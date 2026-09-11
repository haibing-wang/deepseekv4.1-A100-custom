"""Adaptive GPU cache of experts for the single-GPU (CPU expert) mode.

Each layer keeps a fixed number of expert slots in VRAM (loaded from a static routing profile at
start). While decoding, an exponential moving average of expert usage is kept per layer; when a
cold expert becomes clearly more used than the least-used resident one, the resident one is evicted
and the cold expert's weights are copied host -> GPU by a background thread (17.7 MB, ~2 ms over
PCIe). The slot map is updated only once the copy has landed, so the decode graphs never read a
slot that is being rewritten (the victim is dropped from the map before the copy starts)."""
from __future__ import annotations

import queue
import threading

import numpy as np
import torch


class HotCache:
    def __init__(self, blocks, device: torch.device, decay: float = 0.97, margin: float = 2.0, swaps_per_token: int = 2, cpus=None):
        self.device = device
        self.cpus = cpus  # cpus for the copy thread (reserved ones near the GPU), or None
        self.decay, self.margin = decay, margin
        self.swaps_per_token = swaps_per_token
        self.layers = {}
        for blk in blocks:
            moe = blk.ffn
            if not moe.hot:
                continue
            hot = moe.hot
            n = hot["dummy"]
            slot_expert = np.full(n, -1, dtype=np.int64)
            for e, k in hot["slot"].items():
                slot_expert[k] = e
            self.layers[blk.layer_id] = {"moe": moe, "usage": np.zeros(moe.n_experts, dtype=np.float32), "slot_expert": slot_expert,
                                         "pending": set(), "done": []}
        self.lock = threading.Lock()
        self.q: queue.Queue = queue.Queue()
        self.budget = swaps_per_token
        self.n_swaps = 0
        self.n_hits = 0
        self.n_total = 0
        self.stream = torch.cuda.Stream(device=device)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def new_token(self):
        self.budget = self.swaps_per_token

    # ---- called by the decode loop for layer `lid` with this token's routed expert ids (before splitting hot/cold)
    def update(self, lid: int, ids: list[int]):
        L = self.layers.get(lid)
        if L is None:
            return
        moe, usage, slot_expert = L["moe"], L["usage"], L["slot_expert"]
        smap = moe.hot["slot"]
        with self.lock:
            done, L["done"] = L["done"], []
        for slot, e in done:  # copies that landed: publish
            slot_expert[slot] = e
            smap[e] = slot
            L["pending"].discard(slot)
        usage *= self.decay
        usage[ids] += 1.0
        self.n_total += len(ids)
        self.n_hits += sum(1 for e in ids if e in smap)
        if self.budget <= 0:
            return
        cold = [e for e in ids if e not in smap]
        if not cold:
            return
        # eviction candidates: resident experts not used by this token and not being loaded (vectorised)
        resident = slot_expert >= 0
        su = np.where(resident, usage[np.maximum(slot_expert, 0)], np.inf)
        su[np.isin(slot_expert, ids)] = np.inf
        if L["pending"]:
            su[list(L["pending"])] = np.inf
        for e in sorted(cold, key=lambda e: -usage[e]):
            if self.budget <= 0:
                break
            victim = int(np.argmin(su))
            if not np.isfinite(su[victim]) or usage[e] < su[victim] + self.margin:
                break
            su[victim] = np.inf
            ve = int(slot_expert[victim])
            del smap[ve]  # cold from now on; the slot is off-limits until the copy lands
            slot_expert[victim] = -1
            L["pending"].add(victim)
            self.budget -= 1
            self.n_swaps += 1
            self.q.put((lid, victim, int(e)))

    def _worker(self):
        import ctypes
        if self.cpus:
            try:
                import os
                os.sched_setaffinity(0, set(self.cpus))
            except OSError:
                pass
        # Staging through pinned memory with a plain memmove: a torch copy from pageable host memory would run
        # its CPU side through the shared OpenMP pool (stalling the expert kernel's parallel regions for tens
        # of ms) and hold the driver busy; pinned -> device is a pure async DMA.
        stage = {}
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            while True:
                lid, slot, e = self.q.get()
                L = self.layers[lid]
                moe, hot = L["moe"], L["moe"].hot
                for key, src in (("w13", moe.w13), ("s13", moe.s13), ("w2", moe.w2), ("s2", moe.s2)):
                    row = src[e]
                    buf = stage.get(key)
                    if buf is None or buf.numel() != row.numel():
                        buf = stage[key] = torch.empty(row.numel(), dtype=torch.uint8, pin_memory=True)
                    ctypes.memmove(buf.data_ptr(), row.data_ptr(), row.numel())
                    hot[key][slot].view(-1).copy_(buf, non_blocking=True)
                    self.stream.synchronize()  # the staging buffer is reused for the next piece
                with self.lock:
                    L["done"].append((slot, e))

    def stats(self) -> str:
        return f"hot cache: hit rate {self.n_hits / max(self.n_total, 1) * 100:.1f}% ({self.n_hits}/{self.n_total}), swaps {self.n_swaps}"
