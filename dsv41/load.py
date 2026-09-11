"""Checkpoint loading with layer placement across GPUs and Engram tables in host RAM."""
from __future__ import annotations

import json
import os
import time

import torch

from .engram import Engram, EngramLayout, HostEngramTable, NgramHashState
from .model import Args, Block, Transformer
from .quant import dequant_fp8_block
from .stio import Checkpoint

LAYER_GB = 7.1  # experts (6.72 GiB fp4 + scales) + dense bf16 (0.34 GiB) per layer
RESERVE_GB = 2.0  # activations / temporaries per device


def plan_placement(n_layers: int, devices: list[int], budgets_gb: dict[int, float] | None = None) -> list[torch.device]:
    """Greedy: fill each device (in the given order) with as many layers as its free memory allows."""
    free = {}
    for d in devices:
        if budgets_gb and d in budgets_gb:
            free[d] = budgets_gb[d]
        else:
            f, _ = torch.cuda.mem_get_info(d)
            free[d] = f / 2**30
    placement: list[torch.device] = []
    for d in devices:
        n = int(max(0, free[d] - RESERVE_GB - (2.6 if d == devices[0] else 0)) // LAYER_GB)
        for _ in range(n):
            if len(placement) < n_layers:
                placement.append(torch.device(f"cuda:{d}"))
    if len(placement) < n_layers:
        raise RuntimeError(f"not enough GPU memory: placed {len(placement)}/{n_layers} layers with free={free}")
    return placement


def _dense(ckpt: Checkpoint, name: str, device) -> torch.Tensor:
    """A Linear weight as bf16 on device (FP8 block-scaled -> dequantized; bf16 stays)."""
    dtype, _ = ckpt.meta(name)
    w = ckpt.get(name, device)
    if dtype == "F8_E4M3":
        s = ckpt.get(name.replace(".weight", ".scale"), device)
        return dequant_fp8_block(w, s)
    return w


def load_layer(ckpt: Checkpoint, i: int, device) -> dict:
    p = f"layers.{i}."
    w: dict[str, torch.Tensor] = {}
    for n in ckpt.names(p):
        key = n[len(p):]
        if ".experts." in key or key.startswith("engram.embed") or key.endswith(".scale") or "bias_vl" in key:
            continue
        if key.endswith(".weight") and ckpt.meta(n)[0] in ("F8_E4M3", "BF16"):
            w[key] = _dense(ckpt, n, device)
        else:
            w[key] = ckpt.get(n, device)
    # experts, stacked: w13 = [w1; w3] along N
    e_names = sorted({int(n.split(".experts.")[1].split(".")[0]) for n in ckpt.names(p + "ffn.experts.")})
    E = len(e_names)
    inter, dimh = ckpt.meta(p + "ffn.experts.0.w1.weight")[1]
    dim = ckpt.meta(p + "ffn.experts.0.w2.weight")[1][0]
    w13 = torch.empty(E, 2 * inter, dimh, dtype=torch.uint8, device=device)
    s13 = torch.empty(E, 2 * inter, dimh * 2 // 32, dtype=torch.uint8, device=device)
    w2 = torch.empty(E, dim, inter // 2, dtype=torch.uint8, device=device)
    s2 = torch.empty(E, dim, inter // 32, dtype=torch.uint8, device=device)
    for e in e_names:
        q = f"{p}ffn.experts.{e}."
        w13[e, :inter] = ckpt.get(q + "w1.weight").view(torch.uint8).to(device, non_blocking=True)
        w13[e, inter:] = ckpt.get(q + "w3.weight").view(torch.uint8).to(device, non_blocking=True)
        s13[e, :inter] = ckpt.get(q + "w1.scale").to(device, non_blocking=True)
        s13[e, inter:] = ckpt.get(q + "w3.scale").to(device, non_blocking=True)
        w2[e] = ckpt.get(q + "w2.weight").view(torch.uint8).to(device, non_blocking=True)
        s2[e] = ckpt.get(q + "w2.scale").to(device, non_blocking=True)
    w.update({"experts.w13": w13, "experts.s13": s13, "experts.w2": w2, "experts.s2": s2})
    torch.cuda.synchronize(device)
    return w


def load_model(ckpt_path: str, devices: list[int], max_seq_len: int = 16384, max_batch: int = 1,
               budgets_gb: dict[int, float] | None = None, n_layers: int | None = None, engram: bool = True,
               tokenizer=None) -> Transformer:
    cfg = json.load(open(os.path.join(ckpt_path, "inference", "config.json")))
    args = Args(cfg, max_batch_size=max_batch, max_seq_len=max_seq_len)
    ckpt = Checkpoint(ckpt_path)
    n_layers = n_layers or cfg["n_layers"]
    placement = plan_placement(n_layers, devices, budgets_gb)
    print("placement:", {str(d): placement.count(d) for d in dict.fromkeys(placement)}, flush=True)
    model = Transformer(args)
    t0 = time.time()
    # per-(owner layer, device) mirrors of the shared compressed-KV and index-key caches
    for owner in cfg["kv_source_layers"]:
        if owner >= n_layers:
            continue
        rows = max_seq_len // cfg["compress_ratios"][owner] + 1  # + one dummy row for the static decode path
        for d in dict.fromkeys(placement):
            model.shared.compress_kv[(owner, d)] = torch.zeros(max_batch, rows, cfg["head_dim"], dtype=torch.bfloat16, device=d)
            model.shared.index_k[(owner, d)] = torch.zeros(max_batch, rows, cfg["index_head_dim"], dtype=torch.bfloat16, device=d)
    for i in range(n_layers):
        dev = placement[i]
        w = load_layer(ckpt, i, dev)
        model.blocks.append(Block(args, i, w, dev, model.shared))
        del w
        print(f"  layer {i:2d} -> {dev}  ({time.time() - t0:5.0f}s)", flush=True)
    dev0, devL = placement[0], placement[n_layers - 1]
    model.embed = ckpt.get("embed.weight", dev0)
    model.head = ckpt.get("head.weight", devL)
    model.norm_w = ckpt.get("norm.weight", devL)
    if engram and cfg.get("engram_layer_ids"):
        layout = EngramLayout(cfg)
        assert tokenizer is not None, "the Engram hash needs the tokenizer"
        model.engram_hash = NgramHashState(cfg, layout, tokenizer, max_batch, max_seq_len, dev0)
        for li, lid in enumerate(layout.layer_ids):
            if lid >= n_layers:
                continue
            p = f"layers.{lid}.engram."
            t1 = time.time()
            weight = ckpt.get(p + "embed.weight").view(torch.uint8).clone()  # into RAM (sequential read)
            scale = ckpt.get(p + "embed.scale").clone()
            print(f"  engram table layer {lid}: {weight.numel() / 2**30:.1f} GiB in host RAM ({time.time() - t1:.0f}s)", flush=True)
            dev = placement[lid]
            blk = model.blocks[lid]
            blk.engram = Engram(cfg["dim"], cfg["hc_mult"], layout, HostEngramTable(weight, scale),
                                _dense(ckpt, p + "wkv.weight", dev), ckpt.get(p + "q_weight", dev), ckpt.get(p + "k_weight", dev), cfg["norm_eps"])
            blk.engram.layer_hash_index = li
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    return model
