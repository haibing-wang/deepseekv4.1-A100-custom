"""Generation engine shared by the REPL and the OpenAI-compatible server: loads the model once, keeps
the CUDA-graph decode runtime, and streams tokens for one request at a time (the runtime is single
sequence; requests are serialized with a lock)."""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

import torch

from .decode import DecodeRuntime
from .load import load_model

CKPT = "/mnt/ssd/models/DeepSeek-V4.1-Flash"


@dataclass
class GenParams:
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


def sample_token(logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator | None) -> int:
    if temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if 0 < top_p < 1:
        sp, si = probs.sort(descending=True)
        keep = (sp.cumsum(-1) - sp) < top_p  # keep tokens until cumulative mass passes top_p
        sp = sp * keep
        idx = torch.multinomial(sp / sp.sum(), 1, generator=gen)
        return int(si.gather(-1, idx).item())
    return int(torch.multinomial(probs, 1, generator=gen).item())


MTP_STATS = os.environ.get("DSV41_MTP_STATS") == "1"


def mtp_policy(n_contexts: int) -> int:
    """Drafts per step that maximise aggregate throughput for n contexts on one 4-GPU replica (measured 2026-09-12,
    64 mixed prompts): 5 up to 8 contexts, 3 up to 40, none beyond (see README). A scheduler with continuous
    batching should re-evaluate this per step (and lower K when the recent acceptance rate is low)."""
    if n_contexts <= 8:
        return 5
    if n_contexts <= 40:
        return 3
    return 0


class Engine:
    def __init__(self, ckpt: str = CKPT, devices: list[int] | None = None, max_seq_len: int = 8192,
                 budgets: dict[int, float] | None = None, use_graphs: bool = True, thinking_mode: str = "chat",
                 offload_experts=False, hot_experts: int = 0, route_stats: str = "", ep: bool = False, ep_shards: list[int] | None = None,
                 mtp: int = 0):
        from transformers import AutoTokenizer

        sys.path.insert(0, os.path.join(ckpt, "encoding"))
        from encoding import encode_messages, parse_message_from_completion_text  # type: ignore

        self._encode = encode_messages
        self._parse = parse_message_from_completion_text
        self.thinking_mode = thinking_mode
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.max_seq_len = max_seq_len
        self.mtp = mtp  # DSpark drafts verified per step (0 = plain decode)
        self.model = load_model(ckpt, devices or list(range(torch.cuda.device_count())), max_seq_len=max_seq_len,
                                budgets_gb=budgets, tokenizer=self.tok, offload_experts=offload_experts,
                                hot_experts=hot_experts, route_stats=route_stats, ep=ep, ep_shards=ep_shards,
                                max_batch=1 + mtp, max_seqs=1, bf16_copies=False, last_light=12 if mtp else 3)
        if offload_experts:
            from .decode import OffloadDecodeRuntime
            self.rt = OffloadDecodeRuntime(self.model, use_graphs=use_graphs)
        elif ep:
            from .ep import EPRuntime
            self.rt = EPRuntime(self.model, use_graphs=use_graphs)
        else:
            self.rt = DecodeRuntime(self.model, use_graphs=use_graphs)
        if mtp:
            from .dspark import DSparkRows
            from .load import Checkpoint
            m = self.model
            self.ds = DSparkRows(Checkpoint(ckpt), m.args, m.blocks[-1].device, m.embed, m.head, m.shared, len(m.blocks), self.rt)
            m.collect_main_hidden = self.ds.targets
        if not offload_experts:
            from .load import keep_bf16_copies
            from .w8 import BF16_COPY
            if BF16_COPY:  # after every weight is loaded, before the graphs bake in the kernel choice
                keep_bf16_copies(self.model, extra_reserve_gb={self.model.blocks[-1].device: 3.0} if mtp else None)
        if use_graphs:
            self.rt.capture()
        if mtp:
            self.ds.capture(1)
        self.lock = threading.Lock()
        self.eos = self.tok.eos_token_id
        self.model_name = "deepseek-v4.1-flash"

    # ---------------------------------------------------------------- prompts
    def chat_prompt(self, messages: list[dict], thinking_mode: str | None = None) -> str:
        return self._encode(messages, thinking_mode=thinking_mode or self.thinking_mode)

    def parse_completion(self, text: str, thinking_mode: str | None = None) -> dict:
        """Structured assistant message (content / reasoning_content / tool_calls). The official parser
        wants the completion to end with the EOS string; we generate without it, so append it."""
        eos = self.tok.eos_token or ""
        try:
            return self._parse(text + eos, thinking_mode=thinking_mode or self.thinking_mode)
        except Exception:
            return {"role": "assistant", "content": text, "reasoning_content": None, "tool_calls": []}

    # ---------------------------------------------------------------- generation
    @torch.inference_mode()
    def generate(self, prompt_ids: list[int], p: GenParams) -> Iterator[tuple[int, str]]:
        """Yields (token_id, text_piece) as they are produced. Holds the engine lock for the duration."""
        assert len(prompt_ids) < self.max_seq_len, f"prompt of {len(prompt_ids)} tokens exceeds max_seq_len={self.max_seq_len}"
        max_new = min(p.max_new_tokens, self.max_seq_len - len(prompt_ids) - 1 - self.mtp)
        gen = None
        if p.seed is not None:
            gen = torch.Generator(device=self.model.blocks[-1].device)
            gen.manual_seed(p.seed)
        with self.lock:
            t_pre = time.perf_counter()
            logits = self.model.forward(torch.tensor([prompt_ids], dtype=torch.long), 0)
            out: list[int] = []
            state = {"decoded_upto": 0, "pending": ""}

            def emit(t: int):
                """Incremental decode of one accepted token: yields (t, text) pieces, or the sentinel None to stop."""
                out.append(t)
                text = self.tok.decode(out[state["decoded_upto"]:])
                if "�" in text:  # hold back a partial multi-byte character
                    return
                piece, state["decoded_upto"] = text, len(out)
                if piece:
                    state["pending"] += piece
                    if p.stop and any(s in state["pending"] for s in p.stop):
                        cut = min(state["pending"].find(s) for s in p.stop if s in state["pending"])
                        if cut > 0:
                            yield t, state["pending"][:cut]
                        yield None
                        return
                    yield t, state["pending"]
                    state["pending"] = ""

            stopped = False
            self.last_stats = {"prefill_s": 0.0, "tokens": 0}
            if self.mtp:
                tokens = self._generate_mtp(prompt_ids, logits, p, gen, max_new)
            else:
                tokens = self._generate_plain(prompt_ids, logits, p, gen, max_new)
            for t in tokens:
                if self.last_stats["tokens"] == 0:
                    self.last_stats["prefill_s"] = time.perf_counter() - t_pre  # up to the first sampled token
                self.last_stats["tokens"] += 1
                if t == self.eos:
                    break
                for item in emit(t):
                    if item is None:
                        stopped = True
                        break
                    yield item
                if stopped:
                    break
            if not stopped and out[state["decoded_upto"]:]:
                tail = self.tok.decode(out[state["decoded_upto"]:])
                if tail:
                    yield out[-1], tail

    def _generate_plain(self, prompt_ids, logits, p, gen, max_new) -> Iterator[int]:
        pos = len(prompt_ids)
        for _ in range(max_new):
            t = sample_token(logits[0], p.temperature, p.top_p, gen)
            yield t
            if t == self.eos:
                return
            logits = self.rt.step(t, pos)
            pos += 1

    def _generate_mtp(self, prompt_ids, logits, p, gen, max_new) -> Iterator[int]:
        """Speculative decoding with the DSpark draft: every step verifies [bonus, K drafts] as K+1 rows of one batched
        step (positions p+1 .. p+K+1 of sequence 0); row i is sampled from the target distribution given the accepted
        prefix, so the output is an exact sample whatever the temperature (drafts only decide how many rows survive)."""
        K = self.mtp
        ds, rt, m = self.ds, self.rt, self.model
        last = m.blocks[-1].device
        T = len(prompt_ids)
        ds.write_main_rows(m.main_hidden[0], torch.zeros(T, dtype=torch.int64, device=last), torch.arange(T, device=last))
        mh = m.main_hidden[0, -1]
        p_last = T - 1
        bonus = sample_token(logits[0], p.temperature, p.top_p, gen)
        yield bonus
        n = 1
        if bonus == self.eos:
            return
        rows = torch.zeros(K + 1, dtype=torch.int64, device=last)  # sequence ids (all 0)
        written_max = p_last  # newest position written to the main rings (positions beyond p_last are rejected drafts)
        steps = n_acc = 0
        t_step = t_draft = 0.0
        while n < max_new:
            t0 = time.perf_counter()
            drafts = ds.draft_rows(torch.tensor([bonus], device=last), torch.tensor([p_last], device=last), mh[None],
                                   torch.tensor([written_max], device=last))[0, :K].tolist()
            toks = [bonus] + drafts
            poss = [p_last + 1 + i for i in range(K + 1)]
            t1 = time.perf_counter()
            lg = rt.step(toks, poss, seq=[0] * (K + 1), pmax=[p_last + K + 1] * (K + 1))
            mh_all = torch.cat([rt.main_hid[l].to(last) for l in ds.targets], dim=-1)  # [K+1, 3*dim]
            ds.write_main_rows(mh_all, rows, torch.tensor(poss, device=last))
            acc = 0
            done = False
            while True:  # row acc: the target's sample after the accepted prefix
                t = sample_token(lg[acc], p.temperature, p.top_p, gen)
                yield t
                n += 1
                if t == self.eos or n >= max_new:
                    done = True
                    break
                if acc < K and t == drafts[acc]:
                    acc += 1
                    continue
                break
            steps += 1
            n_acc += acc
            t_draft += t1 - t0
            t_step += time.perf_counter() - t1
            if done:
                break
            bonus = t
            written_max = poss[-1]
            p_last = p_last + 1 + acc
            mh = mh_all[acc]
        if MTP_STATS and steps:
            print(f"[mtp] {steps} steps, {n_acc / steps:.2f} of {K} drafts accepted per step, draft {t_draft / steps * 1000:.1f} ms, "
                  f"verify+sample {t_step / steps * 1000:.1f} ms per step", file=sys.stderr, flush=True)

    def generate_text(self, prompt_ids: list[int], p: GenParams) -> tuple[str, int]:
        pieces, n = [], 0
        for _, piece in self.generate(prompt_ids, p):
            pieces.append(piece)
            n += 1
        return "".join(pieces), n


def parse_budgets(spec: str) -> dict[int, float] | None:
    return {int(k): float(v) for k, v in (kv.split(":") for kv in spec.split(",") if kv)} or None
