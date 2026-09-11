"""Generate text with DeepSeek-V4.1-Flash on A100s.
usage: .venv-lc/bin/python -m dsv41.run --devices 0,1,2,3,4,5,6,7 --prompt "..." [--max-new-tokens 64]"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsv41.load import load_model  # noqa: E402

CKPT = "/mnt/ssd/models/DeepSeek-V4.1-Flash"


def sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--budgets", default="", help="per-device GB overrides, e.g. 0:60,1:35")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--chat", action="store_true", help="wrap the prompt with the chat template")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--n-layers", type=int, default=None, help="load only the first N layers (plumbing test)")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--offload-experts", action="store_true", help="experts in host RAM (single-GPU mode)")
    ap.add_argument("--profile", action="store_true", help="per-component timing of the decode steps")
    ap.add_argument("--decode", default="graph", choices=["eager", "static", "graph"], help="decode path")
    ap.add_argument("--kernel-profile", action="store_true", help="after generation, profile 8 decode steps and list the top CUDA kernels")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    devices = [int(d) for d in a.devices.split(",")]
    budgets = {int(k): float(v) for k, v in (kv.split(":") for kv in a.budgets.split(",") if kv)} or None
    model = load_model(a.ckpt, devices, max_seq_len=a.max_seq_len, budgets_gb=budgets, n_layers=a.n_layers,
                       engram=not a.no_engram, tokenizer=tok, offload_experts=a.offload_experts)

    if a.chat:
        sys.path.insert(0, os.path.join(a.ckpt, "encoding"))
        from encoding import encode_messages  # type: ignore
        text = encode_messages([{"role": "user", "content": a.prompt}], thinking_mode="chat")
        ids = tok.encode(text)
    else:
        ids = tok.encode(a.prompt)
    print(f"prompt tokens: {len(ids)}", flush=True)
    input_ids = torch.tensor([ids], dtype=torch.long)

    rt = None
    if a.decode != "eager":
        # capture before the prefill: the warm-up/capture runs scribble on the caches at position 0,
        # and the prefill rewrites everything they touched
        from dsv41.decode import DecodeRuntime
        rt = DecodeRuntime(model, use_graphs=(a.decode == "graph" and not a.offload_experts))
        if a.decode == "graph":
            tc = time.time()
            rt.capture()
            print(f"[captured {len(rt.graphs)} CUDA graphs in {time.time() - tc:.1f}s]", flush=True)
    torch.cuda.synchronize()
    t0 = time.time()
    logits = model.forward(input_ids, 0)
    torch.cuda.synchronize()
    t_prefill = time.time() - t0
    out = []
    pos = input_ids.size(1)
    if a.profile:
        from collections import defaultdict
        import dsv41.model as M
        M.PROF = defaultdict(float)
    nxt = sample(logits, a.temperature)
    t1 = time.time()
    for _ in range(a.max_new_tokens):
        out.append(int(nxt.item()))
        print(tok.decode(out[-1:]), end="", flush=True)
        if out[-1] == tok.eos_token_id:
            break
        if rt is not None:
            logits = rt.step(out[-1], pos)
        else:
            logits = model.forward(nxt.view(1, 1), pos)
        pos += 1
        nxt = sample(logits, a.temperature)
    torch.cuda.synchronize()
    t_dec = time.time() - t1
    print(f"\n\nprefill {len(ids)} tok in {t_prefill:.2f}s ({len(ids)/t_prefill:.1f} tok/s); decode {len(out)} tok in {t_dec:.2f}s ({len(out)/max(t_dec,1e-9):.2f} tok/s)")
    if a.kernel_profile and rt is not None:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for i in range(8):
                rt.step(out[-1], pos + i)
            torch.cuda.synchronize()
        rows = [(e.key, e.device_time_total, e.count) for e in prof.key_averages() if e.device_time_total > 0]
        tot = sum(r[1] for r in rows)
        print(f"\n[kernel profile: {tot / 8 / 1000:.1f} ms of GPU time per token]")
        for k, t, c in sorted(rows, key=lambda r: -r[1])[:22]:
            print(f"  {t / 8 / 1000:7.2f} ms/token  {c // 8:5d}/token  {k[:90]}")
    if a.profile:
        import dsv41.model as M
        tot = sum(M.PROF.values())
        for k, v in sorted(M.PROF.items(), key=lambda kv: -kv[1]):
            print(f"  {k:10s} {v / max(len(out), 1) * 1000:7.1f} ms/token  ({v / tot * 100:4.1f}%)")


if __name__ == "__main__":
    main()
