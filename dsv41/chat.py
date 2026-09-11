"""Interactive chat in the terminal.
usage: python -m dsv41.chat --devices 2,0,1,4,5,6,7,3 [--thinking] [--temperature 0.6]
commands: /clear (forget the conversation), /system <text>, /exit"""
import argparse
import sys
import time

import os as _os
_os.environ.setdefault("OMP_WAIT_POLICY", "active")  # CPU expert threads keep spinning between layers (libgomp reads this once)
from .engine import Engine, GenParams, parse_budgets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--devices", default="2,0,1,4,5,6,7,3")
    ap.add_argument("--budgets", default="")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--thinking", action="store_true", help="thinking mode (reasoning before the answer)")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--offload-experts", nargs="?", const="cpu", default=False, choices=["gpu", "cpu"], help="single-GPU mode: experts in host RAM; 'cpu' computes them on the CPU (default), 'gpu' streams them over PCIe")
    ap.add_argument("--hot-experts", type=int, default=0, help="cpu offload mode: experts per layer kept on the GPU (by usage stats)")
    ap.add_argument("--hot-stats", default="", help="route stats .pt used to pick the hot experts (default: results/route_stats.pt)")
    ap.add_argument("--ep", action="store_true", help="expert parallelism: experts sharded over the devices (e.g. --devices 2,3,0,1 --ep-shards 100,100,100,84)")
    ap.add_argument("--ep-shards", default="", help="experts per device for --ep (default: even split)")
    a = ap.parse_args()
    kw = dict(devices=[int(d) for d in a.devices.split(",")], max_seq_len=a.max_seq_len, budgets=parse_budgets(a.budgets),
              use_graphs=not a.no_graphs, thinking_mode="thinking" if a.thinking else "chat", offload_experts=a.offload_experts, hot_experts=a.hot_experts, route_stats=a.hot_stats,
              ep=a.ep, ep_shards=[int(v) for v in a.ep_shards.split(",")] if a.ep_shards else None)
    eng = Engine(a.ckpt, **kw) if a.ckpt else Engine(**kw)
    print("DeepSeek-V4.1-Flash on A100. /clear /system <text> /exit", flush=True)
    messages: list[dict] = []
    params = GenParams(max_new_tokens=a.max_new_tokens, temperature=a.temperature, top_p=a.top_p)
    while True:
        try:
            line = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line.strip():
            continue
        if line.strip() == "/exit":
            break
        if line.strip() == "/clear":
            messages.clear()
            print("[conversation cleared]")
            continue
        if line.startswith("/system "):
            messages = [m for m in messages if m["role"] != "system"]
            messages.insert(0, {"role": "system", "content": line[8:].strip()})
            print("[system prompt set]")
            continue
        messages.append({"role": "user", "content": line})
        ids = eng.tok.encode(eng.chat_prompt(messages))
        t0 = time.time()
        n = 0
        text = []
        try:
            for _, piece in eng.generate(ids, params):
                sys.stdout.write(piece)
                sys.stdout.flush()
                text.append(piece)
                n += 1
        except KeyboardInterrupt:
            print("\n[interrupted]")
        dt = time.time() - t0
        completion = "".join(text)
        messages.append(eng.parse_completion(completion))
        print(f"\n[{n} tokens, {n / max(dt, 1e-9):.1f} tok/s, prompt {len(ids)} tokens]")


if __name__ == "__main__":
    main()
