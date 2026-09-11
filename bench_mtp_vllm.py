"""vLLM (inside the container): decode speed with/without MTP speculative decoding on real coding prompts.
usage: python3 bench_mtp_vllm.py <model_dir> <num_spec_tokens: 0 = off>"""
import sys, time
from vllm import LLM, SamplingParams

def main():
    model, n = sys.argv[1], int(sys.argv[2])
    util = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45
    kw: dict = dict(model=model, dtype="float16", kv_cache_dtype="auto", max_model_len=4096, gpu_memory_utilization=util,
              max_num_seqs=8, trust_remote_code=True)
    if n > 0:
        kw["speculative_config"] = {"method": "qwen3_5_mtp", "num_speculative_tokens": n}
    llm = LLM(**kw)
    prompts = [
     "Write a complete Python implementation of an LRU cache class with get/put methods, docstrings and a small test suite using unittest.",
     "Implement Dijkstra's shortest path algorithm in C++17 with a priority queue, reading a graph from stdin and printing distances. Include comments.",
     "Write a React component (TypeScript) for a paginated table with sorting and a search box, plus a short explanation of the design.",
    ]
    sp = SamplingParams(temperature=0, max_tokens=512)
    llm.generate(prompts[:1], sp)  # warmup
    for label, batch in (("concurrency 1", [[p] for p in prompts]), ("concurrency 3", [prompts])):
        tot_tok = tot_t = 0
        for b in batch:
            t = time.perf_counter(); outs = llm.generate(b, sp); dt = time.perf_counter() - t
            tot_tok += sum(len(o.outputs[0].token_ids) for o in outs); tot_t += dt
        print(f"RESULT mtp_n={n} {label}: {tot_tok} tok in {tot_t:.1f}s = {tot_tok/tot_t:.1f} tok/s (includes prefill)")
    try:
        for m in llm.get_metrics():  # type: ignore[attr-defined]
            if "spec_decode" in m.name and hasattr(m, "value"): print(f"METRIC {m.name} = {m.value}")
    except Exception as e:
        print("metrics unavailable:", e)

if __name__ == "__main__":
    main()
