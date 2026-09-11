#!/bin/bash
# llama.cpp: decode speed with and without MTP speculative decoding (real coding prompts, greedy).
# usage: ./bench_mtp_llamacpp.sh <gpu> <gguf> [n_draft_list default "0 2 3 4"]
set -u
GPU=$1; M=$2; NLIST=${3:-"0 2 3 4"}
cd /mnt/ssdraid/git/deepseekv4.1
BIN=./llama.cpp/build/bin
PORT=18080
name=$(basename "$M" .gguf)
OUT=results/mtp-$name.txt
: > $OUT
for N in $NLIST; do
  if [ "$N" = "0" ]; then SPEC=""; else SPEC="--spec-type draft-mtp --spec-draft-n-max $N --spec-draft-n-min 0"; fi
  CUDA_VISIBLE_DEVICES=$GPU $BIN/llama-server -m "$M" -ngl 999 -fa 1 -c 8192 -np 1 -b 2048 -ub 512 --port $PORT --host 127.0.0.1 $SPEC \
      > results/mtp-server-$name-n$N.log 2>&1 &
  SP=$!
  for i in $(seq 1 120); do curl -s http://127.0.0.1:$PORT/health | grep -q '"ok"' && break; sleep 2; done
  echo "=== $name  MTP draft n=$N ===" | tee -a $OUT
  python3 - $PORT $N <<'EOF' | tee -a $OUT
import json, sys, urllib.request, time
port, n = sys.argv[1], sys.argv[2]
prompts = [
 "Write a complete Python implementation of an LRU cache class with get/put methods, docstrings and a small test suite using unittest.",
 "Implement Dijkstra's shortest path algorithm in C++17 with a priority queue, reading a graph from stdin and printing distances. Include comments.",
 "Write a React component (TypeScript) for a paginated table with sorting and a search box, plus a short explanation of the design.",
]
tot_tok = tot_ms = 0; acc = dr = 0
for p in prompts:
    body = json.dumps({"prompt": p, "n_predict": 512, "temperature": 0, "cache_prompt": False}).encode()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"}), timeout=600)
    t = json.loads(r.read())["timings"]
    tot_tok += t["predicted_n"]; tot_ms += t["predicted_ms"]
    acc += t.get("draft_n_accepted", 0) or 0; dr += t.get("draft_n", 0) or 0
    print(f"  {t['predicted_n']:4d} tok  {t['predicted_per_second']:6.1f} tok/s  draft_n={t.get('draft_n',0)} accepted={t.get('draft_n_accepted',0)}")
print(f"  TOTAL n={n}: {tot_tok} tok in {tot_ms/1000:.1f}s = {tot_tok/tot_ms*1000:.1f} tok/s" + (f", acceptance {acc/dr*100:.0f}%" if dr else ""))
EOF
  kill $SP; wait $SP 2>/dev/null
done
echo MTP_BENCH_DONE | tee -a $OUT
