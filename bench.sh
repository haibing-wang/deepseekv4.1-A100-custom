#!/bin/bash
# NVFP4-on-A100 benchmark: llama.cpp int8 path (NVFP4 -> int8 LUT -> dp4a / INT8 mma)
# usage: ./bench.sh <gpu_index> <gguf> [<gguf> ...]
set -u
GPU=$1; shift
BIN=/mnt/ssdraid/git/deepseekv4.1/llama.cpp/build/bin
OUT=/mnt/ssdraid/git/deepseekv4.1/results
mkdir -p $OUT
for M in "$@"; do
  name=$(basename "$M" .gguf)
  echo "=== $name ($(du -h "$M" | cut -f1)) on GPU $GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU $BIN/llama-bench -m "$M" -ngl 999 -fa 1 \
      -p 512,2048 -n 128 -b 2048 -ub 512 -r 3 -o md 2>/dev/null | tee "$OUT/bench-$name.md"
  # batched decode throughput (simulates concurrent requests): 1, 8, 32 sequences
  CUDA_VISIBLE_DEVICES=$GPU $BIN/llama-batched-bench -m "$M" -ngl 999 -fa 1 -c 8192 \
      -npp 512 -ntg 128 -npl 1,8,32 2>/dev/null | tail -n 6 | tee "$OUT/batched-$name.txt"
done
