#!/bin/bash
# perplexity on wikitext-2 test (first N chunks of 2048 tokens)
# usage: ./ppl.sh <gpu_index> <gguf> [<gguf> ...]
set -u
GPU=$1; shift
BIN=/mnt/ssdraid/git/deepseekv4.1/llama.cpp/build/bin
DATA=/mnt/ssdraid/git/deepseekv4.1/data/wikitext-2-raw/wiki.test.raw
OUT=/mnt/ssdraid/git/deepseekv4.1/results
CHUNKS=${CHUNKS:-40}
mkdir -p $OUT
for M in "$@"; do
  name=$(basename "$M" .gguf)
  echo "=== ppl $name (chunks=$CHUNKS, ctx=2048) ==="
  CUDA_VISIBLE_DEVICES=$GPU $BIN/llama-perplexity -m "$M" -ngl 999 -fa 1 -c 2048 -b 2048 --chunks $CHUNKS -f $DATA \
      2>&1 | grep -E "Final estimate|\[1\]" | tail -n 2 | tee "$OUT/ppl-$name.txt"
done
