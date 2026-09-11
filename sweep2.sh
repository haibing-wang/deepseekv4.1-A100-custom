#!/bin/bash
# Sweep v2: waits for the re-quantized baselines (output tensor left as NVFP4 in every variant),
# then llama-bench + batched decode (+ perplexity) per variant. Q8_0 attention everywhere; only MLP differs.
set -u
cd /mnt/ssdraid/git/deepseekv4.1
BIN=./llama.cpp/build/bin
G=/mnt/ssd/models/gguf
OUT=results
GPU_MAIN=${GPU_MAIN:-4}
GPU_F16=${GPU_F16:-4,5}

run_variant() {  # $1 = gpu list, $2 = variant name
  local M=$G/$2.gguf
  echo "=== $2 ($(du -h $M | cut -f1)) on GPU $1 ==="
  CUDA_VISIBLE_DEVICES=$1 $BIN/llama-bench -m $M -ngl 999 -fa 1 -p 512,2048 -n 128 -b 2048 -ub 512 -r 5 -o md \
      2>$OUT/bench-$2.err | tee $OUT/bench-$2.md | grep "^| qwen"
  CUDA_VISIBLE_DEVICES=$1 $BIN/llama-batched-bench -m $M -ngl 999 -fa 1 -c 24576 -npp 512 -ntg 128 -npl 1,8,32 \
      2>$OUT/batched-$2.err | tee $OUT/batched-$2.txt | grep "^|" | grep -v -E "PP \||---"
}
ppl_variant() {  # $1 = gpu list, $2 = variant name
  printf "%-32s " $2
  CUDA_VISIBLE_DEVICES=$1 $BIN/llama-perplexity -m $G/$2.gguf -ngl 999 -fa 1 -c 2048 -b 2048 --chunks 40 \
      -f data/wikitext-2-raw/wiki.test.raw 2>&1 | grep "Final estimate" | tee $OUT/ppl-$2.txt
}

until [ -e $OUT/.requant_done ]; do sleep 10; done
for name in Qwen3.8-27B-NVFP4-attnQ8 Qwen3.8-27B-mlpQ4_0-attnQ8 Qwen3.8-27B-mlpQ8_0-attnQ8; do
  run_variant $GPU_MAIN $name
done
echo "=== perplexity (wikitext-2 test, 40 chunks x 2048) ==="
for name in Qwen3.8-27B-mlpQ4_0-attnQ8 Qwen3.8-27B-mlpQ8_0-attnQ8; do
  ppl_variant $GPU_MAIN $name
done
# wait until no vLLM bench container has been running for ~45 s
idle=0
while [ $idle -lt 3 ]; do
  if docker ps --format '{{.Command}}' | grep -q "vllm bench"; then idle=0; else idle=$((idle+1)); fi
  sleep 15
done
run_variant $GPU_F16 Qwen3.8-27B-mlpF16-attnQ8
echo "=== perplexity F16 ==="
ppl_variant $GPU_F16 Qwen3.8-27B-mlpF16-attnQ8
echo SWEEP2_DONE
