#!/bin/bash
# vLLM Marlin NVFP4 (W4A16: FP4 -> FP16 dequant in-kernel, FP16 tensor cores) on A100
# usage: ./bench_vllm.sh <gpu_index>
set -u
GPU=$1
IMG=vllm/vllm-openai:qwen38-flash-next
MODEL=/models/Qwen3.8-27B-INT8-W8A16-MTP
OUT=/mnt/ssdraid/git/deepseekv4.1/results
mkdir -p $OUT
run() {  # $1 = batch size
  docker run --rm --gpus "device=$GPU" --ipc=host \
    -v /mnt/ssd/models:/models -v /mnt/ssdraid/git/deepseekv4.1/results:/results \
    -e VLLM_LOGGING_LEVEL=WARNING -v /mnt/ssdraid/git/deepseekv4.1/nvml173/libnvidia-ml.so.1:/opt/nvml/libnvidia-ml.so.1:ro -e LD_LIBRARY_PATH=/opt/nvml \
    --entrypoint vllm $IMG bench latency \
      --model $MODEL --dtype float16 --kv-cache-dtype auto \
      --max-model-len 4096 --gpu-memory-utilization 0.45 --max-num-seqs 64 \
      --input-len 512 --output-len 128 --batch-size $1 \
      --num-iters-warmup 2 --num-iters 5 \
      --output-json /results/vllm-int8w8a16-latency-bs$1.json 2>&1 | grep -v -E "^INFO|^WARNING|^\s*$" | tail -n 12
}
for bs in 1 8 32; do echo "=== vLLM Marlin NVFP4 batch=$bs ==="; run $bs; done
