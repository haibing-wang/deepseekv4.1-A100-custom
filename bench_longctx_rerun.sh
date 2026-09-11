#!/bin/bash
# reruns: 128K x 8 (FP8 KV) after the previous container released the GPU, and 256K x 2 within the model's 262144 limit
set -u
GPU=$1; UTIL=0.90
OUT=results/longctx-vllm.txt
run() {
  echo "=== in=$1 out=$2 prompts=$3 max_seqs=$4 kv=$5 util=$UTIL ===" | tee -a $OUT
  docker run --rm --gpus "device=$GPU" --ipc=host -v /mnt/ssd/models:/models \
    -v /mnt/ssdraid/git/deepseekv4.1/nvml173/libnvidia-ml.so.1:/opt/nvml/libnvidia-ml.so.1:ro -e LD_LIBRARY_PATH=/opt/nvml \
    -e VLLM_LOGGING_LEVEL=WARNING --entrypoint vllm vllm/vllm-openai:qwen38-flash-next bench throughput \
      --model /models/Qwen3.8-27B-NVFP4 --dtype float16 --kv-cache-dtype $5 --trust-remote-code \
      --max-model-len $6 --gpu-memory-utilization $UTIL --max-num-seqs $4 --max-num-batched-tokens 8192 \
      --enable-chunked-prefill --dataset-name random --random-input-len $1 --random-output-len $2 --num-prompts $3 \
      > results/longctx-vllm-in$1-n$3-kv$5.log 2>&1
  grep -E "Throughput|Error|error:|ValueError|RuntimeError" results/longctx-vllm-in$1-n$3-kv$5.log | grep -v -i "min_frames\|max_frames" | tail -n 2 | tee -a $OUT
  sleep 30
}
sleep 30
run 131072 256 8 8 fp8 139264
run 261632 256 2 2 fp8 262144
echo LONGCTX_RERUN_DONE | tee -a $OUT
