#!/bin/bash
# vLLM long-context throughput on one A100 (Marlin NVFP4 W4A16), random prompts.
# usage: ./bench_longctx_vllm.sh <gpu> [gpu_mem_util default 0.90]
set -u
GPU=$1; UTIL=${2:-0.90}
cd /mnt/ssdraid/git/deepseekv4.1
OUT=results/longctx-vllm.txt; : > $OUT
run() { # $1=input_len $2=output_len $3=num_prompts $4=max_num_seqs $5=kv_dtype $6=max_model_len
  echo "=== in=$1 out=$2 prompts=$3 max_seqs=$4 kv=$5 util=$UTIL ===" | tee -a $OUT
  docker run --rm --gpus "device=$GPU" --ipc=host -v /mnt/ssd/models:/models \
    -v /mnt/ssdraid/git/deepseekv4.1/nvml173/libnvidia-ml.so.1:/opt/nvml/libnvidia-ml.so.1:ro -e LD_LIBRARY_PATH=/opt/nvml \
    -e VLLM_LOGGING_LEVEL=WARNING --entrypoint vllm vllm/vllm-openai:qwen38-flash-next bench throughput \
      --model /models/Qwen3.8-27B-NVFP4 --dtype float16 --kv-cache-dtype $5 --trust-remote-code \
      --max-model-len $6 --gpu-memory-utilization $UTIL --max-num-seqs $4 --max-num-batched-tokens 8192 \
      --enable-chunked-prefill --input-len $1 --output-len $2 --num-prompts $3 \
      > results/longctx-vllm-in$1-n$3-kv$5.log 2>&1
  grep -E "Throughput|Error|error:|ValueError|RuntimeError" results/longctx-vllm-in$1-n$3-kv$5.log | grep -v -i "min_frames\|max_frames" | tail -n 3 | tee -a $OUT
}
run 32768  256 16 16 auto 40960
run 32768  256 16 16 fp8  40960
run 131072 256 4  4  auto 139264
run 131072 256 8  8  fp8  139264
run 262144 256 2  2  fp8  270336
run 4096   512 64 64 auto 8192
echo LONGCTX_DONE | tee -a $OUT
