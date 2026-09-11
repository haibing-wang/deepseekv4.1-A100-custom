#!/bin/bash
# usage: ./bench_mtp_vllm.sh <gpu> <model_dir_under_/mnt/ssd/models> [n list default "0 3"]
set -u
GPU=$1; MD=$2; NLIST=${3:-"0 3"}
cd /mnt/ssdraid/git/deepseekv4.1
OUT=results/mtp-vllm-$MD.txt; : > $OUT
for N in $NLIST; do
  echo "=== vLLM $MD mtp n=$N ===" | tee -a $OUT
  docker run --rm --gpus "device=$GPU" --ipc=host -v /mnt/ssd/models:/models -v /mnt/ssdraid/git/deepseekv4.1:/work \
    -v /mnt/ssdraid/git/deepseekv4.1/nvml173/libnvidia-ml.so.1:/opt/nvml/libnvidia-ml.so.1:ro -e LD_LIBRARY_PATH=/opt/nvml \
    -e VLLM_LOGGING_LEVEL=WARNING --entrypoint python3 vllm/vllm-openai:qwen38-flash-next /work/bench_mtp_vllm.py /models/$MD $N ${UTIL:-0.45} \
    > results/mtp-vllm-$MD-n$N.log 2>&1
  grep -E "^RESULT|^METRIC|Error|error:" results/mtp-vllm-$MD-n$N.log | grep -v -i "min_frames\|max_frames" | tail -n 8 | tee -a $OUT
done
echo VLLM_MTP_DONE | tee -a $OUT
