"""Make an INT8 W8A8 (per-channel weights, dynamic per-token activations) checkpoint of Qwen3.8-27B
with llm-compressor (SmoothQuant + GPTQ), for vLLM's INT8 CUTLASS kernels on A100.
Calibration text: local wikitext-2 train split (no network needed).
usage: CUDA_VISIBLE_DEVICES=<gpu> .venv-lc/bin/python quant_w8a8.py <src_dir> <dst_dir>"""
import sys, socket, random
# force IPv4 for anything that touches the network (IPv6 is dead on this host)
_orig = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [ai for ai in _orig(*a, **k) if ai[0] == socket.AF_INET]

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

src, dst = sys.argv[1], sys.argv[2]
NUM_SAMPLES, MAX_LEN = 256, 1024

tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map=None)

text = open("/mnt/ssdraid/git/deepseekv4.1/data/wikitext-2-raw/wiki.train.raw").read()
paras = [p for p in text.split("\n \n") if len(p) > 2000]
random.seed(0); random.shuffle(paras)
ds = Dataset.from_dict({"text": paras[:NUM_SAMPLES]})
ds = ds.map(lambda b: tok(b["text"], max_length=MAX_LEN, truncation=True, padding=False), remove_columns=["text"])

ignore = ["lm_head", "re:.*mtp.*", "re:.*visual.*", "re:.*vision.*"]
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8, ignore=ignore),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=ignore),
]
oneshot(model=model, processor=tok, dataset=ds, recipe=recipe, max_seq_length=MAX_LEN, num_calibration_samples=NUM_SAMPLES)
model.save_pretrained(dst, save_compressed=True)
tok.save_pretrained(dst)
print("saved", dst)
