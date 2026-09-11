"""Compare the fused-2 decode layer against the previous static path: same tokens, logits close.
usage: .venv-lc/bin/python -m dsv41.test_fused2 --devices 2 --n-layers 6"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsv41 import decode as D  # noqa: E402
from dsv41.load import load_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--devices", default="2")
ap.add_argument("--n-layers", type=int, default=6)
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--prompt", default="The capital of France is Paris. The capital of Germany is")
a = ap.parse_args()
from transformers import AutoTokenizer
ckpt = "/mnt/ssd/models/DeepSeek-V4.1-Flash"
tok = AutoTokenizer.from_pretrained(ckpt)
model = load_model(ckpt, [int(d) for d in a.devices.split(",")], max_seq_len=4096, n_layers=a.n_layers, engram=True, tokenizer=tok)
ids = tok.encode(a.prompt)
rt = D.DecodeRuntime(model, use_graphs=False)
logits = model.forward(torch.tensor([ids]), 0)
pos = len(ids)
nxt = int(logits.argmax(-1))
worst = 0.0
for step in range(a.steps):
    D.FUSED2 = False
    rt.kv_owner = rt.index_owner = -1
    ref = rt.step(nxt, pos).clone()
    D.FUSED2 = True
    rt.kv_owner = rt.index_owner = -1
    new = rt.step(nxt, pos).clone()
    diff = (ref - new).abs().max().item()
    worst = max(worst, diff)
    print(f"step {step} pos {pos}: token {nxt!r} {tok.decode([nxt])!r} argmax ref {int(ref.argmax())} new {int(new.argmax())}  max|dlogit| {diff:.4f}  (|logit| max {ref.abs().max().item():.1f})")
    nxt = int(new.argmax(-1))
    pos += 1
print("worst max|dlogit|", worst)
# timing (graphs) of the fused path
D.FUSED2 = True
rt2 = D.DecodeRuntime(model, use_graphs=True)
rt2.capture()
for _ in range(3):
    rt2.step(nxt, pos)
torch.cuda.synchronize()
t0 = time.time()
for i in range(20):
    rt2.step(nxt, pos + i)
torch.cuda.synchronize()
print(f"fused2 graph decode: {(time.time() - t0) / 20 * 1000:.2f} ms/token for {a.n_layers} layers")
D.FUSED2 = False
rt3 = D.DecodeRuntime(model, use_graphs=True)
rt3.capture()
for _ in range(3):
    rt3.step(nxt, pos)
torch.cuda.synchronize()
t0 = time.time()
for i in range(20):
    rt3.step(nxt, pos + i)
torch.cuda.synchronize()
print(f"old graph decode:    {(time.time() - t0) / 20 * 1000:.2f} ms/token for {a.n_layers} layers")
