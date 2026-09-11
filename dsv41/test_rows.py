"""Rows at different positions / sequences: two prompts of different lengths decoded together must give exactly
the logits of their single-sequence runs.  usage: python -m dsv41.test_rows --devices 2 --n-layers 8"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dsv41.load import load_model
from dsv41.decode import DecodeRuntime

ap = argparse.ArgumentParser()
ap.add_argument("--devices", default="2")
ap.add_argument("--n-layers", type=int, default=8)
ap.add_argument("--steps", type=int, default=6)
a = ap.parse_args()
from transformers import AutoTokenizer
ckpt = "/mnt/ssd/models/DeepSeek-V4.1-Flash"
tok = AutoTokenizer.from_pretrained(ckpt)
devs = [int(d) for d in a.devices.split(",")]
PA = tok.encode("The capital of France is Paris. The capital of Germany is")
PB = tok.encode("Once upon a time in a small village by the sea, there lived an old fisherman who")
feed = [19920, 36300, 118795, 28859, 17706, 100003, 2000, 3000]
model = load_model(ckpt, devs, max_seq_len=4096, max_batch=2, n_layers=a.n_layers, engram=True, tokenizer=tok)
rt = DecodeRuntime(model, use_graphs=True)
rt.capture()

def single(prompt, slot):
    """reference: the prompt alone in slot 0, rows = [(0, pos), (dummy row: slot 1 at pos 0... use the same seq for both rows? no:
    use row 1 as a copy of row 0 (same seq/pos/token) so it is a pure duplicate)."""
    model.forward(torch.tensor([prompt]), 0)
    pos = len(prompt)
    outs = []
    for t in feed[: a.steps]:
        lg = rt.step([t, t], [pos, pos], seq=[0, 0], pmax=[pos, pos])
        outs.append(lg[0].clone().cpu())
        pos += 1
    return outs

ref_a = single(PA, 0)
ref_b = single(PB, 0)
# both together: B into slot 0 -> copy to slot 1, then A into slot 0
model.forward(torch.tensor([PB]), 0)
rt.copy_seq(0, 1)
model.forward(torch.tensor([PA]), 0)
pa, pb = len(PA), len(PB)
worst = 0.0
for i, t in enumerate(feed[: a.steps]):
    lg = rt.step([t, t], [pa, pb], seq=[0, 1], pmax=[pa, pb])
    da = (lg[0].cpu() - ref_a[i]).abs().max().item()
    db = (lg[1].cpu() - ref_b[i]).abs().max().item()
    worst = max(worst, da, db)
    print(f"step {i}: A pos {pa} max|d| {da:.4f} argmax {int(lg[0].argmax())}/{int(ref_a[i].argmax())}   B pos {pb} max|d| {db:.4f} argmax {int(lg[1].argmax())}/{int(ref_b[i].argmax())}")
    pa += 1
    pb += 1
print("worst", worst)

# ---- several positions of one sequence in a single step (the verification shape)
model2 = None
K = 6
if a.steps >= K:
    model.forward(torch.tensor([PA]), 0)
    pa = len(PA)
    seqk = [0] * K
    # reference: sequential single rows (row 1 duplicates row 0)
    ref = []
    for i in range(K):
        lg = rt.step([feed[i], feed[i]], [pa + i, pa + i], seq=[0, 0], pmax=[pa + i, pa + i])
        ref.append(lg[0].clone().cpu())
    model.forward(torch.tensor([PA]), 0)
    rt2 = rt
    # the runtime has 2 rows; use a 6-row runtime for the batched check
    from dsv41.load import load_model as _lm
    print("[six rows of one sequence in one step]")
    m6 = _lm(ckpt, [3], max_seq_len=4096, max_batch=K, n_layers=a.n_layers, engram=True, tokenizer=tok)
    r6 = DecodeRuntime(m6, use_graphs=True)
    r6.capture()
    m6.forward(torch.tensor([PA]), 0)
    lg = r6.step(feed[:K], [pa + i for i in range(K)], seq=[0] * K, pmax=[pa + K - 1] * K)
    for i in range(K):
        d = (lg[i].cpu() - ref[i]).abs().max().item()
        print(f"row {i}: max|d| {d:.4f} argmax {int(lg[i].argmax())}/{int(ref[i].argmax())}")
