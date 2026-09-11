"""Compare the expert-parallel runtime with the pipeline runtime: same tokens (teacher forcing), logits close.
usage: python -m dsv41.test_ep --mode ep --devices 2,3,0,1 --n-layers 8 --out /tmp/ep.pt
       python -m dsv41.test_ep --mode pp --devices 2 --n-layers 8 --out /tmp/pp.pt
       python -m dsv41.test_ep --compare /tmp/ep.pt /tmp/pp.pt"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="ep", choices=["ep", "pp"])
ap.add_argument("--devices", default="2,3,0,1")
ap.add_argument("--n-layers", type=int, default=8)
ap.add_argument("--out", default="")
ap.add_argument("--compare", nargs=2)
a = ap.parse_args()
if a.compare:
    x, y = torch.load(a.compare[0]), torch.load(a.compare[1])
    for i, (lx, ly) in enumerate(zip(x, y)):
        print(f"step {i}: argmax {int(lx.argmax())} vs {int(ly.argmax())}  max|dlogit| {(lx - ly).abs().max().item():.4f}  (|logit| max {lx.abs().max().item():.1f})")
    sys.exit(0)
from transformers import AutoTokenizer
from dsv41.load import load_model
ckpt = "/mnt/ssd/models/DeepSeek-V4.1-Flash"
tok = AutoTokenizer.from_pretrained(ckpt)
devs = [int(d) for d in a.devices.split(",")]
model = load_model(ckpt, devs, max_seq_len=4096, n_layers=a.n_layers, engram=True, tokenizer=tok, ep=(a.mode == "ep"))
ids = tok.encode("The capital of France is Paris. The capital of Germany is")
if a.mode == "ep":
    from dsv41.ep import EPRuntime
    rt = EPRuntime(model, use_graphs=True)
else:
    from dsv41.decode import DecodeRuntime
    rt = DecodeRuntime(model, use_graphs=True)
rt.capture()
model.forward(torch.tensor([ids]), 0)
pos = len(ids)
feed = [19920, 36300, 118795, 28859, 17706, 100003, 2000, 3000, 4000, 5000]  # fixed tokens (teacher forcing)
outs = []
for t in feed:
    outs.append(rt.step(t, pos).clone().cpu())
    pos += 1
torch.save(outs, a.out)
print("saved", a.out)
