"""Expert routing telemetry per task: how much of the routing weight mass the top-N experts of each layer
cover, and how different the expert distributions of different tasks are (Jensen-Shannon divergence).
usage: DSV41_ROUTE_LOG=1 python -m dsv41.route_telemetry --devices 2,0,1,4,5,6,7,3 --tokens 300"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DSV41_ROUTE_LOG"] = "1"
import torch
from dsv41.load import load_model
from dsv41.decode import DecodeRuntime

TASKS = {
    "japanese": "日本の江戸時代の身分制度について、農民の暮らしを中心に800字程度で説明してください。",
    "coding": "Write a Python module that implements an LRU cache with TTL expiry, thread safety, and unit tests. Explain the design choices.",
    "math": "Prove that there are infinitely many primes, then explain the distribution of primes and the prime number theorem with examples.",
    "english": "Write a short essay about how cities can adapt to climate change, covering transport, housing and green spaces.",
    "translation": "Translate the following into Japanese and then into French: 'The committee postponed its decision until the financial audit is complete, citing concerns about transparency and the reliability of the reported figures.'",
}
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
ap.add_argument("--devices", default="2,0,1,4,5,6,7,3")
ap.add_argument("--tokens", type=int, default=300)
ap.add_argument("--out", default="results/route_telemetry.pt")
a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.ckpt)
sys.path.insert(0, os.path.join(a.ckpt, "encoding"))
from encoding import encode_messages
model = load_model(a.ckpt, [int(d) for d in a.devices.split(",")], max_seq_len=8192, engram=True, tokenizer=tok)
rt = DecodeRuntime(model, use_graphs=True)
rt.capture()
E = model.args.n_routed_experts
nl = len(model.blocks)
mass = {t: torch.zeros(nl, E, dtype=torch.float64) for t in TASKS}
count = {t: torch.zeros(nl, E, dtype=torch.float64) for t in TASKS}
for task, prompt in TASKS.items():
    ids = tok.encode(encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat"))
    logits = model.forward(torch.tensor([ids]), 0)
    pos = len(ids)
    nxt = int(logits.argmax(-1))
    out = []
    for _ in range(a.tokens):
        out.append(nxt)
        if nxt == tok.eos_token_id:
            break
        logits = rt.step(nxt, pos)
        eid, wt = rt.route_snapshot()
        for l in range(nl):
            mass[task][l].index_add_(0, eid[l], wt[l].double())
            count[task][l].index_add_(0, eid[l], torch.ones(eid.shape[1], dtype=torch.float64))
        pos += 1
        nxt = int(logits.argmax(-1))
    print(f"[{task}] {len(out)} tokens: {tok.decode(out)[:120]!r}", flush=True)
torch.save({"mass": mass, "count": count, "tasks": TASKS}, a.out)

def coverage(m, N):
    p = m / m.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return p.sort(dim=1, descending=True).values[:, :N].sum(dim=1)  # [layers]

print("\nrouting weight mass covered by the top-N experts of each layer (mean over layers, [min-max]):")
print(f"{'task':12s}" + "".join(f"{'N=' + str(N):>16s}" for N in (32, 64, 96, 128, 192)))
for t in TASKS:
    row = f"{t:12s}"
    for N in (32, 64, 96, 128, 192):
        c = coverage(mass[t], N)
        row += f"{c.mean() * 100:8.1f}% [{c.min() * 100:3.0f}-{c.max() * 100:3.0f}]"
    print(row)
print("\nsame, by selection count:")
for t in TASKS:
    row = f"{t:12s}"
    for N in (32, 64, 96, 128, 192):
        c = coverage(count[t], N)
        row += f"{c.mean() * 100:8.1f}% [{c.min() * 100:3.0f}-{c.max() * 100:3.0f}]"
    print(row)
# cross-task: what a task's top-80 covers of another task's mass, and JS divergence
def top_set(m, N):
    return m.sort(dim=1, descending=True).indices[:, :N]
print("\ncross-task: mass of column task covered by the top-80 experts (per layer) of the row task:")
names = list(TASKS)
print(f"{'':12s}" + "".join(f"{n:>12s}" for n in names))
for t in names:
    S = top_set(mass[t], 80)
    row = f"{t:12s}"
    for u in names:
        p = mass[u] / mass[u].sum(dim=1, keepdim=True).clamp_min(1e-12)
        cov = p.gather(1, S).sum(dim=1).mean()
        row += f"{cov * 100:11.1f}%"
    print(row)
def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda x, y: (x * (x.clamp_min(1e-12) / y.clamp_min(1e-12)).log()).sum(dim=1)
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)) / math.log(2)
print("\nJensen-Shannon divergence (bits, mean over layers) between task expert distributions:")
print(f"{'':12s}" + "".join(f"{n:>12s}" for n in names))
for t in names:
    p = mass[t] / mass[t].sum(dim=1, keepdim=True).clamp_min(1e-12)
    row = f"{t:12s}"
    for u in names:
        q = mass[u] / mass[u].sum(dim=1, keepdim=True).clamp_min(1e-12)
        row += f"{js(p, q).mean():12.3f}"
    print(row)
jl = js(mass['coding'] / mass['coding'].sum(1, keepdim=True), mass['japanese'] / mass['japanese'].sum(1, keepdim=True))
print("\nJS(coding, japanese) per layer:", " ".join(f"{v:.2f}" for v in jl.tolist()))
c64 = coverage(mass['coding'], 64)
print("coding top-64 coverage per layer:", " ".join(f"{v * 100:.0f}" for v in c64.tolist()))
