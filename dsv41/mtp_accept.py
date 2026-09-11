"""Measure DSpark draft acceptance on greedy decoding: after every main step, draft 5 tokens and compare them
with what the main model actually produces next. No verification machinery: pure acceptance statistics.
usage: python -m dsv41.mtp_accept --devices 2,0,1,4,5,6,7,3 --chat --prompt "..." --max-new-tokens 200"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dsv41.load import load_model, Checkpoint
from dsv41.decode import DecodeRuntime
from dsv41.dspark import DSpark

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
ap.add_argument("--devices", default="2,0,1,4,5,6,7,3")
ap.add_argument("--prompt", default="Explain how mixture-of-experts routing works in transformer language models.")
ap.add_argument("--chat", action="store_true")
ap.add_argument("--max-new-tokens", type=int, default=200)
a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.ckpt)
devices = [int(d) for d in a.devices.split(",")]
model = load_model(a.ckpt, devices, max_seq_len=8192, engram=True, tokenizer=tok)
last = model.blocks[-1].device
ds = DSpark(Checkpoint(a.ckpt), model.args, last, model.embed, model.head, model.shared, len(model.blocks))
print("DSpark loaded", flush=True)
if a.chat:
    sys.path.insert(0, os.path.join(a.ckpt, "encoding"))
    from encoding import encode_messages
    ids = tok.encode(encode_messages([{"role": "user", "content": a.prompt}], thinking_mode="chat"))
else:
    ids = tok.encode(a.prompt)
rt = DecodeRuntime(model, use_graphs=True)
rt.capture()
model.collect_main_hidden = ds.targets
logits = model.forward(torch.tensor([ids]), 0)
ds.write_main(model.main_hidden[0], 0)  # main kv for every prompt position
pos = len(ids)
nxt = int(logits.argmax(-1))
drafts = {}  # position p -> drafted ids for positions p+1..p+5 (p = position of the input token of the draft)
generated = []
t_draft = 0.0
for step in range(a.max_new_tokens):
    generated.append(nxt)
    if nxt == tok.eos_token_id:
        break
    # draft from (t_pos = nxt, main state up to pos-1): predictions for positions pos+1 .. pos+5
    t0 = time.time()
    d, conf = ds.draft(nxt, pos - 1)
    torch.cuda.synchronize(last)
    t_draft += time.time() - t0
    drafts[pos] = (d, conf.tolist())
    logits = rt.step(nxt, pos)
    mh = torch.cat([rt.main_hid[l].to(last) for l in ds.targets], dim=-1)
    ds.write_main(mh, pos)
    pos += 1
    nxt = int(logits.argmax(-1))
print(tok.decode(generated))
# acceptance: for draft made at position p (input token = generated token at index p - len(ids)), compare with generated[p+1-len(ids) ...]
base = len(ids)
hist = [0] * (ds.block + 1)
per_pos = [0] * ds.block
n = 0
for p, (d, conf) in drafts.items():
    k = p - base  # index of the input token in `generated`
    actual = generated[k + 1 : k + 1 + ds.block]
    if len(actual) < ds.block:
        continue
    acc = 0
    for i in range(ds.block):
        if d[i] == actual[i]:
            acc += 1
            per_pos[i] += 1
        else:
            break
    hist[acc] += 1
    n += 1
mean_acc = sum(i * c for i, c in enumerate(hist)) / max(n, 1)
print(f"\n[DSpark acceptance over {n} steps] mean accepted drafts {mean_acc:.2f} (+1 bonus = {mean_acc + 1:.2f} tokens/step)")
print("  histogram (accepted 0..5):", hist)
print("  cumulative accuracy of draft position i:", [f"{c / max(n, 1) * 100:.0f}%" for c in per_pos])
print(f"  draft time (eager): {t_draft / max(len(drafts), 1) * 1000:.1f} ms per draft")
