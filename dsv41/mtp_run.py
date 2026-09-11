"""Speculative decoding with DSpark on the batched runtime: S sequences, each step verifies [token, 5 drafts] rows.
usage: python -m dsv41.mtp_run --devices 2,3,0,1 --ep --seqs 8 --prompts dsv41/batch_prompts16.txt --max-new-tokens 64"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dsv41.load import load_model, Checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
ap.add_argument("--devices", default="2,3,0,1")
ap.add_argument("--ep", action="store_true")
ap.add_argument("--ep-shards", default="")
ap.add_argument("--budgets", default="", help="per-GPU GiB for the pipeline placement, e.g. 2:70,3:70")
ap.add_argument("--seqs", type=int, default=8)
ap.add_argument("--prompts", default="dsv41/batch_prompts16.txt")
ap.add_argument("--max-new-tokens", type=int, default=64)
ap.add_argument("--no-mtp", action="store_true", help="plain batched decode with the same harness (baseline)")
ap.add_argument("--n-layers", type=int, default=None, help="truncated model (plumbing test)")
a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.ckpt)
sys.path.insert(0, os.path.join(a.ckpt, "encoding"))
from encoding import encode_messages
S = a.seqs
K = 1 if a.no_mtp else 6
devs = [int(d) for d in a.devices.split(",")]
model = load_model(a.ckpt, devs, max_seq_len=8192, max_batch=S * K, engram=True, tokenizer=tok, ep=a.ep, n_layers=a.n_layers,
                   ep_shards=[int(v) for v in a.ep_shards.split(",")] if a.ep_shards else None,
                   budgets_gb={int(k): float(v) for k, v in (kv.split(":") for kv in a.budgets.split(",") if kv)} or None)
if a.ep:
    from dsv41.ep import EPRuntime
    rt = EPRuntime(model, use_graphs=True)
else:
    from dsv41.decode import DecodeRuntime
    rt = DecodeRuntime(model, use_graphs=True)
if a.n_layers:  # truncated model: read the last three layers instead of DSpark's target layers (plumbing only)
    nl = len(model.blocks)
    tl_ = [nl - 3, nl - 2, nl - 1]
    rt.target_layers = tl_
    rt.main_hid = {lid: torch.zeros(S * K, model.args.dim, dtype=torch.bfloat16, device=model.blocks[lid].device) for lid in tl_}
rt.capture()
last = model.blocks[-1].device
ds = None
if not a.no_mtp:
    from dsv41.dspark import DSparkRows
    ds = DSparkRows(Checkpoint(a.ckpt), model.args, last, model.embed, model.head, model.shared, len(model.blocks), rt)
    if a.n_layers:
        ds.targets = rt.target_layers
    model.collect_main_hidden = ds.targets
lines = [l.strip() for l in open(a.prompts) if l.strip()][:S]
assert len(lines) == S
prompts = [tok.encode(encode_messages([{"role": "user", "content": l}], thinking_mode="chat")) for l in lines]
# prefill one sequence at a time into slot 0, copy the state to its slot (last one stays in slot 0)
p_last, bonus, mh_last = [0] * S, [0] * S, [None] * S
for s in range(S - 1, -1, -1):
    ids = prompts[s]
    logits = model.forward(torch.tensor([ids]), 0)
    if ds is not None:
        T = len(ids)
        ds.write_main_rows(model.main_hidden[0], torch.zeros(T, dtype=torch.int64, device=last), torch.arange(T, device=last))
        mh_last[s] = model.main_hidden[0, -1].clone()
    if s > 0:
        rt.copy_seq(0, s)
        if ds is not None:
            for blk in ds.blocks:
                blk.attn.window_kv_cache[s].copy_(blk.attn.window_kv_cache[0])
    p_last[s] = len(ids) - 1
    bonus[s] = int(logits.argmax(-1))
generated = [[] for _ in range(S)]
written_max = [p_last[s] for s in range(S)]
drafts = None
if ds is not None:
    dr = ds.draft_rows(torch.tensor(bonus, device=last), torch.tensor(p_last, device=last), torch.stack(mh_last), torch.tensor(written_max, device=last))
    drafts = dr.tolist()
torch.cuda.synchronize()
t0 = time.time()
steps = 0
n_acc = 0
done = [False] * S
while steps < a.max_new_tokens and not all(done):
    toks, poss, seqs, pmaxs = [], [], [], []
    for s in range(S):
        row_toks = [bonus[s]] + (drafts[s] if drafts is not None else [])
        for i in range(K):
            toks.append(row_toks[i]); poss.append(p_last[s] + 1 + i); seqs.append(s); pmaxs.append(p_last[s] + K)
    logits = rt.step(toks, poss, seq=seqs, pmax=pmaxs)
    am = logits.argmax(-1).tolist()
    if ds is not None:
        mh_all = torch.cat([rt.main_hid[l].to(last) for l in ds.targets], dim=-1)  # [S*K, 3*dim]
        ds.write_main_rows(mh_all, torch.tensor(seqs, device=last), torch.tensor(poss, device=last))
    new_bonus, new_p, mh_rows = [], [], []
    for s in range(S):
        acc = 0
        if drafts is not None:
            for i in range(K - 1):
                if am[s * K + i] == drafts[s][i]:
                    acc += 1
                else:
                    break
        new_tokens = (drafts[s][:acc] if drafts is not None else []) + [am[s * K + acc]]
        if not done[s]:
            generated[s].extend(new_tokens)
            if tok.eos_token_id in new_tokens:
                done[s] = True
        n_acc += acc
        new_bonus.append(am[s * K + acc])
        new_p.append(p_last[s] + 1 + acc)
        mh_rows.append(s * K + acc)
    written_max = [p_last[s] + K for s in range(S)]
    bonus, p_last = new_bonus, new_p
    if ds is not None:
        dr = ds.draft_rows(torch.tensor(bonus, device=last), torch.tensor(p_last, device=last), mh_all[mh_rows], torch.tensor(written_max, device=last))
        drafts = dr.tolist()
    steps += 1
torch.cuda.synchronize()
dt = time.time() - t0
total = sum(len(g) for g in generated)
print(tok.decode(generated[0][:60]))
print(f"\n[{S} sequences, {steps} steps, {dt:.2f}s] {total} tokens generated: {total / dt:.1f} tok/s aggregate, "
      f"{dt / steps * 1000:.1f} ms/step, {total / steps / S:.2f} tokens per sequence per step" + (f", accepted drafts per step per seq {n_acc / steps / S:.2f}" if ds else ""))
