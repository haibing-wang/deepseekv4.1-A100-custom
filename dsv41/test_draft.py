"""Batched DSpark draft (DSparkRows) vs the eager reference (DSpark) on one sequence: same drafts expected."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dsv41.load import load_model, Checkpoint
from dsv41.decode import DecodeRuntime
from dsv41.dspark import DSpark, DSparkRows
from transformers import AutoTokenizer
ckpt = "/mnt/ssd/models/DeepSeek-V4.1-Flash"
tok = AutoTokenizer.from_pretrained(ckpt)
nl = int(os.environ.get("NL", "8"))
model = load_model(ckpt, [2], max_seq_len=4096, max_batch=1, n_layers=nl, engram=True, tokenizer=tok)
rt = DecodeRuntime(model, use_graphs=False)
last = model.blocks[-1].device
ck = Checkpoint(ckpt)
ref = DSpark(ck, model.args, last, model.embed, model.head, model.shared, nl)
new = DSparkRows(ck, model.args, last, model.embed, model.head, model.shared, nl, rt)
targets = [nl - 3, nl - 2, nl - 1]
ref.targets = new.targets = targets
model.collect_main_hidden = targets
ids = tok.encode("The capital of France is Paris. The capital of Germany is")
logits = model.forward(torch.tensor([ids]), 0)
T = len(ids)
mh = model.main_hidden[0]  # [T, 3*dim]
ref.write_main(mh, 0)
new.write_main_rows(mh, torch.zeros(T, dtype=torch.int64, device=last), torch.arange(T, device=last))
for blk_r, blk_n in zip(ref.blocks, new.blocks):
    d = (blk_r.attn.window_kv_cache.float() - blk_n.attn.window_kv_cache.float()).abs().max().item()
    print("main ring max|d|", d)
token = int(logits.argmax(-1))
d_ref, conf = ref.draft(token, T - 1)
d_new = new.draft_rows(torch.tensor([token], device=last), torch.tensor([T - 1], device=last), mh[-1:].clone(), torch.tensor([T - 1], device=last))
print("ref drafts", d_ref)
print("new drafts", d_new.tolist()[0])
