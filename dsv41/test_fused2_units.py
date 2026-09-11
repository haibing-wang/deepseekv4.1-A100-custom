"""Unit tests: each fused2 kernel against the torch/fused-1 composition it replaces."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from dsv41 import fused2 as F2
from dsv41.fused import fake_quant_fp8, hc_split_sinkhorn, rmsnorm, rope_dev_, sparse_attn_decode_split
from dsv41.model import _hc_mix_proj, _hc_post, _hc_pre

dev = torch.device("cuda:2")
torch.manual_seed(0)
hc, D, eps, hc_eps = 4, 5120, 1e-6, 1e-6
h = torch.randn(1, 1, hc, D, device=dev, dtype=torch.bfloat16)
fn = torch.randn(24, hc * D, device=dev) * 0.02
scale = torch.tensor([0.5, 0.3, 0.7], device=dev)
base = torch.randn(24, device=dev)
w = torch.rand(D, device=dev, dtype=torch.bfloat16) + 0.5
pre_in = torch.softmax(torch.randn(hc, device=dev), 0)

def rep(name, a, b):
    a, b = a.float(), b.float()
    print(f"{name:22s} max|d| {(a - b).abs().max().item():.3e}  rel {((a - b).abs().max() / (b.abs().max() + 1e-9)).item():.3e}  (max |b| {b.abs().max().item():.3g})")

# hc_mix
m_ref = _hc_mix_proj(h, fn, eps).flatten()
m_new = F2.hc_mix(h, fn, eps)
rep("hc_mix", m_new, m_ref)
# hc_pre_norm_quant
pre_r, post_r, comb_r = hc_split_sinkhorn(m_ref.view(1, 1, -1), scale, base, hc, 20, hc_eps)
y_r = rmsnorm(_hc_pre(h, pre_in.view(1, 1, hc)), w, eps)
yq_r = fake_quant_fp8(y_r, 32)
pre_n, post_n, comb_n, y_n, yq_n, yf_n = F2.hc_pre_norm_quant(h, pre_in, m_ref, scale, base, w, eps, hc_eps, 20, want_f32=True)
rep("sinkhorn pre", pre_n, pre_r.flatten()); rep("sinkhorn post", post_n, post_r.flatten()); rep("sinkhorn comb", comb_n, comb_r.view(hc, hc))
rep("hc_pre+norm y", y_n.flatten(), y_r.flatten()); rep("hc_pre+norm yq", yq_n.flatten(), yq_r.flatten()); rep("yf", yf_n.flatten(), y_r.flatten())
# norm_quant
x = torch.randn(1, 1280, device=dev, dtype=torch.bfloat16) * 3
wq = torch.rand(1280, device=dev, dtype=torch.bfloat16) + 0.5
rep("norm_quant", F2.norm_quant(x, wq, eps), fake_quant_fp8(rmsnorm(x, wq, eps), 32))
# kv_write
win, rd = 128, 64
cos = torch.randn(4096, rd // 2, device=dev); sin = torch.randn(4096, rd // 2, device=dev)
pos = torch.tensor(300, device=dev)
kvw = torch.rand(512, device=dev, dtype=torch.bfloat16) + 0.5
kvx = torch.randn(1, 512, device=dev, dtype=torch.bfloat16) * 2
cache_r = torch.zeros(1, win, 512, device=dev, dtype=torch.bfloat16); cache_n = cache_r.clone()
kv = rmsnorm(kvx.view(1, 1, 512), kvw, eps).contiguous(); rope_dev_(kv, rd, cos, sin, pos); kv = fake_quant_fp8(kv, 32)
cache_r.index_copy_(1, torch.remainder(pos, win).view(1), kv)
F2.kv_write(kvx, kvw, cos, sin, pos, cache_n, rd, eps, torch.zeros(1, dtype=torch.int64, device=dev))
rep("kv_write", cache_n, cache_r)
# sattn2 (window only, and with compressed cache)
H, hd = 64, 512
q = torch.randn(1, 1, H, hd, device=dev, dtype=torch.bfloat16)
sink = torch.randn(H, device=dev)
cache = torch.randn(1, win, hd, device=dev, dtype=torch.bfloat16)
for p in (50, 300):
    pos = torch.tensor(p, device=dev)
    slot = torch.remainder(pos, win)
    widx = torch.remainder(torch.arange(win, device=dev) + slot + 1, win)
    widx = torch.where(widx > pos, -1, widx).to(torch.int32).view(1, 1, win)
    o_r = sparse_attn_decode_split(q, cache, None, sink, widx, hd**-0.5); rope_dev_(o_r, rd, cos, sin, pos, inverse=True)
    seq1 = torch.zeros(1, dtype=torch.int64, device=dev)
    o_n = F2.sattn2(q, cache, None, None, pos.view(1), sink, cos, sin, rd, hd**-0.5, seq1, pos.view(1))
    rep(f"sattn2 win pos{p}", o_n, o_r)
    ckv = torch.randn(1, 4097, hd, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, 150, (1, 1, 512), device=dev, dtype=torch.int32); idx[..., 400:] = -1
    o_r = sparse_attn_decode_split(q, cache, ckv, sink, torch.cat([widx, torch.where(idx >= 0, idx + win, -1)], -1), hd**-0.5); rope_dev_(o_r, rd, cos, sin, pos, inverse=True)
    o_n = F2.sattn2(q, cache, ckv, idx, pos.view(1), sink, cos, sin, rd, hd**-0.5, seq1, pos.view(1))
    rep(f"sattn2 win+c pos{p}", o_n, o_r)
# gate_topk
E, topk = 384, 6
sc = torch.randn(1, E, device=dev) * 2
bias = torch.randn(E, device=dev) * 0.1
s = F.softplus(sc / 1.0).sqrt(); ind = (s + bias).topk(topk, dim=-1)[1]; wt = s.gather(1, ind); wt = wt / (wt.sum(-1, keepdim=True) + 1e-20) * 2.5
eid_n, wt_n = F2.gate_topk(sc, bias, 1.0, topk, 2.5, "sqrtsoftplus", True)
print("gate ids ref", ind.flatten().tolist(), "new", eid_n.tolist()); rep("gate wt", wt_n, wt.flatten())
# hc_post2
a = torch.randn(1, D, device=dev, dtype=torch.bfloat16)
r_ref = _hc_post(a.view(1, 1, D), h, post_r, comb_r)
r_new = h.clone(); F2.hc_post2_(a, r_new, post_n, comb_n)
rep("hc_post2 attn", r_new, r_ref)
y2 = torch.randn(topk, D, device=dev); ys = torch.randn(1, D, device=dev, dtype=torch.bfloat16)
yy = (y2.view(1, topk, D).sum(1) + ys.float()).to(torch.bfloat16)
r_ref = _hc_post(yy.view(1, 1, D), h, post_r, comb_r)
r_new = h.clone(); F2.hc_post2_(None, r_new, post_n, comb_n, y2=y2, ys=ys)
rep("hc_post2 moe", r_new, r_ref)
