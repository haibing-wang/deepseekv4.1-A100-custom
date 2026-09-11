import os, sys, time
os.environ["DSV41_COMPILE"] = "0"
import torch
sys.path.insert(0, "/mnt/ssdraid/git/deepseekv4.1")
from dsv41 import fused, quant, kernels, model

torch.manual_seed(0)
dev = "cuda"


def check(name, a, b, tol=0):
    err = (a.float() - b.float()).abs().max().item()
    print(f"{name:28s} max abs err {err:.3e} {'OK' if err <= tol else 'FAIL'}")


x = torch.randn(3, 7, 5120, device=dev, dtype=torch.bfloat16) * 4
check("fake_quant_fp8", fused.fake_quant_fp8(x, 32), quant.fake_quant_fp8(x, 32))
check("fake_quant_fp4 e8m0", fused.fake_quant_fp4(x, 32), quant.fake_quant_fp4(x, 32))
check("fake_quant_fp4 e4m3", fused.fake_quant_fp4(x[..., :512], 16, True), quant.fake_quant_fp4(x[..., :512], 16, True))
w = torch.randn(5120, device=dev, dtype=torch.bfloat16)
check("rmsnorm", fused.rmsnorm(x, w, 1e-20), model.rmsnorm(x, w, 1e-20), tol=0.0625)
mix = torch.randn(2, 5, 24, device=dev)
sc, ba = torch.randn(3, device=dev), torch.randn(24, device=dev)
p1, q1, c1 = fused.hc_split_sinkhorn(mix, sc, ba, 4, 20, 1e-6)
p2, q2, c2 = kernels.hc_split_sinkhorn(mix, sc, ba, 4, 20, 1e-6)
check("sinkhorn pre", p1, p2, 1e-6); check("sinkhorn post", q1, q2, 1e-6); check("sinkhorn comb", c1, c2, 1e-5)
h = torch.randn(2, 5, 4, 5120, device=dev, dtype=torch.bfloat16)
pre = torch.rand(2, 5, 4, device=dev)
check("hc_pre", fused.hc_pre(h, pre), model._hc_pre(h, pre), 0.0625)
post = torch.rand(2, 5, 4, device=dev); comb = torch.rand(2, 5, 4, 4, device=dev)
xx = torch.randn(2, 5, 5120, device=dev, dtype=torch.bfloat16)
check("hc_post", fused.hc_post(xx, h, post, comb), model._hc_post(xx, h, post, comb), 0.125)

# decode-shape timing: torch chain vs fused
x1 = torch.randn(1, 1, 5120, device=dev, dtype=torch.bfloat16); h1 = torch.randn(1, 1, 4, 5120, device=dev, dtype=torch.bfloat16)
m1 = torch.randn(1, 1, 24, device=dev)
for name, f_t, f_f in [("fake_quant_fp8", lambda: quant.fake_quant_fp8(x1, 32), lambda: fused.fake_quant_fp8(x1, 32)),
                       ("rmsnorm", lambda: model.rmsnorm(x1, w, 1e-20), lambda: fused.rmsnorm(x1, w, 1e-20)),
                       ("sinkhorn", lambda: kernels.hc_split_sinkhorn(m1, sc, ba), lambda: fused.hc_split_sinkhorn(m1, sc, ba)),
                       ("hc_post", lambda: model._hc_post(x1, h1, post[:1, :1], comb[:1, :1]), lambda: fused.hc_post(x1, h1, post[:1, :1], comb[:1, :1]))]:
    res = []
    for f in (f_t, f_f):
        for _ in range(5): f()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(100): f()
        torch.cuda.synchronize(); res.append((time.perf_counter() - t) / 100 * 1e6)
    print(f"{name:16s} torch {res[0]:7.1f} us   fused {res[1]:6.1f} us")

# ---- rope vs reference complex rotation
from dsv41.model import apply_rotary_emb, precompute_freqs_cis
freqs = precompute_freqs_cis(64, 4096, 65536, 160000.0, 16.0, 32, 1, torch.device(dev))
cs = torch.view_as_real(freqs).contiguous(); cos, sin = cs[..., 0].contiguous(), cs[..., 1].contiguous()
q = torch.randn(1, 3, 64, 512, device=dev, dtype=torch.bfloat16)
q1 = q.clone(); apply_rotary_emb(q1[..., -64:], freqs[100:103])
q2 = q.clone(); fused.rope_(q2, 64, cos, sin, 100)
check("rope 4d", q1, q2, 0.0625)
q1 = q.clone(); apply_rotary_emb(q1[..., -64:], freqs[100:103], True)
q2 = q.clone(); fused.rope_(q2, 64, cos, sin, 100, inverse=True)
check("rope 4d inverse", q1, q2, 0.0625)
k = torch.randn(1, 3, 512, device=dev, dtype=torch.bfloat16)
k1 = k.clone(); apply_rotary_emb(k1[..., -64:], freqs[7:10]); k2 = k.clone(); fused.rope_(k2, 64, cos, sin, 7)
check("rope 3d", k1, k2, 0.0625)

# ---- decode sparse attention vs torch version
qd = torch.randn(1, 1, 64, 512, device=dev, dtype=torch.bfloat16)
kvd = torch.randn(1, 700, 512, device=dev, dtype=torch.bfloat16)
sink = torch.randn(64, device=dev)
idx = torch.randint(0, 700, (1, 1, 640), device=dev, dtype=torch.int32); idx[0, 0, 5:40] = -1
o1 = kernels.sparse_attn(qd, kvd, sink, idx, 512**-0.5); o2 = fused.sparse_attn_decode(qd, kvd, sink, idx, 512**-0.5)
check("sparse_attn_decode", o1, o2, 0.02)
for name, f_t, f_f in [("sparse_attn", lambda: kernels.sparse_attn(qd, kvd, sink, idx, 512**-0.5), lambda: fused.sparse_attn_decode(qd, kvd, sink, idx, 512**-0.5)),
                       ("rope", lambda: apply_rotary_emb(q1[..., -64:], freqs[100:103]), lambda: fused.rope_(q2, 64, cos, sin, 100))]:
    res = []
    for f in (f_t, f_f):
        for _ in range(5): f()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(100): f()
        torch.cuda.synchronize(); res.append((time.perf_counter() - t) / 100 * 1e6)
    print(f"{name:16s} torch {res[0]:7.1f} us   fused {res[1]:6.1f} us")

# ---- swiglu_quant vs torch chain
gu = torch.randn(9, 2 * 2304, device=dev) * 5
wt = torch.rand(9, device=dev)
h_ref = model._swiglu(gu[:, :2304], gu[:, 2304:], 10.0)
h_ref = quant.fake_quant_fp8((wt[:, None] * h_ref).to(torch.bfloat16), 32)
check("swiglu_quant", fused.swiglu_quant(gu, wt, 2304, 10.0), h_ref)
h_ref2 = quant.fake_quant_fp8(model._swiglu(gu[:, :2304], gu[:, 2304:], 10.0).to(torch.bfloat16), 32)
check("swiglu_quant (no w)", fused.swiglu_quant(gu, None, 2304, 10.0), h_ref2)

# ---- split decode attention vs single-program version
kv1 = torch.randn(1, 128, 512, device=dev, dtype=torch.bfloat16); kv2 = torch.randn(1, 2049, 512, device=dev, dtype=torch.bfloat16)
idx2 = torch.cat([torch.randint(0, 128, (1, 1, 128), device=dev), 128 + torch.randint(0, 2049, (1, 1, 512), device=dev)], dim=-1).to(torch.int32); idx2[0, 0, 3:20] = -1
o_a = fused.sparse_attn_decode2(qd, kv1, kv2, sink, idx2, 512**-0.5); o_b = fused.sparse_attn_decode_split(qd, kv1, kv2, sink, idx2, 512**-0.5)
check("sparse_attn split", o_a, o_b, 0.02)
for name, f in [("decode2", lambda: fused.sparse_attn_decode2(qd, kv1, kv2, sink, idx2, 512**-0.5)), ("split", lambda: fused.sparse_attn_decode_split(qd, kv1, kv2, sink, idx2, 512**-0.5))]:
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(100): f()
    torch.cuda.synchronize(); print(f"{name:10s} {(time.perf_counter()-t)/100*1e6:6.1f} us")
