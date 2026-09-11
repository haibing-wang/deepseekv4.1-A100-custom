"""Unit tests: FP4 GEMM kernel vs slow dequant reference; sparse attention vs dense masked softmax."""
import sys, time
import torch
sys.path.insert(0, "/mnt/ssdraid/git/deepseekv4.1")
from dsv41.quant import dequant_fp4, fake_quant_fp8, fake_quant_fp4, round_e2m1, FP4_TABLE
from dsv41.kernels import fp4_gemm, sparse_attn

torch.manual_seed(0)
dev = "cuda"

# ---- FP4 GEMM
for (M, N, K) in [(1, 2304, 5120), (7, 5120, 2304), (64, 2304, 5120), (300, 5120, 2304)]:
    packed = torch.randint(0, 256, (N, K // 2), device=dev, dtype=torch.uint8)
    scale = torch.randint(110, 135, (N, K // 32), device=dev, dtype=torch.uint8)
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    ref = a.float() @ dequant_fp4(packed, scale).float().T
    out = fp4_gemm(a, packed, scale)
    err = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    print(f"fp4_gemm M={M:4d} N={N} K={K}: max rel err {err:.2e}", "OK" if err < 2e-3 else "FAIL")

# timing: one expert's three GEMMs at decode (M=1) and prefill (M=512)
packed1 = torch.randint(0, 256, (2304, 2560), device=dev, dtype=torch.uint8); s1 = torch.full((2304, 160), 120, device=dev, dtype=torch.uint8)
packed2 = torch.randint(0, 256, (5120, 1152), device=dev, dtype=torch.uint8); s2 = torch.full((5120, 72), 120, device=dev, dtype=torch.uint8)
for M in (1, 8, 64, 512):
    x = torch.randn(M, 5120, device=dev, dtype=torch.bfloat16); y = torch.randn(M, 2304, device=dev, dtype=torch.bfloat16)
    for _ in range(3): fp4_gemm(x, packed1, s1); fp4_gemm(x, packed1, s1); fp4_gemm(y, packed2, s2)
    torch.cuda.synchronize(); t = time.perf_counter(); iters = 20
    for _ in range(iters): fp4_gemm(x, packed1, s1); fp4_gemm(x, packed1, s1); fp4_gemm(y, packed2, s2)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / iters
    bytes_w = 2304 * 2560 * 2 + 5120 * 1152 + 2304 * 160 * 2 + 5120 * 72
    print(f"expert fwd M={M:3d}: {dt*1e6:7.0f} us  ({bytes_w/dt/1e9:6.0f} GB/s weight read, {2*M*3*5120*2304/dt/1e12:5.1f} TFLOPS)")

# ---- sparse attention vs dense
b, s, h, d, n = 1, 5, 4, 64, 40
q = torch.randn(b, s, h, d, device=dev, dtype=torch.bfloat16)
kv = torch.randn(b, n, d, device=dev, dtype=torch.bfloat16)
sink = torch.randn(h, device=dev)
idx = torch.randint(0, n, (b, s, 8), device=dev, dtype=torch.int32); idx[0, 0, :3] = -1
o = sparse_attn(q, kv, sink, idx, d**-0.5)
# dense reference
ref = torch.empty_like(o, dtype=torch.float32)
for i in range(s):
    sel = idx[0, i]; valid = sel >= 0
    kk = kv[0, sel[valid].long()].float()
    sc = (q[0, i].float() @ kk.T) * d**-0.5  # [h, t]
    sc = torch.cat([sc, sink[:, None]], dim=1)
    p = sc.softmax(dim=-1)[:, :-1]
    ref[0, i] = p @ kk
print("sparse_attn max abs err", (o.float() - ref).abs().max().item())

# ---- fp4 rounding sanity: every grid value is a fixed point, midpoints round to even
g = FP4_TABLE.to(dev)
assert torch.equal(round_e2m1(g), g)
print("round_e2m1(0.75, 1.25, 1.75, 2.5, 3.5, 5) =", round_e2m1(torch.tensor([0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dev)).tolist())
x = torch.randn(4, 64, device=dev, dtype=torch.bfloat16) * 3
print("fake_quant_fp8 rel err", ((fake_quant_fp8(x) - x).float().norm() / x.float().norm()).item())
print("fake_quant_fp4 rel err", ((fake_quant_fp4(x) - x).float().norm() / x.float().norm()).item())
