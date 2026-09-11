"""Grouped FP4 expert GEMM with large groups (fp4_tcw.cu, up to 64 tokens per expert) vs the exact reference and vs
the 16-token kernel with split groups (the previous behaviour), for a realistic mix of group sizes."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dsv41 import cukern
from dsv41.quant import dequant_fp4, fake_quant_fp8

dev = torch.device("cuda:6")
torch.manual_seed(0)
E, N, K = 48, 4608, 5120
w = torch.randint(0, 256, (E, N, K // 2), device=dev, dtype=torch.uint8)
s = torch.randint(112, 128, (E, N, K // 32), device=dev, dtype=torch.uint8)
wb = torch.stack([dequant_fp4(w[e], s[e]) for e in range(E)]).float()

def gtime(fn, it=10):
    st = torch.cuda.Stream(device=dev)
    with torch.cuda.device(dev), torch.cuda.stream(st):
        for _ in range(3): fn()
        torch.cuda.synchronize(dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            for _ in range(it): fn()
    torch.cuda.synchronize(dev)
    g.replay(); torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(dev)
    return (time.perf_counter() - t0) / (5 * it)

def groups(sizes, split):
    """group tables for experts 0.. with the given token counts, runs split at `split` tokens"""
    ge, gs, tok = [], [0], []
    T = 0
    for e, m in enumerate(sizes):
        toks = list(range(T, T + m)); T += m
        for i in range(0, m, split):
            ge.append(e); tok += toks[i:i + split]; gs.append(len(tok))
    t = lambda v: torch.tensor(v, device=dev, dtype=torch.int32)
    return t(ge), t(gs), t(tok), T

cases = {"uniform 1": [1] * 48, "uniform 8": [8] * 48, "uniform 16": [16] * 24, "uniform 32": [32] * 12, "uniform 64": [64] * 6,
         "decode B=32 mix": [1] * 24 + [2] * 12 + [3] * 6 + [6] * 3 + [12, 20, 40],
         "verify 192 rows mix": [2] * 16 + [4] * 12 + [8] * 8 + [16] * 6 + [32] * 4 + [64] * 2}
print(f"{'case':22s} {'pairs':>5s} {'err new':>9s} {'err old':>9s} {'new us':>7s} {'GB/s':>5s} {'old us':>7s} {'GB/s':>5s}")
for name, sizes in cases.items():
    ge, gs, tok, T = groups(sizes, 64)
    ge16, gs16, tok16, _ = groups(sizes, 16)
    x = fake_quant_fp8(torch.randn(T, K, device=dev, dtype=torch.bfloat16), 32)
    xp = cukern.permute_x(x)
    P = T
    ref = torch.cat([x[tok[gs[i]:gs[i + 1]].long()].float() @ wb[ge[i]].T for i in range(len(ge))]) if False else None
    # reference in the (split-64) pair order
    ref = torch.empty(P, N, device=dev)
    for i in range(len(ge)):
        a, b = int(gs[i]), int(gs[i + 1]); ref[a:b] = x[tok[a:b].long()].float() @ wb[int(ge[i])].T
    cukern.FP4_W_LAYOUT = True
    y = cukern.fp4_gemm_tc(xp, w, s, ge, gs, tok, P, min(max(sizes), 64))
    e1 = ((y - ref).abs().max() / ref.abs().max()).item()
    cukern.FP4_W_LAYOUT = False
    y0 = cukern.fp4_gemm_tc(xp, w, s, ge16, gs16, tok16, P, min(max(sizes), 16))
    # old path: pairs in split-16 order; same token order here (tokens are consecutive) so rows match
    e0 = ((y0 - ref).abs().max() / ref.abs().max()).item()
    cukern.FP4_W_LAYOUT = True
    d1 = gtime(lambda: cukern.fp4_gemm_tc(xp, w, s, ge, gs, tok, P, min(max(sizes), 64)))
    cukern.FP4_W_LAYOUT = False
    d0 = gtime(lambda: cukern.fp4_gemm_tc(xp, w, s, ge16, gs16, tok16, P, min(max(sizes), 16)))
    nbytes = len(sizes) * N * (K // 2 + K // 32)
    print(f"{name:22s} {P:5d} {e1:9.2e} {e0:9.2e} {d1*1e6:7.1f} {nbytes/d1/1e9:5.0f} {d0*1e6:7.1f} {nbytes/d0/1e9:5.0f}")
