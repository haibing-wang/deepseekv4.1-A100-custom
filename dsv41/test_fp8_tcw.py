"""Swapped-layout FP8 GEMM (fp8_tcw.cu, up to 64 rows per pass) vs the first layout and the bf16 reference."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from dsv41 import cukern
from dsv41.quant import dequant_fp8_block, fake_quant_fp8
from dsv41.w8 import permute_k, tile, TILED as W8_TILED

dev = torch.device("cuda:6")
torch.manual_seed(0)

def rand_fp8(N, K):
    w = torch.randint(0, 256, (N, K), device=dev, dtype=torch.uint8)
    w[(w & 0x7F) == 0x7F] = 0
    s = torch.randint(110, 130, (N // 32, K // 32), device=dev, dtype=torch.uint8)
    return w, s

def gtime(fn, it=20):
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

def run(x, w, s, gc, wl):
    cukern.FP8_W_LAYOUT = wl
    return cukern.fp8_gemm_tc(x, wp, s, group_cols=gc, tiled=W8_TILED)

print(f"{'case':32s} {'err W':>9s} {'err old':>9s} {'new us':>8s} {'GB/s':>6s} {'old us':>8s} {'GB/s':>6s}")
for (M, N, K, gc) in [(1, 1280, 5120, 0), (6, 1280, 5120, 0), (8, 5120, 8192, 0), (16, 5120, 8192, 0), (32, 5120, 8192, 0), (48, 5120, 8192, 0), (64, 5120, 8192, 0),
                      (32, 4608, 5120, 0), (64, 4608, 5120, 0), (8, 8192, 4096, 1024), (32, 8192, 4096, 1024), (64, 8192, 4096, 1024), (1, 32768, 1280, 0), (48, 32768, 1280, 0)]:
    w, s = rand_fp8(N, K)
    Mx = M * (N // gc) if gc else M
    x = fake_quant_fp8(torch.randn(Mx, K, device=dev, dtype=torch.bfloat16) * 2, 32)
    wb = dequant_fp8_block(w.view(torch.float8_e4m3fn), s)
    wp = tile(permute_k(w)) if W8_TILED else permute_k(w)
    if gc:
        ref = torch.einsum("bgd,grd->bgr", x.view(M, N // gc, K), wb.view(N // gc, gc, K)).reshape(M, N)
    else:
        ref = F.linear(x, wb)
    y1 = run(x, wp, s, gc, True); y0 = run(x, wp, s, gc, False)
    e1 = ((y1.float() - ref.float()).abs().max() / ref.abs().max()).item()
    e0 = ((y0.float() - ref.float()).abs().max() / ref.abs().max()).item()
    d1 = gtime(lambda: run(x, wp, s, gc, True)); d0 = gtime(lambda: run(x, wp, s, gc, False))
    print(f"M={M:2d} N={N:5d} K={K:4d} gc={gc:4d}   {e1:9.2e} {e0:9.2e} {d1*1e6:8.1f} {N*K/d1/1e9:6.0f} {d0*1e6:8.1f} {N*K/d0/1e9:6.0f}")
