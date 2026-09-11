"""FP8-weight tensor-core GEMM vs the bf16 reference (dequant + cuBLAS), plus bandwidth."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from dsv41 import cukern
from dsv41.quant import dequant_fp8_block, fake_quant_fp8

dev = torch.device("cuda:3")
torch.manual_seed(0)

def rand_fp8(N, K):
    w = torch.randint(0, 256, (N, K), device=dev, dtype=torch.uint8)
    w[(w & 0x7F) == 0x7F] = 0  # no NaN codes
    s = torch.randint(110, 130, (N // 32, K // 32), device=dev, dtype=torch.uint8)
    return w, s

for (M, N, K, gc) in [(1, 512, 5120, 0), (1, 1280, 5120, 0), (1, 32768, 1280, 0), (6, 5120, 8192, 0), (16, 4608, 5120, 0), (8, 8192, 4096, 1024), (1, 5120, 2304, 0)]:
    w, s = rand_fp8(N, K)
    Mx = N // gc if gc else M
    x = fake_quant_fp8(torch.randn(Mx, K, device=dev, dtype=torch.bfloat16) * 2, 32)
    wb = dequant_fp8_block(w.view(torch.float8_e4m3fn), s)
    if gc:
        ref = torch.einsum("gd,grd->gr", x, wb.view(N // gc, gc, K)).reshape(1, N)
    else:
        ref = F.linear(x, wb)
    y = cukern.fp8_gemm_tc(x, w, s, group_cols=gc)
    err = (y.float() - ref.float()).abs().max().item()
    torch.cuda.synchronize(dev)
    it = 20
    def gtime(fn):
        # GPU time per call inside a CUDA graph (what the decode graphs see), 20 calls per graph
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
    dt = gtime(lambda: cukern.fp8_gemm_tc(x, w, s, group_cols=gc))
    dt2 = gtime(lambda: F.linear(x, wb) if not gc else torch.einsum("gd,grd->gr", x, wb.view(N // gc, gc, K)))
    print(f"M={M:2d} N={N:5d} K={K:4d} gc={gc:4d}: max|d| {err:.3e} (ref max {ref.abs().max().item():.1f})  fp8-tc {dt*1e6:6.1f} us ({N*K/dt/1e9:6.0f} GB/s)  bf16 cuBLAS {dt2*1e6:6.1f} us ({2*N*K/dt2/1e9:6.0f} GB/s)")
