"""Tiled FP8 GEMM (fp8_tcg.cu, ldmatrix + cp.async, k-permuted weights) vs the reference for M = 32..192."""
import sys, os, time, ctypes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from dsv41 import cukern
from dsv41.cukern import get_function, launch, _cuda, _check
from dsv41.quant import dequant_fp8_block, fake_quant_fp8

dev = torch.device("cuda:6")
torch.manual_seed(0)
PERM = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]

from dsv41.w8 import tile, TILED as W8_TILED
def permute_k(w8):
    N, K = w8.shape
    p = w8.view(N, K // 16, 16)[:, :, PERM].reshape(N, K).contiguous()
    return tile(p) if W8_TILED else p

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

f = get_function("fp8_tcg.cu", "fp8_gemm_tcg", dev)
SHARED = 2 * (128 * 128 + 64 * 256)
_check(_cuda.cuFuncSetAttribute(f, 8, ctypes.c_int(SHARED)), "attr")

def tcg(x, wp, s, N, K, gc, splits):
    M = x.shape[0] // (N // gc) if gc else x.shape[0]
    kps = -(-K // splits); kps = -(-kps // 128) * 128; splits = -(-K // kps)
    part = torch.empty(splits, M, N, device=dev, dtype=torch.float32)
    y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    counters = cukern._tile_counters(dev, (N // 128) * ((M + 63) // 64))
    args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_int(K), ctypes.c_int(M), ctypes.c_void_p(wp.data_ptr()), ctypes.c_void_p(s.data_ptr()),
            ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(K // 32), ctypes.c_void_p(part.data_ptr()), ctypes.c_int(N), ctypes.c_int(kps), ctypes.c_int(gc),
            ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(counters.data_ptr()), ctypes.c_int(splits), ctypes.c_int(1 if W8_TILED else 0)]
    launch(f, (N // 128, (M + 63) // 64, splits), (256, 1, 1), args, dev, shared=SHARED)
    return y

print(f"{'case':30s} {'err':>9s} " + " ".join(f"{'s'+str(sp):>13s}" for sp in [1, 2, 4, 8]) + "   old-layout us")
for (M, N, K, gc) in [(32, 5120, 8192, 0), (64, 5120, 8192, 0), (96, 5120, 8192, 0), (192, 5120, 8192, 0), (64, 4608, 5120, 0), (192, 4608, 5120, 0),
                      (64, 1280, 5120, 0), (192, 1280, 5120, 0), (64, 8192, 4096, 1024), (192, 8192, 4096, 1024), (48, 32768, 1280, 0), (192, 5120, 2304, 0)]:
    w, s = rand_fp8(N, K)
    wp = permute_k(w)
    Mx = M * (N // gc) if gc else M
    x = fake_quant_fp8(torch.randn(Mx, K, device=dev, dtype=torch.bfloat16) * 2, 32)
    wb = dequant_fp8_block(w.view(torch.float8_e4m3fn), s)
    ref = torch.einsum("bgd,grd->bgr", x.view(M, N // gc, K), wb.view(N // gc, gc, K)).reshape(M, N) if gc else F.linear(x, wb)
    y = tcg(x, wp, s, N, K, gc, 4)
    err = ((y.float() - ref.float()).abs().max() / ref.abs().max()).item()
    ts = [gtime(lambda sp=sp: tcg(x, wp, s, N, K, gc, sp)) for sp in [1, 2, 4, 8]]
    cukern.FP8_G_LAYOUT = False
    told = gtime(lambda: cukern.fp8_gemm_tc(x, wp, s, group_cols=gc, tiled=W8_TILED))
    cukern.FP8_G_LAYOUT = True
    tnew = gtime(lambda: cukern.fp8_gemm_tc(x, wp, s, group_cols=gc, tiled=W8_TILED))
    assert cukern.fp8_gemm_tc(x, wp, s, group_cols=gc, tiled=W8_TILED).float().sub(ref.float()).abs().max() <= 2 * (y.float() - ref.float()).abs().max() + 1
    print(f"M={M:3d} N={N:5d} K={K:4d} gc={gc:4d}   {err:9.2e} " + " ".join(f"{t*1e6:6.1f}({N*K/t/1e9:4.0f})" for t in ts) + f"   {told*1e6:6.1f}")
