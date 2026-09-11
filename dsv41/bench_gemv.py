"""Explore GEMV tile shapes and the cost of E2M1 decode for the decode path (M=1)."""
import sys, time
import torch, triton, triton.language as tl
sys.path.insert(0, "/mnt/ssdraid/git/deepseekv4.1")
from dsv41.kernels import _decode_e2m1, fp4_gemm
from dsv41.quant import dequant_fp4

dev = "cuda"
N, K = 2304, 5120
packed = torch.randint(0, 256, (N, K // 2), device=dev, dtype=torch.uint8)
scale = torch.randint(110, 135, (N, K // 32), device=dev, dtype=torch.uint8)
x = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
ref = x.float() @ dequant_fp4(packed, scale).float().T


@triton.jit
def gemv_k(A, B, S, C, N, K, stride_bn, stride_sn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, DECODE: tl.constexpr):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rkh = tl.arange(0, BLOCK_K // 2)
    n_mask = rn < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    kps = K // SPLIT_K
    for k0 in range(pid_k * kps, (pid_k + 1) * kps, BLOCK_K):
        packed = tl.load(B + rn[:, None] * stride_bn + (k0 // 2 + rkh)[None, :], mask=n_mask[:, None], other=0).to(tl.int32)
        rg = tl.arange(0, BLOCK_K // 32)
        sb = tl.load(S + rn[:, None] * stride_sn + (k0 // 32 + rg)[None, :], mask=n_mask[:, None], other=0).to(tl.int32)
        sb = tl.reshape(tl.broadcast_to(tl.reshape(sb, (BLOCK_N, BLOCK_K // 32, 1)), (BLOCK_N, BLOCK_K // 32, 16)), (BLOCK_N, BLOCK_K // 2))
        a_lo = tl.load(A + k0 + 2 * rkh).to(tl.float32)
        a_hi = tl.load(A + k0 + 2 * rkh + 1).to(tl.float32)
        if DECODE == 1:
            w_lo = _decode_e2m1(packed & 0x0F, sb)
            w_hi = _decode_e2m1(packed >> 4, sb)
        else:  # lower bound: no real decode
            w_lo = (packed & 0x0F).to(tl.float32)
            w_hi = (packed >> 4).to(tl.float32) + sb.to(tl.float32)
        acc += tl.sum(a_lo[None, :] * w_lo, axis=1) + tl.sum(a_hi[None, :] * w_hi, axis=1)
    if SPLIT_K == 1:
        tl.store(C + rn, acc, mask=n_mask)
    else:
        tl.atomic_add(C + rn, acc, mask=n_mask)


def run(bn, bk, split, decode, warps):
    c = torch.zeros(N, device=dev, dtype=torch.float32)
    grid = (triton.cdiv(N, bn), split)
    f = lambda: gemv_k[grid](x, packed, scale, c, N, K, packed.stride(0), scale.stride(0), BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, DECODE=decode, num_warps=warps)
    f(); torch.cuda.synchronize()
    if decode == 1 and split == 1:
        err = (c - ref[0]).abs().max().item() / ref.abs().max().item(); assert err < 1e-3, err
    t = time.perf_counter(); it = 50
    for _ in range(it):
        if split > 1: c.zero_()
        f()
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / it
    mb = N * K // 2 + N * K // 32
    print(f"BN={bn:3d} BK={bk:4d} split={split} warps={warps} decode={decode}: {dt*1e6:6.1f} us  {mb/dt/1e9:6.0f} GB/s")

for decode in (0, 1):
    for bn, bk, split, warps in [(8, 256, 1, 4), (16, 128, 1, 4), (32, 128, 1, 4), (64, 64, 1, 4), (64, 128, 1, 8), (128, 64, 1, 8), (32, 256, 2, 4), (64, 128, 2, 8), (16, 512, 1, 8)]:
        try: run(bn, bk, split, decode, warps)
        except Exception as e: print("fail", bn, bk, split, warps, type(e).__name__, str(e)[:80])
