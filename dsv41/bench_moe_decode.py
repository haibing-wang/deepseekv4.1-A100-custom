"""Decode-shape (M=1, 6 experts) MoE kernel variants: current bit-decode vs LUT decode, tile shapes."""
import sys, time
import torch, triton, triton.language as tl
sys.path.insert(0, "/mnt/ssdraid/git/deepseekv4.1")
from dsv41.moe_kernels import GroupedPairs, grouped_fp4_gemm, _decode_e2m1_bf16
from dsv41.quant import dequant_fp4

dev = "cuda"
torch.manual_seed(0)
E, N, K = 8, 4608, 5120  # w13 of one layer (subset of experts)
w = torch.randint(0, 256, (E, N, K // 2), device=dev, dtype=torch.uint8)
s = torch.randint(110, 135, (E, N, K // 32), device=dev, dtype=torch.uint8)
a = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
eid = torch.tensor([0, 1, 2, 3, 4, 5], device=dev)
row_in = torch.zeros(6, dtype=torch.long, device=dev); row_out = torch.arange(6, device=dev)
pairs = GroupedPairs(eid, row_in, row_out, torch.ones(6, device=dev), 16)
ref = torch.stack([a.float() @ dequant_fp4(w[e], s[e]).float().T for e in range(6)]).squeeze(1)

# 256-entry LUT: byte -> (bf16 bits of low nibble value) | (bf16 bits of high nibble value) << 16, unscaled
vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
bits = vals.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
lut = torch.tensor([(bits[b & 0xF] | (bits[b >> 4] << 16)).item() for b in range(256)], dtype=torch.int32, device=dev)


@triton.jit
def lut_kernel(A, W, S, LUT, OUT, ROW_IN, ROW_OUT, TILE_EXPERT, TILE_START, TILE_COUNT, N, K,
               stride_am, stride_we, stride_wn, stride_se, stride_sn, stride_om,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    tile = tl.program_id(0); pid_n = tl.program_id(1); pid_k = tl.program_id(2)
    expert = tl.load(TILE_EXPERT + tile); start = tl.load(TILE_START + tile); count = tl.load(TILE_COUNT + tile)
    lm = tl.arange(0, BLOCK_M); p_mask = lm < count; pr = start + lm
    rin = tl.load(ROW_IN + pr, mask=p_mask, other=0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = rn < N
    rkh = tl.arange(0, BLOCK_K // 2); rg = tl.arange(0, BLOCK_K // 32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    kps = K // SPLIT_K
    w_base = W + expert.to(tl.int64) * stride_we; s_base = S + expert.to(tl.int64) * stride_se
    for k0 in range(pid_k * kps, (pid_k + 1) * kps, BLOCK_K):
        a = tl.load(A + rin[:, None] * stride_am + (k0 + tl.arange(0, BLOCK_K))[None, :], mask=p_mask[:, None], other=0.0)
        packed = tl.load(w_base + rn[:, None] * stride_wn + (k0 // 2 + rkh)[None, :], mask=n_mask[:, None], other=0).to(tl.int32)
        pair = tl.load(LUT + packed)  # int32: lo bf16 bits | hi bf16 bits << 16
        sb = tl.load(s_base + rn[:, None] * stride_sn + (k0 // 32 + rg)[None, :], mask=n_mask[:, None], other=127).to(tl.int32)
        scale = tl.exp2((sb - 127).to(tl.float32))  # [BLOCK_N, BLOCK_K/32]
        scale = tl.reshape(tl.broadcast_to(tl.reshape(scale, (BLOCK_N, BLOCK_K // 32, 1)), (BLOCK_N, BLOCK_K // 32, 16)), (BLOCK_N, BLOCK_K // 2))
        lo = (pair & 0xFFFF).to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32) * scale
        hi = (pair >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32) * scale
        wv = tl.interleave(lo.to(tl.bfloat16), hi.to(tl.bfloat16))
        acc = tl.dot(a, tl.trans(wv), acc)
    rout = tl.load(ROW_OUT + pr, mask=p_mask, other=0)
    o_ptrs = OUT + rout[:, None].to(tl.int64) * stride_om + rn[None, :]
    tl.atomic_add(o_ptrs, acc, mask=p_mask[:, None] & n_mask[None, :])


def run_lut(bm, bn, bk, split, warps, stages):
    out = torch.zeros(6, N, device=dev, dtype=torch.float32)
    grid = (6, triton.cdiv(N, bn), split)
    f = lambda: lut_kernel[grid](a, w, s, lut, out, pairs.row_in, pairs.row_out, pairs.tile_expert, pairs.tile_start, pairs.tile_count, N, K,
                                 a.stride(0), w.stride(0), w.stride(1), s.stride(0), s.stride(1), out.stride(0),
                                 BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, num_warps=warps, num_stages=stages)
    f(); torch.cuda.synchronize()
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    t = time.perf_counter(); it = 30
    for _ in range(it): out.zero_(); f()
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / it
    mb = 6 * (N * K // 2 + N * K // 32)
    print(f"LUT bm={bm} bn={bn:3d} bk={bk:3d} split={split} w={warps} st={stages}: {dt*1e6:6.0f} us {mb/dt/1e9:5.0f} GB/s err {err:.1e}")


out = grouped_fp4_gemm(a, w, s, pairs, 6); torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(30): grouped_fp4_gemm(a, w, s, pairs, 6)
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 30
print(f"current grouped kernel: {dt*1e6:6.0f} us {6*(N*K//2+N*K//32)/dt/1e9:5.0f} GB/s err {((out-ref).abs().max()/ref.abs().max()).item():.1e}")
for cfg in [(16, 64, 128, 2, 4, 3), (16, 64, 128, 4, 4, 3), (16, 32, 128, 4, 4, 3), (16, 64, 256, 2, 4, 2), (16, 128, 128, 2, 8, 3), (16, 64, 64, 4, 4, 4), (16, 32, 256, 4, 4, 2), (16, 64, 128, 8, 4, 3)]:
    try: run_lut(*cfg)
    except Exception as e: print("fail", cfg, type(e).__name__, str(e)[:100])

# ---- CUDA C GEMV (ctypes-launched cubin)
from dsv41 import cukern
row_in32 = row_in.to(torch.int32); eid32 = eid.to(torch.int32); wt1 = torch.ones(6, device=dev)
o = cukern.fp4_gemv_pairs(a, w, s, row_in32, eid32, wt1, 6); torch.cuda.synchronize()
print("cuda gemv err", ((o - ref).abs().max() / ref.abs().max()).item())
t = time.perf_counter()
for _ in range(50): cukern.fp4_gemv_pairs(a, w, s, row_in32, eid32, wt1, 6)
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 50
print(f"cuda gemv: {dt*1e6:6.0f} us {6*(N*K//2+N*K//32)/dt/1e9:5.0f} GB/s")
