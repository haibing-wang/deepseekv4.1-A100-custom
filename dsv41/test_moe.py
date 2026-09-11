import sys, time
import torch
sys.path.insert(0, "/mnt/ssdraid/git/deepseekv4.1")
from dsv41.moe_kernels import GroupedPairs, grouped_fp4_gemm
from dsv41.quant import dequant_fp4

torch.manual_seed(0)
dev = "cuda"
E, N, K = 32, 2304, 5120  # a slice of one layer's experts (w1 shape)
w = torch.randint(0, 256, (E, N, K // 2), device=dev, dtype=torch.uint8)
s = torch.randint(110, 135, (E, N, K // 32), device=dev, dtype=torch.uint8)


def reference(a, pairs_raw, n_out):
    out = torch.zeros(n_out, N, device=dev, dtype=torch.float32)
    for e, ri, ro, wt in pairs_raw:
        out[ro] += wt * (a[ri].float() @ dequant_fp4(w[e], s[e]).float().T)
    return out


for M, topk in [(1, 6), (5, 6), (64, 6), (512, 6)]:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    eid = torch.randint(0, E, (M, topk), device=dev)
    wt = torch.rand(M, topk, device=dev)
    row_in = torch.arange(M, device=dev).repeat_interleave(topk)
    pairs = GroupedPairs(eid.flatten(), row_in, row_in.clone(), wt.flatten(), block_m=16 if M <= 16 else 64)
    out = grouped_fp4_gemm(a, w, s, pairs, M)
    ref = reference(a, list(zip(eid.flatten().tolist(), row_in.tolist(), row_in.tolist(), wt.flatten().tolist())), M)
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    torch.cuda.synchronize(); t = time.perf_counter(); it = 20
    for _ in range(it):
        grouped_fp4_gemm(a, w, s, pairs, M)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / it
    n_exp = len(set(eid.flatten().tolist()))
    mb = n_exp * (N * K // 2 + N * K // 32)
    print(f"M={M:4d} topk={topk} experts_touched={n_exp:3d}: rel err {err:.1e} {'OK' if err < 2e-3 else 'FAIL'}  {dt*1e6:7.0f} us  {mb/dt/1e9:6.0f} GB/s  {2*M*topk*N*K/dt/1e12:5.1f} TFLOPS")
