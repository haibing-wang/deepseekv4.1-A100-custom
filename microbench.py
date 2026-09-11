"""A100 raw ceilings for the Qwen3.8-27B MLP shapes: FP16 tensor-core GEMM vs INT8 tensor-core GEMM,
plus HBM bandwidth (decode ceiling). Run: CUDA_VISIBLE_DEVICES=<idle gpu> python3 microbench.py"""
import torch, time

dev = "cuda"
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

def bench(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters

H, FF = 5120, 17408  # hidden, ffn (Qwen3.8-27B)
print(f"{'shape (MxKxN)':>22} | {'fp16 TFLOPS':>11} | {'int8 TOPS':>9} | int8/fp16")
for M in (1, 8, 32, 512, 2048):
    for (K, N, tag) in ((H, 2 * FF, "up+gate"), (FF, H, "down")):
        a16 = torch.randn(M, K, device=dev, dtype=torch.float16)
        b16 = torch.randn(K, N, device=dev, dtype=torch.float16)
        t16 = bench(lambda: a16 @ b16)
        flops = 2 * M * K * N
        if M >= 32:  # torch._int_mm requires M>16-ish and multiples; skip tiny M
            a8 = torch.randint(-12, 13, (M, K), device=dev, dtype=torch.int8)
            b8 = torch.randint(-12, 13, (K, N), device=dev, dtype=torch.int8)
            t8 = bench(lambda: torch._int_mm(a8, b8))
            print(f"{M:>5}x{K:>5}x{N:>5} {tag:>7} | {flops/t16/1e12:11.1f} | {flops/t8/1e12:9.1f} | {t16/t8:6.2f}x")
        else:
            print(f"{M:>5}x{K:>5}x{N:>5} {tag:>7} | {flops/t16/1e12:11.1f} | {'-':>9} |   (bandwidth-bound: {2*K*N/t16/1e9:6.0f} GB/s read)")

# HBM bandwidth
x = torch.empty(2**30, device=dev, dtype=torch.uint8)  # 1 GiB
y = torch.empty_like(x)
t = bench(lambda: y.copy_(x), iters=20)
print(f"\nHBM copy bandwidth: {2 * x.numel() / t / 1e9:.0f} GB/s (read+write)")
print(f"decode ceiling for 18.5 GiB weights: {x.numel()*2/t/1e9/ (18.5*1.0737):.0f} tok/s, for 27.1 GiB: {x.numel()*2/t/1e9/(27.1*1.0737):.0f} tok/s")
