import sys, time, torch
sys.path.insert(0, "/mnt/ssdraid/git/deepseekv4.1")
from dsv41 import cukern
from dsv41.quant import dequant_fp8_block
dev = "cuda"; torch.manual_seed(0)
for (M, N, K) in [(1, 32768, 1280), (1, 5120, 8192), (1, 1280, 5120), (1, 4608, 5120)]:
    w = (torch.randn(N, K, device=dev) * 2).to(torch.float8_e4m3fn)
    s = torch.randint(115, 135, ((N + 31) // 32, K // 32), device=dev, dtype=torch.uint8)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    ref = x.float() @ dequant_fp8_block(w, s).float().T
    out = cukern.fp8_gemv(x, w, s); torch.cuda.synchronize()
    err = ((out - ref).abs().max() / ref.abs().max()).item()
    t = time.perf_counter(); it = 50
    for _ in range(it): cukern.fp8_gemv(x, w, s)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / it
    wb = torch.nn.functional.linear(x, dequant_fp8_block(w, s)); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): torch.nn.functional.linear(x, wb.new_empty(0)) if False else torch.nn.functional.linear(x, dequant_fp8_block(w, s)) if False else None
    wbf = dequant_fp8_block(w, s); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): torch.nn.functional.linear(x, wbf)
    torch.cuda.synchronize(); dt2 = (time.perf_counter() - t) / it
    print(f"M={M:2d} N={N:5d} K={K:4d}: err {err:.1e}  fp8 gemv {dt*1e6:6.1f} us ({N*K/dt/1e9:5.0f} GB/s)   cuBLAS bf16 {dt2*1e6:6.1f} us ({2*N*K/dt2/1e9:5.0f} GB/s)")
