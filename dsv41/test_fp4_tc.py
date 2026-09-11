"""FP4 tensor-core expert GEMM vs the exact reference and vs the current GEMV, for M = 1..16 tokens per expert."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dsv41 import cukern
from dsv41.quant import dequant_fp4, fake_quant_fp8

dev = torch.device("cuda:2")
torch.manual_seed(0)
E, N, K = 8, 4608, 5120
w = torch.randint(0, 256, (E, N, K // 2), device=dev, dtype=torch.uint8)
s = torch.randint(112, 128, (E, N, K // 32), device=dev, dtype=torch.uint8)
wb = torch.stack([dequant_fp4(w[e], s[e]) for e in range(E)]).float()  # [E, N, K]

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

print(f"{'case':28s} {'max rel err':>12s} {'tc us':>8s} {'tc GB/s':>8s} {'gemv us':>8s} {'gemv GB/s':>9s}")
for (G, M) in [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16), (6, 1), (6, 2), (6, 4), (6, 6)]:
    x = fake_quant_fp8(torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 2, 32)
    xp = cukern.permute_x(x)
    experts = torch.arange(G, device=dev, dtype=torch.int32)
    grp_start = torch.arange(0, G * M + 1, M, device=dev, dtype=torch.int32)
    pair_tok = torch.arange(M, device=dev, dtype=torch.int32).repeat(G)  # every group sees all M tokens
    P = G * M
    ref = torch.stack([x.float() @ wb[e].T for e in range(G)]).view(P, N)
    y = cukern.fp4_gemm_tc(xp, w, s, experts, grp_start, pair_tok, P, M)
    err = ((y - ref).abs().max() / ref.abs().max()).item()
    # current GEMV: one (token, expert) pair per program row, weights re-read per pair
    pair_expert = experts.repeat_interleave(M)
    ones = torch.ones(P, device=dev)
    ref2 = cukern.fp4_gemv_pairs(x, w, s, pair_tok, pair_expert, ones, P)
    err2 = ((ref2 - ref).abs().max() / ref.abs().max()).item()
    dt = gtime(lambda: cukern.fp4_gemm_tc(xp, w, s, experts, grp_start, pair_tok, P, M))
    dt2 = gtime(lambda: cukern.fp4_gemv_pairs(x, w, s, pair_tok, pair_expert, ones, P))
    nbytes = G * N * (K // 2 + K // 32)
    print(f"G={G} experts x M={M:2d} tokens    {err:12.2e} {dt*1e6:8.1f} {nbytes/dt/1e9:8.0f} {dt2*1e6:8.1f} {nbytes/dt2/1e9:9.0f}   (gemv err {err2:.1e})")
