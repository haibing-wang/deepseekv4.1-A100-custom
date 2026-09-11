import os, sys, time, torch
sys.path.insert(0, "/mnt/ssdraid/git/deepseekv4.1")
from dsv41 import cpumoe
from dsv41.quant import dequant_fp4, fake_quant_fp8
torch.manual_seed(0)
bw = cpumoe.bw_test(8.0); print("RAM read bandwidth GB/s:", {k: round(v, 1) for k, v in bw.items()})
E, inter, dim = 48, 2304, 5120   # 48 experts = 0.9 GB: exceeds L3, streams from DRAM
he = cpumoe.HostExperts(E, inter, dim)
ws = {}
for e in range(E):
    w1 = torch.randint(0, 256, (inter, dim // 2), dtype=torch.uint8); w3 = torch.randint(0, 256, (inter, dim // 2), dtype=torch.uint8)
    s1 = torch.randint(118, 128, (inter, dim // 32), dtype=torch.uint8); s3 = torch.randint(118, 128, (inter, dim // 32), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (dim, inter // 2), dtype=torch.uint8); s2 = torch.randint(118, 128, (dim, inter // 32), dtype=torch.uint8)
    he.load_expert(e, w1, w3, s1, s3, w2, s2)
    if e in (0, 2, 3, 5, 6, 7): ws[e] = (w1, w3, s1, s3, w2, s2)
x = fake_quant_fp8(torch.randn(1, dim, dtype=torch.bfloat16) * 2, 32)
ids = [0, 2, 3, 5, 6, 7]; wts = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]
ref = torch.zeros(dim)
for e, wt in zip(ids, wts):
    w1, w3, s1, s3, w2, s2 = ws[e]
    g = x.float() @ dequant_fp4(w1, s1).float().T; u = x.float() @ dequant_fp4(w3, s3).float().T
    u = u.clamp(-10, 10); g = g.clamp(max=10)
    h = fake_quant_fp8((wt * torch.nn.functional.silu(g) * u).to(torch.bfloat16), 32)
    ref += (h.float() @ dequant_fp4(w2, s2).float().T)[0]
L = cpumoe.lib()
mb = 6 * (2 * inter * dim // 2 + 2 * inter * dim // 32 + dim * inter // 2 + dim * inter // 32)
for mode in (0, 1):
    L.cpumoe_set_int8(mode)
    out = he.forward(x[0], ids, wts, 10.0).clone()
    err = ((out - ref).abs().max() / ref.abs().max()).item()
    # rotate expert ids each iteration so the weights come from DRAM, not cache
    t = time.perf_counter(); it = 40
    for i in range(it): he.forward(x[0], [(e + 6 * i) % E for e in ids], wts, 10.0)
    dt = (time.perf_counter() - t) / it
    print(f"{'int8 VNNI' if mode else 'bf16     '}: rel err {err:.2e}  layer {dt*1e3:.2f} ms -> {mb/dt/1e9:.0f} GB/s ; per token (40 layers) {dt*40*1e3:.0f} ms")
