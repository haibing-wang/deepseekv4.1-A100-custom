"""Numeric formats of the DeepSeek-V4.1 checkpoint, implemented with plain torch ops so they run on
Ampere (no FP8/FP4 tensor cores needed).

- E8M0 scales: one byte b, value 2^(b-127).
- FP8 block weights: float8_e4m3fn [N, K] with an E8M0 scale per 32x32 block -> dequantized to bf16 at load.
- FP4 expert weights: E2M1 packed two per byte along K (low nibble = even k), one E8M0 scale per row per 32 k.
- Activation "fake quantization" (quantize then dequantize) reproduces what the reference kernels do
  before feeding FP8/FP4 tensor cores, so our bf16 GEMMs see the same rounded values.
"""
import os

import torch


def maybe_compile(fn):
    """torch.compile fuses our elementwise quantization chains into single kernels (set DSV41_COMPILE=0 to disable)."""
    if os.environ.get("DSV41_COMPILE", "0") == "1":
        return torch.compile(fn, dynamic=True)
    return fn


FP8_MAX = 448.0
FP4_MAX = 6.0
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def e8m0_to_float(b: torch.Tensor) -> torch.Tensor:
    """uint8 (or float8_e8m0fnu bit pattern) -> float32 power of two."""
    if b.dtype != torch.uint8:
        b = b.view(torch.uint8)
    return torch.exp2(b.to(torch.float32) - 127.0)


def ceil_log2(x: torch.Tensor) -> torch.Tensor:
    """ceil(log2(x)) for x > 0, exactly, via frexp (x = m * 2^e, m in [0.5, 1))."""
    m, e = torch.frexp(x)
    return e - (m == 0.5).to(e.dtype)


def pow2_scale(amax: torch.Tensor, qmax: float) -> torch.Tensor:
    """The reference fast_round_scale: 2^ceil(log2(amax / qmax))."""
    return torch.exp2(ceil_log2(amax / qmax).to(torch.float32))


@maybe_compile
def fake_quant_fp8(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """act_quant(..., inplace=True) of the reference: per-row groups of `block` along the last dim,
    power-of-two scale, round to e4m3, dequantize. Returns x's dtype."""
    shape = x.shape
    xg = x.float().reshape(-1, block)
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    s = pow2_scale(amax, FP8_MAX)
    q = (xg / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (q * s).reshape(shape).to(x.dtype)


def quant_fp8(x: torch.Tensor, block: int = 32):
    """act_quant without dequant: (float8_e4m3fn values, float32 scales [..., K/block])."""
    shape = x.shape
    xg = x.float().reshape(-1, block)
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    s = pow2_scale(amax, FP8_MAX)
    q = (xg / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(shape), s.reshape(*shape[:-1], shape[-1] // block)


def round_e2m1(v: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even onto the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} (|v| <= 6 assumed)."""
    a = v.abs()
    r = torch.where(a < 2.0, torch.round(a * 2.0) / 2.0, torch.where(a < 4.0, torch.round(a), torch.round(a / 2.0) * 2.0))
    return torch.copysign(r.clamp(max=FP4_MAX), v)


@maybe_compile
def fake_quant_fp4(x: torch.Tensor, block: int = 32, scale_e4m3: bool = False) -> torch.Tensor:
    """fp4_act_quant(..., inplace=True): E2M1 values with an E8M0 (default) or E4M3 scale per group."""
    shape = x.shape
    xg = x.float().reshape(-1, block)
    amax = xg.abs().amax(dim=-1, keepdim=True)
    if scale_e4m3:
        amax = amax.clamp_min(6 * 2.0**-9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2.0**-126)
        s = pow2_scale(amax, FP4_MAX)
    q = round_e2m1((xg / s).clamp(-FP4_MAX, FP4_MAX))
    return (q * s).reshape(shape).to(x.dtype)


def dequant_fp8_block(w: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """float8_e4m3fn [N, K] with E8M0 scales [ceil(N/32), ceil(K/32)] -> bf16 [N, K]."""
    n, k = w.shape
    s = e8m0_to_float(scale)[: (n + block - 1) // block, : (k + block - 1) // block]
    s = s.repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    return (w.float() * s).to(torch.bfloat16)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """int8/uint8 [N, K/2] -> float32 [N, K] E2M1 values (unscaled)."""
    p = packed.view(torch.uint8)
    lo, hi = (p & 0x0F).long(), (p >> 4).long()
    tab = FP4_TABLE.to(p.device)
    return torch.stack([tab[lo], tab[hi]], dim=-1).flatten(-2)


def dequant_fp4(packed: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Reference (slow) dequantization of an expert weight, for tests: -> bf16 [N, K]."""
    v = unpack_fp4(packed)
    s = e8m0_to_float(scale).repeat_interleave(block, 1)
    return (v * s).to(torch.bfloat16)


# --------------------------------------------------------------------------- tiled FP4 expert layout
# The decode kernels (cuda/fp4_tc.cu, cuda/fp4_tcw.cu) read, per warp instruction, 8 rows x 64 bytes of an expert
# (128 k). Stored row-major those are 8 separate 64-byte pieces 2.5 KB apart; tiled as
# [N/16][K/128][16 rows][64 B] (scales [N/16][K/128][16][4]) every warp load is one contiguous 512-byte range
# (fp4_gemm_tc8 at one token per expert: 1.02 -> 1.20 TB/s). Same tensor shapes, permuted content.
def tile_fp4(w: torch.Tensor) -> torch.Tensor:
    """uint8 [E, N, K/2] row-major -> tiled (same shape)."""
    E, N, Kh = w.shape
    assert N % 16 == 0 and Kh % 64 == 0
    return w.view(E, N // 16, 16, Kh // 64, 64).permute(0, 1, 3, 2, 4).reshape(E, N, Kh).contiguous()


def untile_fp4(w: torch.Tensor) -> torch.Tensor:
    E, N, Kh = w.shape
    return w.view(E, N // 16, Kh // 64, 16, 64).permute(0, 1, 3, 2, 4).reshape(E, N, Kh).contiguous()


def tile_fp4_scales(s: torch.Tensor) -> torch.Tensor:
    """uint8 [E, N, K/32] -> tiled [N/16][K/128][16][4] (same shape)."""
    E, N, Ks = s.shape
    assert N % 16 == 0 and Ks % 4 == 0
    return s.view(E, N // 16, 16, Ks // 4, 4).permute(0, 1, 3, 2, 4).reshape(E, N, Ks).contiguous()


def untile_fp4_scales(s: torch.Tensor) -> torch.Tensor:
    E, N, Ks = s.shape
    return s.view(E, N // 16, Ks // 4, 16, 4).permute(0, 1, 3, 2, 4).reshape(E, N, Ks).contiguous()
