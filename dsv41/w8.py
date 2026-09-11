"""Dense weights kept as FP8 (e4m3 bytes + E8M0 [32x32] block scales) instead of bf16.

Decode (M <= 16 rows) runs them through the tensor-core kernel in cuda/fp8_tc.cu, which reads half
the bytes of the bf16 cuBLAS path (1.1-1.2 TB/s effective on A100 vs 1.4 TB/s of bf16, i.e. ~1.6x
faster per GEMV) and computes exactly the same numbers (exact dequantization, fp32 accumulation).
Prefill (M > 16) dequantizes to a temporary bf16 matrix and uses cuBLAS.
Set DSV41_W8=0 to keep bf16 copies (old behaviour)."""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .quant import dequant_fp8_block

ENABLED = os.environ.get("DSV41_W8", "1") == "1"
MAX_TC_ROWS = 16


class W8:
    __slots__ = ("w8", "s8", "shape", "device")

    def __init__(self, w8: torch.Tensor, s8: torch.Tensor):
        assert w8.dtype == torch.uint8 and s8.dtype == torch.uint8 and w8.dim() == 2
        self.w8 = w8.contiguous()
        self.s8 = s8.contiguous()
        self.shape = tuple(w8.shape)
        self.device = w8.device

    @property
    def dtype(self):
        return torch.bfloat16

    def bf16(self) -> torch.Tensor:
        return dequant_fp8_block(self.w8.view(torch.float8_e4m3fn), self.s8)

    @staticmethod
    def cat(ws: list) -> "W8":
        """Concatenate along N (row counts must be multiples of 32)."""
        assert all(w.shape[0] % 32 == 0 for w in ws)
        return W8(torch.cat([w.w8 for w in ws], dim=0), torch.cat([w.s8 for w in ws], dim=0))


def linear_w(x: torch.Tensor, w) -> torch.Tensor:
    """F.linear for a bf16 tensor weight or a W8 weight (bf16 x, bf16 result)."""
    if not isinstance(w, W8):
        return F.linear(x, w)
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if x2.shape[0] <= MAX_TC_ROWS:
        from .cukern import fp8_gemm_tc
        y = fp8_gemm_tc(x2.contiguous().to(torch.bfloat16), w.w8, w.s8)
    else:
        y = F.linear(x2, w.bf16())
    return y.view(*lead, w.shape[0])


def oproj_a(o: torch.Tensor, wo_a, n_groups: int, rank: int) -> torch.Tensor:
    """The block-diagonal o-projection: o [b, s, g, d] x wo_a (rows g*rank..(g+1)*rank use only group g) -> [b, s, g*rank]."""
    b, s, g, d = o.shape
    if not isinstance(wo_a, W8):
        return torch.einsum("bsgd,grd->bsgr", o, wo_a.view(n_groups, rank, -1)).flatten(2)
    if b * s <= MAX_TC_ROWS:
        from .cukern import fp8_gemm_tc
        return fp8_gemm_tc(o.reshape(b * s * g, d).contiguous(), wo_a.w8, wo_a.s8, group_cols=rank).view(b, s, -1)
    return torch.einsum("bsgd,grd->bsgr", o, wo_a.bf16().view(n_groups, rank, -1)).flatten(2)
