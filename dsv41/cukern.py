"""Our CUDA C kernels, compiled with nvcc to cubin and launched through the driver API (ctypes), so
they work with any torch build and inside CUDA-graph capture."""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NVCC = os.environ.get("DSV41_NVCC", "/usr/local/cuda-12.8/bin/nvcc")
_cuda = ctypes.CDLL("libcuda.so.1")
_modules: dict[tuple[str, int], ctypes.c_void_p] = {}
_funcs: dict[tuple[str, int], ctypes.c_void_p] = {}


def _check(err, what):
    if err != 0:
        raise RuntimeError(f"{what} failed with CUDA error {err}")


def _cubin(src_name: str) -> bytes:
    src = os.path.join(HERE, "cuda", src_name)
    code = open(src, "rb").read()
    tag = hashlib.sha1(code).hexdigest()[:12]
    out = os.path.join(HERE, "cuda", f".{src_name}.{tag}.sm80.cubin")
    if not os.path.exists(out):
        subprocess.run([NVCC, "-cubin", "-arch=sm_80", "-O3", "-o", out, src], check=True)
    return open(out, "rb").read()


def get_function(src_name: str, func: str, device: torch.device) -> ctypes.c_void_p:
    key = (src_name, device.index)
    if key not in _modules:
        with torch.cuda.device(device):
            torch.cuda.current_stream()  # make sure the context exists
            image = _cubin(src_name)
            mod = ctypes.c_void_p()
            _check(_cuda.cuModuleLoadData(ctypes.byref(mod), image), "cuModuleLoadData")
            _modules[key] = mod
    fkey = (src_name + ":" + func, device.index)
    if fkey not in _funcs:
        f = ctypes.c_void_p()
        _check(_cuda.cuModuleGetFunction(ctypes.byref(f), _modules[key], func.encode()), "cuModuleGetFunction")
        _funcs[fkey] = f
    return _funcs[fkey]


def launch(func, grid, block, args, device: torch.device, shared: int = 0):
    """args: list of ctypes values (c_void_p for pointers)."""
    ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    with torch.cuda.device(device):  # the module handle belongs to this device's primary context
        stream = torch.cuda.current_stream(device).cuda_stream
        _check(_cuda.cuLaunchKernel(func, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared,
                                    ctypes.c_void_p(stream), ptrs, None), "cuLaunchKernel")


def fp4_gemv_pairs(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, row_in: torch.Tensor, expert: torch.Tensor,
                   wt: torch.Tensor, n_pairs: int) -> torch.Tensor:
    """x: bf16 [rows_in, K]; w: uint8 [E, N, K/2]; s: uint8 [E, N, K/32]; row_in/expert: int32 [pairs];
    wt: fp32 [pairs]. Returns fp32 [pairs, N] = wt * x[row_in] @ W[expert]^T (one program per row: no atomics)."""
    E, N, Kh = w.shape
    K = Kh * 2
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1 and K <= 5120
    out = torch.empty(n_pairs, N, device=x.device, dtype=torch.float32)
    f = get_function("fp4_gemv.cu", "fp4_gemv_pairs", x.device)
    rows_per_block = 8 * 4
    grid = ((N + rows_per_block - 1) // rows_per_block, n_pairs, 1)
    args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_int(x.stride(0)),
            ctypes.c_void_p(w.data_ptr()), ctypes.c_longlong(w.stride(0)), ctypes.c_int(w.stride(1)),
            ctypes.c_void_p(s.data_ptr()), ctypes.c_longlong(s.stride(0)), ctypes.c_int(s.stride(1)),
            ctypes.c_void_p(row_in.data_ptr()), ctypes.c_void_p(expert.data_ptr()), ctypes.c_void_p(wt.data_ptr()),
            ctypes.c_void_p(out.data_ptr()), ctypes.c_int(out.stride(0)), ctypes.c_int(N), ctypes.c_int(K)]
    launch(f, grid, (256, 1, 1), args, x.device)
    return out


def fp8_gemv(x: torch.Tensor, w_fp8: torch.Tensor, s_u8: torch.Tensor) -> torch.Tensor:
    """x: bf16 [M<=16, K]; w_fp8: float8_e4m3fn/uint8 [N, K]; s_u8: uint8 [ceil(N/32), K/32] (E8M0). fp32 [M, N]."""
    M, K = x.shape
    N = w_fp8.shape[0]
    assert M == 1 and K % 32 == 0 and K <= 8192 and x.stride(1) == 1
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)
    f = get_function("fp8_gemv.cu", "fp8_gemv", x.device)
    rows_per_block = 8 * 4
    grid = ((N + rows_per_block - 1) // rows_per_block, 1, 1)
    smem = 4 * (256 + 16 * M * (K // 16))
    args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_int(x.stride(0)), ctypes.c_int(M),
            ctypes.c_void_p(w_fp8.data_ptr()), ctypes.c_int(w_fp8.stride(0)),
            ctypes.c_void_p(s_u8.data_ptr()), ctypes.c_int(s_u8.stride(0)),
            ctypes.c_void_p(out.data_ptr()), ctypes.c_int(out.stride(0)), ctypes.c_int(N), ctypes.c_int(K)]
    launch(f, grid, (256, 1, 1), args, x.device, shared=smem)
    return out


# --------------------------------------------------------------------------- FP8-weight tensor-core GEMM (M <= 16)
def _splits_for(N: int, K: int) -> int:
    """Split-K factor so that at least ~1024 warps stream the weights (each warp owns 8 columns)."""
    warps = N // 8
    s = 1
    while warps * s < 4096 and (K // (s * 2)) >= 256:
        s *= 2
    return s


def fp8_gemm_tc(x: torch.Tensor, w8: torch.Tensor, s8: torch.Tensor, group_cols: int = 0, out_dtype=torch.bfloat16) -> torch.Tensor:
    """x: bf16 [M, K] (M <= 16, contiguous); w8: uint8 (e4m3 bits) [N, K]; s8: uint8 (E8M0) [ceil(N/32), K/32].
    Returns x @ dequant(w8, s8)^T as [M, N] (fp32 accumulation, rounded to out_dtype).
    group_cols > 0: block-diagonal use (x: [N/group_cols, K]; column n uses x row n // group_cols) -> [1, N]."""
    M, K = x.shape
    N = w8.shape[0]
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and w8.is_contiguous() and s8.is_contiguous()
    assert K % 64 == 0 and N % 8 == 0 and (group_cols == 0 and M <= 16 or group_cols > 0 and group_cols % 8 == 0)
    Mo = 1 if group_cols else M
    splits = _splits_for(N, K)
    kps = -(-K // splits)
    kps = -(-kps // 128) * 128
    splits = -(-K // kps)
    part = torch.empty(splits, Mo, N, device=x.device, dtype=torch.float32)
    f = get_function("fp8_tc.cu", "fp8_gemm_tc8" if Mo <= 8 else "fp8_gemm_tc16", x.device)
    WARPS = 4
    grid = ((N // 8 + WARPS - 1) // WARPS, splits, 1)
    fused_epilogue = out_dtype == torch.bfloat16
    y = torch.empty(Mo, N, device=x.device, dtype=torch.bfloat16) if fused_epilogue else None
    counters = _tile_counters(x.device, N // 8) if fused_epilogue else None
    args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_int(x.stride(0)), ctypes.c_int(Mo),
            ctypes.c_void_p(w8.data_ptr()), ctypes.c_void_p(s8.data_ptr()), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(s8.shape[1]),
            ctypes.c_void_p(part.data_ptr()), ctypes.c_int(N), ctypes.c_int(kps), ctypes.c_int(group_cols),
            ctypes.c_void_p(y.data_ptr() if y is not None else 0), ctypes.c_void_p(counters.data_ptr() if counters is not None else 0), ctypes.c_int(splits)]
    launch(f, grid, (WARPS * 32, 1, 1), args, x.device)
    if fused_epilogue:
        return y
    y = part[0] if splits == 1 else part.sum(dim=0)
    return y.to(out_dtype)


_counters: dict = {}


def _tile_counters(device, n_tiles: int) -> torch.Tensor:
    """Zeroed per-tile counters for the split-K epilogue (self-resetting; one buffer per device, grown on demand)."""
    c = _counters.get(device)
    if c is None or c.numel() < n_tiles:
        c = _counters[device] = torch.zeros(max(n_tiles, 8192), dtype=torch.int32, device=device)
    return c


# --------------------------------------------------------------------------- FP4 expert GEMM on tensor cores (grouped by expert)
_PERM8 = [0, 4, 2, 6, 1, 5, 3, 7]


def permute_x(x: torch.Tensor) -> torch.Tensor:
    """bf16 [M, K] -> the 8-k permuted layout fp4_gemm_tc expects (within every 8 k: 0,4,2,6,1,5,3,7)."""
    M, K = x.shape
    return x.view(M, K // 8, 8)[:, :, _PERM8].reshape(M, K).contiguous()


def fp4_gemm_tc(xp: torch.Tensor, w: torch.Tensor, s: torch.Tensor, grp_expert: torch.Tensor, grp_start: torch.Tensor,
                pair_tok: torch.Tensor, n_pairs: int, max_tokens: int) -> torch.Tensor:
    """xp: permuted bf16 [rows, K]; w: uint8 [E, N, K/2]; s: uint8 [E, N, K/32]; groups g: expert grp_expert[g] with pairs
    grp_start[g]..grp_start[g+1]-1 (<= 16 each), pair p uses x row pair_tok[p]. Returns fp32 [n_pairs, N]."""
    E, N, Kh = w.shape
    K = Kh * 2
    assert xp.dtype == torch.bfloat16 and xp.is_contiguous() and K % 128 == 0 and N % 8 == 0 and max_tokens <= 16
    G = grp_expert.numel()
    out = torch.empty(n_pairs, N, device=xp.device, dtype=torch.float32)
    f = get_function("fp4_tc.cu", "fp4_gemm_tc8" if max_tokens <= 8 else "fp4_gemm_tc16", xp.device)
    WARPS = 4
    grid = ((N // 8 + WARPS - 1) // WARPS, G, 1)
    args = [ctypes.c_void_p(xp.data_ptr()), ctypes.c_int(xp.stride(0)),
            ctypes.c_void_p(w.data_ptr()), ctypes.c_longlong(w.stride(0)), ctypes.c_void_p(s.data_ptr()), ctypes.c_longlong(s.stride(0)),
            ctypes.c_void_p(grp_expert.data_ptr()), ctypes.c_void_p(grp_start.data_ptr()), ctypes.c_void_p(pair_tok.data_ptr()),
            ctypes.c_void_p(out.data_ptr()), ctypes.c_int(N), ctypes.c_int(N), ctypes.c_int(K)]
    launch(f, grid, (WARPS * 32, 1, 1), args, xp.device)
    return out
