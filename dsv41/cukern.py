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
