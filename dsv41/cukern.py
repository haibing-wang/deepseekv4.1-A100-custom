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
    group_cols > 0: block-diagonal use (x: [B * N/group_cols, K]; output row b, column n uses x row
    b * (N/group_cols) + n // group_cols) -> [B, N]."""
    M, K = x.shape
    N = w8.shape[0]
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and w8.is_contiguous() and s8.is_contiguous()
    assert K % 64 == 0 and N % 8 == 0 and (group_cols == 0 and M <= 16 or group_cols > 0 and group_cols % 8 == 0 and N % group_cols == 0)
    Mo = M // (N // group_cols) if group_cols else M
    assert Mo <= 16
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
                pair_tok: torch.Tensor, n_pairs: int, max_tokens: int, shard_start: int = 0, shard_n: int = 1 << 30,
                zero_out: bool = False, out: torch.Tensor | None = None) -> torch.Tensor:
    """xp: permuted bf16 [rows, K]; w: uint8 [E, N, K/2]; s: uint8 [E, N, K/32]; groups g: expert grp_expert[g] with pairs
    grp_start[g]..grp_start[g+1]-1 (<= 16 each), pair p uses x row pair_tok[p]. Returns fp32 [n_pairs, N].
    Expert parallelism: ids are global, this GPU holds [shard_start, shard_start + shard_n); other groups are skipped
    (zero_out: their output rows are zeroed)."""
    E, N, Kh = w.shape
    K = Kh * 2
    assert xp.dtype == torch.bfloat16 and xp.is_contiguous() and K % 128 == 0 and N % 8 == 0 and max_tokens <= 16
    G = grp_expert.numel()
    if out is None:
        out = torch.empty(n_pairs, N, device=xp.device, dtype=torch.float32)
    f = get_function("fp4_tc.cu", "fp4_gemm_tc8" if max_tokens <= 8 else "fp4_gemm_tc16", xp.device)
    WARPS = 4
    grid = ((N // 8 + WARPS - 1) // WARPS, G, 1)
    args = [ctypes.c_void_p(xp.data_ptr()), ctypes.c_int(xp.stride(0)),
            ctypes.c_void_p(w.data_ptr()), ctypes.c_longlong(w.stride(0)), ctypes.c_void_p(s.data_ptr()), ctypes.c_longlong(s.stride(0)),
            ctypes.c_void_p(grp_expert.data_ptr()), ctypes.c_void_p(grp_start.data_ptr()), ctypes.c_void_p(pair_tok.data_ptr()),
            ctypes.c_void_p(out.data_ptr()), ctypes.c_int(N), ctypes.c_int(N), ctypes.c_int(K),
            ctypes.c_int(shard_start), ctypes.c_int(min(shard_n, E)), ctypes.c_int(1 if zero_out else 0)]
    launch(f, grid, (WARPS * 32, 1, 1), args, xp.device)
    return out


# --------------------------------------------------------------------------- device-side GPU messaging (expert parallelism)
def p2p_copy(dst: torch.Tensor, src: torch.Tensor, device: torch.device):
    """Copy src (contiguous, size multiple of 16 B) into dst (possibly on another GPU) with a kernel on `device`."""
    n = src.numel() * src.element_size()
    assert n % 16 == 0 and dst.numel() * dst.element_size() >= n
    f = get_function("p2p.cu", "p2p_copy", device)
    n16 = n // 16
    launch(f, ((n16 + 255) // 256, 1, 1), (256, 1, 1), [ctypes.c_void_p(dst.data_ptr()), ctypes.c_void_p(src.data_ptr()), ctypes.c_int(n16)], device)


def p2p_copy_row(dst_base: torch.Tensor, row_idx: torch.Tensor, src: torch.Tensor, device: torch.device):
    """dst_base[b, row_idx] = src[b] for every batch row b (dst_base: [B, rows, D], src: [B, 1, D]; row index: int64 device scalar)."""
    B = dst_base.shape[0]
    n = src.numel() * src.element_size() // B
    assert n % 16 == 0 and src.numel() == B * dst_base.shape[-1]
    f = get_function("p2p.cu", "p2p_copy_row", device)
    row16 = n // 16
    bstride16 = dst_base.stride(0) * dst_base.element_size() // 16
    launch(f, ((row16 * B + 255) // 256, 1, 1), (256, 1, 1), [ctypes.c_void_p(dst_base.data_ptr()), ctypes.c_void_p(row_idx.data_ptr()), ctypes.c_void_p(src.data_ptr()),
                                                            ctypes.c_int(row16), ctypes.c_int(B), ctypes.c_longlong(bstride16)], device)


def p2p_sum_rows(dst: torch.Tensor, src: torch.Tensor, device: torch.device, groups: int = 1, dst_stride: int | None = None):
    """dst[g] = sum of the `rows` rows of group g of src [groups * rows, n] (dst rows `dst_stride` elements apart)."""
    total, n = src.shape
    rows = total // groups
    if dst_stride is None:
        dst_stride = dst.stride(0) if dst.dim() > 1 else n
    f = get_function("p2p.cu", "p2p_sum_rows", device)
    launch(f, ((n + 255) // 256, groups, 1), (256, 1, 1), [ctypes.c_void_p(dst.data_ptr()), ctypes.c_void_p(src.data_ptr()), ctypes.c_int(rows), ctypes.c_int(n),
                                                          ctypes.c_int(groups), ctypes.c_longlong(dst_stride)], device)


def p2p_signal(flag_ptrs: torch.Tensor, seq: torch.Tensor, device: torch.device):
    """Set the flags at the addresses in flag_ptrs (int64 device tensor on `device`) to the value of seq (int32 device scalar)."""
    f = get_function("p2p.cu", "p2p_signal", device)
    launch(f, (1, 1, 1), (1, 1, 1), [ctypes.c_void_p(flag_ptrs.data_ptr()), ctypes.c_int(flag_ptrs.numel()), ctypes.c_void_p(seq.data_ptr())], device)


def p2p_wait(flags: torch.Tensor, seq: torch.Tensor, device: torch.device):
    """Spin (one thread) until all flags (int32 [n] on `device`) >= seq."""
    f = get_function("p2p.cu", "p2p_wait", device)
    launch(f, (1, 1, 1), (1, 1, 1), [ctypes.c_void_p(flags.data_ptr()), ctypes.c_int(flags.numel()), ctypes.c_void_p(seq.data_ptr())], device)


def p2p_seq_bump(seq: torch.Tensor, device: torch.device):
    f = get_function("p2p.cu", "p2p_seq_bump", device)
    launch(f, (1, 1, 1), (1, 1, 1), [ctypes.c_void_p(seq.data_ptr())], device)


def p2p_multicast(dst_ptrs: torch.Tensor, src: torch.Tensor, flag_ptrs: torch.Tensor | None, seq: torch.Tensor, device: torch.device, counter: torch.Tensor | None = None):
    """One kernel: src (size multiple of 16) into every destination address in dst_ptrs (int64 on `device`), then the last
    block sets the flags at flag_ptrs to seq (counter: an int32 device scalar, zero at first use, self-resetting)."""
    n = src.numel() * src.element_size()
    assert n % 16 == 0
    f = get_function("p2p.cu", "p2p_multicast", device)
    n16 = n // 16
    blocks = (n16 + 1023) // 1024
    if flag_ptrs is not None and counter is None:
        counter = _tile_counters(device, 8192)[-1:]
    launch(f, (blocks, 1, 1), (1024, 1, 1), [ctypes.c_void_p(dst_ptrs.data_ptr()), ctypes.c_int(dst_ptrs.numel()), ctypes.c_void_p(src.data_ptr()), ctypes.c_int(n16),
                                            ctypes.c_void_p(flag_ptrs.data_ptr() if flag_ptrs is not None else 0), ctypes.c_void_p(seq.data_ptr()),
                                            ctypes.c_void_p(counter.data_ptr() if counter is not None else 0)], device)


def p2p_stamp(dst: torch.Tensor, device: torch.device):
    """Write the GPU global timer (ns) into dst (int64 scalar on `device`), stream-ordered."""
    f = get_function("p2p.cu", "p2p_stamp", device)
    launch(f, (1, 1, 1), (1, 1, 1), [ctypes.c_void_p(dst.data_ptr())], device)


_cuda.cuMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]


def memcpy_async(dst: torch.Tensor, src: torch.Tensor, device: torch.device, nbytes: int | None = None):
    """cuMemcpyAsync(dst, src) on `device`'s current stream (unified addressing: dst may live on a peer GPU; the copy
    engines do the transfer, which matters across sockets where kernel-initiated P2P stores crawl)."""
    n = src.numel() * src.element_size() if nbytes is None else nbytes
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device).cuda_stream
        _check(_cuda.cuMemcpyAsync(ctypes.c_void_p(dst.data_ptr()), ctypes.c_void_p(src.data_ptr()), n, ctypes.c_void_p(stream)), "cuMemcpyAsync")
