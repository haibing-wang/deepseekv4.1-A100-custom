# DeepSeek-V4.1-Flash on A100 (sm80) — a from-scratch inference runtime

`dsv41/` runs the official `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint (552B MoE backbone + 196B Engram
conditional memory, FP8/FP4 weights) on 8× A100 80GB, a GPU generation with no FP8/FP4 tensor cores and
no support in DeepSeek's own inference stack, vLLM or SGLang. The Engram hash tables (2 × 92 GiB) live in
host RAM. Single-stream decode reaches ~33 tok/s; the outputs are deterministic.

Nothing here depends on a third-party implementation of the model: the released `inference/model.py`
was used only as the architecture definition. All kernels are ours (Triton and CUDA C).

## Layout

| file | what |
|---|---|
| `dsv41/stio.py` | minimal safetensors reader over `mmap` (handles `F8_E8M0`, `F8_E4M3`, packed `I8` as raw bytes) |
| `dsv41/quant.py` | E8M0 / FP8-block / E2M1 formats, activation fake-quantization matching the reference kernels |
| `dsv41/moe_kernels.py` | Triton grouped FP4 GEMM (prefill): E2M1 decoded to bf16 bit patterns in-kernel, tensor cores |
| `dsv41/cuda/fp4_gemv.cu`, `dsv41/cukern.py` | CUDA C expert GEMV for decode (614 GB/s, no atomics); compiled with nvcc to a cubin and launched through `libcuda` via ctypes, so it works in CUDA-graph capture with any torch build |
| `dsv41/fused.py` | fused Triton kernels: RMSNorm, FP8/FP4 fake quant, Sinkhorn split, hyper-connection pre/post, RoPE, decode sparse attention (two KV sources, split-slot), SwiGLU + rounding |
| `dsv41/engram.py` | n-gram hashing (compressed vocab, per-layer multipliers, prime buckets) and the host-resident tables |
| `dsv41/model.py` | the 40-layer pipeline: sliding window + compressed sparse attention with two-level indexer, candidate blocks, hyper-connections, MoE with 384 experts; caches mirrored per (owner layer, GPU) |
| `dsv41/decode.py` | static-shape decode (position as a device tensor, dummy cache rows instead of branches) with one CUDA graph per GPU |
| `dsv41/load.py` | placement across GPUs by free memory (~7.1 GiB per layer), FP8 dense weights dequantized to bf16, experts kept packed |
| `dsv41/run.py` | CLI |

## Run

```
python -m dsv41.run --devices 2,0,1,4,5,6,7,3 --decode graph --chat \
    --prompt "日本で一番高い山と、その標高を教えてください。"
```

Requirements: torch ≥ 2.10 with `float8_e8m0fnu` (we use 2.13+cu130), triton ≥ 3.5, nvcc for `sm_80`
(`DSV41_NVCC`), transformers/tokenizers/sympy, the checkpoint at `/mnt/ssd/models/DeepSeek-V4.1-Flash`
(`--ckpt`), ~310 GiB of free GPU memory in total and ~190 GiB of host RAM for the Engram tables.

Flags: `--profile` (per-component decode timing), `--kernel-profile` (top CUDA kernels per token),
`--decode eager|static|graph`, `--n-layers N` (plumbing tests), `--no-engram`, `--budgets 3:26` (per-GPU GiB).
`DSV41_DETERMINISTIC=1` disables split-K in the prefill MoE kernel.

## How it maps to Ampere

- FP8 dense weights (per-32×32 E8M0 block scales) → bf16 at load; activations are rounded to FP8 exactly
  where the reference does it, so cuBLAS bf16 GEMMs see the same inputs. cuBLAS already runs these
  M=1 GEMVs at ~1.3 TB/s.
- FP4 experts (E2M1, per-32 E8M0 scales) stay packed in VRAM (292 GiB across the GPUs). Decode: one warp
  streams a row with 16-byte loads, decodes nibbles through a 16-entry shared LUT and accumulates in fp32;
  prefill: Triton tensor-core GEMM with the scale folded into the bf16 exponent.
- Compressed KV / index keys use the same FP4 (E4M3 or E8M0 scale) rounding as the reference (fake quant).
- Engram: hash ids are computed on the GPU from token ids, rows + scales are gathered on the CPU
  (`index_select`, ~1 ms/token) and dequantized on the GPU.
- Decode is one CUDA graph per GPU; between segments only the residual stream, the index buffers and the
  cache rows written by owner layers are copied.

## Memory budget: what needs to live where

Sizes are for the released checkpoint (40 backbone layers; the 3 DSpark/MTP layers and the vision encoder
are not loaded).

| component | format | size | where it lives now |
|---|---|---|---|
| routed experts (384 × 40 layers) | FP4 packed + E8M0 scales | **269 GiB** (6.72 GiB/layer) | GPU, spread over the pipeline |
| dense weights (attention, shared experts, gates, indexers, hyper-connections) | bf16 (from FP8) | 13.6 GiB (0.34 GiB/layer) | GPU, with their layer |
| embedding + output head | bf16 | 2.5 GiB | first / last GPU |
| Engram tables (layers 1 and 14) | FP8 rows + E8M0 scales | **189 GiB** (2 × 94.5 GiB) | **host RAM** |
| Engram projections | bf16 | 0.6 GiB | GPU |
| caches at 4K context (window ring, compressed KV, index keys, mirrors) | bf16 | < 1 GiB per GPU | GPU |

So the fully GPU-resident configuration needs about **290 GiB of GPU memory plus ~2 GiB per GPU of
working space**, i.e. a minimum of 4× A100 80GB (very tight), comfortably 5-6, and **≥ 200 GiB of host
RAM** for the Engram tables (the loader reads them into RAM; page cache for the 475 GiB checkpoint on
top of that makes cold loads faster but is optional).

### Can it run on a single A100 80GB?

Not with the experts resident: 269 GiB of FP4 experts do not fit in 80 GB. A single-GPU variant has to
keep the experts in host memory and stream the 6 selected experts per layer for every token:

| resource | single A100 requirement |
|---|---|
| GPU memory | ~20 GiB (dense weights, embeddings, caches, workspace) — fits with room for long contexts |
| host RAM | 269 GiB experts (pinned) + 189 GiB Engram = **~460 GiB**; ~270 GiB if the Engram tables are served from NVMe instead (24 random 6 KB reads per token, which an NVMe handles easily) |
| PCIe traffic per decoded token | 6 experts × 40 layers × 18.8 MB = **4.5 GB** → ~190 ms at ~24 GB/s (PCIe 4.0 x16) |
| expected decode speed | **~5 tok/s** single stream, bounded by PCIe, not by the GPU |
| prefill | all experts of a layer are needed → the whole 269 GiB streams through once per prompt (~11 s at 24 GB/s) |

This offload path is not implemented in this repository yet (the loader places whole layers on GPUs);
the numbers above follow directly from the sizes and the interconnect. With a machine that has the
RAM, the changes are confined to `MoE.__call__` (gather the selected experts from pinned host tensors
into a per-layer GPU staging buffer before `fp4_gemv_pairs`) and `load.py`.

## Numbers (8× A100 80GB PCIe, shared with other jobs)

| stage | decode tok/s |
|---|---|
| first working version | 4.0 |
| MoE dispatch without host syncs | 7.3 |
| fused Triton elementwise kernels | 15.3 |
| static-shape decode + CUDA graphs | 21.7 |
| CUDA C expert GEMV | 31.6 |
| split-slot decode attention | 33.6 |

Prefill of a 1,413-token prompt: 4.8 s (296 tok/s); decode after it: 33.4 tok/s. Load: ~70 s with the
checkpoint in page cache, ~4 min cold.

Not implemented yet: DSpark (MTP) speculative decoding, the vision encoder, batch > 1, a fast FP8
dense GEMV (`cuda/fp8_gemv.cu` is an unused attempt that does not beat cuBLAS bf16).
