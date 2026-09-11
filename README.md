# DeepSeek-V4.1-Flash on A100 (sm80) — a from-scratch inference runtime

`dsv41/` runs the official `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint (552B MoE backbone + 196B Engram
conditional memory, FP8/FP4 weights) on 8× A100 80GB, a GPU generation with no FP8/FP4 tensor cores and
no support in DeepSeek's own inference stack, vLLM or SGLang. The Engram hash tables (2 × 92 GiB) live in
host RAM. Single-stream decode reaches 52 tok/s on the 8-GPU layer pipeline, 62 tok/s with expert
parallelism across 7 GPUs, and 32-35 tok/s on a single A100 with the experts computed on the CPUs.

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
| `dsv41/engine.py` | generation engine shared by the REPL and the server (streaming, top-p, stop strings, chat template + completion parser from the checkpoint's `encoding/`) |
| `dsv41/chat.py` | interactive terminal chat (multi-turn, `/clear`, `/system`, thinking mode) |
| `dsv41/serve.py` | OpenAI-compatible HTTP server (`/v1/chat/completions` with streaming, `/v1/completions`, `/v1/models`), stdlib only |
| `dsv41/run.py` | one-shot CLI with profiling flags |

## Run

Interactive chat:

```
python -m dsv41.chat --devices 2,0,1,4,5,6,7,3            # add --thinking for reasoning mode
>>> 日本で一番高い山と、その標高を教えてください。
```

OpenAI-compatible server (one request generates at a time; others queue):

```
python -m dsv41.serve --devices 2,0,1,4,5,6,7,3 --port 8000
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash",
  "messages": [{"role": "user", "content": "東京タワーの高さは？"}],
  "max_tokens": 256, "temperature": 0.6, "stream": true}'
```

Supported request fields: `messages` (system/user/assistant/tool, images not supported), `max_tokens`,
`temperature`, `top_p`, `stop`, `seed`, `stream`, and `"thinking": true` (or `reasoning_effort`) to get
the model's reasoning back in `message.reasoning_content`. Tool calls in the completion are parsed into
OpenAI-format `tool_calls`. `/v1/completions` takes a raw `prompt`. Works with the `openai` client by
setting `base_url="http://host:8000/v1"`.

One-shot generation with profiling:

```
python -m dsv41.run --devices 2,0,1,4,5,6,7,3 --decode graph --chat \
    --prompt "日本で一番高い山と、その標高を教えてください。"
```

Requirements: torch ≥ 2.10 with `float8_e8m0fnu` (we use 2.13+cu130), triton ≥ 3.5, nvcc for `sm_80`
(`DSV41_NVCC`), transformers/tokenizers/sympy, the checkpoint at `/mnt/ssd/models/DeepSeek-V4.1-Flash`
(`--ckpt`), ~310 GiB of free GPU memory in total and ~190 GiB of host RAM for the Engram tables.

Flags: `--profile` (per-component decode timing), `--kernel-profile` (top CUDA kernels per token),
`--kernel-trace FILE` (chronological kernel list of one step), `--decode eager|static|graph`, `--n-layers N`
(plumbing tests), `--no-engram`, `--budgets 3:26` (per-GPU GiB). `DSV41_DETERMINISTIC=1` disables split-K
in the prefill MoE kernel.

### Expert parallelism (`--ep`)

```
python -m dsv41.run --devices 2,3,0,1 --ep --chat --prompt "..."                 # 4 GPUs, equal shards
python -m dsv41.run --devices 2,3,0,1,4,5,7 --ep --ep-shards 82,82,68,38,38,38,38 # uneven shards (free memory)
```

`dsv41/ep.py`: the dense layers are pipelined over the GPUs in order and every layer's 384 experts are
sharded over all of them, so each GPU reads only its own experts. Per layer the owner GPU pushes the
quantized activation and the routing to the peers with P2P stores and raises a flag; every GPU computes
the selected experts it holds (a masked grouped tensor-core GEMM) and pushes its partial sum back; the
owner waits for the flags, adds the shared expert and continues. The whole token is one CUDA graph per
GPU with device-side flag synchronisation (no host round trips). Needs P2P between the GPUs: on this box
the 4 GPUs of one socket exchange a message in ~10 us, across sockets it is much slower. 62 tok/s on
7 GPUs vs 52 tok/s for the pipeline; the expert reads per step stay bounded when several tokens are
verified at once, which is what speculative decoding needs.

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

This is the `--offload-experts` mode. It has two variants:

- `--offload-experts cpu` (default): the experts stay in host RAM **and are computed on the CPU**
  (`dsv41/cpu/moe_cpu.cpp`: E2M1 nibbles expanded in registers, AVX-512 VNNI int8 dot products, rows split
  across the two NUMA nodes with node-local first touch, threads pinned to physical cores). Only the
  10 KB activation and the 20 KB MoE output cross PCIe per layer; the GPU runs attention, the dense
  projections and the shared expert (overlapped with the CPU) inside CUDA graphs. Measured on this box
  (2× Xeon Silver 4410Y, DDR5-4000 ×32 DIMMs, 229 GB/s measured read ceiling): CPU experts ~0.13 ms per
  expert plus ~0.05 ms per layer, i.e. ~24 ms/token when every expert is cold.
- `--offload-experts cpu --hot-experts 64`: hybrid. The 64 most used experts of each layer (from a
  routing profile, `--route-stats` / `results/route_stats.pt`; on this text the top 20% of experts take
  82% of the hits) also live on the GPU and are computed there together with the shared expert while the
  CPU computes the cold ones; the two partial sums are added. Uses ~49 GB more GPU memory for 64/layer.
  Measured with 80 hot experts per layer: **35 tok/s** on an English prompt in the profile's domain (78%
  of expert hits on the GPU, GPU side 13 ms/token), 22 tok/s on a Japanese prompt where the static profile
  hits 19%. `dsv41/hotcache.py` replaces resident experts by recent usage (a background thread copies the
  weights over PCIe through pinned staging; `DSV41_ADAPTIVE_HOT=0` disables it, `DSV41_HOT_SWAPS` swaps per
  token): the Japanese hit rate rises to 40-56% over a few hundred tokens.
- `--offload-experts gpu`: the experts are DMA'd from pinned RAM into a GPU staging buffer and computed on
  the GPU. 4.5 GB per token over PCIe 4.0 x16 (25 GB/s measured) → **2.3 tok/s**. Kept for reference.

```
python -m dsv41.chat  --devices 2 --offload-experts          # REPL on one GPU (CUDA_VISIBLE_DEVICES also works)
python -m dsv41.serve --devices 2 --offload-experts --port 8000
```

The dense weights go to the GPU (~20 GiB used in total). Prefill (more than 16 tokens) currently streams
all experts of a layer through the GPU in chunks of 64 (each (token, expert) pair is computed in its
expert's chunk); from pageable NUMA memory this takes ~40 s per prompt and is the next thing to fix.
Decode uses three CUDA graphs per layer (dense part → CPU experts → post-processing) and one host sync
per layer. The CPU path wants huge pages (the expert buffers are `MADV_HUGEPAGE`; with a fragmented
page cache the kernel silently falls back to 4 KiB pages and streaming gets ~10% slower), the workers
pinned to all hardware threads except one core per NUMA node (kept for the main thread, the CUDA driver
and the cache copy thread), `OMP_WAIT_POLICY=active` (set before torch is imported), the expert rows
first-touched under a per-thread `MPOL_BIND` (otherwise the kernel spills a node's rows to the other node
when its free memory is fragmented, and that node's threads run at half speed), `kernel.numa_balancing=0`.

## Numbers (8× A100 80GB PCIe, shared with other jobs)

| stage | decode tok/s |
|---|---|
| first working version | 4.0 |
| MoE dispatch without host syncs | 7.3 |
| fused Triton elementwise kernels | 15.3 |
| static-shape decode + CUDA graphs | 21.7 |
| CUDA C expert GEMV | 31.6 |
| split-slot decode attention | 33.6 |
| fused decode layer (~26 kernels instead of ~100; the 12 ms "rest" was ~4,000 tiny kernels inside the graphs) | 37.9 |
| dense weights kept FP8, decoded in registers to bf16 tensor-core operands (`cuda/fp8_tc.cu`) | 42.6 |
| FP4 experts decoded in registers to bf16 tensor-core operands, grouped by expert (`cuda/fp4_tc.cu`) | 47.4 |
| split-K epilogue in the kernel, hyper-connection split on a side stream | 51.9 |
| expert parallelism over 7 GPUs (`--ep`) | 65.7 |

Several sequences at once (`--batch B`, the rows advance in lockstep; `--batch-prompts FILE` for distinct
prompts): the dense weights are read once per step for all rows and the (token, expert) pairs are bucketed by
expert on the device, so an expert's weights are read once for all the tokens routed to it. Aggregate
throughput with 16 / 32 distinct prompts (`dsv41/batch_prompts*.txt`), routing messages by copy engine for
large batches:

| configuration | B=1 | B=8 | B=16 | B=32 |
|---|---|---|---|---|
| 8-GPU layer pipeline | 52 | – | 201 | – |
| expert parallelism, 8 GPUs | 64 | – | 397 (40 ms/step) | – |
| expert parallelism, 4 GPUs (2,3,0,1) | 64 | 248 | 365 (44 ms) | 456 (70 ms) |
| two independent 4-GPU replicas (all 8 GPUs) | – | – | 714 | **911 tok/s** |

A 4-GPU group is as fast as an 8-GPU one per step (the per-layer critical path is the owner's attention
and dense part, not the expert reads), so two 4-GPU replicas give twice the throughput of one 8-GPU
group. With the measured DSpark acceptance (3.13 tokens per verified step) the two-replica B=32
configuration would reach ~2,800 tok/s if verification were free; that verification step is the next thing to build.

GPU time per token on the pipeline: FP8 dense 6.3 ms, FP4 experts 5.0 ms, the rest ~8 ms (attention,
indexer top-k, small fused kernels). Prefill of a 1,413-token prompt: 4.8 s (296 tok/s). Load: ~70 s with
the checkpoint in page cache, ~4 min cold.

Why A100 can do this without FP8/FP4 units: an E4M3 byte placed as `s<<15 | e<<7 | m<<4` is a bf16 whose
value is the FP8 value times 2^-120 (the subnormals line up too), so one bf16x2 multiply by 2^(scale-7)
turns two packed weights into exactly dequantized bf16 operands for `mma.sync`; E2M1 nibbles work the same
way with 2^126. The weights stay 8-bit / 4-bit in HBM and the tensor cores do the accumulation in fp32.

DSpark (multi-token prediction) is implemented as an eager draft module (`dsv41/dspark.py`; `dsv41/mtp_accept.py`
measures it): on greedy English text 2.13 of the 5 drafts are accepted on average (78% for the first draft),
i.e. 3.13 tokens per verified step. The verification path (6-token steps in the graphs) is not built yet.
`dsv41/route_telemetry.py` records which experts a task uses: the top-64 experts of a layer carry 83-95% of
the routing weight for a given task and the sets differ a lot between tasks (a conversation-local working
set of ~80 experts per layer is what the single-GPU cache should hold).

Not implemented yet: the MTP verification step, continuous batching in the server, the vision encoder.
