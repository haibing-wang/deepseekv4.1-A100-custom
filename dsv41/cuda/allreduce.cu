// Push-style all-reduce for small bf16 vectors across P2P-connected GPUs (single process, no NCCL).
// Every GPU d has: buf[W][n] (peers write row d... i.e. peer p writes its vector into buf_d[p]),
// flags[W] (peer p sets flags_d[p] = seq once its row has landed), and a seq counter.
// Three kernels per round on each GPU (all capturable in a CUDA graph):
//   ar_push:   copy x into peer_bufs[p] + me*n for every peer p (P2P stores), and into own buf[me].
//   ar_signal: fence, set flags_p[me] = seq on every peer, spin until own flags[p] == seq for all p, seq++.
//   ar_reduce: out = sum_p buf[p].
#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __global__ void ar_push(const uint4* __restrict__ x, uint4** peer_rows, int W, int n16) {
    // peer_rows[p] = address of row `me` inside GPU p's buf (already offset); includes self.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n16 * W; i += stride) {
        int p = i / n16, j = i - p * n16;
        peer_rows[p][j] = x[j];
    }
}

extern "C" __global__ void ar_signal(int** peer_flags, volatile int* my_flags, int* seq_ptr, int W) {
    // single thread
    __threadfence_system();
    int seq = *seq_ptr;
    for (int p = 0; p < W; p++) {
        volatile int* f = (volatile int*)peer_flags[p];
        *f = seq;  // peer_flags[p] = &flags_p[me]
    }
    __threadfence_system();
    for (int p = 0; p < W; p++) {
        while (my_flags[p] < seq) { }
    }
    __threadfence_system();
    *seq_ptr = seq + 1;
}

extern "C" __global__ void ar_reduce(const __nv_bfloat162* __restrict__ buf, __nv_bfloat162* __restrict__ out, int W, int n2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n2) return;
    float2 acc = __bfloat1622float2(buf[i]);
    for (int p = 1; p < W; p++) {
        float2 v = __bfloat1622float2(buf[p * n2 + i]);
        acc.x += v.x; acc.y += v.y;
    }
    out[i] = __float22bfloat162_rn(acc);
}

// Generalised signal: set `n_set` peer flags, then wait until `n_wait` own flags reach seq. seq_ptr is
// advanced by `bump` (so two phases of one round can share a counter: phase 1 bump=0, phase 2 bump=1
// with seq offsets handled by the caller through separate flag arrays).
extern "C" __global__ void ar_signal2(int** peer_flags, int n_set, volatile int* my_flags, int n_wait, int* seq_ptr, int bump) {
    __threadfence_system();
    int seq = *seq_ptr;
    for (int p = 0; p < n_set; p++) { volatile int* f = (volatile int*)peer_flags[p]; *f = seq; }
    __threadfence_system();
    for (int p = 0; p < n_wait; p++) { while (my_flags[p] < seq) { } }
    __threadfence_system();
    if (bump) *seq_ptr = seq + 1;
}
