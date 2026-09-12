// Device-side messaging between GPUs of one process (peer access enabled): copies by kernel stores into
// peer memory, flag signal / wait kernels. All capturable in CUDA graphs; used by the expert-parallel
// runtime (dsv41/ep.py) so that a whole token runs as one graph per GPU without host round trips.
#include <stdint.h>

extern "C" __global__ void p2p_copy(uint4* __restrict__ dst, const uint4* __restrict__ src, int n16) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n16) dst[i] = src[i];
}

// for every row b: copy row `row16` uint4 from src[b] into dst_base + seq[b] * bstride16 + row_idx[b] * row16
extern "C" __global__ void p2p_copy_row(uint4* __restrict__ dst_base, const long long* __restrict__ row_idx, const uint4* __restrict__ src,
                                       int row16, int nb, long long bstride16, const long long* __restrict__ seq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row16 * nb) return;
    int b = i / row16, j = i - b * row16;
    dst_base[seq[b] * bstride16 + row_idx[b] * row16 + j] = src[i];
}

// per group g (< groups): dst[g * dst_stride + i] = sum over `rows` rows of src[(g * rows + r) * n + i]
extern "C" __global__ void p2p_sum_rows(float* __restrict__ dst, const float* __restrict__ src, int rows, int n, int groups, long long dst_stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (i >= n || g >= groups) return;
    float acc = 0.f;
    for (int r = 0; r < rows; ++r) acc += src[((long long)g * rows + r) * n + i];
    dst[(long long)g * dst_stride + i] = acc;
}

// dst[g, i] = sum_r src[idx[g * rows + r], col0 + i] for i < n (src rows `src_ld` apart; dst rows `dst_stride` apart):
// the expert output rows of token g summed straight from the expert-sorted order, over one column chunk
extern "C" __global__ void p2p_sum_rows_idx(float* __restrict__ dst, const float* __restrict__ src, const int* __restrict__ idx,
                                            int rows, int n, int groups, long long src_ld, int col0, long long dst_stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (i >= n || g >= groups) return;
    float acc = 0.f;
    for (int r = 0; r < rows; ++r) acc += src[(long long)idx[g * rows + r] * src_ld + col0 + i];
    dst[(long long)g * dst_stride + i] = acc;
}

// after the preceding kernels of this stream have completed: publish `value` (read from seq_ptr) to up to 8 peer flags
extern "C" __global__ void p2p_signal(int** flags, int n, const int* seq_ptr) {
    __threadfence_system();
    int v = *seq_ptr;
    for (int i = 0; i < n; ++i) { volatile int* f = (volatile int*)flags[i]; *f = v; }
    __threadfence_system();
}

// spin until every one of n local flags has reached the value in seq_ptr
extern "C" __global__ void p2p_wait(volatile int* flags, int n, const int* seq_ptr) {
    int v = *seq_ptr;
    for (int i = 0; i < n; ++i) { while (flags[i] < v) { } }
    __threadfence_system();
}

extern "C" __global__ void p2p_seq_bump(int* seq_ptr) { *seq_ptr += 1; }

// one launch: copy `n16` uint4 of src into up to 8 destinations (peer inboxes); the last block to finish
// signals their flags (counter reset for the next launch)
extern "C" __global__ void p2p_multicast(uint4** dsts, int ndst, const uint4* __restrict__ src, int n16,
                                        int** flags, const int* seq_ptr, unsigned int* counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n16) {
        uint4 v = src[i];
        for (int d = 0; d < ndst; ++d) dsts[d][i] = v;
    }
    if (flags == nullptr) return;
    __threadfence_system();
    __syncthreads();
    __shared__ unsigned int last;
    if (threadIdx.x == 0) last = (atomicInc(counter, gridDim.x - 1) == gridDim.x - 1);
    __syncthreads();
    if (last && threadIdx.x == 0) {
        __threadfence_system();
        int v = *seq_ptr;
        for (int d = 0; d < ndst; ++d) { volatile int* f = (volatile int*)flags[d]; *f = v; }
        __threadfence_system();
    }
}

// timeline stamp: the global nanosecond timer (same clock on every GPU of the node)
extern "C" __global__ void p2p_stamp(long long* dst) {
    long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *dst = t;
}


// MoE dispatch in one launch (one block): from the router's (token, expert, weight) pairs build the expert-sorted
// tables the grouped FP4 GEMM wants — histogram, prefix, scatter, groups of <= gmax pairs — instead of a sort and
// a dozen small kernels. The order of a token's pairs within an expert is arbitrary (atomic cursors); every pair
// row is computed independently and summed per token in fixed j order, so the results do not depend on it.
//   eid [n] int32, wt [n] fp32 (pair p = token p / topk) -> tok_sorted [n], wt_sorted [n], inv [n] (sorted slot of
//   pair p), grp_expert [n] (-1 beyond the last group), grp_start [n + 1].
#define DISPATCH_MAX_E 512
extern "C" __global__ void moe_dispatch(const int* __restrict__ eid, const float* __restrict__ wt, int n, int topk, int E, int gmax,
                                        int* __restrict__ tok_sorted, float* __restrict__ wt_sorted, int* __restrict__ inv,
                                        int* __restrict__ grp_expert, int* __restrict__ grp_start) {
    __shared__ int count[DISPATCH_MAX_E], start[DISPATCH_MAX_E], gbase[DISPATCH_MAX_E], cursor[DISPATCH_MAX_E];
    __shared__ int total_groups;
    const int tid = threadIdx.x, nt = blockDim.x;
    for (int e = tid; e < E; e += nt) { count[e] = 0; cursor[e] = 0; }
    __syncthreads();
    for (int p = tid; p < n; p += nt) atomicAdd(&count[eid[p]], 1);
    __syncthreads();
    if (tid == 0) {  // serial prefix over E experts (a few hundred): pair starts and group bases
        int s = 0, g = 0;
        for (int e = 0; e < E; ++e) {
            start[e] = s; gbase[e] = g;
            s += count[e];
            g += (count[e] + gmax - 1) / gmax;
        }
        total_groups = g;
    }
    __syncthreads();
    for (int p = tid; p < n; p += nt) {
        const int e = eid[p];
        const int pos = start[e] + atomicAdd(&cursor[e], 1);
        tok_sorted[pos] = p / topk;
        wt_sorted[pos] = wt[p];
        inv[p] = pos;
    }
    for (int e = tid; e < E; e += nt) {
        const int ng = (count[e] + gmax - 1) / gmax;
        for (int g = 0; g < ng; ++g) { grp_expert[gbase[e] + g] = e; grp_start[gbase[e] + g] = start[e] + g * gmax; }
    }
    const int G = total_groups;
    for (int g = G + tid; g < n; g += nt) { grp_expert[g] = -1; grp_start[g] = n; }
    if (tid == 0) grp_start[G] = n;
    if (tid == 0) grp_start[n] = n;
}
