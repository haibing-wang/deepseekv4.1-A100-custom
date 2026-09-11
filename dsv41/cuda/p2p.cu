// Device-side messaging between GPUs of one process (peer access enabled): copies by kernel stores into
// peer memory, flag signal / wait kernels. All capturable in CUDA graphs; used by the expert-parallel
// runtime (dsv41/ep.py) so that a whole token runs as one graph per GPU without host round trips.
#include <stdint.h>

extern "C" __global__ void p2p_copy(uint4* __restrict__ dst, const uint4* __restrict__ src, int n16) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n16) dst[i] = src[i];
}

// copy a row of `row16` uint4 from src into dst_base + row_idx * row16 (row index read from device memory)
extern "C" __global__ void p2p_copy_row(uint4* __restrict__ dst_base, const long long* __restrict__ row_idx, const uint4* __restrict__ src, int row16) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row16) dst_base[(long long)(*row_idx) * row16 + i] = src[i];
}

// sum `rows` fp32 rows of src [rows, n] into dst [n] (used to push a partial expert output)
extern "C" __global__ void p2p_sum_rows(float* __restrict__ dst, const float* __restrict__ src, int rows, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int r = 0; r < rows; ++r) acc += src[(long long)r * n + i];
    dst[i] = acc;
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
