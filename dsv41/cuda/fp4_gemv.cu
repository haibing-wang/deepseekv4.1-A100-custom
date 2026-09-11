// Decode-time expert GEMV for DeepSeek-V4.1 on A100: out[pair, n] = wt[pair] * sum_k x[row_in[pair], k] * W[expert[pair], n, k]
// W: E2M1 packed two per byte along k (low nibble = even k), one E8M0 scale per (n, 32 k).
// One warp streams whole rows with 16-byte loads (512 contiguous bytes per warp instruction); lane l
// owns the 32-k chunks c = l, l+32, ...; the activation is staged in shared memory transposed
// (xs[j][c] = x[32c + j]) so the 32 lanes read consecutive addresses (no bank conflicts); nibbles are
// decoded through a 16-entry shared LUT and accumulated in fp32. No atomics: one output per (pair, n).
#include <cuda_bf16.h>
#include <stdint.h>

#define WARPS 8
#define ROWS_PER_WARP 4
#define MAX_K 5120
#define MAX_CHUNKS (MAX_K / 32)

extern "C" __global__ void __launch_bounds__(WARPS * 32)
fp4_gemv_pairs(const __nv_bfloat16* __restrict__ X, int x_stride,
               const uint8_t* __restrict__ W, long long stride_we, int stride_wn,
               const uint8_t* __restrict__ S, long long stride_se, int stride_sn,
               const int* __restrict__ row_in, const int* __restrict__ expert, const float* __restrict__ wt,
               float* __restrict__ out, int out_stride, int N, int K)
{
    __shared__ float xs[32 * MAX_CHUNKS];  // transposed: xs[j * chunks + c] = x[32c + j]
    __shared__ float lut[16];
    const int pair = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int chunks = K / 32;
    if (tid < 16) {
        const float mags[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        lut[tid] = (tid & 8) ? -mags[tid & 7] : mags[tid & 7];
    }
    const __nv_bfloat16* xrow = X + (long long)row_in[pair] * x_stride;
    for (int k = tid; k < K; k += WARPS * 32) xs[(k & 31) * chunks + (k >> 5)] = __bfloat162float(xrow[k]);
    __syncthreads();
    const int e = expert[pair];
    const uint8_t* Wb = W + (long long)e * stride_we;
    const uint8_t* Sb = S + (long long)e * stride_se;
    const int n0 = (blockIdx.x * WARPS + warp) * ROWS_PER_WARP;
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int n = n0 + r;
        if (n >= N) break;
        const uint4* wrow = reinterpret_cast<const uint4*>(Wb + (long long)n * stride_wn);
        const uint8_t* srow = Sb + (long long)n * stride_sn;
        float acc = 0.f;
        for (int c = lane; c < chunks; c += 32) {
            const uint4 p = __ldg(wrow + c);
            const int sb = __ldg(srow + c);
            const float* xc = xs + c;  // element j at xc[j * chunks]
            float s = 0.f;
#define NIB(word, b, j) \
            s = fmaf(lut[((word) >> (8 * (b))) & 0xF], xc[(j) * chunks], s); \
            s = fmaf(lut[((word) >> (8 * (b) + 4)) & 0xF], xc[((j) + 1) * chunks], s);
            NIB(p.x, 0, 0) NIB(p.x, 1, 2) NIB(p.x, 2, 4) NIB(p.x, 3, 6)
            NIB(p.y, 0, 8) NIB(p.y, 1, 10) NIB(p.y, 2, 12) NIB(p.y, 3, 14)
            NIB(p.z, 0, 16) NIB(p.z, 1, 18) NIB(p.z, 2, 20) NIB(p.z, 3, 22)
            NIB(p.w, 0, 24) NIB(p.w, 1, 26) NIB(p.w, 2, 28) NIB(p.w, 3, 30)
#undef NIB
            const float scale = (sb == 0) ? 0.f : __int_as_float(sb << 23);  // 2^(sb-127)
            acc = fmaf(s, scale, acc);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) out[(long long)pair * out_stride + n] = wt[pair] * acc;
    }
}
