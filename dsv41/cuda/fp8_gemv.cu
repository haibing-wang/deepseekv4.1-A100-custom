// Decode-time dense GEMV with FP8 (e4m3) weights and E8M0 scales per 32x32 block, for A100:
// out[m, n] = sum_k x[m, k] * W[n, k] * 2^(S[n/32, k/32] - 127), x bf16 [M, K] (M <= 16), out fp32.
// One warp per row: 16-byte loads (16 fp8 values), e4m3 decoded through a 256-entry shared LUT.
// NOTE: currently slower than cuBLAS bf16 (which runs at ~1.3 TB/s on A100); kept for future tuning, unused by the model.
// activations staged transposed in shared memory (x[m][k] at xs[(k & 15) * (M * chunks) + m * chunks + k/16]).
#include <cuda_bf16.h>
#include <stdint.h>

#define WARPS 8
#define ROWS_PER_WARP 4
#define MAX_K 8192
#define MAX_M 1
#define CH 16  // fp8 values per 16-byte chunk

__device__ __forceinline__ float e4m3_to_float(int b) {
    const int sign = b >> 7, exp = (b >> 3) & 0xF, mant = b & 7;
    float v;
    if (exp == 0) v = ldexpf((float)mant, -9);           // subnormal: mant/8 * 2^-6
    else v = ldexpf(1.0f + mant * 0.125f, exp - 7);       // normal
    return sign ? -v : v;
}

extern "C" __global__ void __launch_bounds__(WARPS * 32)
fp8_gemv(const __nv_bfloat16* __restrict__ X, int x_stride, int M,
         const uint8_t* __restrict__ W, int stride_wn,
         const uint8_t* __restrict__ S, int stride_sn,
         float* __restrict__ out, int out_stride, int N, int K)
{
    extern __shared__ float smem[];
    float* lut = smem;                 // 256
    float* xs = smem + 256;            // 16 * M * chunks
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int chunks = K / CH;
    for (int i = tid; i < 256; i += WARPS * 32) lut[i] = e4m3_to_float(i);
    for (int i = tid; i < M * K; i += WARPS * 32) {
        const int m = i / K, k = i - m * K;
        xs[(k & 15) * (M * chunks) + m * chunks + (k >> 4)] = __bfloat162float(X[(long long)m * x_stride + k]);
    }
    __syncthreads();
    const int n0 = (blockIdx.x * WARPS + warp) * ROWS_PER_WARP;
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int n = n0 + r;
        if (n >= N) break;
        const uint4* wrow = reinterpret_cast<const uint4*>(W + (long long)n * stride_wn);
        const uint8_t* srow = S + (long long)(n >> 5) * stride_sn;
        float acc[MAX_M];
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) acc[m] = 0.f;
        for (int c = lane; c < chunks; c += 32) {
            const uint4 p = __ldg(wrow + c);
            const int sb = __ldg(srow + (c >> 1));  // one scale per 32 k = 2 chunks
            const float scale = (sb == 0) ? 0.f : __int_as_float(sb << 23);
            float wv[CH];
            const uint32_t words[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
            for (int wi = 0; wi < 4; ++wi) {
#pragma unroll
                for (int b = 0; b < 4; ++b) wv[wi * 4 + b] = lut[(words[wi] >> (8 * b)) & 0xFF] * scale;
            }
            for (int m = 0; m < M; ++m) {
                const float* xc = xs + m * chunks + c;
                float s = 0.f;
#pragma unroll
                for (int j = 0; j < CH; ++j) s = fmaf(wv[j], xc[j * (M * chunks)], s);
                acc[m] += s;
            }
        }
        for (int m = 0; m < M; ++m) {
            float a = acc[m];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
            if (lane == 0) out[(long long)m * out_stride + n] = a;
        }
    }
}
