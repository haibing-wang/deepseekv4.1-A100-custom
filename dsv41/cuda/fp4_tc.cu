// Grouped expert GEMM with FP4 (E2M1) weights on Ampere tensor cores: the weights stay packed (4 bit)
// in HBM, are decoded to bf16 in registers (exact: place [s e1 e0 m] as s<<15 | e<<7 | m<<6 and
// multiply by 2^126; normals become 2^(e-1)(1+m/2) and the bf16 subnormal 64m*2^-133 = m*2^-127
// becomes m/2, the E2M1 subnormal), the E8M0 block scale is folded into the same bf16x2 multiply
// (factor 2^(s-1)), and mma.m16n8k16 accumulates in fp32. Same numbers as the current GEMV, a
// quarter of the instructions per weight, and all tokens routed to an expert share one weight read.
//
//   out[p, n] = sum_k x[tok[p], k] * W[e_g, n, k] * 2^(S[e_g, n, k/32] - 127)     for pairs p of group g
//
// Groups: grp_expert[g] is the expert, pairs grp_start[g] .. grp_start[g+1]-1 its (<= 16) tokens, and
// pair_tok[p] the x row of pair p. x must be given in the "8-k permuted" layout produced by
// cukern.permute_x (within every 8 consecutive k: order 0,4,2,6,1,5,3,7), which matches the order in
// which the nibbles come out of a 32-bit word (even bytes' low nibbles, odd bytes' low, even high,
// odd high): a dot product does not care about the order of its terms.
//
// Lane (g = lane/4, t = lane%4) of a warp owns weight row n = n0 + g and 32 consecutive k per
// 128-k step (K0 + 32t ..), i.e. one 16-byte load and one scale byte; 8 mma steps per 128 k.
#include <cuda_bf16.h>
#include <stdint.h>

#define WARPS 4

__device__ __forceinline__ uint32_t bf16x2_fma0(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(0u));
    return r;
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// 8 nibbles (one 32-bit word: bytes 0..3 = k 0..7, low nibble = even k) -> 4 bf16x2 words holding the
// k pairs (0,4), (2,6), (1,5), (3,7) as unscaled placed bits, then one fma by the folded scale.
__device__ __forceinline__ void e2m1x8_to_bf16(uint32_t w, uint32_t f2, uint32_t* o) {
    // low nibbles of bytes 0,2 -> halves 0,1 ; magnitude bits [2:0] -> [8:6], sign bit 3 -> 15
    uint32_t a = ((w & 0x00070007u) << 6) | ((w & 0x00080008u) << 12);
    // low nibbles of bytes 1,3: magnitude [10:8] -> [8:6], sign bit 11 -> 15
    uint32_t b = ((w & 0x07000700u) >> 2) | ((w & 0x08000800u) << 4);
    // high nibbles of bytes 0,2: magnitude [6:4] -> [8:6], sign bit 7 -> 15
    uint32_t c = ((w & 0x00700070u) << 2) | ((w & 0x00800080u) << 8);
    // high nibbles of bytes 1,3: magnitude [14:12] -> [8:6], sign bit 15 stays
    uint32_t d = ((w & 0x70007000u) >> 6) | (w & 0x80008000u);
    o[0] = bf16x2_fma0(a, f2);
    o[1] = bf16x2_fma0(b, f2);
    o[2] = bf16x2_fma0(c, f2);
    o[3] = bf16x2_fma0(d, f2);
}

template <bool M8>
__device__ __forceinline__ void fp4_gemm_body(const __nv_bfloat16* __restrict__ X, int ldx,
        const uint8_t* __restrict__ W, long long stride_we, const uint8_t* __restrict__ S, long long stride_se,
        const int* __restrict__ grp_expert, const int* __restrict__ grp_start, const int* __restrict__ pair_tok,
        float* __restrict__ out, int ldo, int N, int K, int shard_start, int shard_n, int zero_out, int min_tok, int max_tok)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int n0 = (blockIdx.x * WARPS + warp) * 8;
    if (n0 >= N) return;
    const int grp = blockIdx.y;
    const int p0 = grp_start[grp], p1 = grp_start[grp + 1];
    const int M = p1 - p0;
    if (M < min_tok || M > max_tok) return;  // groups of other sizes go to fp4_tcw.cu
    int e = grp_expert[grp] - shard_start;
    if (e < 0 || e >= shard_n) {  // not this GPU's expert (expert parallelism): skip, optionally zero the rows
        if (zero_out && lane < 8) {
            for (int r = 0; r < M; ++r) out[(long long)(p0 + r) * ldo + n0 + lane] = 0.f;
        }
        return;
    }
    const int n = n0 + g;
    const uint8_t* wrow = W + (long long)e * stride_we + (long long)n * (K / 2);
    const uint8_t* srow = S + (long long)e * stride_se + (long long)n * (K / 32);
    // x rows for A fragment rows g and g+8
    const __nv_bfloat16* x0 = g < M ? X + (long long)pair_tok[p0 + g] * ldx : nullptr;
    const __nv_bfloat16* x1 = (!M8 && g + 8 < M) ? X + (long long)pair_tok[p0 + g + 8] * ldx : nullptr;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    const uint4 z4 = make_uint4(0, 0, 0, 0);
    uint4 wv = z4, xa[4] = {z4, z4, z4, z4}, xb[4] = {z4, z4, z4, z4};
    int sb = 0;
    auto load = [&](int k0) {
        const int kb = k0 + 32 * t;
        wv = __ldg(reinterpret_cast<const uint4*>(wrow + (kb >> 1)));
        sb = __ldg(srow + (kb >> 5));
        if (x0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) xa[i] = *reinterpret_cast<const uint4*>(x0 + kb + 8 * i);
        }
        if (!M8) {
            if (x1) {
#pragma unroll
                for (int i = 0; i < 4; ++i) xb[i] = *reinterpret_cast<const uint4*>(x1 + kb + 8 * i);
            }
        }
    };
    load(0);
    for (int k0 = 0; k0 < K; k0 += 128) {
        const uint4 wc = wv; const int sc = sb;
        uint4 ya[4] = {xa[0], xa[1], xa[2], xa[3]}, yb[4] = {xb[0], xb[1], xb[2], xb[3]};
        if (k0 + 128 < K) load(k0 + 128);
        const uint32_t fb = (uint32_t)(sc + 126) << 7;  // 2^(s-1) as bf16
        const uint32_t f2 = fb | (fb << 16);
        uint32_t b[16];
        e2m1x8_to_bf16(wc.x, f2, b);
        e2m1x8_to_bf16(wc.y, f2, b + 4);
        e2m1x8_to_bf16(wc.z, f2, b + 8);
        e2m1x8_to_bf16(wc.w, f2, b + 12);
        const uint32_t* xav = reinterpret_cast<const uint32_t*>(ya);  // 16 words: x pairs in the same order as b
        const uint32_t* xbv = reinterpret_cast<const uint32_t*>(yb);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const uint32_t bf[2] = {b[2 * s], b[2 * s + 1]};
            const uint32_t af[4] = {xav[2 * s], M8 ? 0u : xbv[2 * s], xav[2 * s + 1], M8 ? 0u : xbv[2 * s + 1]};
            mma16816(c, af, bf);
        }
    }
    // C: rows g / g+8 (pairs p0+g, p0+g+8), columns n0 + 2t, 2t+1
    if (g < M) { float* o = out + (long long)(p0 + g) * ldo + n0 + 2 * t; o[0] = c[0]; o[1] = c[1]; }
    if (!M8 && g + 8 < M) { float* o = out + (long long)(p0 + g + 8) * ldo + n0 + 2 * t; o[0] = c[2]; o[1] = c[3]; }
}

extern "C" __global__ void __launch_bounds__(WARPS * 32)
fp4_gemm_tc8(const __nv_bfloat16* __restrict__ X, int ldx, const uint8_t* __restrict__ W, long long stride_we,
             const uint8_t* __restrict__ S, long long stride_se, const int* __restrict__ grp_expert, const int* __restrict__ grp_start,
             const int* __restrict__ pair_tok, float* __restrict__ out, int ldo, int N, int K, int shard_start, int shard_n, int zero_out, int min_tok, int max_tok)
{ fp4_gemm_body<true>(X, ldx, W, stride_we, S, stride_se, grp_expert, grp_start, pair_tok, out, ldo, N, K, shard_start, shard_n, zero_out, min_tok, max_tok); }

extern "C" __global__ void __launch_bounds__(WARPS * 32)
fp4_gemm_tc16(const __nv_bfloat16* __restrict__ X, int ldx, const uint8_t* __restrict__ W, long long stride_we,
              const uint8_t* __restrict__ S, long long stride_se, const int* __restrict__ grp_expert, const int* __restrict__ grp_start,
              const int* __restrict__ pair_tok, float* __restrict__ out, int ldo, int N, int K, int shard_start, int shard_n, int zero_out, int min_tok, int max_tok)
{ fp4_gemm_body<false>(X, ldx, W, stride_we, S, stride_se, grp_expert, grp_start, pair_tok, out, ldo, N, K, shard_start, shard_n, zero_out, min_tok, max_tok); }
