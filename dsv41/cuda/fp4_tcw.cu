// Grouped FP4 (E2M1) expert GEMM on Ampere tensor cores, second layout for groups with many tokens: the
// weights are the mma A operand (16 rows n per warp) and the tokens the B operand (8 per tile), so a group of
// up to 64 tokens reads the expert once (fp4_tc.cu re-reads it per 16 tokens). Same numbers as fp4_tc.cu:
// exact E2M1 -> bf16 register decode with the E8M0 block scale folded in, fp32 accumulation.
//
//   out[p, n] = sum_k x[tok[p], k] * W[e_g, n, k] * 2^(S[e_g, n, k/32] - 127)      pairs p of group g
//
// Lane (g = lane/4, t = lane%4) owns weight rows n0+g and n0+g+8 and 32 consecutive k per 128-k iteration
// (K0 + 32t ..: one 16-byte load and one scale byte per row); mma step s (0..7) uses word s of the decoded
// row, i.e. nibble pairs (0,4),(2,6),(1,5),(3,7) of byte group s, and x is given in the matching "8-k
// permuted" layout (cukern.permute_x), loaded per token as the same 32 k (four 16-byte chunks). The x tile
// of a group ([8*MT tokens, 128 k], chunk c of row r at c ^ (r & 7)) is staged in shared memory with
// cp.async (STAGES deep); the weights are prefetched two iterations ahead in registers.
// Only groups with min_tok <= tokens <= max_tok are processed (the launcher pairs this kernel with fp4_tc.cu
// for the small groups); groups of other GPUs' experts are skipped (zero_out: their rows zeroed).
// grid: (N / 64, groups); block 128 threads; dynamic shared memory STAGES * 8 * MT * 256 bytes.
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

__device__ __forceinline__ void e2m1x8_to_bf16(uint32_t w, uint32_t f2, uint32_t* o) {
    uint32_t a = ((w & 0x00070007u) << 6) | ((w & 0x00080008u) << 12);
    uint32_t b = ((w & 0x07000700u) >> 2) | ((w & 0x08000800u) << 4);
    uint32_t c = ((w & 0x00700070u) << 2) | ((w & 0x00800080u) << 8);
    uint32_t d = ((w & 0x70007000u) >> 6) | (w & 0x80008000u);
    o[0] = bf16x2_fma0(a, f2);
    o[1] = bf16x2_fma0(b, f2);
    o[2] = bf16x2_fma0(c, f2);
    o[3] = bf16x2_fma0(d, f2);
}

// 16 bytes (32 nibbles, 32 k) of one row -> 16 bf16x2 words scaled by f2
__device__ __forceinline__ void decode32(const uint4& w, uint32_t f2, uint32_t* o) {
    e2m1x8_to_bf16(w.x, f2, o);
    e2m1x8_to_bf16(w.y, f2, o + 4);
    e2m1x8_to_bf16(w.z, f2, o + 8);
    e2m1x8_to_bf16(w.w, f2, o + 12);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int src_bytes) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N_> __device__ __forceinline__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;" :: "n"(N_) : "memory"); }

template <int MT, int STAGES>
__device__ __forceinline__ void fp4_gemm_tcw_body(const __nv_bfloat16* __restrict__ X, int ldx,
        const uint8_t* __restrict__ W, long long stride_we, const uint8_t* __restrict__ S, long long stride_se,
        const int* __restrict__ grp_expert, const int* __restrict__ grp_start, const int* __restrict__ pair_tok,
        float* __restrict__ out, int ldo, int N, int K, int shard_start, int shard_n, int zero_out, int min_tok, int max_tok)
{
    constexpr int ROWS = 8 * MT;
    constexpr int CH = 16;  // 16-byte chunks per row per 128 k
    extern __shared__ __align__(16) uint4 xs[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int n_blk = blockIdx.x * (WARPS * 16);
    const int n0 = n_blk + warp * 16;
    const int grp = blockIdx.y;
    const int p0 = grp_start[grp], p1 = grp_start[grp + 1];
    const int M = p1 - p0;
    if (M < min_tok || M > max_tok) return;
    const int e = grp_expert[grp] - shard_start;
    if (e < 0 || e >= shard_n) {
        if (zero_out) {
            for (int i = threadIdx.x; i < M * (WARPS * 16); i += WARPS * 32)
                out[(long long)(p0 + i / (WARPS * 16)) * ldo + n_blk + i % (WARPS * 16)] = 0.f;
        }
        return;
    }
    const uint8_t* wrowA = W + (long long)e * stride_we + (long long)(n0 + g) * (K / 2);
    const uint8_t* wrowB = W + (long long)e * stride_we + (long long)(n0 + g + 8) * (K / 2);
    const uint8_t* srowA = S + (long long)e * stride_se + (long long)(n0 + g) * (K / 32);
    const uint8_t* srowB = S + (long long)e * stride_se + (long long)(n0 + g + 8) * (K / 32);
    float c[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) { c[mt][0] = c[mt][1] = c[mt][2] = c[mt][3] = 0.f; }
    const uint4 z4 = make_uint4(0, 0, 0, 0);
    uint4 wa0 = z4, wb0 = z4, wa1 = z4, wb1 = z4;
    int sa0 = 0, sb0 = 0, sa1 = 0, sb1 = 0;
    auto load_w = [&](int kk, uint4& wa, uint4& wb, int& sa, int& sb) {
        const int kb = kk + 32 * t;  // 32 k of this lane
        wa = __ldg(reinterpret_cast<const uint4*>(wrowA + (kb >> 1)));
        wb = __ldg(reinterpret_cast<const uint4*>(wrowB + (kb >> 1)));
        sa = __ldg(srowA + (kb >> 5));
        sb = __ldg(srowB + (kb >> 5));
    };
    // x tile staging: thread copies chunk (tid % 16) of rows tid/16 + 8j
    constexpr int XJ = ROWS * CH / (WARPS * 32);
    const int xr0 = threadIdx.x / CH, xc0 = threadIdx.x % CH;
    const __nv_bfloat16* xsrc[XJ];
#pragma unroll
    for (int j = 0; j < XJ; ++j) {
        const int r = xr0 + 8 * j;
        xsrc[j] = r < M ? X + (long long)pair_tok[p0 + r] * ldx + 8 * xc0 : nullptr;
    }
    const int xoff0 = xr0 * CH + (xc0 ^ (xr0 & 7));
    auto load_x = [&](int kk, int st) {
        if (kk < K) {
            uint4* dst = xs + st * (ROWS * CH) + xoff0;
#pragma unroll
            for (int j = 0; j < XJ; ++j) {
                const bool ok = xsrc[j] != nullptr;
                cp_async16(dst + j * 8 * CH, ok ? xsrc[j] + kk : X, ok ? 16 : 0);
            }
        }
        cp_async_commit();
    };
    auto iterate = [&](int k0, int it, uint4& wa, uint4& wb, int& sa, int& sb) {
        const int st = it % STAGES;
        const uint4 wca = wa, wcb = wb;
        const int sca = sa, scb = sb;
        cp_async_wait_group<STAGES - 2>();
        __syncthreads();
        load_x(k0 + 128 * (STAGES - 1), (it + STAGES - 1) % STAGES);
        if (k0 + 256 < K) load_w(k0 + 256, wa, wb, sa, sb);
        uint32_t da[16], db[16];
        {
            const uint32_t fa = (uint32_t)(sca + 126) << 7, fb = (uint32_t)(scb + 126) << 7;  // 2^(s-1) as bf16
            decode32(wca, fa | (fa << 16), da);
            decode32(wcb, fb | (fb << 16), db);
        }
        const uint4* xt = xs + st * (ROWS * CH) + g * CH;
#pragma unroll
        for (int j = 0; j < 4; ++j) {  // this lane's 4 chunks (32 k) of every token tile
            const uint4* xp = xt + ((4 * t + j) ^ g);
            uint32_t xw[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const uint4 v = xp[mt * 8 * CH];
                xw[mt][0] = v.x; xw[mt][1] = v.y; xw[mt][2] = v.z; xw[mt][3] = v.w;
            }
#pragma unroll
            for (int s2 = 0; s2 < 2; ++s2) {  // two mma steps per chunk
                const int s = 2 * j + s2;
                const uint32_t af[4] = {da[2 * s], db[2 * s], da[2 * s + 1], db[2 * s + 1]};
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const uint32_t bf[2] = {xw[mt][2 * s2], xw[mt][2 * s2 + 1]};
                    mma16816(c[mt], af, bf);
                }
            }
        }
    };
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) load_x(128 * i, i);
    load_w(0, wa0, wb0, sa0, sb0);
    if (128 < K) load_w(128, wa1, wb1, sa1, sb1);
    for (int k0 = 0, it = 0; k0 < K; ) {
        iterate(k0, it, wa0, wb0, sa0, sb0);
        k0 += 128; ++it;
        if (k0 >= K) break;
        iterate(k0, it, wa1, wb1, sa1, sb1);
        k0 += 128; ++it;
    }
    cp_async_wait_group<0>();
    // C[mt]: rows n0+g (c0,c1) / n0+g+8 (c2,c3), columns = tokens 8*mt + 2t, 2t+1
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int t0 = 8 * mt + 2 * t;
        if (t0 < M) { float* o = out + (long long)(p0 + t0) * ldo; o[n0 + g] = c[mt][0]; o[n0 + g + 8] = c[mt][2]; }
        if (t0 + 1 < M) { float* o = out + (long long)(p0 + t0 + 1) * ldo; o[n0 + g] = c[mt][1]; o[n0 + g + 8] = c[mt][3]; }
    }
}

extern "C" __global__ void __launch_bounds__(WARPS * 32)
fp4_gemm_tcw(const __nv_bfloat16* __restrict__ X, int ldx, const uint8_t* __restrict__ W, long long stride_we,
        const uint8_t* __restrict__ S, long long stride_se, const int* __restrict__ grp_expert, const int* __restrict__ grp_start,
        const int* __restrict__ pair_tok, float* __restrict__ out, int ldo, int N, int K, int shard_start, int shard_n, int zero_out,
        int min_tok, int max_tok)
{
    // token-tile count chosen per group (the padded tiles cost mma work, not bytes); dynamic smem sized for MT = 8
    const int M = grp_start[blockIdx.y + 1] - grp_start[blockIdx.y];
    if (M <= 16)
        fp4_gemm_tcw_body<2, 3>(X, ldx, W, stride_we, S, stride_se, grp_expert, grp_start, pair_tok, out, ldo, N, K, shard_start, shard_n, zero_out, min_tok, max_tok);
    else if (M <= 32)
        fp4_gemm_tcw_body<4, 3>(X, ldx, W, stride_we, S, stride_se, grp_expert, grp_start, pair_tok, out, ldo, N, K, shard_start, shard_n, zero_out, min_tok, max_tok);
    else
        fp4_gemm_tcw_body<8, 3>(X, ldx, W, stride_we, S, stride_se, grp_expert, grp_start, pair_tok, out, ldo, N, K, shard_start, shard_n, zero_out, min_tok, max_tok);
}
