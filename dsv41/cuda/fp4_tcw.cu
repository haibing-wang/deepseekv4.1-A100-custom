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
#ifndef NT_16
#define NT_16 2
#endif
#ifndef NT_32
#define NT_32 2
#endif

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

template <int MT, int STAGES, int NW, int NT>  // NW warps x (NT x 16) columns per block; NT n-tiles per warp share the x fragments
__device__ __forceinline__ void fp4_gemm_tcw_body(const __nv_bfloat16* __restrict__ X, int ldx,
        const uint8_t* __restrict__ W, long long stride_we, const uint8_t* __restrict__ S, long long stride_se,
        const int* __restrict__ grp_expert, const int* __restrict__ grp_start, const int* __restrict__ pair_tok,
        float* __restrict__ out, int ldo, int N, int K, int shard_start, int shard_n, int zero_out, int min_tok, int max_tok, int tiled)
{
    constexpr int ROWS = 8 * MT;
    constexpr int CH = 16;  // 16-byte chunks per row per 128 k
    extern __shared__ __align__(16) uint4 xs[];
    constexpr int BN = NW * 16 * NT, NTHR = NW * 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int n_blk = blockIdx.x * BN;
    if (n_blk >= N) return;  // launches use grid.x = N / 64 whatever NT: wider blocks leave the tail idle
    const int n0 = n_blk + warp * 16 * NT;
    const int grp = blockIdx.y;
    const int p0 = grp_start[grp], p1 = grp_start[grp + 1];
    const int M = p1 - p0;
    if (M < min_tok || M > max_tok) return;
    const int e = grp_expert[grp] - shard_start;
    if (e < 0 || e >= shard_n) {
        if (zero_out) {
            for (int i = threadIdx.x; i < M * BN; i += NTHR)
                out[(long long)(p0 + i / BN) * ldo + n_blk + i % BN] = 0.f;
        }
        return;
    }
    // lane's first weight bytes / scale byte of rows n0+g and n0+g+8, and the step per 128 k (row-major or the
    // tiled layout [N/16][K/128][16 rows][64 B], scales [N/16][K/128][16][4]; n0 is a multiple of 16)
    // per n-tile j (rows n0 + 16 j + g and + 8): lane's first weight bytes / scale byte and the step per 128 k
    const uint8_t* wrowA[NT]; const uint8_t* wrowB[NT]; const uint8_t* srowA[NT]; const uint8_t* srowB[NT];
    int wstep, sstep;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int nj = n0 + 16 * j;
        if (tiled) {
            const uint8_t* wt0 = W + (long long)e * stride_we + (long long)(nj >> 4) * (K / 128) * 1024 + 16 * t;
            const uint8_t* st0 = S + (long long)e * stride_se + (long long)(nj >> 4) * (K / 128) * 64 + t;
            wrowA[j] = wt0 + g * 64; wrowB[j] = wt0 + (g + 8) * 64; srowA[j] = st0 + g * 4; srowB[j] = st0 + (g + 8) * 4;
            wstep = 1024; sstep = 64;
        } else {
            wrowA[j] = W + (long long)e * stride_we + (long long)(nj + g) * (K / 2) + 16 * t;
            wrowB[j] = W + (long long)e * stride_we + (long long)(nj + g + 8) * (K / 2) + 16 * t;
            srowA[j] = S + (long long)e * stride_se + (long long)(nj + g) * (K / 32) + t;
            srowB[j] = S + (long long)e * stride_se + (long long)(nj + g + 8) * (K / 32) + t;
            wstep = 64; sstep = 4;
        }
    }
    float c[NT][MT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) { c[j][mt][0] = c[j][mt][1] = c[j][mt][2] = c[j][mt][3] = 0.f; }
    const uint4 z4 = make_uint4(0, 0, 0, 0);
    uint4 wa0[NT], wb0[NT], wa1[NT], wb1[NT];
    int sa0[NT], sb0[NT], sa1[NT], sb1[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) { wa0[j] = wb0[j] = wa1[j] = wb1[j] = z4; sa0[j] = sb0[j] = sa1[j] = sb1[j] = 0; }
    auto load_w = [&](int kk, uint4 (&wa)[NT], uint4 (&wb)[NT], int (&sa)[NT], int (&sb)[NT]) {
        const int it_ = kk >> 7;  // 128-k iteration
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            wa[j] = __ldg(reinterpret_cast<const uint4*>(wrowA[j] + it_ * wstep));
            wb[j] = __ldg(reinterpret_cast<const uint4*>(wrowB[j] + it_ * wstep));
            sa[j] = __ldg(srowA[j] + it_ * sstep);
            sb[j] = __ldg(srowB[j] + it_ * sstep);
        }
    };
    // x tile staging: thread copies chunk (tid % 16) of rows tid/16 + 8j
    constexpr int XJ = (ROWS * CH + NTHR - 1) / NTHR;  // chunks per thread per stage
    constexpr int RSTEP = NTHR / CH;                    // rows covered per pass (8 or 16: (r & 7) stays the same)
    const int xr0 = threadIdx.x / CH, xc0 = threadIdx.x % CH;
    const __nv_bfloat16* xsrc[XJ];
#pragma unroll
    for (int j = 0; j < XJ; ++j) {
        const int r = xr0 + RSTEP * j;
        xsrc[j] = r < M ? X + (long long)pair_tok[p0 + r] * ldx + 8 * xc0 : nullptr;
    }
    const int xoff0 = xr0 * CH + (xc0 ^ (xr0 & 7));
    auto load_x = [&](int kk, int st) {
        if (kk < K) {
            uint4* dst = xs + st * (ROWS * CH) + xoff0;
#pragma unroll
            for (int j = 0; j < XJ; ++j) {
                if (xr0 + RSTEP * j < ROWS) {
                    const bool ok = xsrc[j] != nullptr;
                    cp_async16(dst + j * RSTEP * CH, ok ? xsrc[j] + kk : X, ok ? 16 : 0);
                }
            }
        }
        cp_async_commit();
    };
    auto iterate = [&](int k0, int it, uint4 (&wa)[NT], uint4 (&wb)[NT], int (&sa)[NT], int (&sb)[NT]) {
        const int st = it % STAGES;
        uint4 wca[NT], wcb[NT];
        int sca[NT], scb[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) { wca[j] = wa[j]; wcb[j] = wb[j]; sca[j] = sa[j]; scb[j] = sb[j]; }
        cp_async_wait_group<STAGES - 2>();
        __syncthreads();
        load_x(k0 + 128 * (STAGES - 1), (it + STAGES - 1) % STAGES);
        if (k0 + 256 < K) load_w(k0 + 256, wa, wb, sa, sb);
        uint32_t da[NT][16], db[NT][16];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const uint32_t fa = (uint32_t)(sca[j] + 126) << 7, fb = (uint32_t)(scb[j] + 126) << 7;  // 2^(s-1) as bf16
            decode32(wca[j], fa | (fa << 16), da[j]);
            decode32(wcb[j], fb | (fb << 16), db[j]);
        }
        const uint4* xt = xs + st * (ROWS * CH) + g * CH;
#pragma unroll
        for (int jc = 0; jc < 4; ++jc) {  // this lane's 4 chunks (32 k) of every token tile
            const uint4* xp = xt + ((4 * t + jc) ^ g);
            uint32_t xw[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const uint4 v = xp[mt * 8 * CH];
                xw[mt][0] = v.x; xw[mt][1] = v.y; xw[mt][2] = v.z; xw[mt][3] = v.w;
            }
#pragma unroll
            for (int s2 = 0; s2 < 2; ++s2) {  // two mma steps per chunk
                const int s = 2 * jc + s2;
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const uint32_t bf[2] = {xw[mt][2 * s2], xw[mt][2 * s2 + 1]};
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        const uint32_t af[4] = {da[j][2 * s], db[j][2 * s], da[j][2 * s + 1], db[j][2 * s + 1]};
                        mma16816(c[j][mt], af, bf);
                    }
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
    // C[j][mt]: rows n0+16j+g (c0,c1) / +8 (c2,c3), columns = tokens 8*mt + 2t, 2t+1
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int nj = n0 + 16 * j;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int t0 = 8 * mt + 2 * t;
            if (t0 < M) { float* o = out + (long long)(p0 + t0) * ldo; o[nj + g] = c[j][mt][0]; o[nj + g + 8] = c[j][mt][2]; }
            if (t0 + 1 < M) { float* o = out + (long long)(p0 + t0 + 1) * ldo; o[nj + g] = c[j][mt][1]; o[nj + g + 8] = c[j][mt][3]; }
        }
    }
}

extern "C" __global__ void __launch_bounds__(WARPS * 32)
fp4_gemm_tcw(const __nv_bfloat16* __restrict__ X, int ldx, const uint8_t* __restrict__ W, long long stride_we,
        const uint8_t* __restrict__ S, long long stride_se, const int* __restrict__ grp_expert, const int* __restrict__ grp_start,
        const int* __restrict__ pair_tok, float* __restrict__ out, int ldo, int N, int K, int shard_start, int shard_n, int zero_out,
        int min_tok, int max_tok, int tiled)
{
    // 4 warps; token-tile count per group (dynamic smem sized for MT = 8); groups of 9..32 tokens use 2 n-tiles per
    // warp (128 columns per block: the x fragments feed twice the mmas and the tile is staged half as often)
    const int M = grp_start[blockIdx.y + 1] - grp_start[blockIdx.y];
    if (M <= 8)
        fp4_gemm_tcw_body<1, 3, 4, 1>(X, ldx, W, stride_we, S, stride_se, grp_expert, grp_start, pair_tok, out, ldo, N, K, shard_start, shard_n, zero_out, min_tok, max_tok, tiled);
    else if (M <= 16)
        fp4_gemm_tcw_body<2, 3, 4, NT_16>(X, ldx, W, stride_we, S, stride_se, grp_expert, grp_start, pair_tok, out, ldo, N, K, shard_start, shard_n, zero_out, min_tok, max_tok, tiled);
    else if (M <= 32)
        fp4_gemm_tcw_body<4, 3, 4, NT_32>(X, ldx, W, stride_we, S, stride_se, grp_expert, grp_start, pair_tok, out, ldo, N, K, shard_start, shard_n, zero_out, min_tok, max_tok, tiled);
    else
        fp4_gemm_tcw_body<8, 3, 4, 1>(X, ldx, W, stride_we, S, stride_se, grp_expert, grp_start, pair_tok, out, ldo, N, K, shard_start, shard_n, zero_out, min_tok, max_tok, tiled);
}
