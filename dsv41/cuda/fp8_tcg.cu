// FP8-weight GEMM on Ampere tensor cores for many rows (M >= 32): CUTLASS-style tiles, both operands staged in
// shared memory with a 4-stage cp.async pipeline and read with ldmatrix. Same exact e4m3 -> bf16 register decode
// and E8M0 [32 x 32] block scales as fp8_tc.cu; fp32 accumulation.
//
//   out[m, n] = sum_k x[m, k] * (W[n, k] * 2^(S[n/32, k/32] - 127))
//
// Weight layout: the bytes of every 16-k group are stored in the order k = 0,1,8,9, 2,3,10,11, 4,5,12,13, 6,7,14,15
// (W8.permute_k), so that the 32-bit word ldmatrix hands lane (g, t) — bytes 4t..4t+3 of row g — decodes straight
// into the mma A fragment (logical k 2t, 2t+1 and 2t+8, 2t+9). x keeps its natural order (ldmatrix gives the B
// fragment directly). The other kernels (fp8_tc.cu, fp8_tcw.cu) read the same permuted weights with adjusted x
// word indexing, and W8.bf16() undoes the permutation for prefill.
//
// Block: 256 threads = 8 warps, tile BM=64 tokens x BN=128 weight rows x BK=64 k per stage; warp (wm, wn) owns
// 32 tokens x 32 weight rows = 4 m-tiles x 2 n-tiles. grid: (N/128, ceil(M/64), splits). fp32 partials
// [splits, M, N] with the same last-block epilogue as fp8_tc.cu (counter per (m-block, n-block) tile).
// group_cols > 0 (block-diagonal o-projection, group_cols % 128 == 0): x row = token * (N/group_cols) + n_blk/group_cols.
#include <cuda_bf16.h>
#include <stdint.h>

#define BN 128
#define BK 128
#define STAGES 2
#define NTHR 256

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
__device__ __forceinline__ void e4m3x4_to_bf16(uint32_t w, uint32_t& lo, uint32_t& hi) {
    uint32_t t0 = prmt(w, 0, 0x1404);
    uint32_t t1 = prmt(w, 0, 0x3424);
    lo = ((t0 >> 4) & 0x07F007F0u) | (t0 & 0x80008000u);
    hi = ((t1 >> 4) & 0x07F007F0u) | (t1 & 0x80008000u);
}
__device__ __forceinline__ uint32_t hmul2_bf16(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(0u));
    return r;
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int src_bytes) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N_> __device__ __forceinline__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;" :: "n"(N_) : "memory"); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* smem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t* r, const void* smem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}

// smem layouts (bytes): W stage [BN rows][64 B], chunk c of row r at c ^ ((r >> 1) & 3); X stage [BM rows][128 B],
// chunk c of row r at c ^ (r & 7). Both make the 8-row ldmatrix phases bank-conflict free.
// W rows of 128 B (8 chunks): c ^ (r & 7); X rows of 256 B (16 chunks): (c ^ (r & 7)) keeps 8 consecutive rows on distinct bank groups
__device__ __forceinline__ int w_off(int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); }
__device__ __forceinline__ int x_off(int r, int c) { return r * 256 + ((c ^ (r & 7)) << 4); }

template <int BM>  // token rows per block (64: 4 m-tiles per warp, 2 blocks/SM; 128: 8 m-tiles, the decoded weight
                   // fragment feeds twice the mmas, 1 block/SM)
__device__ __forceinline__ void fp8_gemm_tcg_body(const __nv_bfloat16* __restrict__ X, int ldx, int M, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
             int N, int K, int Kc, float* __restrict__ out, int ldo, int k_per_split, int group_cols,
             __nv_bfloat16* __restrict__ y, unsigned int* __restrict__ counters, int splits, int tiled)
{
    constexpr int MTW = BM / 16;  // m-tiles (8 rows) per warp: the block's BM rows split over the 2 warp rows
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* ws = smem;                              // STAGES * BN * 64
    uint8_t* xs = smem + STAGES * BN * BK;           // STAGES * BM * BK * 2
    __shared__ unsigned int last_flag;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;
    const int g = lane >> 2, t = lane & 3;
    const int n_blk = blockIdx.x * BN, m_blk = blockIdx.y * BM;
    const int ks = blockIdx.z * k_per_split;
    const int ke = min(K, ks + k_per_split);
    const int xgroups = group_cols > 0 ? N / group_cols : 1;
    const int xg = group_cols > 0 ? n_blk / group_cols : 0;
    const int wstep = tiled ? 1024 : 64;  // weight bytes per 64 k from a lane's base pointer
    // global -> smem copy assignments (loop invariant): W: 512 chunks (row = i/4, chunk = i%4); X: 512 chunks (row = i/8, chunk = i%8)
    constexpr int WJ = BN * BK / 16 / NTHR, XJ = BM * BK * 2 / 16 / NTHR;
    const uint8_t* wsrc[WJ]; int wdst[WJ];
    const __nv_bfloat16* xsrc[XJ]; int xdst[XJ]; bool xok[XJ];
#pragma unroll
    for (int j = 0; j < WJ; ++j) {
        const int i = tid + j * NTHR;
        const int wr = i / (BK / 16), wc = i % (BK / 16);
        // row-major, or tiled [N/16][K/64][16 rows][64 B]: chunk wc (16 B of the 128-k stage) sits in tile step wc/4
        wsrc[j] = tiled ? W + (long long)((n_blk + wr) >> 4) * (K / 64) * 1024 + (wc >> 2) * 1024 + ((n_blk + wr) & 15) * 64 + (wc & 3) * 16
                        : W + (long long)(n_blk + wr) * K + wc * 16;
        wdst[j] = w_off(wr, wc);
    }
#pragma unroll
    for (int j = 0; j < XJ; ++j) {
        const int i = tid + j * NTHR;
        const int xr = i / (BK / 8), xc = i % (BK / 8);
        xok[j] = m_blk + xr < M;
        xsrc[j] = xok[j] ? X + ((long long)(m_blk + xr) * xgroups + xg) * ldx + xc * 8 : X;
        xdst[j] = x_off(xr, xc);
    }
    auto load_stage = [&](int kk, int st) {
        if (kk < ke) {
            uint8_t* wd = ws + st * (BN * BK);
            uint8_t* xd = xs + st * (BM * BK * 2);
#pragma unroll
            for (int j = 0; j < WJ; ++j) cp_async16(wd + wdst[j], wsrc[j] + (kk >> 6) * wstep, 16);   // K % BK == 0: always in range
#pragma unroll
            for (int j = 0; j < XJ; ++j) cp_async16(xd + xdst[j], xsrc[j] + kk, xok[j] ? 16 : 0);
        }
        cp_async_commit();
    };
    float acc[2][MTW][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < MTW; ++j) { acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f; }
    // ldmatrix lane addressing: W x4 over [16 rows][32 k bytes]: matrices 0,1 = rows 0-7 / 8-15 at k 0-15; 2,3 = same rows at k 16-31.
    // lane i supplies the row address of matrix i/8, row i%8.
    const int wl_row = (lane & 7) + ((lane >> 3) & 1) * 8;    // row within the 16-row n-tile
    const int wl_chunk = (lane >> 4);                          // 16-byte chunk (0 or 1) within the 32 k bytes
    // X x2 over [8 tokens][16 k]: matrices 0 (k 0-7), 1 (k 8-15); lanes 0-7 address matrix 0 rows, 8-15 matrix 1 rows
    const int xl_row = lane & 7;
    const int xl_chunk = (lane >> 3) & 1;
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) load_stage(ks + BK * i, i);
    // the warp's 32 weight rows share one scale row; its two bytes per stage (k halves) are prefetched a stage ahead
    const uint8_t* srow = S + (long long)((n_blk + wn * 32) >> 5) * Kc;
    uint32_t sc_next = ks < ke ? *reinterpret_cast<const uint32_t*>(srow + (ks >> 5)) : 0u;  // BK/32 = 4 scale bytes
    int it = 0;
    for (int k0 = ks; k0 < ke; k0 += BK, ++it) {
        const uint32_t sc2 = sc_next;
        if (k0 + BK < ke) sc_next = *reinterpret_cast<const uint32_t*>(srow + ((k0 + BK) >> 5));
        cp_async_wait_group<STAGES - 2>();
        __syncthreads();
        load_stage(k0 + BK * (STAGES - 1), (it + STAGES - 1) % STAGES);
        const uint8_t* wt = ws + (it % STAGES) * (BN * BK);
        const uint8_t* xt = xs + (it % STAGES) * (BM * BK * 2);
#pragma unroll
        for (int kh = 0; kh < BK / 32; ++kh) {
            uint32_t araw[2][4];
            uint32_t b[2][MTW][2];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                ldmatrix_x4(araw[nt], wt + w_off(wn * 32 + nt * 16 + wl_row, kh * 2 + wl_chunk));
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
                for (int mt = 0; mt < MTW; ++mt)
                    ldmatrix_x2(b[s][mt], xt + x_off(wm * (BM / 2) + mt * 8 + xl_row, (kh * 2 + s) * 2 + xl_chunk));
            const int sc = (sc2 >> (8 * kh)) & 0xFF;
            const uint32_t fb = (sc + 120 > 0) ? ((uint32_t)(sc + 120) << 7) : 0u;
            const uint32_t f2 = fb | (fb << 16);
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                uint32_t a[2][4];
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    uint32_t lo, hi;
                    e4m3x4_to_bf16(araw[nt][2 * s], lo, hi);
                    a[nt][0] = hmul2_bf16(lo, f2); a[nt][2] = hmul2_bf16(hi, f2);
                    e4m3x4_to_bf16(araw[nt][2 * s + 1], lo, hi);
                    a[nt][1] = hmul2_bf16(lo, f2); a[nt][3] = hmul2_bf16(hi, f2);
                }
#pragma unroll
                for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) mma16816(acc[nt][mt], a[nt], b[s][mt]);
            }
        }
    }
    cp_async_wait_group<0>();
    // store fp32 partials: acc[nt][mt]: c0,c1 = (n = wn*32 + nt*16 + g, tokens wm*32 + mt*8 + 2t, +1); c2,c3 = n + 8
    float* o = out + (long long)blockIdx.z * M * ldo;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int n = n_blk + wn * 32 + nt * 16 + g;
#pragma unroll
        for (int mt = 0; mt < MTW; ++mt) {
            const int m = m_blk + wm * (BM / 2) + mt * 8 + 2 * t;
            if (m < M) { o[(long long)m * ldo + n] = acc[nt][mt][0]; o[(long long)m * ldo + n + 8] = acc[nt][mt][2]; }
            if (m + 1 < M) { o[(long long)(m + 1) * ldo + n] = acc[nt][mt][1]; o[(long long)(m + 1) * ldo + n + 8] = acc[nt][mt][3]; }
        }
    }
    if (y == nullptr) return;
    __threadfence();
    __syncthreads();
    if (tid == 0) last_flag = (splits == 1) ? 1u : (atomicInc(&counters[blockIdx.y * gridDim.x + blockIdx.x], (unsigned)splits - 1) == (unsigned)(splits - 1));
    __syncthreads();
    if (!last_flag) return;
    __threadfence();
    const int mrows = min(BM, M - m_blk);
    for (int i = tid; i < mrows * BN; i += NTHR) {
        const int r = m_blk + i / BN, col = n_blk + i % BN;
        float a = 0.f;
        for (int sp = 0; sp < splits; ++sp) a += out[((long long)sp * M + r) * ldo + col];
        y[(long long)r * N + col] = __float2bfloat16(a);
    }
}

extern "C" __global__ void __launch_bounds__(NTHR, 2)
fp8_gemm_tcg(const __nv_bfloat16* __restrict__ X, int ldx, int M, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
             int N, int K, int Kc, float* __restrict__ out, int ldo, int k_per_split, int group_cols,
             __nv_bfloat16* __restrict__ y, unsigned int* __restrict__ counters, int splits, int tiled)
{ fp8_gemm_tcg_body<64>(X, ldx, M, W, S, N, K, Kc, out, ldo, k_per_split, group_cols, y, counters, splits, tiled); }

extern "C" __global__ void __launch_bounds__(NTHR, 1)
fp8_gemm_tcg128(const __nv_bfloat16* __restrict__ X, int ldx, int M, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
             int N, int K, int Kc, float* __restrict__ out, int ldo, int k_per_split, int group_cols,
             __nv_bfloat16* __restrict__ y, unsigned int* __restrict__ counters, int splits, int tiled)
{ fp8_gemm_tcg_body<128>(X, ldx, M, W, S, N, K, Kc, out, ldo, k_per_split, group_cols, y, counters, splits, tiled); }
