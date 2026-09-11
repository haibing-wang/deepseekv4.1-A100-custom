// FP8-weight skinny GEMM on Ampere tensor cores, second layout: the weights are the A operand (16 rows n per
// warp), the activations the B operand (8 tokens per tile), so one decoded weight fragment is reused by every
// token tile: M <= 8 * MT rows of x per pass with the weights read exactly once (the first layout re-read the
// weights per 16 rows, which made batched decode cost ~linearly in rows).
//
//   out[m, n] = sum_k x[m, k] * (W[n, k] * 2^(S[n/32, k/32] - 127))
//
// Decode of e4m3 -> bf16 is exact (see fp8_tc.cu). A warp owns NT n-tiles of 16 weight rows (every x fragment
// read from shared memory feeds NT mmas: with NT = 1 the 128 B/cycle shared-memory port is the co-limit with the
// tensor cores). Lane (g = lane/4, t = lane%4) owns weight rows n0+g and n0+g+8 of each n-tile and physical k = K0 + 16t .. +15 of each 64-wide k step (one 16-byte load per row); mma step s (0..3)
// uses the physical pairs (16t+4s, +1) and (+2, +3) as logical k (2t, 2t+1) and (2t+8, 2t+9). x is loaded with
// the same permutation (lane: token 8*mt+g, k 16t..16t+15 as two 16-byte loads), so both operands agree.
// C tile: rows = n (n0+g / n0+g+8), columns = tokens (8*mt + 2t, +1); stored transposed to out[m, n].
//
// The x tile of an iteration ([8*MT tokens, 128 k] bf16, swizzled 16-byte chunks: chunk c of row r sits at c ^ (r & 7),
// which makes the fragment reads bank-conflict free) is staged in (dynamic) shared memory with cp.async, STAGES
// deep, so the block's 4 warps (64 weight rows) read x from L2 once instead of once per warp. The weights are
// prefetched two iterations ahead in registers: with one iteration of lookahead a block was latency bound
// (~1.2 us per 128-k iteration whatever M), and 3 blocks per SM only reached ~600 GB/s at M = 64.
// grid: (N / 64, splits). out: fp32 partials [splits, M, N]. Fused split-K epilogue as in fp8_tc.cu (per
// 16-column tile). group_cols > 0 (block-diagonal o-projection): x row = token * (N/group_cols) + n_blk/group_cols.
#include <cuda_bf16.h>
#include <stdint.h>

#define WARPS 4

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

// decode 16 e4m3 bytes (one row, 16 physical k) into 8 bf16x2 words scaled by f2
__device__ __forceinline__ void decode16(const uint4& w, uint32_t f2, uint32_t* o) {
    e4m3x4_to_bf16(w.x, o[0], o[1]);
    e4m3x4_to_bf16(w.y, o[2], o[3]);
    e4m3x4_to_bf16(w.z, o[4], o[5]);
    e4m3x4_to_bf16(w.w, o[6], o[7]);
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = hmul2_bf16(o[i], f2);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int src_bytes) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N_> __device__ __forceinline__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;" :: "n"(N_) : "memory"); }

template <int MT, int STAGES>
__device__ __forceinline__ void fp8_gemm_tcw_body(const __nv_bfloat16* __restrict__ X, int ldx, int M,
            const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int N, int K, int Kc,
            float* __restrict__ out, int ldo, int k_per_split, int group_cols,
            __nv_bfloat16* __restrict__ y, unsigned int* __restrict__ counters, int splits)
{
    constexpr int ROWS = 8 * MT;               // x rows staged per iteration
    constexpr int CH = 16;                     // 16-byte chunks per row (128 k)
    extern __shared__ __align__(16) uint4 xs[];  // [STAGES][ROWS * CH]
    __shared__ unsigned int last[WARPS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int n_blk = blockIdx.x * (WARPS * 16);
    const int n0 = n_blk + warp * 16;
    const int ks = blockIdx.y * k_per_split;
    const int ke = min(K, ks + k_per_split);
    const uint8_t* wrowA = W + (long long)(n0 + g) * K;
    const uint8_t* wrowB = W + (long long)(n0 + g + 8) * K;
    const uint8_t* srow = S + (long long)(n0 >> 5) * Kc;
    const int xgroups = group_cols > 0 ? N / group_cols : 1;
    const int xg = group_cols > 0 ? n_blk / group_cols : 0;
    float c[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) { c[mt][0] = c[mt][1] = c[mt][2] = c[mt][3] = 0.f; }
    const uint4 z4 = make_uint4(0, 0, 0, 0);
    // weights two iterations ahead in two explicit register sets (a runtime-indexed array would go to local memory)
    uint4 wa0[2] = {z4, z4}, wb0[2] = {z4, z4}, wa1[2] = {z4, z4}, wb1[2] = {z4, z4};
    int sb0[2] = {0, 0}, sb1[2] = {0, 0};
    auto load_w = [&](int kk, uint4 (&wa)[2], uint4 (&wb)[2], int (&sb)[2]) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int kb = kk + 64 * h + 16 * t;
            const bool ok = kb < ke;
            wa[h] = ok ? __ldg(reinterpret_cast<const uint4*>(wrowA + kb)) : z4;
            wb[h] = ok ? __ldg(reinterpret_cast<const uint4*>(wrowB + kb)) : z4;
            sb[h] = ok ? __ldg(srow + (kb >> 5)) : 0;
        }
    };
    // x tile [ROWS, 128 k] -> smem stage (rows >= M and k >= ke zero-filled); always commits a group.
    // Thread i copies chunk (i % 16) of rows i/16, i/16 + 8, ...: all loop invariant except the k offset.
    constexpr int XJ = ROWS * CH / (WARPS * 32);  // chunks per thread per stage (MT)
    const int xr0 = threadIdx.x / CH, xc0 = threadIdx.x % CH;
    const __nv_bfloat16* xsrc0 = X + ((long long)xr0 * xgroups + xg) * ldx + 8 * xc0;
    const long long xstride8 = 8LL * xgroups * ldx;       // 8 rows further
    const int xoff0 = xr0 * CH + (xc0 ^ (xr0 & 7));        // (xr0 + 8j) & 7 == xr0 & 7
    auto load_x = [&](int kk, int st) {
        if (kk < ke) {
            uint4* dst = xs + st * (ROWS * CH) + xoff0;
            const bool kok = kk + 8 * xc0 < ke;
#pragma unroll
            for (int j = 0; j < XJ; ++j) {
                const bool ok = kok && (xr0 + 8 * j < M);
                cp_async16(dst + j * 8 * CH, ok ? xsrc0 + j * xstride8 + kk : X, ok ? 16 : 0);
            }
        }
        cp_async_commit();
    };
    auto iterate = [&](int k0, int it, uint4 (&wa)[2], uint4 (&wb)[2], int (&sb)[2]) {
        const int st = it % STAGES;
        uint4 wca[2] = {wa[0], wa[1]}, wcb[2] = {wb[0], wb[1]};
        int sc[2] = {sb[0], sb[1]};
        cp_async_wait_group<STAGES - 2>();
        __syncthreads();  // stage `st` landed for everyone; everyone is done reading stage (it-1) % STAGES
        load_x(k0 + 128 * (STAGES - 1), (it + STAGES - 1) % STAGES);
        if (k0 + 256 < ke) load_w(k0 + 256, wa, wb, sb);
        const uint4* xt = xs + st * (ROWS * CH) + g * CH;  // this lane's row within every token tile
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t fb = (sc[h] + 120 > 0) ? ((uint32_t)(sc[h] + 120) << 7) : 0u;
            const uint32_t f2 = fb | (fb << 16);
            uint32_t da[8], db[8];
            decode16(wca[h], f2, da);
            decode16(wcb[h], f2, db);
            const uint4* x0p = xt + ((8 * h + 2 * t) ^ g);
            const uint4* x1p = xt + ((8 * h + 2 * t + 1) ^ g);
            uint32_t xw[MT][8];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const uint4 x0 = x0p[mt * 8 * CH], x1 = x1p[mt * 8 * CH];
                xw[mt][0] = x0.x; xw[mt][1] = x0.y; xw[mt][2] = x0.z; xw[mt][3] = x0.w;
                xw[mt][4] = x1.x; xw[mt][5] = x1.y; xw[mt][6] = x1.z; xw[mt][7] = x1.w;
            }
            // s outer, token tile inner: consecutive mmas are independent (the 4 k-steps of a tile form a chain)
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const uint32_t af[4] = {da[2 * s], db[2 * s], da[2 * s + 1], db[2 * s + 1]};
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const uint32_t bf[2] = {xw[mt][2 * s], xw[mt][2 * s + 1]};
                    mma16816(c[mt], af, bf);
                }
            }
        }
    };
    // prologue: x stages 0 .. STAGES-2, weights for iterations 0 and 1
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) load_x(ks + 128 * i, i);
    if (ks < ke) load_w(ks, wa0, wb0, sb0);
    if (ks + 128 < ke) load_w(ks + 128, wa1, wb1, sb1);
    for (int k0 = ks, it = 0; k0 < ke; ) {
        iterate(k0, it, wa0, wb0, sb0);
        k0 += 128; ++it;
        if (k0 >= ke) break;
        iterate(k0, it, wa1, wb1, sb1);
        k0 += 128; ++it;
    }
    cp_async_wait_group<0>();
    float* o = out + (long long)blockIdx.y * M * ldo;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int t0 = 8 * mt + 2 * t;
        if (t0 < M) { o[(long long)t0 * ldo + n0 + g] = c[mt][0]; o[(long long)t0 * ldo + n0 + g + 8] = c[mt][2]; }
        if (t0 + 1 < M) { o[(long long)(t0 + 1) * ldo + n0 + g] = c[mt][1]; o[(long long)(t0 + 1) * ldo + n0 + g + 8] = c[mt][3]; }
    }
    if (y == nullptr) return;
    __threadfence();
    if (lane == 0) last[warp] = (splits == 1) ? 1u : (atomicInc(&counters[n0 >> 4], (unsigned)splits - 1) == (unsigned)(splits - 1));
    __syncwarp();
    if (!last[warp]) return;
    __threadfence();
    for (int i = lane; i < M * 16; i += 32) {
        const int r = i >> 4, col = n0 + (i & 15);
        float acc = 0.f;
        for (int sp = 0; sp < splits; ++sp) acc += out[((long long)sp * M + r) * ldo + col];
        y[(long long)r * N + col] = __float2bfloat16(acc);
    }
}

#define KERNEL(MT, ST) \
extern "C" __global__ void __launch_bounds__(WARPS * 32) \
fp8_gemm_tcw##MT##s##ST(const __nv_bfloat16* __restrict__ X, int ldx, int M, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, \
             int N, int K, int Kc, float* __restrict__ out, int ldo, int k_per_split, int group_cols, \
             __nv_bfloat16* __restrict__ y, unsigned int* __restrict__ counters, int splits) \
{ fp8_gemm_tcw_body<MT, ST>(X, ldx, M, W, S, N, K, Kc, out, ldo, k_per_split, group_cols, y, counters, splits); }

KERNEL(2, 3)
KERNEL(4, 3)
KERNEL(8, 3)
KERNEL(2, 4)
KERNEL(4, 4)
KERNEL(8, 4)
