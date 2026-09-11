// CPU MoE for the single-GPU mode: the selected experts are computed on the CPU straight from the
// packed FP4 weights in RAM (E2M1 nibbles -> bf16 via vpermw, AVX-512 BF16 dot products, fp32
// accumulate, E8M0 block scale applied per 32 k). Only x (10 KB) and the MoE output (20 KB) cross PCIe.
// Threads are pinned one per physical core; rows are split between the two NUMA nodes so both
// memory controllers stream weights, and the weight pages are first-touched by the node that reads them.
#include <immintrin.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

static int g_threads = 24, g_cores_per_node = 12, g_nodes = 2;
static int g_cpus[256];
static cpu_set_t g_master_mask;  // the caller's affinity, restored after every entry point
static int g_master_saved = 0;
static void save_master() { if (!g_master_saved) { sched_getaffinity(0, sizeof(g_master_mask), &g_master_mask); g_master_saved = 1; } }
static void restore_master();
// pin the calling OpenMP thread to its CPU. Threads torch's shared runtime re-creates inherit the
// master's mask, so every parallel region re-pins (one cheap syscall per thread per region).
static thread_local int* g_pin_memo = nullptr;
static inline void pin_self() {
    static thread_local int pinned_to = -1;  // each OS thread pins itself once (re-created threads pin on first use)
    g_pin_memo = &pinned_to;
    int t = omp_get_thread_num();
    if (pinned_to == g_cpus[t]) return;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(g_cpus[t], &s);
    sched_setaffinity(0, sizeof(s), &s);
    pinned_to = g_cpus[t];
}
static void restore_master() {
    if (g_master_saved) sched_setaffinity(0, sizeof(g_master_mask), &g_master_mask);
    if (g_pin_memo) *g_pin_memo = -1;  // the master will pin itself again on the next region
}

extern "C" void cpumoe_init(int threads, int cores_per_node) {
    save_master();
    g_threads = threads; g_cores_per_node = cores_per_node; g_nodes = (threads + cores_per_node - 1) / cores_per_node;
    for (int t = 0; t < threads; ++t) g_cpus[t] = t;
    omp_set_dynamic(0); omp_set_num_threads(threads);
}

extern "C" int cpumoe_init_list(const int* cpus, int n, int cores_per_node) {
    save_master();
    g_threads = n; g_cores_per_node = cores_per_node; g_nodes = (n + cores_per_node - 1) / cores_per_node;
    for (int t = 0; t < n; ++t) g_cpus[t] = cpus[t];
    omp_set_dynamic(0); omp_set_num_threads(n);
    int got = 0;
#pragma omp parallel
    {
        pin_self();
#pragma omp master
        got = omp_get_num_threads();
    }
    restore_master();
    return got;  // number of threads actually provided by the runtime
}

extern "C" void* cpumoe_alloc(size_t bytes) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    // THP: in a small benchmark 4 KiB pages streamed faster (524 vs 696 us per 6-expert layer), but with the
    // real 460 GB of expert memory the page walks dominate (decode 37 -> 55 ms, prefill 40 -> 173 s), so huge pages stay on.
    if (!getenv("DSV41_NO_THP")) madvise(p, bytes, MADV_HUGEPAGE);
    return p;
}

// Bind the memory of an [E][N][row_bytes] buffer so that each node's row half lives on that node
// (MPOL_BIND per range, page granular). Without this the kernel spills to the other node when the
// local node's free memory is fragmented by the page cache (measured: 45 GB of node-1 rows on node 0,
// node-1 threads 2x slower). Must be called before the pages are first touched.
#define DSV41_MPOL_BIND 2
extern "C" int cpumoe_bind_rows(uint8_t* base, int E, int N, int row_bytes) {
    const long page = sysconf(_SC_PAGESIZE);
    int bad = 0;
    for (int e = 0; e < E; ++e)
        for (int node = 0; node < g_nodes; ++node) {
            int n0 = (int)((long)N * node / g_nodes), n1 = (int)((long)N * (node + 1) / g_nodes);
            uintptr_t a = (uintptr_t)base + ((size_t)e * N + n0) * row_bytes, b = (uintptr_t)base + ((size_t)e * N + n1) * row_bytes;
            a = (a + page - 1) & ~(uintptr_t)(page - 1); b &= ~(uintptr_t)(page - 1);
            if (b <= a) continue;
            unsigned long mask = 1UL << node;
            if (syscall(SYS_mbind, (void*)a, (size_t)(b - a), DSV41_MPOL_BIND, &mask, 64, 0) != 0) bad++;
        }
    return bad;
}

// Per-thread NUMA policy during first touch: MPOL_BIND to the thread's node, so the copy threads' pages
// really land on their node (with the default policy the kernel spills to the other node when the local
// free memory is fragmented by the page cache: measured 45 GB of node-1 rows on node 0, node-1 threads
// 2x slower). One syscall per thread per region: no VMA splitting (mbind per range hit vm.max_map_count).
#define DSV41_MPOL_DEFAULT 0
static inline void bind_self_node(int node) { unsigned long mask = 1UL << node; syscall(SYS_set_mempolicy, DSV41_MPOL_BIND, &mask, 64); }
static inline void unbind_self() { syscall(SYS_set_mempolicy, DSV41_MPOL_DEFAULT, nullptr, 0); }

// Placement check: fraction of sampled pages (every 2 MiB) of node k's row half that actually live on node k.
extern "C" double cpumoe_local_fraction(const uint8_t* base, int E, int N, int row_bytes) {
    long ok = 0, tot = 0;
    for (int e = 0; e < E; ++e)
        for (int node = 0; node < g_nodes; ++node) {
            int n0 = (int)((long)N * node / g_nodes), n1 = (int)((long)N * (node + 1) / g_nodes);
            const uint8_t* a = base + ((size_t)e * N + n0) * row_bytes; const uint8_t* b = base + ((size_t)e * N + n1) * row_bytes;
            // only huge pages fully inside the half (the boundary pages are shared by design)
            uintptr_t a2 = ((uintptr_t)a + (2 << 20) - 1) & ~(uintptr_t)((2 << 20) - 1);
            for (const uint8_t* p = (const uint8_t*)a2; p + (2 << 20) <= b; p += (2 << 20)) {
                void* pages[1] = {(void*)p}; int status[1] = {-1};
                if (syscall(SYS_move_pages, 0, 1, pages, nullptr, status, 0) == 0 && status[0] >= 0) { tot++; if (status[0] == node) ok++; }
            }
        }
    return tot ? (double)ok / tot : -1.0;
}

// node of the thread that will read row n of an N-row matrix: rows split evenly between nodes
static inline int row_node(int n, int N) { return (int)((long)n * g_nodes / N); }

// copy one expert matrix [N, row_bytes] into dst (first touch by the node that will read each row)
extern "C" void cpumoe_load_rows(uint8_t* dst, const uint8_t* src, int N, int row_bytes) {
    omp_set_dynamic(0); omp_set_num_threads(g_threads);
#pragma omp parallel
    {
        pin_self();
        int t = omp_get_thread_num(), node = t / g_cores_per_node, tin = t % g_cores_per_node;
        int n0 = (int)((long)N * node / g_nodes), n1 = (int)((long)N * (node + 1) / g_nodes);
        int per = (n1 - n0 + g_cores_per_node - 1) / g_cores_per_node;
        int a = n0 + tin * per, b = a + per < n1 ? a + per : n1;
        bind_self_node(node);
        if (a < b) memcpy(dst + (size_t)a * row_bytes, src + (size_t)a * row_bytes, (size_t)(b - a) * row_bytes);
        unbind_self();
    }
    restore_master();
}

// read bandwidth: node = -1 -> every thread reads its own node's half; node = k -> only node k's threads read node k's half
extern "C" double cpumoe_bw_test(const uint8_t* p, size_t bytes, int iters, int node_sel) {
    omp_set_dynamic(0); omp_set_num_threads(g_threads);
    double best = 0;
    static volatile long sinks[64];
    for (int it = 0; it < iters; ++it) {
        double t0 = omp_get_wtime();
#pragma omp parallel
        {
            pin_self();
            int t = omp_get_thread_num(), node = t / g_cores_per_node, tin = t % g_cores_per_node;
            if (node_sel < 0 || node == node_sel) {
                size_t b0 = bytes * node / g_nodes, b1 = bytes * (node + 1) / g_nodes;
                size_t per = (b1 - b0) / g_cores_per_node;
                const uint8_t* a = p + b0 + tin * per;
                __m512i acc = _mm512_setzero_si512();
                for (size_t i = 0; i + 64 <= per; i += 64) acc = _mm512_add_epi64(acc, _mm512_loadu_si512(a + i));
                sinks[t] = _mm512_reduce_add_epi64(acc);
            }
        }
        double dt = omp_get_wtime() - t0;
        double read = (node_sel < 0 ? bytes : bytes / g_nodes);
        double gbs = read / dt / 1e9;
        if (gbs > best) best = gbs;
    }
    restore_master();
    return best;
}

// ---------------------------------------------------------------- FP4 row dot product
alignas(64) static uint16_t LUT_BF16[32];  // E2M1 code -> bf16 bits
static void init_lut() {
    const float v[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    for (int i = 0; i < 32; ++i) { uint32_t u; memcpy(&u, &v[i & 15], 4); LUT_BF16[i] = (uint16_t)(u >> 16); }
}

static inline float u32_as_f32(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

// x: bf16[K] as uint16; w: packed [K/2]; s: e8m0 [K/32]
static inline float dot_row(const uint8_t* __restrict w, const uint8_t* __restrict s, const uint16_t* __restrict x, int K) {
    const __m512i lut = _mm512_load_si512(LUT_BF16);
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    const int C = K / 32;
    for (int c = 0; c < C; c += 2) {
        // block c
        __m128i p0 = _mm_loadu_si128((const __m128i*)(w + 16 * c));
        __m256i b16 = _mm256_cvtepu8_epi16(p0);  // 16 packed bytes -> 16 x u16
        __m256i lo = _mm256_and_si256(b16, _mm256_set1_epi16(0x0F));
        __m256i hi = _mm256_srli_epi16(b16, 4);
        __m512i codes = _mm512_or_si512(_mm512_cvtepu16_epi32(lo), _mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16));
        __m512i wb = _mm512_permutexvar_epi16(codes, lut);
        __m512 d = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh)wb, (__m512bh)_mm512_loadu_si512(x + 32 * c));
        float sc = (s[c] == 0) ? 0.f : u32_as_f32((uint32_t)s[c] << 23);
        acc0 = _mm512_fmadd_ps(_mm512_set1_ps(sc), d, acc0);
        if (c + 1 < C) {
            __m128i p1 = _mm_loadu_si128((const __m128i*)(w + 16 * (c + 1)));
            __m256i c16 = _mm256_cvtepu8_epi16(p1);
            __m256i lo1 = _mm256_and_si256(c16, _mm256_set1_epi16(0x0F));
            __m256i hi1 = _mm256_srli_epi16(c16, 4);
            __m512i codes1 = _mm512_or_si512(_mm512_cvtepu16_epi32(lo1), _mm512_slli_epi32(_mm512_cvtepu16_epi32(hi1), 16));
            __m512i wb1 = _mm512_permutexvar_epi16(codes1, lut);
            __m512 d1 = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh)wb1, (__m512bh)_mm512_loadu_si512(x + 32 * (c + 1)));
            float sc1 = (s[c + 1] == 0) ? 0.f : u32_as_f32((uint32_t)s[c + 1] << 23);
            acc1 = _mm512_fmadd_ps(_mm512_set1_ps(sc1), d1, acc1);
        }
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

static inline float bf16_to_f(uint16_t b);
static inline float u32_as_f32(uint32_t u);
// ---------------------------------------------------------------- INT8 VNNI path (W4A8)
// activations: per-32 symmetric int8 (u8 = q + 128 for vpdpbusd), weights: E2M1 -> 2*value as int8 via vpermb
alignas(64) static int8_t LUT_S8[64];
static void init_lut_s8() {
    const int v[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    for (int i = 0; i < 64; ++i) LUT_S8[i] = (int8_t)v[i & 15];
}

// quantize bf16[K] -> u8[K] (q+128) and per-32 scales (fp32); scale includes the 1/2 of the weight LUT (2*value)
static void quant_x_u8(const uint16_t* x, int K, uint8_t* xu, float* sx, uint8_t* xe = nullptr, uint8_t* xo = nullptr) {
    for (int b = 0; b < K; b += 32) {
        float amax = 0.f, v[32];
        for (int j = 0; j < 32; ++j) { v[j] = bf16_to_f(x[b + j]); float a = fabsf(v[j]); if (a > amax) amax = a; }
        float sc = amax > 0 ? amax / 127.f : 1.f;
        float inv = 1.f / sc;
        for (int j = 0; j < 32; ++j) { int q = (int)rintf(v[j] * inv); if (q > 127) q = 127; if (q < -127) q = -127; xu[b + j] = (uint8_t)(q + 128); }
        sx[b / 32] = sc * 0.5f;  // weights are stored as 2*value
    }
    if (xe) for (int k = 0; k < K; k += 2) { xe[k >> 1] = xu[k]; xo[k >> 1] = xu[k + 1]; }
}

// v2: 128 k per iteration straight from the packed bytes (no 16-bit widening): the low nibbles are the
// even k, the high nibbles the odd k, and the activation is pre-split the same way (xe/xo), so lane j of
// the two accumulated vpdpbusd covers the 8 consecutive k [8j, 8j+8). ~20 vector ops per 128 k for two rows.
#ifndef PF_DIST
#define PF_DIST 4096  // software prefetch distance (bytes) for the weight rows
#endif
alignas(64) static const int32_t IDX4[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
static inline void dot_row_v2(const uint8_t* __restrict w0, const uint8_t* __restrict s0, const uint8_t* __restrict w1, const uint8_t* __restrict s1,
                              const uint8_t* __restrict xe, const uint8_t* __restrict xo, const float* __restrict sx, int K, float* r0, float* r1) {
    const __m512i lut = _mm512_load_si512(LUT_S8);
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m512i ones = _mm512_set1_epi8(1);
    const __m512i idx4 = _mm512_load_si512(IDX4);
    const __m512i zero = _mm512_setzero_si512();
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    for (int c = 0; c < K; c += 128) {
        _mm_prefetch((const char*)(w0 + c / 2 + PF_DIST), _MM_HINT_T0);
        _mm_prefetch((const char*)(w1 + c / 2 + PF_DIST), _MM_HINT_T0);
        const __m512i xev = _mm512_loadu_si512(xe + c / 2);
        const __m512i xov = _mm512_loadu_si512(xo + c / 2);
        const int cb = c / 32;
        const __m512 sxv = _mm512_castps128_ps512(_mm_loadu_ps(sx + cb));
        // row 0
        __m512i p = _mm512_loadu_si512(w0 + c / 2);
        __m512i wlo = _mm512_permutexvar_epi8(_mm512_and_si512(p, m4), lut);
        __m512i whi = _mm512_permutexvar_epi8(_mm512_and_si512(_mm512_srli_epi16(p, 4), m4), lut);
        __m512i d = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(zero, xev, wlo), xov, whi);
        __m512i sw = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(zero, ones, wlo), ones, whi);
        __m512i v = _mm512_sub_epi32(d, _mm512_slli_epi32(sw, 7));
        __m512 sc = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu8_epi32(_mm_cvtsi32_si128(*(const int*)(s0 + cb))), 23));
        sc = _mm512_permutexvar_ps(idx4, _mm512_mul_ps(sc, sxv));
        acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(v), sc, acc0);
        // row 1
        p = _mm512_loadu_si512(w1 + c / 2);
        wlo = _mm512_permutexvar_epi8(_mm512_and_si512(p, m4), lut);
        whi = _mm512_permutexvar_epi8(_mm512_and_si512(_mm512_srli_epi16(p, 4), m4), lut);
        d = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(zero, xev, wlo), xov, whi);
        sw = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(zero, ones, wlo), ones, whi);
        v = _mm512_sub_epi32(d, _mm512_slli_epi32(sw, 7));
        sc = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu8_epi32(_mm_cvtsi32_si128(*(const int*)(s1 + cb))), 23));
        sc = _mm512_permutexvar_ps(idx4, _mm512_mul_ps(sc, sxv));
        acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(v), sc, acc1);
    }
    *r0 = _mm512_reduce_add_ps(acc0);
    *r1 = _mm512_reduce_add_ps(acc1);
}

// w: packed [K/2]; s: e8m0 [K/32]; xu: u8[K]; sx: fp32[K/32]
static inline float dot_row_vnni(const uint8_t* __restrict w, const uint8_t* __restrict s, const uint8_t* __restrict xu, const float* __restrict sx, int K) {
    const __m512i lut = _mm512_load_si512(LUT_S8);
    const __m512i m4 = _mm512_set1_epi16(0x0F);
    const __m512i ones = _mm512_set1_epi8(1);
    __m512 acc = _mm512_setzero_ps();
    for (int c = 0; c < K; c += 64) {
        __m256i p = _mm256_loadu_si256((const __m256i*)(w + c / 2));   // 32 bytes = 64 nibbles
        __m512i b16 = _mm512_cvtepu8_epi16(p);                         // 32 x u16
        __m512i lo = _mm512_and_si512(b16, m4);
        __m512i hi = _mm512_srli_epi16(b16, 4);
        __m512i codes = _mm512_or_si512(lo, _mm512_slli_epi16(hi, 8));   // 64 bytes: 2j = lo, 2j+1 = hi
        __m512i ws = _mm512_permutexvar_epi8(codes, lut);                // 64 x s8 (2*value)
        __m512i xv = _mm512_loadu_si512(xu + c);
        __m512i d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), xv, ws);     // sum (q+128)*w per 4 k
        __m512i sw = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, ws);  // sum w per 4 k
        __m512i v = _mm512_sub_epi32(d, _mm512_slli_epi32(sw, 7));          // - 128 * sum w
        int cb = c / 32;
        float sa = (s[cb] == 0) ? 0.f : u32_as_f32((uint32_t)s[cb] << 23) * sx[cb];
        float sb = (s[cb + 1] == 0) ? 0.f : u32_as_f32((uint32_t)s[cb + 1] << 23) * sx[cb + 1];
        __m512 sv = _mm512_mask_blend_ps(0xFF00, _mm512_set1_ps(sa), _mm512_set1_ps(sb));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(v), sv, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

// two rows (w0, w1) against the same activation; returns both dot products
static inline void dot_row_vnni2(const uint8_t* __restrict w0, const uint8_t* __restrict s0, const uint8_t* __restrict w1, const uint8_t* __restrict s1,
                                 const uint8_t* __restrict xu, const float* __restrict sx, int K, float* r0, float* r1) {
    const __m512i lut = _mm512_load_si512(LUT_S8);
    const __m512i m4 = _mm512_set1_epi16(0x0F);
    const __m512i ones = _mm512_set1_epi8(1);
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    for (int c = 0; c < K; c += 64) {
        __m512i xv = _mm512_loadu_si512(xu + c);
        int cb = c / 32;
        float sxa = sx[cb], sxb = sx[cb + 1];
        // row 0
        __m512i b16 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(w0 + c / 2)));
        __m512i codes = _mm512_or_si512(_mm512_and_si512(b16, m4), _mm512_slli_epi16(_mm512_srli_epi16(b16, 4), 8));
        __m512i ws = _mm512_permutexvar_epi8(codes, lut);
        __m512i v = _mm512_sub_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xv, ws), _mm512_slli_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, ws), 7));
        float sa = (s0[cb] == 0) ? 0.f : u32_as_f32((uint32_t)s0[cb] << 23) * sxa;
        float sb = (s0[cb + 1] == 0) ? 0.f : u32_as_f32((uint32_t)s0[cb + 1] << 23) * sxb;
        acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(v), _mm512_mask_blend_ps(0xFF00, _mm512_set1_ps(sa), _mm512_set1_ps(sb)), acc0);
        // row 1
        b16 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(w1 + c / 2)));
        codes = _mm512_or_si512(_mm512_and_si512(b16, m4), _mm512_slli_epi16(_mm512_srli_epi16(b16, 4), 8));
        ws = _mm512_permutexvar_epi8(codes, lut);
        v = _mm512_sub_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xv, ws), _mm512_slli_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, ws), 7));
        sa = (s1[cb] == 0) ? 0.f : u32_as_f32((uint32_t)s1[cb] << 23) * sxa;
        sb = (s1[cb + 1] == 0) ? 0.f : u32_as_f32((uint32_t)s1[cb + 1] << 23) * sxb;
        acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(v), _mm512_mask_blend_ps(0xFF00, _mm512_set1_ps(sa), _mm512_set1_ps(sb)), acc1);
    }
    *r0 = _mm512_reduce_add_ps(acc0);
    *r1 = _mm512_reduce_add_ps(acc1);
}

static inline float bf16_to_f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t f_to_bf16(float f) {  // round to nearest even
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t r = u + 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(r >> 16);
}
static inline float round_e4m3(float v) {
    float a = fabsf(v);
    if (a == 0.f) return v;
    int e; frexpf(a, &e); e -= 1;           // floor(log2 a)
    if (e < -6) e = -6;
    float ulp = ldexpf(1.f, e - 3);
    float r = rintf(a / ulp) * ulp;
    if (r > 448.f) r = 448.f;
    return v < 0 ? -r : r;
}
static inline float pow2_ceil_log2(float a) {  // 2^ceil(log2 a)
    int e; float m = frexpf(a, &e);         // a = m * 2^e, m in [0.5, 1)
    if (m == 0.5f) e -= 1;
    return ldexpf(1.f, e);
}

// silu(gate)*up with clamps, times routing weight, bf16 rounding, then FP8 fake quant per 32 -> h (bf16)
// for columns [b0, b1) of one expert
static void swiglu_quant_cols(const float* gu, int inter, float wt, float limit, uint16_t* h, int b0, int b1) {
    float tmp[32];
    for (int b = b0; b < b1; b += 32) {
        float amax = 1e-4f;
        for (int j = 0; j < 32; ++j) {
            float g = gu[b + j], u = gu[inter + b + j];
            if (limit > 0) { u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit); }
            float v = wt * (g / (1.f + expf(-g))) * u;
            v = bf16_to_f(f_to_bf16(v));
            tmp[j] = v;
            float av = fabsf(v); if (av > amax) amax = av;
        }
        float sc = pow2_ceil_log2(amax / 448.f);
        for (int j = 0; j < 32; ++j) {
            float q = tmp[j] / sc; q = fminf(fmaxf(q, -448.f), 448.f);
            h[b + j] = f_to_bf16(round_e4m3(q) * sc);
        }
    }
}

// One MoE layer for one token. Pointers per selected expert (E of them).
static int g_int8 = 2;
alignas(64) static int g_ctr1[16 * 8], g_ctr3[16 * 8];  // per-node dynamic work counters (one cache line each)
static int g_ch1 = 96, g_ch3 = 64;  // max rows per dynamic work item (stage 1 / stage 3)
extern "C" void cpumoe_set_chunks(int ch1, int ch3) { g_ch1 = ch1 > 0 ? ch1 : 64; g_ch3 = ch3 > 0 && ch3 <= 256 ? ch3 : 64; }
extern "C" void cpumoe_set_int8(int on) { g_int8 = on; }

extern "C" int cpumoe_forward(const uint8_t* const* w13, const uint8_t* const* s13, const uint8_t* const* w2, const uint8_t* const* s2,
                              int E, const uint16_t* x, const float* wts, float* out, int K, int inter, int dim, float limit,
                              float* gu, uint16_t* h) {
    static int lut_ready = 0;
    if (!lut_ready) { init_lut(); init_lut_s8(); lut_ready = 1; }
    const double t_entry = omp_get_wtime();
    if (omp_get_max_threads() != g_threads) { omp_set_dynamic(0); omp_set_num_threads(g_threads); }  // torch resets the shared OpenMP runtime's thread count
    const int N13 = 2 * inter;
    static uint8_t xu[8192]; static float sx[256];            // quantized x
    static uint8_t xe[4096], xo[4096];                        // even / odd k (v2 kernel)
    static uint8_t hu[16 * 4096]; static float sh[16 * 128];  // quantized h per expert
    static uint8_t he_[16 * 2048], ho_[16 * 2048];
    const int dbg = getenv("DSV41_CPU_DEBUG") != nullptr;
    double t0 = dbg ? omp_get_wtime() : 0, t1 = 0, t2 = 0, t3 = 0;
    if (g_int8) quant_x_u8(x, K, xu, sx, xe, xo);
    if (dbg) t1 = omp_get_wtime();
    static double tdbg[256][4];
    for (int k = 0; k < g_nodes; ++k) { g_ctr1[k * 16] = 0; g_ctr3[k * 16] = 0; }
    // one parallel region for the three stages (barriers instead of three fork/joins)
#pragma omp parallel
    {
        pin_self();
        const int t = omp_get_thread_num(), node = t / g_cores_per_node, tin = t % g_cores_per_node;
        if (dbg) tdbg[t][0] = omp_get_wtime();
        // stage 1: gu[e][n] = x . w13[e][n]. Rows are split by node (NUMA-local halves); within a node the
        // (expert, 32-row chunk) work items are handed out dynamically through an atomic counter, so a thread
        // that gets descheduled or shares a core with another process just takes fewer chunks.
        {
            const int n0 = (int)((long)N13 * node / g_nodes), n1 = (int)((long)N13 * (node + 1) / g_nodes);
            // work item size: ~8 items per thread (balance) but not below 8 rows (streaming); g_ch1 caps it
            int CH = (int)(((long)(n1 - n0) * E) / ((long)g_cores_per_node * 8)) & ~1;
            if (CH < 8) CH = 8;
            if (CH > g_ch1) CH = g_ch1;
            const int nch = (n1 - n0 + CH - 1) / CH, total = E * nch;
            for (;;) {
                int c = __atomic_fetch_add(&g_ctr1[node * 16], 1, __ATOMIC_RELAXED);
                if (c >= total) break;
                int e = c / nch, a = n0 + (c % nch) * CH, b = a + CH < n1 ? a + CH : n1;
                int n = a;
                if (g_int8 == 2)
                    for (; n + 1 < b; n += 2)
                        dot_row_v2(w13[e] + (size_t)n * (K / 2), s13[e] + (size_t)n * (K / 32), w13[e] + (size_t)(n + 1) * (K / 2), s13[e] + (size_t)(n + 1) * (K / 32),
                                   xe, xo, sx, K, &gu[(size_t)e * N13 + n], &gu[(size_t)e * N13 + n + 1]);
                else if (g_int8)
                    for (; n + 1 < b; n += 2)
                        dot_row_vnni2(w13[e] + (size_t)n * (K / 2), s13[e] + (size_t)n * (K / 32), w13[e] + (size_t)(n + 1) * (K / 2), s13[e] + (size_t)(n + 1) * (K / 32),
                                      xu, sx, K, &gu[(size_t)e * N13 + n], &gu[(size_t)e * N13 + n + 1]);
                for (; n < b; ++n)
                    gu[(size_t)e * N13 + n] = g_int8
                        ? dot_row_vnni(w13[e] + (size_t)n * (K / 2), s13[e] + (size_t)n * (K / 32), xu, sx, K)
                        : dot_row(w13[e] + (size_t)n * (K / 2), s13[e] + (size_t)n * (K / 32), x, K);
            }
        }
        if (dbg) tdbg[t][1] = omp_get_wtime();
#pragma omp barrier
        if (dbg && t == 0) t2 = omp_get_wtime();
        // stage 2: h[e] = fp8(bf16(wt * silu(gate) * up)) (+ int8 quantization for the w2 GEMV); blocks of 256 columns
        {
            const int CB = 256, nb = inter / CB;
#pragma omp for schedule(static)
            for (int i = 0; i < E * nb; ++i) {
                int e = i / nb, b0 = (i % nb) * CB;
                swiglu_quant_cols(gu + (size_t)e * N13, inter, wts[e], limit, h + (size_t)e * inter, b0, b0 + CB);
                if (g_int8) quant_x_u8(h + (size_t)e * inter + b0, CB, hu + (size_t)e * inter + b0, sh + (size_t)e * (inter / 32) + b0 / 32,
                                       he_ + ((size_t)e * inter + b0) / 2, ho_ + ((size_t)e * inter + b0) / 2);
            }
        }
        if (dbg && t == 0) t3 = omp_get_wtime();
        if (dbg) tdbg[t][2] = omp_get_wtime();
        // stage 3: out[n] = sum_e h[e] . w2[e][n]  (dynamic 32-row output chunks per node; expert-outer inside a chunk)
        {
            const int n0 = (int)((long)dim * node / g_nodes), n1 = (int)((long)dim * (node + 1) / g_nodes);
            int CH = (int)((n1 - n0) / (g_cores_per_node * 6)) & ~1;
            if (CH < 8) CH = 8;
            if (CH > g_ch3) CH = g_ch3;
            const int nch = (n1 - n0 + CH - 1) / CH;
            for (;;) {
                int c = __atomic_fetch_add(&g_ctr3[node * 16], 1, __ATOMIC_RELAXED);
                if (c >= nch) break;
                int a = n0 + c * CH, b = a + CH < n1 ? a + CH : n1;
                float accs[256];
                for (int i = 0; i < b - a; ++i) accs[i] = 0.f;
                if (g_int8 == 2) {
                    for (int e = 0; e < E; ++e) {
                        int n = a;
                        for (; n + 1 < b; n += 2) {
                            float r0, r1;
                            dot_row_v2(w2[e] + (size_t)n * (inter / 2), s2[e] + (size_t)n * (inter / 32), w2[e] + (size_t)(n + 1) * (inter / 2), s2[e] + (size_t)(n + 1) * (inter / 32),
                                       he_ + (size_t)e * inter / 2, ho_ + (size_t)e * inter / 2, sh + (size_t)e * (inter / 32), inter, &r0, &r1);
                            accs[n - a] += r0; accs[n + 1 - a] += r1;
                        }
                        for (; n < b; ++n)
                            accs[n - a] += dot_row_vnni(w2[e] + (size_t)n * (inter / 2), s2[e] + (size_t)n * (inter / 32), hu + (size_t)e * inter, sh + (size_t)e * (inter / 32), inter);
                    }
                } else {
                    for (int n = a; n < b; ++n)
                        for (int e = 0; e < E; ++e)
                            accs[n - a] += g_int8
                                ? dot_row_vnni(w2[e] + (size_t)n * (inter / 2), s2[e] + (size_t)n * (inter / 32), hu + (size_t)e * inter, sh + (size_t)e * (inter / 32), inter)
                                : dot_row(w2[e] + (size_t)n * (inter / 2), s2[e] + (size_t)n * (inter / 32), h + (size_t)e * inter, inter);
                }
                for (int i = 0; i < b - a; ++i) out[a + i] = accs[i];
            }
        }
        if (dbg) tdbg[t][3] = omp_get_wtime();
    }
    if (dbg) {
        double smin = 1e9, smax = 0, emin = 1e9, emax = 0;
        for (int i = 0; i < g_threads; i++) { double a = (tdbg[i][0] - t1) * 1e6, b = (tdbg[i][1] - t1) * 1e6; if (a < smin) smin = a; if (a > smax) smax = a; if (b < emin) emin = b; if (b > emax) emax = b; }
        double e3min = 1e9, e3max = 0; int slow = -1;
        for (int i = 0; i < g_threads; i++) { double b = (tdbg[i][3] - t3) * 1e6; if (b < e3min) e3min = b; if (b > e3max) { e3max = b; slow = i; } }
        restore_master();
        fprintf(stderr, "[cpumoe] E=%d quant %.0fus stage1 %.0fus (thread start %.0f..%.0f, end %.0f..%.0f) stage2 %.0fus stage3 %.0fus (end %.0f..%.0f slowest t%d) total %.0fus\n", E, (t1 - t0) * 1e6, (t2 - t1) * 1e6, smin, smax, emin, emax, (t3 - t2) * 1e6, (omp_get_wtime() - t3) * 1e6, e3min, e3max, slow, (omp_get_wtime() - t_entry) * 1e6);
        return 0;
    }
    restore_master();
    return 0;
}

// Copy one whole layer (E experts) into the NUMA-split host buffers in a single parallel region.
// dst layouts: w13 [E][2*inter][dim/2] (w1 rows then w3 rows), s13 [E][2*inter][dim/32], w2 [E][dim][inter/2], s2 [E][dim][inter/32].
// Node 0 threads copy the rows their node will read (w1 half of w13, first half of w2), node 1 the rest;
// experts are dealt round-robin to the threads of each node so the page faults on the source mmap run in parallel.
extern "C" void cpumoe_load_layer(uint8_t* w13, uint8_t* s13, uint8_t* w2, uint8_t* s2,
                                  const uint8_t* const* sw1, const uint8_t* const* sw3, const uint8_t* const* ss1, const uint8_t* const* ss3,
                                  const uint8_t* const* sw2, const uint8_t* const* ss2, int E, int inter, int dim) {
    omp_set_dynamic(0); omp_set_num_threads(g_threads);
    const size_t rb13 = dim / 2, rs13 = dim / 32, rb2 = inter / 2, rs2 = inter / 32;
    const size_t b13 = (size_t)2 * inter * rb13, bs13 = (size_t)2 * inter * rs13, b2 = (size_t)dim * rb2, bs2 = (size_t)dim * rs2;
    const int N13 = 2 * inter;
#pragma omp parallel
    {
        pin_self();
        int t = omp_get_thread_num(), node = t / g_cores_per_node, tin = t % g_cores_per_node;
        bind_self_node(node);
        // row ranges owned by this node (same split as the compute)
        int a13 = (int)((long)N13 * node / g_nodes), z13 = (int)((long)N13 * (node + 1) / g_nodes);
        int a2 = (int)((long)dim * node / g_nodes), z2 = (int)((long)dim * (node + 1) / g_nodes);
        for (int e = tin; e < E; e += g_cores_per_node) {
            // w13 / s13: rows [a13, z13) of the concatenated [w1; w3]
            for (int r = a13; r < z13;) {
                int seg_end = r < inter ? inter : N13;
                if (seg_end > z13) seg_end = z13;
                const uint8_t* srcw = (r < inter) ? sw1[e] + (size_t)r * rb13 : sw3[e] + (size_t)(r - inter) * rb13;
                const uint8_t* srcs = (r < inter) ? ss1[e] + (size_t)r * rs13 : ss3[e] + (size_t)(r - inter) * rs13;
                memcpy(w13 + (size_t)e * b13 + (size_t)r * rb13, srcw, (size_t)(seg_end - r) * rb13);
                memcpy(s13 + (size_t)e * bs13 + (size_t)r * rs13, srcs, (size_t)(seg_end - r) * rs13);
                r = seg_end;
            }
            memcpy(w2 + (size_t)e * b2 + (size_t)a2 * rb2, sw2[e] + (size_t)a2 * rb2, (size_t)(z2 - a2) * rb2);
            memcpy(s2 + (size_t)e * bs2 + (size_t)a2 * rs2, ss2[e] + (size_t)a2 * rs2, (size_t)(z2 - a2) * rs2);
        }
        unbind_self();
    }
    restore_master();
}

// Same as cpumoe_forward, but the expert matrices are addressed by id from the layer's base pointers
// (the Python side passes one int32 array instead of four pointer arrays).
extern "C" int cpumoe_forward_ids(const uint8_t* w13, const uint8_t* s13, const uint8_t* w2, const uint8_t* s2,
                                  size_t b13, size_t bs13, size_t b2, size_t bs2, const int* ids, int E,
                                  const uint16_t* x, const float* wts, float* out, int K, int inter, int dim, float limit,
                                  float* gu, uint16_t* h) {
    const uint8_t* p13[16]; const uint8_t* ps13[16]; const uint8_t* p2[16]; const uint8_t* ps2[16];
    if (E > 16) return -1;
    for (int j = 0; j < E; j++) { p13[j] = w13 + (size_t)ids[j] * b13; ps13[j] = s13 + (size_t)ids[j] * bs13; p2[j] = w2 + (size_t)ids[j] * b2; ps2[j] = s2 + (size_t)ids[j] * bs2; }
    return cpumoe_forward(p13, ps13, p2, ps2, E, x, wts, out, K, inter, dim, limit, gu, h);
}

// One expert (w1, w3, s1, s3, w2, s2 source rows) into the layer buffers at index e, with the same node
// split as the compute: w1 rows (first half of w13) by node 0, w3 rows by node 1; w2 rows split in half.
extern "C" void cpumoe_load_expert(uint8_t* w13, uint8_t* s13, uint8_t* w2, uint8_t* s2, int e,
                                   const uint8_t* src_w1, const uint8_t* src_w3, const uint8_t* src_s1, const uint8_t* src_s3,
                                   const uint8_t* src_w2, const uint8_t* src_s2, int inter, int dim) {
    if (omp_get_max_threads() != g_threads) { omp_set_dynamic(0); omp_set_num_threads(g_threads); }
    const int N13 = 2 * inter, rb13 = dim / 2, rs13 = dim / 32, rb2 = inter / 2, rs2 = inter / 32;
#pragma omp parallel
    {
        pin_self();
        int t = omp_get_thread_num(), node = t / g_cores_per_node, tin = t % g_cores_per_node;
        bind_self_node(node);
        // w13 / s13 rows [n0, n1) of this node, split among its threads
        int n0 = (int)((long)N13 * node / g_nodes), n1 = (int)((long)N13 * (node + 1) / g_nodes);
        int per = (n1 - n0 + g_cores_per_node - 1) / g_cores_per_node;
        int a = n0 + tin * per, b = a + per < n1 ? a + per : n1;
        for (int n = a; n < b; ++n) {
            const uint8_t* sw = n < inter ? src_w1 + (size_t)n * rb13 : src_w3 + (size_t)(n - inter) * rb13;
            const uint8_t* ss = n < inter ? src_s1 + (size_t)n * rs13 : src_s3 + (size_t)(n - inter) * rs13;
            memcpy(w13 + ((size_t)e * N13 + n) * rb13, sw, rb13);
            memcpy(s13 + ((size_t)e * N13 + n) * rs13, ss, rs13);
        }
        int m0 = (int)((long)dim * node / g_nodes), m1 = (int)((long)dim * (node + 1) / g_nodes);
        per = (m1 - m0 + g_cores_per_node - 1) / g_cores_per_node;
        a = m0 + tin * per; b = a + per < m1 ? a + per : m1;
        if (a < b) {
            memcpy(w2 + ((size_t)e * dim + a) * rb2, src_w2 + (size_t)a * rb2, (size_t)(b - a) * rb2);
            memcpy(s2 + ((size_t)e * dim + a) * rs2, src_s2 + (size_t)a * rs2, (size_t)(b - a) * rs2);
        }
        unbind_self();
    }
    restore_master();
}
