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
    madvise(p, bytes, MADV_HUGEPAGE);
    return p;
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
        if (a < b) memcpy(dst + (size_t)a * row_bytes, src + (size_t)a * row_bytes, (size_t)(b - a) * row_bytes);
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
static void quant_x_u8(const uint16_t* x, int K, uint8_t* xu, float* sx) {
    for (int b = 0; b < K; b += 32) {
        float amax = 0.f, v[32];
        for (int j = 0; j < 32; ++j) { v[j] = bf16_to_f(x[b + j]); float a = fabsf(v[j]); if (a > amax) amax = a; }
        float sc = amax > 0 ? amax / 127.f : 1.f;
        float inv = 1.f / sc;
        for (int j = 0; j < 32; ++j) { int q = (int)rintf(v[j] * inv); if (q > 127) q = 127; if (q < -127) q = -127; xu[b + j] = (uint8_t)(q + 128); }
        sx[b / 32] = sc * 0.5f;  // weights are stored as 2*value
    }
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
static void swiglu_quant_row(const float* gu, int inter, float wt, float limit, uint16_t* h) {
    float tmp[32];
    for (int b = 0; b < inter; b += 32) {
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
static int g_int8 = 1;
extern "C" void cpumoe_set_int8(int on) { g_int8 = on; }

extern "C" int cpumoe_forward(const uint8_t* const* w13, const uint8_t* const* s13, const uint8_t* const* w2, const uint8_t* const* s2,
                              int E, const uint16_t* x, const float* wts, float* out, int K, int inter, int dim, float limit,
                              float* gu, uint16_t* h) {
    static int lut_ready = 0;
    if (!lut_ready) { init_lut(); init_lut_s8(); lut_ready = 1; }
    omp_set_dynamic(0); omp_set_num_threads(g_threads);  // torch resets the shared OpenMP runtime's thread count
    const int N13 = 2 * inter;
    static uint8_t xu[8192]; static float sx[256];            // quantized x
    static uint8_t hu[16 * 4096]; static float sh[16 * 128];  // quantized h per expert
    const int dbg = getenv("DSV41_CPU_DEBUG") != nullptr;
    double t0 = dbg ? omp_get_wtime() : 0, t1 = 0, t2 = 0, t3 = 0;
    if (g_int8) quant_x_u8(x, K, xu, sx);
    if (dbg) t1 = omp_get_wtime();
    // stage 1: gu[e][n] = x . w13[e][n]
#pragma omp parallel
    {
        pin_self();
        int t = omp_get_thread_num(), node = t / g_cores_per_node, tin = t % g_cores_per_node;
        int n0 = (int)((long)N13 * node / g_nodes), n1 = (int)((long)N13 * (node + 1) / g_nodes);
        int per = (n1 - n0 + g_cores_per_node - 1) / g_cores_per_node;
        int a = n0 + tin * per, b = a + per < n1 ? a + per : n1;
        for (int e = 0; e < E; ++e) {
            int n = a;
            if (g_int8)
                for (; n + 1 < b; n += 2)
                    dot_row_vnni2(w13[e] + (size_t)n * (K / 2), s13[e] + (size_t)n * (K / 32), w13[e] + (size_t)(n + 1) * (K / 2), s13[e] + (size_t)(n + 1) * (K / 32),
                                  xu, sx, K, &gu[(size_t)e * N13 + n], &gu[(size_t)e * N13 + n + 1]);
            for (; n < b; ++n)
                gu[(size_t)e * N13 + n] = g_int8
                    ? dot_row_vnni(w13[e] + (size_t)n * (K / 2), s13[e] + (size_t)n * (K / 32), xu, sx, K)
                    : dot_row(w13[e] + (size_t)n * (K / 2), s13[e] + (size_t)n * (K / 32), x, K);
        }
    }
    if (dbg) t2 = omp_get_wtime();
    // stage 2: h[e] = fp8(bf16(wt * silu(gate) * up)) (+ int8 quantization for the w2 GEMV)
#pragma omp parallel for
    for (int e = 0; e < E; ++e) {
        pin_self();
        swiglu_quant_row(gu + (size_t)e * N13, inter, wts[e], limit, h + (size_t)e * inter);
        if (g_int8) quant_x_u8(h + (size_t)e * inter, inter, hu + (size_t)e * inter, sh + (size_t)e * (inter / 32));
    }
    if (dbg) t3 = omp_get_wtime();
    // stage 3: out[n] = sum_e h[e] . w2[e][n]
#pragma omp parallel
    {
        pin_self();
        int t = omp_get_thread_num(), node = t / g_cores_per_node, tin = t % g_cores_per_node;
        int n0 = (int)((long)dim * node / g_nodes), n1 = (int)((long)dim * (node + 1) / g_nodes);
        int per = (n1 - n0 + g_cores_per_node - 1) / g_cores_per_node;
        int a = n0 + tin * per, b = a + per < n1 ? a + per : n1;
        int n = a;
        if (g_int8)
            for (; n + 1 < b; n += 2) {
                float acc0 = 0.f, acc1 = 0.f, r0, r1;
                for (int e = 0; e < E; ++e) {
                    dot_row_vnni2(w2[e] + (size_t)n * (inter / 2), s2[e] + (size_t)n * (inter / 32), w2[e] + (size_t)(n + 1) * (inter / 2), s2[e] + (size_t)(n + 1) * (inter / 32),
                                  hu + (size_t)e * inter, sh + (size_t)e * (inter / 32), inter, &r0, &r1);
                    acc0 += r0; acc1 += r1;
                }
                out[n] = acc0; out[n + 1] = acc1;
            }
        for (; n < b; ++n) {
            float acc = 0.f;
            for (int e = 0; e < E; ++e)
                acc += g_int8
                    ? dot_row_vnni(w2[e] + (size_t)n * (inter / 2), s2[e] + (size_t)n * (inter / 32), hu + (size_t)e * inter, sh + (size_t)e * (inter / 32), inter)
                    : dot_row(w2[e] + (size_t)n * (inter / 2), s2[e] + (size_t)n * (inter / 32), h + (size_t)e * inter, inter);
            out[n] = acc;
        }
    }
    if (dbg) fprintf(stderr, "[cpumoe] E=%d quant %.0fus stage1 %.0fus stage2 %.0fus stage3 %.0fus\n", E, (t1 - t0) * 1e6, (t2 - t1) * 1e6, (t3 - t2) * 1e6, (omp_get_wtime() - t3) * 1e6);
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
    }
    restore_master();
}
