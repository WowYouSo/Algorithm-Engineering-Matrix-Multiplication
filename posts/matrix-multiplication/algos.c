#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <dispatch/dispatch.h>

#include "bench.h"

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif


static inline void ops_gemm_classic(int M, int N, int K,
                                   unsigned long long *muls,
                                   unsigned long long *adds,
                                   unsigned long long *const_muls,
                                   const bench_config_t *cfg) {
    (void)cfg;
    unsigned long long mm = (unsigned long long)M * (unsigned long long)N * (unsigned long long)K;
    if (muls) *muls = mm;
    if (adds) *adds = mm;
    if (const_muls) *const_muls = 0;
}

static inline size_t extra_none(int M, int N, int K, int threads, const bench_config_t *cfg) {
    (void)M; (void)N; (void)K; (void)threads; (void)cfg;
    return 0;
}

static bool alg_naive(algo_ctx_t ctx, const double *A, const double *B, double *C,
                      int M, int N, int K, int threads) {
    (void)ctx; (void)threads;
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            double aik = A[i*(size_t)K + k];
            const double *brow = &B[k*(size_t)N];
            double *crow = &C[i*(size_t)N];
            for (int j = 0; j < N; ++j) {
                crow[j] += aik * brow[j];
            }
        }
    }
    return true;
}

static bool alg_blocked(algo_ctx_t ctx, const double *A, const double *B, double *C,
                        int M, int N, int K, int threads) {
    (void)ctx; (void)threads;
    const int BS = 64;
    for (int kk = 0; kk < K; kk += BS) {
        int kkmax = kk + BS; if (kkmax > K) kkmax = K;
        for (int ii = 0; ii < M; ii += BS) {
            int iimax = ii + BS; if (iimax > M) iimax = M;
            for (int jj = 0; jj < N; jj += BS) {
                int jjmax = jj + BS; if (jjmax > N) jjmax = N;
                for (int i = ii; i < iimax; ++i) {
                    for (int k = kk; k < kkmax; ++k) {
                        double aik = A[i*(size_t)K + k];
                        const double *brow = &B[k*(size_t)N + jj];
                        double *crow = &C[i*(size_t)N + jj];
                        for (int j = 0; j < (jjmax - jj); ++j) {
                            crow[j] += aik * brow[j];
                        }
                    }
                }
            }
        }
    }
    return true;
}

static bool alg_gcd_mt(algo_ctx_t ctx, const double *A, const double *B, double *C,
                       int M, int N, int K, int threads) {
    (void)ctx;
    const int BS = 64;
    int tile = 128;
    if (M < 512) tile = 32;

    int tasks = (M + tile - 1) / tile;
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

    dispatch_semaphore_t sem = NULL;
    if (threads > 0) {
        sem = dispatch_semaphore_create(threads);
    }

    dispatch_apply(tasks, q, ^(size_t t){
        if (sem) dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

        int i0 = (int)(t * (size_t)tile);
        int i1 = i0 + tile; if (i1 > M) i1 = M;

        for (int kk = 0; kk < K; kk += BS) {
            int kkmax = kk + BS; if (kkmax > K) kkmax = K;
            for (int jj = 0; jj < N; jj += BS) {
                int jjmax = jj + BS; if (jjmax > N) jjmax = N;
                for (int i = i0; i < i1; ++i) {
                    for (int k = kk; k < kkmax; ++k) {
                        double aik = A[i*(size_t)K + k];
                        const double *brow = &B[k*(size_t)N + jj];
                        double *crow = &C[i*(size_t)N + jj];
                        for (int j = 0; j < (jjmax - jj); ++j) {
                            crow[j] += aik * brow[j];
                        }
                    }
                }
            }
        }

        if (sem) dispatch_semaphore_signal(sem);
    });

    return true;
}

#ifdef USE_ACCELERATE
static bool alg_accelerate(algo_ctx_t ctx, const double *A, const double *B, double *C,
                           int M, int N, int K, int threads) {
    (void)ctx; (void)threads;
    const double alpha = 1.0;
    const double beta  = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
    return true;
}
#else
static bool alg_accelerate(algo_ctx_t ctx, const double *A, const double *B, double *C,
                           int M, int N, int K, int threads) {
    (void)ctx; (void)A; (void)B; (void)C; (void)M; (void)N; (void)K; (void)threads;
    return false;
}
#endif

#ifndef SKIP_METAL
__attribute__((weak)) bool metal_gemm_f32(const float *A, const float *B, float *C, int M, int N, int K);

static bool alg_metal(algo_ctx_t ctx, const double *A, const double *B, double *C,
                      int M, int N, int K, int threads) {
    (void)ctx; (void)threads;
    if (!metal_gemm_f32) return false;

    size_t Asz = (size_t)M * (size_t)K;
    size_t Bsz = (size_t)K * (size_t)N;
    size_t Csz = (size_t)M * (size_t)N;

    float *Af = (float*)malloc(Asz * sizeof(float));
    float *Bf = (float*)malloc(Bsz * sizeof(float));
    float *Cf = (float*)calloc(Csz, sizeof(float));
    if (!Af || !Bf || !Cf) { free(Af); free(Bf); free(Cf); return false; }

    for (size_t i = 0; i < Asz; ++i) Af[i] = (float)A[i];
    for (size_t i = 0; i < Bsz; ++i) Bf[i] = (float)B[i];

    bool ok = metal_gemm_f32(Af, Bf, Cf, M, N, K);
    if (ok) {
        for (size_t i = 0; i < Csz; ++i) C[i] = (double)Cf[i];
    }

    free(Af); free(Bf); free(Cf);
    return ok;
}

static size_t extra_metal(int M, int N, int K, int threads, const bench_config_t *cfg) {
    (void)threads; (void)cfg;
    return ((size_t)M*(size_t)K + (size_t)K*(size_t)N + (size_t)M*(size_t)N) * sizeof(float);
}
#else
static bool alg_metal(algo_ctx_t ctx, const double *A, const double *B, double *C,
                      int M, int N, int K, int threads) {
    (void)ctx; (void)A; (void)B; (void)C; (void)M; (void)N; (void)K; (void)threads;
    return false;
}
static size_t extra_metal(int M, int N, int K, int threads, const bench_config_t *cfg) {
    (void)M; (void)N; (void)K; (void)threads; (void)cfg;
    return 0;
}
#endif



typedef struct {
    int M, N, K;
    int half;
    double *rowFactor;
    double *colFactor;
} vinograd_ctx_t;

static algo_ctx_t vinograd_init(int M, int N, int K, int threads, const bench_config_t *cfg) {
    (void)threads; (void)cfg;
    vinograd_ctx_t *x = (vinograd_ctx_t*)calloc(1, sizeof(vinograd_ctx_t));
    if (!x) return NULL;
    x->M = M; x->N = N; x->K = K;
    x->half = K / 2;
    x->rowFactor = (double*)malloc((size_t)M * sizeof(double));
    x->colFactor = (double*)malloc((size_t)N * sizeof(double));
    if (!x->rowFactor || !x->colFactor) {
        free(x->rowFactor); free(x->colFactor); free(x);
        return NULL;
    }
    return (algo_ctx_t)x;
}

static void vinograd_destroy(algo_ctx_t ctx) {
    vinograd_ctx_t *x = (vinograd_ctx_t*)ctx;
    if (!x) return;
    free(x->rowFactor);
    free(x->colFactor);
    free(x);
}

static bool alg_vinograd(algo_ctx_t ctx, const double *A, const double *B, double *C,
                         int M, int N, int K, int threads) {
    (void)threads;
    vinograd_ctx_t *x = (vinograd_ctx_t*)ctx;
    if (!x || x->M != M || x->N != N || x->K != K) return false;

    const int half = x->half;

    for (int i = 0; i < M; ++i) {
        double s = 0.0;
        const double *arow = &A[i*(size_t)K];
        for (int p = 0; p < half; ++p) {
            s += arow[2*p] * arow[2*p + 1];
        }
        x->rowFactor[i] = s;
    }

    for (int j = 0; j < N; ++j) {
        double s = 0.0;
        for (int p = 0; p < half; ++p) {
            const double b0 = B[(2*p)*(size_t)N + j];
            const double b1 = B[(2*p + 1)*(size_t)N + j];
            s += b0 * b1;
        }
        x->colFactor[j] = s;
    }

    const int rem = K & 1;

    for (int i = 0; i < M; ++i) {
        const double rf = x->rowFactor[i];
        const double *arow = &A[i*(size_t)K];
        double *crow = &C[i*(size_t)N];

        for (int j = 0; j < N; ++j) {
            crow[j] = -rf - x->colFactor[j];
        }

        for (int p = 0; p < half; ++p) {
            const double a0 = arow[2*p];
            const double a1 = arow[2*p + 1];
            const double *brow0 = &B[(2*p)*(size_t)N];
            const double *brow1 = &B[(2*p + 1)*(size_t)N];
            for (int j = 0; j < N; ++j) {
                crow[j] += (a0 + brow1[j]) * (a1 + brow0[j]);
            }
        }

        if (rem) {
            const double a_last = arow[K - 1];
            const double *brow_last = &B[(K - 1)*(size_t)N];
            for (int j = 0; j < N; ++j) {
                crow[j] += a_last * brow_last[j];
            }
        }
    }

    return true;
}

static void ops_vinograd(int M, int N, int K,
                         unsigned long long *muls,
                         unsigned long long *adds,
                         unsigned long long *const_muls,
                         const bench_config_t *cfg) {
    (void)cfg;
    const int half = K / 2;
    const int rem = K & 1;


    unsigned long long mul = 0;
    mul += (unsigned long long)M * (unsigned long long)half;
    mul += (unsigned long long)N * (unsigned long long)half;
    mul += (unsigned long long)M * (unsigned long long)N * (unsigned long long)half;
    mul += (unsigned long long)M * (unsigned long long)N * (unsigned long long)rem;


    unsigned long long add = 0;
    add += (unsigned long long)M * (unsigned long long)half;
    add += (unsigned long long)N * (unsigned long long)half;
    add += 2ULL * (unsigned long long)M * (unsigned long long)N;
    add += 3ULL * (unsigned long long)M * (unsigned long long)N * (unsigned long long)half;
    add += (unsigned long long)M * (unsigned long long)N * (unsigned long long)rem;

    if (muls) *muls = mul;
    if (adds) *adds = add;
    if (const_muls) *const_muls = 0;
}

static size_t extra_vinograd(int M, int N, int K, int threads, const bench_config_t *cfg) {
    (void)K; (void)threads; (void)cfg;
    return ((size_t)M + (size_t)N) * sizeof(double);
}



typedef struct {
    int M, N, K;
    int n;
    int threshold;
    bool use_pad;
    bool cached_pads;

    double *Ap;
    double *Bp;
    double *Cp;

    double *ws;
    size_t ws_doubles;
} strassen_ctx_t;

typedef struct {
    double *base;
    size_t cap; 
    size_t off;
} ws_alloc_t;

static inline size_t next_pow2_u32(size_t x) {
    if (x <= 1) return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
#if SIZE_MAX > 0xffffffffu
    x |= x >> 32;
#endif
    return x + 1;
}

static inline bool is_pow2_u32(size_t x) {
    return x && ((x & (x - 1)) == 0);
}

static size_t strassen_ws_doubles(int n, int threshold) {
    size_t ws = 0;
    while (n > threshold) {
        int n2 = n / 2;
        ws += 9ULL * (size_t)n2 * (size_t)n2;
        n = n2;
    }
    return ws;
}

static inline double *ws_alloc(ws_alloc_t *ws, size_t count) {
    if (!ws || ws->off + count > ws->cap) return NULL;
    double *p = ws->base + ws->off;
    ws->off += count;
    return p;
}

static inline void ws_mark(ws_alloc_t *ws, size_t *mark) {
    if (mark) *mark = ws ? ws->off : 0;
}

static inline void ws_reset(ws_alloc_t *ws, size_t mark) {
    if (ws) ws->off = mark;
}

static void mat_add(const double *A, int sa, const double *B, int sb, double *C, int sc, int n) {
    for (int i = 0; i < n; ++i) {
        const double *ar = &A[(size_t)i * (size_t)sa];
        const double *br = &B[(size_t)i * (size_t)sb];
        double *cr = &C[(size_t)i * (size_t)sc];
        for (int j = 0; j < n; ++j) cr[j] = ar[j] + br[j];
    }
}

static void mat_sub(const double *A, int sa, const double *B, int sb, double *C, int sc, int n) {
    for (int i = 0; i < n; ++i) {
        const double *ar = &A[(size_t)i * (size_t)sa];
        const double *br = &B[(size_t)i * (size_t)sb];
        double *cr = &C[(size_t)i * (size_t)sc];
        for (int j = 0; j < n; ++j) cr[j] = ar[j] - br[j];
    }
}

static void mat_copy(const double *A, int sa, double *C, int sc, int n) {
    for (int i = 0; i < n; ++i) {
        memcpy(&C[(size_t)i * (size_t)sc], &A[(size_t)i * (size_t)sa], (size_t)n * sizeof(double));
    }
}

static void mat_zero(double *C, int sc, int n) {
    for (int i = 0; i < n; ++i) {
        memset(&C[(size_t)i * (size_t)sc], 0, (size_t)n * sizeof(double));
    }
}

static void classic_gemm_nn(const double *A, int sa, const double *B, int sb, double *C, int sc, int n) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            const double aik = A[(size_t)i * (size_t)sa + (size_t)k];
            const double *brow = &B[(size_t)k * (size_t)sb];
            double *crow = &C[(size_t)i * (size_t)sc];
            for (int j = 0; j < n; ++j) {
                crow[j] += aik * brow[j];
            }
        }
    }
}

static void strassen_rec(const double *A, int sa, const double *B, int sb, double *C, int sc,
                         int n, int threshold, ws_alloc_t *ws) {
    if (n <= threshold) {
        classic_gemm_nn(A, sa, B, sb, C, sc, n);
        return;
    }

    const int n2 = n / 2;
    const size_t block = (size_t)n2 * (size_t)n2;

    size_t mark = 0;
    ws_mark(ws, &mark);

    double *S1 = ws_alloc(ws, block);
    double *S2 = ws_alloc(ws, block);

    double *P1 = ws_alloc(ws, block);
    double *P2 = ws_alloc(ws, block);
    double *P3 = ws_alloc(ws, block);
    double *P4 = ws_alloc(ws, block);
    double *P5 = ws_alloc(ws, block);
    double *P6 = ws_alloc(ws, block);
    double *P7 = ws_alloc(ws, block);

    if (!S1 || !S2 || !P1 || !P2 || !P3 || !P4 || !P5 || !P6 || !P7) {
        ws_reset(ws, mark);
        classic_gemm_nn(A, sa, B, sb, C, sc, n);
        return;
    }

    const double *A11 = A;
    const double *A12 = A + n2;
    const double *A21 = A + (size_t)n2 * (size_t)sa;
    const double *A22 = A21 + n2;

    const double *B11 = B;
    const double *B12 = B + n2;
    const double *B21 = B + (size_t)n2 * (size_t)sb;
    const double *B22 = B21 + n2;

    double *C11 = C;
    double *C12 = C + n2;
    double *C21 = C + (size_t)n2 * (size_t)sc;
    double *C22 = C21 + n2;

    mat_add(A11, sa, A22, sa, S1, n2, n2);
    mat_add(B11, sb, B22, sb, S2, n2, n2);
    mat_zero(P1, n2, n2);
    strassen_rec(S1, n2, S2, n2, P1, n2, n2, threshold, ws);

    mat_add(A21, sa, A22, sa, S1, n2, n2);
    mat_zero(P2, n2, n2);
    strassen_rec(S1, n2, B11, sb, P2, n2, n2, threshold, ws);

    mat_sub(B12, sb, B22, sb, S2, n2, n2);
    mat_zero(P3, n2, n2);
    strassen_rec(A11, sa, S2, n2, P3, n2, n2, threshold, ws);

    mat_sub(B21, sb, B11, sb, S2, n2, n2);
    mat_zero(P4, n2, n2);
    strassen_rec(A22, sa, S2, n2, P4, n2, n2, threshold, ws);

    mat_add(A11, sa, A12, sa, S1, n2, n2);
    mat_zero(P5, n2, n2);
    strassen_rec(S1, n2, B22, sb, P5, n2, n2, threshold, ws);

    mat_sub(A21, sa, A11, sa, S1, n2, n2);
    mat_add(B11, sb, B12, sb, S2, n2, n2);
    mat_zero(P6, n2, n2);
    strassen_rec(S1, n2, S2, n2, P6, n2, n2, threshold, ws);

    mat_sub(A12, sa, A22, sa, S1, n2, n2);
    mat_add(B21, sb, B22, sb, S2, n2, n2);
    mat_zero(P7, n2, n2);
    strassen_rec(S1, n2, S2, n2, P7, n2, n2, threshold, ws);

    for (int i = 0; i < n2; ++i) {
        double *c11 = &C11[(size_t)i * (size_t)sc];
        double *c12 = &C12[(size_t)i * (size_t)sc];
        double *c21 = &C21[(size_t)i * (size_t)sc];
        double *c22 = &C22[(size_t)i * (size_t)sc];

        const double *p1 = &P1[(size_t)i * (size_t)n2];
        const double *p2 = &P2[(size_t)i * (size_t)n2];
        const double *p3 = &P3[(size_t)i * (size_t)n2];
        const double *p4 = &P4[(size_t)i * (size_t)n2];
        const double *p5 = &P5[(size_t)i * (size_t)n2];
        const double *p6 = &P6[(size_t)i * (size_t)n2];
        const double *p7 = &P7[(size_t)i * (size_t)n2];

        for (int j = 0; j < n2; ++j) {
            c11[j] = p1[j] + p4[j] - p5[j] + p7[j];
            c12[j] = p3[j] + p5[j];
            c21[j] = p2[j] + p4[j];
            c22[j] = p1[j] - p2[j] + p3[j] + p6[j];
        }
    }

    ws_reset(ws, mark);
}

static algo_ctx_t strassen_init(int M, int N, int K, int threads, const bench_config_t *cfg) {
    (void)threads;
    strassen_ctx_t *x = (strassen_ctx_t*)calloc(1, sizeof(strassen_ctx_t));
    if (!x) return NULL;

    x->M = M; x->N = N; x->K = K;
    x->threshold = (cfg && cfg->strassen_threshold > 0) ? cfg->strassen_threshold : 128;

    size_t need = (size_t)M;
    if ((size_t)N > need) need = (size_t)N;
    if ((size_t)K > need) need = (size_t)K;

    bool want_pad = (cfg ? cfg->strassen_pad_pow2 : true);

    if (M == N && N == K && is_pow2_u32((size_t)M)) {
        x->n = M;
        x->use_pad = false;
    } else if (want_pad) {
        x->n = (int)next_pow2_u32(need);
        x->use_pad = true;
    } else {
        free(x);
        return NULL;
    }

    x->ws_doubles = strassen_ws_doubles(x->n, x->threshold);
    x->ws = (double*)malloc(x->ws_doubles * sizeof(double));
    if (!x->ws) { free(x); return NULL; }

    if (x->use_pad) {
        size_t nn = (size_t)x->n * (size_t)x->n;
        x->Ap = (double*)calloc(nn, sizeof(double));
        x->Bp = (double*)calloc(nn, sizeof(double));
        x->Cp = (double*)malloc(nn * sizeof(double));
        if (!x->Ap || !x->Bp || !x->Cp) {
            free(x->Ap); free(x->Bp); free(x->Cp); free(x->ws); free(x);
            return NULL;
        }
        x->cached_pads = false;
    }

    return (algo_ctx_t)x;
}

static void strassen_destroy(algo_ctx_t ctx) {
    strassen_ctx_t *x = (strassen_ctx_t*)ctx;
    if (!x) return;
    free(x->Ap);
    free(x->Bp);
    free(x->Cp);
    free(x->ws);
    free(x);
}

static bool alg_strassen(algo_ctx_t ctx, const double *A, const double *B, double *C,
                         int M, int N, int K, int threads) {
    (void)threads;
    strassen_ctx_t *x = (strassen_ctx_t*)ctx;
    if (!x) return false;
    if (x->M != M || x->N != N || x->K != K) return false;

    ws_alloc_t ws = { .base = x->ws, .cap = x->ws_doubles, .off = 0 };

    if (!x->use_pad) {
        if (!(M == N && N == K)) return false;
        strassen_rec(A, K, B, N, C, N, x->n, x->threshold, &ws);
        return true;
    }

    const int n = x->n;
    const size_t nn = (size_t)n * (size_t)n;

    if (!x->cached_pads) {
        memset(x->Ap, 0, nn * sizeof(double));
        memset(x->Bp, 0, nn * sizeof(double));

        for (int i = 0; i < M; ++i) {
            memcpy(&x->Ap[(size_t)i * (size_t)n], &A[(size_t)i * (size_t)K], (size_t)K * sizeof(double));
        }
        for (int i = 0; i < K; ++i) {
            memcpy(&x->Bp[(size_t)i * (size_t)n], &B[(size_t)i * (size_t)N], (size_t)N * sizeof(double));
        }
        x->cached_pads = true;
    }

    mat_zero(x->Cp, n, n);
    strassen_rec(x->Ap, n, x->Bp, n, x->Cp, n, n, x->threshold, &ws);

    for (int i = 0; i < M; ++i) {
        memcpy(&C[(size_t)i * (size_t)N], &x->Cp[(size_t)i * (size_t)n], (size_t)N * sizeof(double));
    }

    return true;
}

static void ops_strassen_rec(int n, int threshold,
                            unsigned long long *muls,
                            unsigned long long *adds) {
    if (n <= threshold) {
        unsigned long long nn = (unsigned long long)n;
        unsigned long long mm = nn * nn * nn;
        *muls = mm;
        *adds = mm;
        return;
    }

    unsigned long long m2 = 0, a2 = 0;
    ops_strassen_rec(n/2, threshold, &m2, &a2);

    *muls = 7ULL * m2;

    unsigned long long n2 = (unsigned long long)(n/2);
    unsigned long long add_level = 18ULL * n2 * n2;
    *adds = 7ULL * a2 + add_level;
}

static void ops_strassen(int M, int N, int K,
                         unsigned long long *muls,
                         unsigned long long *adds,
                         unsigned long long *const_muls,
                         const bench_config_t *cfg) {
    int threshold = (cfg && cfg->strassen_threshold > 0) ? cfg->strassen_threshold : 128;
    bool want_pad = (cfg ? cfg->strassen_pad_pow2 : true);

    size_t need = (size_t)M;
    if ((size_t)N > need) need = (size_t)N;
    if ((size_t)K > need) need = (size_t)K;

    int n = 0;
    if (M == N && N == K && is_pow2_u32((size_t)M)) {
        n = M;
    } else if (want_pad) {
        n = (int)next_pow2_u32(need);
    } else {
        if (muls) *muls = 0;
        if (adds) *adds = 0;
        if (const_muls) *const_muls = 0;
        return;
    }

    unsigned long long m = 0, a = 0;
    ops_strassen_rec(n, threshold, &m, &a);

    if (muls) *muls = m;
    if (adds) *adds = a;
    if (const_muls) *const_muls = 0;
}

static size_t extra_strassen(int M, int N, int K, int threads, const bench_config_t *cfg) {
    (void)threads;
    int threshold = (cfg && cfg->strassen_threshold > 0) ? cfg->strassen_threshold : 128;
    bool want_pad = (cfg ? cfg->strassen_pad_pow2 : true);

    size_t need = (size_t)M;
    if ((size_t)N > need) need = (size_t)N;
    if ((size_t)K > need) need = (size_t)K;

    int n = 0;
    bool use_pad = false;
    if (M == N && N == K && is_pow2_u32((size_t)M)) {
        n = M;
        use_pad = false;
    } else if (want_pad) {
        n = (int)next_pow2_u32(need);
        use_pad = true;
    } else {
        return 0;
    }

    size_t ws = strassen_ws_doubles(n, threshold) * sizeof(double);

    if (!use_pad) {
        return ws;
    }

    size_t nn = (size_t)n * (size_t)n * sizeof(double);
    return 3ULL * nn + ws;
}



algo_ctx_t alpha48_init(int M, int N, int K, int threads, const bench_config_t *cfg);
void alpha48_destroy(algo_ctx_t ctx);
bool alpha48_run(algo_ctx_t ctx, const double *A, const double *B, double *C,
                 int M, int N, int K, int threads);
void alpha48_ops(int M, int N, int K,
                 unsigned long long *muls,
                 unsigned long long *adds,
                 unsigned long long *const_muls,
                 const bench_config_t *cfg);
size_t alpha48_extra_mem(int M, int N, int K, int threads, const bench_config_t *cfg);



static const algo_t ALGOS_ALL[] = {
    {"naive",      ALG_NAIVE,      NULL,            NULL,            alg_naive,      ops_gemm_classic, extra_none},
    {"blocked",    ALG_BLOCKED,    NULL,            NULL,            alg_blocked,    ops_gemm_classic, extra_none},
    {"gcd_mt",     ALG_GCD_MT,     NULL,            NULL,            alg_gcd_mt,     ops_gemm_classic, extra_none},
    {"accelerate", ALG_ACCELERATE, NULL,            NULL,            alg_accelerate, ops_gemm_classic, extra_none},
    {"metal",      ALG_METAL,      NULL,            NULL,            alg_metal,      ops_gemm_classic, extra_metal},

    {"vinograd",   ALG_VINOGRAD,   vinograd_init,   vinograd_destroy,alg_vinograd,   ops_vinograd,     extra_vinograd},
    {"strassen",   ALG_STRASSEN,   strassen_init,   strassen_destroy,alg_strassen,   ops_strassen,     extra_strassen},
    {"alpha48",    ALG_ALPHA48,    alpha48_init,    alpha48_destroy, alpha48_run,    alpha48_ops,      alpha48_extra_mem},
};

const algo_t *algorithms_list(size_t *count) {
    if (count) *count = sizeof(ALGOS_ALL) / sizeof(ALGOS_ALL[0]);
    return ALGOS_ALL;
}
