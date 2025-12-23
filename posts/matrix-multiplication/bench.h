#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif



typedef struct {
    int runs;
    int threads;
    uint32_t seed;

    bool check;
    const char *ref_algo;
    double check_epsilon;

    int strassen_threshold;
    bool strassen_pad_pow2;

    bool alpha48_pad_4;
} bench_config_t;


typedef enum {
    ALG_NAIVE = 0,
    ALG_BLOCKED = 1,
    ALG_GCD_MT = 2,
    ALG_ACCELERATE = 3,
    ALG_METAL = 4,

    ALG_VINOGRAD = 10,
    ALG_STRASSEN = 11,
    ALG_ALPHA48 = 12
} algo_kind_t;

typedef void *algo_ctx_t;

typedef algo_ctx_t (*algo_init_fn)(int M, int N, int K, int threads, const bench_config_t *cfg);

typedef void (*algo_destroy_fn)(algo_ctx_t ctx);

typedef bool (*algo_run_fn)(algo_ctx_t ctx,
                            const double *A,
                            const double *B,
                            double *C,
                            int M, int N, int K,
                            int threads);



typedef void (*algo_ops_fn)(int M, int N, int K,
                            unsigned long long *muls,
                            unsigned long long *adds,
                            unsigned long long *const_muls,
                            const bench_config_t *cfg);

typedef size_t (*algo_extra_mem_fn)(int M, int N, int K, int threads, const bench_config_t *cfg);

typedef struct {
    const char *name;
    algo_kind_t kind;

    algo_init_fn init;
    algo_destroy_fn destroy;

    algo_run_fn fn;

    algo_ops_fn ops;

    algo_extra_mem_fn extra_mem_bytes;
} algo_t;


double now_seconds(void);


uint64_t current_rss_bytes(void);

void fill_random(double *a, size_t n);
void zero_buf(double *a, size_t n);



typedef struct {
    const algo_t *algo;
    int M, N, K;
    bench_config_t cfg;

    double wall_time_s;
    double min_time_s;

    unsigned long long muls;
    unsigned long long adds;
    unsigned long long const_muls;

    unsigned long long classic_flops;

    uint64_t bytes_abcs;
    uint64_t extra_bytes;
    uint64_t rss_before;
    uint64_t rss_after; 

    double gflops_alg;
    double gflops_classic;

    bool checked;
    double max_abs_err;
    double rms_err;
    double max_rel_err;

    double checksum;

    bool ok;
} bench_result_t;

bench_result_t run_benchmark(const algo_t *algo, int M, int N, int K, const bench_config_t *cfg);

const algo_t *algorithms_list(size_t *count);

bool metal_gemm_f32(const float *A, const float *B, float *C, int M, int N, int K);

#ifdef __cplusplus
}
#endif
