#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <mach/mach_time.h>
#include <mach/mach.h>

#include "bench.h"


double now_seconds(void) {
    static mach_timebase_info_data_t timebase = {0, 0};
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    uint64_t t = mach_absolute_time();
    return (double)t * (double)timebase.numer / (double)timebase.denom / 1e9;
}


uint64_t current_rss_bytes(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                 (task_info_t)&info, &count);
    if (kr != KERN_SUCCESS) return 0;
    return (uint64_t)info.resident_size;
}


void fill_random(double *a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] = (double)rand() / (double)RAND_MAX - 0.5; 
    }
}

void zero_buf(double *a, size_t n) {
    memset(a, 0, n * sizeof(double));
}

static inline double sum_all(const double *c, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += c[i];
    return s;
}

static inline void merge_cfg_defaults(bench_config_t *dst, const bench_config_t *src) {
    bench_config_t d = {
        .runs = 5,
        .threads = 0,
        .seed = 1,
        .check = false,
        .ref_algo = NULL,
        .strassen_threshold = 128,
        .strassen_pad_pow2 = true,
        .alpha48_pad_4 = true,
    };
    if (!src) { *dst = d; return; }

    *dst = *src;
    if (dst->runs <= 0) dst->runs = d.runs;
    if (dst->strassen_threshold <= 0) dst->strassen_threshold = d.strassen_threshold;
}

static const algo_t *find_algo_by_name(const char *name) {
    if (!name) return NULL;
    size_t cnt = 0;
    const algo_t *A = algorithms_list(&cnt);
    for (size_t i = 0; i < cnt; ++i) {
        if (strcmp(A[i].name, name) == 0) return &A[i];
    }
    return NULL;
}

static const algo_t *default_ref_algo(void) {
    const algo_t *acc = find_algo_by_name("accelerate");
    if (acc) return acc;
    const algo_t *blk = find_algo_by_name("blocked");
    if (blk) return blk;
    return find_algo_by_name("naive");
}

static void compute_error_metrics(const double *C, const double *Cref, size_t n,
                                  double *max_abs, double *sum_sq, double *max_rel) {
    const double eps = 1e-12;
    double mabs = 0.0;
    double msq = 0.0;
    double mrel = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double ref = Cref[i];
        double diff = C[i] - ref;
        double ad = fabs(diff);
        if (ad > mabs) mabs = ad;
        msq += diff * diff;
        double denom = fmax(fabs(ref), eps);
        double rel = ad / denom;
        if (rel > mrel) mrel = rel;
    }
    if (max_abs) *max_abs = mabs;
    if (sum_sq) *sum_sq = msq;
    if (max_rel) *max_rel = mrel;
}


bench_result_t run_benchmark(const algo_t *algo, int M, int N, int K, const bench_config_t *cfg_in) {
    bench_result_t R;
    memset(&R, 0, sizeof(R));
    if (!algo || !algo->fn) return R;

    bench_config_t cfg;
    merge_cfg_defaults(&cfg, cfg_in);

    R.algo = algo;
    R.M = M; R.N = N; R.K = K;
    R.cfg = cfg;

    size_t Asz = (size_t)M * (size_t)K;
    size_t Bsz = (size_t)K * (size_t)N;
    size_t Csz = (size_t)M * (size_t)N;

    R.bytes_abcs = (uint64_t)(Asz + Bsz + Csz) * sizeof(double);
    R.classic_flops = 2ULL * (unsigned long long)M * (unsigned long long)N * (unsigned long long)K;

    if (algo->extra_mem_bytes) {
        R.extra_bytes = (uint64_t)algo->extra_mem_bytes(M, N, K, cfg.threads, &cfg);
    }

    if (algo->ops) {
        algo->ops(M, N, K, &R.muls, &R.adds, &R.const_muls, &cfg);
    } else {
        unsigned long long mm = (unsigned long long)M * (unsigned long long)N * (unsigned long long)K;
        R.muls = mm;
        R.adds = mm;
        R.const_muls = 0;
    }

    R.rss_before = current_rss_bytes();

    double *A = (double*)malloc(Asz * sizeof(double));
    double *B = (double*)malloc(Bsz * sizeof(double));
    double *C = (double*)malloc(Csz * sizeof(double));
    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed (%zu, %zu, %zu doubles)\n", Asz, Bsz, Csz);
        free(A); free(B); free(C);
        return R;
    }

    if (cfg.seed == 0) srand((unsigned)time(NULL));
    else srand((unsigned)cfg.seed);

    fill_random(A, Asz);
    fill_random(B, Bsz);

    algo_ctx_t ctx = NULL;
    if (algo->init) {
        ctx = algo->init(M, N, K, cfg.threads, &cfg);
    }

    zero_buf(C, Csz);
    (void)algo->fn(ctx, A, B, C, M, N, K, cfg.threads);

    R.rss_after = current_rss_bytes();

    const algo_t *ref_algo = NULL;
    algo_ctx_t ref_ctx = NULL;
    double *Cref = NULL;

    if (cfg.check) {
        ref_algo = cfg.ref_algo ? find_algo_by_name(cfg.ref_algo) : default_ref_algo();
        if (!ref_algo || !ref_algo->fn) {
            fprintf(stderr, "--check requested but reference algorithm '%s' not found\n",
                    cfg.ref_algo ? cfg.ref_algo : "(auto)");
        } else {
            Cref = (double*)malloc(Csz * sizeof(double));
            if (!Cref) {
                fprintf(stderr, "Failed to allocate reference matrix (%zu doubles)\n", Csz);
            } else {
                if (ref_algo->init) ref_ctx = ref_algo->init(M, N, K, cfg.threads, &cfg);
                zero_buf(Cref, Csz);
                bool ok_ref = ref_algo->fn(ref_ctx, A, B, Cref, M, N, K, cfg.threads);
                if (!ok_ref) {
                    fprintf(stderr, "Reference algorithm '%s' failed; disabling check\n", ref_algo->name);
                    free(Cref); Cref = NULL;
                } else {
                    R.checked = true;
                }
                if (ref_algo->destroy) ref_algo->destroy(ref_ctx);
                ref_ctx = NULL;
            }
        }
    }

    double total = 0.0;
    double tmin = 1e100;
    double checksum_acc = 0.0;

    double worst_max_abs = 0.0;
    double worst_max_rel = 0.0;
    double sum_sq_all = 0.0;
    unsigned long long checked_runs = 0;

    int runs_ok = 0;
    for (int r = 0; r < cfg.runs; ++r) {
        zero_buf(C, Csz);
        double t0 = now_seconds();
        bool ok = algo->fn(ctx, A, B, C, M, N, K, cfg.threads);
        double t1 = now_seconds();

        if (!ok) {
            R.ok = false;
            break;
        }

        double dt = t1 - t0;
        total += dt;
        if (dt < tmin) tmin = dt;
        checksum_acc += sum_all(C, Csz);

        if (Cref) {
            double max_abs = 0.0, sum_sq = 0.0, max_rel = 0.0;
            compute_error_metrics(C, Cref, Csz, &max_abs, &sum_sq, &max_rel);
            if (max_abs > worst_max_abs) worst_max_abs = max_abs;
            if (max_rel > worst_max_rel) worst_max_rel = max_rel;
            sum_sq_all += sum_sq;
            checked_runs++;
        }

        runs_ok++;
        R.ok = true;
    }

    if (algo->destroy) algo->destroy(ctx);
    ctx = NULL;

    R.wall_time_s = (runs_ok > 0) ? total / (double)runs_ok : 0.0;
    R.min_time_s  = (runs_ok > 0) ? tmin : 0.0;
    R.checksum = checksum_acc;

    double ops_alg = (double)R.muls + (double)R.adds + (double)R.const_muls;
    if (R.wall_time_s > 0.0) {
        R.gflops_alg = ops_alg / R.wall_time_s / 1e9;
        R.gflops_classic = (double)R.classic_flops / R.wall_time_s / 1e9;
    }

    if (Cref && checked_runs > 0) {
        R.max_abs_err = worst_max_abs;
        R.max_rel_err = worst_max_rel;
        R.rms_err = sqrt(sum_sq_all / ((double)Csz * (double)checked_runs));
    }

    free(A); free(B); free(C); free(Cref);
    return R;
}


static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [--algo name|all] [--M n] [--N n] [--K n] [--runs r] [--threads t]\n"
        "          [--seed s] [--check] [--ref name] [--eps e] [--strassen-threshold n]\n"
        "          [--json] [--save file]\n\n"
        "Notes:\n"
        "  --seed 0 uses time-based seeding; default is deterministic seed=1.\n"
        "  --check computes a reference result and reports error metrics (slower).\n"
        "  --ref chooses reference algorithm for --check (default: accelerate if available else blocked).\n"
        "  --eps controls the relative-error denominator epsilon (default: 1e-12).\n\n",
        prog);

    size_t cnt = 0;
    const algo_t *A = algorithms_list(&cnt);
    fprintf(stderr, "Algorithms:");
    for (size_t i = 0; i < cnt; ++i) fprintf(stderr, " %s", A[i].name);
    fprintf(stderr, "\n");
}

static void print_result(const bench_result_t *R, bool json, FILE *out) {
    if (json) {
        fprintf(out,
            "{"
            "\"algo\":\"%s\","
            "\"M\":%d,\"N\":%d,\"K\":%d,"
            "\"runs\":%d,\"threads\":%d,\"seed\":%u,"
            "\"avg_s\":%.9f,\"min_s\":%.9f,"
            "\"muls\":%llu,\"adds\":%llu,\"const_muls\":%llu,"
            "\"classic_flops\":%llu,"
            "\"gflops_alg\":%.4f,\"gflops_classic\":%.4f,"
            "\"bytes_abcs\":%llu,\"extra_bytes\":%llu,"
            "\"rss_before\":%llu,\"rss_after\":%llu,"
            "\"checked\":%s,\"max_abs_err\":%.6e,\"rms_err\":%.6e,\"max_rel_err\":%.6e,"
            "\"checksum\":%.6e,\"ok\":%s"
            "}\n",
            R->algo ? R->algo->name : "(null)",
            R->M, R->N, R->K,
            R->cfg.runs, R->cfg.threads, (unsigned)R->cfg.seed,
            R->wall_time_s, R->min_time_s,
            (unsigned long long)R->muls, (unsigned long long)R->adds, (unsigned long long)R->const_muls,
            (unsigned long long)R->classic_flops,
            R->gflops_alg, R->gflops_classic,
            (unsigned long long)R->bytes_abcs, (unsigned long long)R->extra_bytes,
            (unsigned long long)R->rss_before, (unsigned long long)R->rss_after,
            R->checked ? "true" : "false",
            R->max_abs_err, R->rms_err, R->max_rel_err,
            R->checksum,
            R->ok ? "true" : "false");
    } else {
        fprintf(out,
            "Algorithm: %s\n"
            "Size: %dx%d * %dx%d\n"
            "Runs: %d (threads=%d, seed=%u)\n"
            "Time: avg %.6fs (min %.6fs)\n"
            "Ops (theory/run): muls %llu, adds %llu, const_muls %llu\n"
            "GFLOPS: alg %.2f, classic %.2f\n"
            "Memory: A+B+C %llu bytes, extra %llu bytes\n"
            "RSS: before %llu bytes, after %llu bytes\n"
            "Checksum: %.6e\n",
            R->algo ? R->algo->name : "(null)",
            R->M, R->K, R->K, R->N,
            R->cfg.runs, R->cfg.threads, (unsigned)R->cfg.seed,
            R->wall_time_s, R->min_time_s,
            (unsigned long long)R->muls, (unsigned long long)R->adds, (unsigned long long)R->const_muls,
            R->gflops_alg, R->gflops_classic,
            (unsigned long long)R->bytes_abcs, (unsigned long long)R->extra_bytes,
            (unsigned long long)R->rss_before, (unsigned long long)R->rss_after,
            R->checksum);

        if (R->checked) {
            fprintf(out,
                "Check vs reference: max_abs=%.6e, rms=%.6e, max_rel=%.6e\n",
                R->max_abs_err, R->rms_err, R->max_rel_err);
        }
        fprintf(out, "%s\n", R->ok ? "OK" : "FAILED");
    }
}

int main(int argc, char **argv) {
    const char *algo_name = "naive";
    int M = 1024, N = 1024, K = 1024;

    bench_config_t cfg = {
        .runs = 5,
        .threads = 0,
        .seed = 1,
        .check = false,
        .ref_algo = NULL,
        .check_epsilon = 1e-12,
        .strassen_threshold = 128,
        .strassen_pad_pow2 = true,
        .alpha48_pad_4 = true,
    };

    bool json = false;
    const char *save_file = NULL;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--algo") && i+1 < argc) { algo_name = argv[++i]; }
        else if (!strcmp(argv[i], "--M") && i+1 < argc) { M = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--N") && i+1 < argc) { N = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--K") && i+1 < argc) { K = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--runs") && i+1 < argc) { cfg.runs = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--threads") && i+1 < argc) { cfg.threads = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) { cfg.seed = (uint32_t)strtoul(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--check")) { cfg.check = true; }
        else if (!strcmp(argv[i], "--ref") && i+1 < argc) { cfg.ref_algo = argv[++i]; }
        else if (!strcmp(argv[i], "--eps") && i+1 < argc) { cfg.check_epsilon = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--strassen-threshold") && i+1 < argc) { cfg.strassen_threshold = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--json")) { json = true; }
        else if (!strcmp(argv[i], "--save") && i+1 < argc) { save_file = argv[++i]; }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { print_usage(argv[0]); return 0; }
        else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!strcmp(algo_name, "all")) {
        size_t cnt = 0;
        const algo_t *A = algorithms_list(&cnt);
        for (size_t i = 0; i < cnt; ++i) {
            bench_config_t cfg_i = cfg;
            bench_result_t R = run_benchmark(&A[i], M, N, K, &cfg_i);
            if (save_file) {
                FILE *f = fopen(save_file, "a");
                if (f) { print_result(&R, json, f); fclose(f); }
                else { perror("open save file"); }
            }
            print_result(&R, json, stdout);
        }
        return 0;
    }

    const algo_t *algo = find_algo_by_name(algo_name);
    if (!algo) {
        fprintf(stderr, "Unknown algorithm '%s'\n", algo_name);
        print_usage(argv[0]);
        return 2;
    }

    bench_result_t R = run_benchmark(algo, M, N, K, &cfg);

    if (save_file) {
        FILE *f = fopen(save_file, "a");
        if (f) { print_result(&R, json, f); fclose(f); }
        else { perror("open save file"); }
    }

    print_result(&R, json, stdout);
    return R.ok ? 0 : 3;
}
