# Matrix Multiplication Benchmark (macOS / Apple Silicon)

This repository is a small, reproducible benchmark harness for comparing different **matrix multiplication** algorithms on macOS (Apple Silicon).

It measures, per algorithm:
- wall-clock time (avg and min over multiple runs)
- theoretical operation counts (adds / non-constant multiplies / constant scalings)
- memory costs (A+B+C bytes + algorithm-specific extra buffers; RSS before/after warm-up)
- checksum (prevents dead-code elimination)
- optional correctness check vs a reference backend (max abs / RMS / max rel error)

## Algorithms implemented

CPU (double precision, row-major):
- `naive` — straightforward triple loop
- `blocked` — cache-friendly tiling
- `gcd_mt` — multithreaded CPU using GCD (`dispatch_apply`)
- `vinograd` — Winograd-style (Vinograd) matrix multiplication
- `strassen` — recursive Strassen algorithm (thresholded, optional padding to power-of-two)
- `alpha48` — 4×4 rank-48 algorithm (AlphaTensor/AlphaEvolve-style), used as a 4×4 block kernel

Optional backends:
- `accelerate` — Apple Accelerate/BLAS (`cblas_dgemm`)
- `metal` — GPU via Metal Performance Shaders (float32 internally, host input/output are double)

### Notes / constraints
- **Layout:** all CPU algorithms operate on row-major `double` buffers.
- `strassen` is naturally defined for square power-of-two sizes; by default we **pad** to the nearest power-of-two of `max(M,N,K)`.
- `alpha48` is naturally defined for 4×4 blocks; by default we **pad** each dimension up to a multiple of 4.
- Padding is done once (cached) and then reused across repeated benchmark runs on the same inputs.

## Build

Requires macOS + clang.

```bash
make naive        # CPU only (no Accelerate, no Metal)
make accel        # + Accelerate/BLAS backend
make metal        # + Metal/MPS backend (also includes Accelerate)
```

Optional knobs:
```bash
make naive FAST_MATH=1   # adds -ffast-math (faster, less precise)
make naive DEBUG=1       # -O0 -g
```

## Run

### Basic single-algorithm run
```bash
./bench_naive --algo blocked --M 2048 --N 2048 --K 2048 --runs 10
```

### Reproducible run (fixed seed)
```bash
./bench_naive --algo vinograd --M 1024 --N 1024 --K 1024 --runs 5 --seed 123
```

### Correctness check (compares to a reference algorithm)
```bash
# Uses Accelerate if available in the binary, otherwise falls back to `blocked`.
./bench_accel --algo strassen --M 512 --N 512 --K 512 --runs 3 --check

# Explicit reference:
./bench_naive --algo alpha48 --M 256 --N 256 --K 256 --runs 3 --check --ref blocked
```

### Run all algorithms for the same matrix sizes
```bash
./bench_accel --algo all --M 1024 --N 1024 --K 1024 --runs 5 --seed 1 --json --save results.jsonl
```

## CLI options

- `--algo <name|all>`
- `--M <int> --N <int> --K <int>`
- `--runs <int>` — number of timed runs (warm-up is always performed once)
- `--threads <int>` — hint for MT algorithms (only used by `gcd_mt`)
- `--seed <uint>` — RNG seed for A/B generation (`0` = time-based)
- `--json` — emit JSON Lines (one line per result)
- `--save <path>` — append JSON/text results to a file

Correctness options:
- `--check` — compute a reference `C_ref` once, then report error metrics for the tested algorithm
- `--ref <algo>` — select reference algo (default: `accelerate` if available, otherwise `blocked`)
- `--eps <double>` — epsilon used in the relative-error denominator (default: `1e-12`)

Algorithm knobs:
- `--strassen-threshold <n>` — base-case size for Strassen (default: 128)

## Output fields

Text mode prints a human-readable summary.

JSON mode emits one line like:
```json
{"algo":"blocked","M":1024,"N":1024,"K":1024,"runs":5,"threads":0,"seed":1,"avg_s":0.123,"min_s":0.118,
 "muls":1073741824,"adds":1073741824,"const_muls":0,
 "classic_flops":2147483648,"gflops_alg":17.4,"gflops_classic":18.2,
 "bytes_abcs":25165824,"extra_bytes":0,"rss_before":...,
 "rss_after":...,"checksum":...,
 "checked":true,"ref_algo":"accelerate","max_abs_err":...,"rms_err":...,"max_rel_err":...,
 "ok":true}
```

Where:
- `muls` = non-constant multiplications
- `adds` = additions/subtractions
- `const_muls` = multiplies/divides by constants (e.g., `/2`, `*2` in Alpha48)
- `classic_flops` = `2*M*N*K` (useful for an “effective GFLOPS” comparison)
- `gflops_alg` = `(muls+adds+const_muls)/time`
- `gflops_classic` = `classic_flops/time`

## File layout

- `bench.c` — benchmark runner + CLI
- `bench.h` — public types (algo registry, config, result struct)
- `algos.c` — baselines + Vinograd + Strassen + registry
- `alpha48.c` — Alpha48 kernel + wrapper
- `metal_mps.mm` — Metal/MPS backend (optional)

## Practical advice for fair benchmarking

- Prefer sizes that avoid padding for algorithm-specific kernels:
  - `alpha48`: use sizes divisible by 4
  - `strassen`: use square power-of-two sizes (e.g., 256, 512, 1024, 2048)
- Run with a fixed `--seed` for reproducibility.
- Use `min_s` as a “best-case” metric and `avg_s` to see typical performance.

