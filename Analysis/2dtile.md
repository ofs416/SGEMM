# Recap / Motivation
- Where `sgemm_tile_unroll` stands: 48.8 GFLOP/s at n=256, falling to ~36 GFLOP/s at n≥2048.
- Why it falls off: for every K tile (`kt`), the whole of C (M×N) is swept once → L/KC full passes over C.
  - At n=4096, C is 64 MB, far beyond L3, so every pass streams C from DRAM.
- Also note: B tile = KC·N·4 bytes = 512 KB at n=4096 → already over the ~256 KB target from the 1D version.
- Idea: tile across J as well (loop order `kt` → `jt` → `i`), so a KC×NC block of B stays in L2 while it is reused by every row of A/C.

# The Kernel

```cpp
void sgemm_2dtile(std::size_t M, std::size_t N, std::size_t L,
                 float alpha, const float* A, const float* B,
                 float beta, float* C) {

        constexpr std::size_t KC = 64;
        constexpr std::size_t NC = 1024; // KC*NC*4 bytes of B plus NC*4 bytes of C, aim for ≤ ~256 KB for ryzen 5500

        for (std::size_t i = 0; i < M * N; i++) {
            C[i] *= beta;
        }

        for (std::size_t kt = 0; kt < L; kt += KC) {
            std::size_t kend = std::min(L, kt + KC);
            for (std::size_t jt = 0; jt < N; jt += NC) {
                std::size_t jend = std::min(N, jt + NC);
                for (std::size_t i = 0; i < M; i++) {
                    std::size_t k = kt;
                    for (; k + 4 <= kend; k += 4) {
                        const float a0 = alpha * A[i * L + k];
                        const float a1 = alpha * A[i * L + k + 1];
                        const float a2 = alpha * A[i * L + k + 2];
                        const float a3 = alpha * A[i * L + k + 3];
                        const float* b0 = &B[k * N];
                        const float* b1 = &B[(k + 1) * N];
                        const float* b2 = &B[(k + 2) * N];
                        const float* b3 = &B[(k + 3) * N];
                        for (std::size_t j = jt; j < jend; j++) {
                            C[i * N + j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
                        }
                    }
                    for (; k < kend; k++) {
                        const float a = alpha * A[i * L + k];
                        for (std::size_t j = jt; j < jend; j++) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }

```

# Choosing Tile Sizes
- Ryzen 5 5500 cache table: L1D 32 KB, L2 512 KB per core, L3 16 MB shared.
- Constraint: B tile (KC×NC×4 bytes) plus one row segment of C (NC×4 bytes) should fit in L2 (~256 KB target). KC=64, NC=1024 → 256 KB.
- Because J is tiled, the B tile is independent of N, so it no longer blows past L2 at large N.

# GodBolt

Inner-loop is the same as sgemm_tile_unroll.

# llvm-mca

No difference from sgemm_tile_unroll for the same reason as GodBolts

# Google Benchmark


| n | unrolled | 2D tiled | speedup | slower than OpenBLAS |
|---|---|---|---|---|
| 256 | 0.689 ms (48.8 GFLOP/s) | 0.676 ms (49.7 GFLOP/s) | 1.02× | ~2.2× |
| 512 | 5.12 ms (52.6 GFLOP/s) | 5.14 ms (52.3 GFLOP/s) | 1.00× | ~2.2× |
| 1024 | 50.8 ms (42.3 GFLOP/s) | 50.1 ms (42.9 GFLOP/s) | 1.01× | ~2.7× |
| 2048 | 483 ms (35.6 GFLOP/s) | 465 ms (37.0 GFLOP/s) | 1.04× | ~3.1× |
| 4096 | 3792 ms (36.3 GFLOP/s) | 3486 ms (39.4 GFLOP/s) | 1.09× | ~3.0× |
| 8192 | 30533 ms (36.0 GFLOP/s) | 27494 ms (40.0 GFLOP/s) | 1.11× | ~3.0× |

- Litte gain with small N as expected as the cache is still utilised well.
- Performance scales better with N, and stays flat beyond 4096 (39.4 GFLOP/s at 4096, 40.0 at 8192).
- The n=8192 row is from `bench/results/n8192_2026-11-03.csv` (3 repetitions); the other rows are from `bench/results/bench_2026-11-03.csv`.


# AMD uProf

- Follow up on the 1D shortfall (~37 vs ~85 GFLOP/s ceiling).
- Compare unrolled vs 2D
- Confirm B reads now hit L2

# Next Step

Register blocking: a micro-kernel holding a small block of C in ymm registers, so loads from L1 are amortised over many FMAs and the Zn3FPP45 bottleneck finally gives way to FMA throughput.
