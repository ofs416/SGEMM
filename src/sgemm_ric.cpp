#include "../include/sgemm.hpp"

// The Naive version is row-continuous, meaning when accessing across rows during the inner dot-product we jump around in memory location a lot
// Data is loaded from memory with a fixed cache size, almost everytime we need to access multiple rows we get cache misses
// This is fixed by reordering the loops. Mathematically, we compute the outer product in the inner loop as opposed to the inner prodcut

// Ric triple loop. Row-major: A(i, k) = A[i * L + k], B(k, j) = B[k * N + j]
void sgemm_ric(std::size_t M, std::size_t N, std::size_t L,
                 float alpha, const float* A, const float* B,
                 float beta, float* C) {
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j)
            C[i * N + j] *= beta;                       // beta applied once per element
        for (std::size_t k = 0; k < L; ++k) {
            const float a = alpha * A[i * L + k];
            for (std::size_t j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
    }
}

// clang -O3 -march=native, mean of 10 repetitions
// ┌──────┬─────────────────────────┬────────────────────────┬──────────────────────┐
// │ n    │ Ric                     │ OpenBLAS (1 thread)    │ ric slower than BLAS │
// ├──────┼─────────────────────────┼────────────────────────┼──────────────────────┤
// │ 256  │ 0.913 ms (38.0 GFLOP/s) │ 0.318 ms (109 GFLOP/s) │ ~2.9×                │
// ├──────┼─────────────────────────┼────────────────────────┼──────────────────────┤
// │ 512  │ 7.96 ms (34.9 GFLOP/s)  │ 2.45 ms (113 GFLOP/s)  │ ~3.2×                │
// ├──────┼─────────────────────────┼────────────────────────┼──────────────────────┤
// │ 1024 │ 67.5 ms (32.9 GFLOP/s)  │ 19.4 ms (115 GFLOP/s)  │ ~3.5×                │
// ├──────┼─────────────────────────┼────────────────────────┼──────────────────────┤
// │ 2048 │ 1162 ms (15.3 GFLOP/s)  │ 153 ms (116 GFLOP/s)   │ ~7.6×                │
// ├──────┼─────────────────────────┼────────────────────────┼──────────────────────┤
// │ 4096 │ 11924 ms (11.9 GFLOP/s) │ 1203 ms (118 GFLOP/s)  │ ~9.9×                │
// └──────┴─────────────────────────┴────────────────────────┴──────────────────────┘
