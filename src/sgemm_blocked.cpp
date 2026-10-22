#include "../include/sgemm.hpp"

// The Naive version is row-continuous, meaning when accessing across rows during the inner dot-product we jump around in memory location a lot
// Data is loaded from memory with a fixed cache size, almost everytime we need to access multiple rows we get cache misses
// This is fixed by reordering the loops. Mathematically, we compute the outer product in the inner loop as opposed to the inner prodcut

// Blocked triple loop. Row-major: A(i, k) = A[i * L + k], B(k, j) = B[k * N + j]
void sgemm_blocked(std::size_t M, std::size_t N, std::size_t L,
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

// ┌──────┬────────────────────────┬───────────────────────┬─────────────────┐
// │  n   │        Blocked         │  OpenBLAS (1 thread)  │ naive slower by │
// ├──────┼────────────────────────┼───────────────────────┼─────────────────┤
// │ 256  │ 1.16 ms (28.7 GFLOP/s) │ 0.32 ms (108 GFLOP/s) │ ~3.62×          │
// ├──────┼────────────────────────┼───────────────────────┼─────────────────┤
// │ 512  │ 9.51 ms (28.2 GFLOP/s) │ 2.41 ms (114 GFLOP/s) │ ~3.94×          │
// ├──────┼────────────────────────┼───────────────────────┼─────────────────┤
// │ 1024 │ 82.6 ms (26.2 GFLOP/s) │ 19.4 ms (113 GFLOP/s) │ ~4.26×          │
// └──────┴────────────────────────┴───────────────────────┴─────────────────┘
