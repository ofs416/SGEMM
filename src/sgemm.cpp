#include "../include/sgemm.hpp"
#include <stdexcept>
#include <iostream>

// Naive triple loop. Row-major: A(i, k) = A[i * L + k], B(k, j) = B[k * N + j]
void sgemm_naive(std::size_t M, std::size_t N, std::size_t L,
                 float alpha, const float* A, const float* B,
                 float beta, float* C) {
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float acc = 0.0f;                           // Performed in the register for increased speed (not fully naive)
            for (std::size_t k = 0; k < L; ++k)
                acc += A[i * L + k] * B[k * N + j];
            C[i * N + j] = alpha * acc + beta * C[i * N + j];
        }
    }
}


MatrixF matmul(const MatrixF& A, const MatrixF& B) {
    if (A.cols() != B.rows())
        throw std::invalid_argument("matmul: inner dimensions differ");
    MatrixF C(A.rows(), B.cols());   // zeros, so beta = 0 is safe
    sgemm_naive(A.rows(), B.cols(), A.cols(),
                1.0f, A.data(), B.data(), 0.0f, C.data());
    return C;
}


// ┌──────┬────────────────────────┬────────────────┬───────────────────┬───────────────────┐
// │  n   │     no flags (-O0)     │      -O3       │ -O3 -march=native │ -O0 → -O3 speedup │
// ├──────┼────────────────────────┼────────────────┼───────────────────┼───────────────────┤
// │ 256  │ 51.0 ms (0.67 GFLOP/s) │ 21.2 ms (1.61) │ 22.1 ms (1.55)    │ 2.4×              │
// ├──────┼────────────────────────┼────────────────┼───────────────────┼───────────────────┤
// │ 512  │ 413 ms (0.66)          │ 274 ms (1.00)  │ 279 ms (0.98)     │ 1.5×              │
// ├──────┼────────────────────────┼────────────────┼───────────────────┼───────────────────┤
// │ 1024 │ 4068 ms (0.54)         │ 4094 ms (0.54) │ 4130 ms (0.53)    │ 1.0×              │
// └──────┴────────────────────────┴────────────────┴───────────────────┴───────────────────┘

// -O3 -march=native
// ┌──────┬────────────────────────┬───────────────────────┬─────────────────┐
// │  n   │         naive          │  OpenBLAS (1 thread)  │ naive slower by │
// ├──────┼────────────────────────┼───────────────────────┼─────────────────┤
// │ 256  │ 22.2 ms (1.5 GFLOP/s)  │ 0.32 ms (108 GFLOP/s) │ ~70×            │
// ├──────┼────────────────────────┼───────────────────────┼─────────────────┤
// │ 512  │ 275 ms (1.0 GFLOP/s)   │ 2.41 ms (114 GFLOP/s) │ ~115×           │
// ├──────┼────────────────────────┼───────────────────────┼─────────────────┤
// │ 1024 │ 4028 ms (0.54 GFLOP/s) │ 19.4 ms (113 GFLOP/s) │ ~210×           │
// └──────┴────────────────────────┴───────────────────────┴─────────────────┘
