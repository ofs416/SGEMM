#include "../include/sgemm.hpp"
#include <stdexcept>
#include <iostream>

// Naive triple loop. Row-major: A(i, k) = A[i * L + k], B(k, j) = B[k * N + j]
void sgemm_naive(std::size_t M, std::size_t N, std::size_t L,
                 float alpha, const float* A, const float* B,
                 float beta, float* C) {
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
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
