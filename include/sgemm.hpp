#pragma once
#include <cstddef>
#include "matrix.hpp"

// C = alpha * A * B + beta * C, row-major. A is MxL, B is LxN, C is MxN.
void sgemm_naive(std::size_t M, std::size_t N, std::size_t L,
                 float alpha, const float* A, const float* B,
                 float beta, float* C);

void sgemm_ric(std::size_t M, std::size_t N, std::size_t L,
                 float alpha, const float* A, const float* B,
                 float beta, float* C);

void sgemm_tile(std::size_t M, std::size_t N, std::size_t L,
                 float alpha, const float* A, const float* B,
                 float beta, float* C);

// Convenience wrapper over MatrixF
MatrixF matmul(const MatrixF& A, const MatrixF& B);
