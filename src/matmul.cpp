#include "../include/sgemm.hpp"
#include <stdexcept>

MatrixF matmul(const MatrixF& A, const MatrixF& B) {
    if (A.cols() != B.rows())
        throw std::invalid_argument("matmul: inner dimensions differ");
    MatrixF C(A.rows(), B.cols());   // zeros, so beta = 0 is safe
    sgemm_naive(A.rows(), B.cols(), A.cols(),
                1.0f, A.data(), B.data(), 0.0f, C.data());
    return C;
}
