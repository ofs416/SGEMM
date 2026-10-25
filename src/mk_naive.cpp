#include <cstddef>

// Inner loop of sgemm_naive: dot product over k. B is walked with stride N.
float mk_naive(std::size_t L, std::size_t N, const float* __restrict A, const float* __restrict B) {
    float acc = 0.0f;
    for (std::size_t k = 0; k < L; ++k)
        acc += A[k] * B[k * N];
    return acc;
}
