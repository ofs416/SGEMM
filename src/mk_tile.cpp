#include <cstddef>

// Inner loop of sgemm_tile: one row of C += a * one row of B.
void mk_tile(std::size_t N, float a, const float* __restrict B, float* __restrict C) {
    for (std::size_t j = 0; j < N; ++j)
        C[j] += a * B[j];
}
