#include <cstddef>

// Standalone (godbolt-friendly) copy of sgemm_ric's loops: beta scale + i/k/j nest.
void mk_o_ric(std::size_t M, std::size_t N, std::size_t L,
              float alpha, const float* __restrict A, const float* __restrict B,
              float beta, float* __restrict C) {
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j)
            C[i * N + j] *= beta;
        for (std::size_t k = 0; k < L; ++k) {
            const float a = alpha * A[i * L + k];
            for (std::size_t j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
    }
}
