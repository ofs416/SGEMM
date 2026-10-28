#include <algorithm>
#include <cstddef>

// Standalone (godbolt-friendly) copy of sgemm_tile's loops: beta scale + kt/i/k/j nest.
void mk_o_tile(std::size_t M, std::size_t N, std::size_t L,
               float alpha, const float* __restrict A, const float* __restrict B,
               float beta, float* __restrict C) {
    constexpr std::size_t KC = 32;

    for (std::size_t i = 0; i < M * N; i++)
        C[i] *= beta;

    for (std::size_t kt = 0; kt < L; kt += KC) {
        std::size_t kend = std::min(L, kt + KC);
        for (std::size_t i = 0; i < M; i++) {
            for (std::size_t k = kt; k < kend; k++) {
                const float a = alpha * A[i * L + k];
                for (std::size_t j = 0; j < N; j++)
                    C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}
