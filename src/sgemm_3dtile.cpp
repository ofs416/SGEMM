#include "../include/sgemm.hpp"
#include <algorithm>
#include <cstddef>

// 3D tiling: block over all of M, N and L. The (KC x NC) panel of B and the
// (MC x NC) panel of C are reused across the MC rows of the current block.
void sgemm_3dtile(std::size_t M, std::size_t N, std::size_t L,
                  float alpha, const float* A, const float* B,
                  float beta, float* C) {

    constexpr std::size_t MC = 64;
    constexpr std::size_t KC = 64;
    constexpr std::size_t NC = 1024;

    for (std::size_t i = 0; i < M * N; i++) {
        C[i] *= beta;
    }

    for (std::size_t jt = 0; jt < N; jt += NC) {
        const std::size_t jend = std::min(N, jt + NC);
        for (std::size_t kt = 0; kt < L; kt += KC) {
            const std::size_t kend = std::min(L, kt + KC);
            for (std::size_t it = 0; it < M; it += MC) {
                const std::size_t iend = std::min(M, it + MC);
                for (std::size_t i = it; i < iend; i++) {
                    std::size_t k = kt;
                    for (; k + 4 <= kend; k += 4) {
                        const float a0 = alpha * A[i * L + k];
                        const float a1 = alpha * A[i * L + k + 1];
                        const float a2 = alpha * A[i * L + k + 2];
                        const float a3 = alpha * A[i * L + k + 3];
                        const float* b0 = &B[k * N];
                        const float* b1 = &B[(k + 1) * N];
                        const float* b2 = &B[(k + 2) * N];
                        const float* b3 = &B[(k + 3) * N];
                        for (std::size_t j = jt; j < jend; j++) {
                            C[i * N + j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
                        }
                    }
                    for (; k < kend; k++) {
                        const float a = alpha * A[i * L + k];
                        for (std::size_t j = jt; j < jend; j++) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
