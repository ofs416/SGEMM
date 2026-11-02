#include "../include/sgemm.hpp"
#include <cstddef>

// The previous tile version, involves M*N+M*N*L C reads and writes
// We reduce this to  M*N + M*N*L/4. by unrolling the K loop by a factor of 4


void sgemm_tile_unroll(std::size_t M, std::size_t N, std::size_t L,
                 float alpha, const float* A, const float* B,
                 float beta, float* C) {

        constexpr std::size_t KC = 32;   // B tile ≈ KC*N*4 bytes, ≤ ~256 KB for ryzen 5500

        for (std::size_t i = 0; i < M * N; i++) {
            C[i] *= beta;
        }

        for (std::size_t kt = 0; kt < L; kt += KC) {
            std::size_t kend = std::min(L, kt + KC);
            for (std::size_t i = 0; i < M; i++) {
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
                    for (std::size_t j = 0; j < N; j++) {
                        C[i * N + j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
                    }
                }
                for (; k < kend; k++) {
                    const float a = alpha * A[i * L + k];
                    for (std::size_t j = 0; j < N; j++) {
                        C[i * N + j] += a * B[k * N + j];
                    }
                }
            }
        }
    }