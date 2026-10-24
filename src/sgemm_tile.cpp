#include "../include/sgemm.hpp"
#include <cstddef>

// https://siboehm.com/articles/22/Fast-MMM-on-CPU
// Toy example goes over @ between two 6x6 matrices with a L1 cache size for 36 floats
// When we finish the top row of C (6 floats), we have 6 more floats in A in L1 and then 24 floats from B (the last 4 rows)
// now when stepping thropugh the outer loop, there is a cache miss on B as the stack is FIFO and we need the first row that
// has already been pushed out.
// imagine a dog going in circles chasing its tail.


void sgemm_tile(std::size_t M, std::size_t N, std::size_t L,
                 float alpha, const float* A, const float* B,
                 float beta, float* C) {

        constexpr std::size_t KC = 32;   // B tile ≈ KC*N*4 bytes, aim for ≤ ~256 KB for ryzen 5500

        for (std::size_t i = 0; i < M * N; i++) {
            C[i] *= beta;
        }

        for (std::size_t kt = 0; kt < L; kt += KC) {
            std::size_t kend = std::min(L, kt + KC);
            for (std::size_t i = 0; i < M; i++) {
                for (std::size_t k = kt; k < kend; k++) {
                    const float a = alpha * A[i * L + k];
                    for (std::size_t j = 0; j < N; j++) {
                        C[i * N + j] += a * B[k * N + j];
                    }
                }
            }
        }
    }

// ┌──────┬────────────────────────┬───────────────────────┬─────────────────┐
// │  n   │          Tile          │  OpenBLAS (1 thread)  │ tile slower by  │
// ├──────┼────────────────────────┼───────────────────────┼─────────────────┤
// │ 256  │ 1.10 ms (28.8 GFLOP/s) │ 0.29 ms (109 GFLOP/s) │ ~3.8×           │
// ├──────┼────────────────────────┼───────────────────────┼─────────────────┤
// │ 512  │ 8.85 ms (28.8 GFLOP/s) │ 2.23 ms (114 GFLOP/s) │ ~4.0×           │
// ├──────┼────────────────────────┼───────────────────────┼─────────────────┤
// │ 1024 │ 80.4 ms (25.3 GFLOP/s) │ 17.7 ms (115 GFLOP/s) │ ~4.5×           │
// └──────┴────────────────────────┴───────────────────────┴─────────────────┘
