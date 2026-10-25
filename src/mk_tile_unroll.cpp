#include <cstddef>

// Inner loop of sgemm_tile_unroll: one row of C += four rows of B, k unrolled by 4.
void mk_tile_unroll(std::size_t N, float a0, float a1, float a2, float a3,
                    const float* __restrict b0, const float* __restrict b1,
                    const float* __restrict b2, const float* __restrict b3,
                    float* __restrict C) {
    for (std::size_t j = 0; j < N; ++j)
        C[j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
}
