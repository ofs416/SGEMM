#include <cmath>
#include <iostream>
#include <vector>
#include "../include/sgemm.hpp"

// Double-precision reference: C = alpha * A * B + beta * C
static std::vector<double> reference(std::size_t M, std::size_t N, std::size_t L,
                                     float alpha, const std::vector<float>& A,
                                     const std::vector<float>& B, float beta,
                                     const std::vector<float>& C) {
    std::vector<double> R(M * N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            double acc = 0;
            for (std::size_t k = 0; k < L; ++k) acc += double(A[i * L + k]) * B[k * N + j];
            R[i * N + j] = alpha * acc + beta * C[i * N + j];
        }
    return R;
}

using Kernel = decltype(&sgemm_naive);

static bool check(Kernel kernel, const char* name, std::size_t M, std::size_t N, std::size_t L, float alpha, float beta) {
    std::vector<float> A(M * L), B(L * N), C(M * N);
    for (std::size_t i = 0; i < A.size(); ++i) A[i] = float((i * 7) % 13) / 13 - 0.5f;
    for (std::size_t i = 0; i < B.size(); ++i) B[i] = float((i * 5) % 11) / 11 - 0.5f;
    for (std::size_t i = 0; i < C.size(); ++i) C[i] = float((i * 3) % 7) / 7;
    auto R = reference(M, N, L, alpha, A, B, beta, C);
    kernel(M, N, L, alpha, A.data(), B.data(), beta, C.data());
    for (std::size_t i = 0; i < C.size(); ++i)
        if (std::fabs(C[i] - R[i]) > 1e-4 * (1 + std::fabs(R[i]))) {
            std::cerr << "FAIL " << name << " " << M << "x" << N << "x" << L << " alpha=" << alpha
                      << " beta=" << beta << " at " << i << ": " << C[i] << " vs " << R[i] << "\n";
            return false;
        }
    return true;
}

int main() {
    const std::size_t dims[][3] = {{1, 1, 1}, {7, 5, 3}, {63, 65, 64}, {100, 33, 257}, {5, 70, 1}};
    const float ab[][2] = {{1, 0}, {1, 1}, {2.5f, -0.5f}, {0, 1}};
    const struct { Kernel fn; const char* name; } kernels[] = {
        {sgemm_naive, "naive"}, {sgemm_blocked, "blocked"}};
    for (auto& k : kernels)
        for (auto& d : dims)
            for (auto& p : ab)
                if (!check(k.fn, k.name, d[0], d[1], d[2], p[0], p[1])) return 1;

    MatrixF a(3, 5, 2.0), b(5, 3, 3.0);
    MatrixF c = matmul(a, b);
    if (c.rows() != 3 || c.cols() != 3 || c(1, 1) != 30.0f) { std::cerr << "FAIL matmul\n"; return 1; }
    std::cout << "sgemm tests passed\n";
}
