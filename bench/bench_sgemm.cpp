#include <benchmark/benchmark.h>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include "sgemm.hpp"

// Effective core clock: a chain of dependent integer adds retires 1 per cycle.
static double measure_ghz() {
    constexpr long iters = 200'000'000;
    long x = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < iters; ++i)
        asm volatile("add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
                     "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0"
                     : "+r"(x));
    auto t1 = std::chrono::steady_clock::now();
    return iters * 10.0 / std::chrono::duration<double, std::nano>(t1 - t0).count();
}

// FP32 FLOPs/cycle/core assuming 2 FMA pipes: lanes * 2 (mul+add) * 2 pipes
static int flops_per_cycle() {
    if (__builtin_cpu_supports("avx512f")) return 16 * 2 * 2;
    if (__builtin_cpu_supports("fma"))     return 8 * 2 * 2;
    return 4 * 2;  // SSE: separate mul and add pipes
}

static double g_peak_1core = 0.0;  // FLOP/s

static void BM_sgemm_naive(benchmark::State& state) {
    const std::size_t n = state.range(0);
    std::vector<float> A(n * n, 1.0f), B(n * n, 1.0f), C(n * n, 0.0f);
    for (auto _ : state) {
        sgemm_naive(n, n, n, 1.0f, A.data(), B.data(), 0.0f, C.data());
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    // 2*n^3 flops per GEMM, reported as FLOP/s
    const double flops = 2.0 * n * n * n;
    state.counters["FLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);
    // fraction of single-core peak (the naive kernel is single-threaded)
    state.counters["%peak"] = benchmark::Counter(
        100.0 * flops / g_peak_1core, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_sgemm_naive)->RangeMultiplier(2)->Range(64, 1024)
    ->Unit(benchmark::kMillisecond)->Repetitions(5);

int main(int argc, char** argv) {
    const double ghz = measure_ghz();
    const int fpc = flops_per_cycle();
    const unsigned cores = std::thread::hardware_concurrency() / 2;  // assumes SMT
    g_peak_1core = ghz * 1e9 * fpc;
    std::printf("Measured clock: %.2f GHz, %d FP32 FLOPs/cycle/core\n", ghz, fpc);
    std::printf("Peak: %.1f GFLOP/s (1 core), %.1f GFLOP/s (%u cores)\n\n",
                g_peak_1core / 1e9, g_peak_1core * cores / 1e9, cores);

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
