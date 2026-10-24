#include <benchmark/benchmark.h>
#include <cblas.h>
#include <cstdio>
#include <vector>
#include "sgemm.hpp"

// Report GFLOPS (all kernels run single-threaded)
static void set_counters(benchmark::State& state, std::size_t n) {
    // 2*n^3 flops per GEMM, reported as FLOP/s
    const double flops = 2.0 * n * n * n;
    state.counters["FLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);
}

template <auto Kernel>
static void BM_sgemm(benchmark::State& state) {
    const std::size_t n = state.range(0);
    std::vector<float> A(n * n, 1.0f), B(n * n, 1.0f), C(n * n, 0.0f);
    for (auto _ : state) {
        Kernel(n, n, n, 1.0f, A.data(), B.data(), 0.0f, C.data());
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    set_counters(state, n);
}
BENCHMARK(BM_sgemm<sgemm_naive>)->Name("naive")->RangeMultiplier(2)->Range(256, 1024)
    ->Unit(benchmark::kMillisecond)->Repetitions(10);
BENCHMARK(BM_sgemm<sgemm_ric>)->Name("ric")->RangeMultiplier(2)->Range(256, 1024)
    ->Unit(benchmark::kMillisecond)->Repetitions(10);
BENCHMARK(BM_sgemm<sgemm_tile>)->Name("tile")->RangeMultiplier(2)->Range(256, 1024)
    ->Unit(benchmark::kMillisecond)->Repetitions(10);
BENCHMARK(BM_sgemm<sgemm_tile_unroll>)->Name("tile_unroll")->RangeMultiplier(2)->Range(256, 1024)
    ->Unit(benchmark::kMillisecond)->Repetitions(10);

// Reference: OpenBLAS, same row-major layout and no transposes
static void BM_sgemm_openblas(benchmark::State& state) {
    const std::size_t n = state.range(0);
    std::vector<float> A(n * n, 1.0f), B(n * n, 1.0f), C(n * n, 0.0f);
    for (auto _ : state) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                    1.0f, A.data(), n, B.data(), n, 0.0f, C.data(), n);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    set_counters(state, n);
}
BENCHMARK(BM_sgemm_openblas)->Name("openblas")->RangeMultiplier(2)->Range(256, 1024)
    ->Unit(benchmark::kMillisecond)->Repetitions(10);

// Console output limited to the mean, median and cv aggregate rows
class AggregateReporter : public benchmark::ConsoleReporter {
public:
    void ReportRuns(const std::vector<Run>& runs) override {
        std::vector<Run> keep;
        for (const Run& r : runs)
            if (r.run_type == Run::RT_Aggregate && r.aggregate_name != "stddev")
                keep.push_back(r);
        if (!keep.empty()) ConsoleReporter::ReportRuns(keep);
    }
};

int main(int argc, char** argv) {
    // Single-threaded so OpenBLAS is compared like-for-like with our kernels
    openblas_set_num_threads(1);
    std::printf("OpenBLAS: %s, %d thread(s)\n\n", openblas_get_config(), openblas_get_num_threads());

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    AggregateReporter reporter;
    benchmark::RunSpecifiedBenchmarks(&reporter);
    benchmark::Shutdown();
}
