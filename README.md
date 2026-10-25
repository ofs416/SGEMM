# SGEMM

## Analysis

Each kernel's hot inner loop is mirrored in `src/mk_*.cpp` (e.g. `mk_tile_unroll`) so it can be analysed on its own. Analysis has three steps, and each one answers a different question.

1. **Static analysis of the inner loop (`llvm-mca`)**: what is the best this loop could do if every load hit L1?
2. **Isolated microbenchmark**: does the loop reach that estimate on L1-resident data?
3. **Full SGEMM benchmark**: how far below the microkernel number does the full multiply fall? That gap is the memory hierarchy (blocking, packing, edge cases).

The reference compiler is clang (the project builds with it by default), so the asm you analyse is the asm that was benchmarked.

Reference peak for the Ryzen 5 5500 (Zen 3): 2 FMA ports x 8 floats x 2 flops = 32 flops/cycle, about 130 GFLOP/s at ~4 GHz. OpenBLAS (single thread) reaches ~115.

### Step 1: `llvm-mca` on the inner loop

Running `llvm-mca` on the whole function is misleading. It treats prologue, vector loop, remainder loop and `ret` as one straight-line block, and does not model the loop. Extract only the vectorised loop instead:

```sh
clang++ src/mk_tile_unroll.cpp -O3 -march=native -S -o mk.s

# find the loop: the label commented "Inner Loop Header" up to its backward branch
grep -n "Inner Loop Header\|jne" mk.s

# cut those lines out (here 26-53) and analyse them against the actual CPU
sed -n 26,53p mk.s > loop.s
llvm-mca -mcpu=znver3 -iterations=200 loop.s
```

or 

```sh
clang++ src/mk_ric.cpp -O3 -march=native -ffast-math -S -o - \
  | awk '/^\.LBB0_4:/{f=1;next} f{print} /jne[[:space:]]+\.LBB0_4/{exit}' \
  | llvm-mca -mcpu=znver3 -iterations=200
```

Notes:

- Pass `-mcpu=znver3` explicitly rather than `native`. Under WSL2 `native` may not resolve to the real CPU.
- The first loop is the vectorised body. The later `Inner Loop Header` is the scalar remainder, so ignore it.
- Add `-bottleneck-analysis` or read the `Resource pressure by instruction` table to see which ports saturate.
- Also check the loop for spills (stores to the stack) and that it was vectorised at all.

Convert the result to flops/cycle: `flops per iteration / Block RThroughput`, then compare with the 32 flops/cycle peak.

Example, `mk_tile_unroll`: one iteration covers 32 floats (4 vectors of 8) at 8 flops each, i.e. 256 flops. Block RThroughput is 12 cycles, so about 21 flops/cycle, ~67% of peak (~85 GFLOP/s at 4 GHz). Without register blocking the loop is limited by loads (5 vector loads per 4 FMAs) and by not using FMA for every operation, so it cannot reach peak.

`mk_tile` and `mk_ric` need 2 loads and a store per FMA, so expect a much lower ceiling.

### Step 2: isolated microbenchmark

Call the `mk_*` function repeatedly on buffers that fit in L1d (32 KB per core). For `mk_tile_unroll` that is 5 rows of N floats, so N up to ~1024. The measured GFLOP/s should be close to the step 1 estimate. If it is much lower, something `llvm-mca` can't see is in the way (unaligned loads, 4K aliasing between `C` and `B`, loop overhead).

### Step 3: full SGEMM

`bench_sgemm` reports GFLOP/s for the full multiply at several sizes. Compare against the step 2 number and the peak:

- step 3 well below step 2: memory hierarchy (blocking, packing, edge handling). Confirm with `valgrind --tool=cachegrind --cache-sim=yes` or `perf stat -e L1-dcache-load-misses` if available, comparing D1 miss rates at small and large `n`.
- step 2 well below peak: the kernel design (register blocking, tile shape).

## Building, testing and benchmarking

Requires CMake 3.20+, a C++20 compiler (clang++ by default, see `CMakeLists.txt`) and OpenBLAS (the benchmark reference). GoogleTest and Google Benchmark are fetched automatically.

```sh
cmake -S . -B build          # Release by default, built with -march=native
cmake --build build -j
```

Run the tests:

```sh
ctest --test-dir build --output-on-failure
```

Run the benchmarks (single-threaded, naive vs OpenBLAS):

```sh
./build/bench_sgemm
./build/bench_sgemm --benchmark_filter='naive/256'   # a subset
```

All numbers below were measured with clang 18.1.3 on a Ryzen 5 5500 (mean of 10 repetitions, single thread). To benchmark without compiler flags, configure a separate build directory, e.g. `cmake -S . -B build-O0 -DCMAKE_BUILD_TYPE=None`, and edit the `-march=native` option in `CMakeLists.txt`.

## 1. Naive

Accumulation of the inner dot product is done in a register for increased speed (so not fully naive).

In this implementation we rapidly hit cache overloading, meaning memory stalls cause huge delays.

```cpp
for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (std::size_t k = 0; k < L; ++k)
            acc += A[i * L + k] * B[k * N + j];
        C[i * N + j] = alpha * acc + beta * C[i * N + j];
    }
}
```

### 1.1 No compiler flags

| n    | naive                  | OpenBLAS (1 thread)   | naive slower by |
|------|------------------------|-----------------------|-----------------|
| 256  | 50.2 ms (0.68 GFLOP/s) | 0.32 ms (108 GFLOP/s) | ~158×           |
| 512  | 410 ms (0.67 GFLOP/s)  | 2.43 ms (113 GFLOP/s) | ~169×           |
| 1024 | 4004 ms (0.54 GFLOP/s) | 19.0 ms (114 GFLOP/s) | ~211×           |

### 1.2 `-O3 -march=native`

| n    | naive                  | OpenBLAS (1 thread)   | naive slower by | flag speedup |
|------|------------------------|-----------------------|-----------------|--------------|
| 256  | 35.7 ms (0.96 GFLOP/s) | 0.32 ms (108 GFLOP/s) | ~112×           | 1.4×         |
| 512  | 351 ms (0.78 GFLOP/s)  | 2.43 ms (113 GFLOP/s) | ~144×           | 1.2×         |
| 1024 | 4070 ms (0.54 GFLOP/s) | 19.0 ms (114 GFLOP/s) | ~214×           | 1.0×         |

OpenBLAS is a prebuilt library, so its numbers don't depend on these flags.


### 1.3 Naive across compiler flags

| n    | `-O0`                  | `-O3`          | `-O3 -march=native` | `-O0` → `-O3` speedup |
|------|------------------------|----------------|---------------------|-----------------------|
| 256  | 50.2 ms (0.68 GFLOP/s) | 26.8 ms (1.27) | 35.7 ms (0.96)      | 1.9×                  |
| 512  | 410 ms (0.67)          | 298 ms (0.91)  | 351 ms (0.78)       | 1.4×                  |
| 1024 | 4004 ms (0.54)         | 4009 ms (0.54) | 4070 ms (0.54)      | 1.0×                  |


## 2. Loop reorder (`sgemm_ric`)

In the naive version the inner loop walks down a column of `B` (`B[k * N + j]` with `k` varying), a stride of `N` floats. Matrices are row-major and a cache line holds consecutive elements, so almost every access of `B` touches a different cache line and we get a high rate of cache misses.

The fix is to reorder the loops to `i, k, j`. The inner loop now computes a row of the outer-product update `C[i, :] += A[i, k] * B[k, :]` instead of a dot product, so `B` and `C` are both read along rows (unit stride) and `A[i, k]` is a single value held in a register. `beta` is applied once per element up front, then `alpha * A[i, k]` is accumulated into `C`.

```cpp
for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j)
        C[i * N + j] *= beta;
    for (std::size_t k = 0; k < L; ++k) {
        const float a = alpha * A[i * L + k];
        for (std::size_t j = 0; j < N; ++j)
            C[i * N + j] += a * B[k * N + j];
    }
}
```

This only reorders the loops and does no tiling, so `ric` (reordered loops) is not yet the tiled version. Rows of `B` and `C` are still streamed in full for every `i`, so once a row no longer fits in cache the speed drops with `n` (38.1 to 31.9 GFLOP/s below).

### 2.1 `-O3 -march=native`

| n    | reordered              | OpenBLAS (1 thread)   | slower than OpenBLAS | speedup over naive |
|------|------------------------|-----------------------|----------------------|--------------------|
| 256  | 0.896 ms (38.1 GFLOP/s)| 0.32 ms (108 GFLOP/s) | ~2.8×                | ~40×               |
| 512  | 8.08 ms (33.8 GFLOP/s) | 2.43 ms (113 GFLOP/s) | ~3.3×                | ~43×               |
| 1024 | 68.7 ms (31.9 GFLOP/s) | 19.0 ms (114 GFLOP/s) | ~3.6×                | ~59×               |

The gap was closed considerably with these cache hits. Furthermore it now scales far better with n. The scaling is improved by the decrease in cache-hits, whilst the rest of the speed up is actually from thge compilers ability to now vectorise this.


### 2.2 Effect of vectorisation

Same kernel built with and without auto-vectorisation (`-fno-vectorize -fno-slp-vectorize`, placed after `-O3`: clang re-enables vectorisation if they come before it), single thread.

| n    | scalar      | vectorised   | vectorisation speedup |
|------|-------------|--------------|-----------------------|
| 256  | 5.3 GFLOP/s | 38.1 GFLOP/s | ~7.2×                 |
| 512  | 5.3 GFLOP/s | 33.8 GFLOP/s | ~6.4×                 |
| 1024 | 5.3 GFLOP/s | 31.9 GFLOP/s | ~6.1×                 |



## 3. Tiling over `k` (`sgemm_tile`)

The reordered kernel still streams every row of `B` for each `i`. By the time the next `i` needs `B[k, :]` again it has been evicted from cache, so `B` is re-fetched from memory once per row of `C`.

The fix is to tile the middle loop `k`. The outer loop steps `k` in blocks of `KC`, and for each block we sweep all rows `i`. The `KC` rows of `B` in the block (`KC * N * 4` bytes) are loaded once and reused by every `i` while they are still in cache. `KC = 32` is fixed, which keeps the `B` tile at 128 KB for `n = 1024`, well inside the 512 KB L2 of the Ryzen 5 5500. `beta` is applied in a separate pass up front.

```cpp
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
```

The cost is that `C` is now read and written once per tile instead of once in total, so `KC` can't be made arbitrarily small.

### 3.1 `-O3 -march=native`

Mean of 10 repetitions.

| n    | tiled                  | OpenBLAS (1 thread)   | slower than OpenBLAS | speedup over reordered |
|------|------------------------|-----------------------|----------------------|------------------------|
| 256  | 0.932 ms (36.7 GFLOP/s)| 0.32 ms (108 GFLOP/s) | ~2.9×                | ~0.96×                 |
| 512  | 7.47 ms (36.5 GFLOP/s) | 2.43 ms (113 GFLOP/s) | ~3.1×                | ~1.08×                 |
| 1024 | 64.8 ms (33.9 GFLOP/s) | 19.0 ms (114 GFLOP/s) | ~3.4×                | ~1.06×                 |

Tiling `k` alone gives little over the reordered kernel (none at n=256, ~1.06-1.08× at 512 and 1024). Each tile still re-reads and re-writes `C` on every `k` step, so the load/store count is unchanged (see 3.2). The remaining gap to OpenBLAS is mostly register blocking and tiling `j` as well.

### 3.2 Unrolling `k` by 4 (`sgemm_tile_unroll`)

The previous tile version, involves M\*N+M\*N\*L C reads and writes. We reduce this to  M\*N + M\*N\*L/4 by unrolling the K loop by a factor of 4. Each loaded `C[j]` four FMAs instead of one. A scalar loop handles the remainder when the tile length isn't a multiple of 4.

```cpp
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
    for (std::size_t j = 0; j < N; j++)
        C[i * N + j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
}
for (; k < kend; k++) { /* scalar remainder, as in 3 */ }
```

The sum is now added to `C` in a different order, so results can differ from the tiled kernel in the last bits.

Mean of 10 repetitions.

| n    | tiled                  | unrolled                | speedup | slower than OpenBLAS |
|------|------------------------|-------------------------|---------|----------------------|
| 256  | 0.932 ms (36.7 GFLOP/s)| 0.590 ms (58.2 GFLOP/s) | 1.58×   | ~1.9×                |
| 512  | 7.47 ms (36.5 GFLOP/s) | 5.33 ms (50.3 GFLOP/s)  | 1.40×   | ~2.2×                |
| 1024 | 64.8 ms (33.9 GFLOP/s) | 50.5 ms (43.1 GFLOP/s)  | 1.28×   | ~2.7×                |

The gain shrinks as `n` grows because the kernel becomes limited by streaming `B` and `C` through the cache rather than by `C` read-modify-writes. The next steps are tiling `j` as well and a register-blocked microkernel.
