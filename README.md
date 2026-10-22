# SGEMM

## Building, testing and benchmarking

Requires CMake 3.20+, a C++20 compiler and OpenBLAS (the benchmark reference). GoogleTest and Google Benchmark are fetched automatically.

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

To benchmark without compiler flags, configure a separate build directory, e.g. `cmake -S . -B build-O0 -DCMAKE_BUILD_TYPE=None`, and edit the `-march=native` option in `CMakeLists.txt`.

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
| 256  | 51.0 ms (0.67 GFLOP/s) | 0.32 ms (108 GFLOP/s) | ~160×           |
| 512  | 413 ms (0.66 GFLOP/s)  | 2.41 ms (114 GFLOP/s) | ~170×           |
| 1024 | 4068 ms (0.54 GFLOP/s) | 19.4 ms (113 GFLOP/s) | ~210×           |

### 1.2 `-O3 -march=native`

| n    | naive                  | OpenBLAS (1 thread)   | naive slower by | flag speedup |
|------|------------------------|-----------------------|-----------------|--------------|
| 256  | 22.2 ms (1.5 GFLOP/s)  | 0.32 ms (108 GFLOP/s) | ~70×            | 2.3×         |
| 512  | 275 ms (1.0 GFLOP/s)   | 2.41 ms (114 GFLOP/s) | ~115×           | 1.5×         |
| 1024 | 4028 ms (0.54 GFLOP/s) | 19.4 ms (113 GFLOP/s) | ~210×           | 1.0×         |

OpenBLAS is a prebuilt library, so its numbers don't depend on these flags.


### 1.3 Naive across compiler flags

| n    | `-O0`                  | `-O3`          | `-O3 -march=native` | `-O0` → `-O3` speedup |
|------|------------------------|----------------|---------------------|-----------------------|
| 256  | 51.0 ms (0.67 GFLOP/s) | 21.2 ms (1.61) | 22.1 ms (1.55)      | 2.4×                  |
| 512  | 413 ms (0.66)          | 274 ms (1.00)  | 279 ms (0.98)       | 1.5×                  |
| 1024 | 4068 ms (0.54)         | 4094 ms (0.54) | 4130 ms (0.53)      | 1.0×                  |


## 2. Loop reorder (`sgemm_blocked`)

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

This only reorders the loops and does no tiling, so the name `blocked` is a placeholder for the tiled version to come. Rows of `B` and `C` are still streamed in full for every `i`, so once a row no longer fits in cache the speed drops with `n` (28.7 to 26.2 GFLOP/s below).

### 2.1 `-O3 -march=native`

| n    | reordered              | OpenBLAS (1 thread)   | slower than OpenBLAS | speedup over naive |
|------|------------------------|-----------------------|----------------------|--------------------|
| 256  | 1.16 ms (28.7 GFLOP/s) | 0.32 ms (108 GFLOP/s) | ~3.6×                | ~19×               |
| 512  | 9.51 ms (28.2 GFLOP/s) | 2.41 ms (114 GFLOP/s) | ~3.9×                | ~29×               |
| 1024 | 82.6 ms (26.2 GFLOP/s) | 19.4 ms (113 GFLOP/s) | ~4.3×                | ~49×               |

The gap was closed considerably with these cache hits. Furthermore it now scales far better with n. The scaling is improved by the decrease in cache-hits, whilst the rest of the speed up is actually from thge compilers ability to now vectorise this.


### 2.2 Effect of vectorisation

Same kernel built with and without auto-vectorisation (`-fno-tree-vectorize -fno-tree-slp-vectorize`), single thread.

| n    | scalar      | vectorised   | vectorisation speedup |
|------|-------------|--------------|-----------------------|
| 256  | 6.8 GFLOP/s | 29.4 GFLOP/s | ~4.3×                 |
| 512  | 6.4 GFLOP/s | 28.5 GFLOP/s | ~4.4×                 |
| 1024 | 5.9 GFLOP/s | 26.4 GFLOP/s | ~4.4×                 |

