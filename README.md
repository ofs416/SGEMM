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
