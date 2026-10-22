# SGEMM

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
