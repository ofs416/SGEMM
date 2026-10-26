```cpp
void sgemm_naive(std::size_t M, std::size_t N, std::size_t L,
                 float alpha, const float* A, const float* B,
                 float beta, float* C) {
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float acc = 0.0f;                       
            for (std::size_t k = 0; k < L; ++k)
                acc += A[i * L + k] * B[k * N + j];
            C[i * N + j] = alpha * acc + beta * C[i * N + j];
        }
    }
}
```

# GodBolt

with 'x86-64 clang18.1.0 -O3 -march=znver3'.

```asm
mk_naive(unsigned long, unsigned long, float const*, float const*):
        test    rdi, rdi
        je      .LBB0_1
        push    r15
        push    r14
        push    rbx
        mov     eax, edi
        and     eax, 7
        cmp     rdi, 8
        jae     .LBB0_9
        vxorps  xmm0, xmm0, xmm0
        xor     r8d, r8d
        jmp     .LBB0_5
.LBB0_1:
        vxorps  xmm0, xmm0, xmm0
        ret
.LBB0_9:
        lea     r10, [4*rsi]
        mov     r9, rsi
        and     rdi, -8
        shl     r9, 5
        vxorps  xmm0, xmm0, xmm0
        xor     r8d, r8d
        mov     rbx, rcx
        lea     r11, [r10 + r10]
.LBB0_10:
        vmovss  xmm1, dword ptr [rdx + 4*r8]
        vmovss  xmm2, dword ptr [rdx + 4*r8 + 4]
        lea     r14, [rbx + r10]
        vfmadd132ss     xmm1, xmm0, dword ptr [rbx]
        vmovss  xmm0, dword ptr [rdx + 4*r8 + 8]
        add     r14, r10
        vfmadd231ss     xmm1, xmm2, dword ptr [rbx + 4*rsi]
        vmovss  xmm2, dword ptr [rdx + 4*r8 + 24]
        vfmadd132ss     xmm0, xmm1, dword ptr [rbx + 8*rsi]
        vmovss  xmm1, dword ptr [rdx + 4*r8 + 12]
        add     rbx, r9
        vfmadd132ss     xmm1, xmm0, dword ptr [r14 + 4*rsi]
        vmovss  xmm0, dword ptr [rdx + 4*r8 + 16]
        vfmadd132ss     xmm0, xmm1, dword ptr [r14 + 8*rsi]
        add     r14, r11
        vmovss  xmm1, dword ptr [rdx + 4*r8 + 20]
        lea     r15, [r14 + 4*rsi]
        vfmadd132ss     xmm1, xmm0, dword ptr [r14 + 4*rsi]
        vmovss  xmm0, dword ptr [rdx + 4*r8 + 28]
        add     r8, 8
        vfmadd132ss     xmm2, xmm1, dword ptr [r10 + r15]
        add     r15, r10
        vfmadd132ss     xmm0, xmm2, dword ptr [r10 + r15]
        cmp     rdi, r8
        jne     .LBB0_10
.LBB0_5:
        test    rax, rax
        je      .LBB0_8
        lea     rdx, [rdx + 4*r8]
        imul    r8, rsi
        shl     rsi, 2
        xor     edi, edi
        lea     rcx, [rcx + 4*r8]
.LBB0_7:
        vmovss  xmm1, dword ptr [rdx + 4*rdi]
        inc     rdi
        vfmadd231ss     xmm0, xmm1, dword ptr [rcx]
        add     rcx, rsi
        cmp     rax, rdi
        jne     .LBB0_7
.LBB0_8:
        pop     rbx
        pop     r14
        pop     r15
        ret
```


Here the limiting factor is the 8 sequential FMAs from the loop unrolled by a factor of 8. Each takes 4 cycles to complete so each innerloop takes 32 cycles.

With each loop performing 8 * 2 = 16 flops

So that's 0.5 flops per cycle

Assuming that l1 cache is fully utilised. at ~ 4 GHz that's 2 GFLOP/s per chain of FMAs.

 Each `C[i][j]` is its own dot product, and the next one starts with `vxorps` on a renamed register. It executes while the previous chain is still waiting on its 4-cycle FMAs. This is still scalar, one float per FMA, just several FMAs in flight at once.

 An FMA unit is pipelined: latency is 4 cycles, but it can accept a new independent FMA every cycle (`s1`–`s4` are the pipeline stages):

| Cycle   | 1  | 2  | 3  | 4  | 5  | 6  | 7  |
|---------|----|----|----|----|----|----|----|
| FMA #1  | s1 | s2 | s3 | s4 |    |    |    |
| FMA #2  |    | s1 | s2 | s3 | s4 |    |    |
| FMA #3  |    |    | s1 | s2 | s3 | s4 |    |
| FMA #4  |    |    |    | s1 | s2 | s3 | s4 |

From cycle 4 the pipeline is full: one FMA completes and one starts every cycle. A dependent FMA cannot enter until the previous one finishes `s4`, which is why a single chain only manages 1 FMA per 4 cycles.

We know zen3 can start 2 FMA/cycle (16 GFLOP/s), therefore 8 independent chains could be in flight at once. Load limits of 3 loads/cycle keeps us down to 1.5 FMA/cycle (12 GFLOP/s) from the 2 loads needed per FMA. Lastly, limits on the reorder buff mean we can only have 4 chains in flight at once, hence 1 FMA/cycle (8 GFLOP/s).

# Google Benchmark

| n    | Naive `-O0`            | Naive `-O3 -march=native` | OpenBLAS (1 thread)   | Naive slower by |
|------|------------------------|---------------------------|-----------------------|-----------------|
| 256  | 50.2 ms (0.68 GFLOP/s) | 35.7 ms (0.96 GFLOP/s)    | 0.32 ms (108 GFLOP/s) | ~112×           |
| 512  | 410 ms (0.67 GFLOP/s)  | 351 ms (0.78 GFLOP/s)     | 2.43 ms (113 GFLOP/s) | ~144×           |
| 1024 | 4004 ms (0.54 GFLOP/s) | 4070 ms (0.54 GFLOP/s)    | 19.0 ms (114 GFLOP/s) | ~214×           |

 Comparing this to the Google Benchmark results, we observe we only hit a fraction of this. Indicating that we are poorly utilising the cache.

With small N ~= 16, we almost reach this upper-limit, then as we increase N the number of chains is limited by the reorder buffer, which allows lookahead and early starts on instructions. Eventually the number of chains in flight hits 1 (2 GFLOP/s upper-limit), with cache misses restricting us further.

Note that as we scale N, performance further degrades unlike OpenBlas, indicating that we are poorly utilising the caches.


# llvm-mca: `mk_naive` inner loop (znver3)

Inner loop `.LBB0_13` of `src/mk_naive.cpp`, analysed with llvm-mca.

```bash
clang++ ./src/mk_naive.cpp -O3 -march=native -S -o - \
  | awk '/^\.LBB0_13:/{f=1;next} f{print} /jne[[:space:]]+\.LBB0_13/{exit}' \
  | llvm-mca -mcpu=znver3 -iterations=200
```

## Summary

| Metric | Value |
|---|---|
| Iterations | 200 |
| Instructions | 5000 (25 / iter) |
| Total cycles | 6411 (~32 / iter) |         
| Total uOps | 5200 |
| Dispatch width | 6 |
| uOps per cycle | 0.81 |
| IPC | 0.78 |
| Block RThroughput | 8.0 |

Here we see that ~32 cycles per iteration agree with our calculation.
Note we are not resource bound so the Block RThroughput is not the relevant number of cycle per iteration, as it assumes each instruction is independent.

## Resource pressure per iteration

| Resource | Cycles |
|---|---:|
| Zn3AGU0 / 1 / 2 | 0.33 / 0.34 / 0.34 |
| Zn3ALU0–3 | 1.75 each |
| Zn3BRU1 | 1.00 |
| Zn3FPP0 / FPP1 | 4.00 each |
| Zn3FPP2 / FPP3 | - |
| Zn3FPP45 (x2) | 8.00 each |
| Zn3FPSt | - |
| Zn3LSU (x3) | 5.33 / 5.34 / 5.34 |
| Zn3Load (x3) | 5.33 / 5.34 / 5.34 |
| Zn3Store (x2) | - |

# Next-Step

Currently we are poorly utilising caches and relying on sequential FMA which only use 1 of the 8 available lanes at a time. In the next iteration RIC, we will perform an optimisation which enable vectorisation and better utilises the cache.
