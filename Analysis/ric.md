```cpp
void sgemm_ric(std::size_t M, std::size_t N, std::size_t L,
                 float alpha, const float* A, const float* B,
                 float beta, float* C) {
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j)
            C[i * N + j] *= beta;                   
        for (std::size_t k = 0; k < L; ++k) {
            const float a = alpha * A[i * L + k];
            for (std::size_t j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
    }
}
```

# GodBolt

with 'x86-64 clang18.1.0 -O3 -march=znver3'.

```asm
mk_ric(unsigned long, float, float const*, float*):
        test    rdi, rdi
        je      .LBB0_7
        cmp     rdi, 32
        jae     .LBB0_3
        xor     eax, eax
        jmp     .LBB0_6
.LBB0_3:
        vbroadcastss    ymm1, xmm0
        mov     rax, rdi
        and     rax, -32
        xor     ecx, ecx
.LBB0_4:
        vmovups ymm2, ymmword ptr [rsi + 4*rcx]
        vmovups ymm3, ymmword ptr [rsi + 4*rcx + 32]
        vmovups ymm4, ymmword ptr [rsi + 4*rcx + 64]
        vmovups ymm5, ymmword ptr [rsi + 4*rcx + 96]
        vfmadd213ps     ymm2, ymm1, ymmword ptr [rdx + 4*rcx]
        vfmadd213ps     ymm3, ymm1, ymmword ptr [rdx + 4*rcx + 32]
        vfmadd213ps     ymm4, ymm1, ymmword ptr [rdx + 4*rcx + 64]
        vfmadd213ps     ymm5, ymm1, ymmword ptr [rdx + 4*rcx + 96]
        vmovups ymmword ptr [rdx + 4*rcx], ymm2
        vmovups ymmword ptr [rdx + 4*rcx + 32], ymm3
        vmovups ymmword ptr [rdx + 4*rcx + 64], ymm4
        vmovups ymmword ptr [rdx + 4*rcx + 96], ymm5
        add     rcx, 32
        cmp     rax, rcx
        jne     .LBB0_4
        cmp     rax, rdi
        je      .LBB0_7
.LBB0_6:
        vmovss  xmm1, dword ptr [rsi + 4*rax]
        vfmadd213ss     xmm1, xmm0, dword ptr [rdx + 4*rax]
        vmovss  dword ptr [rdx + 4*rax], xmm1
        inc     rax
        cmp     rdi, rax
        jne     .LBB0_6
.LBB0_7:
        vzeroupper
        ret
```


Here we know have v-instrucitons ending in ps, meaning that they are utilising all 8 lanes (256-bits for zen3 fits 8 single precision floats). We know 8 is in use due to the presence of ymm*. So our upper bound on FMA throughput with now sits at 2 * 8 * 2 * 4GHz ~= 130 GFLOP/s.
Furthermore unlike the naive implementation there is no dependency between these 4 instructions and hence no latency bound arising from the previous sequential behaviour

In reality we arent limited by FMA throughput. Zen 3 can do 2 loads, 1 store and 2 FMAs per cycle.

Each iteration above contains 8 loads, 4 stores (from 4 FMAs) and 4 FMA, hence we are limited to 4 cycles per iteration. In one iteration we perform 8 * 4 * 2 = 64 flops. Which is 16 flops per cycle and hence an upper-limit of ~=64 GFLOP/s @ 4GHz.

Thus far we have not checked the limit from load latency (see llvm-mca below, which finds a tighter 6 cycle limit)


# llvm-mca: `mk_ric` inner loop (znver3)

Inner loop `.LBB0_4` of `src/mk_ric.cpp`, analysed with llvm-mca.

```bash
clang++ ./src/mk_ric.cpp -O3 -march=native -S -o - \
  | awk '/^\.LBB0_4:/{f=1;next} f{print} /jne[[:space:]]+\.LBB0_4/{exit}' \
  | llvm-mca -mcpu=znver3 -iterations=200
```

## Summary

| Metric | Value |
|---|---|
| Iterations | 200 |
| Instructions | 3000 (15 / iter) |
| Total cycles | 1210 (~6 / iter) |
| Total uOps | 3000 |
| Dispatch width | 6 |
| uOps per cycle | 2.48 |
| IPC | 2.48 |
| Block RThroughput | 6.0 |

Unlike naive there is no dependency chain, so here Block RThroughput is the relevant number: ~6 cycles per iteration, i.e. 64 flops / 6 = ~10.7 flops/cycle, ~43 GFLOP/s @ 4 GHz.
This is slower than the 4 cycles (64 GFLOP/s) estimated above from load, store and FMA counts alone.

## Resource pressure per iteration

| Resource | Cycles |
|---|---:|
| Zn3AGU0 / 1 / 2 | - |
| Zn3ALU0–3 | 0.60 each |
| Zn3BRU1 | 0.60 |
| Zn3FPP0 / FPP1 | 2.00 each |
| Zn3FPP2 / FPP3 | - |
| Zn3FPP45 (x2) | 6.00 each |
| Zn3FPSt | 4.00 |
| Zn3LSU (x3) | 4.00 each |
| Zn3Load (x3) | 2.67 each |
| Zn3Store (x2) | 2.00 each |

The bottleneck is `Zn3FPP45`: all 12 vector memory uOps (4 loads, 4 FMA memory operands, 4 stores) go through its 2 units, 12 / 2 = 6 cycles.
The FMA pipes (`FPP0/1`) are only busy 2 of 6 cycles, so the kernel is memory-op bound, not FMA bound.
Stores (`FPSt`, 4 cycles) and the LSUs (4 cycles) are close behind but not binding.


# Google Benchmark

| n    | Ric                     | OpenBLAS (1 thread)   | Ric slower than BLAS |
|------|-------------------------|-----------------------|----------------------|
| 256  | 0.896 ms (38.1 GFLOP/s) | 0.32 ms (108 GFLOP/s) | ~2.8×                |
| 512  | 8.08 ms (33.8 GFLOP/s)  | 2.43 ms (113 GFLOP/s) | ~3.3×                |
| 1024 | 68.7 ms (31.9 GFLOP/s)  | 19.0 ms (114 GFLOP/s) | ~3.6×                |


Here we observe that we approached near the calculated upper bound, falling short due to memory utilisation which will be seen in AMD uProf. The results above show we are now far close to the performance of OpenBlas, not just raw speed, but scaling-wise too with increasing N (although still not scaling as well). The table below indicates where these speedups arise.

| n    | scalar      | vectorised   | vectorisation speedup |
|------|-------------|--------------|-----------------------|
| 256  | 5.3 GFLOP/s | 38.1 GFLOP/s | ~7.2×                 |
| 512  | 5.3 GFLOP/s | 33.8 GFLOP/s | ~6.4×                 |
| 1024 | 5.3 GFLOP/s | 31.9 GFLOP/s | ~6.1×                 |

So the more contiguous memory access allowed by performing the outer-product yields better scaling, whilst most of the raw throughput arises from the vectorised instructions such as vfmadd213ps.

# Next-Step

There are still plenty of gains left on the table. Next we will introduce tiling, in particular 1D cahce tiling, which aims to make the most of each load into memory. Following this we will explore register tiling.
