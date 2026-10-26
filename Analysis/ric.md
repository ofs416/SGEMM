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

Thus far we have not checked the limit from load latency


# Google Benchmark

┌──────┬─────────────────────────┬───────────────────────┬──────────────────────┐
│  n   │           Ric           │  OpenBLAS (1 thread)  │ Ric slower than BLAS │
├──────┼─────────────────────────┼───────────────────────┼──────────────────────┤
│  256 │ 0.896 ms (38.1 GFLOP/s) │ 0.32 ms (108 GFLOP/s) │                ~2.8× │
├──────┼─────────────────────────┼───────────────────────┼──────────────────────┤
│  512 │  8.08 ms (33.8 GFLOP/s) │ 2.43 ms (113 GFLOP/s) │                ~3.3× │
├──────┼─────────────────────────┼───────────────────────┼──────────────────────┤
│ 1024 │  68.7 ms (31.9 GFLOP/s) │ 19.0 ms (114 GFLOP/s) │                ~3.6× │
└──────┴─────────────────────────┴───────────────────────┴──────────────────────┘
