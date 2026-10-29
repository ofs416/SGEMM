```cpp
void sgemm_tile(std::size_t M, std::size_t N, std::size_t L,
                 float alpha, const float* A, const float* B,
                 float beta, float* C) {

        constexpr std::size_t KC = 32;   // B tile ≈ KC*N*4 bytes, aim for ≤ ~256 KB for ryzen 5500

        for (std::size_t i = 0; i < M * N; i++) {
            C[i] *= beta;
        }

        for (std::size_t kt = 0; kt < L; kt += KC) {
            std::size_t kend = std::min(L, kt + KC);
            for (std::size_t i = 0; i < M; i++) {
                for (std::size_t k = kt; k < kend; k++) {
                    const float a = alpha * A[i * L + k];
                    for (std::size_t j = 0; j < N; j++) {
                        C[i * N + j] += a * B[k * N + j];
                    }
                }
            }
        }
    }
```

# GodBolt and llvm-mca

Note that the inner loop is the same as ric, hence the compiled inner-loop is identical.

The same is also witnessed in the llvm-mca.

That means the we are still load limited and hence operating with an upperbound at ~43 GFLOP/s @ 4 GHz.

Looking at the the compiled code for mk_o_tile.cpp and mk_o_ric.cpp (ommitted due to size), we see the only difference is that ric unrolls K by a factor of 8, whilst tile unrolls I by a factor of 8. This explains the near identical performance, with the unroll across I proving ever so slightly more effective.

Both these unrolls can help direct the next step!

The tile version, involves M\*N+M\*N\*L C reads and writes. We reduce this to  M\*N + M\*N\*L/4 by unrolling the K loop by a factor of 4. Each loaded `C[j]` partakes in four FMAs instead of one. A scalar loop handles the remainder when the tile length isn't a multiple of 4.

'''cpp
void sgemm_tile_unroll(std::size_t M, std::size_t N, std::size_t L,
                 float alpha, const float* A, const float* B,
                 float beta, float* C) {

        constexpr std::size_t KC = 32;   // B tile ≈ KC*N*4 bytes, ≤ ~256 KB for ryzen 5500

        for (std::size_t i = 0; i < M * N; i++) {
            C[i] *= beta;
        }

        for (std::size_t kt = 0; kt < L; kt += KC) {
            std::size_t kend = std::min(L, kt + KC);
            for (std::size_t i = 0; i < M; i++) {
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
                    for (std::size_t j = 0; j < N; j++) {
                        C[i * N + j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
                    }
                }
                for (; k < kend; k++) {
                    const float a = alpha * A[i * L + k];
                    for (std::size_t j = 0; j < N; j++) {
                        C[i * N + j] += a * B[k * N + j];
                    }
                }
            }
        }
    }
'''

Now GodBolt and llvm-mca will produce something different!

# GodBolt

```asm
mk_tile_unroll(unsigned long, float, float, float, float, float const*, float const*, float const*, float const*, float*):
        test    rdi, rdi
        je      .LBB0_7
        cmp     rdi, 32
        jae     .LBB0_3
        xor     eax, eax
        jmp     .LBB0_6
.LBB0_3:
        vbroadcastss    ymm4, xmm1
        vbroadcastss    ymm5, xmm0
        vbroadcastss    ymm6, xmm2
        vbroadcastss    ymm7, xmm3
        mov     rax, rdi
        and     rax, -32
        xor     r10d, r10d
.LBB0_4:
        vmulps  ymm8, ymm4, ymmword ptr [rdx + 4*r10]
        vmulps  ymm9, ymm4, ymmword ptr [rdx + 4*r10 + 32]
        vmulps  ymm10, ymm4, ymmword ptr [rdx + 4*r10 + 64]
        vmulps  ymm11, ymm4, ymmword ptr [rdx + 4*r10 + 96]
        vfmadd231ps     ymm8, ymm5, ymmword ptr [rsi + 4*r10]
        vfmadd231ps     ymm9, ymm5, ymmword ptr [rsi + 4*r10 + 32]
        vfmadd231ps     ymm10, ymm5, ymmword ptr [rsi + 4*r10 + 64]
        vfmadd231ps     ymm11, ymm5, ymmword ptr [rsi + 4*r10 + 96]
        vfmadd231ps     ymm8, ymm6, ymmword ptr [rcx + 4*r10]
        vfmadd231ps     ymm9, ymm6, ymmword ptr [rcx + 4*r10 + 32]
        vfmadd231ps     ymm10, ymm6, ymmword ptr [rcx + 4*r10 + 64]
        vfmadd231ps     ymm11, ymm6, ymmword ptr [rcx + 4*r10 + 96]
        vfmadd231ps     ymm8, ymm7, ymmword ptr [r8 + 4*r10]
        vfmadd231ps     ymm9, ymm7, ymmword ptr [r8 + 4*r10 + 32]
        vfmadd231ps     ymm10, ymm7, ymmword ptr [r8 + 4*r10 + 64]
        vfmadd231ps     ymm11, ymm7, ymmword ptr [r8 + 4*r10 + 96]
        vaddps  ymm8, ymm8, ymmword ptr [r9 + 4*r10]
        vaddps  ymm9, ymm9, ymmword ptr [r9 + 4*r10 + 32]
        vaddps  ymm10, ymm10, ymmword ptr [r9 + 4*r10 + 64]
        vaddps  ymm11, ymm11, ymmword ptr [r9 + 4*r10 + 96]
        vmovups ymmword ptr [r9 + 4*r10], ymm8
        vmovups ymmword ptr [r9 + 4*r10 + 32], ymm9
        vmovups ymmword ptr [r9 + 4*r10 + 64], ymm10
        vmovups ymmword ptr [r9 + 4*r10 + 96], ymm11
        add     r10, 32
        cmp     rax, r10
        jne     .LBB0_4
        cmp     rax, rdi
        je      .LBB0_7
.LBB0_6:
        vmulss  xmm4, xmm1, dword ptr [rdx + 4*rax]
        vfmadd231ss     xmm4, xmm0, dword ptr [rsi + 4*rax]
        vfmadd231ss     xmm4, xmm2, dword ptr [rcx + 4*rax]
        vfmadd231ss     xmm4, xmm3, dword ptr [r8 + 4*rax]
        vaddss  xmm4, xmm4, dword ptr [r9 + 4*rax]
        vmovss  dword ptr [r9 + 4*rax], xmm4
        inc     rax
        cmp     rdi, rax
        jne     .LBB0_6
.LBB0_7:
        vzeroupper
        ret
```

So this version has 20 loads, 4 stores, 16 VFMA/VMUL and 4 VADD. Which respectively need 10, 4, 8, 2 cycles. Note the sequential dependency of the VADD on the VMUL is not limiting here.

So theoretically we could acheive, 256 FLOPs per iteration hency 25.6 FLOPS per cycle and therefore ~102 GFLOP/s. A considerable potential performance uplift.

# llvm-mca

```bash
clang++ ./src/sgemm_tile_unroll.cpp -O3 -march=native -S -o - \
  | awk '/^\.LBB0_23:/{f=1;next} f{print} /jne[[:space:]]+\.LBB0_23/{exit}' \
  | llvm-mca -mcpu=znver3 -iterations=200
```

## Summary

| Metric | Value |
|---|---|
| Iterations | 200 |
| Instructions | 6000 (30 / iter) |
| Total cycles | 2418 (~12 / iter) |
| Total uOps | 6200 |
| Dispatch width | 6 |
| uOps per cycle | 2.56 |
| IPC | 2.48 |
| Block RThroughput | 12.0 |

llvm-mca gives ~12 cycles per iteration, so 256 / 12 = ~21.3 flops/cycle, yielding an upper limit of ~85 GFLOP/s @ 4 GHz.
That is 2x the ric/tile kernel (~43 GFLOP/s), but below the 25.6 flops/cycle (~102 GFLOP/s) estimated above from the 2 loads/cycle limit (10 cycles).

## Resource pressure per iteration

| Resource | Cycles |
|---|---:|
| Zn3AGU0 / 1 / 2 | 1.00 each |
| Zn3ALU0–3 | 0.80 each |
| Zn3BRU1 | 0.80 |
| Zn3FPP0 / FPP1 | 8.00 each |
| Zn3FPP2 / FPP3 | 2.00 each |
| Zn3FPP45 (x2) | 12.00 each |
| Zn3FPSt | 4.00 |
| Zn3LSU (x3) | 8.33 each |
| Zn3Load (x3) | 7.00 each |
| Zn3Store (x2) | 2.00 each |

The bottleneck is again `Zn3FPP45`: all 24 vector memory uOps (20 loads, 4 stores) go through its 2 units, 24 / 2 = 12 cycles.
The FMA pipes (`FPP0/1`) are busy 8 of 12 cycles, so the kernel is still memory-op bound, but much closer to FMA bound than ric (2 of 6).

# Google Benchmark

| n    | tiled                  | unrolled                | speedup | slower than OpenBLAS |
|------|------------------------|-------------------------|---------|----------------------|
| 256  | 0.967 ms (36.1 GFLOP/s)  | 0.586 ms (59.9 GFLOP/s) | 1.65×   | ~1.8×                |
| 512  | 7.50 ms (37.0 GFLOP/s)   | 5.50 ms (50.5 GFLOP/s)  | 1.36×   | ~2.2×                |
| 1024 | 64.0 ms (34.8 GFLOP/s)   | 49.6 ms (44.8 GFLOP/s)  | 1.29×   | ~2.6×                |
| 2048 | 574 ms (31.0 GFLOP/s)    | 487 ms (36.6 GFLOP/s)   | 1.18×   | ~3.2×                |
| 4096 | 5637 ms (25.3 GFLOP/s)   | 3747 ms (37.9 GFLOP/s)  | 1.50×   | ~3.1×                |

Here we observe the tiling across both I and K leads to a respectable speed up. So tiling and reusing the data already in the cache, followed by an unroll which decrease mem-ops per iteration, has yielded the results we expected, altough we fall short of the upper limit we calculated, the cause of which will likely be found with AMD uProf.

# Next Step

So far we have only tiled across one dimension, which increases the number of times B is used once loaded into cache. Next we will extend this to 2-dimensional tiling; tiling across not just J, but also across I. Meaning accessing C is more efficient as the C tile stays in cache. This can then be extended to tiling across J, I and K, allowing more efficient access of A and B at the expense of C and is only worthwhile when N is large. 

The next major performance boost will come from register blocking. Tiling focuses on reducing traffic to slow memory by ensuring as many loads come from L1, whilst register blocking focuses on allowing it to consume L1 as efficiently as possible.
