#include <cstddef>

// Inner loop of sgemm_ric: one row of C += a * one row of B.
void mk_ric(std::size_t N, float a, const float* __restrict B, float* __restrict C) {
    for (std::size_t j = 0; j < N; ++j)
        C[j] += a * B[j];
}


// clang++ src/mk_ric.cpp -O3 -march=native -ffast-math -S -o - \
  | awk '/^\.LBB0_4:/{f=1;next} f{print} /jne[[:space:]]+\.LBB0_4/{exit}' \
  | llvm-mca -mcpu=znver3 -iterations=200

// Iterations:        200                                   // simulated the loop  200 times
// Instructions:      3000                                   // 15 instructions x 200
// Total Cycles:      1210                                      // 1210 / 200 = ~6.05 cycles per loop pass  <-- the headline number
// Total uOps:        3000                                   // 1 micro-op per instruction (all simple)
//
// Dispatch Width:    6                                         // CPU front end can issue up to 6 uops/cycle
// uOps Per Cycle:    2.48                                   // only using 2.48 of 6, so the front end is NOT the bottleneck
// IPC:               2.48                                   // same thing here (1 uop per instruction)
// Block RThroughput: 6.0                                    // best-case cycles per pass given the busiest unit (see table below)
//                                                          // 64 flops / 6 cycles = ~10.7 flops/cycle
//
// Instruction Info:                                       // [1]=uops [2]=latency (cycles to result) [3]=reciprocal throughput
//
//  1  8   0.50  *   vmovups (%rsi,%rcx,4), %ymm2               // load 8 floats of B   (latency 8, can start 2/cycle)
//  1  8   0.50  *   vmovups 32(%rsi,%rcx,4), %ymm3             // load next 8 floats of B
//  1  8   0.50  *   vmovups 64(%rsi,%rcx,4), %ymm4             // ...
//  1  8   0.50  *   vmovups 96(%rsi,%rcx,4), %ymm5             // 4 loads x 8 floats = 32 elements per pass
//
//  1  11  0.50  *   vfmadd213ps (%rdx,%rcx,4), %ymm1, %ymm2   // ymm2 = B*a + C[0..7]   (C is loaded here
//  1  11  0.50  *   vfmadd213ps 32(%rdx,%rcx,4), ...ymm3      //   as a memory operand, which is why the
//  1  11  0.50  *   vfmadd213ps 64(%rdx,%rcx,4), ...ymm4      //   FMA is marked "MayLoad" (*) and has
//  1  11  0.50  *   vfmadd213ps 96(%rdx,%rcx,4), ...ymm5      //   latency 11 = 8 load + ~3-4 FMA)
//                                                             // 4 FMAs x 8 lanes x 2 flops = 64 flops/pass
//
//  1  1   1.00     * vmovups %ymm2, (%rdx,%rcx,4)              // store result back to C   (marked "MayStore")
//  1  1   1.00     * vmovups %ymm3, 32(%rdx,%rcx,4)         // each store: throughput 1.00 = one per cycle
//  1  1   1.00     * vmovups %ymm4, 64(%rdx,%rcx,4)             //   => 4 stores need at least 4 cycles
//  1  1   1.00     * vmovups %ymm5, 96(%rdx,%rcx,4)
//
//  1  1   0.25     addq $32, %rcx                        // index += 32 elements (cheap: any of 4 integer ALUs)
//  1  1   0.25     cmpq %rcx, %rax                       // reached the end?
//  1  1   0.50     jne  .LBB0_4                         // no -> loop again (branch unit)
//
// Resources:                                            // the hardware units the model knows about
// [0-2]   AGU          address generation (computes the memory addresses)
// [3-6]   ALU          integer math (add, cmp)
// [7]     BRU          branch unit
// [8-11]  FPP0-3       floating-point pipes. FPP0/FPP1 are where FMAs run
// [12.x]  FPP45        two FP pipes the model uses for vector loads/stores data
// [13]    FPSt         FP store-data unit
// [14.x]  LSU          load/store unit (3 of them: up to 3 memory ops/cycle)
// [15.x]  Load         load pipes
// [16.x]  Store        store pipes
//
// Resource pressure per iteration:                      // cycles of work per loop pass landing on each unit.
//                                                   // THE LARGEST NUMBER IS THE BOTTLENECK.
//
// [0-2]  AGU    -                                // not shown as busy
// [3-6]  ALU    0.60 each                             // integer work is trivial
// [7]    BRU    0.60                              // one branch, trivial
// [8,9]  FPP0/1 2.00 each                          // the FMA pipes: 4 FMAs over 2 pipes = 2 cycles. Only ~1/3 busy
//                                                //   => the compute hardware is mostly idle
// [12.0] FPP45  6.00                              // <-- BOTTLENECK. 12 uops (4 loads + 4 FMAs + 4 stor
// [12.1] FPP45  6.00                             //     spread over 2 units = 6 cycles. This sets Block RThroughput = 6.0
// [13]   FPSt   4.00                             // 4 stores, one per cycle = 4 cycles (my earlier prediction; not binding)
// [14.x] LSU    4.00 each                             // 12 memory ops / 3 units = 4 cycles (also not bindi
// [15.x] Load   2.67 each                         // 8 loads / 3 units
// [16.x] Store  2.00 each                         // 4 stores / 2 units
//
// Resource pressure by instruction:                            // same table, per instruction. Use it to see WHO loads each unit.
//
// vmovups (load B)   -> FPP45 ~1.0, LSU ~1.0, Load ~1.0     // each load uses one FPP45 slot + a load pipe
// vfmadd213ps        -> FPP0 or FPP1 (1.0), FPP45 ~1.0, LSU, Load   // FMA + its memory operand load
// vmovups (store C)  -> FPP45 ~1.0, FPSt 1.0, LSU, Store            // each store also eats
// addq/cmpq/jne      -> ALU / BRU only                              // never touch the busy units
