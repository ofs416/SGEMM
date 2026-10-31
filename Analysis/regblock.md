# Register Blocking — Outline

## Recap / Motivation
- Where the tiled kernels stand: tiling moved loads from DRAM into cache, but the core still issues the same instructions.
- The inner loop is still bound by `Zn3FPP45`: 24 vector memory uOps per 16 FMAs, ~12 cycles/iter → ~85 GFLOP/s ceiling.
- Root cause: every FMA result goes back to memory (C is loaded and stored per k-group), so loads/stores scale with FLOPs.
- Idea: keep a small block of C in ymm registers for the whole K loop, so each load feeds many FMAs and C touches memory only once per block.

## The Micro-Kernel Idea
- Compute an MR×NR block of C as a sum of KC rank-1 updates (outer products).
- Per k step:
  - load NR/8 ymm vectors from a row of B,
  - broadcast MR scalars from a column of A,
  - do MR·NR/8 FMAs into accumulator registers.
- C is loaded once before the K loop and stored once after it.
- Diagram: MR×NR accumulator grid, B row across the top, A column down the side.

## Choosing MR × NR
- Constraints on Zen 3 (AVX2):
  - **Register budget:** 16 ymm registers → MR·NR/8 accumulators + NR/8 for B + 1 for the broadcast ≤ 16.
  - **Latency hiding:** FMA latency 4 cycles × 2 FMA pipes → need ≥ 8 independent accumulators.
  - **Memory uOps vs FMAs:** (NR/8 + MR) memory uOps over 2 FPP45 units vs MR·NR/8 FMAs over 2 FMA pipes.
- Candidate table (per k step):

| MR×NR | accumulators | regs used | mem uOps | FMAs | mem cycles | FMA cycles | bound |
|---|---|---|---|---|---|---|---|
| 4×16 | 8 | 11 | 6 | 8 | 3 | 4 | FMA |
| 8×8 | 8 | 10 | 9 | 8 | 4.5 | 4 | memory |
| 6×16 | 12 | 15 | 8 | 12 | 4 | 6 | FMA |
| 4×24 | 12 | 16 | 7 | 12 | 3.5 | 6 | FMA |

- Pick 6×16 (BLIS's choice for Haswell/Zen): FMA-bound with slack, one register spare.
- Check: does `vbroadcastss` from memory go through FPP45 on Zen 3? Confirm in llvm-mca.

## The Kernel
- Code block: `sgemm_microkernel_6x16(...)` with AVX2 intrinsics (`_mm256_loadu_ps`, `_mm256_broadcast_ss`, `_mm256_fmadd_ps`).
- Why intrinsics: the compiler won't reliably keep 12 accumulators in registers from plain C++.
- Where alpha goes: scale once at the end (12 multiplies) instead of per A element.
- Edge handling: M % 6 and N % 16 remainders — padded tile vs masked loads/stores (`_mm256_maskload_ps`) vs scalar fallback.
- Driver loop: tile loops from the 2D/3D version, then `jr` (NR) and `ir` (MR) loops calling the micro-kernel.

## Packing
- Problem: the micro-kernel reads a column of A (stride L) and rows of B (stride N) → strided access, TLB misses, cache-set conflicts at power-of-two n.
- Pack A's MC×KC block into MR-wide row panels, stored k-major.
- Pack B's KC×NC block into NR-wide column panels, stored k-major.
- Result: the micro-kernel reads both operands with unit stride.
- Cost of packing vs reuse: O(MC·KC + KC·NC) copies amortised over O(MC·NC·KC) FLOPs.
- Benchmark with and without packing to show the effect.

## GodBolt
- Show the micro-kernel inner loop: expect 2 loads, 6 broadcasts, 12 FMAs, plus loop overhead, no stores.
- Compare with `sgemm_tile_unroll`: 20 loads + 4 stores per 16 FMAs → 8 memory uOps per 12 FMAs.
- Check that no accumulators spill to the stack.

## llvm-mca
- Same command, targeting the micro-kernel loop label.
- Expect the bottleneck to move from `Zn3FPP45` to `Zn3FPP0/1` (the FMA pipes).
- Expected ~6 cycles/iter → 192 FLOPs / 6 = 32 FLOPs/cycle → ~128 GFLOP/s @ 4 GHz (the per-core peak for AVX2 FMA on Zen 3).
- Resource pressure table for comparison with the unrolled version.

## Google Benchmark
| n | 2D tiled | register blocked | + packing | speedup | slower than OpenBLAS |
|---|---|---|---|---|---|
| 256 | | | | | |
| 512 | | | | | |
| 1024 | | | | | |
| 2048 | | | | | |
| 4096 | | | | | |
- Expectation: big jump at every n, since this changes instruction mix, not just cache behaviour.
- Fraction of single-core peak reached.

## AMD uProf
- FMA pipe utilisation: how close to 2 FMAs/cycle?
- L1 hit rate for packed panels, L2/L3 misses during packing.
- Remaining stalls: front end, branch, or memory.

## Tuning Block Sizes
- With MR×NR fixed, retune KC (A panel + B panel in L1), MC (packed A in L2), NC (packed B in L3).
- Sweep table: KC × MC → GFLOP/s at n=1024 and n=4096.

## Next Step
- Software prefetching of the next A/B panels and of C before the store.
- Unrolling the k loop inside the micro-kernel to cut loop overhead.
- Multithreading across the 6 cores (parallelise the `ic` or `jr` loop), and how shared L3 changes NC.
- Final comparison against OpenBLAS.
- 
