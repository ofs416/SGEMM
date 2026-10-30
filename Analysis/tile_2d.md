# 2D Tiling (I and K) — Outline

## Recap / Motivation
- Where `sgemm_tile_unroll` stands: 59.9 GFLOP/s at n=256, falling to ~37 GFLOP/s at n≥2048.
- Why it falls off: for every K tile (`kt`), the whole of C (M×N) is swept once → L/KC full passes over C.
  - At n=4096, C is 64 MB, far beyond L3, so every pass streams C from DRAM.
- Also note: B tile = KC·N·4 bytes = 512 KB at n=4096 → already over the ~256 KB target from the 1D version.
- Idea: tile across I as well, so a block of C rows stays in cache across all K tiles.

## The Kernel
- New block size `MC` (rows of C per tile).
- Loop order: `it` (I tiles) → `kt` (K tiles) → `i` → `k` (unrolled ×4, scalar remainder) → `j`.
- Code block: `sgemm_tile2d(...)` — same inner body as `sgemm_tile_unroll`.
- Brief explanation of what stays resident where:
  - C tile: MC·N·4 bytes, reused across all L/KC K tiles.
  - B tile: KC·N·4 bytes, reused across MC rows, now re-read M/MC times overall.
  - A: MC×KC sub-block, small.

## Choosing Tile Sizes
- Ryzen 5 5500 cache table: L1D 32 KB, L2 512 KB per core, L3 16 MB shared.
- Constraint: C tile + B tile should fit in L2 (or C tile in L2, B tile in L3).
- The tension: both tiles scale with N → at large N, MC and KC are forced small. (Sets up 3D tiling.)
- Small sweep table: MC × KC → GFLOP/s at n=1024 and n=4096.

## Memory Traffic Analysis
- Count reads/writes of A, B and C for: naive → 1D tile → 1D tile + unroll → 2D tile.
- Show the C traffic drop (from L/KC passes over C to 1 pass per I tile) and the B traffic increase (M/MC reloads).
- Estimate arithmetic intensity (FLOPs per byte from DRAM) for each.

## GodBolt
- Inner loop expected to be identical to `sgemm_tile_unroll` (same 30-instruction block: 20 loads, 4 stores, 16 VFMA/VMUL, 4 VADD).
- Point: 2D tiling changes *where* loads come from, not *how many* the core issues.
- Mention any differences in the outer loops (e.g. which loop clang unrolls).

## llvm-mca
- Same command, confirm same summary (~12 cycles/iter, Zn3FPP45 bottleneck).
- So the throughput ceiling is unchanged: ~85 GFLOP/s @ 4 GHz.
- llvm-mca assumes all loads hit L1, so it can't show the benefit of tiling — that has to come from the benchmark and profiler.

## Google Benchmark
| n | unrolled | 2D tiled | speedup | slower than OpenBLAS |
|---|---|---|---|---|
| 256 | | | | |
| 512 | | | | |
| 1024 | | | | |
| 2048 | | | | |
| 4096 | | | | |
- Expectation: little change at small n (already cache-resident), biggest gain at n≥2048.
- Does performance now stay flat as n grows?

## AMD uProf
- Follow up on the 1D shortfall (~37 vs ~85 GFLOP/s ceiling).
- Compare unrolled vs 2D: L1/L2/L3 miss rates, DRAM bandwidth, cycles stalled on memory.
- Confirm C traffic actually moved from DRAM into L2.

## Next Step
- 3D tiling: add `NC` (J tiles) so B tile is KC×NC and C tile is MC×NC, both independent of N — matters when N is large.
- Packing A and B into contiguous buffers to avoid TLB misses and cache-set conflicts.
- Register blocking: a micro-kernel holding a small block of C in ymm registers, so loads from L1 are amortised over many FMAs and the Zn3FPP45 bottleneck finally gives way to FMA throughput.

