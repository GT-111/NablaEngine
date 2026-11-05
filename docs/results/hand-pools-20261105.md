# Engine analysis for the hand-pool speed campaign (2026-11-05)

Scope: the five production hand models (NV 26 / 36 / 36 / 44 / 56, `MAXCON` 256, `NEFC` 1038-1112), four 5 ms native
substeps per decision, four ranks, ~410 worlds per rank and model, `iterations` 100, tolerance 1e-10. No GPU and no nvcc
were available for this note: the state sizes below come from the field definitions in `g1_engine/step.py` and the
generated model constants (`#define NB`, `NV`, `MAXCON`, `NEFC`, `KDOF`, `NPAIR`, `NGEOMC`) of the 2026-11-04 hand
sources; the register/spill numbers are the ptxas lines of those builds (`sm_90`, the production caps).

## Per-world state sizes and the split capacities

`SPLIT_NCON / SPLIT_NEFC` are the two-root defaults from `step.py`: 96 contacts / `NFLOSS + NEQ + NV + 4*96` rows.
A world beyond them runs the same solver on `struct SimNfull` inside `struct Hand` (global memory, the `big` path).

| model | NV | Sim (fused) | SimP (`k_pre`) | SimN (`k_post`) | Hand (work, per world) | SPLIT_NCON/NEFC | `k_post` blocks/SM (smem) |
|---|---|---|---|---|---|---|---|
| inspire_rh56f1 | 36 | 129.5 KB | 20.2 KB | 19.4 KB | 133.2 KB | 96 / 432 | 11 |
| inspire_rh56dftp | 36 | 129.5 KB | 20.2 KB | 19.4 KB | 133.2 KB | 96 / 432 | 11 |
| sharpa_wave | 56 | 155.3 KB | 27.8 KB | 35.1 KB | 159.8 KB | 96 / 484 | 6 |
| unitree_dex3_1 | 26 | 115.0 KB | 14.9 KB | 14.0 KB | 119.7 KB | 96 / 410 | 16 |
| wonik_allegro_v5 | 44 | 136.4 KB | 22.3 KB | 25.0 KB | 140.5 KB | 96 / 460 | 9 |
| mjlab G1 (reference) | 35 | 37.0 KB | 10.3 KB | 14.4 KB | 35.1 KB | 55 / 192 | 15 |

Consequences at 410 worlds on an H200 (132 SMs):

- The split forward is one resident wave for every hand model: `k_pre` 8-15 blocks/SM, `k_post` 6-16 blocks/SM by
  shared memory (16 by 128 registers), and 410 / 132 = 3.1 blocks per SM are needed. The sm_90 register caps of
  `DK_SM90_REG_CAPS` were tuned for 4096 worlds (2-4 waves); at B=410 they do not buy occupancy, they only cost the
  spills they force (`k_post` at `__maxnreg__(128)`: 250-330 B of spill stores per thread on the hand models).
- The fused forward is one block per SM for every hand model: `sizeof(Sim)` is 115-155 KB because `struct Sim` holds
  the full `MAXCON` 256 / `NEFC` ~1100 contact and row storage that the split `SimN` caps at 96/432-484 in shared
  memory. 410 worlds then run in four waves. This is the opposite of the small-batch case the split/fused choice was
  tuned for; `DK_FORWARD_SPLIT=0` is a switch to confirm the measurement, not an expected win.
- The per-call work buffer is 410 x Hand = 49-65 MB (now kept across calls by the `torch_ffi` change).

## Launch structure and overlap

- The training path (`dexforge_tracking/training/learners/physical_nabla._engine_step`) gives each hand pool its own
  CUDA stream and launches all five between `stream.wait_stream(caller)` and `caller.wait_stream(stream)`: the five
  hands overlap inside a substep, and the epoch ends with a barrier per substep. The 2026-11-04 four-rank B410 H2
  measurement of exactly this structure (`/tmp/a26059/plant-b410-analysis.json`) has serial 76.2 ms against parallel
  31.1 ms per rank, `all_equal_bits_serial_parallel: true`: the overlap already works and is worth 59%.
- Per substep and hand the split path is: one `cudaMemsetAsync` of the order counters, `k_pre`, `k_post`, plus a
  `k_kin` for live wrist feedback and, with the Torch interface's counters on, `dk_torch_diag_fwd` (and
  `dk_torch_diag_bwd` per backward). Nothing in the forward/backward path synchronizes or reads device status:
  `torch_ffi` only allocates and launches; the constants upload and `read_diagnostics` are the only synchronizations.
  DexForge's `torch._assert_async` is a device-side assert, not a D2H.
- At ~410 worlds a launch is one resident wave (`k_post` 6-16 blocks/SM by shared memory; 410 / 132 = 3.1 blocks per
  SM are needed for one pool), so its duration is the slowest world's phase and a 2-4 us launch tail is 1-2% of a
  `k_post` (0.99 ms measured). The launch count was the story of the old trace only because its failure-capture
  variant added a per-call `info.cpu()` and serialized the hand streams (the Lead's diagnosis); at this batch the cost
  is the per-iteration work inside the solver (measured: 3-5 Newton iterations, so ~0.2-0.3 ms per iteration at the
  low per-stream occupancy), not the launches.

## The 9edd Newton/Cholesky reuse, for these models

- `struct Sim`/`SimNT` reuse `H`'s storage for the refined reciprocal pivots (`h_inv`): `H` is dead after the
  factorization, and the tape stores the factor, not `H`; `DK_TAPE_VERIFY`'s re-assembly writes `H` again after the
  solve, when `h_inv` is no longer needed.
- The reuse guard (`zsame && !refact`, gated `NR == 2 && !CHOL_BCAST && !CHOL_COMPACT`) is active for every hand model:
  `NR` is 2 (bilateral), `CHOL_BCAST` is `32 < NV <= 35` (false for 26-56) and `CHOL_COMPACT` is 0 on Ampere+.
  `chol_resolve_reg_cached` repeats the factorization's forward/back substitution with the stored reciprocals and the
  same `L` it just read, so a reused iteration is **bitwise** a re-factorization's solve (unlike the older
  `NEWTON_REUSE` path, which the configuration table marks as differing by roundings).
- `zsame` is set from `row_piece` before/after each Newton step: the Hessian is `M + J' C J` with `C` from the rows'
  zones, so an unchanged set of zones means an unchanged H. A world whose last step changed a zone is never reused, and
  the tape path re-assembles at the final zones (`refact = zchanged`) before storing `L_H`. `DK_TAPE_VERIFY` is the
  check that would flag a violated guard (tape flag bit 11).
- The register cost of the reuse is the reason to re-tune the cap: `chol_solve_reg_save` holds `a0[NV] + a1[NV] +
  dinv[NV]` floats per lane, i.e. 78 / 108 / 132 / 168 registers for NV 26 / 36 / 44 / 56 before the rest of the
  kernel. 128 is below that for NV 36 and up, so the factorization spills; a resolve (the reuse path) needs only
  `dinv` (from shared) and `L`, so the value of the reuse is partly eaten by the spills it is compiled under.
- 9edd's guard is stricter than the Hessian needs. H = M + J' D J depends on a row only through its curvature `dd`,
  which is `D` exactly on the quadratic zone and 0 on the inactive zone and on both linear pieces of a frictionloss
  row (`row_piece` / `row_eval`); the contact rows and limits only have quadratic/inactive. Since the register-resolve
  path is bitwise a re-factorization's solve, testing "did a row enter or leave the quadratic zone" instead of "did
  its zone change" is bitwise-equivalent and reuses the factor across a frictionloss row's 2 <-> 3 flip (the tape's
  zone word records the quadratic rows only, so the record is unaffected except its REFACT diagnostic bit). The hand
  pools have 0 / 32 / 44 frictionloss dofs, so this is where the remaining re-factorizations are.

## What to measure, in order

1. `DK_DIAGNOSTICS` off (default now): one launch fewer per native forward and backward.
2. `DK_SPLIT_POST_MAXNREG` 128 (default) against 168 / 0 (or `DK_SPLIT_POST_MINB=12`) and `DK_SPLIT_PRE_MINB=16`
   against 0: at B=410 an uncapped `k_post` still fits one wave (8 blocks/SM at 255 registers), and it removes the
   250-330 B/thread of spills. Code-generation change: compare `--tape --save-outputs` bitwise.
3. `DK_SPLIT_ORDER=0`: at one wave the ordering has no makespan to recover, and its memset, per-world bucket atomics
   and per-launch warp scan remain. Scheduling only (host.py).
4. `DK_FORWARD_SPLIT=0` only to confirm the four-wave prediction above; expect a loss for all five hand models at B=410.
5. `DK_SPLIT_NCON` / `DK_SPLIT_NEFC` only if the worlds take the `big` path (a captured tape's `OFF_HDR` word is
   `ncon | nefc << 16`; `ncon > 96` or `nefc > SPLIT_NEFC` means global-memory state).

## Measured on the five-hand B410 H2 gate (2026-11-05)

Physical parity passes for the counters-off / work-buffer arm (`46252a2`, `ecd6287`, `9d9a2f7`) and for the uncapped
`k_post` arm (`2cdb47f`); the diagnostics counters were the only differences and are informational. Timings (10
rounds, median device ms, rank 0): base 29.122 against 28.991 (-0.45%) for the first, base 29.19 against 28.965
(-0.8%) for the uncapped caps, i.e. within the 0.2-3% base-to-base variance: the 128-register cap is not the
per-iteration cost in practice.

The solver facts from the retained per-substep `info` (rank 0, four substeps, five pools) are what this note did not
have: `nit` mean 2.4-4.7, max 11; `ncon` mean 1.0-11.3, max 17; `nefc` mean 7.2-79.5, max 104; no world takes the
`big` path, no refact and no non-converged tape record. So the 100-iteration budget is never exhausted (3-5 iterations
suffice), the `ZONE_CHANGED` relaxation has nothing to reuse here (`refact = 0`), and the two-root split capacity is
not a production issue for these five pools. `k_post`'s 0.987 ms per 410-world launch is therefore per-iteration
Hessian / Cholesky work at low per-stream occupancy: one pool is 410 blocks over 132 SMs, i.e. about 3.1 blocks per
SM (the five streams overlap only ~2.45x, not 5x, and the per-substep join caps the overlap). An instruction-level
win would need a `DK_PROFILE` phase breakdown; the structural widening (many more worlds per launch) is blocked by
the compile-time model constants (`NQ`, `NV`, `NB`, `MAXCON`, `KDOF` are `#define`s of each model's library): the five
pools cannot share one kernel launch, so the only way to fill the GPU is more stream overlap, which is a caller-side
change.
