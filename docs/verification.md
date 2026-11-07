# Verification

The engine is checked against MuJoCo Warp 3.17.0 (the PyPI release), MuJoCo C in float64 and MJX, on public corpora,
with committed references so that the checks run without MuJoCo Warp installed. `scripts/verify.sh` runs the set and
prints one PASS / FAIL line per check; every check is also a script of its own that writes a JSON result (with a
`provenance` block: commit, GPU, driver, nvcc, package versions) and exits with status 1 when a gate fails.

```bash
scripts/verify.sh --quick [OUTDIR]   # a few minutes on one GPU: env, one step, one gradient group, FFI bitwise, capability
scripts/verify.sh [OUTDIR]           # about 15 minutes on a V100: every check on mf_s0 and mf_s1
```

`OUTDIR` defaults to `runs/verify`; each check leaves `<name>.log` and `<name>.json` there.

## Data

| data | file | what it is |
|---|---|---|
| the model | `fixtures/mjlab/g1_train.mjb`, `g1_meta.json` | the mjlab Unitree G1 (29 actuated hinges, a free root, 14 sole capsules, explicit contact pairs; `fixtures/mjlab/README.md`) in MuJoCo 3.17.0's binary format, and its metadata (foot-floor pairs, torso body, sizes and options) |
| corpora | `benchmarks/data/mf_s0.npz` .. `mf_s3.npz` | 4096 states each from MuJoCo C rollouts of the G1 under random joint set points and pushes (`benchmarks/bench/corpus/gen_mf.py`), with per-state friction and torso COM offset, three stored output cotangents per state and labels; `menagerie_s0.npz`: states of MuJoCo Menagerie's own G1 (the benchmark's track B) |
| float64 truth | `benchmarks/data/truth/<corpus>.npz` | MuJoCo C float64 at tight solver options (2000 Newton iterations, tolerance 1e-13): the forward truth of sampled states, a model check per state, the state groups, and the gradient truth (the VJPs of the stored cotangents by centered finite differences at h = 1e-6, with their floor) of the sampled states of each group (`benchmarks/data/truth/README.md`) |
| MuJoCo Warp references | `fixtures/mjlab/mjw317_mf_s0.npz`, `mjw317_mf_s1.npz` | native mujoco-warp 3.17.0, one control step at 50 iterations of every state, per-substep counts and the contacts at the input states (`scripts/make_mjw_reference.py`) |
| forward digests | `fixtures/mjlab/engine_digests.json` | sha256 of the forward's outputs and tape per corpus, run configuration and build (GPU architecture, nvcc release, forward flags and `-D` options); V100 (sm_70, nvcc 12.4) entries |
| identity manifest | `docs/results/identity_g1_mjlab.json` | the hashes of every production build unit per target and toolchain (`scripts/identity_gate.py`) |

`model_ok` states are those where MuJoCo C and native MuJoCo Warp agree on the state's contacts, constraint zones and
contact frames; elsewhere the two reference simulators differ in the contact model itself, so the gates use the
`model_ok` subset and report the rest.

## Checks

Pass criteria are the scripts' own (`GATES` / `THRESHOLDS`, documented in each script); the measured values are from
a V100 (sm_70, nvcc 12.4) at 50 Newton iterations, 4 substeps, 4096 worlds.

### `env` (`scripts/env_check.py`)

Versions, GPU, nvcc and target architecture, the fixture model, a test kernel compiled and run, JAX's devices, and
Warp / MuJoCo Warp when installed (optional). Fails on: no nvcc or one older than release 12, a model that does not
load, a wrong kernel result, JAX missing or without a GPU.

### `step_<corpus>`, `step_truth_<corpus>`, `contacts_<corpus>` (`scripts/engine_step_check.py --gate`)

One control step of the standalone forward binary against the MuJoCo Warp reference, MuJoCo C and the float64 truth,
relative error per world |d| / |ref|:

| gate | threshold | mf_s0 | mf_s1 |
|---|---|---|---|
| qvel vs MuJoCo Warp, median / p99 | 5e-6 / 5e-5 | 1.4e-6 / 1.2e-5 | 1.4e-6 / 1.0e-5 |
| qpos vs MuJoCo Warp, median / p99 | 5e-7 / 5e-6 | 9.6e-8 / 5.3e-7 | 9.3e-8 / 5.3e-7 |
| qvel vs truth (`model_ok`), median / p99 | 5e-6 / 5e-5 | 7.4e-7 / 5.1e-6 | 7.4e-7 / 1.1e-5 |
| qpos vs truth (`model_ok`), median / p99 | 5e-7 / 5e-6 | 6.6e-8 / 2.9e-7 | 6.8e-8 / 7.6e-7 |
| contact sets vs MuJoCo Warp at the input state (one substep), share of worlds | >= 0.999 | 1.0 | 1.0 |
| worlds with supported contacts | 1 | 1 | 1 |

For scale: MuJoCo Warp against a rerun of itself differs by qvel median 3e-7, p99 3.5e-6. `step_<corpus>` runs with
`--tape`: the forward-tape checks T1-T8 (below) must pass, and the outputs' digests must match the committed manifest
(`match`; `mismatch` fails; `unanchored` where no entry exists for the GPU architecture and nvcc release at hand:
`engine_step_check.py --digests write --digests-file fixtures/mjlab/engine_digests.json` adds one).

### `gradient` (`scripts/gradient_check.py`)

The VJPs of one control step through `nablaengine.jax_ffi` in the exact gradient mode (`clamp_kv=1, normal=0`, the
truth's semantics) against the finite-difference truth, per state group G0, G4, G5, G6 (G1 reported), per input block
(qpos tangent 35, qvel 35, ctrl 29), error e = |g - fd| / max(|fd|, 1e-9) floored at the truth's own floor:

| block | gate (p99, `model_ok`) | measured (worst over G0 / G4 / G5 / G6, mf_s0 and mf_s1) |
|---|---|---|
| qpos | <= 2e-3 | 6.4e-4 |
| qvel | <= 5e-4 | 1.5e-4 |
| ctrl | <= 1e-4 | 2.4e-5 |
| cosine, 1st percentile | >= 0.99999 | 0.9999995 (smallest) |

Every gradient must be finite and every gated group must have usable states. A self-test moves the gradients by a
relative 3e-3 in a random direction and requires the G0 gate to fail. The `all` subset is reported, not gated
(states where MuJoCo C and MuJoCo Warp disagree on the contacts reach p99 9e-2).

### `ffi` (`scripts/jax_ffi_check.py`)

The JAX interface, sections `bitwise`, `jit`, `window`, `zeros`, `nonfinite`, `options`, `api`, `timing`:

- `bitwise`: the FFI against the adjoint check binary (`nablaengine.checks`, the same engine source) on the same inputs,
  every corpus, 50 and 10 iterations, the adjoint variants (1, 0), (0, 3), (1, 3), (0, 0), (2, 0), (2, 3): the forward
  (outputs, info, tape) and the gradients of the three cotangents bitwise, the tape-free forward equal to the taped
  one, the backward deterministic; at 50 iterations the forward also bitwise equal to this tree's standalone forward
  binary (`engine_step_check.py --tape --save-outputs`), whose digests are checked against the manifest.
- `jit`: no retracing or recompilation across repeated calls and batch sizes.
- `window`: a 64-step differentiated rollout (`lax.scan`) against a manual chain of per-step VJPs (bitwise for a
  linear loss), and the remat modes (with `remat_policy()` one forward call per step and the plain scan's gradients).
- `zeros`: the cotangents of warm, friction and com are exactly zero.
- `nonfinite`: no non-finite rows on the corpora; injected NaN / inf rows are counted exactly.
- `options`: gradient decay (both forms and their laws), the soft clamp, argument errors.
- `api`: `EnginePhysics`, the model check, negative shape tests (`InvalidArgument`), the adjoint's launch.
- `timing`: the FFI calls against the bare kernels (reported).

### `kinematics` (`scripts/kinematics_check.py`)

`jax_ffi.kinematics` and `step(..., kinematics=True)` against MJX's `kinematics` + `com_pos` + `com_vel`. Gates: xpos,
subtree_com <= 1e-5 m, xquat <= 1e-6, cvel median <= 1e-6 and p99 <= 1e-5, VJP median <= 1e-5 and p99 <= 1e-4; the
step's kinematics outputs and their VJP bitwise those of the separate call. Measured: xpos 4.7e-7, xquat 4.8e-7, cvel
p99 4.3e-7, VJP p99 8.2e-7.

### `h1` (`scripts/h1_check.py`)

The Unitree H1 (`fixtures/h1`, nv 25: the solves' masked lanes past NV) against MuJoCo C in float64, on 512 states of
MuJoCo C rollouts under random setpoints and pushes (standing, staggering, falling, lying), at 50 Newton iterations.
No MuJoCo Warp reference is committed for the H1. Measured on an RTX 4060 Laptop GPU (sm_89, nvcc 12.4):

| gate | threshold | measured |
|---|---|---|
| capability: no error, warnings exactly H3, H5, U2 | | H3, H5, U2 |
| forward: qvel relative error, median (outputs finite and bitwise deterministic) | 1e-5 | 3.6e-7 (p90 9.3e-3, p99 4.3e-2: contact-rich states) |
| exact-mode gradient (ctrl, qvel) vs central differences, contact-free states, median / p90 | 1e-5 / 1e-4 | 1.5e-7 / 3.0e-7 (ctrl), 1.3e-7 / 2.3e-7 (qvel) |
| exact-mode gradient (ctrl, qvel), the states as drawn, median | 1e-4 | 5.8e-7 (ctrl), 1.6e-6 (qvel) |

The tails of the gradient errors are states at a kink that central differences straddle: an actuator force at its
limit or a self contact. The G1 run through the same script gives the same picture (forward median 7.4e-7, p90
1.1e-2; contact-free gradients 3.3e-7 / 5.4e-7).

### `capability` (`scripts/capability_check.py`)

The capability checker (`nablaengine.model.capability`): every `MjModel` field of the installed MuJoCo is classified,
the G1 has no error or warning, and each of about 20 deliberately modified G1 variants (contype contacts, joint
damping, an equality, Euler integration, a tendon, the CG solver, a motor actuator, ...) is rejected with exactly its
own codes by `capability.require` and by `jax_ffi.init(model=...)`, before anything is built. CPU only.

### Not in `verify.sh`

- `scripts/identity_gate.py`: regenerates every production build unit (the FFI library's 13 translation units and the
  check binaries, per target sm_70, sm_86, sm_90, sm_120) and compares hashes at three levels with the manifest: L0
  the generated source, `-D` options, packed constants and nvcc command; L1 the preprocessed source; L2 normalized
  PTX, SASS, resource usage and host code. No GPU needed; L1 / L2 need a manifest entry for the local toolchain (nvcc,
  host compiler, ptxas, cuobjdump; `$CUOBJDUMP` must name a cuobjdump that knows sm_120). The default mode passes a
  unit at the first identical level that can clear its L0 difference; `--full` requires every level identical.
- `scripts/engine_check.py`: the smooth dynamics (mass matrix, bias forces) against MuJoCo C.
- `scripts/make_mjw_reference.py`: regenerates a MuJoCo Warp reference (needs mujoco-warp 3.17.0).
- `scripts/make_h1_model.py`: regenerates `fixtures/h1/h1_train.mjb` from MuJoCo Menagerie (byte for byte).
- `scripts/visualize_rollout.py`: a visual check: a 5 s open-loop rollout (squat and arm swing from a keyframe, ending
  in a fall) by the engine and by MuJoCo C, rendered side by side into `rollout.mp4` with the qpos deviation per step
  (float32 rounding level, 1e-7 to 2e-5 through the fall; gate `--tol`, default 1e-4) and the gradient of the final pelvis height. Needs ffmpeg and EGL (or another
  `MUJOCO_GL`).
- The benchmark harness (`benchmarks/README.md`) scores the forward and the gradients of every engine against the
  same truth files: `docs/results/paper_accuracy_heldout_v100.md`.

## Labels

### Forward-tape checks T1-T8 (`engine_step_check.py --tape`)

| label | check |
|---|---|
| T1 | outputs bitwise the same with the tape on and off (same binary), and the engine deterministic run to run |
| T2 | the state chain: record 0 is the input, record k the state of a k-substep run, qvel' = qvel + h qacc within 2 ulp |
| T3 | the contact table: record 0 against a one-substep run's contacts and MuJoCo Warp's, valid slots, row counts |
| T4 | the factor L_Q against MuJoCo C's `mj_fullM` + h diag(kv) in float64 |
| T5 | the factor L_H against M + J_A' D_A J_A from MuJoCo C at the record's state on matched contacts (T5b with the engine's contact geometry and D, T5c reported) |
| T6 | REFACT / NONCONV / smallest self-contact axis distance / clamp statistics (reported) |
| T7 | time with the tape on against off (reported) |
| T8 | every record word written, the padding between fields zero |

T1-T5b and T8 are gated.

### State groups G0-G9 (`nablaengine.reference.adjoint_groups`, `benchmarks/bench/truth`)

A control step is labelled by what happens in any of its substeps:

| group | meaning |
|---|---|
| G1 | a sphere / capsule self contact whose centre or axis distance is below 5 mm (the normal's derivative grows like 1 / distance) |
| G2 | a contact flip: the contact set differs from the reference, or a contact distance below 1e-6 m |
| G3 | a kink: a constraint row at the edge of its zone, \|x_e\| / (\|J_e a\| + \|aref_e\|) < 1e-4 |
| G4 | a PD actuator force at or beyond its force range (clamped); G4_near: within 2 % of it |
| G5 | a joint-limit row |
| G6 | a contact between two robot geoms |
| G7 | a contact frame near its basis switch |
| G8 | two near-parallel capsules |
| G9 | the solver at its iteration limit |
| G0 | none of G1, G2, G3, G7, G9: the states where a smooth derivative exists |

Subgroups: G4, G5 and G6 intersected with G0 (as the gradient gates use them), and G0 without G4.

### Adjoint phases A0-A8 (`nablaengine.g1_engine.adjoint`)

Per substep, in reverse order: A0 the tape record; A1 the integrator and implicitfast VJP; A2 the implicit-function
solve lambda = H^-1 a bar; A3 root-to-leaf sweeps (poses and the recursive Newton-Euler channels, replayed bitwise as
the forward computed them); A4 each body's local RNE VJP; A5 / A6 the constraint rows (joint limits, then contacts:
replay, impedance, the contact VJPs) between A4 and A7; A7 the leaf-to-root sweep; A8 the actuators and the
composition of the input cotangents. The module docstring has the details.

### Unit-test labels U1-U11

The labels of the unit tests of the adjoint's device functions against float64 JAX ports
(`nablaengine.reference.jax64`) or MuJoCo C:
U1 the quaternion integration VJP, U2 the solves on the tape's factors, U3 the tree sweeps (A3, A4, A7), U4 the
impedance derivatives, U5 the limit rows, U6 the contact rows, U7 the plane contacts and the geometry-to-body VJP, U8
the self-contact primitives and the frame, U9 the approximate normal derivatives (`DK_ADJ_NORMAL` 1-3), U10 the pose
sweep, U11 the constraint-row stage of the adjoint kernel. The code comments cite them by label; the unit tests
themselves are not part of this repository. The gates above (`gradient`, `ffi`) check the adjoint as a whole.

The capability checker's codes are a separate namespace (`nablaengine.model.capability`): H* the assumptions of the
engine (an error when broken), U* code paths the G1 does not run (warnings), I* recorded values, F0 a field the
classification does not know.
