# The JAX interface: `nablaengine.jax_ffi`

`nablaengine.jax_ffi` exposes the engine as JAX operations: one control step of B simulated worlds as an XLA FFI
custom call with an analytic VJP (`jax.custom_vjp`), the kinematics outputs with their VJP, lower-level forward and
backward calls, gradient decay, a remat policy and diagnostic counters. The module docstring of
`src/nablaengine/jax_ffi.py` is the complete reference; this page is the overview.

The library is generated from the engine's CUDA source for the model and built by nvcc on first use (about 15 s per
GPU architecture), then loaded from a cache (`$DK_CACHE_DIR`, default `~/.cache/nablaengine`) keyed by a hash of the
sources, the model constants, the flags, the nvcc version and XLA's FFI headers. Everything is float32.

## Shapes and units

For the mjlab Unitree G1 (`fixtures/mjlab/g1_train.mjb`): `nq = 36`, `nv = 35`, `nu = 29`; for the Unitree H1
(`fixtures/h1/h1_train.mjb`, `jax_ffi.init(model=m, foot_pairs=meta["foot_pairs"])`): `nq = 26`, `nv = 25`, `nu = 19`. B, the number of worlds, is
read from the arrays at run time: one library serves every batch size, and JAX traces once per shape.

| argument | shape | meaning |
|---|---|---|
| `qpos` | (B, 36) | joint positions (free joint: position, unit quaternion w x y z; then 29 hinges) |
| `qvel` | (B, 35) | joint velocities |
| `warm` | (B, 35) | the Newton warm start (`qacc_warmstart`); the step returns the next one |
| `ctrl` | (B, 29) | position targets of the 29 actuators, held over the control step |
| `friction` | (B,) | per-world friction of the foot-floor contact pairs (domain randomisation) |
| `com` | (B, 3) | per-world offset of the torso body's centre of mass (domain randomisation) |

A control step is `substeps` engine substeps of the model's timestep (4 x 5 ms by default).

## `init`

```python
info = jax_ffi.init(model_dir=None, model=None, foot_pairs=None)
```

Loads the library for the model (building it on first use), hands it the model constants and registers the FFI
targets. Idempotent; every traced call goes through it, so an explicit call is optional. `model_dir` is a fixtures
directory (default `$DK_FIXTURES`, else `fixtures/mjlab` of the checkout); `model` / `foot_pairs` pass an `MjModel`
and its foot-floor pair indices instead (for an installed package without the fixtures). The returned dict holds the
library path and cache key, the architecture and its flags (`arch`, `arch_flags`, `forward_defines`), the tape record
size and whether the library was built now (`built`); log it with results.

## `step` and `make_step`

```python
qpos1, qvel1, warm1 = jax_ffi.step(qpos, qvel, warm, ctrl, friction, com, *, iterations=10, substeps=4,
                                   clamp_kv=0, normal=3, clamp_soft=None, grad_decay=1.0, kinematics=False)
step_fn = jax_ffi.make_step(iterations=10, substeps=4, clamp_kv=0, normal=3, clamp_soft=None, grad_decay=1.0,
                            kinematics=False)
```

One differentiable control step. The options are static Python values; each combination is its own `custom_vjp`
function (`make_step` returns it, normalized and cached, so jit caches stay valid). Without differentiation `step`
runs the forward kernels without a tape. Under `jax.vjp` / `jax.grad` the forward also writes a tape (per world and
substep `rec_words()` float32 values; `tape_bytes(nworld)` gives the size) that the backward replays: the adjoint runs
on the forward's own contacts, constraint rows and solver state, and is deterministic. The VJP returns the
cotangents of `qpos` (in the 36-dimensional qpos form, as MuJoCo Warp), `qvel` and `ctrl` (summed over the substeps),
and zeros for `warm`, `friction` and `com`; the cotangent of `warm'` is ignored.

| option | default | meaning |
|---|---|---|
| `iterations` | 10 | Newton iterations per substep (the reference corpora of the checks use 50) |
| `substeps` | 4 | substeps per control step |
| `clamp_kv` | 0 | derivative on a dof whose actuator force is clamped at its limit: 0 keeps the actuator damping term -kv in the qvel VJP (as MuJoCo Warp's adjoint), 1 drops it (the exact derivative), 2 the soft clamp (a surrogate gradient of width `clamp_soft`) |
| `normal` | 3 | derivative of the contact normal of self contacts between spheres and capsules: 3 frozen where the normal is shorter than 1e-3 m (finite gradients), 0 exact, 1 and 2 other approximations |
| `clamp_soft` | None | the soft clamp's width for `clamp_kv=2` (None: 0.2); must stay None for `clamp_kv` 0 and 1 |
| `grad_decay` | 1.0 | the physics-only gradient decay: the backward multiplies the cotangents it returns for qpos and qvel by this factor (1.0: no multiply) |
| `kinematics` | False | also return the kinematics outputs of the output state (below), computed in the same call, their cotangents in the same VJP |

`clamp_kv=1, normal=0` is the exact derivative of the forward ("exact mode", the one compared with finite
differences); `clamp_kv=0, normal=3` are the interface defaults. The forward does not depend on `clamp_kv`,
`normal`, `clamp_soft` or `grad_decay`: one tape serves every adjoint variant, and the forward outputs are bitwise
the same in every mode.

A VJP through the custom call is a plain JAX VJP:

```python
outs, pull = jax.vjp(lambda q, v, u: jax_ffi.step(q, v, warm, u, friction, com)[:2], qpos, qvel, ctrl)
g_qpos, g_qvel, g_ctrl = pull((cot_qpos, cot_qvel))
```

## Kinematics outputs

```python
qpos_n, xpos, xquat, cvel, subtree_com = jax_ffi.kinematics(qpos, qvel, com)
*state, qpos_n, xpos, xquat, cvel, subtree_com = jax_ffi.step(..., kinematics=True)
```

MJX's `mjx.kinematics` + `com_pos` + `com_vel` on the model with the world's torso centre-of-mass offset, with their
VJP: `qpos_n` (B, 36) with the free joint's quaternion normalized, `xpos` (B, nbody, 3) and `xquat` (B, nbody, 4) body
frames (world body 0 included), `cvel` (B, nbody, 6) twists [angular; linear] at the reference point
`subtree_com[1]`, `subtree_com` (B, nbody, 3). `step(..., kinematics=True)` computes them for the output state inside
the step's call; `kinematics()` is for states the engine did not produce (initial or external states), and on a step's
output it gives the same bits. `kinematics_forward` / `kinematics_backward` are the same without `custom_vjp`.

## Lower-level calls

```python
qpos1, qvel1, warm1, info, tape = jax_ffi.forward(qpos, qvel, warm, ctrl, friction, com, *, iterations=10,
                                                  substeps=4, tape=True, kinematics=False)
g_qpos, g_qvel, g_ctrl, binfo = jax_ffi.backward(tape, friction, com, g_qpos1, g_qvel1, *, substeps=4, clamp_kv=0,
                                                 normal=3, ctrl=None, clamp_soft=None, kinematics=None)
```

Traceable, without `custom_vjp`. `info` (B, 4) int32 per world: Newton iterations of the last substep, contacts,
constraint rows, unsupported bits (a world whose contacts exceed what the engine models; `tape.py`
`FLAG_UNSUPPORTED`). `binfo` (B,) int32: the adjoint's flags (the union of the records' tape flags,
`adjoint.ADJ_ROWS_SKIPPED`, `ADJ_REPLAY_MISMATCH`) or-ed with `NONFINITE_* << NONFINITE_SHIFT` (which of the world's
gradients has a NaN / inf, and whether its incoming cotangent had one). `clamp_kv=2` needs the step's `ctrl`. The
gradients of `backward` are never scaled by `grad_decay`.

## Gradient decay: `scale_grad`

```python
state = jax_ffi.scale_grad(state, decay)
```

The identity on a pytree whose cotangent the backward multiplies by `decay` (a static Python float). Applied to the
whole carried state at the top of a `lax.scan` body ("full-state form") it multiplies every derivative that crosses n
step boundaries by exactly `decay ** n`, whatever the path. Applied only to the physics call's `qpos`, `qvel`
("physics-only form", the same gradients as `step(..., grad_decay=decay)`) it scales only the part of the state's
cotangent that flows back through the physics, not the controller's path. The module docstring has both forms and
their laws.

## Remat

The custom VJP names its residual tape `"dk_tape"` (`TAPE_NAME`) and its outputs `"dk_out"` (`OUT_NAME`) with
`jax.ad_checkpoint.checkpoint_name`; `kinematics()` names its outputs and residual state `"dk_kin"` (`KIN_NAME`).
`REMAT_NAMES` holds all three. Use

```python
body = jax.checkpoint(body, policy=jax_ffi.remat_policy())   # = save_only_these_names(*REMAT_NAMES, *extra)
```

for a differentiated rollout under `jax.checkpoint`: one forward call per control step, as without remat. A plain
`jax.checkpoint(body)` saves none of them, and the backward reruns the forward for the tape (two forward calls per
control step). The gradients equal those without remat up to XLA's rounding of the recomputed parts of the body.
`remat_policy(*extra_names)` adds the caller's own checkpoint names.

## Diagnostics

The library keeps integer counters in device memory per GPU (exact and order-independent atomics); every forward and
backward call adds to them, also inside `jit` and `lax.scan`:

`fwd_calls`, `fwd_worlds`, `fwd_unsupported_worlds`, `fwd_iteration_limit_worlds` (the last substep used every Newton
iteration), `tape_records`, `refact_records`, `nonconv_records`, `singular_records` (from the tape), `bwd_calls`,
`bwd_rows`, `nonfinite_rows`, `nonfinite_qpos_rows`, `nonfinite_qvel_rows`, `nonfinite_ctrl_rows`,
`nonfinite_cotangent_rows`, `replay_mismatch_worlds`, `fwd_contact_overflow_worlds`, `fwd_row_overflow_worlds`.

| call | where | returns |
|---|---|---|
| `read_diagnostics(reset=True)` | outside jit (host sync) | `{name: int}` since the last reset |
| `diagnostics(*deps, reset=True)` | inside jit, after `deps` are computed | (18, 2) uint32 low / high words; `decode_diagnostics(arr)` -> `{name: int}` |
| `last_forward(nworld=None)` | outside jit | the last forward call's record for that batch size: `worlds`, `contacts` (summed over the worlds), `contacts_max`, `rows_max` |
| `launch_info(device=0)` | outside jit | the adjoint kernel's launch per variant: shared memory and worlds per SM |

**Capacity.** Each world holds at most 64 contacts and 192 constraint rows (a frictional contact takes 4 rows, a joint
limit 1). A world beyond them, typically after deep interpenetration with the floor or between links (a fall), keeps
the first ones and drops the rest, so it misses part of its support and friction forces; its step is then not the
model's. Such worlds are flagged in `forward()`'s `info[:, 3]` (`OVERFLOW_CONTACTS = 4`, `OVERFLOW_ROWS = 16`) and
counted per call and world in `fwd_contact_overflow_worlds` / `fwd_row_overflow_worlds` (both also in
`fwd_unsupported_worlds`). With `set_overflow_warnings(True)` (or `DK_OVERFLOW_WARN=1`; off by default)
`decode_diagnostics` / `read_diagnostics` issue a `RuntimeWarning` when either count is non-zero (`warn_overflow`),
and the counts nobody read are reported once more at exit. Mask or reset the flagged worlds
of a batch before using its results. On the four held-out benchmark corpora (16,384 states) none occur: at most 30 contacts and 114 rows.

## `EnginePhysics`

```python
phys = jax_ffi.EnginePhysics(mjm, nworld, foot_pairs, torso_id, *, iterations=None, substeps=4, clamp_kv=0,
                             normal=3, clamp_soft=None, grad_decay=1.0, kinematics=False, use_model=False)
qpos1, qvel1, warm1 = phys.step(qpos, qvel, warm, ctrl, friction, com)
phys.stats()          # nacon, naconmax, nefc_max, njmax of the last forward call with this batch size
phys.diagnostics()    # read_diagnostics()
```

A batched physics object with MuJoCo Warp's capacity names (`nconmax` 64 contacts and `njmax` 192 constraint rows per
world). The model constants come from the fixtures, and `mjm` must then be the same model (`check_model`), unless
`use_model=True`: then from `mjm`. Other keyword options of a MuJoCo Warp based physics object are accepted and
ignored.

## Unsupported models

The engine supports one model, the mjlab G1. `init`, `check_model`, `EnginePhysics` and the scripts' model loading
run the capability checker (`nablaengine.model.capability`) first. A model outside its scope raises
`nablaengine.model.capability.UnsupportedModel` (a subclass of `ValueError` and `AssertionError`) before anything is
built, with the failing checks' codes and values; it never simulates other physics silently. `check_model(mjm,
foot_pairs, torso_id)` also raises `ValueError` when a supported model differs from the one whose constants the
library holds (sizes, timestep, foot pairs, torso id, the packed physics constants compared bitwise). The checker's
codes and the `MjModel` fields it inspects are in `src/nablaengine/model/capability.py` and `fields.py`;
`scripts/capability_check.py` is its test. `$DK_ALLOW_UNSUPPORTED` (`docs/configuration.md`) turns named codes into
warnings for experiments; results of such a build are not comparable with the checks'.

Errors at call time: the FFI handlers check the rank and dimensions of every buffer before they read or write any
and raise `InvalidArgument` (a `JaxRuntimeError`) on a mismatch; `make_step` raises `ValueError` for an adjoint
variant that does not exist or a non-finite `grad_decay`.
