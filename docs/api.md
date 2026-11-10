# The JAX interface: `nablaengine.jax_ffi`

`nablaengine.jax_ffi` exposes the engine as JAX operations: one control step of B simulated worlds as an XLA FFI
custom call with an analytic VJP (`jax.custom_vjp`), the kinematics outputs with their VJP, lower-level forward and
backward calls, gradient decay, a remat policy and diagnostic counters. The module docstring of
`src/nablaengine/jax_ffi.py` is the complete reference; this page is the overview.

The library is generated from the engine's CUDA source for the model and built by nvcc on first use (about 15 s per
GPU architecture), then loaded from a cache (`$DK_CACHE_DIR`, else `$XDG_CACHE_HOME/nablaengine`, else
`~/.cache/nablaengine`) keyed by a hash of the
sources, the model constants, the flags, the nvcc version and XLA's FFI headers. Everything is float32.

CUDA library compilation uses `MAX_JOBS` for concurrent translation units and
`DK_COMPILE_THREADS` for each unit's `nvcc --split-compile` optimizer threads
(default 1). Their product is bounded by process affinity and the container's
`cpu.max` quota; the operator must still assign the process its authorized CPU
share. For a cold build assigned 24 CPUs, use `MAX_JOBS=12 DK_COMPILE_THREADS=2`.
The final host link remains one command. Both Torch and JAX use this builder;
there is no Ninja layer or separate `MAX_JOBS` owner.
Compile and link subprocesses use their build directory as the working directory,
independent of the caller's shared-home CWD; select a local `DK_CACHE_DIR`.

Changing only `MAX_JOBS` reuses existing artifacts. Changing optimizer threads
above 1 produces a distinct artifact because the generated code must be
qualified separately. Keep `DK_COMPILE_THREADS=1` when reusing qualified libraries.
CUDA 12.9's [split compilation](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-compiler-driver-nvcc/index.html#split-compile-number-split-compile)
parallelizes optimizer passes; `--threads` mainly parallelizes compilation for
multiple target architectures, so it is not added to this single-architecture
builder. Neither setting enables fast math or lowers optimization levels.

## Shapes and units

For independent models, `create_group(contexts, iterations, grad_decay=...)`
reuses their existing compiled libraries. `group.step(inputs, substeps=1)` takes
one tuple of eight native operands per model and returns one
`(qpos, qvel, warm, info)` tuple per model. Each model executes on its own CUDA
stream; events order its inputs and return completed outputs to XLA's stream.
The forward kernels, analytic adjoint and per-model tapes are unchanged. An
empty model batch remains empty. Controllers that recompute forces every
physics substep must continue calling the group once per substep.

`group.step(inputs, substeps=1, kinematics=True)` also returns the five native
successor kinematics arrays after `info`, in the same order as the context's
`step(..., kinematics=True)`. They remain differentiable: their cotangents are
pulled back into successor Q/V before the existing adjoint. A controller can
carry these exact arrays into its next feedback calculation and its final
readout. Initial/reset states still need their own factual kinematics; the
group does not cache geometry across states or optimizer updates.

The dispatcher compiles one small additional CUDA unit, without rebuilding
the model libraries. Group setup initializes the existing contexts before
worker dispatch. `group.read_diagnostics(reset=False)` returns the existing
diagnostic counter schema for each group member; those counters belong to
the group, separately from calls made directly through each context.

Each context supplies its compiled `nq`, `nv`, `nu`, `nbody` and FREE-root count `nr`.
Group members may have different dimensions and independent batch sizes, including zero.
Create contexts and the group once for the selected models; a step does not rebuild them.
B is read from each model's arrays: one library serves every batch size, while JAX traces once per shape.
For the mjlab Unitree G1 fixture only, `nq = 36`, `nv = 35`, `nu = 29` and `nr = 1`.

| argument | shape | meaning |
|---|---|---|
| `qpos` | (B, nq) | joint positions in the model's compiled order; each FREE joint has position and unit quaternion w x y z |
| `qvel` | (B, nv) | joint velocities in the model's compiled order |
| `warm` | (B, nv) | the Newton warm start (`qacc_warmstart`); the step returns the next one |
| `ctrl` | (B, nu) | position targets in native actuator order, held over the call's substeps; passive hinges have no control slot |
| `friction` | (B,) | per-world friction of the selected generated contact pairs; it replaces those pairs' own friction, so pass the model's value where it is not randomised |
| `com` | (B, 3*nr) | offsets of the selected COM bodies for the model's one or two FREE trees; selected by `com_body_id` |
| `qfrc_applied` | (B, nv) | applied force in the model's generalized coordinates; default zeros |
| `kp_scale` | (B,) | actual native actuator Kp scale; default ones, retained for replay and stopped in the VJP |

A control step is `substeps` engine substeps of the model's timestep (4 x 5 ms by default).

## `init`

```python
info = jax_ffi.init(model_dir=None, model=None, foot_pairs=None, com_body=None)
```

Loads the library for the model (building it on first use), hands it the model constants and registers the FFI
targets. Idempotent; every traced call goes through it, so an explicit call is optional. `model_dir` is a fixtures
directory (default `$DK_FIXTURES`, else `fixtures/mjlab` of the checkout); `model` / `foot_pairs` pass an `MjModel`
and its foot-floor pair indices instead (for an installed package without the fixtures, or another robot: below),
`com_body` the name of the body that takes the per-world COM offset `com` (default `torso_link`, or the fixtures'
`torso_body`). The returned dict holds the
library path and cache key, the architecture and its flags (`arch`, `arch_flags`, `forward_defines`), the tape record
size and whether the library was built now (`built`); log it with results.

## Model contexts

Use `create_context(model=..., foot_pairs=..., com_body_id=...)` for multiple
models in one process. The selected model owns its dimensions, packed
constants and registered FFI targets. The full selected COM-body tuple is used
for construction and `context.check_model`, including bilateral FREE trees;
it is not inferred from the G1 torso name.

```python
context = jax_ffi.create_context(model=model, foot_pairs=selected_pairs,
                                 com_body_id=selected_com_bodies)
q1, v1, warm1, info = context.step(qpos, qvel, warm, ctrl, friction, com,
                                  qfrc_applied, kp_scale, iterations=100,
                                  substeps=1, return_info=True)
qn, xpos, xquat, cvel, subtree_com = context.kinematics(q1, v1, com)
```

State and tape arrays belong to the caller and can coexist across models and
JIT scans. Different compiled models have different target names; custom VJP
closures retain their actual context. Creating the same compiled model again
returns its existing context and model-scoped diagnostics, rather than claiming
independent counters for replicas. The default `init`/module functions retain
their single selected model; explicit contexts do not replace it. Model context
metadata is not a portable physical-state snapshot.

## `step` and `make_step`

```python
qpos1, qvel1, warm1 = jax_ffi.step(qpos, qvel, warm, ctrl, friction, com, qfrc_applied=None, kp_scale=None, *, iterations=10, substeps=4,
                                   clamp_kv=0, normal=3, clamp_soft=None, grad_decay=1.0, kinematics=False, return_info=False)
step_fn = jax_ffi.make_step(iterations=10, substeps=4, clamp_kv=0, normal=3, clamp_soft=None, grad_decay=1.0,
                            kinematics=False)
```

One differentiable control step. The options are static Python values; each combination is its own `custom_vjp`
function (`make_step` returns it, normalized and cached, so jit caches stay valid). Without differentiation `step`
runs the forward kernels without a tape. Under `jax.vjp` / `jax.grad` the forward also writes a tape (per world and
substep `rec_words()` float32 values; `tape_bytes(nworld)` gives the size) that the backward replays: the adjoint runs
on the forward's own contacts, constraint rows and solver state, and is deterministic. The VJP returns the
cotangents of `qpos` (in the 36-dimensional qpos form, as MuJoCo Warp), `qvel` and `ctrl` (summed over the substeps),
and applied generalized force, with zeros for `warm`, `friction`, `com` and `kp_scale`; the cotangent of `warm'` is ignored. The forward and adjoint use the same applied force and Kp values; these inputs are never silently dropped.

| option | default | meaning |
|---|---|---|
| `iterations` | 10 | Newton iterations per substep (the reference corpora of the checks use 50) |
| `substeps` | 4 | substeps per control step |
| `clamp_kv` | 0 | derivative on a dof whose actuator force is clamped at its limit: 0 keeps the actuator damping term -kv in the qvel VJP (as MuJoCo Warp's adjoint), 1 drops it (the exact derivative), 2 the soft clamp (a surrogate gradient of width `clamp_soft`) |
| `normal` | 3 | derivative of the contact normal of self contacts between spheres and capsules: 3 frozen where the normal is shorter than 1e-3 m (finite gradients), 0 exact, 1 and 2 other approximations |
| `clamp_soft` | None | the soft clamp's width for `clamp_kv=2` (None: 0.2); must stay None for `clamp_kv` 0 and 1 |
| `grad_decay` | 1.0 | the physics-only gradient decay: the backward multiplies the cotangents it returns for qpos and qvel by this factor (1.0: no multiply) |
| `kinematics` | False | also return the kinematics outputs of the output state (below), computed in the same call, their cotangents in the same VJP |
| `return_info` | False | return the native int32 status after warm, before any kinematics outputs; info has no derivative |

`clamp_kv=1, normal=0` is the exact derivative of the forward ("exact mode", the one compared with finite
differences); `clamp_kv=0, normal=3` are the interface defaults. The defaults differ from the exact derivative only
on dofs whose actuator force is at its limit (and on near-singular self contacts): negligible for the G1, but a robot
whose actuators often saturate (e.g. MuJoCo Playground's H1, kp 800 against limits of 18-40 N m) sees median errors
of 1-4% against finite differences in the default mode and about 1e-6 in exact mode. The forward does not depend on `clamp_kv`,
`normal`, `clamp_soft` or `grad_decay`: one tape serves every adjoint variant, and the forward outputs are bitwise
the same in every mode.

The native adjoint uses one power-of-two scale per world's incoming state cotangents to avoid intermediate
overflow, then restores that scale on every returned physics gradient. This is numerical preconditioning;
the gradient convention, clipping and decay options are unchanged. Gradients outside the float32 output range
remain nonfinite.

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

MJX's `mjx.kinematics` + `com_pos` + `com_vel` on the model with the selected bodies' centre-of-mass offsets, with their
VJP: `qpos_n` (B, nq) with every FREE joint's quaternion normalized, `xpos` (B, nbody, 3) and `xquat` (B, nbody, 4) body
frames (world body 0 included), `cvel` (B, nbody, 6) twists [angular; linear] at the reference point
`subtree_com[body_rootid]`, `subtree_com` (B, nbody, 3). `step(..., kinematics=True)` computes them for the output state inside
the step's call; `kinematics()` is for states the engine did not produce (initial or external states), and on a step's
output it gives the same bits. `kinematics_forward` / `kinematics_backward` are the same without `custom_vjp`.

## Lower-level calls

```python
qpos1, qvel1, warm1, info, tape = jax_ffi.forward(qpos, qvel, warm, ctrl, friction, com, qfrc_applied, kp_scale, iterations=10,
                                                  substeps=4, tape=True, kinematics=False)
g_qpos, g_qvel, g_ctrl, binfo, g_qfrc = jax_ffi.backward(tape, friction, com, g_qpos1, g_qvel1, substeps=4, clamp_kv=0,
                                                 normal=3, qfrc_applied=qfrc_applied, kp_scale=kp_scale)
```

Traceable, without `custom_vjp`. `info` (B, 4) int32 per world: Newton iterations of the last substep, contacts,
constraint rows, unsupported bits (a world whose contacts exceed what the engine models; `tape.py`
`FLAG_UNSUPPORTED`). `binfo` (B,) int32: the adjoint's flags (the union of the records' tape flags,
`adjoint.ADJ_ROWS_SKIPPED`, `ADJ_REPLAY_MISMATCH`) or-ed with `NONFINITE_* << NONFINITE_SHIFT` (which of the world's
gradients has a NaN / inf, and whether its incoming cotangent had one). `clamp_kv=2` needs the step's `ctrl`. The
gradients of `backward` are never scaled by `grad_decay`. Supply the same physical operands as the forward.
Omitting applied force keeps the original four-result backward interface; supplying it appends its gradient.

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

**Capacity.** Each library compiles point and row capacities from its model. Point storage uses a positive model
`nconmax`, or an automatic budget of 256 points, limited by the sum of the eligible pairs' maximum manifold sizes
with a 32-point minimum allocation. Row storage covers every stored point (4 rows for a frictional contact, 1 for a
frictionless contact), every frictionloss and soft equality row, and one limit row per limited hinge. A larger positive
model `njmax` is preserved. Read the actual capacities from `init()`'s `maxcon` and `nefc` metadata.
A world beyond its capacities, for example during deep interpenetration with the floor or between links, keeps
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

A batched physics object with MuJoCo Warp's capacity names (`nconmax` contact points and `njmax` constraint rows per
world), set from the compiled model capacities above. The model constants come from the fixtures, and `mjm` must then be the same model (`check_model`), unless
`use_model=True`: then from `mjm`. Other keyword options of a MuJoCo Warp based physics object are accepted and
ignored.

## Unsupported models

The engine accepts one or two top-level FREE roots and one hinge at the origin of every other body,
with `6 <= nv <= 64`, at most eight children per body and 12 hinge ancestors. Forward and kinematics allow
64 bodies; the adjoint allows 31 bodies for one FREE root and 64 for two. These are kernel limits, independent
of robot names. Position actuators may drive any unique subset of hinges, with independent `nu`; passive
hinges remain in `qpos` and `qvel`. Linear joint damping, dof frictionloss and up to `6 * nr` active soft linear hinge
equalities are represented. Explicit and eligible `contype` / `conaffinity` contact pairs are generated; pairs may
also be absent. The capability checker validates the supported plane, sphere, capsule and box combinations,
including the plane-box and box-box adjoint hooks. Convex-mesh collisions, more than two FREE roots, welded,
multi-joint, ball and slide bodies, tendons, flexes and plugin dynamics remain unsupported.
The published numerical results belong to the models and builds named in their result files; CPU generation
checks for the extended model class are described in `docs/verification.md`.
Disabling contacts changes the simulated condition; a successful check of such a model does not qualify its original
contact-enabled scene.

`init`, `check_model`, `EnginePhysics` and the scripts' model loading
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

## Other robots

The engine generates its code from each model under the model class and kernel limits above
(`nablaengine.model.capability`), using MuJoCo's Newton solver, a pyramidal friction cone and implicitfast integration.
Robot names, controller cadence, observation roles and training schedules are not inputs to code generation.

A model outside the class is rejected with the failing checks. Pass a supported model as
`create_context(model=m, foot_pairs=[...], com_body_id=...)`, set its `friction` input to the selected pairs' friction unless it is
randomised, and set `substeps` to its control step.

Validated robots (`docs/results/robots_check_v100.json`, V100). Forward: one control step of about 4000 states
against MuJoCo Warp 3.17.0, with the qvel relative error at the 99th percentile and the share of worlds with the same
contact count. Gradients: against float64 central differences of MuJoCo C, median relative error of the qvel
gradient, on the worlds whose forward agrees with MuJoCo C (MuJoCo C's box-box collision and frictionloss differ from
MuJoCo Warp's).

| robot | model | nv | contacts | forward qvel p99 | same contact count | qvel gradient |
|---|---|---|---|---|---|---|
| Unitree G1 | mjlab (the fixture) | 35 | plane-sphere / capsule, capsule-capsule | `docs/verification.md` | | 1.0e-5 |
| Unitree Go1 | mjlab | 18 | plane-sphere / capsule / box | 5.7e-6 | 99.97 % | 7.3e-7 |
| Unitree Go2 | MuJoCo Menagerie | 18 | plane-sphere | 2.1e-5 | 99.97 % | 1.5e-6 |
| Unitree A1 | MuJoCo Menagerie | 18 | plane-sphere | 1.7e-5 | 100 % | 1.8e-6 |
| ANYbotics ANYmal C | MuJoCo Menagerie | 18 | plane-sphere | 2.1e-4 | 100 % | 2.3e-6 |
| Boston Dynamics Spot | MuJoCo Playground | 18 | plane-sphere | 1.2e-5 | 100 % | 3.3e-6 |
| Unitree H1 | MuJoCo Playground | 25 | plane-capsule | 6.4e-6 | 100 % | 5.7e-7 |
| Booster T1 | MuJoCo Playground | 29 | plane-sphere, box-box | 5.7e-5 | 99.8 % | 4.7e-6 |
| ROBOTIS OP3 | MuJoCo Playground | 26 | plane-box, box-box | 1.2e-4 | 99.97 % | 1.2e-6 |
| Berkeley Humanoid | MuJoCo Playground | 18 | plane-box, box-box | 5.7e-5 | 99.9 % | 6.0e-6 |

Notes on the models:
- MuJoCo Playground's are its feet-only collision scenes.
- MuJoCo Menagerie's are its `scene.xml` with feet-only collision and the usual locomotion contact settings: pyramidal
  cone, condim 3 (Menagerie's files use an elliptic cone and condim 6).
- The Go2's torque motors are replaced by position servos (kp 35, kd 0.5).

Exporting and checking a robot, and the time per control step of each: `docs/robots.md`. Terrain: `docs/terrain.md`.

## Context interface qualification

`scripts/jax_context_check.py` checks context routing, typed shapes, force VJP,
stopped physical parameters, successor kinematics/status, a 4×H64 scan and
three actual CPU Adam updates using explicit analytic FFI doubles. It also
checks bilateral COM constant packing with mocked library loading/admission.
Those CPU checks do not qualify CUDA solver numerics or hand contacts. The
native FFI handlers use the existing CUDA forward/adjoint kernels on XLA's
provided stream; actual model/physics qualification must use the selected
compiled source and runtime.
