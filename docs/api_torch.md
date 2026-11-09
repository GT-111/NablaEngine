# The PyTorch interface: `nablaengine.torch_ffi`

The same engine as [`nablaengine.jax_ffi`](api.md), called from PyTorch without JAX: the same kernels, the same options
and defaults, and bitwise the same outputs and gradients on the same GPU (`scripts/torch_ffi_check.py`). Read
[api.md](api.md) for the meaning of the inputs, outputs and options; this page lists what differs.

```python
import torch
from nablaengine import fixtures, torch_ffi

torch_ffi.init()                         # builds the CUDA library on first use (nvcc), then loads it from the cache
z = fixtures.load_corpus("mf_s0")
B = 256
qpos, qvel, warm, ctrl, friction, com = (torch.from_numpy(z[k][:B]).cuda() for k in
                                         ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction", "com"))

ctrl.requires_grad_()
qpos1, qvel1, warm1 = torch_ffi.step(qpos, qvel, warm, ctrl, friction, com)  # one control step: 4 substeps of 5 ms
(qvel1 ** 2).sum().backward()            # ctrl.grad: the analytic adjoint
```

## Installation and the build artifact

`pip install nablaengine[torch]`: NumPy, MuJoCo 3.17 and PyTorch, no JAX (`nablaengine[jax]` is the JAX interface's
extra). The CUDA library is built once per architecture into an artifact in the cache directory (`DK_CACHE_DIR`):

```bash
python -m nablaengine.torch_ffi build                      # the first GPU's architecture
python -m nablaengine.torch_ffi build --arch sm_90 --arch sm_120   # or named ones (no GPU needed; nvcc must support them: sm_120 needs CUDA 12.8)
```

This needs mujoco 3.17 and nvcc (a CUDA 12.x toolkit) and writes `libdk_g1_torch_<arch>_<key>.so` (the CUDA runtime
linked statically; the units of the JAX library without its XLA FFI unit, so the kernels are the same), the packed
model constants (`.npy`) and the model facts (`.json`), and makes it the architecture's current artifact. A process
that only runs the interface needs PyTorch, NumPy and the NVIDIA driver: `init()` loads the artifact with ctypes and
imports no JAX and no mujoco and calls no nvcc (an artifact built elsewhere: copy its three files and set
`DK_TORCH_ARTIFACT` to its `.json`). The artifact records a hash of the package sources it was generated from; after
an update of the package `init()` rebuilds it where mujoco and nvcc are available and refuses it otherwise. Without
any artifact, `init()` builds one first.

Concurrent processes requesting the same library key share one compilation under a Linux file lock; different
keys can compile in parallel. Constants, artifact metadata and the current-artifact pointer are published by
atomic replacement, so another rank always reads complete files.

On a V100 (sm_70) install a PyTorch build that still targets sm_70, e.g. the CUDA 12.6 wheels
(`--index-url https://download.pytorch.org/whl/cu126`); the CUDA 12.8 and 12.9 wheels start at sm_75.

## Entries

| entry | as in jax_ffi | notes |
|---|---|---|
| `init(device=None, artifact=None, model_dir=None, model=None, foot_pairs=None, com_body_id=None)` | `init` | prepares the default context and returns its model facts; the first call on a device synchronizes it |
| `create_context(device=None, artifact=None, model_dir=None, model=None, foot_pairs=None, com_body_id=None)` | | returns an independent `ModelContext` with its own immutable model library, dimensions, constants, readiness and diagnostics |
| `build(arch, ...)`, `python -m nablaengine.torch_ffi build` | | the artifact for an architecture (mujoco and nvcc needed) |
| `step(qpos, qvel, warm, ctrl, friction, com, qfrc_applied=None, kp_scale=None, *, iterations=10, substeps=4, clamp_kv=0, normal=3, clamp_soft=None, grad_decay=1.0, kinematics=False, return_info=False, context=None)` | `step` | differentiable in Q, V, control and applied generalized force; the tape is written only when an input requires grad; `warm'` is not differentiable |
| `kinematics(qpos, qvel, com)` | `kinematics` | differentiable in `qpos` and `qvel` |
| `scale_grad(tree, gamma)` | `scale_grad` | on a pytree of tensors (`torch.utils._pytree`); non-floating leaves pass through |
| `forward(...)`, `backward(...)` | `forward`, `backward` | without autograd; the same outputs, `info` included |
| `diagnostics(device=None)`, `read_diagnostics(device=None, reset=True)` | `diagnostics`, `read_diagnostics` | the counters of `jax_ffi.DIAG`, kept per device (below) |
| `set_diagnostics(enabled)` | | the counters on (default) or off; returns the previous setting |
| `tape_bytes(nworld, substeps=4, steps=1)` | `tape_bytes` | |
| `set_terrain(terrain, device=None, cell=1.0)` | `set_terrain` | uploads a `nablaengine.terrain.Terrain` for a terrain build (`init(terrain=TerrainConfig(...))`, `docs/terrain.md`) |
| `grid_pattern(size, resolution, direction, device=None)`, `raycast(pos, rot, offsets, directions, alignment="base", max_distance=10.0)` | `grid_pattern`, `raycast` | the height scan of a terrain build (no gradient) |

Every tensor is float32, contiguous and on one CUDA device, with the worlds as the leading dimension; another dtype,
device, layout or shape raises `TypeError` (nothing is cast or copied). Every call runs on PyTorch's current CUDA
stream of the inputs' device, with that device made current for the call.

`qfrc_applied` is `(B, nv)` in the model's generalized force coordinates; omitted means zero. It is held for the
native substeps in that call. `kp_scale` is `(B,)` and multiplies the actual position servo Kp, leaving Kd unchanged;
omitted means one. It is a stopped physical randomization parameter, with no gradient. `backward(...,
qfrc_applied=..., kp_scale=...)` takes the same operands as its forward and appends `g_qfrc_applied` to its usual four
outputs when an applied force is supplied.

`return_info=True` inserts the same `(B, 4)` int32 forward status as `forward()` after `warm'`, before any optional
kinematics outputs. It is not differentiable. Its columns are the last substep's Newton iterations, contact count,
constraint-row count and unsupported/overflow flags. This lets a differentiated rollout check capacity on the
device, for example `torch._assert_async((info[:, 3] == 0).all(), "unsupported native step")`, without a host read.

## Independent models in one process

Module calls without `context=` keep their original default G1 API. `init()` prepares that context once; it never
switches the default model after loading. Build or load additional models explicitly:

```python
left = torch_ffi.create_context(model=left_model, foot_pairs=[], com_body_id=left_root_body)
right = torch_ffi.create_context(model=right_model, foot_pairs=[], com_body_id=right_root_body)
q1, v1, warm1 = left.step(qpos_left, qvel_left, warm_left, finger_ctrl_left, friction_left, com_left,
                        qfrc_applied_left, kp_scale_left, substeps=1)
# Equivalently: torch_ffi.step(..., context=left).
```

Each context has `step`, `forward`, `backward`, `kinematics`, `init`, `tape_bytes`, `diagnostics`, `read_diagnostics`
and `set_diagnostics` methods. Native dimensions and artifact facts are in `context.info`. The optional numeric
`com_body_id` is a numeric body ID for one FREE root, or an ordered tuple of two body IDs for a coupled two-FREE
model. Each body's authored local inertial position receives its own three-component offset: `com` is `(B, 3*nr)`
and the G1 default remains `(B, 3)` with its torso binding. `context.info` records `nr`, `com_width`,
`com_body_ids` and the actual FREE Q/V addresses. Contexts are retained for the process lifetime because delayed autograd and
captured CUDA graphs reference their libraries and constants. There is no model reload or global constant switch.

Custom ops register once and carry both the process-local context ID and the existing artifact model key.
The key distinguishes model shapes in persistent compiler caches when a new process reuses a context number.
Their fake shapes use that context's Q/V, actuator, body and tape dimensions. Forward saves both identities for
delayed backward, including after another model has run. Prepare each context on every target device with `context.init(device)` before capture.
`scripts/torch_context_check.py` checks routing and fake shapes on CPU; `--cuda ARTIFACT_A ARTIFACT_B` additionally
checks real delayed VJPs and compiled/graph execution on CUDA. CPU routing checks are not CUDA qualification.

## Rollouts and memory

A rollout is a Python loop over `step`. Each differentiated step keeps its tape until the backward:
`tape_bytes(4096) = 96.5 MB` per control step at 4096 worlds (23.6 kB per world), 6.2 GB for 64 steps. `torch.utils.checkpoint` can drop
the tapes and rerun the steps in the backward: the forward is deterministic, so a recomputed tape is bitwise the first
one.

## CUDA graphs and `torch.compile`

After `init(device)`, `step`, its backward and `kinematics` only launch kernels on the current stream and allocate
through PyTorch's caching allocator: they never synchronize, so a rollout with fixed shapes can be captured with
`torch.cuda.graph` (call `init` before the capture; a capture that would upload the constants raises). The library
calls are `torch.library` custom ops (`nablaengine::forward`, `nablaengine::backward`, `nablaengine::kinematics`,
`nablaengine::kinematics_backward`) with fake implementations, so `torch.compile(..., fullgraph=True)` keeps them as
opaque nodes without graph breaks. In eager mode the same functions are called directly, without the custom ops'
dispatch.

## Diagnostics

Every forward and backward adds its counts (worlds at the solver's iteration limit, tape flags, rows with non-finite
gradients or cotangents, replay mismatches; `jax_ffi.DIAG`) to an int64 tensor on its device, with device operations
only. `diagnostics()` returns that tensor without reading it; `read_diagnostics()` copies it to the host (a
synchronization) and by default resets it. The backward's `info` carries the same per-world bits as `jax_ffi.backward`.

The artifact's `maxcon` and `nefc` record its model-derived point and row capacities. A positive model `nconmax`
owns the point budget; automatic models use a bounded 256-point budget, limited by the pair graph with a 32-point
minimum allocation. Rows cover every stored contact plus frictionloss, soft equalities and limits; a larger positive
model `njmax` is preserved. Artifact version 4 records the widened tape counts; older artifacts must be rebuilt.
Contacts or constraint rows beyond a world's compiled capacity are dropped and flagged as in the JAX
interface (`docs/api.md`, Diagnostics, Capacity): `info[:, 3] & OVERFLOW_CONTACTS / OVERFLOW_ROWS` of `forward()` or
`step(..., return_info=True)`, the
counters `fwd_contact_overflow_worlds` / `fwd_row_overflow_worlds`, and with `set_overflow_warnings(True)` (or
`DK_OVERFLOW_WARN=1`; off by default) a `RuntimeWarning` from `read_diagnostics()` and once more at exit for counts
nobody read.

`set_diagnostics(False)` switches the counters and the backward's non-finite bits off (the backward's `info` then holds
the adjoint's flags only); outputs and gradients are the same. The counters are one small kernel per forward and per
backward in the library, so they cost almost nothing (V100, 4096 worlds: forward + backward 10.56 ms on, 10.53 ms off).
A captured CUDA graph keeps the setting it was captured with.

## Several GPUs

Each call runs on the device of its inputs (all on one device), whatever the current device is; the model constants
are uploaded to each device on its first use (`init(device)` does it ahead, e.g. before a capture), and the counters
are kept per device. Data-parallel use puts one batch of worlds on each GPU. On GPUs of one architecture the
results are bitwise the same on every device (`scripts/torch_ffi_check.py`, section `multigpu`, which also checks
`jax_ffi` on several JAX devices).

