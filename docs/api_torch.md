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

On a V100 (sm_70) install a PyTorch build that still targets sm_70, e.g. the CUDA 12.6 wheels
(`--index-url https://download.pytorch.org/whl/cu126`); the CUDA 12.8 and 12.9 wheels start at sm_75.

## Entries

| entry | as in jax_ffi | notes |
|---|---|---|
| `init(device=None, artifact=None, model_dir=None, model=None, foot_pairs=None, terrain=None, com_body=None)` | `init` | loads the artifact (building it if needed) and uploads the model constants to `device` (default: the current CUDA device); the first call on a device synchronizes it; one architecture per process |
| `build(arch, ...)`, `python -m nablaengine.torch_ffi build` | | the artifact for an architecture (mujoco and nvcc needed) |
| `step(qpos, qvel, warm, ctrl, friction, com, *, iterations=10, substeps=4, clamp_kv=0, normal=3, clamp_soft=None, grad_decay=1.0, kinematics=False)` | `step` | a `torch.autograd.Function`; the tape is written only when an input requires grad; `warm'` is not differentiable |
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

Contacts or constraint rows beyond a world's capacity (64 contacts, 192 rows) are dropped and flagged as in the JAX
interface (`docs/api.md`, Diagnostics, Capacity): `info[:, 3] & OVERFLOW_CONTACTS / OVERFLOW_ROWS` of `forward()`, the
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

