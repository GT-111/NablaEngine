# Configuration: environment variables and compile-time switches

Two different kinds of knobs exist, and they are easy to confuse because both are spelled `DK_*`:

- **Environment variables** are read by Python (or by a check binary) at run time.
- **Compile-time switches** are C preprocessor macros of the CUDA source (`-DNAME=VALUE`). They take effect only in a
  build that passes them: `scripts/engine_step_check.py --define NAME=VALUE`, or a check script that sets them for its
  own builds. Exporting one as an environment variable does nothing: `export DK_ADJ_CLAMP_KV=0` changes no build. The
  JAX interface's library (`nablaengine.jax_ffi`) is built with every other macro at its default and one translation
  unit per adjoint variant (with `DK_SM120_FRONTEND=split` also a forward / kinematics unit of its own); its
  `clamp_kv` and `normal` arguments choose the variant at call time. The build target's own architecture flags
  (`nablaengine.cuda.arch_flags`: on sm_120 every unit from compute_90 PTX by default) and `-D` options
  (`nablaengine.cuda.unit_defines`: the register caps of `DK_SM90_REG_CAPS` and `DK_SM120_PRE_CAP`) go to every build
  in the same way: the FFI library and the check binaries.

"Bitwise" below says whether a non-default value changes the engine's results bit for bit (forward outputs, tape or
gradients) compared with the default build.

The benchmark harness (`benchmarks/README.md`: `DK_BENCH_*`) and the timing kits (`scripts/paper_bench/README.md`:
`PAPER_BENCH_*`; `scripts/physx_bench/README.md`: `PHYSX_BENCH_*`) document their own variables; none of them changes
the engine's results.

## Environment variables

| Variable | Default | Read by | What it does | Bitwise |
|---|---|---|---|---|
| `NVCC` | search order below | `nablaengine.cuda.nvcc()` | The nvcc of every engine build. Search order: `$NVCC`, `nvcc` on `PATH`, `$CUDA_HOME/bin/nvcc`, a pip `nvidia.cuda_nvcc` that contains nvcc (the `nvidia-cuda-nvcc-cu12` wheel does not: only ptxas), `/usr/local/cuda/bin/nvcc`. Use CUDA 12.x: CUDA 13 dropped sm_70, sm_120 needs 12.8 or newer. | yes: another nvcc release gives other binaries (the FFI library's cache key includes `nvcc --version`; the forward digests are keyed by the release) |
| `CUDA_HOME` | unset | `cuda.nvcc()` | `$CUDA_HOME/bin/nvcc` is a candidate. | as `NVCC` |
| `DK_ARCH` | the first visible GPU's | `cuda.arch()`, `jax_ffi` | The build target (`-arch=sm_XY`), e.g. `sm_70` (V100), `sm_86` (RTX 3090), `sm_90` (H100), `sm_120` (RTX PRO 6000). Set it to build without a GPU (`DK_ARCH=sm_120 JAX_PLATFORMS=cpu`). | yes when it is not the GPU's own architecture (a warning says so): code for an older target runs that target's code paths, e.g. `CHOL_COMPACT` |
| `DK_SM120_FRONTEND` | `n7` | `cuda.arch_flags()`: every build (check binaries, FFI library) | On sm_120, `n7`: every unit (forward / kinematics `k_pre`, `k_post`, `k_substep`, `k_kin`, `k_kin_vjp`; the 12 adjoint variants; the FFI unit) is built from compute_90 PTX, the NVVM 7 front end (`-arch=compute_90 -code=sm_120`). On the RTX PRO 6000 the native forward is 4.75% slower and the native adjoint 12% slower (the NVVM 7 adjoint 0.893: about 127 registers against 238, 9 worlds per SM instead of 8). `split`: only the forward / kinematics unit NVVM 7, the adjoint units `-arch=sm_120` (NVVM 20 with CUDA 12.8 / 12.9). `native`: every unit `-arch=sm_120`. `n7` / `split` apply only where the PTX headers confirm compute_90 is NVVM 7 and compute_120 is not (otherwise every unit native, with a note on stderr). In the FFI library's cache key; `jax_ffi.init()` reports `arch_flags` and `front_end`. Other architectures ignore it. | yes on sm_120: the forward's bits between `native` and `n7` / `split`, the gradients' between `n7` and `split` / `native`; no elsewhere |
| `DK_SM120_FORWARD_FRONTEND` | unset | `cuda.frontend()` | Deprecated alias of `DK_SM120_FRONTEND`, read only where that is unset, with a DeprecationWarning: `n7` -> `n7` (every unit), `native` -> `native`. | as `DK_SM120_FRONTEND` |
| `DK_SM90_REG_CAPS` | `1` | `cuda.unit_defines()` (`REG_CAPS`): every build of the forward kernels | On sm_90 / sm_90a the unit with the forward kernels gets `-DSPLIT_PRE_MINB=16 -DSPLIT_POST_MAXNREG=128`: `k_pre` at 128 registers (no spills), `k_post` `__maxnreg__(128)` (some spills), 16 worlds per SM each instead of 12; forward 10.8% faster on the H100 SXM5 (4096 worlds). `0`: the launch bounds of the other targets. Decided by the build target, not by `__CUDA_ARCH__` (sm_120's forward unit is compiled from compute_90 PTX); other targets ignore it. A caller's `-DSPLIT_PRE_MINB` / `-DSPLIT_POST_MAXNREG` / `-DSPLIT_POST_MINB` replaces the cap of that kernel. Needs nvcc 12.4+ (`__maxnreg__`): with an older nvcc the build keeps the launch bounds and says so on stderr. In the FFI library's cache key; `jax_ffi.init()` reports the applied options as `forward_defines`. | no (checked on the H100 and on the V100: outputs, contacts and tape; register allocation only) |
| `DK_SM120_PRE_CAP` | `1` | `cuda.unit_defines()` (`REG_CAPS`): every build of the forward kernels | On sm_120 with the NVVM 7 forward (`DK_SM120_FRONTEND` `n7` or `split`, not `native`) the unit with the forward kernels gets `-DSPLIT_PRE_MINB=16`: `k_pre` 162 -> 128 registers without spills (ptxas 12.9), 1.3% faster forward on the RTX PRO 6000 (shared memory keeps `k_pre` at 12 worlds per SM, so it is code generation only); `k_post`'s `__maxnreg__(128)` is not taken there. `0`: without it. A caller's `-DSPLIT_PRE_MINB` replaces it. In the FFI library's cache key and `init()`'s `forward_defines`. | no (register allocation only) |
| `DK_TEST_FORWARD_VIRTUAL_ARCH` | unset | `cuda.virtual_arch()` | Testing only: the forward / kinematics unit from this virtual architecture's PTX on any GPU (e.g. `compute_70` on a V100: `-arch=compute_70 -code=sm_70`), so the split structure (`DK_SM120_FRONTEND=split`: a forward unit of its own, check binaries in pairs) can be exercised elsewhere; it warns when set. | no where that PTX is the native one (compute_70 on sm_70); otherwise as a front-end change |
| `DK_TEST_VIRTUAL_ARCH` | unset | `cuda.virtual_arch()` | Testing only: every unit from this virtual architecture's PTX on any GPU (e.g. `compute_70` on a V100), so sm_120's default structure (one set of flags for every unit) can be exercised elsewhere; it warns when set. `DK_TEST_FORWARD_VIRTUAL_ARCH` takes precedence for the forward unit. | no where that PTX is the native one (compute_70 on sm_70); otherwise as a front-end change |
| `CUDA_VISIBLE_DEVICES` | all GPUs | CUDA, `cuda.arch()`, the scripts | The GPU(s) a process uses; `cuda.arch()` and the result JSONs' GPU record follow its first entry. | no (on one GPU type) |
| `DK_FIXTURES` | unset (or empty): `<checkout>/fixtures/mjlab` | `nablaengine.fixtures.fixtures_dir()`, which `jax_ffi.init()` and every script use | The fixtures directory: the model (`g1_train.mjb`, `g1_meta.json`), the MuJoCo Warp references (`mjw317_<corpus>.npz`) and the forward digests; a relative path is taken from the current directory. Outside a source checkout set it, or call `jax_ffi.init(model=, foot_pairs=)`. | selects data |
| `DK_BENCH_DATA` | unset (or empty): `<checkout>/benchmarks/data` | `nablaengine.fixtures.bench_dir()` | Where the corpora (`<name>.npz`) and their truth files (`truth/<name>.npz`) are read from by the library and the checks (`fixtures.load_corpus`, `load_truth`). | selects data |
| `DK_DIGESTS` | unset | `nablaengine.checks` (the forward digest check of `engine_step_check.py` and `jax_ffi_check.py`) | A further manifest of forward digests, read after `fixtures/mjlab/engine_digests.json` (for example entries made on another GPU architecture). | no (which digests a run is compared with) |
| `DK_CACHE_DIR` | `$XDG_CACHE_HOME/nablaengine`, else `~/.cache/nablaengine` | `jax_ffi.cache_dir()`, `torch_ffi.cache_dir()` | Where the FFI library (`libdk_g1_<arch>_<key>.so`, about 5.5 MB per architecture and key, never pruned) is kept. | no |
| `MAX_JOBS` | process affinity, bounded by `cpu.max` and unit count | `jax_ffi._compile_parallelism()`: JAX and Torch builds | Maximum concurrent CUDA translation units, sharing the process's allocated CPU budget with optimizer threads. | no; scheduling alone does not change the cache key |
| `DK_COMPILE_THREADS` | `1` | `jax_ffi._compile_parallelism()`: JAX and Torch builds | `nvcc --split-compile` optimizer threads per unit, with jobs times threads bounded by the same CPU budget. | values above 1 have a distinct artifact key and need numerical qualification |
| `DK_TORCH_ARTIFACT` | unset: the cache's `torch_<arch>_current.json` | `torch_ffi.init()` | The PyTorch interface's artifact to load (the `.json` that `python -m nablaengine.torch_ffi build` wrote, next to its library and constants), e.g. one built on another machine. | no (a stale or foreign artifact is refused) |
| `DK_OVERFLOW_WARN` | `0` (off) | `jax_ffi`, `torch_ffi` (`set_overflow_warnings`) | `1`: `read_diagnostics()` (and `jax_ffi.decode_diagnostics`) warn when worlds exceeded their compiled contact or constraint-row capacity, and the unread counts are reported at exit. Read actual capacities from context metadata; counters are kept either way. | no |
| `XDG_CACHE_HOME` | `~/.cache` | `jax_ffi.cache_dir()` | See `DK_CACHE_DIR`. | no |
| `DK_MAXCON` | positive model `nconmax`, else 256, bounded by the pair graph with a 32-point minimum allocation | `g1_engine.step.maxcon()` | Overrides contact capacity per world in every build, the FFI library included. An experiment switch: it warns when set. | yes (capacity and memory layout) |
| `DK_KEEP_WORK` | unset: removed | `nablaengine.workdir` | `1` keeps the scripts' temporary work directories (generated sources, binaries, inputs, outputs; also `--keep-work`). | no |
| `TMPDIR` | `/tmp` | the scripts' work directories | Where work directories are made (a check binary's run needs a few hundred MB while it runs). | no |
| `DK_SPLIT_ONLY` | unset | `engine_step_check.py`'s binary | `1` / `2`: the timed launches run only `k_pre` / `k_post` (per-kernel time; results invalid). | timing only |
| `DK_TAPE_IN` | unset | the adjoint check binary (`nablaengine.checks`) | Internal (set by the scripts for split builds, where the forward unit has other flags than the adjoint, as on sm_120): the adjoint runs on another binary's tape. | n/a |
| `CUOBJDUMP` | next to nvcc, then `PATH`, then a conda package cache (`$CONDA_PKGS_DIRS`, `$CONDA_EXE`'s installation, `~/miniconda3`, `~/miniforge3`, `~/anaconda3`) | `scripts/identity_gate.py` (`nablaengine.identity.find_tool`) | The CUDA binary tool for the SASS digests of the identity gate. Its cuobjdump must know every target it disassembles (sm_120: CUDA 12.8 or newer) and is part of the manifest's toolchain key. | no (reporting) |
| `PTXAS` | unset: the pip package `nvidia.cuda_nvcc`'s ptxas (the JAX stack's), then the one next to nvcc | `scripts/identity_gate.py` (`nablaengine.identity.find_ptxas`) | A ptxas for a build target the local nvcc cannot compile for (sm_120 with nvcc 12.4): the gate builds such a unit from its virtual architecture's PTX (`-arch=compute_90`) and assembles it with this ptxas (`-arch=sm_120 -m64`, as nvcc runs it); the first candidate that assembles for the target is used. Never read by the engine's own builds. | no (the identity gate only) |
| `DK_ALLOW_UNSUPPORTED` | unset: every capability error raises | `nablaengine.model.capability.require()`: `jax_ffi.init`, `jax_ffi.check_model`, `EnginePhysics`, `fixtures.load_model`, `g1_engine.kinematics.check_model` | Escape hatch for experiments: a comma-separated list of capability codes (e.g. `H22,H23`; `all` for every code) whose errors become a `RuntimeWarning` instead of `UnsupportedModel`. The engine then simulates other physics than MuJoCo for that model (the codes say what it leaves out); results of such a build are not comparable with the checks'. Never set it for the G1, which has no capability errors. | yes for a model it lets through (that is its point); no effect on a supported model |
| `ROBOT_DROP_BOX` | unset | `scripts/robots/robot_check.py` | `1`: the robot model without its box-geom pairs (an experiment for comparing with and without box contacts). | yes (other contacts) |
| `JAX_PLATFORMS` | JAX's | JAX | Several scripts set `cpu`, and `reference/jax64.py` sets it (and x64) at import. | no |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | JAX's (`true`) | JAX | Set `false` when JAX shares a GPU with another engine (16 GB cards); `verify.sh` sets it. | no |

## Compile-time switches of the engine

Defaults are those of the source (`g1_engine/*.py`); a model-derived default is computed by the Python code that
generates the header.

| Macro | Default | Where | What it does | Bitwise |
|---|---|---|---|---|
| `DK_ADJ_CLAMP_KV` | `1` | `adjoint.py` | Clamped actuators in the adjoint: 0 keeps -kv lambda (as MuJoCo Warp's adjoint), 1 drops it (exact), 2 soft clamp (delta at run time). The FFI's `clamp_kv` (default 0). | yes (gradients) |
| `DK_ADJ_NORMAL` | `3` | `adjoint_contact.py` | Normal derivative of the sphere core of self contacts: 0 exact, 1 frozen frame, 2 regularized, 3 frozen below eps. The FFI's `normal` (default 3). | yes (gradients) |
| `DK_ADJ_NORMAL_EPS` | `1e-3f` | `adjoint_contact.py` | The eps (m) of `DK_ADJ_NORMAL` 2 and 3. | yes (gradients) |
| `DK_ADJ_PARALLEL` | `DK_ADJ_NORMAL != 0` | `adjoint_contact.py` | 1 holds near-parallel free capsule-capsule segment parameters constant (finite gradients); 0 exact. | yes (gradients of near-parallel states; the corpora are bitwise unchanged) |
| `DK_ADJ_PARALLEL_EPS` | `1.2e-3f` | `adjoint_contact.py` | The sin^2(theta) threshold of `DK_ADJ_PARALLEL`. | yes (gradients) |
| `DK_ADJ_STAGE` | `3` | `adjoint.py` | Stages of the adjoint, for bisection: 1 smooth dynamics, integrator, actuators; 2 + limit rows and plane contacts; 3 + self contacts. | yes |
| `DK_ADJ_WORLDS_PER_SM` | `-1` | `adjoint.py` | Worlds per SM of `k_adjoint`: -1 the GPU's choice (4 on V100, no cap elsewhere), 0 as many as fit, N at most N. | no (scheduling) |
| `DK_ADJ_DEBUG` | undefined | `adjoint.py` | Writes the per-substep intermediates of the adjoint with MuJoCo Warp's adjoint names (bisection against it). | no (extra outputs; checked equal to the production build) |
| `DK_SPLIT` | `1` | `host.py` | Forward: 1 the split kernels `k_pre` / `k_post`, 0 the fused `k_substep` (`engine_step_check.py --fused`; the adjoint check binary uses it). | no (checked: outputs, contacts, tape) |
| `DK_SPLIT_ORDER` | `1` | `host.py` | `k_post` dispatched by row count, most rows first (`SPLIT_NBUCKET` buckets; at most 32 substeps per call). | no (scheduling) |
| `SPLIT_NBUCKET`, `SPLIT_ROW_SHIFT` | `64`, `1` | `host.py` | `DK_SPLIT_ORDER`'s buckets: their number and width (2^shift rows; bucket 0 also takes the worlds above the range). | no (scheduling) |
| `SPLIT_PRE_MINB`, `SPLIT_POST_MINB` | `12` (sm_90 and sm_120's NVVM 7 forward: `SPLIT_PRE_MINB=16`, see `DK_SM90_REG_CAPS`, `DK_SM120_PRE_CAP`) | `host.py` | `__launch_bounds__` minimum blocks per SM of `k_pre` / `k_post` (a register cap); 0 none. | checked unchanged at the default and at 16 (sm_90, sm_120, V100); other values: check (register pressure can change ptxas's contraction) |
| `SPLIT_POST_MAXNREG` | undefined (sm_90: `128`, see `DK_SM90_REG_CAPS`) | `host.py` | Defined and > 0: `k_post`'s register cap as `__maxnreg__(N)` (nvcc 12.4+) in place of `__launch_bounds__(32, SPLIT_POST_MINB)`; 0 or undefined: the launch bound. | no at 128 (checked on the H100 and the V100: outputs, contacts and tape); other values: check |
| `SPLIT_NCON`, `SPLIT_NEFC` | One root or `DK_TAPE_VERIFY`: `min(MAXCON, 32)`, `min(NEFC, 128)`; other two-root builds: `min(MAXCON, 96)`, `min(NEFC, NFLOSS + NEQ + NV + 4 * SPLIT_NCON)` | `step.py` | `k_post`'s shared-memory capacity; larger worlds run the same algorithm on the global-memory state. The verification matrix needs the smaller default to fit the normal 48 KiB shared-memory limit for all supported models. | check (CUDA parity and performance for the new two-root default are pending) |
| `DK_SPLIT_ALL_BIG` | undefined | `host.py` | Debug: every world on the global-memory state. | no (checked) |
| `DK_UNIFORM`, `DK_JTF_GATHER`, `DK_JV_VEC`, `DK_HESS_DOFPAR`, `DK_GRAD_COST` | `1` | `step.py` | Optimization toggles (warp-uniform votes, J'f gather, 128-bit loads, dof-parallel Hessian rows, cost in the gradient reduction). | no (bitwise equal to all off) |
| `NEWTON_REUSE` | `NV > 32 && NV <= 35` (on for the G1) | `step.py` | Reuse the last Cholesky factor when no row changed zone. | yes: 0 differs by roundings in 90-96% of worlds |
| `CHOL_BCAST` | `NV > 32 && NV <= 35` (on for the G1) | `source.py` | The broadcast Cholesky; with 0, `CHOL_COMPACT` picks one of the others. | yes (another factorization) |
| `CHOL_COMPACT` | 1 below sm_80, else 0 | `source.py` | With `CHOL_BCAST=0`: the compact (1) or unrolled (0) register Cholesky. Follows `__CUDA_ARCH__`. | yes |
| `CHOL_INLINE` | `__noinline__` | `source.py` | Inlining of the unrolled register Cholesky. | can (contraction follows the context) |
| `DK_INL_GRAD`, `DK_INL_MATVEC`, `DK_INL_COLLIDE`, `DK_INL_ROWSJV`, `DK_INL_COST` | `__noinline__`, `__forceinline__`, empty, empty, empty | `step.py` | Inlining of helpers (code size against call overhead). | can (contraction follows the context); check with `--tape --save-outputs` |
| `DKC` | `__device__` | `source.py` | Memory space of the model arrays (global, L1-cached); `__constant__` for comparison. | performance experiment; not checked |
| `BOX_BOX_CCD` | `1` | `step.py` | Box-box contacts (a model with boxes only; the mjlab G1 has none): 1 as MuJoCo Warp's native CCD single contact, 0 the full separating-axis primitive. | yes (box-box contacts) |
| `DK_PROFILE` | undefined | `step.py`, `host.py`, `adjoint.py` | Per-phase cycle counters (`engine_step_check.py --profile`). | timing build (the phase marks synchronize in every build) |
| `DK_TAPE_VERIFY` | undefined | `step.py` | Debug: re-assembles the Newton Hessian at the final zones and flags differences in the tape (bit 11). | debug |
| `KIN_WPB` | `4` | `kinematics.py` | Worlds per block of the kinematics kernels. | no (layout) |

Generated from the model, not meant to be overridden: `MAXCON` (`step.maxcon`, see `DK_MAXCON`), `NEFC` (stored contact rows, frictionloss, equalities and hinge limits, preserving larger positive `njmax`),
`HAS_BOX`, `FLOOR_G`, `DOF_BODY_ROOT6`, `SOLIMP_POW2`, `KDOF` (`step.max_support`: the most dofs a contact Jacobian
touches, 16 for the G1; the `DK_JV_VEC` loads need 16), `ADJ_SOLIMP_POWER` (defined only when every row has the same
solimp power, which the adjoint then uses as a constant), `NQ`, `NV`, `NU`, `NB`, the other model sizes (`NPAIR`,
`NGEOMC`, ...) and the tape layout (`REC_WORDS`, `OFF_*`).
