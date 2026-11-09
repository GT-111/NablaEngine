"""PyTorch interface of the engine: the same CUDA library and semantics as nablaengine.jax_ffi, on torch's current CUDA
stream.

  step(qpos, qvel, warm, ctrl, friction, com, qfrc_applied=None, kp_scale=None, *, iterations=10, substeps=4, clamp_kv=0, normal=3, clamp_soft=None,
       grad_decay=1.0, kinematics=False, return_info=False) -> (qpos', qvel', warm'[, info][, qpos_n, xpos, xquat, cvel, subtree_com])
  kinematics(qpos, qvel, com) -> (qpos_n, xpos, xquat, cvel, subtree_com)
  create_context(...) -> ModelContext (independent model library and dimensions)
  scale_grad(tree, gamma)

Arguments and outputs as jax_ffi.step / jax_ffi.kinematics (see that module's docstring): float32 tensors on one CUDA
device, contiguous, leading dimension B (the worlds): qpos (B, nq), qvel (B, nv), warm (B, nv), ctrl (B, nu), friction
(B,), com (B, 3*nr) for one or two FREE roots (mjlab G1: nr 1, nq 36, nv 35, nu 29). Other dtypes, devices or shapes raise TypeError; nothing is cast or
copied implicitly. step is differentiable (torch.autograd): when any input requires grad, the forward also writes the
tape (B, substeps * REC_WORDS float32, 5888 B per world and substep) and keeps it for the backward (the analytic
adjoint, deterministic), which returns d/dqpos (36-dimensional qpos form), d/dqvel, d/dctrl and d/dqfrc_applied,
no gradient for warm, friction, com and kp_scale. Omitted qfrc_applied/kp_scale retain the default JAX semantics.

A rollout is a Python loop: every differentiated step keeps its tape until the backward (tape_bytes(); 6 GB for
4096 worlds x 64 steps). torch.utils.checkpoint may rerun a step: the forward is deterministic, so a recomputed tape
is bitwise the first one.

The library: the engine's CUDA source for the model (jax_ffi.build_plan's units without the XLA FFI unit, plus
dk_torch.cu's diagnostic counters), compiled by build() for one architecture into an artifact in the cache directory
(libdk_g1_torch_<arch>_<key>.so with the CUDA runtime linked statically, the packed constants .npy, the model facts
.json): `python -m nablaengine.torch_ffi build [--arch sm_XY ...]` where mujoco 3.17 and nvcc are installed. init()
loads the artifact for the GPU's architecture (or $DK_TORCH_ARTIFACT) with ctypes: a process that only loads it needs
torch, NumPy and the NVIDIA driver, and imports no JAX, no mujoco and no nvcc. The artifact records a hash of the
package sources it was generated from (sources_sha256); a stale one is rebuilt where nvcc and mujoco are available,
else init() raises. Every call goes to the library's C entries (<unit>_forward, <unit>_adjoint, <unit>_kinematics,
<unit>_kinematics_vjp) with torch's current stream and the tensors' device pointers, the inputs' device made current.
Each ModelContext owns its library, immutable constants, dimensions, diagnostics and per-device readiness.
Module calls retain a default context; create_context() adds models without replacing it. Model constants are
uploaded to a device on its first use, which synchronizes once: call context.init(device) before
capturing a CUDA graph. After that, step, its backward and kinematics only launch kernels and allocate through torch's
caching allocator (no host synchronization), so they can be captured in a CUDA graph with fixed shapes. The four
entries are torch.library custom ops (nablaengine::forward, backward, kinematics, kinematics_backward) with fake
implementations while torch.compile traces (opaque nodes); eager calls run the same functions directly.

Diagnostics (set_diagnostics switches them): per device, the counters of jax_ffi.DIAG (forward calls and worlds, worlds at the iteration limit,
unsupported worlds, tape flags; backward calls and rows, rows with non-finite gradients or cotangents, replay
mismatches) are added up on the device by every call, in an int64 tensor that is never read implicitly:
diagnostics(device) returns it, read_diagnostics() reads it into a dict (a synchronization). The backward's info
(B,) carries the adjoint's flags | NONFINITE_* << NONFINITE_SHIFT as jax_ffi.backward's.
"""
import ctypes
import hashlib
import json
import math
import os
import tempfile
import threading
from pathlib import Path

import numpy as np

# the interface constants of nablaengine.jax_ffi (repeated here so that the run time imports neither jax_ffi nor mujoco;
# scripts/torch_ffi_check.py checks that they are equal)
ITERATIONS, SUBSTEPS, CLAMP_KV, NORMAL = 10, 4, 0, 3
CLAMP_SOFT, GRAD_DECAY = 0.2, 1.0
DIAG = ("fwd_calls", "fwd_worlds", "fwd_unsupported_worlds", "fwd_iteration_limit_worlds", "tape_records",
        "refact_records", "nonconv_records", "singular_records", "bwd_calls", "bwd_rows", "nonfinite_rows",
        "nonfinite_qpos_rows", "nonfinite_qvel_rows", "nonfinite_ctrl_rows", "nonfinite_cotangent_rows",
        "replay_mismatch_worlds", "fwd_contact_overflow_worlds", "fwd_row_overflow_worlds")
OVERFLOW_CONTACTS, OVERFLOW_ROWS = 4, 16  # forward info[:, 3] bits: contacts / constraint rows beyond capacity dropped
_OVERFLOW_WARN = [os.environ.get("DK_OVERFLOW_WARN", "0") not in ("", "0")]  # set_overflow_warnings
NONFINITE_QPOS, NONFINITE_QVEL, NONFINITE_CTRL, NONFINITE_COT, NONFINITE_SHIFT = 1, 2, 4, 8, 16
VARIANTS = tuple((kv, n) for kv in (0, 1, 2) for n in (0, 1, 2, 3))  # adjoint builds (clamp_kv, normal)
FWD_UNIT = "dk_fwd"
ARTIFACT_VERSION = 4
# the package files the library is generated from: an artifact records their hash, a different hash means stale
SOURCES = ("cuda.py", "jax_ffi.py", "torch_ffi.py", "fixtures.py", "engines.py", "ccd.py", "terrain.py", "g1_engine/*.py",
           "model/*.py")


def _tag(v):
    return f"dk_kv{v[0]}_n{v[1]}"


_LOCK = threading.Lock()
_CONTEXTS = {}  # stable integer -> ModelContext; also keeps recorded autograd/CUDA graphs alive
_DEFAULT_CONTEXT = None
_DEFAULT_DIAGNOSTICS_ON = True  # preserves set_diagnostics() before the default context is loaded


class ModelContext:
    """One immutable model library, dimensions, constants and per-device diagnostics.

    Construct with create_context(). A context never replaces another model's
    constants; autograd and compiled graphs retain its context_id and artifact
    model key. Integer context IDs alone can repeat in another process.
    """

    def __init__(self, context_id):
        self.context_id = context_id
        self._state = {}
        self._ready = set()
        self._diag = {}
        self._diagnostics_on = True

    @property
    def info(self):
        return self._state["info"]

    def init(self, device=None):
        with _LOCK:
            return _init_context(self, _device_index(device))

    def step(self, *args, **kwargs):
        return step(*args, context=self, **kwargs)

    def forward(self, *args, **kwargs):
        return forward(*args, context=self, **kwargs)

    def backward(self, *args, **kwargs):
        return backward(*args, context=self, **kwargs)

    def kinematics(self, *args, **kwargs):
        return kinematics(*args, context=self, **kwargs)

    def tape_bytes(self, nworld, substeps=SUBSTEPS, steps=1):
        return tape_bytes(nworld, substeps, steps, context=self)

    def diagnostics(self, device=None):
        return diagnostics(device, context=self)

    def read_diagnostics(self, device=None, reset=True):
        return read_diagnostics(device, reset, context=self)

    def set_diagnostics(self, enabled):
        return set_diagnostics(enabled, context=self)


def _err(msg, exc=TypeError):
    return exc(f"nablaengine.torch_ffi: {msg}")


TORCH_UNIT = r"""
#include <cuda_runtime.h>
// the diagnostic counters of jax_ffi's FFI unit (k_diag_fwd / k_diag_bwd, the same counts) into a caller's
// (NDIAG,) uint64 array: one launch per forward / backward instead of one elementwise kernel per count
@DEFINES@
namespace {
__device__ __forceinline__ void diag_add(unsigned long long* d, int k, unsigned v) {
  __syncwarp();
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0 && v) atomicAdd(&d[k], (unsigned long long)v);
}
__device__ __forceinline__ bool all_finite(const float* x, int n) {
  bool ok = true;
  for (int i = 0; i < n; i++) ok &= isfinite(x[i]);
  return ok;
}
__global__ void k_tdiag_fwd(unsigned long long* d, int nw, int nsub, int iterations, const int* info,
                            const float* tape) {
  unsigned unsup = 0, lim = 0, refact = 0, nonconv = 0, sing = 0, ocon = 0, orow = 0;
  for (int w = blockIdx.x * blockDim.x + threadIdx.x; w < nw; w += gridDim.x * blockDim.x) {
    unsup += info[4 * w + 3] != 0;
    ocon += (info[4 * w + 3] & OVERFLOW_CONTACTS) != 0; orow += (info[4 * w + 3] & OVERFLOW_ROWS) != 0;
    lim += info[4 * w] >= iterations;
    if (tape)
      for (int k = 0; k < nsub; k++) {
        unsigned f = __float_as_uint(tape[((size_t)w * nsub + k) * REC_WORDS + OFF_FLAGS]);
        refact += (f & TAPE_REFACT) != 0; nonconv += (f & TAPE_NONCONV) != 0; sing += (f & TAPE_SINGULAR) != 0;
      }
  }
  diag_add(d, D_FWD_UNSUPPORTED_WORLDS, unsup); diag_add(d, D_FWD_ITERATION_LIMIT_WORLDS, lim);
  diag_add(d, D_REFACT_RECORDS, refact); diag_add(d, D_NONCONV_RECORDS, nonconv);
  diag_add(d, D_SINGULAR_RECORDS, sing);
  diag_add(d, D_FWD_CONTACT_OVERFLOW_WORLDS, ocon); diag_add(d, D_FWD_ROW_OVERFLOW_WORLDS, orow);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd(&d[D_FWD_CALLS], 1ull); atomicAdd(&d[D_FWD_WORLDS], (unsigned long long)nw);
    if (tape) atomicAdd(&d[D_TAPE_RECORDS], (unsigned long long)nw * nsub);
  }
}
__global__ void k_tdiag_bwd(unsigned long long* d, int nw, const float* gq, const float* gv, const float* gu,
                            const float* cq, const float* cv, int* info) {
  unsigned rows = 0, rq = 0, rv = 0, ru = 0, rc = 0, rm = 0;
  for (int w = blockIdx.x * blockDim.x + threadIdx.x; w < nw; w += gridDim.x * blockDim.x) {
    unsigned b = (all_finite(gq + (size_t)w * NQ, NQ) ? 0u : (unsigned)NONFINITE_QPOS)
                 | (all_finite(gv + (size_t)w * NV, NV) ? 0u : (unsigned)NONFINITE_QVEL)
                 | (all_finite(gu + (size_t)w * NU, NU) ? 0u : (unsigned)NONFINITE_CTRL)
                 | (all_finite(cq + (size_t)w * NQ, NQ) && all_finite(cv + (size_t)w * NV, NV) ? 0u
                                                                                                : (unsigned)NONFINITE_COT);
    int f = info[w];
    info[w] = f | (int)(b << NONFINITE_SHIFT);
    rows += (b & 7u) != 0; rq += (b & NONFINITE_QPOS) != 0; rv += (b & NONFINITE_QVEL) != 0;
    ru += (b & NONFINITE_CTRL) != 0; rc += (b & NONFINITE_COT) != 0; rm += (f & ADJ_REPLAY_MISMATCH) != 0;
  }
  diag_add(d, D_NONFINITE_ROWS, rows); diag_add(d, D_NONFINITE_QPOS_ROWS, rq); diag_add(d, D_NONFINITE_QVEL_ROWS, rv);
  diag_add(d, D_NONFINITE_CTRL_ROWS, ru); diag_add(d, D_NONFINITE_COTANGENT_ROWS, rc);
  diag_add(d, D_REPLAY_MISMATCH_WORLDS, rm);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd(&d[D_BWD_CALLS], 1ull); atomicAdd(&d[D_BWD_ROWS], (unsigned long long)nw);
  }
}
unsigned diag_grid(long long nw) { long long g = (nw + 255) / 256; return (unsigned)(g < 1 ? 1 : g > 160 ? 160 : g); }
}  // namespace
extern "C" int dk_torch_diag_fwd(cudaStream_t st, unsigned long long* d, int nw, int nsub, int iterations,
                                 const int* info, const float* tape) {
  k_tdiag_fwd<<<diag_grid(nw), 256, 0, st>>>(d, nw, nsub, iterations, info, tape);
  return (int)cudaGetLastError();
}
extern "C" int dk_torch_diag_bwd(cudaStream_t st, unsigned long long* d, int nw, const float* gq, const float* gv,
                                 const float* gu, const float* cq, const float* cv, int* info) {
  k_tdiag_bwd<<<diag_grid(nw), 256, 0, st>>>(d, nw, gq, gv, gu, cq, cv, info);
  return (int)cudaGetLastError();
}
"""


def _torch_unit(m, rec_words=None):
    """The torch library's own unit dk_torch.cu (TORCH_UNIT with the model's sizes and jax_ffi's constants; rec_words:
    the tape record's length, default the model's)."""
    from nablaengine import engines
    E = engines.modules(m)
    g1_adjoint, g1_tape = E.adjoint, E.tape
    defs = [f"#define {k} {v}" for k, v in (
        ("NQ", m.nq), ("NV", m.nv), ("NU", m.nu), ("REC_WORDS", rec_words or g1_tape.rec_words(m)),
        ("OFF_FLAGS", g1_tape.layout(m)["flags"][0]), ("TAPE_REFACT", f"{g1_tape.FLAG_REFACT}u"),
        ("TAPE_NONCONV", f"{g1_tape.FLAG_NONCONV}u"), ("TAPE_SINGULAR", f"{g1_tape.FLAG_SINGULAR}u"),
        ("ADJ_REPLAY_MISMATCH", g1_adjoint.ADJ_REPLAY_MISMATCH), ("NONFINITE_QPOS", NONFINITE_QPOS),
        ("NONFINITE_QVEL", NONFINITE_QVEL), ("NONFINITE_CTRL", NONFINITE_CTRL), ("NONFINITE_COT", NONFINITE_COT),
        ("NONFINITE_SHIFT", NONFINITE_SHIFT), ("OVERFLOW_CONTACTS", OVERFLOW_CONTACTS),
        ("OVERFLOW_ROWS", OVERFLOW_ROWS))]
    defs += [f"#define D_{name.upper()} {i}" for i, name in enumerate(DIAG)]
    return TORCH_UNIT.replace("@DEFINES@", "\n".join(defs))


def cache_dir():
    """$DK_CACHE_DIR, else $XDG_CACHE_HOME/nablaengine, else ~/.cache/nablaengine (as jax_ffi.cache_dir)."""
    d = Path(os.environ.get("DK_CACHE_DIR") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
             / "nablaengine")
    d.mkdir(parents=True, exist_ok=True)
    return d


def sources_sha256():
    """sha256 over the package files the library is generated from (SOURCES, sorted by path) and the default model
    files (fixtures/mjlab: the .mjb and its meta.json) where they are installed."""
    root = Path(__file__).resolve().parent
    files = sorted({f for pat in SOURCES for f in root.glob(pat)})
    fix = Path(os.environ.get("DK_FIXTURES") or root.parent.parent / "fixtures" / "mjlab")
    files += [f for f in (fix / "g1_train.mjb", fix / "g1_meta.json") if f.exists()]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def build(arch, model_dir=None, model=None, foot_pairs=None, set_current=True, terrain=None, com_body=None, com_body_id=None):
    """Generate and compile the library for the model and architecture arch (sm_XY) and write its artifact into the
    cache directory: the library libdk_g1_torch_<arch>_<key>.so (the CUDA runtime linked statically), the packed model
    constants (.npy) and the model facts (.json); returns the .json's path. Needs mujoco 3.17 and nvcc (CUDA 12.x), no
    torch and no JAX; cached (the same sources reuse the library). set_current: make it the architecture's artifact
    that init() loads by default (torch_<arch>_current.json); False for another model's, loaded by its path. terrain:
    a nablaengine.terrain.TerrainConfig for a terrain build (never made current; set_terrain() uploads the terrain).
    com_body: the body that takes the per-world COM offset (as jax_ffi.init)."""
    import nablaengine
    from nablaengine import jax_ffi
    from nablaengine import engines
    from nablaengine.g1_engine import source as g1_source
    from nablaengine.model import capability
    import mujoco
    m, fp, com = jax_ffi._model(model_dir, model, foot_pairs, com_body)
    E = engines.modules(m)
    g1_kin, g1_step, g1_tape = E.kinematics, E.step, E.tape
    com_body_id = com_body_id if com_body_id is not None else (int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, com)),)
    capability.require(m, fp, capability.COM_BODY if com_body_id is None else com_body_id, "engine", "torch_ffi.build")
    plan = jax_ffi.build_plan(m, fp, arch, terrain=terrain, com_body_id=com_body_id)
    tgeoms = None
    if terrain is not None:
        from nablaengine import terrain as _terrain
        set_current = False
        tgeoms = [[int(g), int(p), int(f)] for g, p, f in _terrain.robot_terrain_geoms(m, fp, terrain.geoms)]
    units = {k: v for k, v in plan["units"].items() if k != "dk_ffi.cu"}
    defines = {k: v for k, v in plan["defines"].items() if k in units}
    unit_arch = {k: v for k, v in (plan["unit_arch"] or {}).items() if k in units}
    mc = terrain.maxcon if terrain is not None and terrain.maxcon else g1_step.maxcon(m)
    rec = g1_tape.rec_words(m, mc, g1_tape.kind(terrain))
    units["dk_torch.cu"], defines["dk_torch.cu"] = _torch_unit(m, rec), []  # the device-side diagnostic counters
    if unit_arch:  # the adjoint units' architecture flags (sm_120: every unit through NVVM 7 by default)
        adj = unit_arch.get(f"{_tag(VARIANTS[0])}.cu")
        if adj:
            unit_arch["dk_torch.cu"] = adj
    consts = np.ascontiguousarray(plan["consts"], np.float32)
    path, built, key = jax_ffi._build(units, defines, arch, None, consts, unit_arch or None, prefix="libdk_g1_torch")
    try:  # the kinematics outputs' assumptions (kinematics.check_model); the kinematics entries raise without them
        g1_kin.check_model(m)
        kin_error = None
    except AssertionError as e:
        kin_error = f"the model does not meet the kinematics outputs' assumptions ({e!r})"
    stem = path.with_suffix("")
    nr = len(g1_source.free_roots(m))
    com_ids = (int(m.body("torso_link").id),) if com_body_id is None else tuple(int(x) for x in np.atleast_1d(com_body_id))
    meta = dict(artifact_version=ARTIFACT_VERSION, library=path.name, consts=stem.with_suffix(".npy").name, key=key,
                arch=arch, version=nablaengine.__version__, sources_sha256=sources_sha256(),
                terrain=terrain.key() if terrain is not None else None, terrain_geoms=tgeoms,
                consts_sha256=hashlib.sha256(consts.tobytes()).hexdigest(), rec_words=rec,
                nq=m.nq, nv=m.nv, nu=m.nu, nbody=m.nbody, nr=nr, com_width=3*nr, com_body_ids=com_ids,
                qpos0=np.asarray(m.qpos0).tolist(),
                root_qpos_addresses=[int(m.jnt_qposadr[m.body_jntadr[r]]) for r in g1_source.free_roots(m)],
                root_dof_addresses=[int(m.jnt_dofadr[m.body_jntadr[r]]) for r in g1_source.free_roots(m)],
                maxcon=mc, nefc=g1_step.nefc(m),
                off_flags=int(g1_tape.layout(m)["flags"][0]), forward_unit=plan["forward_unit"],
                variants=[list(v) for v in VARIANTS], kinematics_error=kin_error, timestep=float(m.opt.timestep),
                foot_pairs=[int(p) for p in fp], arch_flags=plan["arch_flags"],
                forward_defines=plan["forward_defines"], built=built,
                default_model=model_dir is None and model is None)
    js = stem.with_suffix(".json")
    # Other ranks can return from the shared library build together. Publish
    # complete files so one rank never loads another's truncated rewrite.
    with tempfile.TemporaryDirectory(prefix=f"artifact_{key}_", dir=js.parent) as directory:
        work = Path(directory)
        constants = stem.with_suffix(".npy")
        np.save(work / constants.name, consts)
        os.replace(work / constants.name, constants)
        (work / js.name).write_text(json.dumps(meta, indent=1) + "\n")
        os.replace(work / js.name, js)
        if set_current:
            current = cache_dir() / f"torch_{arch}_current.json"
            (work / current.name).write_text(json.dumps({"artifact": js.name}) + "\n")
            os.replace(work / current.name, current)
    return js


def _artifact_path(arch, artifact):
    """The artifact to load: the argument, else $DK_TORCH_ARTIFACT, else the cache's current one for arch (None: none
    built yet)."""
    a = artifact or os.environ.get("DK_TORCH_ARTIFACT")
    if a:
        return Path(a)
    ptr = cache_dir() / f"torch_{arch}_current.json"
    return cache_dir() / json.loads(ptr.read_text())["artifact"] if ptr.exists() else None


def _stale(js, arch):
    """Why the artifact js cannot be loaded for arch (None: it can): missing files, another artifact version or
    architecture, or a library generated from other package sources than the installed ones."""
    if js is None or not js.exists():
        return "no artifact"
    meta = json.loads(js.read_text())
    if meta.get("artifact_version") != ARTIFACT_VERSION:
        return f"artifact version {meta.get('artifact_version')}, this module reads {ARTIFACT_VERSION}"
    if meta["arch"] != arch:
        return f"built for {meta['arch']}, the GPU is {arch}"
    if not (js.parent / meta["library"]).exists() or not (js.parent / meta["consts"]).exists():
        return "its library or constants file is missing"
    if meta.get("default_model") and meta["sources_sha256"] != sources_sha256():
        return "built from other package sources (the package changed since): rebuild it"
    return None


def create_context(device=None, artifact=None, model_dir=None, model=None, foot_pairs=None, com_body_id=None, *, terrain=None, com_body=None):
    """Load an independent model and prepare its constants before graph capture.

    Unlike init(), this never changes the module's default G1 context.
    """
    import torch
    dev = _device_index(device)
    with _LOCK:
        context = ModelContext(len(_CONTEXTS))
        _load(context, torch, dev, artifact, model_dir, model, foot_pairs, com_body_id, terrain, com_body)
        _init_context(context, dev)
        _CONTEXTS[context.context_id] = context
        return context


def init(device=None, artifact=None, model_dir=None, model=None, foot_pairs=None, com_body_id=None, *, terrain=None, com_body=None):
    """Load the library and upload the model constants to device (default torch's current CUDA device); idempotent
    per device, for the module's default context. The library comes from an artifact (build()): artifact (a .json path),
    else $DK_TORCH_ARTIFACT, else the cache's current artifact for the GPU's architecture. Loading an artifact needs
    torch, NumPy and the NVIDIA driver only: no nvcc, mujoco or JAX. Without a usable artifact (none, or one of other
    package sources), init() builds one first (mujoco and nvcc needed then). model_dir / model / foot_pairs: another
    model than the default (as jax_ffi.init): built (or found in the cache) for it, not made current. Returns the
    artifact's facts (path, key, arch, sizes, ...). The first call on a device synchronizes it once: call this
    before capturing a CUDA graph."""
    global _DEFAULT_CONTEXT
    with _LOCK:
        if _DEFAULT_CONTEXT is None:
            import torch
            dev = _device_index(device)
            context = ModelContext(len(_CONTEXTS))
            context._diagnostics_on = _DEFAULT_DIAGNOSTICS_ON
            _load(context, torch, dev, artifact, model_dir, model, foot_pairs, com_body_id, terrain, com_body)
            _init_context(context, dev)
            _CONTEXTS[context.context_id] = context
            _DEFAULT_CONTEXT = context
        elif artifact is not None or model_dir is not None or model is not None or terrain is not None:
            raise _err("the default model is already loaded; use create_context() for another model", RuntimeError)
        else:
            _init_context(_DEFAULT_CONTEXT, _device_index(device))
        return _DEFAULT_CONTEXT.info


def _init_context(context, dev):
    import torch
    cc = torch.cuda.get_device_capability(dev)
    if f"sm_{cc[0]}{cc[1]}" != context.info["arch"]:
        raise _err(f"cuda:{dev} is sm_{cc[0]}{cc[1]}, this context is {context.info['arch']}", RuntimeError)
    if dev not in context._ready:
        _upload(context, dev)
    return context.info


def _context(context_id):
    return _CONTEXTS[context_id]


def _resolve_context(context):
    if context is None:
        if _DEFAULT_CONTEXT is None:
            init()
        context = _DEFAULT_CONTEXT
    return context.context_id


def _load(context, torch, dev, artifact, model_dir, model, foot_pairs, com_body_id, terrain, com_body):
    cc = torch.cuda.get_device_capability(dev)
    arch = os.environ.get("DK_ARCH") or f"sm_{cc[0]}{cc[1]}"
    if model_dir is not None or model is not None or terrain is not None:
        js = build(arch, model_dir, model, foot_pairs, set_current=False, com_body_id=com_body_id, terrain=terrain, com_body=com_body)
    else:
        js = _artifact_path(arch, artifact)
        why = _stale(js, arch)
        if why and artifact is not None:
            raise _err(f"artifact {js}: {why}", RuntimeError)
        if why:
            try:
                js = build(arch)
            except ImportError as e:
                raise _err(f"no usable artifact for {arch} ({why}), and building one needs mujoco 3.17 and nvcc: "
                           f"run `python -m nablaengine.torch_ffi build` where they are installed, or set "
                           f"DK_TORCH_ARTIFACT ({e})", RuntimeError) from e
    meta = json.loads(js.read_text())
    consts = np.ascontiguousarray(np.load(js.parent / meta["consts"]), np.float32)
    if hashlib.sha256(consts.tobytes()).hexdigest() != meta["consts_sha256"]:
        raise _err(f"{meta['consts']} does not match its artifact {js}", RuntimeError)
    lib = ctypes.cdll.LoadLibrary(str(js.parent / meta["library"]))
    info = dict(meta, path=str(js.parent / meta["library"]), artifact=str(js))
    fns = _bind(lib, info)
    context._state.update(lib=lib, info=info, fns=fns, consts=consts, off_flags=int(meta["off_flags"]))
    if not _CONTEXTS:
        import atexit
        atexit.register(_exit_report)  # with set_overflow_warnings(True): counts nobody read
    _custom_ops()  # Register opaque ops and autograd classes before a fullgraph trace.
    _fn("step", _make_step_fn)
    _fn("kin", _make_kin_fn)


_TERRAIN = {}  # device index -> the terrain's device tensors (kept alive)


def set_terrain(terrain, device=None, cell=1.0, *, context=None):
    """Upload a nablaengine.terrain.Terrain to device (default the current one) for a terrain build (init(terrain=));
    replaces the previous one on that device. cell: the broad-phase grid's cell size in metres."""
    import torch
    dev = _device_index(device)
    owner = _context(_resolve_context(context))
    owner.init(dev)
    if "set_terrain" not in owner._state["fns"]:
        raise _err("set_terrain needs a terrain build: init(terrain=TerrainConfig(...))", RuntimeError)
    packed = terrain.pack(cell)
    order = ("boxes", "hf_head", "hf_int", "hf_data", "grid", "cell_start", "cell_items", "elem_cell")
    with torch.cuda.device(dev):
        tens = {k: torch.from_numpy(np.ascontiguousarray(packed[k]).reshape(-1)).to(f"cuda:{dev}") for k in order}
        for k in order:  # never a null pointer (the kernel checks only grid)
            if tens[k].numel() == 0:
                tens[k] = torch.zeros(1, dtype=tens[k].dtype, device=f"cuda:{dev}")
        ptrs = (ctypes.c_void_p * 8)(*[tens[k].data_ptr() for k in order])
        for f in owner._state["fns"]["set_terrain"]:
            _rc(f(dev, ctypes.cast(ptrs, ctypes.c_void_p)), "the terrain upload")
        torch.cuda.synchronize(dev)
    _TERRAIN[owner.context_id, dev] = tens



# ---- height scan: rays against the terrain (mjlab's RayCastSensor on MuJoCo Warp's rays) ------------------------------

def grid_pattern(size=(1.0, 1.0), resolution=0.1, direction=(0.0, 0.0, -1.0), device=None):
    """(offsets (N, 3), directions (N, 3)) of a grid of parallel rays in the frame, as mjlab's
    GridPatternCfg(size, resolution, direction).generate_rays."""
    import torch
    x = torch.arange(-size[0] / 2, size[0] / 2 + resolution * 0.5, resolution, device=device, dtype=torch.float32)
    y = torch.arange(-size[1] / 2, size[1] / 2 + resolution * 0.5, resolution, device=device, dtype=torch.float32)
    gx, gy = torch.meshgrid(x, y, indexing="xy")
    off = torch.zeros((gx.numel(), 3), device=device, dtype=torch.float32)
    off[:, 0], off[:, 1] = gx.flatten(), gy.flatten()
    d = torch.tensor(direction, device=device, dtype=torch.float32)
    return off, (d / d.norm()).unsqueeze(0).expand(gx.numel(), 3).clone()


def _yaw_rotation(rot):
    """mjlab RayCastSensor._extract_yaw_rotation: the rotation about z of the frame's x axis (its y axis where x is
    within 0.1 of vertical)."""
    import torch
    x = rot[:, :, 0].clone()
    x[:, 2] = 0
    xn = x.norm(dim=1)
    sing = xn < 0.1
    if sing.any():
        y = rot[:, :, 1].clone()
        y[:, 2] = 0
        y = y / y.norm(dim=1).clamp(min=1e-6).unsqueeze(-1)
        xy = torch.zeros_like(y)
        xy[:, 0], xy[:, 1] = y[:, 1], -y[:, 0]
        x[sing] = xy[sing]
        xn[sing] = 1.0
    x = x / xn.clamp(min=1e-6).unsqueeze(-1)
    out = torch.zeros((rot.shape[0], 3, 3), device=rot.device, dtype=rot.dtype)
    out[:, 0, 0], out[:, 1, 0], out[:, 0, 1], out[:, 1, 1], out[:, 2, 2] = x[:, 0], x[:, 1], -x[:, 1], x[:, 0], 1
    return out


def raycast(pos, rot, offsets, directions, alignment="base", max_distance=10.0, *, context=None):
    """Height scan / depth rays against the terrain of a terrain build (set_terrain), as mjlab's RayCastSensor: rays
    from the frame at pos (B, 3) with rotation rot (B, 3, 3; e.g. a body's from kinematics()), pattern offsets and unit
    directions (N, 3) in the frame (grid_pattern), alignment "base" (the frame's rotation), "yaw" (its yaw only) or
    "world" (none). Returns dict(distances (B, N): to the nearest terrain hit, -1 for none or beyond max_distance,
    normals_w (B, N, 3): its world normal (0 for none), hit_pos_w (B, N, 3): the hit (the ray origin for none)).
    Only the terrain is hit (not the robot's geoms); forward only (no gradient). Intersections are MuJoCo Warp's
    ray_box / ray_hfield."""
    import torch
    dev = _device_index(pos.device)
    owner = _context(_resolve_context(context))
    owner.init(dev)
    if "terrain_rays" not in owner._state["fns"]:
        raise _err("raycast needs a terrain build: init(terrain=TerrainConfig(...)) and set_terrain()", RuntimeError)
    if alignment == "base":
        R = rot
    elif alignment == "yaw":
        R = _yaw_rotation(rot)
    elif alignment == "world":
        R = torch.eye(3, device=rot.device, dtype=rot.dtype).unsqueeze(0).expand(rot.shape[0], -1, -1)
    else:
        raise ValueError(f"alignment {alignment!r}: 'base', 'yaw' or 'world'")
    B, N = pos.shape[0], offsets.shape[0]
    origins = (pos.unsqueeze(1) + torch.einsum("bij,nj->bni", R, offsets)).contiguous()
    rays = torch.einsum("bij,nj->bni", R, directions).contiguous()
    dist = torch.empty((B, N), dtype=torch.float32, device=pos.device)
    nrm = torch.empty((B, N, 3), dtype=torch.float32, device=pos.device)
    with torch.cuda.device(dev):
        st = torch.cuda.current_stream(dev).cuda_stream
        _rc(owner._state["fns"]["terrain_rays"](st, B * N, origins.data_ptr(), rays.data_ptr(), float(max_distance),
                                          dist.data_ptr(), nrm.data_ptr()), "the terrain rays")
    hit = dist >= 0
    hit_pos = torch.where(hit.unsqueeze(-1), origins + rays * dist.clamp(min=0).unsqueeze(-1), origins)
    return dict(distances=dist, normals_w=nrm, hit_pos_w=hit_pos)

def _device_index(device):
    import torch
    if device is None:
        return torch.cuda.current_device()
    d = torch.device(device)
    if d.type != "cuda":
        raise _err(f"device must be a CUDA device, got {d}")
    return torch.cuda.current_device() if d.index is None else d.index


def _bind(lib, info):
    """The library's C entries (argument types set), after a check of every unit's compiled sizes against the
    artifact's model facts."""
    vp, i32, f32 = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    fwd = info["forward_unit"]
    fns = dict(forward=getattr(lib, f"{fwd}_forward"), work_bytes=getattr(lib, f"{fwd}_work_bytes"),
               kin=getattr(lib, f"{fwd}_kinematics"), kin_vjp=getattr(lib, f"{fwd}_kinematics_vjp"))
    fns["forward"].argtypes = [vp, i32] + [vp] * 6 + [i32, i32] + [vp] * 9
    fns["work_bytes"].argtypes, fns["work_bytes"].restype = [i32], ctypes.c_longlong
    fns["kin"].argtypes = [vp, i32] + [vp] * 8
    fns["kin_vjp"].argtypes = [vp, i32] + [vp] * 12
    fns["adjoint"] = {}
    for v in VARIANTS:
        t = _tag(v)
        f = getattr(lib, f"{t}_adjoint")
        f.argtypes = [vp, i32, i32] + [vp] * 9 + [vp, f32] + [vp] * 3
        fns["adjoint"][v] = f
        sz = np.zeros(9, np.int64)  # as jax_ffi.init's check of dk_sizes
        getattr(lib, f"{t}_sizes").argtypes = [vp]
        getattr(lib, f"{t}_sizes")(sz.ctypes.data)
        want = (info["rec_words"], info["nq"], info["nv"], info["maxcon"], v[0], v[1], info["off_flags"])
        if tuple(int(x) for x in sz[2:9]) != tuple(int(x) for x in want):
            raise _err(f"unit {t}: compiled sizes {sz.tolist()} do not match the artifact {want}", RuntimeError)
    units = [_tag(v) for v in VARIANTS] + ([fwd] if fwd == FWD_UNIT else [])
    fns["upload"] = []
    for u in units:
        f = getattr(lib, f"{u}_upload")
        f.argtypes = [vp, vp]
        fns["upload"].append((u, f))
    if info.get("terrain"):
        fns["set_terrain"] = []
        for u in units:
            f = getattr(lib, f"{u}_set_terrain")
            f.argtypes = [i32, vp]
            fns["set_terrain"].append(f)
        fns["terrain_rays"] = getattr(lib, f"{units[0]}_terrain_rays")
        fns["terrain_rays"].argtypes = [vp, i32, vp, vp, f32, vp, vp]
    for name in ("dk_torch_diag_fwd", "dk_torch_diag_bwd"):
        fns[name] = getattr(lib, name)
    fns["dk_torch_diag_fwd"].argtypes = [vp, vp, i32, i32, i32, vp, vp]
    fns["dk_torch_diag_bwd"].argtypes = [vp, vp, i32] + [vp] * 6
    return fns


def _upload(context, dev):
    import torch
    with torch.cuda.device(dev):
        if torch.cuda.is_current_stream_capturing():
            raise _err(f"the model constants are not on cuda:{dev} yet: call context.init({dev}) before capturing a "
                       "CUDA graph", RuntimeError)
        st = torch.cuda.current_stream(dev)
        c = context._state["consts"]
        for unit, f in context._state["fns"]["upload"]:
            rc = f(c.ctypes.data, st.cuda_stream)
            if rc:
                raise _err(f"constant upload of {unit} on cuda:{dev} failed ({rc})", RuntimeError)
        st.synchronize()
        context._diag[dev] = torch.zeros(len(DIAG), dtype=torch.int64, device=f"cuda:{dev}")
    context._ready.add(dev)


def _ready(t, context):
    """The device index of tensor t, its constants uploaded (init)."""
    dev = t.device.index
    if dev not in context._ready:
        context.init(dev)
    return dev


def _on(dev):
    """dev as the current CUDA device for a library call (the kernels, their constants and torch's stream are per
    device): a no-op where it already is."""
    import contextlib
    import torch
    return contextlib.nullcontext() if torch.cuda.current_device() == dev else torch.cuda.device(dev)


def _on_input_device(fn):
    """fn(first, ...) run with first's CUDA device current (_on), so its launches and torch's stream match."""
    import functools

    @functools.wraps(fn)
    def run(first, *args):
        if getattr(first, "is_cuda", False):
            with _on(first.device.index):
                return fn(first, *args)
        return fn(first, *args)
    return run


def _ptr(t):
    return None if t is None else ctypes.c_void_p(t.data_ptr())


def _check(name, x, shape, dev, dtype=None):
    import torch
    dtype = dtype or torch.float32
    if not isinstance(x, torch.Tensor):
        raise _err(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if x.dtype != dtype or tuple(x.shape) != tuple(shape) or x.device.type != "cuda" or not x.is_contiguous():
        raise _err(f"{name} must be a contiguous {dtype} CUDA tensor {tuple(shape)}, got {x.dtype} {tuple(x.shape)} on "
                   f"{x.device}{'' if x.is_contiguous() else ' (not contiguous)'}")
    if dev is not None and x.device.index != dev:
        raise _err(f"{name} is on {x.device}, the other tensors on cuda:{dev}")


def _rc(rc, what):
    if rc:
        raise _err(f"{what} failed (CUDA error {rc})", RuntimeError)


def _soft(clamp_kv, clamp_soft):
    """jax_ffi._soft without numpy (torch.compile traces it): the soft clamp's delta for clamp_kv 2 (None:
    CLAMP_SOFT), None for 0 / 1 (where it must be None); the library call rounds it to float32."""
    if int(clamp_kv) != 2:
        if clamp_soft is not None:
            raise _err(f"clamp_soft is the soft clamp's (clamp_kv=2), not clamp_kv={clamp_kv}", ValueError)
        return None
    d = float(CLAMP_SOFT if clamp_soft is None else clamp_soft)
    if not (math.isfinite(d) and d >= 0.0):
        raise _err(f"clamp_soft must be finite and >= 0, got {clamp_soft}", ValueError)
    return d


_LAST_CON = {}

def last_contacts(device=None, *, context=None):
    """Last native contact table for the selected model and CUDA device."""
    import torch
    owner = _context(_resolve_context(context))
    table = _LAST_CON.get((owner.context_id, _device_index(device)))
    if table is None:
        return None
    values = table.view(table.shape[0], owner.info["maxcon"], 8)
    result = dict(pair=values[..., 0].to(torch.int32), dist=values[..., 1],
                  pos=values[..., 2:5], normal=values[..., 5:8])
    result["valid"] = result["pair"] >= 0
    return result

def _sizes(context_id):
    i = _context(context_id).info
    return i["nq"], i["nv"], i["nu"], i["nbody"], i["rec_words"]


def _com_width(context_id):
    return _context(context_id).info["com_width"]


def tape_bytes(nworld, substeps=SUBSTEPS, steps=1, *, context=None):
    """Bytes of the tapes of `steps` differentiated control steps."""
    return 4 * _context(_resolve_context(context)).info["rec_words"] * substeps * nworld * steps


# ---- custom ops (torch.library): the library calls, opaque to torch.compile ------------------------------------------

@_on_input_device
def _forward_impl(qpos, qvel, warm, ctrl, friction, com, qfrc_applied, kp_scale, iterations, substeps, tape, kinematics,
                  context_id):
    import torch
    context = _context(context_id)
    dev = _ready(qpos, context)
    nq, nv, nu, nb, rec = _sizes(context_id)
    B = qpos.shape[0]
    for name, x, shape in (("qpos", qpos, (B, nq)), ("qvel", qvel, (B, nv)), ("warm", warm, (B, nv)),
                           ("ctrl", ctrl, (B, nu)), ("friction", friction, (B,)), ("com", com, (B, _com_width(context_id)))):
        _check(name, x, shape, dev)
    if qfrc_applied is not None:
        _check("qfrc_applied", qfrc_applied, (B, nv), dev)
    if kp_scale is not None:
        _check("kp_scale", kp_scale, (B,), dev)
    if substeps < 1 or substeps > 32 or iterations < 0:
        raise _err("substeps in 1..32 and iterations >= 0", ValueError)
    kw = dict(dtype=torch.float32, device=qpos.device)
    q1, v1, w1 = torch.empty((B, nq), **kw), torch.empty((B, nv), **kw), torch.empty((B, nv), **kw)
    info = torch.empty((B, 4), dtype=torch.int32, device=qpos.device)
    tp = torch.empty((B, substeps * rec), **kw) if tape else torch.empty((0, substeps * rec), **kw)
    kin = _kin_empty(B, qpos.device, context_id) if kinematics else []
    if B == 0:
        return [q1, v1, w1, info, tp] + kin
    fns = context._state["fns"]
    con = torch.empty((B, context.info["maxcon"] * 8), **kw)
    _LAST_CON[context_id, dev] = con
    work = torch.empty((max(int(fns["work_bytes"](B)), 1),), dtype=torch.uint8, device=qpos.device)
    st = torch.cuda.current_stream(dev).cuda_stream
    _rc(fns["forward"](st, B, *map(_ptr, (qpos, qvel, warm, ctrl, friction, com)), iterations, substeps, _ptr(q1),
                       _ptr(v1), _ptr(w1), _ptr(info), _ptr(con), _ptr(tp) if tape else None, _ptr(work),
                       _ptr(qfrc_applied), _ptr(kp_scale)),
        "the forward launch")
    if kinematics:
        _rc(fns["kin"](st, B, _ptr(q1), _ptr(v1), _ptr(com), *map(_ptr, kin)), "the kinematics launch")
    if context._diagnostics_on:
        _diag_forward(context, dev, st, info, tp if tape else None, iterations, substeps)
    return [q1, v1, w1, info, tp] + kin


def _kin_empty(B, device, context_id):
    import torch
    nq, _, _, nb, _ = _sizes(context_id)
    return [torch.empty(s, dtype=torch.float32, device=device) for s in
            ((B, nq), (B, nb, 3), (B, nb, 4), (B, nb, 6), (B, nb, 3))]


@_on_input_device
def _backward_impl(tape, friction, com, g_qpos, g_qvel, ctrl, kin_state, kin_cot, substeps, clamp_kv, normal,
                   clamp_soft, qfrc_applied, kp_scale, context_id):
    import torch
    context = _context(context_id)
    dev = _ready(tape, context)
    nq, nv, nu, nb, rec = _sizes(context_id)
    B = tape.shape[0]
    for name, x, shape in (("tape", tape, (B, substeps * rec)), ("friction", friction, (B,)), ("com", com, (B, _com_width(context_id))),
                           ("g_qpos", g_qpos, (B, nq)), ("g_qvel", g_qvel, (B, nv))):
        _check(name, x, shape, dev)
    v = (int(clamp_kv), int(normal))
    if v not in VARIANTS:
        raise _err(f"no adjoint variant clamp_kv={clamp_kv} normal={normal}", ValueError)
    if (v[0] == 2) != (ctrl is not None):
        raise _err("clamp_kv=2 (the soft clamp) needs the step's ctrl; 0 and 1 take none", ValueError)
    if ctrl is not None:
        _check("ctrl", ctrl, (B, nu), dev)
    if qfrc_applied is not None:
        _check("qfrc_applied", qfrc_applied, (B, nv), dev)
    if kp_scale is not None:
        _check("kp_scale", kp_scale, (B,), dev)
    kw = dict(dtype=torch.float32, device=tape.device)
    gq, gv, gu = torch.empty((B, nq), **kw), torch.empty((B, nv), **kw), torch.empty((B, nu), **kw)
    gf = torch.empty((B, nv), **kw)
    info = torch.empty((B,), dtype=torch.int32, device=tape.device)
    if B == 0:
        return [gq, gv, gu, info, gf]
    fns = context._state["fns"]
    st = torch.cuda.current_stream(dev).cuda_stream
    cq, cv = g_qpos, g_qvel
    if kin_state:  # the kinematics outputs' cotangents pulled back onto (qpos', qvel'), added to theirs last
        q1, v1 = kin_state
        _check("qpos'", q1, (B, nq), dev)
        _check("qvel'", v1, (B, nv), dev)
        for name, x, shape in zip(("g_qpos_n", "g_xpos", "g_xquat", "g_cvel", "g_subtree_com"), kin_cot,
                                  ((B, nq), (B, nb, 3), (B, nb, 4), (B, nb, 6), (B, nb, 3))):
            _check(name, x, shape, dev)
        cq, cv = torch.empty((B, nq), **kw), torch.empty((B, nv), **kw)
        _rc(fns["kin_vjp"](st, B, _ptr(q1), _ptr(v1), _ptr(com), *map(_ptr, kin_cot), _ptr(g_qpos), _ptr(g_qvel),
                           _ptr(cq), _ptr(cv)), "the kinematics VJP launch")
    _rc(fns["adjoint"][v](st, B, substeps, _ptr(tape), _ptr(friction), _ptr(com), _ptr(cq), _ptr(cv), _ptr(gq),
                          _ptr(gv), _ptr(gu), _ptr(info), _ptr(ctrl), ctypes.c_float(clamp_soft),
                          _ptr(qfrc_applied), _ptr(kp_scale), _ptr(gf)),
        "the adjoint launch")
    if context._diagnostics_on:
        _diag_backward(context, dev, st, gq, gv, gu, cq, cv, info)
    return [gq, gv, gu, info, gf]


@_on_input_device
def _kinematics_impl(qpos, qvel, com, context_id):
    import torch
    context = _context(context_id)
    dev = _ready(qpos, context)
    nq, nv, _, _, _ = _sizes(context_id)
    B = qpos.shape[0]
    for name, x, shape in (("qpos", qpos, (B, nq)), ("qvel", qvel, (B, nv)), ("com", com, (B, _com_width(context_id)))):
        _check(name, x, shape, dev)
    out = _kin_empty(B, qpos.device, context_id)
    if B:
        st = torch.cuda.current_stream(dev).cuda_stream
        _rc(context._state["fns"]["kin"](st, B, _ptr(qpos), _ptr(qvel), _ptr(com), *map(_ptr, out)), "the kinematics launch")
    return out


@_on_input_device
def _kinematics_backward_impl(qpos, qvel, com, g_qpos_n, g_xpos, g_xquat, g_cvel, g_subtree_com, context_id):
    import torch
    context = _context(context_id)
    dev = _ready(qpos, context)
    nq, nv, _, nb, _ = _sizes(context_id)
    B = qpos.shape[0]
    for name, x, shape in (("qpos", qpos, (B, nq)), ("qvel", qvel, (B, nv)), ("com", com, (B, _com_width(context_id))),
                           ("g_qpos_n", g_qpos_n, (B, nq)), ("g_xpos", g_xpos, (B, nb, 3)),
                           ("g_xquat", g_xquat, (B, nb, 4)), ("g_cvel", g_cvel, (B, nb, 6)),
                           ("g_subtree_com", g_subtree_com, (B, nb, 3))):
        _check(name, x, shape, dev)
    kw = dict(dtype=torch.float32, device=qpos.device)
    gq, gv = torch.empty((B, nq), **kw), torch.empty((B, nv), **kw)
    if B:
        st = torch.cuda.current_stream(dev).cuda_stream
        _rc(context._state["fns"]["kin_vjp"](st, B, _ptr(qpos), _ptr(qvel), _ptr(com), _ptr(g_qpos_n), _ptr(g_xpos),
                                     _ptr(g_xquat), _ptr(g_cvel), _ptr(g_subtree_com), None, None, _ptr(gq), _ptr(gv)),
            "the kinematics VJP launch")
    return [gq, gv]


_OPS = {}
# The final model key specializes compiled graphs; native launches still route by context_id.
_IMPL = dict(forward=lambda *a: _forward_impl(*a[:-1]), backward=lambda *a: _backward_impl(*a[:-1]),
             kinematics=lambda *a: _kinematics_impl(*a[:-1]), kinematics_backward=lambda *a: _kinematics_backward_impl(*a[:-1]))


def _ops():
    """The library calls: the torch.library custom ops while torch.compile traces (opaque nodes), the same functions
    called directly in eager mode (no custom-op dispatch: about 0.3 ms less host time per forward + backward)."""
    import torch
    if not torch.compiler.is_compiling():
        return _IMPL
    return _custom_ops()


def _custom_ops():
    """The custom ops, defined on first use (torch imported lazily)."""
    if _OPS:
        return _OPS
    import torch
    from typing import List, Optional
    from torch import Tensor

    def model_op(name):
        def register(fn):
            # Model identity must specialize even when Dynamo generalizes tensor shapes.
            schema = torch.library.infer_schema(fn, mutates_args=()).replace("SymInt context_id", "int context_id")
            return torch.library.custom_op(f"nablaengine::{name}", fn, mutates_args=(), schema=schema)
        return register

    @model_op("forward")
    def forward_op(qpos: Tensor, qvel: Tensor, warm: Tensor, ctrl: Tensor, friction: Tensor, com: Tensor,
                   qfrc_applied: Optional[Tensor], kp_scale: Optional[Tensor], iterations: int, substeps: int,
                   tape: bool, kinematics: bool, context_id: int, model_key: str) -> List[Tensor]:
        return _forward_impl(qpos, qvel, warm, ctrl, friction, com, qfrc_applied, kp_scale, iterations, substeps, tape,
                             kinematics, context_id)

    @forward_op.register_fake
    def _(qpos, qvel, warm, ctrl, friction, com, qfrc_applied, kp_scale, iterations, substeps, tape, kinematics, context_id, model_key):
        nq, nv, _, nb, rec = _sizes(context_id)
        B = qpos.shape[0]
        out = [qpos.new_empty((B, nq)), qpos.new_empty((B, nv)), qpos.new_empty((B, nv)),
               qpos.new_empty((B, 4), dtype=torch.int32), qpos.new_empty((B if tape else 0, substeps * rec))]
        if kinematics:
            out += [qpos.new_empty(s) for s in ((B, nq), (B, nb, 3), (B, nb, 4), (B, nb, 6), (B, nb, 3))]
        return out

    @model_op("backward")
    def backward_op(tape: Tensor, friction: Tensor, com: Tensor, g_qpos: Tensor, g_qvel: Tensor,
                    ctrl: Optional[Tensor], kin_state: List[Tensor], kin_cot: List[Tensor], substeps: int,
                    clamp_kv: int, normal: int, clamp_soft: float, qfrc_applied: Optional[Tensor],
                    kp_scale: Optional[Tensor], context_id: int, model_key: str) -> List[Tensor]:
        return _backward_impl(tape, friction, com, g_qpos, g_qvel, ctrl, kin_state, kin_cot, substeps, clamp_kv,
                              normal, clamp_soft, qfrc_applied, kp_scale, context_id)

    @backward_op.register_fake
    def _(tape, friction, com, g_qpos, g_qvel, ctrl, kin_state, kin_cot, substeps, clamp_kv, normal, clamp_soft,
          qfrc_applied, kp_scale, context_id, model_key):
        nq, nv, nu, _, _ = _sizes(context_id)
        B = tape.shape[0]
        return [tape.new_empty((B, nq)), tape.new_empty((B, nv)), tape.new_empty((B, nu)),
                tape.new_empty((B,), dtype=torch.int32), tape.new_empty((B, nv))]

    @model_op("kinematics")
    def kinematics_op(qpos: Tensor, qvel: Tensor, com: Tensor, context_id: int, model_key: str) -> List[Tensor]:
        return _kinematics_impl(qpos, qvel, com, context_id)

    @kinematics_op.register_fake
    def _(qpos, qvel, com, context_id, model_key):
        nq, _, _, nb, _ = _sizes(context_id)
        B = qpos.shape[0]
        return [qpos.new_empty(s) for s in ((B, nq), (B, nb, 3), (B, nb, 4), (B, nb, 6), (B, nb, 3))]

    @model_op("kinematics_backward")
    def kinematics_backward_op(qpos: Tensor, qvel: Tensor, com: Tensor, g_qpos_n: Tensor, g_xpos: Tensor,
                               g_xquat: Tensor, g_cvel: Tensor, g_subtree_com: Tensor, context_id: int,
                               model_key: str) -> List[Tensor]:
        return _kinematics_backward_impl(qpos, qvel, com, g_qpos_n, g_xpos, g_xquat, g_cvel, g_subtree_com, context_id)

    @kinematics_backward_op.register_fake
    def _(qpos, qvel, com, g_qpos_n, g_xpos, g_xquat, g_cvel, g_subtree_com, context_id, model_key):
        nq, nv, _, _, _ = _sizes(context_id)
        B = qpos.shape[0]
        return [qpos.new_empty((B, nq)), qpos.new_empty((B, nv))]

    _OPS.update(forward=forward_op, backward=backward_op, kinematics=kinematics_op,
                kinematics_backward=kinematics_backward_op)
    return _OPS


# ---- diagnostics (device counters, as jax_ffi's k_diag_fwd / k_diag_bwd) ---------------------------------------------

def _diag_forward(context, dev, st, info, tape, iterations, substeps):
    """The forward's counters added on the device (one kernel, dk_torch_diag_fwd: as jax_ffi's k_diag_fwd)."""
    _rc(context._state["fns"]["dk_torch_diag_fwd"](st, _ptr(context._diag[dev]), info.shape[0], substeps, iterations, _ptr(info),
                                            _ptr(tape)), "the forward diagnostics launch")


def _diag_backward(context, dev, st, gq, gv, gu, cq, cv, info):
    """The backward's counters added on the device and the non-finite bits into info (one kernel, dk_torch_diag_bwd:
    as jax_ffi's k_diag_bwd)."""
    _rc(context._state["fns"]["dk_torch_diag_bwd"](st, _ptr(context._diag[dev]), info.shape[0], *map(_ptr, (gq, gv, gu, cq, cv)),
                                            _ptr(info)), "the backward diagnostics launch")


def set_diagnostics(enabled, *, context=None):
    """Switch the device counters and the backward's non-finite bits on (the default) or off; returns the previous
    setting. They are one small kernel per forward and per backward (dk_torch_diag_fwd / _bwd); off leaves the
    counters as they are and the backward's info with the adjoint's flags only.
    Outputs and gradients are the same either way. A captured CUDA graph keeps the setting of its capture."""
    global _DEFAULT_DIAGNOSTICS_ON
    if context is None and _DEFAULT_CONTEXT is None:
        prev = _DEFAULT_DIAGNOSTICS_ON
        _DEFAULT_DIAGNOSTICS_ON = bool(enabled)
        return prev
    owner = _context(_resolve_context(context))
    prev = owner._diagnostics_on
    owner._diagnostics_on = bool(enabled)
    return prev


def diagnostics(device=None, *, context=None):
    """The device's counters (int64 (len(DIAG),) CUDA tensor, DIAG order), added up by every call since init or the
    last reset; not copied or synchronized (graph-safe to read into another tensor)."""
    dev = _device_index(device)
    owner = _context(_resolve_context(context))
    if dev not in owner._ready:
        owner.init(dev)
    return owner._diag[dev]


def read_diagnostics(device=None, reset=True, *, context=None):
    """The device's counters as {name: int} (synchronizes); reset=True zeroes them afterwards. With
    set_overflow_warnings(True) it warns (warn_overflow) when worlds exceeded the contact or constraint-row capacity."""
    d = diagnostics(device, context=context)
    out = dict(zip(DIAG, (int(x) for x in d.cpu())))
    if reset:
        d.zero_()
    if _OVERFLOW_WARN[0]:
        warn_overflow(out)
    return out


def warn_overflow(d, where="nablaengine"):
    """A RuntimeWarning when the counters d count worlds whose contacts or constraint rows exceeded the per-world
    compiled capacity: the ones beyond were dropped, so those worlds miss support and friction forces
    (as jax_ffi.warn_overflow). The worlds of a call: info[:, 3] & OVERFLOW_CONTACTS / OVERFLOW_ROWS of forward()."""
    import warnings
    nc, nr = d.get("fwd_contact_overflow_worlds", 0), d.get("fwd_row_overflow_worlds", 0)
    if nc or nr:
        warnings.warn(f"{where}: {nc} world-steps exceeded the compiled contact capacity and {nr} the constraint-"
                      f"row capacity; their extra contacts / rows were dropped, so those worlds miss support and "
                      f"friction forces (e.g. deep interpenetration). See docs/api_torch.md, Diagnostics.",
                      RuntimeWarning, stacklevel=3)
    return nc + nr


def set_overflow_warnings(enabled):
    """Switch the capacity-overflow warnings on or off (default off, or $DK_OVERFLOW_WARN=1): when on,
    read_diagnostics warns (warn_overflow) if worlds exceeded the per-world contact or
    constraint-row capacity, and the counts nobody read are reported once more at exit. The counters and info's bits
    are kept either way. Returns the previous setting."""
    prev = _OVERFLOW_WARN[0]
    _OVERFLOW_WARN[0] = bool(enabled)
    return prev


def _exit_report():
    """At exit: the overflow warning for each device's counts since the last read (no reset); never raises."""
    try:
        if not _OVERFLOW_WARN[0]:
            return
        for context in _CONTEXTS.values():
            if context._diagnostics_on:
                for dev in sorted(context._diag):
                    warn_overflow(dict(zip(DIAG, (int(x) for x in context._diag[dev].cpu()))),
                                  where=f"nablaengine (context {context.context_id}, cuda:{dev})")
    except Exception:  # noqa: BLE001  (interpreter or CUDA shutting down)
        pass


# ---- autograd --------------------------------------------------------------------------------------------------------

def _make_step_fn():
    import torch

    class _Step(torch.autograd.Function):
        @staticmethod
        def forward(ctx, qpos, qvel, warm, ctrl, friction, com, qfrc_applied, kp_scale, iterations, substeps,
                    clamp_kv, normal, clamp_soft, grad_decay, kin, return_info, context_id, model_key):
            r = _ops()["forward"](qpos, qvel, warm, ctrl, friction, com, qfrc_applied, kp_scale, iterations, substeps,
                                   True, kin, context_id, model_key)
            q1, v1, w1, info, tp = r[:5]
            ctx.save_for_backward(tp, friction, com, ctrl, q1, v1, qfrc_applied, kp_scale)
            ctx.opts = (substeps, clamp_kv, normal, clamp_soft, grad_decay, kin, return_info)
            ctx.context_id = context_id
            ctx.model_key = model_key
            if return_info:
                ctx.mark_non_differentiable(w1, info)
            else:
                ctx.mark_non_differentiable(w1)
            return (q1, v1, w1) + ((info,) if return_info else ()) + tuple(r[5:])

        @staticmethod
        def backward(ctx, g_q1, g_v1, g_w1, *g_extra):
            tp, friction, com, ctrl, q1, v1, qfrc_applied, kp_scale = ctx.saved_tensors
            substeps, clamp_kv, normal, clamp_soft, grad_decay, kin, return_info = ctx.opts
            g_kin = g_extra[1:] if return_info else g_extra
            g_q1 = torch.zeros_like(q1) if g_q1 is None else g_q1.contiguous()
            g_v1 = torch.zeros_like(v1) if g_v1 is None else g_v1.contiguous()
            kin_state, kin_cot = [], []
            if kin and any(g is not None for g in g_kin):
                kin_state = [q1, v1]
                kin_cot = [torch.zeros(s, dtype=torch.float32, device=q1.device) if g is None else g.contiguous()
                           for g, s in zip(g_kin, _kin_shapes(q1.shape[0], ctx.context_id))]
            gq, gv, gu, _, gf = _ops()["backward"](tp, friction, com, g_q1, g_v1, ctrl if clamp_kv == 2 else None,
                                               kin_state, kin_cot, substeps, clamp_kv, normal,
                                               0.0 if clamp_soft is None else clamp_soft, qfrc_applied, kp_scale,
                                               ctx.context_id, ctx.model_key)
            if grad_decay != 1.0:  # the state cotangent through this step's physics times it, ctrl's not
                gq, gv = gq * grad_decay, gv * grad_decay  # grad_decay a float32 value: one float32 product
            return gq, gv, None, gu, None, None, gf if qfrc_applied is not None else None, None, None, None, None, None, None, None, None, None, None, None

    return _Step


def _kin_shapes(B, context_id):
    nq, _, _, nb, _ = _sizes(context_id)
    return (B, nq), (B, nb, 3), (B, nb, 4), (B, nb, 6), (B, nb, 3)


def _make_kin_fn():
    import torch

    class _Kin(torch.autograd.Function):
        @staticmethod
        def forward(ctx, qpos, qvel, com, context_id, model_key):
            out = _ops()["kinematics"](qpos, qvel, com, context_id, model_key)
            ctx.save_for_backward(qpos, qvel, com)
            ctx.context_id = context_id
            ctx.model_key = model_key
            return tuple(out)

        @staticmethod
        def backward(ctx, *g):
            qpos, qvel, com = ctx.saved_tensors
            g = [torch.zeros(s, dtype=torch.float32, device=qpos.device) if x is None else x.contiguous()
                 for x, s in zip(g, _kin_shapes(qpos.shape[0], ctx.context_id))]
            gq, gv = _ops()["kinematics_backward"](qpos, qvel, com, *g, ctx.context_id, ctx.model_key)
            return gq, gv, None, None, None

    return _Kin


_FNS = {}


def _fn(name, make):
    if name not in _FNS:
        _FNS[name] = make()
    return _FNS[name]


def step(qpos, qvel, warm, ctrl, friction, com, qfrc_applied=None, kp_scale=None, *, iterations=ITERATIONS,
         substeps=SUBSTEPS, clamp_kv=CLAMP_KV, normal=NORMAL, clamp_soft=None, grad_decay=GRAD_DECAY,
         kinematics=False, return_info=False, context=None):
    """One control step: (qpos', qvel', warm'); kinematics=True: (qpos', qvel', warm', qpos_n, xpos, xquat, cvel,
    subtree_com), the output state's kinematics. Options as jax_ffi.step (clamp_kv, normal, clamp_soft and
    grad_decay act on the backward only). return_info=True inserts the nondifferentiable (B, 4) int32 forward status
    after warm', before any kinematics outputs. Differentiable in qpos, qvel, ctrl and applied generalized force
    when any input requires grad (then the forward also writes the tape); warm' is not differentiable."""
    import torch
    if int(clamp_kv) not in (0, 1, 2) or int(normal) not in (0, 1, 2, 3):
        raise _err("clamp_kv in {0, 1, 2}, normal in 0..3", ValueError)
    soft = _soft(clamp_kv, clamp_soft)
    gd = float(grad_decay)  # used as float32 (torch's float32 product), as jax_ffi's np.float32(grad_decay)
    if not math.isfinite(gd):
        raise _err(f"grad_decay must be finite, got {grad_decay}", ValueError)
    context_id = _resolve_context(context)
    model_key = _context(context_id).info["key"]
    args = (qpos, qvel, warm, ctrl, friction, com, qfrc_applied, kp_scale)
    if torch.is_grad_enabled() and any(isinstance(x, torch.Tensor) and x.requires_grad for x in args):
        return _fn("step", _make_step_fn).apply(*args, int(iterations), int(substeps), int(clamp_kv), int(normal),
                                                soft, gd, bool(kinematics), bool(return_info), context_id, model_key)
    r = _ops()["forward"](*args, int(iterations), int(substeps), False, bool(kinematics), context_id, model_key)
    return tuple(r[:4] if return_info else r[:3]) + tuple(r[5:])


def kinematics(qpos, qvel, com, *, context=None):
    """Differentiable kinematics outputs of states (qpos, qvel, com): (qpos_n, xpos, xquat, cvel, subtree_com), as
    jax_ffi.kinematics (gradients for qpos and qvel, none for com); kinematics(q', v', com) of a step's output is
    bitwise step(..., kinematics=True)'s."""
    context_id = _resolve_context(context)
    return _fn("kin", _make_kin_fn).apply(qpos, qvel, com, context_id, _context(context_id).info["key"])


def forward(qpos, qvel, warm, ctrl, friction, com, qfrc_applied=None, kp_scale=None, *, iterations=ITERATIONS,
            substeps=SUBSTEPS, tape=False, kinematics=False, context=None):
    """One control step without autograd: (qpos', qvel', warm', info (B, 4) int32[, tape][, the kinematics outputs]),
    as jax_ffi.forward."""
    context_id = _resolve_context(context)
    r = _ops()["forward"](qpos, qvel, warm, ctrl, friction, com, qfrc_applied, kp_scale, int(iterations), int(substeps),
                          bool(tape), bool(kinematics), context_id, _context(context_id).info["key"])
    return tuple(r[:4]) + ((r[4],) if tape else ()) + tuple(r[5:])


def backward(tape, friction, com, g_qpos, g_qvel, *, substeps=SUBSTEPS, clamp_kv=CLAMP_KV, normal=NORMAL, ctrl=None,
             clamp_soft=None, kinematics=None, qfrc_applied=None, kp_scale=None, context=None):
    """The adjoint of one control step from its tape: (g_qpos, g_qvel, g_ctrl, info (B,) int32), as
    jax_ffi.backward (kinematics: None or (qpos', qvel', g_qpos_n, g_xpos, g_xquat, g_cvel, g_subtree_com))."""
    soft = _soft(clamp_kv, clamp_soft)
    if soft is not None and ctrl is None:
        raise _err("clamp_kv=2 (the soft clamp) needs the step's ctrl", ValueError)
    ks, kc = ([], []) if kinematics is None else (list(kinematics[:2]), list(kinematics[2:]))
    if kinematics is not None and len(kc) != 5:
        raise _err("kinematics = (qpos', qvel', g_qpos_n, g_xpos, g_xquat, g_cvel, g_subtree_com)", ValueError)
    context_id = _resolve_context(context)
    out = tuple(_ops()["backward"](tape, friction, com, g_qpos, g_qvel, ctrl if soft is not None else None, ks, kc,
                                   int(substeps), int(clamp_kv), int(normal), 0.0 if soft is None else soft,
                                   qfrc_applied, kp_scale, context_id, _context(context_id).info["key"]))
    return out if qfrc_applied is not None else out[:4]


def _make_scale_fn():
    import torch

    class _Scale(torch.autograd.Function):
        @staticmethod
        def forward(ctx, gamma, *xs):
            ctx.gamma = gamma
            return tuple(x.view_as(x) for x in xs)

        @staticmethod
        def backward(ctx, *g):
            return (None,) + tuple(None if x is None else x * ctx.gamma for x in g)

    return _Scale


def scale_grad(tree, gamma):
    """The identity on a pytree of tensors whose cotangent the backward multiplies by gamma (floating tensors only;
    others pass through), as jax_ffi.scale_grad: put it on the whole carried state at the top of each rollout step for
    an exact decay^n per step boundary (see jax_ffi's module docstring)."""
    import torch
    from torch.utils import _pytree
    g = float(gamma)  # used as float32 (torch's float32 product)
    if not math.isfinite(g):
        raise _err(f"scale_grad's gamma must be finite, got {gamma}", ValueError)
    leaves, spec = _pytree.tree_flatten(tree)
    idx = [i for i, x in enumerate(leaves) if isinstance(x, torch.Tensor) and x.is_floating_point()]
    if idx:
        out = _fn("scale", _make_scale_fn).apply(g, *(leaves[i] for i in idx))
        for i, x in zip(idx, out):
            leaves[i] = x
    return _pytree.tree_unflatten(leaves, spec)


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m nablaengine.torch_ffi",
                                 description="build the PyTorch interface's library and artifact (mujoco and nvcc "
                                             "needed here; the processes that load it need neither)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build for one or more architectures and make them current")
    b.add_argument("--arch", action="append", help="sm_XY, repeatable (default: $DK_ARCH, else the first GPU's)")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        from nablaengine import cuda
        for arch in a.arch or [cuda.arch()]:
            print(f"nablaengine.torch_ffi: artifact {build(arch)}")


if __name__ == "__main__":
    _main()

