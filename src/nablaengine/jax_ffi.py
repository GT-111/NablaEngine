"""The G1 engine as a differentiable JAX op: one control step with an analytic VJP, and the kinematics outputs of MJX's
mjx.kinematics + com_pos + com_vel with their VJP, for differentiable rollouts (BPTT through lax.scan).

  step(qpos, qvel, warm, ctrl, friction, com, *, iterations=10, substeps=4, clamp_kv=0, normal=3, clamp_soft=None,
       grad_decay=1.0, kinematics=False) -> (qpos', qvel', warm'[, qpos_n, xpos, xquat, cvel, subtree_com])
  kinematics(qpos, qvel, com) -> (qpos_n, xpos, xquat, cvel, subtree_com)

one control step (substeps engine substeps of h = 5 ms) for B worlds; all float32, shapes (B, nq), (B, nv), (B, nv),
(B, nu), (B,), (B, 3*nr) for one or two FREE roots (mjlab G1: nr 1, nq 36, nv 35, nu 29); warm is the Newton warm start (warm' = the last substep's
acceleration, for the next call); friction is each world's selected-pair sliding coefficient, com the selected bodies' local COM offsets. B is read
from the arrays at run time: one library serves every batch size, and JAX traces once per shape. jax.custom_vjp: the
forward under differentiation (dk_step_fwd) also writes the tape, (B, substeps * REC_WORDS) float32 (5888 B per world
and substep), kept with friction and com as the residuals; the backward (dk_step_bwd, the analytic adjoint k_adjoint,
deterministic) returns d/dqpos (36-dimensional qpos form, as MuJoCo Warp), d/dqvel and d/dctrl (summed over the
substeps), zeros for warm, friction and com; the cotangent of warm' is ignored. Without differentiation step runs
dk_step, the same forward kernel without the tape (bitwise the same outputs). Static options (Python values; each
combination is its own custom_vjp function, make_step):
  iterations  Newton iterations per substep (default 10; the reference corpora use 50)
  substeps    substeps per control step (4)
  clamp_kv    DK_ADJ_CLAMP_KV of the adjoint, the derivative on a dof whose actuator force is clamped at its limit:
              0 (default: as MuJoCo Warp's adjoint, the actuator damping -kv lambda kept in the qvel VJP), 1 (exact:
              dropped); 2: the soft clamp, a surrogate gradient: on a clamped dof the actuator's three paths (qpos,
              qvel, ctrl) carry w lam, w = exp(-o / clamp_soft), o = (|f| - tau_max) / tau_max >= 0 the relative
              overshoot of the unclamped PD force (adjoint.py lam_soft); the custom_vjp then keeps ctrl as a residual
              too
  normal      DK_ADJ_NORMAL: 3 (default: the self contacts' normal frozen where nu < 1e-3 m), 0 exact (MuJoCo Warp's),
              1, 2 (other approximations of the normal derivative; g1_engine.adjoint_contact)
  clamp_soft  the soft clamp's delta for clamp_kv=2 (None: CLAMP_SOFT = 0.2; rounded to float32; 0 gives exactly
              clamp_kv=1's gradients, larger values a softer clamp); must stay None for clamp_kv 0 / 1
  grad_decay  the factor of the physics-only form of the gradient decay (below): the backward multiplies the
              cotangents it returns for qpos and qvel by it, not the ctrl's. 1.0 (default) leaves the computation as
              without the option (no multiply). The same gradients bitwise: scale_grad on the physics call's qpos, qvel
              (below)
The forward does not depend on clamp_kv, normal, clamp_soft or grad_decay: one tape serves every adjoint variant.

Kinematics outputs (the semantics, with MJX source references, in nablaengine.g1_engine.kinematics): MJX's
mjx.kinematics + smooth.com_pos + smooth.com_vel on the model with the world's torso COM offset (body_ipos[torso] +=
com), float32:
  qpos_n (B, nq)  qpos with the free joint's quaternion normalized and written back (as mjx.kinematics does)
  xpos (B, nbody, 3), xquat (B, nbody, 4)  body frames, world body 0 included (0 and the identity)
  cvel (B, nbody, 6)  [angular; linear] twists at the reference point subtree_com[1] (body 0: zeros)
  subtree_com (B, nbody, 3)  subtree COMs with the torso's COM offset com (subtree_com[0] = subtree_com[1])
Two entries, the same kernels (k_kin, k_kin_vjp: g1_engine.kinematics.KIN_DEVICE, built into the forward's unit;
poses and twists by source.DEVICE's fwd_body steps, so their bits are those the engine computes for that state):
  step(..., kinematics=True) -> (qpos', qvel', warm', qpos_n, xpos, xquat, cvel, subtree_com): the kinematics of the
      output state (qpos', qvel') with the step's com, k_kin launched inside the step's forward call after the last
      substep (dk_step_kin / dk_fwd_kin); the outputs' cotangents enter the same VJP: the step's backward call
      (dk_bwd_kin / dk_bwd_soft_kin) first pulls them back onto (qpos', qvel') (k_kin_vjp from the residual qpos',
      qvel', added to their own cotangents last), then runs the adjoint, so its gradients are bitwise those of
      composing step's VJP with kinematics' VJP
  kinematics(qpos, qvel, com) -> (qpos_n, xpos, xquat, cvel, subtree_com): for states not produced by the engine
      (e.g. initial or externally supplied states), a custom_vjp (dk_kin, dk_kin_bwd): cotangents of qpos (the
      quaternion through its normalization: the radial part removed) and qvel, zeros for com; kinematics(q', v', com)
      of a step's output is bitwise its kinematics outputs
  kinematics_forward / kinematics_backward: the same without custom_vjp; forward(..., kinematics=True) and
  backward(..., kinematics=(qpos', qvel', g_qpos_n, g_xpos, g_xquat, g_cvel, g_subtree_com)) the lower-level calls
Remat: the step's kinematics outputs are named OUT_NAME with its other outputs, kinematics()'s outputs and residual
state KIN_NAME ("dk_kin"); remat_policy() saves both, so neither kernel reruns in the backward.

Gradient decay with factor decay (a Python float, e.g. 0.98), in two forms with different laws. scale_grad(tree, decay)
is the identity on a pytree whose cotangent the backward multiplies by decay (a custom_vjp over its inexact leaves;
ints, bools and PRNG keys pass through untouched); it works on any differentiable function, the engine's step or
another.
  Full-state form: scale the whole carried state at the top of each scan body step, before anything reads it. Every
  derivative that crosses n step boundaries is then multiplied by exactly decay^n, whatever the path (physics,
  controller, cost; with closed-loop controls too): a cost computed in step k (from that step's output state) reaches
  the control of step t <= k with decay^(k - t) and the rollout's initial state with decay^(k + 1), and a cost of the
  end state after a scan of H steps reaches the first control with decay^(H - 1) (0.98^63 ~ 0.28). For example:

    from nablaengine import jax_ffi

    def body(carry, x):
        state, total = carry
        state = jax_ffi.scale_grad(state, decay)  # first: everything below sees the scaled state
        ctrl = controller(params, state, x)
        qpos, qvel, warm = jax_ffi.step(state.qpos, state.qvel, state.warm, ctrl, friction, com)
        state = State(qpos, qvel, warm)
        return (state, total + cost(state, ctrl)), None

  Another carry that is a function of earlier states and enters later costs multiplicatively must be scaled with it
  for the law to hold on its paths too (state, w = scale_grad((state, w), decay)); the objective's accumulators (total)
  must not be: they add each cost once, and scaling them would decay a cost by its distance to the rollout's end
  instead.
  Physics-only form: scale only the state arguments of the physics call (grad_decay=decay of step / make_step /
  EnginePhysics, or scale_grad on qpos, qvel at the call; bitwise the same gradients):

    qpos, qvel = jax_ffi.scale_grad((state.qpos, state.qvel), decay)
    qpos, qvel, warm = jax_ffi.step(qpos, qvel, state.warm, ctrl, friction, com)

  Only the part of the state's cotangent that flows back through the physics' qpos, qvel inputs is scaled; ctrl's and
  that of everything else the state feeds are not. Per control step the state's cotangent is then propagated by
  decay F_x + F_u pi_x instead of F_x + F_u pi_x (F_x, F_u: the physics' Jacobians with respect to the state and ctrl,
  pi_x: the controller's with respect to the state), so the full-state form's law decay^(k - t) for a cost of step k
  and the control of step t holds here only for controls that do not depend on the state (open loop); through a
  controller that reads the state the path state -> controller -> ctrl -> physics is not decayed.
  scripts/jax_ffi_check.py (section options) checks both forms on a rollout with state feedback. Not on
  qacc_warmstart (no gradient either way).

Remat (differentiable rollouts under jax.checkpoint): the custom_vjp's fwd names its residual tape TAPE_NAME
("dk_tape") and its outputs OUT_NAME ("dk_out") with jax.ad_checkpoint.checkpoint_name. A plain jax.checkpoint(body)
saves none of them, so the backward reruns dk_step_fwd for the tape (two forward calls per control step); saving the
tape alone still reruns it wherever the body's own VJP needs the step's outputs (a loss of qpos', the kinematics of
the next step's input). Use jax.checkpoint(body, policy=remat_policy()) (= save_only_these_names("dk_tape", "dk_out",
"dk_kin"); "dk_kin": kinematics()'s), or no remat: one dk_step_fwd per control step. The gradients equal those without
remat up to XLA's rounding of the recomputed parts of the body (bitwise for jax_ffi_check's rollout body; ~1e-7
relative for a body with an MLP controller).

Lower-level calls (traceable, no custom_vjp): forward(..., tape=False/True) -> (qpos', qvel', warm', info[, tape]) with
info (B, 4) int32 = (Newton iterations of the last substep, contacts, constraint rows, unsupported bits: tape.py
FLAG_UNSUPPORTED) per world; backward(tape, friction, com, g_qpos', g_qvel', ..., ctrl=None, clamp_soft=None) -> (g_qpos,
g_qvel, g_ctrl, info) (clamp_kv=2 needs the step's ctrl; the gradients are not scaled by any grad_decay) with info (B,)
int32 = the adjoint's flags (the union of the records' tape flags, adjoint.ADJ_ROWS_SKIPPED /
ADJ_REPLAY_MISMATCH) | NONFINITE_* << 16 (which of the world's gradients have a NaN / inf, and whether its incoming
cotangent had one).

Diagnostic counters (device memory of the library, per GPU, integer atomics: exact and order-independent): every
dk_step / dk_step_fwd / dk_step_bwd call adds to DIAG (names below): forward calls and worlds, worlds with unsupported
contacts, worlds whose last substep used all Newton iterations, and from the tape (dk_step_fwd only: the forward
computes REFACT / NONCONV / SINGULAR only where it writes the tape) the records with those flags; backward calls and
rows, the rows (worlds) of (g_qpos, g_qvel, g_ctrl) with any NaN / inf (and per output), the rows whose incoming
cotangent was non-finite, the worlds with a replay mismatch. They accumulate across calls, also inside jit and
lax.scan (the custom_vjp's bwd cannot return them). Read them outside jit with read_diagnostics(reset=True) -> {name:
int} (it runs on the GPU's stream after the work already dispatched); inside jit, diagnostics(*deps, reset=...) ->
(NDIAG, 2) uint32 (low, high words; decode_diagnostics) runs after the computation of its deps (e.g. the gradients).
Besides, every forward call overwrites its record LAST (worlds, contacts summed over the worlds, contacts and constraint
rows of the most loaded world: the last substep's, as MuJoCo Warp's nacon / nefc) for its batch size and for any batch
size: last_forward(nworld) reads it without reset, EnginePhysics.stats() reports it as nacon / naconmax / nefc_max /
njmax. EnginePhysics checks that the caller's model is one the engine supports (the input validation of
nablaengine.model.capability: physics the packed constants cannot see, such as joint damping, the integrator or
contype contacts) and the library's (check_model: timestep, foot pairs, torso id, the packed physics constants), so the
fixtures' constants never silently stand in for another model; init() checks the model it builds for the same way
(UnsupportedModel, a ValueError and an AssertionError, with the failing codes).

The library: generated from the engine's CUDA source for the model (nablaengine.g1_engine: adjoint.cuda_source and
host.KERNEL, the same text as the check binaries of nablaengine.checks) and built by nvcc on first use for the GPU's
architecture ($DK_ARCH, else the first JAX GPU's compute capability), cached in $DK_CACHE_DIR (default
~/.cache/nablaengine) under a hash of the sources, the model constants, the flags, the nvcc version and XLA's FFI
headers. One translation unit per adjoint variant (clamp_kv 0 / 1 / 2 x normal 0-3, each wrapped in an anonymous
namespace and compiled with its -D options, so each is the check binary's code), one with the XLA FFI handlers (typed
FFI, jaxlib's xla/ffi/api/ffi.h) on XLA's CUDA stream (clamp_kv 2 has its own backward target, dk_step_bwd_soft, which
also takes ctrl and the float attribute clamp_soft). On sm_120 every unit is built from compute_90 PTX, the NVVM 7
front end (nablaengine.cuda.arch_flags; $DK_SM120_FRONTEND, default n7). Where the forward / kinematics kernels have
other architecture flags than the adjoint units ($DK_SM120_FRONTEND=split: the forward NVVM 7, the adjoint native; or
$DK_TEST_FORWARD_VIRTUAL_ARCH) they are a unit of their own (FWD_UNIT, the text of FWD_VARIANT's unit); native builds
every unit -arch=sm_120; on every other architecture nothing changes. The unit with the forward kernels gets the
target's register caps (cuda.unit_defines, bitwise unchanged): on sm_90 -DSPLIT_PRE_MINB=16 -DSPLIT_POST_MAXNREG=128
(forward about 11% faster on the H100; $DK_SM90_REG_CAPS=0 builds without them), on sm_120 with the NVVM 7 forward
-DSPLIT_PRE_MINB=16 (1.3% on the RTX PRO 6000; $DK_SM120_PRE_CAP=0). The model constants (adjoint.engine_constants of
the mjlab model: $DK_FIXTURES, else fixtures/mjlab, or init(model=...)) are handed to the library once per process
(init) and uploaded to each GPU before its first call; the upload also fixes k_adjoint's launch on that GPU
(adjoint.adj_launch_smem, the check binaries' launch: on V100 more shared memory than sizeof(Adj), at most 4 worlds
per SM; launch_info()). Every handler checks the rank and dimensions of all its buffers before it reads or writes any
(InvalidArgument otherwise).
"""

import ctypes
import fcntl
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import nablaengine
from nablaengine import cuda, fixtures, provenance
from nablaengine.model import capability
from nablaengine.g1_engine import adjoint, host, kinematics as g1_kin, source as g1_source, step as g1_step, tape as g1_tape

ITERATIONS, SUBSTEPS, CLAMP_KV, NORMAL = 10, 4, 0, 3  # the interface defaults
CLAMP_SOFT, GRAD_DECAY = 0.2, 1.0  # clamp_kv=2's default delta; the default gradient decay factor (no decay)
VARIANTS = tuple((kv, n) for kv in (0, 1, 2) for n in (0, 1, 2, 3))  # adjoint builds (DK_ADJ_CLAMP_KV, DK_ADJ_NORMAL)
FWD_VARIANT = (0, 3)  # the unit that also holds the forward kernel k_substep (the forward has no adjoint options)
FWD_UNIT = "dk_fwd"  # the forward / kinematics unit where it has flags of its own (cuda.arch_flags: sm_120 split)
NONFINITE_QPOS, NONFINITE_QVEL, NONFINITE_CTRL, NONFINITE_COT = 1, 2, 4, 8  # backward info bits >> 16
NONFINITE_SHIFT = 16
DIAG = ("fwd_calls", "fwd_worlds", "fwd_unsupported_worlds", "fwd_iteration_limit_worlds", "tape_records",
        "refact_records", "nonconv_records", "singular_records", "bwd_calls", "bwd_rows", "nonfinite_rows",
        "nonfinite_qpos_rows", "nonfinite_qvel_rows", "nonfinite_ctrl_rows", "nonfinite_cotangent_rows",
        "replay_mismatch_worlds", "fwd_contact_overflow_worlds", "fwd_row_overflow_worlds")
# info[:, 3] bits of a forward (the step kernels' s.unsupported, over the call's substeps): a world had more contacts
# than compiled MAXCON or more constraint rows than NEFC; the ones beyond are dropped (their forces are missing)
OVERFLOW_CONTACTS, OVERFLOW_ROWS = 4, 16
_OVERFLOW_WARN = [os.environ.get("DK_OVERFLOW_WARN", "0") not in ("", "0")]  # set_overflow_warnings
# the last forward call's record (k_last_fwd; EnginePhysics.stats): worlds, contacts summed over the worlds, contacts
# and constraint rows of the most loaded world (the last substep's counts, as MuJoCo Warp's nacon / nefc)
LAST = ("worlds", "contacts", "contacts_max", "rows_max")
NSLOT = 16  # last-forward records: slot 0 of any batch size, then one per batch size (the first 15 seen)
MAX_DEVICES = 64
UPLOAD_ERR_SMEM = -1  # a unit's upload: adj_launch_smem() returned 0
TAPE_NAME, OUT_NAME = "dk_tape", "dk_out"  # checkpoint names of the custom_vjp's residual tape and its outputs (remat)
KIN_NAME = "dk_kin"  # jax_ffi.kinematics's outputs and residual state (step's kinematics outputs are OUT_NAME)
REMAT_NAMES = (TAPE_NAME, OUT_NAME, KIN_NAME)

VARIANT_UNIT = r"""
#include <cuda_runtime.h>
@ENGINE_INCLUDES@
namespace {  // internal linkage: every variant's kernels, device functions and DKC constants are its own
@ENGINE@
// per device: k_adjoint's dynamic shared memory, adj_launch_smem()'s production choice for that GPU (on V100 more
// than sizeof(Adj), so that at most 4 worlds share an SM); 0 until the device's upload
size_t g_adj_smem[@MAX_DEVICES@];
}  // namespace

// the packed model constants -> this unit's DKC arrays, and k_adjoint's launch configuration on the current device;
// returns a cudaError_t, or UPLOAD_ERR_SMEM if adj_launch_smem() failed (returned 0)
extern "C" int @PFX@_upload(const float* p, cudaStream_t st) {
@LOAD_CONSTS@
  int dev = -1;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return (int)e;
  if (dev < 0 || dev >= @MAX_DEVICES@) return (int)cudaErrorInvalidDevice;
  const size_t smem = adj_launch_smem();  // also sets k_adjoint's shared-memory opt-in to it
  if (smem == 0) return @UPLOAD_ERR_SMEM@;
  g_adj_smem[dev] = smem;
@FWD_ATTR@
  return (int)cudaGetLastError();
}
// ctrl, clamp_soft: the soft clamp's (DK_ADJ_CLAMP_KV=2: the step's ctrl, required, and delta); the other units ignore
// them (their k_adjoint gets nullptr as before and has no delta argument)
extern "C" int @PFX@_adjoint(cudaStream_t st, int nw, int nsub, const float* tape, const float* fric, const float* com,
                           const float* gq_out, const float* gv_out, float* gq_in, float* gv_in, float* gctrl,
                           int* info, const float* ctrl, float clamp_soft, const float* qfrc,
                           const float* kp_scale, float* gqfrc) {
  int dev = -1;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return (int)e;
  if (dev < 0 || dev >= @MAX_DEVICES@ || g_adj_smem[dev] == 0)  // ensure_ready() uploads before the first launch
    return (int)cudaErrorInvalidConfiguration;
  if (DK_ADJ_CLAMP_KV == 2 && !ctrl) return (int)cudaErrorInvalidValue;
  (void)clamp_soft;
  k_adjoint<<<nw, 32, g_adj_smem[dev], st>>>(nw, nsub, tape, fric, com, gq_out, gv_out, gq_in, gv_in, gctrl, info,
                                             nullptr, DK_ADJ_CLAMP_KV == 2 ? ctrl : nullptr ADJ_SOFT_ARG(clamp_soft),
                                             qfrc, kp_scale, gqfrc);
  return (int)cudaGetLastError();
}
// the launch on device dev (after its upload): o[0] the dynamic shared memory, o[1] the resident worlds per SM
extern "C" int @PFX@_launch(int dev, long long* o) {
  o[0] = dev >= 0 && dev < @MAX_DEVICES@ ? (long long)g_adj_smem[dev] : 0;
  o[1] = 0;
  if (!o[0]) return 0;
  int old = -1, n = 0;
  cudaGetDevice(&old);
  cudaSetDevice(dev);
  cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k_adjoint, 32, (size_t)o[0]);
  cudaSetDevice(old);
  o[1] = n;
  return (int)e;
}
extern "C" void @PFX@_sizes(long long* o) {  // checked by the Python side against tape.py and the model
  o[0] = (long long)sizeof(Sim); o[1] = (long long)sizeof(Adj); o[2] = REC_WORDS; o[3] = NQ; o[4] = NV; o[5] = MAXCON;
  o[6] = DK_ADJ_CLAMP_KV; o[7] = DK_ADJ_NORMAL; o[8] = OFF_FLAGS;
}
@FWD_SHIM@
"""

FWD_SHIM = r"""
// the forward (host.KERNEL, DK_SPLIT): the split kernels with work memory of @PFX@_work_bytes(nw) bytes (the handler's
// scratch memory), or k_substep (none)
extern "C" long long @PFX@_work_bytes(int nw) { return DK_SPLIT ? (long long)dk_work_bytes(nw) : 0; }
extern "C" int @PFX@_forward(cudaStream_t st, int nw, const float* qpos, const float* qvel, const float* warm,
                           const float* ctrl, const float* fric, const float* com, int iterations, int nsub, float* oq,
                           float* ov, float* ow, int* info, float* ocon, float* tape, void* work,
                           const float* qfrc, const float* kp_scale) {
#if DK_SPLIT
  return (int)dk_forward_split(st, nw, qpos, qvel, warm, ctrl, fric, com, iterations, nsub, oq, ov, ow, info, ocon,
                               tape, work, sizeof(SimP), sizeof(SimN), qfrc, kp_scale);
#else
  (void)work;
  k_substep<<<nw, 32, sizeof(Sim), st>>>(nw, qpos, qvel, warm, ctrl, fric, com, iterations, nsub, oq, ov, ow, info, ocon,
                                         tape, qfrc, kp_scale);
  return (int)cudaGetLastError();
#endif
}
// the kinematics outputs of a state (g1_engine.kinematics.KIN_DEVICE), and their VJP (gq_in / gv_in: null, or
// added to the result last)
extern "C" int @PFX@_kinematics(cudaStream_t st, int nw, const float* qpos, const float* qvel, const float* com,
                              float* qn, float* xpos, float* xquat, float* cvel, float* sc) {
  k_kin<<<kin_grid(nw), 32 * KIN_WPB, 0, st>>>(nw, qpos, qvel, com, qn, xpos, xquat, cvel, sc);
  return (int)cudaGetLastError();
}
extern "C" int @PFX@_kinematics_vjp(cudaStream_t st, int nw, const float* qpos, const float* qvel, const float* com,
                                  const float* gqn, const float* gxpos, const float* gxquat, const float* gcvel,
                                  const float* gsc, const float* gq_in, const float* gv_in, float* gq, float* gv) {
  k_kin_vjp<<<kin_grid(nw), 32 * KIN_WPB, 0, st>>>(nw, qpos, qvel, com, gqn, gxpos, gxquat, gcvel, gsc, gq_in, gv_in,
                                                   gq, gv);
  return (int)cudaGetLastError();
}
"""

FWD_ATTR = r"""  // the forward kernels' dynamic shared memory, above the 48 KiB default (the cudaFuncSetAttribute opt-in): k_substep
  // always (the fused forward's whole state, struct Sim), k_pre / k_post for a split forward whose halves exceed it
  cudaFuncSetAttribute(k_substep, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Sim));
  cudaFuncSetAttribute(k_pre, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(SimP));
  cudaFuncSetAttribute(k_post, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(SimN));
"""

FFI_UNIT = r"""
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

@DEFINES@
typedef int (*UploadFn)(const float*, cudaStream_t);
typedef int (*AdjointFn)(cudaStream_t, int, int, const float*, const float*, const float*, const float*, const float*,
                         float*, float*, float*, int*, const float*, float, const float*, const float*, float*);
typedef int (*ForwardFn)(cudaStream_t, int, const float*, const float*, const float*, const float*, const float*,
                         const float*, int, int, float*, float*, float*, int*, float*, float*, void*, const float*,
                         const float*);
typedef int (*LaunchFn)(int, long long*);
typedef int (*KinFn)(cudaStream_t, int, const float*, const float*, const float*, float*, float*, float*, float*,
                     float*);
typedef int (*KinVjpFn)(cudaStream_t, int, const float*, const float*, const float*, const float*, const float*,
                        const float*, const float*, const float*, const float*, const float*, float*, float*);
@DECLS@

__device__ unsigned long long g_diag[NDIAG];  // the diagnostic counters (jax_ffi.DIAG), zero at module load
// the last forward call's high-water marks (jax_ffi.LAST; EnginePhysics.stats): slot 0 of any batch size, slot s > 0 of
// the batch size g_slot_B[s]; overwritten by every forward call, read without reset
__device__ unsigned g_last[NSLOT][NLAST];

// one counter += the block's sum of v (warp shuffles, then one integer atomic per warp: exact in any order); every
// thread of the block calls it (no early exit before it), so the full-mask shuffle is in converged control flow
__device__ __forceinline__ void diag_add(int k, unsigned v) {
  __syncwarp();
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0 && v) atomicAdd(&g_diag[k], (unsigned long long)v);
}

// after the forward: info (nw, 4) of k_substep, the tape (null for dk_step) for its records' flags
__global__ void k_diag_fwd(int nw, int nsub, int iterations, const int* info, const float* tape) {
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
  diag_add(D_FWD_UNSUPPORTED_WORLDS, unsup); diag_add(D_FWD_ITERATION_LIMIT_WORLDS, lim);
  diag_add(D_REFACT_RECORDS, refact); diag_add(D_NONCONV_RECORDS, nonconv); diag_add(D_SINGULAR_RECORDS, sing);
  diag_add(D_FWD_CONTACT_OVERFLOW_WORLDS, ocon); diag_add(D_FWD_ROW_OVERFLOW_WORLDS, orow);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd(&g_diag[D_FWD_CALLS], 1ull); atomicAdd(&g_diag[D_FWD_WORLDS], (unsigned long long)nw);
    if (tape) atomicAdd(&g_diag[D_TAPE_RECORDS], (unsigned long long)nw * nsub);
  }
}

// after the forward, one block: the call's record (worlds, contacts summed over the worlds, contacts and constraint rows
// of the most loaded world; info's counts are the last substep's, as MuJoCo Warp's nacon / nefc after its last step)
// into slot 0 and slot (0 < slot < NSLOT, else only slot 0)
__global__ void k_last_fwd(int nw, const int* info, int slot) {
  __shared__ unsigned sm[3][32];
  unsigned sum = 0, cmax = 0, rmax = 0;
  for (int w = threadIdx.x; w < nw; w += blockDim.x) {
    unsigned c = (unsigned)max(info[4 * w + 1], 0), r = (unsigned)max(info[4 * w + 2], 0);
    sum += c; cmax = max(cmax, c); rmax = max(rmax, r);
  }
  for (int o = 16; o > 0; o >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, o);
    cmax = max(cmax, __shfl_down_sync(0xffffffffu, cmax, o));
    rmax = max(rmax, __shfl_down_sync(0xffffffffu, rmax, o));
  }
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
  if (lane == 0) { sm[0][wid] = sum; sm[1][wid] = cmax; sm[2][wid] = rmax; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < nwarp; i++) { sum += sm[0][i]; cmax = max(cmax, sm[1][i]); rmax = max(rmax, sm[2][i]); }
    const unsigned rec[NLAST] = {(unsigned)nw, sum, cmax, rmax};
    for (int k = 0; k < NLAST; k++) {
      g_last[0][k] = rec[k];
      if (slot > 0 && slot < NSLOT) g_last[slot][k] = rec[k];
    }
  }
}
__global__ void k_last_read(unsigned* out) {  // (NSLOT, NLAST)
  for (int k = threadIdx.x; k < NSLOT * NLAST; k += blockDim.x) out[k] = g_last[k / NLAST][k % NLAST];
}

__device__ __forceinline__ bool all_finite(const float* x, int n) {
  bool ok = true;
  for (int i = 0; i < n; i++) ok &= isfinite(x[i]);
  return ok;
}
// after the adjoint: per world which gradients (and incoming cotangents) are non-finite, into info's bits 16-19
__global__ void k_diag_bwd(int nw, const float* gq, const float* gv, const float* gu, const float* cq, const float* cv,
                           int* info) {
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
  diag_add(D_NONFINITE_ROWS, rows); diag_add(D_NONFINITE_QPOS_ROWS, rq); diag_add(D_NONFINITE_QVEL_ROWS, rv);
  diag_add(D_NONFINITE_CTRL_ROWS, ru); diag_add(D_NONFINITE_COTANGENT_ROWS, rc);
  diag_add(D_REPLAY_MISMATCH_WORLDS, rm);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd(&g_diag[D_BWD_CALLS], 1ull); atomicAdd(&g_diag[D_BWD_ROWS], (unsigned long long)nw);
  }
}
__global__ void k_diag_read(unsigned* out, int reset) {  // (NDIAG, 2) low / high words
  for (int k = threadIdx.x; k < NDIAG; k += blockDim.x) {
    unsigned long long v = g_diag[k];
    out[2 * k] = (unsigned)v; out[2 * k + 1] = (unsigned)(v >> 32);
    if (reset) g_diag[k] = 0ull;
  }
}
static unsigned diag_grid(long long nw) { long long g = (nw + 255) / 256; return (unsigned)(g < 1 ? 1 : g > 160 ? 160 : g); }

static std::vector<float> g_consts;  // the packed model constants (dk_set_constants, once per process)
static std::mutex g_mu;
static bool g_ready[MAX_DEVICES];  // per device: constants uploaded and shared-memory limits set in every unit
static long long g_slot_B[NSLOT];  // the batch size of each last-forward slot > 0 (0: free); under g_mu

extern "C" int dk_set_constants(const float* p, long long n) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (n != NCONST) return 1;
  std::vector<float> c(p, p + n);
  if (!g_consts.empty() && c != g_consts) return 2;  // another model: the units were uploaded with this one
  g_consts.swap(c);
  return 0;
}
extern "C" void dk_sizes(long long* o) {  // variant order of the units (VARIANTS), then their sizes
  for (int v = 0; v < NVARIANT; v++) kSizes[v](o + 9 * v);
}
// per variant (VARIANTS order) k_adjoint's launch on device dev: (dynamic shared memory, resident worlds per SM); zeros
// before the device's first call
extern "C" int dk_launch_info(int dev, long long* o) {
  int rc = 0;
  for (int v = 0; v < NVARIANT; v++) rc |= kLaunch[v](dev, o + 2 * v);
  return rc;
}
extern "C" void dk_last_slots(long long* o) {  // the batch size of each last-forward slot (slot 0: any)
  std::lock_guard<std::mutex> lock(g_mu);
  for (int s = 0; s < NSLOT; s++) o[s] = g_slot_B[s];
}
static int last_slot(long long B) {  // the slot of batch size B (a new one on its first call; 0 when all are taken)
  std::lock_guard<std::mutex> lock(g_mu);
  for (int s = 1; s < NSLOT; s++)
    if (g_slot_B[s] == B) return s;
  for (int s = 1; s < NSLOT; s++)
    if (g_slot_B[s] == 0) { g_slot_B[s] = B; return s; }
  return 0;
}

static ffi::Error cuda_error(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return ffi::Error::Success();
  return ffi::Error::Internal(std::string("nablaengine.jax_ffi: ") + what + ": " + cudaGetErrorString(e));
}
// before the first call on a device: every unit's constants, its kernels' dynamic shared-memory limit and k_adjoint's
// launch configuration (adj_launch_smem), on XLA's stream (then synchronized once)
static ffi::Error ensure_ready(cudaStream_t st) {
  int dev = -1;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return cuda_error(e, "cudaGetDevice");
  std::lock_guard<std::mutex> lock(g_mu);
  if (dev < 0 || dev >= MAX_DEVICES) return ffi::Error::Internal("nablaengine.jax_ffi: device ordinal out of range");
  if (g_ready[dev]) return ffi::Error::Success();
  if ((long long)g_consts.size() != NCONST)
    return ffi::Error::Internal("nablaengine.jax_ffi: no model constants (dk_set_constants; use jax_ffi.init())");
  for (int v = 0; v < NVARIANT; v++) {
    int rc = kUpload[v](g_consts.data(), st);
    if (rc == UPLOAD_ERR_SMEM)
      return ffi::Error::Internal("nablaengine.jax_ffi: k_adjoint's launch configuration failed (adj_launch_smem "
                                  "returned 0)");
    if (rc) return cuda_error((cudaError_t)rc, "constant upload");
  }
  e = cudaStreamSynchronize(st);
  if (e != cudaSuccess) return cuda_error(e, "constant upload (synchronize)");
  g_ready[dev] = true;
  return ffi::Error::Success();
}

template <class B> static bool dims_are(const B& b, std::initializer_list<int64_t> want) {  // rank and every dimension
  auto d = b.dimensions();
  if (d.size() != want.size()) return false;
  size_t i = 0;
  for (int64_t n : want)
    if (d[i++] != n) return false;
  return true;
}
static ffi::Error bad(const std::string& m) { return ffi::Error::InvalidArgument("nablaengine.jax_ffi: " + m); }

typedef ffi::Buffer<ffi::F32> BufF;
typedef ffi::Result<ffi::Buffer<ffi::F32>> RetF;
// the kinematics outputs (or their cotangents) of B worlds: qpos_n (B, nq), xpos (B, nbody, 3), xquat (B, nbody,
// 4), cvel (B, nbody, 6), subtree_com (B, nbody, 3)
static bool kin_shapes(int64_t B, const BufF& qn, const BufF& xp, const BufF& xq, const BufF& cv, const BufF& sc) {
  return dims_are(qn, {B, NQ}) && dims_are(xp, {B, NB, 3}) && dims_are(xq, {B, NB, 4}) && dims_are(cv, {B, NB, 6})
         && dims_are(sc, {B, NB, 3});
}
#define KIN_SHAPES "qpos_n (B, nq), xpos (B, nbody, 3), xquat (B, nbody, 4), cvel (B, nbody, 6), subtree_com (B, nbody, 3)"

// outputs con: k_substep's contacts of the last substep (B, MAXCON * 8), not used by the interface (XLA's buffer); tape:
// null for dk_step; scratch: the split forward's work memory (B records of struct Hand, freed when the handler returns,
// in stream order); kin: null, or the five kinematics outputs of the output state (k_kin after the last k_post).
// Every buffer's rank and dimensions are checked before any is read or written
static ffi::Error forward_impl(cudaStream_t st, ffi::ScratchAllocator& scratch, ffi::Buffer<ffi::F32> qpos,
                               ffi::Buffer<ffi::F32> qvel,
                               ffi::Buffer<ffi::F32> warm, ffi::Buffer<ffi::F32> ctrl, ffi::Buffer<ffi::F32> fric,
                               ffi::Buffer<ffi::F32> com, int32_t iterations, int32_t substeps,
                               ffi::Result<ffi::Buffer<ffi::F32>> qo, ffi::Result<ffi::Buffer<ffi::F32>> vo,
                               ffi::Result<ffi::Buffer<ffi::F32>> wo, ffi::Result<ffi::Buffer<ffi::S32>> info,
                               ffi::Result<ffi::Buffer<ffi::F32>> con, ffi::Result<ffi::Buffer<ffi::F32>>* tape,
                               RetF* kin = nullptr) {
  auto d = qpos.dimensions();
  if (d.size() != 2 || d[1] != NQ) return bad("qpos must be (B, nq) float32");
  const int64_t B = d[0];
  if (!dims_are(qvel, {B, NV}) || !dims_are(warm, {B, NV}) || !dims_are(ctrl, {B, NU}) || !dims_are(fric, {B})
      || !dims_are(com, {B, COM_WIDTH}))
    return bad("shapes: qpos (B, nq), qvel (B, nv), warm (B, nv), ctrl (B, nu), friction (B,), com (B, COM_WIDTH)");
  if (substeps < 1 || iterations < 0) return bad("substeps >= 1 and iterations >= 0");
  if (!dims_are(*qo, {B, NQ}) || !dims_are(*vo, {B, NV}) || !dims_are(*wo, {B, NV}) || !dims_are(*info, {B, 4}))
    return bad("outputs: qpos' (B, nq), qvel' (B, nv), warm' (B, nv) float32, info (B, 4) int32");
  if (!dims_are(*con, {B, MAXCON * 8})) return bad("contact output must be (B, MAXCON * 8) float32");
  if (tape && !dims_are(**tape, {B, (int64_t)substeps * REC_WORDS}))
    return bad("tape must be (B, substeps * REC_WORDS) float32");
  if (kin && !kin_shapes(B, *kin[0], *kin[1], *kin[2], *kin[3], *kin[4]))
    return bad("kinematics outputs: " KIN_SHAPES " float32");
  if (B > INT_MAX) return bad("B too large");
  if (B == 0) return ffi::Error::Success();
  ffi::Error r = ensure_ready(st);
  if (r.failure()) return r;
  float* tp = tape ? (*tape)->typed_data() : nullptr;
  void* work = nullptr;
  if (const long long wb = kWorkBytes((int)B)) {  // (XLA's scratch allocator takes alignments up to 16)
    std::optional<void*> a = scratch.Allocate((size_t)wb, 16);
    if (!a) return ffi::Error::Internal("nablaengine.jax_ffi: no scratch memory for the forward's work buffer");
    work = *a;
  }
  int rc = kForward(st, (int)B, qpos.typed_data(), qvel.typed_data(), warm.typed_data(), ctrl.typed_data(),
                    fric.typed_data(), com.typed_data(), iterations, substeps, qo->typed_data(), vo->typed_data(),
                    wo->typed_data(), info->typed_data(), con->typed_data(), tp, work, nullptr, nullptr);
  if (rc) return cuda_error((cudaError_t)rc, "k_substep launch");
  if (kin) {  // the output state's kinematics (the outputs are complete after the last k_post, in stream order)
    rc = kKin(st, (int)B, qo->typed_data(), vo->typed_data(), com.typed_data(), kin[0]->typed_data(),
              kin[1]->typed_data(), kin[2]->typed_data(), kin[3]->typed_data(), kin[4]->typed_data());
    if (rc) return cuda_error((cudaError_t)rc, "k_kin launch");
  }
  k_diag_fwd<<<diag_grid(B), 256, 0, st>>>((int)B, substeps, iterations, info->typed_data(), tp);
  ffi::Error rd = cuda_error(cudaGetLastError(), "k_diag_fwd launch");
  if (rd.failure()) return rd;
  k_last_fwd<<<1, 1024, 0, st>>>((int)B, info->typed_data(), last_slot(B));
  return cuda_error(cudaGetLastError(), "k_last_fwd launch");
}

static ffi::Error step_impl(cudaStream_t st, ffi::ScratchAllocator scratch, ffi::Buffer<ffi::F32> qpos,
                            ffi::Buffer<ffi::F32> qvel, ffi::Buffer<ffi::F32> warm, ffi::Buffer<ffi::F32> ctrl,
                            ffi::Buffer<ffi::F32> fric, ffi::Buffer<ffi::F32> com, int32_t iterations, int32_t substeps,
                            ffi::Result<ffi::Buffer<ffi::F32>> qo, ffi::Result<ffi::Buffer<ffi::F32>> vo,
                            ffi::Result<ffi::Buffer<ffi::F32>> wo, ffi::Result<ffi::Buffer<ffi::S32>> info,
                            ffi::Result<ffi::Buffer<ffi::F32>> con) {
  return forward_impl(st, scratch, qpos, qvel, warm, ctrl, fric, com, iterations, substeps, qo, vo, wo, info, con,
                      nullptr);
}

static ffi::Error step_fwd_impl(cudaStream_t st, ffi::ScratchAllocator scratch, ffi::Buffer<ffi::F32> qpos,
                                ffi::Buffer<ffi::F32> qvel, ffi::Buffer<ffi::F32> warm, ffi::Buffer<ffi::F32> ctrl,
                                ffi::Buffer<ffi::F32> fric, ffi::Buffer<ffi::F32> com, int32_t iterations,
                                int32_t substeps,
                                ffi::Result<ffi::Buffer<ffi::F32>> qo, ffi::Result<ffi::Buffer<ffi::F32>> vo,
                                ffi::Result<ffi::Buffer<ffi::F32>> wo, ffi::Result<ffi::Buffer<ffi::S32>> info,
                                ffi::Result<ffi::Buffer<ffi::F32>> con, ffi::Result<ffi::Buffer<ffi::F32>> tape) {
  return forward_impl(st, scratch, qpos, qvel, warm, ctrl, fric, com, iterations, substeps, qo, vo, wo, info, con,
                      &tape);
}

// dk_step / dk_step_fwd with the output state's kinematics (qpos_n, xpos, xquat, cvel, subtree_com) as five more
// outputs
static ffi::Error step_kin_impl(cudaStream_t st, ffi::ScratchAllocator scratch, BufF qpos, BufF qvel, BufF warm,
                                BufF ctrl, BufF fric, BufF com, int32_t iterations, int32_t substeps, RetF qo, RetF vo,
                                RetF wo, ffi::Result<ffi::Buffer<ffi::S32>> info, RetF con, RetF qn, RetF xp, RetF xq,
                                RetF cv, RetF sc) {
  RetF kin[5] = {qn, xp, xq, cv, sc};
  return forward_impl(st, scratch, qpos, qvel, warm, ctrl, fric, com, iterations, substeps, qo, vo, wo, info, con,
                      nullptr, kin);
}
static ffi::Error step_fwd_kin_impl(cudaStream_t st, ffi::ScratchAllocator scratch, BufF qpos, BufF qvel, BufF warm,
                                    BufF ctrl, BufF fric, BufF com, int32_t iterations, int32_t substeps, RetF qo,
                                    RetF vo, RetF wo, ffi::Result<ffi::Buffer<ffi::S32>> info, RetF con, RetF tape,
                                    RetF qn, RetF xp, RetF xq, RetF cv, RetF sc) {
  RetF kin[5] = {qn, xp, xq, cv, sc};
  return forward_impl(st, scratch, qpos, qvel, warm, ctrl, fric, com, iterations, substeps, qo, vo, wo, info, con,
                      &tape, kin);
}

// the step's backward with its kinematics outputs: the output state (qpos', qvel') and the cotangents of qpos_n,
// xpos, xquat, cvel, subtree_com, pulled back onto (qpos', qvel') by k_kin_vjp and added to their cotangents before
// k_adjoint
struct KinBwd {
  const BufF *q1, *v1, *gqn, *gxp, *gxq, *gcv, *gsc;
};

// ctrl: the control step's (B, nu) for clamp_kv 2 (dk_step_bwd_soft), null for 0 / 1 (dk_step_bwd); clamp_soft: 2's
// delta; kin (with scratch): null, or the kinematics outputs' VJP inputs (the combined cotangents in scratch memory)
static ffi::Error bwd_impl(cudaStream_t st, ffi::Buffer<ffi::F32> tape, ffi::Buffer<ffi::F32> fric,
                           ffi::Buffer<ffi::F32> com, const ffi::Buffer<ffi::F32>* ctrl, ffi::Buffer<ffi::F32> gq_out,
                           ffi::Buffer<ffi::F32> gv_out, int32_t substeps, int32_t clamp_kv, int32_t normal,
                           float clamp_soft, ffi::Result<ffi::Buffer<ffi::F32>> gq,
                           ffi::Result<ffi::Buffer<ffi::F32>> gv, ffi::Result<ffi::Buffer<ffi::F32>> gu,
                           ffi::Result<ffi::Buffer<ffi::S32>> info, const KinBwd* kin = nullptr,
                           ffi::ScratchAllocator* scratch = nullptr) {
  auto d = tape.dimensions();
  if (substeps < 1 || d.size() != 2 || d[1] != (int64_t)substeps * REC_WORDS)
    return bad("tape must be (B, substeps * REC_WORDS) float32");
  const int64_t B = d[0];
  if (!dims_are(fric, {B}) || !dims_are(com, {B, COM_WIDTH}) || !dims_are(gq_out, {B, NQ}) || !dims_are(gv_out, {B, NV}))
    return bad("shapes: friction (B,), com (B, COM_WIDTH), g_qpos (B, nq), g_qvel (B, nv)");
  if (ctrl && !dims_are(*ctrl, {B, NU})) return bad("ctrl must be (B, nu) float32");
  if (kin && (!dims_are(*kin->q1, {B, NQ}) || !dims_are(*kin->v1, {B, NV})
              || !kin_shapes(B, *kin->gqn, *kin->gxp, *kin->gxq, *kin->gcv, *kin->gsc)))
    return bad("kinematics: qpos' (B, nq), qvel' (B, nv) and the cotangents of " KIN_SHAPES " float32");
  if (!dims_are(*gq, {B, NQ}) || !dims_are(*gv, {B, NV}) || !dims_are(*gu, {B, NU}) || !dims_are(*info, {B}))
    return bad("outputs: g_qpos (B, nq), g_qvel (B, nv), g_ctrl (B, nu) float32, info (B,) int32");
  if (clamp_kv < 0 || clamp_kv > 2 || normal < 0 || normal > 3) return bad("clamp_kv in {0, 1, 2}, normal in 0..3");
  if ((clamp_kv == 2) != (ctrl != nullptr))
    return bad("clamp_kv 2 (the soft clamp) through dk_step_bwd_soft with ctrl, 0 and 1 through dk_step_bwd");
  if (ctrl && !(clamp_soft >= 0.f && clamp_soft < INFINITY)) return bad("clamp_soft must be finite and >= 0");
  AdjointFn adj = kAdjoint[clamp_kv][normal];
  if (!adj) return bad("this adjoint variant was not built");
  if (B > INT_MAX) return bad("B too large");
  if (B == 0) return ffi::Error::Success();
  ffi::Error r = ensure_ready(st);
  if (r.failure()) return r;
  const float *cq = gq_out.typed_data(), *cv = gv_out.typed_data();  // the cotangents of (qpos', qvel')
  if (kin) {  // + the kinematics outputs' pulled back onto (qpos', qvel'), in scratch memory (freed in stream order)
    std::optional<void*> a = scratch->Allocate((size_t)B * (NQ + NV) * sizeof(float), 16);
    if (!a) return ffi::Error::Internal("nablaengine.jax_ffi: no scratch memory for the combined cotangents");
    float* sq = static_cast<float*>(*a);
    float* sv = sq + (size_t)B * NQ;
    int rk = kKinVjp(st, (int)B, kin->q1->typed_data(), kin->v1->typed_data(), com.typed_data(),
                     kin->gqn->typed_data(), kin->gxp->typed_data(), kin->gxq->typed_data(), kin->gcv->typed_data(),
                     kin->gsc->typed_data(), cq, cv, sq, sv);
    if (rk) return cuda_error((cudaError_t)rk, "k_kin_vjp launch");
    cq = sq;
    cv = sv;
  }
  int rc = adj(st, (int)B, substeps, tape.typed_data(), fric.typed_data(), com.typed_data(), cq, cv, gq->typed_data(),
               gv->typed_data(), gu->typed_data(), info->typed_data(), ctrl ? ctrl->typed_data() : nullptr,
               clamp_soft, nullptr, nullptr, nullptr);
  if (rc) return cuda_error((cudaError_t)rc, "k_adjoint launch");
  k_diag_bwd<<<diag_grid(B), 256, 0, st>>>((int)B, gq->typed_data(), gv->typed_data(), gu->typed_data(), cq, cv,
                                            info->typed_data());
  return cuda_error(cudaGetLastError(), "k_diag_bwd launch");
}

static ffi::Error step_bwd_impl(cudaStream_t st, ffi::Buffer<ffi::F32> tape, ffi::Buffer<ffi::F32> fric,
                                ffi::Buffer<ffi::F32> com, ffi::Buffer<ffi::F32> gq_out, ffi::Buffer<ffi::F32> gv_out,
                                int32_t substeps, int32_t clamp_kv, int32_t normal,
                                ffi::Result<ffi::Buffer<ffi::F32>> gq, ffi::Result<ffi::Buffer<ffi::F32>> gv,
                                ffi::Result<ffi::Buffer<ffi::F32>> gu, ffi::Result<ffi::Buffer<ffi::S32>> info) {
  return bwd_impl(st, tape, fric, com, nullptr, gq_out, gv_out, substeps, clamp_kv, normal, 0.f, gq, gv, gu, info);
}

// clamp_kv 2: the soft clamp's backward, with the step's ctrl (B, nu) and delta (attribute clamp_soft, float32)
static ffi::Error step_bwd_soft_impl(cudaStream_t st, ffi::Buffer<ffi::F32> tape, ffi::Buffer<ffi::F32> fric,
                                     ffi::Buffer<ffi::F32> com, ffi::Buffer<ffi::F32> ctrl,
                                     ffi::Buffer<ffi::F32> gq_out, ffi::Buffer<ffi::F32> gv_out, int32_t substeps,
                                     int32_t normal, float clamp_soft, ffi::Result<ffi::Buffer<ffi::F32>> gq,
                                     ffi::Result<ffi::Buffer<ffi::F32>> gv, ffi::Result<ffi::Buffer<ffi::F32>> gu,
                                     ffi::Result<ffi::Buffer<ffi::S32>> info) {
  return bwd_impl(st, tape, fric, com, &ctrl, gq_out, gv_out, substeps, 2, normal, clamp_soft, gq, gv, gu, info);
}

// dk_step_bwd / dk_step_bwd_soft of a step with kinematics outputs: + the output state q1, v1 and the cotangents
// of qpos_n, xpos, xquat, cvel, subtree_com
static ffi::Error step_bwd_kin_impl(cudaStream_t st, ffi::ScratchAllocator scratch, BufF tape, BufF fric, BufF com,
                                    BufF q1, BufF v1, BufF gq_out, BufF gv_out, BufF gqn, BufF gxp, BufF gxq,
                                    BufF gcv, BufF gsc, int32_t substeps, int32_t clamp_kv, int32_t normal, RetF gq,
                                    RetF gv, RetF gu, ffi::Result<ffi::Buffer<ffi::S32>> info) {
  const KinBwd k{&q1, &v1, &gqn, &gxp, &gxq, &gcv, &gsc};
  return bwd_impl(st, tape, fric, com, nullptr, gq_out, gv_out, substeps, clamp_kv, normal, 0.f, gq, gv, gu, info, &k,
                  &scratch);
}
static ffi::Error step_bwd_soft_kin_impl(cudaStream_t st, ffi::ScratchAllocator scratch, BufF tape, BufF fric,
                                         BufF com, BufF ctrl, BufF q1, BufF v1, BufF gq_out, BufF gv_out, BufF gqn,
                                         BufF gxp, BufF gxq, BufF gcv, BufF gsc, int32_t substeps, int32_t normal,
                                         float clamp_soft, RetF gq, RetF gv, RetF gu,
                                         ffi::Result<ffi::Buffer<ffi::S32>> info) {
  const KinBwd k{&q1, &v1, &gqn, &gxp, &gxq, &gcv, &gsc};
  return bwd_impl(st, tape, fric, com, &ctrl, gq_out, gv_out, substeps, 2, normal, clamp_soft, gq, gv, gu, info, &k,
                  &scratch);
}

// jax_ffi.kinematics: the kinematics outputs of a state (qpos, qvel, com), and their VJP (the cotangents of the
// five outputs -> of qpos and qvel)
static ffi::Error kin_state_dims(const BufF& qpos, const BufF& qvel, const BufF& com, int64_t* B) {
  auto d = qpos.dimensions();
  if (d.size() != 2 || d[1] != NQ) return bad("qpos must be (B, nq) float32");
  *B = d[0];
  if (!dims_are(qvel, {*B, NV}) || !dims_are(com, {*B, COM_WIDTH})) return bad("shapes: qpos (B, nq), qvel (B, nv), com (B, COM_WIDTH)");
  if (*B > INT_MAX) return bad("B too large");
  return ffi::Error::Success();
}
static ffi::Error kin_impl(cudaStream_t st, BufF qpos, BufF qvel, BufF com, RetF qn, RetF xp, RetF xq, RetF cv,
                           RetF sc) {
  int64_t B = 0;
  ffi::Error e = kin_state_dims(qpos, qvel, com, &B);
  if (e.failure()) return e;
  if (!kin_shapes(B, *qn, *xp, *xq, *cv, *sc)) return bad("outputs: " KIN_SHAPES " float32");
  if (B == 0) return ffi::Error::Success();
  ffi::Error r = ensure_ready(st);
  if (r.failure()) return r;
  int rc = kKin(st, (int)B, qpos.typed_data(), qvel.typed_data(), com.typed_data(), qn->typed_data(), xp->typed_data(),
                xq->typed_data(), cv->typed_data(), sc->typed_data());
  return rc ? cuda_error((cudaError_t)rc, "k_kin launch") : ffi::Error::Success();
}
static ffi::Error kin_bwd_impl(cudaStream_t st, BufF qpos, BufF qvel, BufF com, BufF gqn, BufF gxp, BufF gxq, BufF gcv,
                               BufF gsc, RetF gq, RetF gv) {
  int64_t B = 0;
  ffi::Error e = kin_state_dims(qpos, qvel, com, &B);
  if (e.failure()) return e;
  if (!kin_shapes(B, gqn, gxp, gxq, gcv, gsc)) return bad("cotangents: " KIN_SHAPES " float32");
  if (!dims_are(*gq, {B, NQ}) || !dims_are(*gv, {B, NV})) return bad("outputs: g_qpos (B, nq), g_qvel (B, nv) float32");
  if (B == 0) return ffi::Error::Success();
  ffi::Error r = ensure_ready(st);
  if (r.failure()) return r;
  int rc = kKinVjp(st, (int)B, qpos.typed_data(), qvel.typed_data(), com.typed_data(), gqn.typed_data(),
                   gxp.typed_data(), gxq.typed_data(), gcv.typed_data(), gsc.typed_data(), nullptr, nullptr,
                   gq->typed_data(), gv->typed_data());
  return rc ? cuda_error((cudaError_t)rc, "k_kin_vjp launch") : ffi::Error::Success();
}

static ffi::Error diag_impl(cudaStream_t st, ffi::RemainingArgs, int32_t reset, ffi::Result<ffi::Buffer<ffi::U32>> out) {
  if (!dims_are(*out, {NDIAG, 2})) return bad("diagnostics output must be (NDIAG, 2) uint32");
  k_diag_read<<<1, 32, 0, st>>>(out->typed_data(), reset);
  return cuda_error(cudaGetLastError(), "k_diag_read launch");
}

static ffi::Error last_impl(cudaStream_t st, ffi::RemainingArgs, ffi::Result<ffi::Buffer<ffi::U32>> out) {
  if (!dims_are(*out, {NSLOT, NLAST})) return bad("last-forward output must be (NSLOT, NLAST) uint32");
  k_last_read<<<1, 64, 0, st>>>(out->typed_data());
  return cuda_error(cudaGetLastError(), "k_last_read launch");
}

#define FWD_BIND(B) B.Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()                        \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()                          \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()                          \
    .Attr<int32_t>("iterations").Attr<int32_t>("substeps")                                                           \
    .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::S32>>() \
    .Ret<ffi::Buffer<ffi::F32>>()
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStep, step_impl, FWD_BIND(ffi::Ffi::Bind()));
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepFwd, step_fwd_impl, FWD_BIND(ffi::Ffi::Bind()).Ret<ffi::Buffer<ffi::F32>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepBwd, step_bwd_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
                                  .Attr<int32_t>("substeps").Attr<int32_t>("clamp_kv").Attr<int32_t>("normal")
                                  .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepBwdSoft, step_bwd_soft_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
                                  .Attr<int32_t>("substeps").Attr<int32_t>("normal").Attr<float>("clamp_soft")
                                  .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>());
#define F32ARG .Arg<ffi::Buffer<ffi::F32>>()
#define F32RET .Ret<ffi::Buffer<ffi::F32>>()
#define KIN5(X) X X X X X  // the five kinematics outputs (or their cotangents)
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepKin, step_kin_impl, FWD_BIND(ffi::Ffi::Bind()) KIN5(F32RET));
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepFwdKin, step_fwd_kin_impl, FWD_BIND(ffi::Ffi::Bind()) F32RET KIN5(F32RET));
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepBwdKin, step_bwd_kin_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
                                  F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG KIN5(F32ARG)
                                  .Attr<int32_t>("substeps").Attr<int32_t>("clamp_kv").Attr<int32_t>("normal")
                                  F32RET F32RET F32RET .Ret<ffi::Buffer<ffi::S32>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepBwdSoftKin, step_bwd_soft_kin_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
                                  F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG KIN5(F32ARG)
                                  .Attr<int32_t>("substeps").Attr<int32_t>("normal").Attr<float>("clamp_soft")
                                  F32RET F32RET F32RET .Ret<ffi::Buffer<ffi::S32>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkKin, kin_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() F32ARG F32ARG F32ARG
                                  KIN5(F32RET));
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkKinBwd, kin_bwd_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() F32ARG F32ARG F32ARG
                                  KIN5(F32ARG) F32RET F32RET);
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkDiag, diag_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs()
                                  .Attr<int32_t>("reset").Ret<ffi::Buffer<ffi::U32>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkLast, last_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs()
                                  .Ret<ffi::Buffer<ffi::U32>>());
"""

_LOCK = threading.Lock()
_STATE = {}  # the loaded library (init)
STATS = dict(builds=0, build_seconds=0.0, loads=0)  # this process: nvcc builds (cache misses) and library loads


def _tag(v):
    return f"dk_kv{v[0]}_n{v[1]}"


def _model(model_dir=None, model=None, foot_pairs=None):
    """(MjModel, foot_pairs): the given model, else the fixtures directory (model_dir, else fixtures.fixtures_dir():
    $DK_FIXTURES, else fixtures/mjlab)."""
    if model is not None:
        assert foot_pairs is not None, "init(model=...) needs foot_pairs (the foot-floor pair indices)"
        return model, [int(p) for p in foot_pairs]
    d = model_dir or fixtures.fixtures_dir()
    m, mt = fixtures.load_model(d, check=None)  # init checks the engine scope next
    return m, mt["foot_pairs"]


def library_sources(m, foot_pairs, forward_unit=False, target=None, com_body_id=None):
    """({file name: CUDA source} of the library's translation units, the packed constants, the per-unit -D options).
    forward_unit: the forward and kinematics kernels in a unit of their own, FWD_UNIT.cu (the text of FWD_VARIANT's unit,
    built with other architecture flags: cuda.arch_flags('forward')), every adjoint variant's unit without them. target:
    the build target (default cuda.arch()), whose forward-unit -D options (cuda.unit_defines: the register caps on
    sm_90) the unit with the forward kernels gets."""
    consts, lays, sizes = adjoint.engine_constants(m, foot_pairs, com_body_id=com_body_id)
    load = host.loader(lays, sizes)
    load = re.sub(r"cudaMemcpyToSymbol\(([^;]*)\);", r"cudaMemcpyToSymbolAsync(\1, 0, cudaMemcpyHostToDevice, st);", load)
    assert "cudaMemcpyToSymbol(" not in load
    engine = adjoint.cuda_source(m, com_body_id=com_body_id)
    # Native collision's standard declarations must stay outside the variant namespace.
    includes = []
    for header in ("stdint.h", "stddef.h", "float.h", "math.h"):
        line = f"#include <{header}>\n"
        if line in engine:
            includes.append(line.rstrip())
            engine = engine.replace(line, "")
    assert engine.count("#include <cuda_runtime.h>\n") == 1 and engine.count("#include") == 1, \
        "the engine source has an unexpected include inside the variant namespace"
    engine = engine.replace("#include <cuda_runtime.h>\n", "")
    units, defines = {}, {}

    def unit(v, fwd, pfx):
        src = (VARIANT_UNIT.replace("@FWD_SHIM@", FWD_SHIM if fwd else "")
               .replace("@ENGINE_INCLUDES@", "\n".join(includes))
               .replace("@FWD_ATTR@", FWD_ATTR if fwd else "")
               .replace("@PFX@", pfx).replace("@MAX_DEVICES@", str(MAX_DEVICES))
               .replace("@UPLOAD_ERR_SMEM@", str(UPLOAD_ERR_SMEM)).replace("@LOAD_CONSTS@", load)
               .replace("@ENGINE@", engine + (host.KERNEL + g1_kin.KIN_DEVICE if fwd else "")))  # last: never scanned
        units[f"{pfx}.cu"] = src
        defines[f"{pfx}.cu"] = [f"-DDK_ADJ_CLAMP_KV={v[0]}", f"-DDK_ADJ_NORMAL={v[1]}"]

    for v in VARIANTS:
        unit(v, v == FWD_VARIANT and not forward_unit, _tag(v))
    if forward_unit:  # FWD_VARIANT's unit text as the forward unit FWD_UNIT (its k_adjoint is not called)
        unit(FWD_VARIANT, True, FWD_UNIT)
    fp = FWD_UNIT if forward_unit else _tag(FWD_VARIANT)  # the unit with the forward and kinematics entries
    defines[f"{fp}.cu"] += cuda.unit_defines("forward", target, defines[f"{fp}.cu"])  # sm_90: the register caps
    defines[f"{fp}.cu"] += cuda.forward_build_defines()  # $DK_FORWARD_SPLIT / $DK_SPLIT_ORDER (bitwise unchanged)
    defs = [f"#define {k} {val}" for k, val in (
        ("NQ", m.nq), ("NV", m.nv), ("NU", m.nu), ("NB", m.nbody), ("COM_WIDTH", 3*len(g1_source.free_roots(m))),
        ("MAXCON", g1_step.maxcon(m)),
        ("REC_WORDS", g1_tape.rec_words(m)),
        ("OFF_FLAGS", g1_tape.layout(m)["flags"][0]), ("TAPE_REFACT", f"{g1_tape.FLAG_REFACT}u"),
        ("TAPE_NONCONV", f"{g1_tape.FLAG_NONCONV}u"), ("TAPE_SINGULAR", f"{g1_tape.FLAG_SINGULAR}u"),
        ("ADJ_REPLAY_MISMATCH", adjoint.ADJ_REPLAY_MISMATCH), ("NONFINITE_QPOS", NONFINITE_QPOS),
        ("NONFINITE_QVEL", NONFINITE_QVEL), ("NONFINITE_CTRL", NONFINITE_CTRL), ("NONFINITE_COT", NONFINITE_COT),
        ("NONFINITE_SHIFT", NONFINITE_SHIFT), ("NCONST", len(consts)), ("NVARIANT", len(VARIANTS)),
        ("NDIAG", len(DIAG)), ("MAX_DEVICES", MAX_DEVICES), ("NSLOT", NSLOT), ("NLAST", len(LAST)),
        ("UPLOAD_ERR_SMEM", UPLOAD_ERR_SMEM), ("OVERFLOW_CONTACTS", OVERFLOW_CONTACTS),
        ("OVERFLOW_ROWS", OVERFLOW_ROWS))]
    defs += [f"#define D_{name.upper()} {i}" for i, name in enumerate(DIAG)]
    decls = []
    for v in VARIANTS:
        t = _tag(v)
        decls.append(f'extern "C" int {t}_upload(const float*, cudaStream_t);')
        decls.append(f'extern "C" int {t}_adjoint(cudaStream_t, int, int, const float*, const float*, const float*, '
                     f'const float*, const float*, float*, float*, float*, int*, const float*, float, '
                     f'const float*, const float*, float*);')
        decls.append(f'extern "C" void {t}_sizes(long long*);')
        decls.append(f'extern "C" int {t}_launch(int, long long*);')
    decls.append(f'extern "C" int {fp}_forward(cudaStream_t, int, const float*, const float*, '
                 f'const float*, const float*, const float*, const float*, int, int, float*, float*, float*, int*, '
                 f'float*, float*, void*, const float*, const float*);')
    decls.append(f'extern "C" long long {fp}_work_bytes(int);')
    decls.append(f'extern "C" int {fp}_kinematics(cudaStream_t, int, const float*, const float*, '
                 f'const float*, float*, float*, float*, float*, float*);')
    decls.append(f'extern "C" int {fp}_kinematics_vjp(cudaStream_t, int, const float*, const float*, '
                 f'const float*, const float*, const float*, const float*, const float*, const float*, const float*, '
                 f'const float*, float*, float*);')
    decls.append(f"static const KinFn kKin = {fp}_kinematics;")
    decls.append(f"static const KinVjpFn kKinVjp = {fp}_kinematics_vjp;")
    upload = [f"{_tag(v)}_upload" for v in VARIANTS]
    if forward_unit:  # the forward unit's constants and attributes with FWD_VARIANT's unit's
        t = _tag(FWD_VARIANT)
        decls.append(f'extern "C" int {FWD_UNIT}_upload(const float*, cudaStream_t);')
        decls.append(f"static int {t}_upload_fwd(const float* p, cudaStream_t st) {{ const int rc = {t}_upload(p, st); "
                     f"return rc ? rc : {FWD_UNIT}_upload(p, st); }}")
        upload[VARIANTS.index(FWD_VARIANT)] = f"{t}_upload_fwd"
    decls.append("static const UploadFn kUpload[NVARIANT] = {" + ", ".join(upload) + "};")
    decls.append("typedef void (*SizesFn)(long long*);")
    decls.append("static const SizesFn kSizes[NVARIANT] = {" + ", ".join(f"{_tag(v)}_sizes" for v in VARIANTS) + "};")
    decls.append("static const LaunchFn kLaunch[NVARIANT] = {" + ", ".join(f"{_tag(v)}_launch" for v in VARIANTS)
                 + "};")
    adj = [[f"{_tag((kv, n))}_adjoint" if (kv, n) in VARIANTS else "nullptr" for n in range(4)] for kv in range(3)]
    decls.append("static const AdjointFn kAdjoint[3][4] = {" + ", ".join("{" + ", ".join(r) + "}" for r in adj) + "};")
    decls.append(f"static const ForwardFn kForward = {fp}_forward;")
    decls.append(f"static long long (*const kWorkBytes)(int) = {fp}_work_bytes;")
    units["dk_ffi.cu"] = FFI_UNIT.replace("@DEFINES@", "\n".join(defs)).replace("@DECLS@", "\n".join(decls))
    defines["dk_ffi.cu"] = []
    return units, consts, defines


def build_plan(m, foot_pairs, arch, com_body_id=None):
    """How init() builds the library for build target arch: dict(units, consts, defines (library_sources), unit_arch
    ({unit: architecture flags} where they are not -arch=<arch>: every unit on sm_120 by default, the forward unit only
    with $DK_SM120_FRONTEND=split; cuda.arch_flags), arch_flags (forward, adjoint), split (the forward / kinematics
    kernels in a unit of their own, FWD_UNIT), forward_unit (the prefix of the forward / kinematics entries),
    forward_defines (the -D options the forward unit gets for the target: cuda.unit_defines)). Also the build plan of
    scripts/identity_gate.py."""
    fwd_flags = cuda.arch_flags("forward", arch)  # the forward / kinematics unit's (NVVM 7 on sm_120)
    adj_flags = cuda.arch_flags(None, arch)  # the adjoint units' and the FFI unit's (NVVM 7 on sm_120 with n7)
    split = fwd_flags != adj_flags
    units, consts, defines = library_sources(m, foot_pairs, forward_unit=split, target=arch, com_body_id=com_body_id)
    fu = FWD_UNIT if split else _tag(FWD_VARIANT)
    fwd_defines = [d for d in defines[f"{fu}.cu"]
                   if not d.startswith(("-DDK_ADJ_CLAMP_KV=", "-DDK_ADJ_NORMAL="))]  # (the register caps)
    unit_arch = {n: f for n, f in ((n, fwd_flags if split and n == f"{FWD_UNIT}.cu" else adj_flags) for n in units)
                 if f != [f"-arch={arch}"]}
    return dict(units=units, consts=consts, defines=defines, unit_arch=unit_arch,
                arch_flags=dict(forward=fwd_flags, adjoint=adj_flags), split=split, forward_unit=fu,
                forward_defines=fwd_defines)


def common_flags(arch):
    """The nvcc flags of every library unit that has no architecture flags of its own (build_plan's unit_arch)."""
    return ["-O3", f"-arch={arch}", "-std=c++17", "-Xcompiler", "-fPIC"]


def unit_command(name, arch, defines, unit_arch=None, include=None):
    """nvcc's arguments (after the nvcc path, before "-c -o <object> <source>") for the library's translation unit
    name: common_flags(arch), or -O3 with the unit's own architecture flags (unit_arch) and the rest of them, then its
    -D options (defines[name]), and for the FFI unit dk_ffi.cu -I<include> (XLA's FFI headers)."""
    common = common_flags(arch)
    unit_arch = unit_arch or {}
    flags = ["-O3", *unit_arch[name], *common[2:]] if name in unit_arch else common
    inc = [f"-I{include}"] if name == "dk_ffi.cu" else []
    return [*flags, *defines[name], *inc]


def _device_arch():
    """sm_XY of the first JAX GPU, None without one."""
    try:
        import jax
        return "sm_" + str(jax.devices("gpu")[0].compute_capability).replace(".", "")
    except Exception:  # noqa: BLE001
        return None


def _arch():
    """$DK_ARCH (nablaengine.cuda.arch(), which warns when it is not the GPU's), else the first JAX GPU's compute
    capability, else cuda.arch() (nvidia-smi)."""
    if os.environ.get("DK_ARCH"):
        return cuda.arch()
    return _device_arch() or cuda.arch()


def cache_dir():
    d = Path(os.environ.get("DK_CACHE_DIR") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
             / "nablaengine")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build(units, defines, arch, include, consts, unit_arch=None, prefix="libdk_g1"):
    """The library for these sources: from the cache, or built by nvcc (units in parallel, then linked). The key also
    covers the model constants (uploaded at run time, not compiled in: a library per model), the flags, the nvcc
    version and XLA's FFI headers. unit_arch: {unit: architecture flags} in place of -arch=<arch> (every unit on
    sm_120 by default, the forward unit with split: cuda.arch_flags), recorded in the key, as $DK_SM120_FRONTEND is on
    sm_120 and the register-cap switch where the target has one (cuda.reg_caps; the caps' -D options are in
    `defines`). include: XLA's FFI headers for the FFI unit dk_ffi.cu; None for a library without that unit (the
    PyTorch interface's, nablaengine.torch_ffi), whose key then has no header hash. prefix: the library's file name
    prefix in the cache. Returns (path, built: bool, the cache key)."""
    nvcc = cuda.nvcc()
    ver = subprocess.run([nvcc, "--version"], check=True, stdout=subprocess.PIPE, universal_newlines=True).stdout
    if include is None:
        assert "dk_ffi.cu" not in units, "the FFI unit dk_ffi.cu needs XLA's FFI headers (include)"
        hdr = None
    else:
        hdr = hashlib.sha256(b"".join((Path(include) / "xla" / "ffi" / "api" / f).read_bytes()
                                      for f in ("api.h", "ffi.h", "c_api.h"))).hexdigest()
    common = common_flags(arch)
    unit_arch = dict(unit_arch or {})
    kd = dict(units=units, defines=defines, common=common, nvcc=ver, ffi_headers=hdr,
              consts=hashlib.sha256(consts.tobytes()).hexdigest())
    if unit_arch:
        kd["unit_arch"] = unit_arch
    if arch in cuda.N7_VIRTUAL:
        kd["sm120_frontend"] = cuda.frontend()
    if arch in cuda.REG_CAPS:
        kd["reg_caps"] = cuda.reg_caps(arch)
    key = hashlib.sha256(json.dumps(kd, sort_keys=True).encode()).hexdigest()[:20]
    out = cache_dir() / f"{prefix}_{arch}_{key}.so"
    if out.exists():
        return out, False, key
    # Keep this inode in the cache: deleting a lock file can split waiters
    # between two independent locks. Different artifacts use different locks.
    with out.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if out.exists():
            return out, False, key
        t0 = time.time()
        work = Path(tempfile.mkdtemp(prefix=f"build_{key}_", dir=cache_dir()))
        try:
            def compile_unit(name):
                (work / name).write_text(units[name])
                r = subprocess.run([nvcc, *unit_command(name, arch, defines, unit_arch, include), "-c", "-o",
                                    str(work / (name + ".o")), str(work / name)], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, universal_newlines=True)
                if r.returncode:
                    raise RuntimeError(f"nvcc failed on {name}:\n{r.stdout[-4000:]}")
            with ThreadPoolExecutor(max_workers=max(1, min(len(units), os.cpu_count() or 1))) as ex:
                list(ex.map(compile_unit, sorted(units)))
            r = subprocess.run([nvcc, "-shared", f"-arch={arch}", "-o", str(work / "lib.so"),
                                *(str(work / (n + ".o")) for n in sorted(units))], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)
            if r.returncode:
                raise RuntimeError(f"nvcc link failed:\n{r.stdout[-4000:]}")
            os.replace(work / "lib.so", out)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        STATS["builds"] += 1
        STATS["build_seconds"] += time.time() - t0
        return out, True, key


def init(model_dir=None, model=None, foot_pairs=None, com_body_id=None):
    """Load the library for the model (build it on first use), hand it the model constants and register the FFI
    targets; idempotent (later calls return the loaded library; a different model raises). model_dir: a fixtures
    directory (default $DK_FIXTURES, else fixtures/mjlab); model, foot_pairs: an MjModel instead (e.g. the same mjlab
    model when the fixtures are not installed) and its foot-floor pair indices. A model the engine does not support
    raises nablaengine.model.capability.UnsupportedModel (a ValueError and an AssertionError) before anything is
    built, with the failing checks' codes and values (the input validation capability.check, engine scope: forward
    and adjoint); $DK_ALLOW_UNSUPPORTED lets named codes through with a warning (experiments only;
    docs/configuration.md). Returns a dict
    (path, key, arch, rec_words, sizes per variant, built, arch_flags {forward, adjoint}, forward_defines: the -D options
    the forward unit gets for the target, e.g. sm_90's register caps, [] elsewhere; forward_unit: the prefix of the
    library's forward / kinematics entries, <forward_unit>_forward / _kinematics / _kinematics_vjp: dk_fwd where that
    unit has flags of its own, else FWD_VARIANT's unit dk_kv0_n3)."""
    import jax
    with _LOCK:
        if _STATE and model_dir is None and model is None:  # every traced call comes here: no model reload
            return _STATE["info"]
        m, fp = _model(model_dir, model, foot_pairs)
        # the model must be one the engine supports (nablaengine.model.capability), checked before anything is built
        capability.require(m, fp, capability.COM_BODY if com_body_id is None else com_body_id, "engine", "jax_ffi.init")
        arch = _arch()
        plan = build_plan(m, fp, arch, com_body_id=com_body_id)
        units, consts, defines, unit_arch = plan["units"], plan["consts"], plan["defines"], plan["unit_arch"]
        fwd_flags, adj_flags, split = plan["arch_flags"]["forward"], plan["arch_flags"]["adjoint"], plan["split"]
        fwd_defines = plan["forward_defines"]
        chash = hashlib.sha256(consts.tobytes()).hexdigest()
        if _STATE:
            if _STATE["consts_hash"] != chash:
                raise RuntimeError("nablaengine.jax_ffi: already initialised with another model (one per process)")
            return _STATE["info"]
        path, built, key = _build(units, defines, arch, jax.ffi.include_dir(), consts, unit_arch or None)
        lib = ctypes.cdll.LoadLibrary(str(path))
        STATS["loads"] += 1
        lib.dk_set_constants.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        c = np.ascontiguousarray(consts, np.float32)
        rc = lib.dk_set_constants(c.ctypes.data, len(c))
        if rc:
            raise RuntimeError(f"nablaengine.jax_ffi: dk_set_constants failed ({rc})")
        sz = np.zeros(9 * len(VARIANTS), np.int64)
        lib.dk_sizes(ctypes.c_void_p(sz.ctypes.data))
        sz = sz.reshape(len(VARIANTS), 9)
        rec = g1_tape.rec_words(m)
        for v, row in zip(VARIANTS, sz):
            assert tuple(row[2:9]) == (rec, m.nq, m.nv, g1_step.maxcon(m), v[0], v[1], g1_tape.layout(m)["flags"][0]), \
                (v, row)
        lib.dk_launch_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
        lib.dk_last_slots.argtypes = [ctypes.c_void_p]
        targets = {}
        for kind, sym in (("step", "DkStep"), ("fwd", "DkStepFwd"), ("bwd", "DkStepBwd"), ("bwd_soft", "DkStepBwdSoft"),
                          ("diag", "DkDiag"), ("last", "DkLast"), ("step_kin", "DkStepKin"), ("fwd_kin", "DkStepFwdKin"),
                          ("bwd_kin", "DkStepBwdKin"), ("bwd_soft_kin", "DkStepBwdSoftKin"), ("kin", "DkKin"),
                          ("kin_bwd", "DkKinBwd")):
            targets[kind] = f"dk_{kind}_{key[:8]}"
            jax.ffi.register_ffi_target(targets[kind], jax.ffi.pycapsule(getattr(lib, sym)), platform="CUDA")
        com_ids = (int(m.body("torso_link").id),) if com_body_id is None else tuple(int(x) for x in np.atleast_1d(com_body_id))
        nr = len(g1_source.free_roots(m))
        torso = com_ids[0]
        try:  # the kinematics outputs' assumptions (kinematics.check_model); the kinematics entries raise without them
            g1_kin.check_model(m)
            kin_error = None
        except AssertionError as e:
            kin_error = f"the model does not meet the kinematics outputs' assumptions ({e!r})"
        # also version (nablaengine.__version__) and commit / dirty (the source checkout's revision, None outside one)
        commit, dirty = provenance.source_commit()
        info = dict(path=str(path), key=key, arch=arch, built=built, version=nablaengine.__version__, commit=commit,
                    dirty=dirty, rec_words=rec, nq=m.nq, nv=m.nv, nu=m.nu,
                    nbody=m.nbody, nr=nr, com_width=3*nr, com_body_ids=com_ids, kinematics_error=kin_error,
                    maxcon=g1_step.maxcon(m), nefc=g1_step.nefc(m),
                    sizeof_sim=int(sz[0, 0]), sizeof_adj={_tag(v): int(r[1]) for v, r in zip(VARIANTS, sz)},
                    variants=[list(v) for v in VARIANTS], timestep=float(m.opt.timestep), torso_id=int(torso),
                    foot_pairs=[int(p) for p in fp], consts_sha256=chash,
                    arch_flags=dict(forward=fwd_flags, adjoint=adj_flags), forward_defines=fwd_defines,
                    forward_unit=FWD_UNIT if split else _tag(FWD_VARIANT))
        if unit_arch or arch in cuda.N7_VIRTUAL:  # the front ends (PTX headers) of the forward and the adjoint units
            info["front_end"] = {k: cuda.nvvm_version(cuda.virtual_arch(u, arch) or "compute_" + arch[3:])
                                 for k, u in (("forward", "forward"), ("adjoint", None))}
        _STATE.update(lib=lib, targets=targets, info=info, consts_hash=chash, consts=consts, m=m)
        import atexit
        atexit.register(_exit_report)  # with set_overflow_warnings(True): the counts nobody read, reported at exit
        return info


def rec_words():
    """Words (float32) per tape record (one world, one substep): 1472 for the mjlab model."""
    return init()["rec_words"]


def tape_bytes(nworld, substeps=SUBSTEPS, steps=1):
    """Bytes of the tape residuals of `steps` control steps (a differentiated rollout of H steps keeps H of them)."""
    return 4 * rec_words() * substeps * nworld * steps


def _check(name, x, shape):
    if x.dtype != np.float32 or tuple(x.shape) != tuple(shape):
        raise TypeError(f"nablaengine.jax_ffi: {name} must be float32 {tuple(shape)}, got {x.dtype} {tuple(x.shape)}")


def _kin_lib():
    """init(), raising ValueError if the model does not meet the kinematics outputs' assumptions."""
    lib = init()
    if lib["kinematics_error"]:
        raise ValueError(f"nablaengine.jax_ffi: {lib['kinematics_error']}")
    return lib


def _kin_structs(lib, B):
    """The five kinematics outputs' ShapeDtypeStructs (g1_engine.kinematics.OUTPUTS order)."""
    import jax
    import jax.numpy as jnp
    return [jax.ShapeDtypeStruct(s, jnp.float32) for s in g1_kin.shapes(_STATE["m"], B).values()]


def _check_kin(lib, B, arrays, what):
    for name, x, shape in zip(g1_kin.OUTPUTS, arrays, g1_kin.shapes(_STATE["m"], B).values()):
        _check(f"{what}{name}", x, shape)


def forward(qpos, qvel, warm, ctrl, friction, com, *, iterations=ITERATIONS, substeps=SUBSTEPS, tape=False,
            kinematics=False):
    """One control step without custom_vjp: (qpos', qvel', warm', info (B, 4) int32[, tape (B, substeps * REC_WORDS)
    float32][, qpos_n, xpos, xquat, cvel, subtree_com]); dk_step, or dk_step_fwd with tape=True; kinematics=True:
    the output state's kinematics (kinematics_forward(qpos', qvel', com), bitwise) from the same call
    (dk_step_kin / dk_fwd_kin)."""
    import jax
    import jax.numpy as jnp
    lib = _kin_lib() if kinematics else init()
    B = qpos.shape[0]
    nq, nv, nu = lib["nq"], lib["nv"], lib["nu"]
    for name, x, shape in (("qpos", qpos, (B, nq)), ("qvel", qvel, (B, nv)), ("warm", warm, (B, nv)),
                           ("ctrl", ctrl, (B, nu)), ("friction", friction, (B,)), ("com", com, (B, lib["com_width"]))):
        _check(name, x, shape)
    f32 = jnp.float32
    outs = [jax.ShapeDtypeStruct((B, nq), f32), jax.ShapeDtypeStruct((B, nv), f32), jax.ShapeDtypeStruct((B, nv), f32),
            jax.ShapeDtypeStruct((B, 4), jnp.int32), jax.ShapeDtypeStruct((B, lib["maxcon"] * 8), f32)]  # + contacts
    if tape:
        outs.append(jax.ShapeDtypeStruct((B, substeps * lib["rec_words"]), f32))
    if kinematics:
        outs += _kin_structs(lib, B)
    call = jax.ffi.ffi_call(_STATE["targets"][("fwd" if tape else "step") + ("_kin" if kinematics else "")], outs)
    res = call(qpos, qvel, warm, ctrl, friction, com, iterations=np.int32(iterations), substeps=np.int32(substeps))
    return tuple(res[:4]) + tuple(res[5:])  # the contact output (k_substep's last-substep contacts) is dropped


def kinematics_forward(qpos, qvel, com):
    """The kinematics outputs of the states (qpos (B, nq), qvel (B, nv), com (B, 3*nr)) without custom_vjp (dk_kin):
    (qpos_n (B, nq), xpos (B, nbody, 3), xquat (B, nbody, 4), cvel (B, nbody, 6), subtree_com (B, nbody, 3)), MJX's
    mjx.kinematics + com_pos + com_vel (g1_engine.kinematics)."""
    import jax
    lib = _kin_lib()
    B = qpos.shape[0]
    for name, x, shape in (("qpos", qpos, (B, lib["nq"])), ("qvel", qvel, (B, lib["nv"])), ("com", com, (B, lib["com_width"]))):
        _check(name, x, shape)
    return tuple(jax.ffi.ffi_call(_STATE["targets"]["kin"], _kin_structs(lib, B))(qpos, qvel, com))


def kinematics_backward(qpos, qvel, com, g_qpos_n, g_xpos, g_xquat, g_cvel, g_subtree_com):
    """The VJP of kinematics_forward at (qpos, qvel, com) for the cotangents of its five outputs: (g_qpos (B, nq),
    g_qvel (B, nv)) (dk_kin_bwd; com has none)."""
    import jax
    import jax.numpy as jnp
    lib = _kin_lib()
    B = qpos.shape[0]
    nq, nv = lib["nq"], lib["nv"]
    for name, x, shape in (("qpos", qpos, (B, nq)), ("qvel", qvel, (B, nv)), ("com", com, (B, lib["com_width"]))):
        _check(name, x, shape)
    gk = (g_qpos_n, g_xpos, g_xquat, g_cvel, g_subtree_com)
    _check_kin(lib, B, gk, "g_")
    outs = [jax.ShapeDtypeStruct((B, nq), jnp.float32), jax.ShapeDtypeStruct((B, nv), jnp.float32)]
    return tuple(jax.ffi.ffi_call(_STATE["targets"]["kin_bwd"], outs)(qpos, qvel, com, *gk))


@functools.lru_cache(maxsize=1)
def _kinematics_fn():
    import jax
    import jax.numpy as jnp
    from jax.ad_checkpoint import checkpoint_name

    @jax.custom_vjp
    def kin(qpos, qvel, com):
        return kinematics_forward(qpos, qvel, com)

    def kin_fwd(qpos, qvel, com):
        # named for remat_policy (KIN_NAME): the outputs (the caller's VJP needs them) and the state the backward
        # recomputes the kinematics from; identities without remat
        out = tuple(checkpoint_name(x, KIN_NAME) for x in kinematics_forward(qpos, qvel, com))
        return out, (checkpoint_name(qpos, KIN_NAME), checkpoint_name(qvel, KIN_NAME), com)

    def kin_bwd(res, cot):
        qpos, qvel, com = res
        g_q, g_v = kinematics_backward(qpos, qvel, com, *cot)
        return g_q, g_v, jnp.zeros_like(com)

    kin.defvjp(kin_fwd, kin_bwd)
    return kin


def kinematics(qpos, qvel, com):
    """Differentiable kinematics outputs of states not produced by the engine (e.g. initial or externally supplied
    states): (qpos_n, xpos, xquat, cvel, subtree_com), as MJX's mjx.kinematics + smooth.com_pos + smooth.com_vel on
    the model with the torso COM offset com added to the torso's body_ipos (as step's); a jax.custom_vjp whose backward
    (dk_kin_bwd) returns the cotangents of qpos (through the quaternion's normalization: its radial part removed) and
    qvel, zeros for com. The same kernels as step(..., kinematics=True): kinematics(q', v', com) of a step's output is
    bitwise its kinematics outputs. Any B, float32; see the module docstring."""
    return _kinematics_fn()(qpos, qvel, com)


def _soft(clamp_kv, clamp_soft):
    """The soft clamp's delta as a float32 value for clamp_kv 2 (None: CLAMP_SOFT), None for 0 / 1 (where it must be
    None); ValueError for an unknown variant or a delta that is negative or not finite."""
    if int(clamp_kv) != 2:
        if clamp_soft is not None:
            raise ValueError(f"nablaengine.jax_ffi: clamp_soft is the soft clamp's (clamp_kv=2), not clamp_kv={clamp_kv}")
        return None
    d = float(np.float32(CLAMP_SOFT if clamp_soft is None else clamp_soft))
    if not (np.isfinite(d) and d >= 0.0):
        raise ValueError(f"nablaengine.jax_ffi: clamp_soft must be finite and >= 0, got {clamp_soft}")
    return d


def backward(tape, friction, com, g_qpos, g_qvel, *, substeps=SUBSTEPS, clamp_kv=CLAMP_KV, normal=NORMAL, ctrl=None,
             clamp_soft=None, kinematics=None):
    """The adjoint of one control step from its tape and the cotangents of (qpos', qvel'): (g_qpos (B, nq), g_qvel (B,
    nv), g_ctrl (B, nu), info (B,) int32: the adjoint's flags | NONFINITE_* << NONFINITE_SHIFT). clamp_kv 2 (the soft
    clamp, delta clamp_soft: None for CLAMP_SOFT) needs the step's ctrl (B, nu); 0 and 1 ignore ctrl. kinematics:
    None, or (qpos', qvel', g_qpos_n, g_xpos, g_xquat, g_cvel, g_subtree_com) of a step with kinematics outputs: their
    cotangents are pulled back onto (qpos', qvel') (kinematics_backward) and added to g_qpos, g_qvel in the same call,
    before the adjoint (dk_bwd_kin / dk_bwd_soft_kin); the non-finite cotangent bit then refers to those sums."""
    import jax
    import jax.numpy as jnp
    lib = _kin_lib() if kinematics is not None else init()
    B = tape.shape[0]
    nq, nv, nu = lib["nq"], lib["nv"], lib["nu"]
    for name, x, shape in (("tape", tape, (B, substeps * lib["rec_words"])), ("friction", friction, (B,)),
                           ("com", com, (B, lib["com_width"])), ("g_qpos", g_qpos, (B, nq)), ("g_qvel", g_qvel, (B, nv))):
        _check(name, x, shape)
    if (int(clamp_kv), int(normal)) not in VARIANTS:
        raise ValueError(f"nablaengine.jax_ffi: no adjoint variant clamp_kv={clamp_kv} normal={normal}")
    soft = _soft(clamp_kv, clamp_soft)
    kin, sfx = (), ""
    if kinematics is not None:
        kin, sfx = tuple(kinematics), "_kin"
        if len(kin) != 7:
            raise ValueError("nablaengine.jax_ffi: kinematics = (qpos', qvel', g_qpos_n, g_xpos, g_xquat, g_cvel, "
                             "g_subtree_com)")
        _check("qpos'", kin[0], (B, nq))
        _check("qvel'", kin[1], (B, nv))
        _check_kin(lib, B, kin[2:], "g_")
        kin = (kin[0], kin[1], g_qpos, g_qvel) + kin[2:]
    else:
        kin = (g_qpos, g_qvel)
    f32 = jnp.float32
    outs = [jax.ShapeDtypeStruct((B, nq), f32), jax.ShapeDtypeStruct((B, nv), f32), jax.ShapeDtypeStruct((B, nu), f32),
            jax.ShapeDtypeStruct((B,), jnp.int32)]
    if soft is not None:
        if ctrl is None:
            raise ValueError("nablaengine.jax_ffi: clamp_kv=2 (the soft clamp) needs the step's ctrl")
        _check("ctrl", ctrl, (B, nu))
        call = jax.ffi.ffi_call(_STATE["targets"]["bwd_soft" + sfx], outs)
        return tuple(call(tape, friction, com, ctrl, *kin, substeps=np.int32(substeps), normal=np.int32(normal),
                          clamp_soft=np.float32(soft)))
    call = jax.ffi.ffi_call(_STATE["targets"]["bwd" + sfx], outs)
    return tuple(call(tape, friction, com, *kin, substeps=np.int32(substeps), clamp_kv=np.int32(clamp_kv),
                      normal=np.int32(normal)))


def make_step(iterations=ITERATIONS, substeps=SUBSTEPS, clamp_kv=CLAMP_KV, normal=NORMAL, clamp_soft=None,
              grad_decay=GRAD_DECAY, kinematics=False):
    """step(qpos, qvel, warm, ctrl, friction, com) -> (qpos', qvel', warm') with these static options, a
    jax.custom_vjp function (one per option set, so jit caches stay valid: the options are normalized first, so every
    spelling of a set gives the same function). kinematics=True: -> (qpos', qvel', warm', qpos_n, xpos, xquat,
    cvel, subtree_com), the output state's kinematics in the same call and the same VJP."""
    if (int(clamp_kv), int(normal)) not in VARIANTS:
        raise ValueError(f"nablaengine.jax_ffi: no adjoint variant clamp_kv={clamp_kv} normal={normal}")
    gd = float(np.float32(grad_decay))
    if not np.isfinite(gd):
        raise ValueError(f"nablaengine.jax_ffi: grad_decay must be finite, got {grad_decay}")
    return _make_step(int(iterations), int(substeps), int(clamp_kv), int(normal), _soft(clamp_kv, clamp_soft), gd,
                      bool(kinematics))


@functools.lru_cache(maxsize=None)
def _make_step(iterations, substeps, clamp_kv, normal, clamp_soft, grad_decay, kin=False):
    import jax
    import jax.numpy as jnp
    soft = clamp_soft is not None  # clamp_kv 2: ctrl is a residual too
    gd = np.float32(grad_decay)

    @jax.custom_vjp
    def step_fn(qpos, qvel, warm, ctrl, friction, com):
        r = forward(qpos, qvel, warm, ctrl, friction, com, iterations=iterations, substeps=substeps, kinematics=kin)
        return tuple(r[:3]) + tuple(r[4:])  # (info dropped) the kinematics outputs with kin

    def step_fwd(qpos, qvel, warm, ctrl, friction, com):
        from jax.ad_checkpoint import checkpoint_name
        r = forward(qpos, qvel, warm, ctrl, friction, com, iterations=iterations, substeps=substeps, tape=True,
                    kinematics=kin)
        # named for jax.checkpoint policies (remat_policy): saving the tape alone still reruns dk_step_fwd in the
        # backward wherever the body's own VJP needs the step's outputs (a loss or kinematics of qpos'), so they are
        # named too, the kinematics outputs with them (a body's cost or controller may read them); identities without
        # remat
        outs = tuple(checkpoint_name(x, OUT_NAME) for x in tuple(r[:3]) + tuple(r[5:]))
        res = (checkpoint_name(r[4], TAPE_NAME), friction, com) + (outs[:2] if kin else ()) + ((ctrl,) if soft else ())
        return outs, res

    def step_bwd(res, cot):
        tp, friction, com = res[:3]
        g_q, g_v = cot[:2]  # the warm start's cotangent (cot[2]) is ignored
        kc = tuple(res[3:5]) + tuple(cot[3:]) if kin else None  # (qpos', qvel', the kinematics outputs' cotangents)
        dq, dv, du, _ = backward(tp, friction, com, g_q, g_v, substeps=substeps, clamp_kv=clamp_kv, normal=normal,
                                 ctrl=res[-1] if soft else None, clamp_soft=clamp_soft, kinematics=kc)
        if grad_decay != 1.0:  # gradient decay: the state cotangent through this step's physics times it, ctrl's not
            dq, dv = dq * gd, dv * gd
        return dq, dv, jnp.zeros_like(dv), du, jnp.zeros_like(friction), jnp.zeros_like(com)

    step_fn.defvjp(step_fwd, step_bwd)
    return step_fn


def step(qpos, qvel, warm, ctrl, friction, com, *, iterations=ITERATIONS, substeps=SUBSTEPS, clamp_kv=CLAMP_KV,
         normal=NORMAL, clamp_soft=None, grad_decay=GRAD_DECAY, kinematics=False):
    """One control step, differentiable (see the module docstring): (qpos', qvel', warm'); kinematics=True:
    (qpos', qvel', warm', qpos_n, xpos, xquat, cvel, subtree_com), the kinematics of the output state (qpos', qvel')
    computed in the same FFI call after the last substep, their cotangents in the same VJP."""
    return make_step(iterations, substeps, clamp_kv, normal, clamp_soft, grad_decay, kinematics)(
        qpos, qvel, warm, ctrl, friction, com)


def scale_grad(tree, gamma):
    """The identity on a pytree whose cotangent the backward multiplies by gamma (a Python float, static; the multiply
    in each leaf's dtype): the gradient decay of the module docstring, for any differentiable function. Only the leaves
    with an inexact dtype (float, complex) go through the custom_vjp; the others (ints, bools, PRNG keys, Python
    scalars and other non-array leaves) are returned untouched, so it can wrap a whole carried state (the full-state
    form of the module docstring: at the top of the scan body, every derivative across n step boundaries gets
    gamma^n). On the state arguments of a physics call
    alone (qpos, qvel: the physics-only form) it gives the gradients of make_step(grad_decay=gamma) bitwise
    (scripts/jax_ffi_check.py, section options). gamma = 1.0 returns tree itself."""
    g = float(gamma)
    if not np.isfinite(g):
        raise ValueError(f"nablaengine.jax_ffi: scale_grad's gamma must be finite, got {gamma}")
    if g == 1.0:
        return tree
    import jax
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    idx = [i for i, x in enumerate(leaves)
           if hasattr(x, "dtype") and jax.dtypes.issubdtype(x.dtype, np.inexact)]
    if not idx:
        return tree
    scaled = _scale_grad_fn(g)([leaves[i] for i in idx])
    for i, x in zip(idx, scaled):
        leaves[i] = x
    return jax.tree_util.tree_unflatten(treedef, leaves)


@functools.lru_cache(maxsize=None)
def _scale_grad_fn(gamma):
    import jax

    @jax.custom_vjp
    def ident(tree):
        return tree

    def fwd(tree):
        return tree, None

    def bwd(_, cot):
        return (jax.tree_util.tree_map(lambda c: c * np.asarray(gamma, c.dtype), cot),)

    ident.defvjp(fwd, bwd)
    return ident


def remat_policy(*extra_names):
    """The jax.checkpoint policy for a BPTT body that calls step: save the tape and the step's outputs (REMAT_NAMES,
    plus extra_names of the caller's own checkpoint_name values), recompute the rest. A plain jax.checkpoint(body)
    saves nothing, so the backward reruns dk_step_fwd for the tape (2 forward calls per control step); with this policy
    each step runs dk_step_fwd once; the gradients equal those without remat up to XLA's rounding of the recomputed body
    (~1e-7 relative; bitwise for jax_ffi_check's rollout body). step(..., kinematics=True)'s kinematics outputs are
    named OUT_NAME with the step's outputs, jax_ffi.kinematics's outputs and residual state KIN_NAME ("dk_kin"): saved
    too, so neither kernel reruns in the backward (step: B x 532 float32 per call, 8.7 MB at B = 4096; kinematics
    called inside the body: B x 603, its residual qpos, qvel included, 9.9 MB)."""
    import jax
    return jax.checkpoint_policies.save_only_these_names(*REMAT_NAMES, *extra_names)


def launch_info(device=0):
    """{variant tag: {smem, worlds_per_sm}}: k_adjoint's launch on the CUDA device ordinal (adjoint.adj_launch_smem,
    set at the device's first call; zeros before)."""
    init()
    o = np.zeros(2 * len(VARIANTS), np.int64)
    rc = _STATE["lib"].dk_launch_info(int(device), ctypes.c_void_p(o.ctypes.data))
    if rc:
        raise RuntimeError(f"nablaengine.jax_ffi: dk_launch_info failed ({rc})")
    return {_tag(v): dict(smem=int(o[2 * i]), worlds_per_sm=int(o[2 * i + 1])) for i, v in enumerate(VARIANTS)}


def diagnostics(*deps, reset=True):
    """The counters (NDIAG, 2) uint32 (low, high words; decode_diagnostics), read on the GPU after the arrays deps are
    computed (inside jit: pass what must run first, e.g. the gradients); reset=True zeroes them."""
    import jax
    import jax.numpy as jnp
    init()
    call = jax.ffi.ffi_call(_STATE["targets"]["diag"], jax.ShapeDtypeStruct((len(DIAG), 2), jnp.uint32),
                            has_side_effect=True)
    return call(*deps, reset=np.int32(bool(reset)))


def decode_diagnostics(arr):
    """(NDIAG, 2) uint32 -> {name: int}; with set_overflow_warnings(True) it warns (warn_overflow) when worlds
    exceeded the contact or row capacity."""
    a = np.asarray(arr).astype(np.uint64)
    d = {name: int(a[i, 0] | (a[i, 1] << np.uint64(32))) for i, name in enumerate(DIAG)}
    if _OVERFLOW_WARN[0]:
        warn_overflow(d)
    return d


def warn_overflow(d, where="nablaengine"):
    """A RuntimeWarning when the counters d count worlds whose contacts or constraint rows exceeded the per-world
    compiled capacity: the ones beyond were dropped, so those worlds miss support and
    friction forces. The worlds of a call: info[:, 3] & OVERFLOW_CONTACTS / OVERFLOW_ROWS of forward()."""
    import warnings
    nc, nr = d.get("fwd_contact_overflow_worlds", 0), d.get("fwd_row_overflow_worlds", 0)
    if nc or nr:
        warnings.warn(f"{where}: {nc} world-steps exceeded the compiled contact capacity and {nr} the constraint-"
                      f"row capacity; their extra contacts / rows were dropped, so those worlds miss support and "
                      f"friction forces (e.g. deep interpenetration). See docs/api.md, Diagnostics.", RuntimeWarning,
                      stacklevel=3)
    return nc + nr


def set_overflow_warnings(enabled):
    """Switch the capacity-overflow warnings on or off (default off, or $DK_OVERFLOW_WARN=1): when on,
    read_diagnostics / decode_diagnostics warn (warn_overflow) if worlds exceeded the per-world contact or
    constraint-row capacity, and the counts nobody read are reported once more at exit. The counters and info's bits
    are kept either way. Returns the previous setting."""
    prev = _OVERFLOW_WARN[0]
    _OVERFLOW_WARN[0] = bool(enabled)
    return prev


def _exit_report():
    """At exit: the overflow warning for the counts since the last read (no reset), if any; never raises."""
    try:
        if _OVERFLOW_WARN[0] and _STATE and STATS.get("loads"):
            read_diagnostics(reset=False)
    except Exception:  # noqa: BLE001  (interpreter or backend shutting down)
        pass


@functools.lru_cache(maxsize=2)
def _diag_fn(reset):
    import jax
    return jax.jit(lambda: diagnostics(reset=reset))


def read_diagnostics(reset=True):
    """{name: int} of the counters since the last reset (host sync; call outside jit, after the work to count)."""
    init()
    return decode_diagnostics(_diag_fn(bool(reset))())


@functools.lru_cache(maxsize=1)
def _last_fn():
    import jax
    import jax.numpy as jnp
    return jax.jit(lambda: jax.ffi.ffi_call(_STATE["targets"]["last"], jax.ShapeDtypeStruct((NSLOT, len(LAST)),
                                                                                             jnp.uint32),
                                            has_side_effect=True)())


def last_forward(nworld=None):
    """{LAST name: int} of the last forward call (dk_step or dk_step_fwd) on the device with batch size nworld (None:
    of any batch size); zeros before the first such call. A read on the GPU's stream after the work already
    dispatched (host sync; call outside jit), without reset."""
    init()
    slots = np.zeros(NSLOT, np.int64)
    _STATE["lib"].dk_last_slots(ctypes.c_void_p(slots.ctypes.data))
    rec = np.asarray(_last_fn()())
    if nworld is None:
        s = 0
    else:
        hit = np.flatnonzero(slots[1:] == int(nworld))
        if not len(hit):  # never called with this batch size (or more batch sizes than slots)
            return {name: 0 for name in LAST}
        s = int(hit[0]) + 1
    return {name: int(rec[s, i]) for i, name in enumerate(LAST)}


def check_model(mjm, foot_pairs=None, torso_id=None):
    """Raise ValueError unless the MjModel mjm (with its foot-floor pair indices and torso body id) is a model the
    engine supports (nablaengine.model.capability.require, engine scope: UnsupportedModel, a ValueError,
    lists the failing checks; the packed constants below cannot see physics the engine does not model, such as joint
    damping or the integrator) and the model whose constants the library holds (init()): nq / nv / nu, the timestep,
    the foot pairs (in order), the torso id and the packed physics constants (adjoint.engine_constants, compared
    bitwise; the solver iterations are an option of the call, not a constant). Returns the library info."""
    info = init()
    fp = info["foot_pairs"] if foot_pairs is None else [int(p) for p in np.asarray(foot_pairs).ravel()]
    capability.require(mjm, fp, capability.COM_BODY, "engine", "jax_ffi.check_model")
    msg = []
    if (mjm.nq, mjm.nv, mjm.nu) != (info["nq"], info["nv"], info["nu"]):
        msg.append(f"nq / nv / nu {(mjm.nq, mjm.nv, mjm.nu)} != {(info['nq'], info['nv'], info['nu'])}")
    if float(mjm.opt.timestep) != info["timestep"]:
        msg.append(f"timestep {mjm.opt.timestep} != {info['timestep']}")
    if fp != info["foot_pairs"]:
        msg.append(f"foot pairs {fp} != {info['foot_pairs']}")
    if torso_id is not None and int(torso_id) != info["torso_id"]:
        msg.append(f"torso id {int(torso_id)} != {info['torso_id']}")
    if not msg:
        try:
            consts, lays, sizes = adjoint.engine_constants(mjm, fp)
        except AssertionError as e:  # a model the engine's constant packing does not support
            raise ValueError(f"nablaengine.jax_ffi: the model is not the engine's mjlab G1 ({e!r})") from None
        mine = _STATE["consts"]
        if len(consts) != len(mine):
            msg.append(f"{len(consts)} packed constants != {len(mine)}")
        elif not np.array_equal(consts.view(np.uint32), np.asarray(mine, np.float32).view(np.uint32)):
            names, at = [], 0  # the constant arrays that differ, in packing order
            for sym, kind, count, width in lays:  # host.loader's packing ("u": two 16-bit halves per value)
                n = (sizes[count] if isinstance(count, str) else int(count)) * width * ({"u": 2, "q": 4}.get(kind, 1))
                if not np.array_equal(consts[at:at + n].view(np.uint32), np.asarray(mine[at:at + n]).view(np.uint32)):
                    names.append(sym)
                at += n
            msg.append(f"packed constants differ ({', '.join(names) or 'layout'})")
    if msg:
        raise ValueError("nablaengine.jax_ffi: the model differs from the library's (the fixtures' unless "
                         "init(model=...)): " + "; ".join(msg) + ". Use the same mjlab model, or EnginePhysics(..., "
                         "use_model=True) / init(model=, foot_pairs=) to build the library for this one")
    return info


class EnginePhysics:
    """A batched physics object for a model: EnginePhysics(mjm, nworld, foot_pairs, torso_id, ...) with .step(qpos,
    qvel, warm, ctrl, friction, com), .stats() (the last forward call's high-water marks: nacon, naconmax, nefc_max,
    njmax), .nconmax / .njmax (the engine's per-world contact and constraint-row capacities) and .diagnostics() (the
    cumulative counters). nworld is the batch size stats() reports (the library serves any B); iterations default to
    mjm.opt.iterations; clamp_kv, normal, clamp_soft, grad_decay and kinematics as make_step's (grad_decay: the
    physics-only form of the gradient decay, see the module docstring; kinematics=True: .step also returns the output
    state's qpos_n, xpos, xquat, cvel, subtree_com). The model constants come from the fixtures (init()), and mjm (with
    foot_pairs and torso_id) must then be the same model (check_model, else ValueError), unless use_model=True: then
    from mjm. Either way a model the engine does not support raises nablaengine.model.capability.UnsupportedModel (a
    ValueError) before anything runs. Other keyword options (those of a MuJoCo Warp based physics object, such as
    graph modes, capacities or a warm-started recompute) are accepted and ignored: the engine has none of them, and
    its tape replaces the recompute."""

    def __init__(self, mjm=None, nworld=None, foot_pairs=None, torso_id=None, *, iterations=None, substeps=SUBSTEPS,
                 clamp_kv=CLAMP_KV, normal=NORMAL, clamp_soft=None, grad_decay=GRAD_DECAY, kinematics=False,
                 use_model=False, **_ignored):
        if use_model:
            self.info = init(model=mjm, foot_pairs=foot_pairs)
            if torso_id is not None:
                check_model(mjm, foot_pairs, torso_id)
        else:
            self.info = init()
            if mjm is not None:
                check_model(mjm, foot_pairs, torso_id)
        self.nworld = None if nworld is None else int(nworld)
        self.nconmax, self.njmax = int(self.info["maxcon"]), int(self.info["nefc"])
        self.iterations = int(iterations or (mjm.opt.iterations if mjm is not None else ITERATIONS))
        self.options = dict(iterations=self.iterations, substeps=int(substeps), clamp_kv=int(clamp_kv),
                            normal=int(normal), clamp_soft=_soft(clamp_kv, clamp_soft),
                            grad_decay=float(np.float32(grad_decay)), kinematics=bool(kinematics))
        self.step = make_step(self.iterations, int(substeps), int(clamp_kv), int(normal), clamp_soft, grad_decay,
                              kinematics)

    def stats(self):
        """High-water marks of the last forward call with this batch size (host sync; call outside jit), with MuJoCo
        Warp's names: nacon (contacts of the last substep, summed over the worlds), naconmax (nconmax x worlds),
        nefc_max (constraint rows of the most loaded world), njmax. A read: nothing is reset."""
        r = last_forward(self.nworld)
        nw = self.nworld if self.nworld is not None else r["worlds"]
        return {"nacon": r["contacts"], "naconmax": self.nconmax * nw, "nefc_max": r["rows_max"], "njmax": self.njmax}

    def diagnostics(self, reset=True):
        """The cumulative diagnostic counters since the last reset (read_diagnostics; host sync, call outside jit)."""
        return read_diagnostics(reset)
