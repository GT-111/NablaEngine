"""CUDA source of the G1 engine's analytic adjoint: the VJP of a control step from its forward tape.

One warp per world, state in shared memory (struct Adj), one launch per control step: k_adjoint reads the forward tape
(nablaengine.g1_engine.tape) and runs the VJPs of the substeps in reverse order (adj_substep, k = nsub - 1 .. 0), with
the cotangents of the control step's output state as input and the gradients of its input qpos, qvel and ctrl (the sum
over the substeps) as output; the warm start, friction and COM offset get none. Per substep, the phases A0-A8 (the
labels of PROF_DESIGN and of the code's comments):

  A0  the record: input state q, v, final Newton acceleration a*, a_adv = Q^-1 M a*, clamp mask, header
  A1  integrator VJP (the quaternion in MuJoCo Warp's smooth form, also where the forward skipped the integration at
      w' = 0) and implicitfast: y = Q^-1 a_adv bar with L_Q, a* bar = M y = a_adv bar - h kv y (Q y = a_adv bar; the
      product form L_Q (L_Q' y) - h kv y loses ~cond(Q) eps in float32)
  A2  IFT: lambda = H^-1 a* bar with L_H
  A3  root-to-leaf sweeps, lane b = body b, one tree level at a time: poses and the RNE's channel 1 (twist of v,
      accelerations at a* with gravity) by the forward's fwd_body in adj_poses (explicit roundings, source.DEVICE: the
      poses, and the contact replay from them, are bitwise the forward's), then hinge axes, channel 2
      (accelerations at a~ = a* - a_adv, v = 0, no gravity) and the twists of lambda and -y
  A4  each body's local RNE VJP: cotangents of its twists, accelerations and own pose (f bar, tau bar)
  A5  A6  the constraint rows (DK_ADJ_STAGE >= 2), between A4 and A7 (they add per-body pose cotangents f bar, tau bar
      and per-dof P_q, P_v; the device functions are nablaengine.g1_engine.adjoint_contact):
      adj_limits  one lane per dof: the hinge-limit rows replayed with the forward's test, the tape's zone bit,
                  kbimp_d / efc_D_d / limit_row_vjp -> P_q, P_v of the dof
      adj_contacts  the collision geoms' world poses (geom_poses of the forward, bitwise), then the tape's contacts
                  in batches of 32, one lane per contact: replay (adj_pair_geom: the forward's dist, pos, frame
                  bitwise), impedance with the forward's expressions (mu: the world's foot friction on foot pairs),
                  the contact point's motions V_* = sum over the support dofs of s_i col_i (v, a*, lam)_i (the dofs on
                  exactly one of the two bodies' chains, as the forward's Jacobian) and their angular parts,
                  contact_psi_vjp, p bar in closed form (contact_pbar_motion), the primitive's VJP and
                  geom_to_body_vjp, into the contact lane's registers; then one lane per body sums the batch in
                  contact order (shuffles from the contacts' lanes; deterministic, no atomics): the support-dof terms
                  of its own dof (G_i -> f bar, tau bar of the body, P_v of the dof; lanes 0 and 31 take two of the
                  root's rotation dofs and hand them to lane 1) and the geometry's pose cotangents of the body
  A7  leaf-to-root sweep: the RNE reverse of both channels and both twists and the pose sweep in one pass (a parent
      pulls its children's finished cotangents: deterministic) -> P_q (dof tangent), P_v
  A8  actuator and the composition: q bar = q bar_dir - dof_to_qpos(P_q; xi_k), v bar = v' bar - P_v, u bar += kp
      lambda on the unclamped dofs (DK_ADJ_CLAMP_KV=2: also kp w lambda on the clamped ones)

Compile-time options (defaults: the exact clamp derivative DK_ADJ_CLAMP_KV=1 and the frozen self-contact normal
DK_ADJ_NORMAL=3; the JAX interface's defaults are clamp_kv=0, normal=3: jax_ffi.CLAMP_KV, jax_ffi.NORMAL):
  DK_ADJ_STAGE     3 (default): limit rows and all contacts; 2: limit rows and the plane contacts (self contacts are
                   skipped and set ADJ_ROWS_SKIPPED); 1: no constraint rows (records with contacts or rows set
                   ADJ_ROWS_SKIPPED). 1 and 2 are for bisection
  DK_ADJ_CLAMP_KV  the derivative on a dof whose actuator force is clamped at its limit (the record's clamp mask):
                   1 (default): the actuator damping -kv lambda is dropped from the qvel VJP of clamped dofs (exact);
                   0: kept (as MuJoCo Warp's adjoint); 2: the soft clamp, a surrogate gradient: on a clamped dof all
                   three actuator paths (P_q kp, P_v kv, u bar kp) take w lambda instead of the exact derivative's 0,
                   w = exp(-o / delta), o = (|f| - tau_max) / tau_max >= 0 the relative overshoot of the unclamped PD
                   force f = kp (u - q) - kv v over its bound, so the derivative is continuous at the bound (w = 1)
                   and delta -> 0 is 1's; delta = k_adjoint's run-time argument clamp_soft (delta <= 0: exactly 1's
                   gradients), and the kernel needs ctrl (the tape has q, v per substep, not u)
  DK_ADJ_NORMAL, DK_ADJ_NORMAL_EPS  the normal derivative of the self contacts (adjoint_contact.header): default 3
                   (frozen where nu < 1e-3 m), 0 exact (MuJoCo Warp's)
  DK_ADJ_PARALLEL, DK_ADJ_PARALLEL_EPS  the segment parameters of a capsule-capsule contact with both free (the
                   near-parallel capsule rule, adjoint_contact.header): 1 held constant where sin^2(theta) < 1.2e-3
                   (finite), 0 exact (MuJoCo Warp's 1 / sin^2 derivative); default DK_ADJ_NORMAL != 0, so 1 with the
                   default normal option and 0 with the exact one (DK_ADJ_NORMAL=0)
  DK_ADJ_DEBUG     the kernel writes, per world and substep, its intermediates with MuJoCo Warp's adjoint names and
                   conventions (DEBUG_FIELDS: res_* in qpos form, split by source: RNE, actuator, limit rows,
                   contacts), per contact its replayed distance, the cotangents of (dist, pos, frame), row values x_e,
                   force and the diagnostics of the state groups (nablaengine.reference.adjoint_groups), and the
                   residual r(a*) = ID + arm a* - tau - J'f, a built-in check of the solves (J'f from the replayed
                   rows' forces -D x on their quadratic zones); needs ctrl for tau
  DK_PROFILE       per-phase clock64() cycles (PROF_PHASES); the marks are empty in every other build, so the
                   production code does not change with them
  DK_ADJ_WORLDS_PER_SM  the launch's worlds per SM (host, adj_launch_smem): -1 (default) the choice for the GPU (4 on
                   V100, as many as fit elsewhere), 0 as many as fit, N > 0 at most N

Launch (host, in this compilation unit): smem = adj_launch_smem() (or adj_launch_smem(-1, &err)) on the current device,
then k_adjoint<<<nw, 32, smem, stream>>>(...) (stream: e.g. XLA's in the FFI handler; the default stream when left
out). adj_launch_smem() also sets the kernel's shared-memory opt-in on that device, so it is called once per device
(e.g. where the constants are uploaded) and its size kept for the launches; it returns 0 on failure (the CUDA error in
err), which the caller must treat as an error: a launch with 0 bytes is not rejected, the kernel would use shared
memory it was not given.

Records the forward flagged (REFACT, NONCONV, SINGULAR, unsupported bits) are processed as any other and only
reported in info. Self-checks in info: ADJ_REPLAY_MISMATCH (a replayed contact is not penetrating, or the replayed limit
rows are not the tape's count).

The adjoint is for the mjlab model (no boxes, no frictionloss rows; header() asserts both). cuda_source(m) is one
compilation unit with the forward (source.DEVICE, step.STEP_DEVICE / STEP_KERNEL: shared DKC constants and device
functions) and the contact VJPs (adjoint_contact.source); the caller adds its __global__ forward kernel and host code
(the adjoint check binary of nablaengine.checks).
"""

import mujoco
import numpy as np

from nablaengine.g1_engine import adjoint_contact, source, step, tape

ADJ_ROWS_SKIPPED = 1 << 12  # info: a record has contacts or constraint rows that this build does not differentiate
ADJ_REPLAY_MISMATCH = 1 << 13  # info: a replayed contact is not penetrating, or the replayed limit rows differ in number
NO_ROW = 1e30  # debug: the kink and basis diagnostics of a contact or dof without that row / property

# -DDK_PROFILE: clock64() cycles of lane 0 per phase, summed over the substeps, as the
# forward's step.PROF_PHASES. The first PROF_PARTITION phases partition the kernel: per substep record (A0), poses (A3,
# adj_poses), integrator (A1), solve_Q (A1: L_Q, y, a* bar), solve_H (A2: L_H, lambda), tree_down (A3 + A4),
# limits (A5 / A6), contact_setup (geom poses, root axes), contact_rows (per batch: zones and adj_contact_one),
# contact_reduce (per batch: the lane-per-body sums), tree_up (A7), compose (A8); io is the kernel's prologue and
# epilogue (cotangents in, gradients out). The counts follow (per kernel: summed over substeps): contacts in the tape's
# tables, contacts with a quadratic row (the ones adj_contact_one differentiates), limit rows
PROF_PHASES = ("record", "poses", "integrator", "solve_Q", "solve_H", "tree_down", "limits", "contact_setup",
               "contact_rows", "contact_reduce", "tree_up", "compose", "io", "count_contacts",
               "count_contacts_active", "count_limit_rows")
PROF_PARTITION = 13
PROF_DESIGN = dict(record="A0", poses="A3", integrator="A1", solve_Q="A1", solve_H="A2", tree_down="A3+A4",
                   limits="A5+A6", contact_setup="A5", contact_rows="A5+A6", contact_reduce="A6", tree_up="A7",
                   compose="A8", io="kernel")  # the phases' labels A0-A8 (module docstring)

# per world and substep with -DDK_ADJ_DEBUG (float32 words); lengths in units of "nq", "nv", "nu", "mc" (the contact
# capacity) or a number, optionally times a width: (unit, width) -> (..., unit, width) when decoded. Keys and
# conventions of MuJoCo Warp's adjoint: qpos-form residuals res_* = dof_to_qpos(P_source; xi_k) (the engine's P is
# MuJoCo Warp's res), adj_qpos_M = +grad_q[y' M(q) a~] (the integrator's mass-matrix term), w_dir = a~
DEBUG_FIELDS = (
    ("g_qpos_out", "nq"), ("g_qvel_out", "nv"),  # cotangents of the substep's output state
    ("adj_qpos_int", "nq"), ("adj_qvel_int", "nv"), ("adj_qacc_adv", "nv"),  # integrator VJP alone (A1)
    ("y", "nv"), ("adj_qacc", "nv"), ("lam", "nv"), ("w_dir", "nv"),  # A1, A2
    ("adj_qpos_M", "nq"), ("res_qpos_rne", "nq"), ("res_qpos_act", "nq"), ("res_qvel_rne", "nv"), ("res_qvel_act", "nv"),
    ("res_qpos_contact", "nq"), ("res_qvel_contact", "nv"), ("res_qpos_limit", "nq"), ("res_qvel_limit", "nv"),
    ("grad_qpos_in", "nq"), ("grad_qvel_in", "nv"), ("grad_ctrl_k", "nu"),  # the substep's input cotangents
    ("resid", "nv"), ("resid_norm", 1), ("tau_norm", 1), ("id_norm", 1),  # r(a*) = ID + arm a* - tau - J'f
    ("ma_norm", 1),  # |M a*| (= L_Q (L_Q' a*) - h kv a*): the scale of the forward's solve M a* = tau - bias + J'f
    ("jtf_norm", 1),  # |J'f| of the replayed rows
    # per contact of the tape's table (stages >= 2; zeros past ncon and for skipped contacts): replayed distance and
    # position, MuJoCo Warp's con_dist_bar / con_pos_bar / con_frame_bar (zero when no row is on its quadratic zone),
    # x_e = J_e a* - aref_e of its rows (pyramid order), its force sum_e -D x_e u_e (world), the adj_contact_geom branch
    # bits, and the group diagnostics: kink min_e |x_e| / (|J_e a*| + |aref_e|), basis margin (self contact
    # | |n_y| - 0.5 |, plane-capsule |beta - 0.5|, else NO_ROW), sin(angle) of capsule-capsule with both parameters free
    # (else NO_ROW)
    ("con_dist", "mc"), ("con_pos", ("mc", 3)), ("con_dist_bar", "mc"), ("con_pos_bar", ("mc", 3)),
    ("con_frame_bar", ("mc", 9)), ("con_x", ("mc", 4)), ("con_force", ("mc", 3)), ("con_br", "mc"),
    ("con_kink", "mc"), ("con_basis", "mc"), ("con_sin_free", "mc"),
    # per dof: its limit row (if any) x, force -D x (on the quadratic zone, else 0) and kink (NO_ROW without a row)
    ("lim_x", "nv"), ("lim_force", "nv"), ("lim_kink", "nv"),
    # the poses the contact stage replays from: bodies (quaternion, origin) and collision geoms (centre, quaternion)
    ("body_q", ("nb", 4)), ("body_x", ("nb", 3)), ("geom_x", ("ng", 3)), ("geom_q", ("ng", 4)),
)


def debug_layout(m, max_contacts=None):
    """{name: (offset, length, shape)} of a -DDK_ADJ_DEBUG record (words; shape: the decoded trailing shape), and its
    length."""
    sizes = {"nq": m.nq, "nv": m.nv, "nu": m.nv - 6, "mc": max_contacts or step.maxcon(m), "nb": m.nbody,
             "ng": len(step.collision_geoms(m))}
    out, off = {}, 0
    for name, n in DEBUG_FIELDS:
        unit, width = n if isinstance(n, tuple) else (n, 1)
        cnt = sizes[unit] if isinstance(unit, str) else unit
        shape = () if (cnt == 1 and width == 1) else ((cnt,) if width == 1 else (cnt, width))
        out[name], off = (off, cnt * width, shape), off + cnt * width
    return out, off


def decode_debug(arr, m, max_contacts=None):
    """-DDK_ADJ_DEBUG records (..., debug words) -> dict of arrays (leading shape kept; scalars per record, per-contact
    vectors as (..., maxcon, width))."""
    lay, nw = debug_layout(m, max_contacts)
    a = np.asarray(arr, np.float32)
    assert a.shape[-1] == nw, (a.shape, nw)
    return {k: a[..., o:o + n].reshape(a.shape[:-1] + shape) for k, (o, n, shape) in lay.items()}


def solimp_powers(m):
    """The solimp powers of the model's constraint rows as the engine packs them (float32): the hinge limits'
    (step.pair_constants: c_jsolimp, one set for all joints) and the contact pairs' parameter sets (c_pset)."""
    sets = np.asarray(step.pair_param_sets(m)[0], np.float32)
    return sorted({float(np.float32(m.jnt_solimp[1][4]))} | {float(x) for x in sets[:, 3 + 4]})


def header(m):
    """#defines of the adjoint (stage, options, debug layout, the model's solimp power); asserts the mjlab-only scope."""
    assert not any(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in step.collision_geoms(m)), \
        "the adjoint is for the mjlab model: no box geoms"
    assert not (m.dof_frictionloss > 0).any(), "the adjoint is for the mjlab model: no frictionloss rows"
    assert m.nu == m.nv - 6, "one position actuator per hinge dof"
    lay, nw = debug_layout(m)
    lines = [f"#define ADJ_ROWS_SKIPPED {ADJ_ROWS_SKIPPED}u", f"#define ADJ_REPLAY_MISMATCH {ADJ_REPLAY_MISMATCH}u",
             f"#define DBG_NO_ROW {NO_ROW:.0e}f", f"#define DBG_WORDS {nw}  // -DDK_ADJ_DEBUG record",
             f"#define NAPROF {len(PROF_PHASES)}  // -DDK_PROFILE: PROF_PHASES"]
    lines += [f"#define APROF_{k.upper()} {i}" for i, k in enumerate(PROF_PHASES)]
    dof0 = int(m.body_dofadr[2]) - 2  # adj_contact_one walks the support's hinge bodies in index order
    assert m.body_dofadr[1] == 0 and m.body_dofnum[1] == 6 and all(
        m.body_dofnum[b] == 1 and m.body_dofadr[b] == b + dof0 for b in range(2, m.nbody)), \
        "a free root (dofs 0-5), then one hinge dof per body, in body order"
    lines.append(f"#define ADJ_HINGE_DOF0 {dof0}  // the hinge dof of body b >= 2 is b + ADJ_HINGE_DOF0")
    pw = solimp_powers(m)
    if len(pw) == 1:  # one power for every row: adj_row_impedance passes it as a constant
        lines.append(f"#define ADJ_SOLIMP_POWER {pw[0].hex()}f  // {pw[0]:g}, every row's solimp power")
    lines += [f"#define DBG_{k.upper()} {o}" for k, (o, n, _) in lay.items()]
    return "\n" + "\n".join(lines) + "\n"


ADJ_DEVICE = r"""
#ifndef DK_ADJ_STAGE
#define DK_ADJ_STAGE 3  // 1: smooth dynamics, integrator, actuator; 2: + limit rows, plane contacts; 3: + self contacts
#endif
#ifndef DK_ADJ_CLAMP_KV
#define DK_ADJ_CLAMP_KV 1  // 1: -kv lambda dropped on clamped dofs (exact); 0: kept (as MuJoCo Warp's adjoint)
#endif                     // 2: the soft clamp (lam_soft)
static_assert(DK_ADJ_CLAMP_KV >= 0 && DK_ADJ_CLAMP_KV <= 2, "DK_ADJ_CLAMP_KV: 0, 1 or 2");
#if DK_ADJ_CLAMP_KV == 2
// the soft clamp's delta is k_adjoint's run-time argument clamp_soft after ctrl (ADJ_SOFT_PARAM; a launch passes
// ADJ_SOFT_ARG(delta)); the other builds' kernel signature, and their code, are unchanged
#define ADJ_SOFT_PARAM , float clamp_soft
#define ADJ_SOFT_ARG(x) , (x)
#else
#define ADJ_SOFT_PARAM
#define ADJ_SOFT_ARG(x)
#endif
static_assert(HAS_BOX == 0 && NFLOSS == 0, "the adjoint is for the mjlab model (no boxes, no frictionloss rows)");
static_assert(DK_ADJ_STAGE >= 1 && DK_ADJ_STAGE <= 3, "DK_ADJ_STAGE: 1, 2 or 3");
static_assert(NB <= 31, "the contact reduction: lanes 0 and 31 take two of the root rotation dofs");
#define CS_N 24  // per-contact words (A6, in registers of the contact's lane): W_v, W_a, W_lam (0-8), contact point
                 // (9-11), the pose cotangents from its geometry of body 1 (f bar 12-14, tau bar 15-17) and of body 2
                 // (18-23)

// -DDK_PROFILE phase marks (adjoint.PROF_PHASES): APROF(k) adds lane 0's cycles since the previous mark to phase k.
// Unlike the forward's PROF they are empty in every other build (no __syncwarp() either), so the production code is
// the same with and without the marks; the marks sit right after a __syncwarp() of the code (the kernel's last one
// after its stores).
// APROF_COUNT(k, n) adds n (evaluated by all lanes: it may hold a full-mask ballot) to the count k
#ifdef DK_PROFILE
#ifdef DK_ADJ_DEBUG
#error "-DDK_PROFILE times the production path: build it without -DDK_ADJ_DEBUG"
#endif
#define APROF_START() do { __syncwarp(); if (lane == 0) s.aprof_t = clock64(); } while (0)
#define APROF(k) do { __syncwarp(); if (lane == 0) { long long t_ = clock64(); s.aprof[k] += t_ - s.aprof_t; s.aprof_t = t_; } __syncwarp(); } while (0)
#define APROF_COUNT(k, n) do { const long long n_ = (n); if (lane == 0) s.aprof[k] += n_; } while (0)
#else
#define APROF_START() do {} while (0)
#define APROF(k) do {} while (0)
#define APROF_COUNT(k, n) do {} while (0)
#endif
static_assert(NAPROF <= 32, "one profile counter per lane");

// Shared memory per world (10108 B on mjlab's G1, against 15740 B without the unions and the register-held contact
// words below: 9 instead of 6 worlds per SM on the 100 KB SMs of sm_86 / sm_120 (1 KB reserved per block, 128 B
// granularity: <= 10240 B for 9; the native sm_120 build, NVVM 20, has about 237 registers, which allow 8) and 16
// instead of 13 on sm_90a, where 128 registers then cap it): what is live through a whole substep, then two unions of
// arrays whose lifetimes within a substep are disjoint (the phases in adj_substep's order, a __syncwarp() between the
// last read of one alternative and the first write of the next; no function touches two alternatives of a union that
// overlap), and a contact batch's per-contact words in the registers of the contacts' lanes (adj_contacts: shuffles
// instead of a 3.2 KB scratch). The arithmetic and its order are unchanged: the gradients are bitwise the same.
//   UA: V, aadv (A0 .. A1: V only inside adj_poses' recursion, aadv until adj_integrator) | L, y, abar (A1 .. A2,
//       adj_solves) | the cotangents b*, fb, tb (A4 .. A7: adj_tree_down to adj_tree_up)
//   UB: A, A2, Vl, Vy (A3 .. A4: adj_poses / adj_tree_down to body_local_vjp) | {gx, gq, zr (A5 / A6: adj_contacts) |
//       rb (A7: adj_tree_up)} with P_q, P_v (A5 .. A8: adj_tree_down sets them to 0 after its last read of A .. Vy)
// Body 0's entries of the union arrays are never read (the root's recursions do not read their parent's). A caller
// that runs adj_tree_down again after adj_contacts needs A again, and P across adj_tree_down: -DDK_ADJ_DEBUG keeps A,
// zr, P_q, P_v outside UB (its pass by source does that); the U11 unit test (docs/verification.md) reruns adj_poses
struct Adj {
  // the substep (A0: from the record) and the adjoint's vectors, dof (NV) or qpos (NQ) indexed
  float q[NQ], v[NV], a[NV], at[NV];  // input state, a*, a~ = a* - a_adv
  float ny[NV], lam[NV];  // -y, lambda
  float qbar[NQ], vbar[NV], ubar[NV - 6];  // cotangents of the substep's output, then of its input; ctrl (summed)
  float com[3], fric;
  unsigned clamp[(NV + 31) / 32];  // the record's actuator clamp mask (dof bits)
  unsigned zone[(NEFC + 31) / 32];  // the record's quadratic-zone bits of the constraint rows (forward row order)
  unsigned con[(MAXCON + 1) / 2];  // the record's contact table: pair | slot << 9 per 16 bits
  int hdr;  // the record's header: ncon | nefc << 8 | nlim << 16 | niter << 24
  int flags;  // union of the records' flags (tape.py), ADJ_ROWS_SKIPPED, ADJ_REPLAY_MISMATCH
  // bodies, lane b = body b (body 0: the world, at rest). Forward quantities (A3) read up to A7:
  float Q[NB][4], X[NB][3], z[NB][3];  // pose (X_b: origin = hinge anchor), hinge axis z_b = R_b axis_b (b >= 2)
  float W[NB][3], AL[NB][3];  // channel 1 (fwd_body): angular velocity W of v, angular acceleration AL at a*
  float AL2[NB][3];  // channel 2: angular acceleration at a~ with v = 0, no gravity (the twist recursion of a~)
  float Wl[NB][3], Wy[NB][3];  // angular parts of the twists of lambda, -y
#ifdef DK_ADJ_DEBUG  // outside UB: the debug build's pass by source runs adj_tree_down after adj_contacts
  float A[NB][3], zr[3][3], Pq[NV], Pv[NV];
#endif
  union {  // UA
    struct {  // A0 .. A1
      float V[NB][3];  // channel 1's linear velocity of the origin: only adj_poses' recursion reads it
      float aadv[NV];  // a_adv (the record's): adj_integrator
    };
    struct {  // A1, A2 (adj_solves)
      float L[NTRI];  // a factor from the record: L_Q (A1), then L_H (A2)
      float y[NV], abar[NV];  // y, a_adv bar then a* bar
    };
    struct {  // cotangents (A4, then completed in A7): channel 1, channel 2, the lambda and -y twists
      float bW[NB][3], bAL[NB][3], bA[NB][3], bAL2[NB][3], bA2[NB][3], bWl[NB][3], bVl[NB][3], bWy[NB][3], bVy[NB][3];
      float fb[NB][3], tb[NB][3];  // pose cotangents f bar_b, tau bar_b (A4, A6), then the subtree sums F_b, N_b (A7)
    };
  };
  union {  // UB
    struct {  // A3 .. A4: read last by body_local_vjp (adj_tree_down)
#ifndef DK_ADJ_DEBUG
      float A[NB][3];  // channel 1's linear acceleration of the origin at a*, with gravity
#endif
      float A2[NB][3];  // channel 2's linear acceleration
      float Vl[NB][3], Vy[NB][3];  // linear parts of the twists of lambda, -y
    };
    struct {  // A5 .. A8
      union {
        struct {  // A5 / A6 (adj_contacts; a batch's per-contact words stay in the registers of the contacts' lanes)
          float gx[NGEOMC][3], gq[NGEOMC][4];  // world poses of the collision geoms (the forward's geom_poses)
#ifndef DK_ADJ_DEBUG
          float zr[3][3];  // the root's rotation axes R_1 e_k (the anchor of their Jacobian columns is X_1)
#endif
        };
        float rb[NB][3];  // A7: r bar_b of r_b = X_b - X_parent (the parent subtracts it from its F)
      };
#ifndef DK_ADJ_DEBUG
      // P = grad Psi: dof tangent (root: world translation, local rotation), qvel; set to 0 by
      // adj_tree_down after its last read of A .. Vy, then summed by adj_limits, adj_contacts, adj_tree_up, adj_compose
      float Pq[NV], Pv[NV];
#endif
    };
  };
#ifdef DK_ADJ_DEBUG
  float dF[NB][3], dN[NB][3], did[NV], jtf[NV], ma_norm;  // residual check: subtree forces of channel 1, ID, J'f, |M a*|
  float tmp[NV], zero[NV];  // scratch, zeros (the split by source); only the debug build has them
#endif
#ifdef DK_PROFILE
  long long aprof[NAPROF], aprof_t;  // phase cycles and counts of lane 0 (APROF), the last mark
#endif
#if DK_ADJ_CLAMP_KV == 2
  float u[NV - 6], soft;  // the world's ctrl and delta (clamp_soft) for lam_soft (120 B: still 9 worlds per SM)
#endif
};

// LANE_BOUND(lane) at the top of a __noinline__ function: its lane parameter carries no bound for the compiler (the
// kernel's lane = threadIdx.x & 31 does not cross the call; see host.py k_substep), so each lane-strided loop
// (for i = lane; i < n; i += 32) would get an unknown trip count, unrolled by 4 with a remainder loop. In adj_solves,
// adj_tree_down, adj_tree_up, adj_limits and adj_contacts; bitwise the same gradients. k_adjoint on the RTX 3090
// (the interface defaults, a 4096-state corpus, 10 iterations): 1.9% / 1.7% faster at 1787 / 4096 worlds on top of
// the blocked
// tri_solve_packed; adj_solves 830 -> 595 SASS instructions (sm_86); k_adjoint 128 -> 140 registers on sm_120 (whose
// 6 worlds per SM were set by the 15.7 KB of shared memory, not by the registers), 127 -> 128 on sm_90a
#define LANE_BOUND(lane) __builtin_assume((lane) >= 0 && (lane) < 32)

// y / d as the fast path of the compiler's IEEE division (div.rn.f32 on sm_70 and sm_86, CUDA 12: r = MUFU.RCP(d), one
// Newton step r2 = r + r (1 - d r), q = y r2, q + r2 (y - d q)), whose reciprocal part depends on d alone. The
// division's own FCHK test sends operands outside the fast path's range (small |y|, |d| or the quotient near the
// exponent limits) to a slow path; this form has no branch, so a solve's divisions schedule ahead of the chain of y.
// Bitwise y / d where the residual y - d q is exact and d, 1 / d and the quotient are normal floats: |y| >= 2^-102 (the
// residual's last bit is then >= 2^-149). Below 2^-102 the residual can underflow and the quotient be off by an ulp
// (at most 1; with 2^32 random y per binade and the tapes' divisors L_kk, 0.063 .. 98, mismatches occur in every binade
// below 2^-102, rarer by about 2x per binade upwards); y = -0 gives +0, an overflowing quotient NaN instead of +-inf.
// Through the solves and sweeps such an ulp grows: the corpora's cotangents (N(0, 1) entries) scaled by s >= 1e-28 give gradients bitwise
// equal to a build with '/' (both corpora, 4 substeps, 3 cotangents each); s = 1e-29 changes 146 entries for one of the
// six, 1e-30 52 to 327 of the 409600 (up to 2816 ulp), 1e-32 about 4%, 1e-36 about 92%. So cotangents of realistic
// size (entries > 1e-20) give the gradients of '/' bitwise (at s = 1 ncu showed the slow path of '/' never ran here)
__device__ __forceinline__ float rcp_step(float d) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(d));
  return __fmaf_rn(r, __fmaf_rn(-d, r, 1.f), r);
}
__device__ __forceinline__ float div_fast(float y, float d, float r2) {
  const float q = __fmaf_rn(y, r2, 0.f);
  return __fmaf_rn(r2, __fmaf_rn(-d, q, y), q);
}

// x = (L L')^-1 b in place for a packed lower-triangular L (TRI) in shared memory and a shared NV vector b; one copy
// for both solves. The arithmetic of chol_solve_compact's solve, in its order (y_r -= L_rk x_k for k ascending, then
// x_r = y_r / L_rr; backwards likewise), so bitwise its result on the corpora; the division is div_fast (no branch,
// the reciprocal of L_kk off the chain). Two forms of the same arithmetic in the same plain C (ptxas fuses every update
// into one fma in both, on sm_70, sm_86, sm_90a and sm_120), so bitwise the same gradients:
// DK_ADJ_TRI_BLOCKED=1 (the default where 32 < NV <= 35) solves rows 0..31 in blocks of 8 as the forward's
// chol_resolve: a block's 8 broadcasts are independent, its 8 x 8 triangle is
// solved redundantly on every lane (broadcast loads of L; the chain: 8 quotients with their in-block updates), then
// each lane applies the block's 8 columns to its row, so one round of shuffles per block instead of one shuffle per
// row on the chain; the tail rows 32.. on every lane. k_adjoint on the RTX 3090 (the interface defaults, a 4096-state
// corpus, 10 iterations, with the lane mask): 5.9% / 6.0% faster at 1787 / 4096 worlds.
// =0: column by column (lane l holds entries l and l + 32, column k broadcast by __shfl_sync), written for a short
// step: the step's entries L_kk, L_rk are loaded before the broadcast, at a clamped index valid for every lane,
// the updates are selects instead of branches, so a step's critical path is the broadcast, three fused multiply-adds
// of the quotient and the one of the update (unrolled by 5)
#ifndef DK_ADJ_TRI_BLOCKED
#define DK_ADJ_TRI_BLOCKED (NV > 32 && NV <= 35)
#endif
#if DK_ADJ_TRI_BLOCKED
static_assert(NV > 32 && NV <= 35, "rows 0..31 on the lanes, a tail of 1..3 rows on every lane");
#define TRI_NT (NV - 32)  // tail rows
__device__ __noinline__ void tri_solve_packed(const float* L, float* b, int lane) {
  const int r = lane, r1 = min(32 + lane, NV - 1);  // lanes 0 .. TRI_NT - 1: tail row 32 + lane (y1); others a dummy
  const int tr = TRI(r, 0), tr1 = TRI(r1, 0);  // the rows' starts
  __syncwarp();
  float y = b[r], y1 = b[r1];
#pragma unroll
  for (int kb = 0; kb < 32; kb += 8) {  // forward: L y = b on rows 0..31
    float yb[8];
#pragma unroll
    for (int j = 0; j < 8; j++) yb[j] = __shfl_sync(0xffffffffu, y, kb + j);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float u = yb[j];
#pragma unroll
      for (int i = 0; i < j; i++) u = u - L[TRI(kb + j, kb + i)] * yb[i];
      const float d = L[TRI(kb + j, kb + j)];
      yb[j] = div_fast(u, d, rcp_step(d));
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
      y = r >= kb + 8 ? y - L[tr + min(kb + j, r)] * yb[j] : (r == kb + j ? yb[j] : y);
      y1 = y1 - L[tr1 + kb + j] * yb[j];
    }
  }
  float yt[TRI_NT];  // the tail rows on every lane: forward, backward, then their part of rows 0..31
#pragma unroll
  for (int t = 0; t < TRI_NT; t++) {
    float u = __shfl_sync(0xffffffffu, y1, t);
#pragma unroll
    for (int i = 0; i < t; i++) u = u - L[TRI(32 + t, 32 + i)] * yt[i];
    const float d = L[TRI(32 + t, 32 + t)];
    yt[t] = div_fast(u, d, rcp_step(d));
  }
#pragma unroll
  for (int t = TRI_NT - 1; t >= 0; t--) {
    float u = yt[t];
#pragma unroll
    for (int i = TRI_NT - 1; i > t; i--) u = u - L[TRI(32 + i, 32 + t)] * yt[i];
    const float d = L[TRI(32 + t, 32 + t)];
    yt[t] = div_fast(u, d, rcp_step(d));
  }
#pragma unroll
  for (int t = TRI_NT - 1; t >= 0; t--) y = y - L[TRI(32 + t, r)] * yt[t];
#pragma unroll
  for (int kb = 24; kb >= 0; kb -= 8) {  // backward: L' x = y on rows 0..31
    float xb[8];
#pragma unroll
    for (int j = 0; j < 8; j++) xb[j] = __shfl_sync(0xffffffffu, y, kb + j);
#pragma unroll
    for (int j = 7; j >= 0; j--) {
      float u = xb[j];
#pragma unroll
      for (int i = 7; i > j; i--) u = u - L[TRI(kb + i, kb + j)] * xb[i];
      const float d = L[TRI(kb + j, kb + j)];
      xb[j] = div_fast(u, d, rcp_step(d));
    }
#pragma unroll
    for (int j = 7; j >= 0; j--)
      y = r < kb ? y - L[TRI(kb + j, min(r, kb + j))] * xb[j] : (r == kb + j ? xb[j] : y);
  }
  __syncwarp();
  b[r] = y;
  if (lane == 0) {  // (lane t writing entry 32 + t would index yt dynamically)
#pragma unroll
    for (int t = 0; t < TRI_NT; t++) b[32 + t] = yt[t];
  }
  __syncwarp();
}
#undef TRI_NT
#else
__device__ __noinline__ void tri_solve_packed(const float* L, float* b, int lane) {
  static_assert(NV <= 64, "two entries per lane");
  // r0 / r1: clamped to a valid row (only lanes with has0 / has1 use them; has0 is the constant true for NV >= 32)
  const int r0 = NV >= 32 ? lane : min(lane, NV - 1), r1 = min(lane + 32, NV - 1);
  const bool has0 = NV >= 32 || lane < NV, has1 = lane + 32 < NV;
  const int t0 = TRI(r0, 0), t1 = TRI(r1, 0);  // the rows' starts
  __syncwarp();
  float y0 = has0 ? b[r0] : 0.f, y1 = has1 ? b[r1] : 0.f;
#pragma unroll 5
  for (int k = 0; k < NV; k++) {  // forward: L y = b
    const float d = L[TRI(k, k)], l0 = L[t0 + min(k, r0)], l1 = L[t1 + min(k, r1)];
    const float yk = div_fast(__shfl_sync(0xffffffffu, k < 32 ? y0 : y1, k & 31), d, rcp_step(d));
    y0 = r0 == k ? yk : (r0 > k ? y0 - l0 * yk : y0);
    y1 = r1 == k ? yk : (has1 && r1 > k ? y1 - l1 * yk : y1);
  }
#pragma unroll 5
  for (int k = NV - 1; k >= 0; k--) {  // backward: L' x = y
    const int tk = TRI(k, 0);
    const float d = L[tk + k], l0 = L[tk + min(r0, k)], l1 = L[tk + min(r1, k)];
    const float xk = div_fast(__shfl_sync(0xffffffffu, k < 32 ? y0 : y1, k & 31), d, rcp_step(d));
    y0 = r0 == k ? xk : (r0 < k ? y0 - l0 * xk : y0);
    y1 = r1 == k ? xk : (has1 && r1 < k ? y1 - l1 * xk : y1);
  }
  __syncwarp();
  if (has0) b[r0] = y0;  // (a lane past NV ran row NV - 1's arithmetic on its own registers; it is never read)
  if (has1) b[r1] = y1;
  __syncwarp();
}
#endif

// out = L (L' x) for a packed lower-triangular L (Q x for L = L_Q); tmp: NV scratch; all shared (-DDK_ADJ_DEBUG: |M a*|)
__device__ void tri_mul_llt(const float* L, const float* x, float* tmp, float* out, int lane) {
  for (int j = lane; j < NV; j += 32) {  // tmp = L' x, column j
    float acc = 0.f;
    for (int i = j; i < NV; i++) acc += L[TRI(i, j)] * x[i];
    tmp[j] = acc;
  }
  __syncwarp();
  for (int i = lane; i < NV; i += 32) {  // out = L tmp, row i
    float acc = 0.f;
    for (int j = 0; j <= i; j++) acc += L[TRI(i, j)] * tmp[j];
    out[i] = acc;
  }
  __syncwarp();
}

__device__ __forceinline__ float dot4(const float* a, const float* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
__device__ __forceinline__ float adj_sinc(float x) { return x * x < 1e-4f ? 1.f - x * x / 6.f : sinf(x) / x; }
__device__ __forceinline__ float adj_dsinc_x(float x) {  // sinc'(x) / x = (x cos x - sin x) / x^3, smooth and even
  return x * x < 1e-4f ? -1.f / 3.f + x * x / 30.f : (x * cosf(x) - sinf(x)) / (x * x * x);
}

// MuJoCo Warp's _adj_quat_integrate: the VJP of q' = normalize(normalize(q) rho), rho = (cos x, (h/2) sinc(x)
// w), x = (h/2) |w| (the engine's axis-angle update in a form that is smooth at w = 0), for the cotangent g of q':
// qbar = its cotangent of q (4), wbar = of w (3)
__device__ void quat_integrate_vjp(const float* q, const float* w, float h, const float* g, float* qbar, float* wbar) {
  float nq2 = dot4(q, q);
  if (nq2 == 0.f) {
    for (int k = 0; k < 4; k++) qbar[k] = 0.f;
    for (int k = 0; k < 3; k++) wbar[k] = 0.f;
    return;
  }
  float hh = 0.5f * h, x = hh * sqrtf(dot3(w, w)), sc = adj_sinc(x), sh = hh * sc, nq = sqrtf(nq2);
  float rho[4] = {cosf(x), sh * w[0], sh * w[1], sh * w[2]}, qn[4], u[4], gu[4], gqn[4], gr[4];
  for (int k = 0; k < 4; k++) qn[k] = q[k] / nq;
  qmul(qn, rho, u);
  float nu = sqrtf(dot4(u, u)), od = 0.f;
  for (int k = 0; k < 4; k++) od += u[k] / nu * g[k];
  for (int k = 0; k < 4; k++) gu[k] = (g[k] - u[k] / nu * od) / nu;  // the final normalize
  float rc[4] = {rho[0], -rho[1], -rho[2], -rho[3]}, qc[4] = {qn[0], -qn[1], -qn[2], -qn[3]};
  qmul(gu, rc, gqn);  // u = qn rho: the transposes of left / right multiplication are products with the conjugates
  qmul(qc, gu, gr);
  float qd = dot4(qn, gqn);
  for (int k = 0; k < 4; k++) qbar[k] = (gqn[k] - qn[k] * qd) / nq;  // qn = normalize(q)
  // rho's dependence on w through x collapses to smooth even functions of x times w (no w / |w|)
  float coef = hh * hh * (-gr[0] * sc + hh * dot3(w, gr + 1) * adj_dsinc_x(x));
  for (int k = 0; k < 3; k++) wbar[k] = sh * gr[1 + k] + coef * w[k];
}

// twist (W, V) of body b's origin for the generalized vector s from its parent's (W angular, V the origin's velocity;
// z, r: hinge axis and r_b = X_b - X_parent; sb = s at the body's hinge dof); the root's is (R_1 s_rot, s_trans). The
// same recursion carries channel 2 (s = a~ at v = 0 without gravity: W = AL2, V = A2)
__device__ __forceinline__ void adj_twist(int b, const float* Q, const float* z, const float* r, const float* Wp,
                                          const float* Vp, const float* s, float sb, float* W, float* V) {
  if (b == 1) {
    qrot(Q, s + 3, W);
    for (int k = 0; k < 3; k++) V[k] = s[k];
    return;
  }
  float u[3];
  cross(Wp, r, u);
  for (int k = 0; k < 3; k++) { W[k] = Wp[k] + z[k] * sb; V[k] = Vp[k] + u[k]; }
}

// o = I x for the body inertia I = R_i diag(d) R_i' (qi: the inertia frame), as body_force
__device__ __forceinline__ void inert_mul(const float* qi, const float* d, const float* x, float* o) {
  float l[3];
  qrot_inv(qi, x, l);
  for (int k = 0; k < 3; k++) l[k] *= d[k];
  qrot(qi, l, o);
}

// A4 for body b alone, a pure function of its forward quantities (ipos: COM offset in the body frame, with
// the torso's DR offset). Its RNE forces about the origin X_b in both channels, F1 = m (A + AL x c + W x (W x c)),
// N1 = I AL + W x I W + c x F1 (= body_force), F2 = m (A2 + AL2 x c), N2 = I AL2 + c x F2 (c = R_b ipos), and the VJP of
// psi_b = F1.Vl + N1.Wl + F2.Vy + N2.Wy (virtual work: lambda' ID - y' M a~ = sum_b psi_b): the cotangents of its twists
// (bVl = F1, bWl = N1, bVy = F2, bWy = N2), of channel 1 (bW, bAL, bA) and channel 2 (bAL2, bA2), and of its own
// rotation through c and I (tb; its position does not enter psi_b)
__device__ void body_local_vjp(int b, const float* Q, const float* ipos, const float* W, const float* AL, const float* A,
                               const float* AL2, const float* A2, const float* Wl, const float* Vl, const float* Wy,
                               const float* Vy, float* bW, float* bAL, float* bA, float* bAL2, float* bA2, float* bWl,
                               float* bVl, float* bWy, float* bVy, float* tb) {
  float c[3], qi[4], t[3], u[3], v3[3];
  qrot(Q, ipos, c);
  qmul(Q, c_iquat + 4 * b, qi);
  const float* d = c_inert + 3 * b;
  const float m = c_mass[b];
  float IW[3], IL[3], IL2[3], IWl[3], IWy[3], X[3], IX[3];
  inert_mul(qi, d, W, IW);
  inert_mul(qi, d, AL, IL);
  inert_mul(qi, d, AL2, IL2);
  inert_mul(qi, d, Wl, IWl);
  inert_mul(qi, d, Wy, IWy);
  cross(Wl, W, X);  // Wl x W
  inert_mul(qi, d, X, IX);
  // forces (bVl = F1, bWl = N1, bVy = F2, bWy = N2)
  float wc[3], wwc[3], F1[3], F2[3];
  cross(W, c, wc);
  cross(W, wc, wwc);
  cross(AL, c, t);
  for (int k = 0; k < 3; k++) F1[k] = m * (A[k] + t[k] + wwc[k]);
  cross(AL2, c, t);
  for (int k = 0; k < 3; k++) F2[k] = m * (A2[k] + t[k]);
  cross(W, IW, t);
  cross(c, F1, u);
  for (int k = 0; k < 3; k++) { bVl[k] = F1[k]; bWl[k] = IL[k] + t[k] + u[k]; }
  cross(c, F2, u);
  for (int k = 0; k < 3; k++) { bVy[k] = F2[k]; bWy[k] = IL2[k] + u[k]; }
  // channel 1: g = m (Vl + Wl x c) is the cotangent of F1 (Vl directly, Wl through c x F1)
  float g[3], gw[3], cb[3], tau[3];
  cross(Wl, c, t);
  for (int k = 0; k < 3; k++) g[k] = m * (Vl[k] + t[k]);
  cross(c, g, t);
  for (int k = 0; k < 3; k++) { bA[k] = g[k]; bAL[k] = t[k] + IWl[k]; }
  cross(g, W, gw);
  cross(wc, g, t);
  cross(c, gw, u);
  cross(IW, Wl, v3);
  for (int k = 0; k < 3; k++) bW[k] = t[k] + u[k] + v3[k] + IX[k];
  cross(F1, Wl, t);  // c bar
  cross(g, AL, u);
  cross(gw, W, v3);
  for (int k = 0; k < 3; k++) cb[k] = t[k] + u[k] + v3[k];
  cross(IL, Wl, t);  // rotation of I: (I x) x y bar - x x (I y bar) for each I x
  cross(AL, IWl, u);
  for (int k = 0; k < 3; k++) tau[k] = t[k] - u[k];
  cross(IW, X, t);
  cross(W, IX, u);
  for (int k = 0; k < 3; k++) tau[k] += t[k] - u[k];
  // channel 2 (twist -y): g2 = m (Vy + Wy x c)
  cross(Wy, c, t);
  for (int k = 0; k < 3; k++) g[k] = m * (Vy[k] + t[k]);
  cross(c, g, t);
  for (int k = 0; k < 3; k++) { bA2[k] = g[k]; bAL2[k] = t[k] + IWy[k]; }
  cross(F2, Wy, t);
  cross(g, AL2, u);
  for (int k = 0; k < 3; k++) cb[k] += t[k] + u[k];
  cross(IL2, Wy, t);
  cross(AL2, IWy, u);
  for (int k = 0; k < 3; k++) tau[k] += t[k] - u[k];
  cross(c, cb, t);  // c = R_b ipos rotates with the body
  for (int k = 0; k < 3; k++) tb[k] = tau[k] + t[k];
}

// A3, first part: poses (Q_b, X_b) and channel 1 (the twist of v, the accelerations at a* with gravity) by the forward's
// fwd_body, one tree level at a time. The contact replay needs the poses bitwise the forward's: fwd_body is written
// with explicit roundings (source.DEVICE), so they are in any kernel and compilation unit (before, with the plain C
// fused by ptxas by context, only this loop inlined into the kernel as in substep_pre gave them, and a non-inlined
// version differed by an ulp in 30% of the bodies' quaternions)
__device__ __forceinline__ void adj_poses(Adj& s, int lane) {
  const int b = lane, d = b < NB ? c_depth[b] : -1, p = b < NB ? c_parent[b] : 0;
  for (int L = 1; L <= MAXD; L++) {
    if (d == L)
      fwd_body(b, s.Q[p], s.X[p], s.W[p], s.V[p], s.AL[p], s.A[p], s.q, s.v, s.a, s.Q[b], s.X[b], s.W[b], s.V[b],
               s.AL[b], s.A[b]);
    __syncwarp();
  }
}

// A3 + A4: the root-to-leaf sweep after adj_poses (hinge axes, channel 2, the twists of lam and ny = -y; one tree level
// at a time) and the bodies' local VJP; f bar = 0, tau bar from A4 (the contact stage adds to both), and P_q = P_v = 0
// (after the last read of A .. Vy, whose memory they share). lam and ny are the seeds of the two channels (a zero
// vector switches one off, both off leave f bar = tau bar = 0: the debug build's split by source)
__device__ __noinline__ void adj_tree_down(Adj& s, const float* lam, const float* ny, int lane) {
  LANE_BOUND(lane);
  const int b = lane, d = b < NB ? c_depth[b] : -1, p = b < NB ? c_parent[b] : 0;
  // what does not depend on the parent's twists comes before the sweep (off the levels' chain): r_b and the three
  // seeds at the body's dof
  const int i = b >= 2 && b < NB ? c_dadr[b] : 0;
  float r[3];
  for (int k = 0; k < 3; k++) r[k] = b < NB ? s.X[b][k] - s.X[p][k] : 0.f;
  const float sa = s.at[i], sl = lam[i], sy = ny[i];
  for (int L = 1; L <= MAXD; L++) {
    if (d == L) {
      if (b >= 2) qrot(s.Q[b], c_axis + 3 * b, s.z[b]);
      adj_twist(b, s.Q[b], s.z[b], r, s.AL2[p], s.A2[p], s.at, sa, s.AL2[b], s.A2[b]);
      adj_twist(b, s.Q[b], s.z[b], r, s.Wl[p], s.Vl[p], lam, sl, s.Wl[b], s.Vl[b]);
      adj_twist(b, s.Q[b], s.z[b], r, s.Wy[p], s.Vy[p], ny, sy, s.Wy[b], s.Vy[b]);
    }
    __syncwarp();
  }
  if (b >= 1 && b < NB) {
    float ip[3];
    body_ipos(b, s.com, ip);
    body_local_vjp(b, s.Q[b], ip, s.W[b], s.AL[b], s.A[b], s.AL2[b], s.A2[b], s.Wl[b], s.Vl[b], s.Wy[b], s.Vy[b],
                   s.bW[b], s.bAL[b], s.bA[b], s.bAL2[b], s.bA2[b], s.bWl[b], s.bVl[b], s.bWy[b], s.bVy[b], s.tb[b]);
    for (int k = 0; k < 3; k++) s.fb[b][k] = 0.f;
  }
  __syncwarp();
  for (int i = lane; i < NV; i += 32) { s.Pq[i] = 0.f; s.Pv[i] = 0.f; }
  __syncwarp();
}

// A7: the leaf-to-root sweep, one tree level at a time. Body b first pulls its children's finished
// cotangents (the reverse of their recursions from b's quantities; F_b, N_b collect the children's subtree sums as
// accumulate() does), then finishes its own: P_v of its dof, z bar and r bar, its pose cotangent (tau bar += z x z bar,
// f bar += r bar, the parent's f bar -= r bar when it pulls), and P_q = z . N_b; the root P_v (angular), P_q (F_1,
// R_1' N_1). Adds to s.Pq, s.Pv. lam, ny: the seeds of adj_tree_down
__device__ __noinline__ void adj_tree_up(Adj& s, const float* lam, const float* ny, int lane) {
  LANE_BOUND(lane);
  const int b = lane, d = b < NB ? c_depth[b] : -1, p = b < NB ? c_parent[b] : 0;
  for (int L = MAXD; L >= 1; L--) {
    if (d == L) {
      float bW[3], bAL[3], bA[3], bAL2[3], bA2[3], bWl[3], bVl[3], bWy[3], bVy[3], F[3], N[3], t[3], u[3];
      for (int k = 0; k < 3; k++) {
        bW[k] = s.bW[b][k]; bAL[k] = s.bAL[b][k]; bA[k] = s.bA[b][k]; bAL2[k] = s.bAL2[b][k]; bA2[k] = s.bA2[b][k];
        bWl[k] = s.bWl[b][k]; bVl[k] = s.bVl[b][k]; bWy[k] = s.bWy[b][k]; bVy[k] = s.bVy[b][k];
        F[k] = s.fb[b][k]; N[k] = s.tb[b][k];
      }
      const float* Wb = s.W[b];
      for (int i = 0; i < c_nchild[b]; i++) {
        const int c = c_child[3 * b + i];
        float r[3], zv[3], vc = s.v[c_dadr[c]];
        for (int k = 0; k < 3; k++) { r[k] = s.X[c][k] - s.X[b][k]; zv[k] = s.z[c][k] * vc; }
        const float *cW = s.bW[c], *cAL = s.bAL[c], *cA = s.bA[c];
        // channel 1: W_c = W_b + z qd, AL_c = AL_b + z qdd + W_b x z qd, A_c = A_b + AL_b x r + W_b x (W_b x r)
        cross(zv, cAL, t);
        for (int k = 0; k < 3; k++) bW[k] += cW[k] + t[k];
        cross(Wb, r, u);
        cross(u, cA, t);
        for (int k = 0; k < 3; k++) bW[k] += t[k];
        cross(cA, Wb, u);
        cross(r, u, t);
        for (int k = 0; k < 3; k++) bW[k] += t[k];
        cross(r, cA, t);
        for (int k = 0; k < 3; k++) { bAL[k] += cAL[k] + t[k]; bA[k] += cA[k]; }
        // channel 2 and the twists: W_c = W_b + z s, V_c = V_b + W_b x r
        cross(r, s.bA2[c], t);
        for (int k = 0; k < 3; k++) { bAL2[k] += s.bAL2[c][k] + t[k]; bA2[k] += s.bA2[c][k]; }
        cross(r, s.bVl[c], t);
        for (int k = 0; k < 3; k++) { bWl[k] += s.bWl[c][k] + t[k]; bVl[k] += s.bVl[c][k]; }
        cross(r, s.bVy[c], t);
        for (int k = 0; k < 3; k++) { bWy[k] += s.bWy[c][k] + t[k]; bVy[k] += s.bVy[c][k]; }
        // pose sweep: F_b += F_c - r bar_c (the parent's share of r bar_c), N_b += N_c + r x F_c
        cross(r, s.fb[c], t);
        for (int k = 0; k < 3; k++) { F[k] += s.fb[c][k] - s.rb[c][k]; N[k] += s.tb[c][k] + t[k]; }
      }
      if (b >= 2) {
        const int i = c_dadr[b];
        const float *z = s.z[b], *Wp = s.W[p];
        const float qd = s.v[i];
        float zb[3], rb[3];
        cross(Wp, z, t);
        s.Pv[i] += dot3(z, bW) + dot3(bAL, t);
        cross(bAL, Wp, t);
        for (int k = 0; k < 3; k++)
          zb[k] = qd * (bW[k] + t[k]) + s.a[i] * bAL[k] + s.at[i] * bAL2[k] + lam[i] * bWl[k] + ny[i] * bWy[k];
        cross(bA, s.AL[p], rb);
        cross(bA, Wp, u);
        cross(u, Wp, t);
        for (int k = 0; k < 3; k++) rb[k] += t[k];
        cross(bA2, s.AL2[p], t);
        for (int k = 0; k < 3; k++) rb[k] += t[k];
        cross(bVl, s.Wl[p], t);
        for (int k = 0; k < 3; k++) rb[k] += t[k];
        cross(bVy, s.Wy[p], t);
        for (int k = 0; k < 3; k++) rb[k] += t[k];
        cross(z, zb, t);  // z_b = R_b axis rotates with the body
        for (int k = 0; k < 3; k++) { N[k] += t[k]; F[k] += rb[k]; s.rb[b][k] = rb[k]; }
        s.Pq[i] += dot3(z, N);
      } else {  // root: W_1 = R_1 w, AL_1 = R_1 a_w, AL2_1, Wl_1, Wy_1 likewise; its A, A2, V's do not depend on the pose
        float l[3];
        qrot_inv(s.Q[1], bW, l);
        for (int k = 0; k < 3; k++) s.Pv[3 + k] += l[k];
        cross(s.W[1], bW, t);
        for (int k = 0; k < 3; k++) N[k] += t[k];
        cross(s.AL[1], bAL, t);
        for (int k = 0; k < 3; k++) N[k] += t[k];
        cross(s.AL2[1], bAL2, t);
        for (int k = 0; k < 3; k++) N[k] += t[k];
        cross(s.Wl[1], bWl, t);
        for (int k = 0; k < 3; k++) N[k] += t[k];
        cross(s.Wy[1], bWy, t);
        for (int k = 0; k < 3; k++) N[k] += t[k];
        qrot_inv(s.Q[1], N, l);
        for (int k = 0; k < 3; k++) { s.Pq[k] += F[k]; s.Pq[3 + k] += l[k]; }
      }
      for (int k = 0; k < 3; k++) {  // publish for the parent
        s.bW[b][k] = bW[k]; s.bAL[b][k] = bAL[k]; s.bA[b][k] = bA[k]; s.bAL2[b][k] = bAL2[k]; s.bA2[b][k] = bA2[k];
        s.bWl[b][k] = bWl[k]; s.bVl[b][k] = bVl[k]; s.bWy[b][k] = bWy[k]; s.bVy[b][k] = bVy[k];
        s.fb[b][k] = F[k]; s.tb[b][k] = N[k];
      }
    }
    __syncwarp();
  }
}

// A1: the integrator's VJP. In: qbar, vbar = the cotangents of q' = q (+) h v', v' = v + h a_adv; out:
// qbar = its direct part (translation, hinges unchanged; the quaternion through quat_integrate_vjp), vbar = v' bar,
// abar = a_adv bar = h v' bar. w' = v'_rot rebuilt as the forward computes it (fmaf), bitwise the same
__device__ void adj_integrator(Adj& s, int lane) {
  const float h = c_misc[0];
  for (int i = lane; i < NV; i += 32) {
    if (i < 3) s.vbar[i] += h * s.qbar[i];
    else if (i >= 6) s.vbar[i] += h * s.qbar[i + 1];
  }
  if (lane == 0) {
    float w[3], qb[4], wb[3];
    for (int k = 0; k < 3; k++) w[k] = fmaf(h, s.aadv[3 + k], s.v[3 + k]);
    quat_integrate_vjp(s.q + 3, w, h, s.qbar + 3, qb, wb);
    for (int k = 0; k < 4; k++) s.qbar[3 + k] = qb[k];
    for (int k = 0; k < 3; k++) s.vbar[3 + k] += wb[k];
  }
  __syncwarp();
  for (int i = lane; i < NV; i += 32) s.abar[i] = h * s.vbar[i];
  __syncwarp();
}

// A1 / A2, from abar = a_adv bar: y = Q^-1 a_adv bar (L_Q), abar = a* bar = M y = a_adv bar - h kv y (Q = M + h
// diag(kv) and Q y = a_adv bar; exact to rounding, unlike the product form L_Q (L_Q' y) - h kv y, whose float32 error
// grows with cond(Q) ~ 4e3), ny = -y, lam = H^-1 a* bar (L_H). rec: the substep's record
// (the factors, read once: streaming loads)
__device__ __noinline__ void adj_solves(Adj& s, const float* rec, int lane) {
  LANE_BOUND(lane);
  const float h = c_misc[0];
  for (int e = lane; e < NTRI; e += 32) s.L[e] = __ldcs(rec + OFF_LQ + e);
  for (int i = lane; i < NV; i += 32) s.y[i] = s.abar[i];
  tri_solve_packed(s.L, s.y, lane);
  for (int i = lane; i < NV; i += 32) { s.abar[i] -= h * c_kv[i] * s.y[i]; s.ny[i] = -s.y[i]; }
  __syncwarp();
  APROF(APROF_SOLVE_Q);
#ifdef DK_ADJ_DEBUG
  {  // |M a*| for the residual check's scale
    tri_mul_llt(s.L, s.a, s.tmp, s.did, lane);
    float acc = 0.f;
    for (int i = lane; i < NV; i += 32) { float x = s.did[i] - h * c_kv[i] * s.a[i]; acc += x * x; }
    acc = warp_sum(acc);
    if (lane == 0) s.ma_norm = sqrtf(acc);
    __syncwarp();
  }
#endif
  for (int e = lane; e < NTRI; e += 32) s.L[e] = __ldcs(rec + OFF_LH + e);
  for (int i = lane; i < NV; i += 32) s.lam[i] = s.abar[i];
  tri_solve_packed(s.L, s.lam, lane);
}

// lam on the dofs whose actuator force is not clamped (the record's mask; MuJoCo Warp's actuator_lam), 0 elsewhere
__device__ __forceinline__ float lam_unclamped(const Adj& s, int i) {
  return i >= 6 && !((s.clamp[i >> 5] >> (i & 31)) & 1u) ? s.lam[i] : 0.f;
}

#if DK_ADJ_CLAMP_KV == 2
// the soft clamp's lam~ (DK_ADJ_CLAMP_KV=2): lam on an unclamped dof (lam_unclamped's value),
// w lam on a clamped one (the record's mask), w = exp(-o / delta), o = max((|f| - tau_max) / tau_max, 0) the relative
// overshoot of the unclamped PD force f = kp (u - q) - kv v at the substep's input (the forward's expression; the
// model's clamp is per dof: jnt_actfrcrange of the dof's hinge, symmetric, one gear-1 actuator per hinge,
// source.constants asserts it), delta = s.soft. A dof clamped by the tape whose recomputed f rounds to within its
// bound gets o = 0, w = 1 (the derivative is continuous at the bound, so the FMA contraction of f does not matter as
// it would for the mask). delta <= 0, or w underflowing to 0, gives 0: the build with DK_ADJ_CLAMP_KV=1 bitwise
__device__ __forceinline__ float lam_soft(const Adj& s, int i) {
  if (i < 6) return 0.f;
  if (!((s.clamp[i >> 5] >> (i & 31)) & 1u)) return s.lam[i];
  if (!(s.soft > 0.f)) return 0.f;
  const float f = c_kp[i] * (s.u[i - 6] - s.q[i + 1]) - c_kv[i] * s.v[i], fmax = c_frclim[i];
  const float w = expf(-fmaxf((fabsf(f) - fmax) / fmax, 0.f) / s.soft);
  return w > 0.f ? w * s.lam[i] : 0.f;
}
#define LAM_ACT(s, i) lam_soft(s, i)
#else
#define LAM_ACT(s, i) lam_unclamped(s, i)
#endif

// A8: the actuator's part of P (P_q += kp lam~, P_v += kv lam (DK_ADJ_CLAMP_KV: kv lam~), u bar +=
// kp lam~; DK_ADJ_CLAMP_KV=2: lam~ of lam_soft), then the substep's input cotangents q bar = q bar_dir -
// dof_to_qpos(P_q; xi_k) (MuJoCo Warp's _dof_to_qpos: 2 xi (0, P_rot), xi not normalized) and v bar = v' bar - P_v
__device__ void adj_compose(Adj& s, int lane) {
  for (int i = 6 + lane; i < NV; i += 32) {
    float lt = LAM_ACT(s, i), kl = c_kp[i] * lt;
    s.Pq[i] += kl;
    s.Pv[i] += c_kv[i] * (DK_ADJ_CLAMP_KV ? lt : s.lam[i]);
    s.ubar[i - 6] += kl;
  }
  __syncwarp();
  for (int i = lane; i < NV; i += 32) {
    if (i < 3) s.qbar[i] -= s.Pq[i];
    else if (i >= 6) s.qbar[i + 1] -= s.Pq[i];
    s.vbar[i] -= s.Pv[i];
  }
  if (lane == 0) {
    float g[4] = {0.f, s.Pq[3], s.Pq[4], s.Pq[5]}, t[4];
    qmul(s.q + 3, g, t);
    for (int k = 0; k < 4; k++) s.qbar[3 + k] -= 2.f * t[k];
  }
  __syncwarp();
}

#if DK_ADJ_STAGE >= 2
// kbimp_d and efc_D_d of one row in a single non-inlined copy for the limit rows and the contacts (their solimp power
// branch inlines powf): out = (k, b, imp, dimp, D, dD) of the row with impedance parameters solref, solimp at pos and
// invweight invw, the forward's expressions (substep_pre). ADJ_SOLIMP_POWER (header(m): the model's rows share one
// power, 2 for mjlab's G1) replaces solimp[4] by that constant, the same value: kbimp's and kbimp_d's branches on the
// power fold at compile time and powf is not compiled for the powers 1 and 2 (1324 -> 310 SASS instructions)
__device__ __noinline__ void adj_row_impedance(const float* solref, const float* solimp, float pos, float invw,
                                               float* out) {
#ifdef ADJ_SOLIMP_POWER
  const float si[5] = {solimp[0], solimp[1], solimp[2], solimp[3], ADJ_SOLIMP_POWER};
  kbimp_d(solref, si, pos, c_misc[0], out, out + 1, out + 2, out + 3);
#else
  kbimp_d(solref, solimp, pos, c_misc[0], out, out + 1, out + 2, out + 3);
#endif
  efc_D_d(invw, out[2], out[3], out + 4, out + 5);
}

// A5 / A6, joint limits, one lane per dof: the hinge-limit rows replayed with the forward's test
// (substep_pre: one IEEE subtraction per side, so bitwise the forward's rows), row r = the number of limit rows of the
// dofs before (dof order: the first rows); a row on its quadratic zone (the tape's bit) adds limit_row_vjp's P_phi,
// P_v to its dof. A count other than the tape header's sets ADJ_REPLAY_MISMATCH. DBG (the debug build's pass by
// source, -DDK_ADJ_DEBUG): writes every dof's limit-row x, row force and kink to dbg. The production pass is the <false>
// instance in every build, so its code, and with it its rounding (ptxas fuses mul / add by context), is the same with
// and without -DDK_ADJ_DEBUG (as one function with a run-time dbg, 5 of a corpus's 4096 worlds differed by 1e-7)
template <bool DBG>
__device__ __noinline__ void adj_limits(Adj& s, int lane, float* dbg_in) {
  LANE_BOUND(lane);
  float* const dbg = DBG ? dbg_in : nullptr;
  (void)dbg;  // no debug entries without -DDK_ADJ_DEBUG
  const unsigned below = (1u << lane) - 1u;
  int nl = 0;
#pragma unroll 1
  for (int i0 = 0; i0 < NV; i0 += 32) {
    const int i = min(i0 + lane, NV - 1);  // clamped so that the loads need no branch (as the forward)
    float dmin, dmax;
    limit_dists(i, s.q[i + 1], &dmin, &dmax);  // the forward's test (step.py)
    const bool on = i0 + lane < NV && i >= 6 && c_jlim[i] != 0.f && fminf(dmin, dmax) < 0.f;
    const unsigned bal = __ballot_sync(0xffffffffu, on);
    if (on) {
      const int r = nl + __popc(bal & below);
      const float pos = fminf(dmin, dmax), sg = dmin < dmax ? 1.f : -1.f;  // lower limit: J = +e_i, upper: -e_i
      const bool act = (s.zone[r >> 5] >> (r & 31)) & 1u;
      float ri[6], pq, pv;
      adj_row_impedance(c_jsolref, c_jsolimp, pos, c_dinvw[i], ri);
      const float k = ri[0], b = ri[1], imp = ri[2], dimp = ri[3], D = ri[4], dD = ri[5];
      if (act) {
        limit_row_vjp(sg, pos, s.a[i], s.v[i], s.lam[i], k, b, imp, dimp, D, dD, &pq, &pv);
        s.Pq[i] += pq;
        s.Pv[i] += pv;
      }
#ifdef DK_ADJ_DEBUG
      if (dbg) {
        const float ja = sg * s.a[i], aref = -k * imp * pos - b * (sg * s.v[i]), x = ja - aref;
        dbg[DBG_LIM_X + i] = x;
        dbg[DBG_LIM_FORCE + i] = act ? -D * x : 0.f;
        dbg[DBG_LIM_KINK + i] = fabsf(x) / fmaxf(fabsf(ja) + fabsf(aref), 1e-30f);
      }
#endif
    }
#ifdef DK_ADJ_DEBUG
    else if (dbg && i0 + lane < NV) {
      dbg[DBG_LIM_X + i0 + lane] = 0.f; dbg[DBG_LIM_FORCE + i0 + lane] = 0.f; dbg[DBG_LIM_KINK + i0 + lane] = DBG_NO_ROW;
    }
#endif
    nl += __popc(bal);
  }
  if (lane == 0 && nl != (int)(((unsigned)s.hdr >> 16) & 255u)) s.flags |= (int)ADJ_REPLAY_MISMATCH;
  __syncwarp();
  APROF_COUNT(APROF_COUNT_LIMIT_ROWS, nl);
  APROF(APROF_LIMITS);
}

// A5 / A6 of contact c of the tape's table, (pair, slot) with the zone bits of its rows (bit e: row e on its quadratic
// zone), on its own lane: the replay (adj_pair_geom: dist, pos, frame bitwise the forward's), the rows' impedance with
// the forward's expressions (substep_pre: mu = the world's foot friction on a foot pair, the pyramid invweight for
// condim 3, kbimp_d / efc_D_d), the contact point's motions V_v, V_a, V_lam = sum over the support dofs of
// s_i col_i (v_i, a*_i, lam_i) and their angular parts Om_* = sum over the rotational ones of s_i z_i (.)_i (support:
// the dofs on exactly one of the two chains, s_i = +1 on body 2's; col_i = z_i x (p - x_i), x_i the origin of the dof's
// body, X_1 for the root rotations; e_k for the root translations), contact_psi_vjp, p bar = contact_pbar_motion, the
// primitive's VJP (adj_pair_geom with the cotangents) and geom_to_body_vjp of both geoms (the floor takes none). Writes
// cs[CS_N] (the lane's registers): W_v, W_a, W_lam, p, the geometry's (f bar, tau bar) of body 1 and of body 2.
// Returns the replayed distance. dbg (-DDK_ADJ_DEBUG, may be null): the contact's debug entries
__device__ __forceinline__ float adj_contact_one(Adj& s, int c, int pair, int slot, unsigned zone, float* cs,
                                                 float* dbg) {
  const int g1 = PG1(pair), g2 = PG2(pair), b1 = c_gbody[g1], b2 = c_gbody[g2], dim = PDIM(pair);
  float d, p[3], fr[9], nu;
  const int br = adj_pair_geom(pair, slot, s.gx, s.gq, &d, p, fr, &nu, nullptr, nullptr);
  const float mu = PFOOT(pair) ? s.fric : PMU(pair);
  float ri[6];
  adj_row_impedance(PSOLREF(pair), PSOLIMP(pair), d, contact_invw(pair, mu), ri);
  const float k = ri[0], b = ri[1], imp = ri[2], dimp = ri[3], D = ri[4], dD = ri[5];
  float Vv[3] = {0.f, 0.f, 0.f}, Va[3] = {0.f, 0.f, 0.f}, Vl[3] = {0.f, 0.f, 0.f};
  float Ov[3] = {0.f, 0.f, 0.f}, Oa[3] = {0.f, 0.f, 0.f}, Ol[3] = {0.f, 0.f, 0.f};
  const unsigned m1 = c_ancmask[b1], m2 = c_ancmask[b2], sup = m1 ^ m2;  // body bits on exactly one chain
  if ((sup >> 1) & 1u) {  // the root in the support (a floor contact): its translations, columns e_k
    const float sg = (m2 >> 1) & 1u ? 1.f : -1.f;
    for (int t = 0; t < 3; t++) { Vv[t] = sg * s.v[t]; Va[t] = sg * s.a[t]; Vl[t] = sg * s.lam[t]; }
  }
  // the support's rotational dofs in ascending order, the same terms in the same order as a loop over all dofs that
  // skips the others (a lane runs only its own, about half of the 32): the root's rotations 3, 4, 5 if the root is
  // in the support, then one hinge dof i = b + ADJ_HINGE_DOF0 per body bit b >= 2, lowest first (header(m) asserts it)
  unsigned hb = sup & ~3u;  // the support's hinge bodies (bit 0: the world, 1: the root)
  int i = (sup >> 1) & 1u ? 3 : (hb ? __ffs(hb) - 1 + ADJ_HINGE_DOF0 : NV);
#pragma unroll 1
  for (; i < NV; i = i < 5 ? i + 1 : (hb ? __ffs(hb) - 1 + ADJ_HINGE_DOF0 : NV)) {
    const int bi = i < 6 ? 1 : i - ADJ_HINGE_DOF0;
    if (i >= 6) hb &= hb - 1u;
    const float sg = (m2 >> bi) & 1u ? 1.f : -1.f, vi = sg * s.v[i], ai = sg * s.a[i], li = sg * s.lam[i];
    const float* z = i < 6 ? s.zr[i - 3] : s.z[bi];
    float r[3], col[3];
    for (int t = 0; t < 3; t++) r[t] = p[t] - s.X[bi][t];
    cross(z, r, col);
    for (int t = 0; t < 3; t++) {
      Vv[t] += vi * col[t]; Va[t] += ai * col[t]; Vl[t] += li * col[t];
      Ov[t] += vi * z[t]; Oa[t] += ai * z[t]; Ol[t] += li * z[t];
    }
  }
  float Wv[3], Wa[3], Wl[3], cot[13], fg[12];  // cot = (d bar, p bar, frame bar): adj_contact_geom's cotangent
  contact_psi_vjp(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4);
  cot[1] = cot[2] = cot[3] = 0.f;
  contact_pbar_motion(Wv, Wa, Wl, Ov, Oa, Ol, cot + 1);
  for (int t = 0; t < 12; t++) fg[t] = 0.f;
  if (zone) {  // the primitive's VJP (it replays again) and the geoms' pose cotangents on their bodies
    float d2, p2[3], fr2[9], nu2, gb[12];
    adj_pair_geom(pair, slot, s.gx, s.gq, &d2, p2, fr2, &nu2, cot, gb);
    if (b1) geom_to_body_vjp(s.gx[g1], s.X[b1], gb, gb + 3, fg, fg + 3);
    geom_to_body_vjp(s.gx[g2], s.X[b2], gb + 6, gb + 9, fg + 6, fg + 9);
  }
  for (int t = 0; t < 3; t++) {
    cs[t] = Wv[t]; cs[3 + t] = Wa[t]; cs[6 + t] = Wl[t]; cs[9 + t] = p[t];
  }
  for (int t = 0; t < 12; t++) cs[12 + t] = fg[t];
#ifdef DK_ADJ_DEBUG
  if (dbg) {  // replay, cotangents, rows (x, kink, force) and the state groups' diagnostics (reference.adjoint_groups)
    dbg[DBG_CON_DIST + c] = d;
    dbg[DBG_CON_DIST_BAR + c] = cot[0];
    dbg[DBG_CON_BR + c] = (float)br;
    for (int t = 0; t < 3; t++) { dbg[DBG_CON_POS + 3 * c + t] = p[t]; dbg[DBG_CON_POS_BAR + 3 * c + t] = cot[1 + t]; }
    for (int t = 0; t < 9; t++) dbg[DBG_CON_FRAME_BAR + 9 * c + t] = cot[4 + t];
    float vf[3], af[3], F[3] = {0.f, 0.f, 0.f}, kink = DBG_NO_ROW;
    for (int j = 0; j < 3; j++) { vf[j] = dot3(fr + 3 * j, Vv); af[j] = dot3(fr + 3 * j, Va); }
    for (int e = 0; e < (dim == 3 ? 4 : 1); e++) {
      const int t = 1 + (e >> 1);
      const float sm = dim == 3 ? ((e & 1) ? -mu : mu) : 0.f;
      const float uv = vf[0] + sm * vf[t], ua = af[0] + sm * af[t], aref = -k * imp * d - b * uv, x = ua - aref;
      dbg[DBG_CON_X + 4 * c + e] = x;
      kink = fminf(kink, fabsf(x) / fmaxf(fabsf(ua) + fabsf(aref), 1e-30f));
      if ((zone >> e) & 1u)
        for (int q = 0; q < 3; q++) F[q] += -D * x * (fr[q] + sm * fr[3 * t + q]);
    }
    for (int q = 0; q < 3; q++) dbg[DBG_CON_FORCE + 3 * c + q] = F[q];
    dbg[DBG_CON_KINK + c] = kink;
    float basis = DBG_NO_ROW, sfree = DBG_NO_ROW, a1[3], a2[3], w[3];
    const int t1 = c_gtype[g1], t2 = c_gtype[g2];
    zaxis_rn(s.gq[g1], a1);
    zaxis_rn(s.gq[g2], a2);
    if (t1 != 0) basis = fabsf(fabsf(fr[1]) - 0.5f);  // self contact: make_frame's basis test on n_y
    else if (t2 == 2) {  // plane-capsule: the projected axis length beta against 0.5
      const float na = dot3(fr, a2);
      for (int q = 0; q < 3; q++) w[q] = a2[q] - fr[q] * na;
      basis = fabsf(sqrtf(dot3(w, w)) - 0.5f);
    }
    if (t1 == 2 && t2 == 2 && (br & ADJ_BR_FREE1) && (br & ADJ_BR_FREE2)) {
      cross(a1, a2, w);
      sfree = sqrtf(dot3(w, w));
    }
    dbg[DBG_CON_BASIS + c] = basis;
    dbg[DBG_CON_SIN_FREE + c] = sfree;
  }
#endif
  (void)nu;
  return d;
}

// A5 / A6, contacts: the collision geoms' world poses from the body poses (geom_poses of the forward,
// so the replay is bitwise its collision) and the root's rotation axes, then the tape's contacts in batches of 32 with
// one lane per contact (adj_contact_one into the lane's registers; row offsets as substep_pre: the limit rows first,
// then per contact 4 pyramid rows (condim 3) or 1, rows past NEFC dropped), and one lane per body that sums the batch
// in contact order, each contact's words broadcast from its lane by shuffles (no shared scratch, 3200 B per world;
// the body-1 words only for a self contact; the batch loops are uniform for the compiler, see below): for each contact
// whose support holds its dof, G_i = s_i (v_i W_v + a*_i W_a + lam_i W_lam), P_v,i += s_i col_i . W_v,
// f bar += z_i x G_i, tau bar += z_i x ((p - x_i) x G_i) (contact_dof_vjp; lane 1: the root translations and rotation
// dof 3, lanes 0 and 31: rotation dofs 4 and 5, handed to lane 1 after the batch), and the geometry's pose cotangents
// of its body. Adds to s.fb, s.tb (bodies) and s.Pv (dofs): a fixed order, independent of scheduling. DK_ADJ_STAGE 2:
// plane contacts only (records with self contacts set ADJ_ROWS_SKIPPED). DBG: the debug build's pass by source, which
// writes every contact's debug entries (also without a quadratic row) to dbg; production: <false> (see adj_limits)
template <bool DBG>
__device__ __noinline__ void adj_contacts(Adj& s, int lane, float* dbg_in) {
  LANE_BOUND(lane);
  float* const dbg = DBG ? dbg_in : nullptr;
  (void)dbg;  // no debug entries without -DDK_ADJ_DEBUG
  // ncon through votes (uniform_int, step.py), and UNIFORM for the per-contact branches: the batch loops and the
  // shuffles in them are then uniform for the compiler (no divergence checks; see step.py UNIFORM). Without the votes
  // the shuffles cost 5% of k_adjoint (RTX 3090, at 6 worlds per SM)
  const int ncon = uniform_int(s.hdr & 255, 8), nlim = ((unsigned)s.hdr >> 16) & 255;
  for (int g = lane; g < NGEOMC; g += 32) {  // geom_poses (step.py)
    const int b = c_gbody[g];
    geom_pose(g, s.Q[b], s.X[b], s.gx[g], s.gq[g]);
  }
  if (lane < 3) {
    const float e[3] = {lane == 0 ? 1.f : 0.f, lane == 1 ? 1.f : 0.f, lane == 2 ? 1.f : 0.f};
    qrot(s.Q[1], e, s.zr[lane]);
  }
#ifdef DK_ADJ_DEBUG
  if (dbg) {
    for (int g = lane; g < NGEOMC; g += 32)
      for (int k = 0; k < 4; k++) {
        if (k < 3) dbg[DBG_GEOM_X + 3 * g + k] = s.gx[g][k];
        dbg[DBG_GEOM_Q + 4 * g + k] = s.gq[g][k];
      }
    for (int b = lane; b < NB; b += 32)
      for (int k = 0; k < 4; k++) {
        if (k < 3) dbg[DBG_BODY_X + 3 * b + k] = s.X[b][k];
        dbg[DBG_BODY_Q + 4 * b + k] = s.Q[b][k];
      }
  }
#endif
  __syncwarp();
  APROF_COUNT(APROF_COUNT_CONTACTS, ncon);
  APROF(APROF_CONTACT_SETUP);
  const unsigned below = (1u << lane) - 1u;
  const bool body = lane >= 1 && lane < NB, rot = lane == 0 || lane == 31;
  const int rb = body ? lane : 1, rk = lane == 0 ? 1 : (lane == 31 ? 2 : 0), di = rb >= 2 ? c_dadr[rb] : 3 + rk;
  int ntri = 0;  // condim-3 contacts before the batch
  for (int c0 = 0; c0 < ncon; c0 += 32) {
    const int c = c0 + lane, nb = min(32, ncon - c0);
    const bool on = c < ncon;
    const unsigned e = on ? (s.con[c >> 1] >> (16 * (c & 1))) & 0xFFFFu : 0u;
    const int pair = e & 511u, slot = e >> 9, dim = on ? PDIM(pair) : 0;
    const unsigned b3 = __ballot_sync(0xffffffffu, dim == 3);
    const int r0 = nlim + c + 3 * (ntri + __popc(b3 & below));  // the contact's first row
    ntri += __popc(b3);
    const bool use = on && (DK_ADJ_STAGE >= 3 || c_gbody[PG1(pair)] == 0);
    unsigned zone = 0u;
    if (use && r0 < NEFC) {
      const int nr = min(dim == 3 ? 4 : 1, NEFC - r0), w = r0 >> 5;
      unsigned long long zz = s.zone[w];
      if (w + 1 < (NEFC + 31) / 32) zz |= (unsigned long long)s.zone[w + 1] << 32;
      zone = (unsigned)(zz >> (r0 & 31)) & ((1u << nr) - 1u);
    }
    bool bad = false;
    int cb = -1;  // body 1 | body 2 << 8 of a contact with a quadratic row, -1: nothing to add
    float cw[CS_N];  // the lane's contact (adj_contact_one), read by the body lanes by shuffles
#pragma unroll
    for (int k = 0; k < CS_N; k++) cw[k] = 0.f;
    if (use && (zone || dbg)) {  // a contact without a quadratic row adds nothing (MuJoCo Warp's _contact_any_active)
      bad = !(adj_contact_one(s, c, pair, slot, zone, cw, dbg) < 0.f);
      if (zone) cb = c_gbody[PG1(pair)] | c_gbody[PG2(pair)] << 8;
    }
    const unsigned skip = __ballot_sync(0xffffffffu, on && !use), mis = __ballot_sync(0xffffffffu, bad);
    if (lane == 0) s.flags |= (skip ? (int)ADJ_ROWS_SKIPPED : 0) | (mis ? (int)ADJ_REPLAY_MISMATCH : 0);
    __syncwarp();
    APROF_COUNT(APROF_COUNT_CONTACTS_ACTIVE, __popc(__ballot_sync(0xffffffffu, cb >= 0)));
    APROF(APROF_CONTACT_ROWS);
    float f[3] = {0.f, 0.f, 0.f}, t[3] = {0.f, 0.f, 0.f}, pt[3] = {0.f, 0.f, 0.f}, pv = 0.f;
#pragma unroll 1
    for (int j = 0; j < nb; j++) {  // the batch in contact order
      const int cj = __shfl_sync(0xffffffffu, cb, j);
      if (UNIFORM(cj < 0)) continue;
      const int c1 = cj & 255, c2 = cj >> 8;
      float w[CS_N];
#pragma unroll
      for (int k = 0; k < 12; k++) w[k] = __shfl_sync(0xffffffffu, cw[k], j);
      const unsigned i2 = (c_ancmask[c2] >> rb) & 1u, i1 = (c_ancmask[c1] >> rb) & 1u;
      if ((body || rot) && i1 != i2) {  // this lane's dof in the support (contact_dof_vjp)
        const float sg = i2 ? 1.f : -1.f, vi = sg * s.v[di], ai = sg * s.a[di], li = sg * s.lam[di];
        const float* z = rb >= 2 ? s.z[rb] : s.zr[rk];
        float G[3], r[3], wv[3], col[3], u[3], x[3], y[3];
        for (int k = 0; k < 3; k++) {
          wv[k] = w[k];
          G[k] = vi * wv[k] + ai * w[3 + k] + li * w[6 + k];
          r[k] = w[9 + k] - s.X[rb][k];
        }
        cross(z, r, col);
        pv += sg * dot3(col, wv);
        cross(z, G, u);
        cross(r, G, x);
        cross(z, x, y);
        for (int k = 0; k < 3; k++) { f[k] += u[k]; t[k] += y[k]; }
        if (lane == 1)
          for (int k = 0; k < 3; k++) pt[k] += sg * wv[k];  // the root translations: columns e_k, no pose term
      }
      if (UNIFORM(c1 != 0)) {  // the floor, body 0, takes no pose cotangent
#pragma unroll
        for (int k = 12; k < 18; k++) w[k] = __shfl_sync(0xffffffffu, cw[k], j);
        if (body && rb == c1)
          for (int k = 0; k < 3; k++) { f[k] += w[12 + k]; t[k] += w[15 + k]; }
      }
#pragma unroll
      for (int k = 18; k < 24; k++) w[k] = __shfl_sync(0xffffffffu, cw[k], j);
      if (body && rb == c2)
        for (int k = 0; k < 3; k++) { f[k] += w[18 + k]; t[k] += w[21 + k]; }
    }
    float o[7] = {f[0], f[1], f[2], t[0], t[1], t[2], pv}, o0[7], o31[7];  // root rotations 4, 5: to lane 1
    for (int k = 0; k < 7; k++) { o0[k] = __shfl_sync(0xffffffffu, o[k], 0); o31[k] = __shfl_sync(0xffffffffu, o[k], 31); }
    if (body) {
      if (lane == 1) {
        for (int k = 0; k < 3; k++) {
          f[k] += o0[k] + o31[k]; t[k] += o0[3 + k] + o31[3 + k]; s.Pv[k] += pt[k];
        }
        s.Pv[4] += o0[6];
        s.Pv[5] += o31[6];
      }
      for (int k = 0; k < 3; k++) { s.fb[rb][k] += f[k]; s.tb[rb][k] += t[k]; }
      s.Pv[di] += pv;
    }
    __syncwarp();
    APROF(APROF_CONTACT_REDUCE);
  }
}
#endif

#ifdef DK_ADJ_DEBUG
__device__ void dbg_put(float* dst, const float* src, int n, float sg, int lane) {
  for (int i = lane; i < n; i += 32) dst[i] = sg * src[i];
}
// dst = sg dof_to_qpos(P; xi) (NQ words)
__device__ void dbg_qpos(float* dst, const float* P, const float* q, float sg, int lane) {
  for (int i = lane; i < NV; i += 32) {
    if (i < 3) dst[i] = sg * P[i];
    else if (i >= 6) dst[i + 1] = sg * P[i];
  }
  if (lane == 0) {
    float g[4] = {0.f, P[3], P[4], P[5]}, t[4];
    qmul(q + 3, g, t);
    for (int k = 0; k < 4; k++) dst[3 + k] = sg * 2.f * t[k];
  }
}
// the built-in check of the solves: r(a*) = ID(q, v, a*) + arm a* - tau - J'f from the bodies' channel-1 forces right
// after A4 (bVl = F1, bWl = N1), summed over the subtrees as the forward does; tau from ctrl, J'f = s.jtf (adj_jtf, the
// replayed rows' forces; 0 with DK_ADJ_STAGE 1)
__device__ void adj_residual(Adj& s, const float* ctrl, float* dbg, int lane) {
  const int b = lane, d = b < NB ? c_depth[b] : -1;
  if (b >= 1 && b < NB)
    for (int k = 0; k < 3; k++) { s.dF[b][k] = s.bVl[b][k]; s.dN[b][k] = s.bWl[b][k]; }
  __syncwarp();
  for (int L = MAXD - 1; L >= 1; L--) {
    if (d == L)
      for (int i = 0; i < c_nchild[b]; i++) {
        int c = c_child[3 * b + i];
        accumulate(s.X[b], s.X[c], s.dF[c], s.dN[c], s.dF[b], s.dN[b]);
      }
    __syncwarp();
  }
  if (b >= 1 && b < NB) torques(b, s.Q[b], s.dF[b], s.dN[b], s.did);
  __syncwarp();
  float rr = 0.f, tt = 0.f, ii = 0.f, ff = 0.f;
  for (int i = lane; i < NV; i += 32) {
    float tau = 0.f;
    if (i >= 6 && ctrl) {
      tau = c_kp[i] * (ctrl[i - 6] - s.q[i + 1]) - c_kv[i] * s.v[i];
      tau = fminf(fmaxf(tau, -c_frclim[i]), c_frclim[i]);
    }
    float r = s.did[i] + c_armature[i] * s.a[i] - tau - s.jtf[i];
    dbg[DBG_RESID + i] = r;
    rr += r * r; tt += tau * tau; ii += s.did[i] * s.did[i]; ff += s.jtf[i] * s.jtf[i];
  }
  rr = warp_sum(rr); tt = warp_sum(tt); ii = warp_sum(ii); ff = warp_sum(ff);
  if (lane == 0) {
    dbg[DBG_RESID_NORM] = sqrtf(rr); dbg[DBG_TAU_NORM] = sqrtf(tt); dbg[DBG_ID_NORM] = sqrtf(ii);
    dbg[DBG_MA_NORM] = s.ma_norm; dbg[DBG_JTF_NORM] = sqrtf(ff);
  }
  __syncwarp();
}
#if DK_ADJ_STAGE >= 2
// J'f of the replayed rows for the residual check, one lane per dof, from the debug entries of adj_limits and
// adj_contacts (written by other lanes before the caller's __syncwarp): the limit row's sigma f, and s_i col_i . F_c
// of every contact c whose support holds the dof (F_c: its force sum_e -D x_e u_e at its point)
__device__ void adj_jtf(Adj& s, const float* dbg, int lane) {
  const int ncon = s.hdr & 255;
  for (int i = lane; i < NV; i += 32) {
    const int bi = c_dofbody[i];
    float acc = 0.f;
    if (i >= 6) {
      const float q = s.q[i + 1];
      acc = (q - c_jrange[2 * i] < c_jrange[2 * i + 1] - q ? 1.f : -1.f) * dbg[DBG_LIM_FORCE + i];
    }
    for (int c = 0; c < ncon; c++) {
      const int pair = (s.con[c >> 1] >> (16 * (c & 1))) & 511u;
      const unsigned m1 = c_ancmask[c_gbody[PG1(pair)]], m2 = c_ancmask[c_gbody[PG2(pair)]];
      if (!(((m1 ^ m2) >> bi) & 1u)) continue;
      const float sg = (m2 >> bi) & 1u ? 1.f : -1.f, *F = dbg + DBG_CON_FORCE + 3 * c, *p = dbg + DBG_CON_POS + 3 * c;
      if (i < 3) { acc += sg * F[i]; continue; }
      const float* z = i < 6 ? s.zr[i - 3] : s.z[bi];
      float r[3], col[3];
      for (int t = 0; t < 3; t++) r[t] = p[t] - s.X[bi][t];
      cross(z, r, col);
      acc += sg * dot3(col, F);
    }
    s.jtf[i] = acc;
  }
  __syncwarp();
}
#endif
#endif

// the world body (at rest), and the debug build's zero vector; the caller syncs
__device__ void adj_init(Adj& s, int lane) {
#ifdef DK_ADJ_DEBUG
  for (int i = lane; i < NV; i += 32) { s.zero[i] = 0.f; s.jtf[i] = 0.f; }  // DK_ADJ_STAGE 1 has no rows
#endif
  if (lane == 0) { s.Q[0][0] = 1.f; s.Q[0][1] = s.Q[0][2] = s.Q[0][3] = 0.f; s.flags = 0; }
  if (lane < 3) {
    s.X[0][lane] = s.z[0][lane] = s.W[0][lane] = s.V[0][lane] = s.AL[0][lane] = s.A[0][lane] = 0.f;
    s.AL2[0][lane] = s.A2[0][lane] = s.Wl[0][lane] = s.Vl[0][lane] = s.Wy[0][lane] = s.Vy[0][lane] = 0.f;
  }
}

// one substep's VJP (A0-A8): s.qbar, s.vbar hold the cotangents of its output state and become those of its input;
// s.ubar accumulates. rec: its tape record. dbg, ctrl: -DDK_ADJ_DEBUG (this world's debug record of the substep, the
// world's ctrl); null otherwise
__device__ void adj_substep(Adj& s, const float* rec, int lane, float* dbg, const float* ctrl) {
  // A0
  for (int i = lane; i < NQ; i += 32) s.q[i] = __ldcs(rec + OFF_QPOS + i);
  for (int i = lane; i < NV; i += 32) {
    float a = __ldcs(rec + OFF_QACC + i), aadv = __ldcs(rec + OFF_QADV + i);
    s.v[i] = __ldcs(rec + OFF_QVEL + i); s.a[i] = a; s.aadv[i] = aadv; s.at[i] = a - aadv;
  }
  if (lane < (NV + 31) / 32) s.clamp[lane] = __float_as_uint(__ldcs(rec + OFF_CLAMP + lane));
  if (lane < (NEFC + 31) / 32) s.zone[lane] = __float_as_uint(__ldcs(rec + OFF_ZONE + lane));
  for (int i = lane; i < (MAXCON + 1) / 2; i += 32) s.con[i] = __float_as_uint(__ldcs(rec + OFF_CON + i));
  if (lane == 0) {
    unsigned hd = __float_as_uint(__ldcs(rec + OFF_HDR));
    s.hdr = (int)hd;
    s.flags |= (int)__float_as_uint(__ldcs(rec + OFF_FLAGS));
    if (DK_ADJ_STAGE < 2 && (hd & 0xFFFFu)) s.flags |= ADJ_ROWS_SKIPPED;  // contacts (ncon) or rows (nefc)
  }
  __syncwarp();
  APROF(APROF_RECORD);
  adj_poses(s, lane);  // A3: poses and channel 1
  APROF(APROF_POSES);
#ifdef DK_ADJ_DEBUG
  if (dbg) { dbg_put(dbg + DBG_G_QPOS_OUT, s.qbar, NQ, 1.f, lane); dbg_put(dbg + DBG_G_QVEL_OUT, s.vbar, NV, 1.f, lane); }
  __syncwarp();  // lane 0 of adj_integrator overwrites qbar[3..6], vbar[3..5] that other lanes just read
#endif
  adj_integrator(s, lane);  // A1
  APROF(APROF_INTEGRATOR);
#ifdef DK_ADJ_DEBUG
  if (dbg) {
    dbg_put(dbg + DBG_ADJ_QPOS_INT, s.qbar, NQ, 1.f, lane); dbg_put(dbg + DBG_ADJ_QVEL_INT, s.vbar, NV, 1.f, lane);
    dbg_put(dbg + DBG_ADJ_QACC_ADV, s.abar, NV, 1.f, lane);
  }
#endif
  adj_solves(s, rec, lane);  // A1, A2 (marks solve_Q inside)
  APROF(APROF_SOLVE_H);
#ifdef DK_ADJ_DEBUG
  if (dbg) {  // the split of P by source: the RNE's channel 1 (lambda) and 2 (-y) alone, then the actuator
    dbg_put(dbg + DBG_Y, s.y, NV, 1.f, lane); dbg_put(dbg + DBG_ADJ_QACC, s.abar, NV, 1.f, lane);
    dbg_put(dbg + DBG_LAM, s.lam, NV, 1.f, lane); dbg_put(dbg + DBG_W_DIR, s.at, NV, 1.f, lane);
#if DK_ADJ_STAGE >= 2
    // the constraint rows alone: the limit rows (direct P terms), then the contacts (their pose cotangents through the
    // pose sweep, with zero RNE seeds: adj_tree_down then leaves f bar = tau bar = 0), and their forces J'f
    adj_tree_down(s, s.zero, s.zero, lane);
    for (int i = lane; i < NV; i += 32) { s.Pq[i] = 0.f; s.Pv[i] = 0.f; }
    __syncwarp();
    adj_limits<true>(s, lane, dbg);
    dbg_qpos(dbg + DBG_RES_QPOS_LIMIT, s.Pq, s.q, 1.f, lane); dbg_put(dbg + DBG_RES_QVEL_LIMIT, s.Pv, NV, 1.f, lane);
    __syncwarp();
    for (int i = lane; i < NV; i += 32) { s.Pq[i] = 0.f; s.Pv[i] = 0.f; }
    __syncwarp();
    adj_contacts<true>(s, lane, dbg);
    adj_tree_up(s, s.zero, s.zero, lane);
    dbg_qpos(dbg + DBG_RES_QPOS_CONTACT, s.Pq, s.q, 1.f, lane);
    dbg_put(dbg + DBG_RES_QVEL_CONTACT, s.Pv, NV, 1.f, lane);
    __syncwarp();
    adj_jtf(s, dbg, lane);
    for (int i = lane; i < NV; i += 32) { s.Pq[i] = 0.f; s.Pv[i] = 0.f; }
    __syncwarp();
#endif
    adj_tree_down(s, s.lam, s.zero, lane);
    adj_residual(s, ctrl, dbg, lane);
    adj_tree_up(s, s.lam, s.zero, lane);
    dbg_qpos(dbg + DBG_RES_QPOS_RNE, s.Pq, s.q, 1.f, lane); dbg_put(dbg + DBG_RES_QVEL_RNE, s.Pv, NV, 1.f, lane);
    __syncwarp();
    for (int i = lane; i < NV; i += 32) { s.Pq[i] = 0.f; s.Pv[i] = 0.f; }
    adj_tree_down(s, s.zero, s.ny, lane);
    adj_tree_up(s, s.zero, s.ny, lane);
    dbg_qpos(dbg + DBG_ADJ_QPOS_M, s.Pq, s.q, -1.f, lane);  // P of the seed -y is -grad_q[y' M a~]
    __syncwarp();
    for (int i = lane; i < NV; i += 32) {
      float lt = LAM_ACT(s, i);
      s.tmp[i] = c_kp[i] * lt;
      dbg[DBG_RES_QVEL_ACT + i] = i >= 6 ? c_kv[i] * (DK_ADJ_CLAMP_KV ? lt : s.lam[i]) : 0.f;
      if (i >= 6) dbg[DBG_GRAD_CTRL_K + i - 6] = c_kp[i] * lt;
      s.Pq[i] = 0.f; s.Pv[i] = 0.f;
    }
    __syncwarp();
    dbg_qpos(dbg + DBG_RES_QPOS_ACT, s.tmp, s.q, 1.f, lane);
    __syncwarp();
  }
#endif
  adj_tree_down(s, s.lam, s.ny, lane);  // A3 (channel 2, twists), A4
  APROF(APROF_TREE_DOWN);
#if DK_ADJ_STAGE >= 2
  adj_limits<false>(s, lane, nullptr);  // A5, A6: P_q, P_v of the limit rows' dofs (marks limits)
  adj_contacts<false>(s, lane, nullptr);  // A5, A6: the contacts' pose cotangents s.fb / s.tb, P_v of the support dofs
                                          // (marks contact_setup, then contact_rows and contact_reduce per batch)
#endif
  adj_tree_up(s, s.lam, s.ny, lane);  // A7
  APROF(APROF_TREE_UP);
  adj_compose(s, lane);  // A8
  APROF(APROF_COMPOSE);
#ifdef DK_ADJ_DEBUG
  if (dbg) { dbg_put(dbg + DBG_GRAD_QPOS_IN, s.qbar, NQ, 1.f, lane); dbg_put(dbg + DBG_GRAD_QVEL_IN, s.vbar, NV, 1.f, lane); }
  __syncwarp();
#endif
}
"""

ADJ_KERNEL = r"""
#ifdef DK_PROFILE
// PROF_PHASES summed over the worlds of every launch since it was last zeroed (zero at load): the host zeroes it
// (cudaMemcpyToSymbol) before the launch it profiles and reads it right after (the adjoint check binary, which also
// prints the launches since the zero; nablaengine.checks.run rejects a reading that is not of one launch)
__device__ unsigned long long g_aprof[NAPROF];
#endif
// the adjoint of one control step (nsub substeps) per world: tape = nw * nsub records (record w * nsub + k), gq_out /
// gv_out the cotangents of the step's output qpos / qvel, gq_in / gv_in / gctrl the gradients of its input qpos, qvel
// and ctrl (summed over the substeps), info the union of the records' flags, ADJ_ROWS_SKIPPED and ADJ_REPLAY_MISMATCH.
// fric: the worlds' foot friction (the foot pairs' mu, as the forward). dbg, ctrl: -DDK_ADJ_DEBUG (nw * nsub records of
// DBG_WORDS, ctrl for the residual check), may be null. DK_ADJ_CLAMP_KV=2: ctrl (nw * nu, the control step's) is
// required, and clamp_soft (ADJ_SOFT_PARAM) is the soft clamp's delta
__global__ void k_adjoint(int nw, int nsub, const float* tape, const float* fric, const float* com, const float* gq_out,
                          const float* gv_out, float* gq_in, float* gv_in, float* gctrl, int* info, float* dbg,
                          const float* ctrl ADJ_SOFT_PARAM) {
  extern __shared__ __align__(16) unsigned char dk_smem[];  // dynamic, as the forward
  Adj& s = *reinterpret_cast<Adj*>(dk_smem);
  const int lane = threadIdx.x & 31, w = blockIdx.x;  // (& 31: see host.py k_substep)
  if (w >= nw) return;
#ifdef DK_PROFILE
  if (lane < NAPROF) s.aprof[lane] = 0;
#endif
  APROF_START();
  adj_init(s, lane);
  for (int i = lane; i < NQ; i += 32) s.qbar[i] = gq_out[(size_t)w * NQ + i];
  for (int i = lane; i < NV; i += 32) s.vbar[i] = gv_out[(size_t)w * NV + i];
  for (int i = lane; i < NV - 6; i += 32) s.ubar[i] = 0.f;
  if (lane < 3) s.com[lane] = com[w * 3 + lane];
  if (lane == 0) s.fric = fric ? fric[w] : 0.f;
#if DK_ADJ_CLAMP_KV == 2
  for (int i = lane; i < NV - 6; i += 32) s.u[i] = ctrl[(size_t)w * (NV - 6) + i];
  if (lane == 0) s.soft = clamp_soft;
#endif
  __syncwarp();
  APROF(APROF_IO);
  for (int k = nsub - 1; k >= 0; k--) {
    const size_t r = (size_t)w * nsub + k;
    adj_substep(s, tape + r * REC_WORDS, lane, dbg ? dbg + r * DBG_WORDS : nullptr,
                ctrl ? ctrl + (size_t)w * (NV - 6) : nullptr);
  }
  for (int i = lane; i < NQ; i += 32) gq_in[(size_t)w * NQ + i] = s.qbar[i];
  for (int i = lane; i < NV; i += 32) gv_in[(size_t)w * NV + i] = s.vbar[i];
  for (int i = lane; i < NV - 6; i += 32) gctrl[(size_t)w * (NV - 6) + i] = s.ubar[i];
  if (lane == 0) info[w] = s.flags;
#ifdef DK_PROFILE
  APROF(APROF_IO);
  if (lane < NAPROF) atomicAdd(&g_aprof[lane], (unsigned long long)s.aprof[lane]);
#endif
}

#ifndef DK_ADJ_WORLDS_PER_SM
#define DK_ADJ_WORLDS_PER_SM -1  // -1: adj_worlds_per_sm()'s choice for the GPU; 0: as many as fit; N > 0: at most N
#endif
// host: the worlds-per-SM cap of a k_adjoint launch on the current device: DK_ADJ_WORLDS_PER_SM if >= 0, else the
// choice for the GPU. V100 (sm_70): 4 (at 15740 B per world, when 6 fitted: 6 -> 4 gave 5.1 -> 4.5 ms per control
// step; with 5 or 6 worlds some warps share a scheduler and slow down; at 10108 B 9 fit and 8 / 9 run 6-11% faster
// than 4, not adopted: V100 is not a target GPU); the RTX 3090 (sm_86) gains up to
// the 9 that fit, so no cap elsewhere (0). -1 if the device cannot be queried, with the CUDA error in *err
// (cudaSuccess otherwise; err may be null)
static inline int adj_worlds_per_sm(cudaError_t* err = nullptr) {
  cudaError_t e = cudaSuccess;
  int dev = 0, major = 0, minor = 0;
  if (DK_ADJ_WORLDS_PER_SM < 0) {
    e = cudaGetDevice(&dev);
    if (e == cudaSuccess) e = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (e == cudaSuccess) e = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  }
  if (err) *err = e;
  if (e != cudaSuccess) return -1;
  if (DK_ADJ_WORLDS_PER_SM >= 0) return DK_ADJ_WORLDS_PER_SM;
  return major == 7 && minor == 0 ? 4 : 0;
}
// host: the dynamic shared memory of a k_adjoint launch on the current device; also sets k_adjoint's opt-in to it on
// that device (cudaFuncSetAttribute: once per device is enough, e.g. with the constant upload, and not per launch).
// worlds_per_sm: -1 (default) adj_worlds_per_sm()'s choice, 0 no cap (sizeof(Adj)), N > 0 at most N worlds per SM:
// (shared memory per SM / N - 1 KB) rounded down to 256 B (1 KB: the per-block reservation of sm_80 and later; 256 B:
// the allocation granularity, 128 B on sm_86), more than the kernel needs (the rest of the SM's shared memory is left
// unused: limiting the occupancy by the carveout instead, with a larger L1, measured slower on V100), in signed
// arithmetic and never below sizeof(Adj) (a cap above what fits is no cap) nor above the device's per-block opt-in
// limit. Returns 0 on failure, with the CUDA error in *err (cudaSuccess otherwise; err may be null): no current device,
// a refused attribute query or opt-in (e.g. sizeof(Adj) above the opt-in limit), or worlds_per_sm < -1
// (cudaErrorInvalidValue). Callers must check for 0: a launch with 0 bytes is not rejected, the kernel would use shared
// memory it was not given. Launch: k_adjoint<<<nw, 32, smem, stream>>>(...)
static inline size_t adj_launch_smem(int worlds_per_sm = -1, cudaError_t* err = nullptr) {
  cudaError_t e = cudaSuccess;
  if (worlds_per_sm == -1) worlds_per_sm = adj_worlds_per_sm(&e);
  else if (worlds_per_sm < -1) e = cudaErrorInvalidValue;
  long long smem = (long long)sizeof(Adj);
  int dev = 0, optin = 0, per_sm = 0;
  if (e == cudaSuccess) e = cudaGetDevice(&dev);
  if (e == cudaSuccess) e = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (e == cudaSuccess && worlds_per_sm > 0) {
    e = cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    long long want = ((long long)per_sm / worlds_per_sm - 1024) / 256 * 256;  // < 0 for large worlds_per_sm
    if (want > optin) want = optin;
    if (e == cudaSuccess && want > smem) smem = want;
  }
  if (e == cudaSuccess) e = cudaFuncSetAttribute(k_adjoint, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  if (err) *err = e;
  return e == cudaSuccess ? (size_t)smem : 0;
}
"""


def cuda_source(m, max_contacts=None):
    """The device code of the forward and the adjoint as one compilation unit, with k_adjoint's launch helpers
    (adj_launch_smem; no __global__ forward kernel, no host main)."""
    maxd = source.constants(m)[1]
    return (source.header(m, maxd, 1) + step.EXTRA_HEADER + step.step_header(m, max_contacts)
            + tape.header(m, max_contacts) + header(m) + source.DEVICE + step.STEP_DEVICE + step.STEP_KERNEL
            + adjoint_contact.source(m) + ADJ_DEVICE + ADJ_KERNEL)


def engine_constants(m, foot_pairs):
    """(constants as one float32 array, [(symbol, kind, count, width)] in packing order, sizes) for the host loader
    (scripts/engine_step_check.py loader()), as engine_step_check packs them."""
    c1, _ = source.constants(m)
    c2 = step.pair_constants(m, foot_pairs)
    dofbody, mask = step.extra_constants(m)
    halves = np.stack([mask & 0xFFFF, mask >> 16], 1).ravel()
    consts = np.concatenate([c1, c2, dofbody.astype(np.float32), halves.astype(np.float32)])
    sizes = {"nb": m.nbody, "nv": m.nv, "one": 1, "nv2": 2 * m.nv}
    lays = (list(source.CONST_LAYOUT) + [(s, k, n, 1) for s, k, n in step.const_layout(m)]
            + [("c_dofbody", "i", "nv", 1), ("c_ancmask", "u", "nb", 1)])
    return consts, lays, sizes
