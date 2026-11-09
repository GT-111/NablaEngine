"""JAX float64 ports of the G1 engine's forward pieces that the analytic adjoint differentiates.

Each function mirrors a device function of nablaengine.g1_engine.source / step (same formulas, same branch structure, the
same constants: model(m) packs them like source.constants and step.pair_constants), so that jax.vjp of a port is the
reference for the engine's VJP device function on the same (float32) inputs. Branches are jnp.where selections with safe
denominators, so the unselected side never poisons a gradient; discrete decisions the adjoint takes from the forward
(contact slot, active rows, clamped dofs) are explicit arguments.

Pieces: the fwd_body chain (kinematics, velocities, accelerations), inverse dynamics ID(q, v, a) (RNE, no armature),
the CRB mass matrix (with armature), the clamped PD actuator force, kbimp / efc_D, the geom poses, the five primitives of
the mjlab model (plane-sphere, plane-capsule, sphere-sphere, sphere-capsule, capsule-capsule) with make_frame, the
normal option (DK_ADJ_NORMAL) and the near-parallel capsule option (DK_ADJ_PARALLEL), the contact Jacobian and
constraint rows (contacts, joint limits), the row terms Psi_e / Psi_c of the residual, quaternion integration in MuJoCo
Warp's smooth (sinc) form, and the qpos <-> dof-tangent maps. Self-checked against MuJoCo C in float64.

Third-party code: the five primitives and make_frame mirror g1_engine/step.py's CUDA translations of MuJoCo Warp
(mujoco_warp/_src/collision_primitive_core.py, math.py; Copyright 2025 The Newton Developers, Apache License 2.0).

Conventions as in the engine: q = (p, quat, hinge angles), v = (world linear, local angular, hinge rates); spatial
quantities in the world frame; body 0 is the world, body 1 the free root, hinges at the body origins.

Importing this module enables float64 in JAX (jax_enable_x64) and defaults JAX to the CPU (JAX_PLATFORMS).
"""

import dataclasses
import functools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

PLANE, SPHERE, CAPSULE = 0, 1, 2  # engine geom type codes (step.GEOM_TYPE); no boxes in the mjlab model
EX, EY, EZ = np.eye(3)


@dataclasses.dataclass(frozen=True, eq=False)
class Model:
    """Model constants of the engine (numpy, static under jit). Bodies: parent, depth, qadr, dadr, children, pos, quat,
    axis, ipos, iquat, mass, inert, anc (hinge ancestors, nearest first), grav, armature, kp, kv, frclim, timestep, torso.
    Collision geoms (step.collision_geoms order): gbody, gtype, gsize, gpos, gquat. Pairs (geoms ordered by type, as the
    engine's pair table): pg1, pg2, pdim, pmu, psolref, psolimp, pfoot. Solver: binvw, dinvw, jrange, jlim, jsolref,
    jsolimp, impratio. dofbody, ancmask as step.extra_constants."""
    nb: int
    nq: int
    nv: int
    parent: np.ndarray
    depth: np.ndarray
    qadr: np.ndarray
    dadr: np.ndarray
    children: tuple
    pos: np.ndarray
    quat: np.ndarray
    axis: np.ndarray
    ipos: np.ndarray
    iquat: np.ndarray
    mass: np.ndarray
    inert: np.ndarray
    anc: tuple
    grav: np.ndarray
    armature: np.ndarray
    kp: np.ndarray
    kv: np.ndarray
    frclim: np.ndarray
    timestep: float
    torso: int
    geoms: tuple
    gbody: np.ndarray
    gtype: np.ndarray
    gsize: np.ndarray
    gpos: np.ndarray
    gquat: np.ndarray
    pg1: np.ndarray
    pg2: np.ndarray
    pdim: np.ndarray
    pmu: np.ndarray
    psolref: np.ndarray
    psolimp: np.ndarray
    pfoot: np.ndarray
    binvw: np.ndarray
    dinvw: np.ndarray
    jrange: np.ndarray
    jlim: np.ndarray
    jsolref: np.ndarray
    jsolimp: np.ndarray
    impratio: float
    dofbody: np.ndarray
    ancmask: np.ndarray


def model(m, foot_pairs=(), f32=False):
    """Model constants of MjModel m as the engine packs them (source.constants, step.pair_constants), in float64; f32=True
    rounds every float constant to float32 first (the engine's values; for VJP references on the engine's inputs)."""
    from nablaengine.g1_engine import step  # the engine's geom table and pair order

    # copies (the model's arrays are often edited per world, e.g. body_ipos for the COM offset)
    r = (lambda x: np.array(np.asarray(x, np.float64).astype(np.float32), np.float64)) if f32 else \
        (lambda x: np.array(x, np.float64))
    nb = m.nbody
    depth, children = np.zeros(nb, int), [[] for _ in range(nb)]
    for b in range(1, nb):
        depth[b] = depth[m.body_parentid[b]] + 1
        children[m.body_parentid[b]].append(b)
    assert m.jnt_type[m.body_jntadr[1]] == mujoco.mjtJoint.mjJNT_FREE and m.body_jntnum[1] == 1
    qadr, dadr, axis = np.zeros(nb, int), np.zeros(nb, int), np.zeros((nb, 3))
    for b in range(2, nb):
        j = m.body_jntadr[b]
        assert m.body_jntnum[b] == 1 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and not m.jnt_pos[j].any()
        qadr[b], dadr[b], axis[b] = m.jnt_qposadr[j], m.jnt_dofadr[j], m.jnt_axis[j]
    anc = []
    for b in range(nb):
        k, lst = m.body_parentid[b] if b >= 2 else 0, []
        while k >= 2:
            lst.append(int(k))
            k = m.body_parentid[k]
        anc.append(tuple(lst))
    kp, kv, frclim = np.zeros(m.nv), np.zeros(m.nv), np.full(m.nv, 1e30)
    for a in range(m.nu):
        j = m.actuator_trnid[a, 0]
        assert m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and m.actuator_gear[a, 0] == 1.0
        dof = m.jnt_dofadr[j]
        kp[dof], kv[dof] = m.actuator_gainprm[a, 0], -m.actuator_biasprm[a, 2]
        if m.jnt_actfrclimited[j]:
            frclim[dof] = m.jnt_actfrcrange[j, 1]
    geoms = step.collision_geoms(m)
    gid = {g: i for i, g in enumerate(geoms)}
    pg1, pg2 = np.zeros(m.npair, int), np.zeros(m.npair, int)
    for i in range(m.npair):
        g1, g2 = int(m.pair_geom1[i]), int(m.pair_geom2[i])
        if step.GEOM_TYPE[int(m.geom_type[g1])] > step.GEOM_TYPE[int(m.geom_type[g2])]:
            g1, g2 = g2, g1
        pg1[i], pg2[i] = gid[g1], gid[g2]
    jr, jlim = np.zeros((m.nv, 2)), np.zeros(m.nv, bool)
    for j in range(1, m.njnt):
        jr[m.jnt_dofadr[j]], jlim[m.jnt_dofadr[j]] = m.jnt_range[j], m.jnt_limited[j]
    dofbody, mask = step.extra_constants(m)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    foot = np.zeros(m.npair, bool)
    foot[list(foot_pairs)] = True
    return Model(
        nb=nb, nq=m.nq, nv=m.nv, parent=np.asarray(m.body_parentid, int), depth=depth, qadr=qadr, dadr=dadr,
        children=tuple(tuple(c) for c in children), pos=r(m.body_pos), quat=r(m.body_quat), axis=r(axis),
        ipos=r(m.body_ipos), iquat=r(m.body_iquat), mass=r(m.body_mass), inert=r(m.body_inertia), anc=tuple(anc),
        grav=r(m.opt.gravity), armature=r(m.dof_armature), kp=r(kp), kv=r(kv), frclim=r(frclim),
        timestep=float(r(m.opt.timestep)), torso=int(torso), geoms=tuple(geoms),
        gbody=np.asarray(m.geom_bodyid[geoms], int), gtype=np.asarray([step.GEOM_TYPE[int(m.geom_type[g])] for g in geoms]),
        gsize=r(m.geom_size[geoms]), gpos=r(m.geom_pos[geoms]), gquat=r(m.geom_quat[geoms]), pg1=pg1, pg2=pg2,
        pdim=np.asarray(m.pair_dim, int), pmu=r(m.pair_friction[:, 0]), psolref=r(m.pair_solref),
        psolimp=r(m.pair_solimp), pfoot=foot, binvw=r(m.body_invweight0[:, 0]), dinvw=r(m.dof_invweight0), jrange=r(jr),
        jlim=jlim, jsolref=r(m.jnt_solref[1]), jsolimp=r(m.jnt_solimp[1]), impratio=float(r(m.opt.impratio)),
        dofbody=np.asarray(dofbody, int), ancmask=np.asarray(mask, np.int64))


# ---------------------------------------------------------------- small helpers (source.DEVICE)

_I1, _I2 = np.array([1, 2, 0]), np.array([2, 0, 1])


def cross(a, b):  # vectorized (few XLA ops: the tree recursions are unrolled over the bodies)
    return a[_I1] * b[_I2] - a[_I2] * b[_I1]


def qmul(a, b):  # the engine's qmul: (a0 b0 - av . bv, a0 bv + b0 av + av x bv)
    av, bv = a[1:], b[1:]
    return jnp.concatenate([(a[0] * b[0] - jnp.dot(av, bv))[None], a[0] * bv + b[0] * av + cross(av, bv)])


def qrot(q, v):  # q v q^-1, the engine's formula: t = 2 qv x v, v + q0 t + qv x t
    t = 2.0 * cross(q[1:], v)
    return v + q[0] * t + cross(q[1:], t)


def qconj(q):
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def qrot_inv(q, v):
    return qrot(qconj(q), v)



def _safe_norm(v):
    """|v| with a zero (not NaN) gradient at v = 0."""
    s = jnp.dot(v, v)
    pos = s > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, s, 1.0)), 0.0)


def normalize3(v):  # the engine's normalize3: unchanged when |v| = 0
    n = _safe_norm(v)
    return jnp.where(n > 0.0, v / jnp.where(n > 0.0, n, 1.0), v)


# ---------------------------------------------------------------- tree recursions (source.DEVICE)

def body_ipos(mdl, b, com):
    """COM offset of body b in its frame, with the per-world torso offset (domain randomization)."""
    return mdl.ipos[b] + (com if b == mdl.torso else 0.0)


def fwd_body(mdl, b, Qp, Xp, Wp, Vp, ALp, Ap, qpos, qvel, qacc):
    """Pose Q, X, angular / linear velocity W, V and acceleration AL, A of body b's origin from its parent's (A is the
    classical acceleration minus gravity: the root starts at qacc_p - g)."""
    if b == 1:
        q = qpos[3:7]
        Q = q / jnp.sqrt(jnp.dot(q, q))
        return Q, qpos[:3], qrot(Q, qvel[3:6]), qvel[:3], qrot(Q, qacc[3:6]), qacc[:3] - mdl.grav
    q0 = qmul(Qp, mdl.quat[b])
    X = Xp + qrot(Qp, mdl.pos[b])
    ang, qd, qdd = qpos[mdl.qadr[b]], qvel[mdl.dadr[b]], qacc[mdl.dadr[b]]
    ha = jnp.concatenate([jnp.cos(0.5 * ang)[None], jnp.sin(0.5 * ang) * mdl.axis[b]])
    Q = qmul(q0, ha)
    aw = qrot(Q, mdl.axis[b])
    r = X - Xp
    W = Wp + aw * qd
    V = Vp + cross(Wp, r)
    AL = ALp + aw * qdd + cross(Wp, aw * qd)
    A = Ap + cross(ALp, r) + cross(Wp, cross(Wp, r))
    return Q, X, W, V, AL, A


def kinematics(mdl, qpos, qvel=None, qacc=None):
    """The fwd_body chain over the tree: dict of (NB, .) arrays Q, X, W, V, AL, A (body 0 = world, at rest)."""
    qvel = jnp.zeros(mdl.nv) if qvel is None else qvel
    qacc = jnp.zeros(mdl.nv) if qacc is None else qacc
    z3 = jnp.zeros(3)
    out = [(jnp.array([1.0, 0.0, 0.0, 0.0]), z3, z3, z3, z3, z3)]
    for b in range(1, mdl.nb):  # model order: parents first
        out.append(fwd_body(mdl, b, *out[mdl.parent[b]], qpos, qvel, qacc))
    return {k: jnp.stack([o[i] for o in out]) for i, k in enumerate(("Q", "X", "W", "V", "AL", "A"))}


def body_force(mdl, b, Q, W, AL, A, ipos):
    """Force F and moment N about the body origin that body b alone needs (Newton-Euler at its COM)."""
    c = qrot(Q, ipos)
    m = mdl.mass[b]
    F = m * (A + cross(AL, c) + cross(W, cross(W, c)))
    qi = qmul(Q, mdl.iquat[b])
    Iw = qrot(qi, mdl.inert[b] * qrot_inv(qi, W))
    Ia = qrot(qi, mdl.inert[b] * qrot_inv(qi, AL))
    return F, Ia + cross(W, Iw) + cross(c, F)


def accumulate(Xb, Xc, Fc, Nc, F, N):
    return F + Fc, N + Nc + cross(Xc - Xb, Fc)


def torques(mdl, b, Q, F, N):
    """{dof: generalized force} of body b's joint from its subtree force and moment about its origin."""
    if b == 1:
        l = qrot_inv(Q, N)
        return {k: (F[k] if k < 3 else l[k - 3]) for k in range(6)}
    return {int(mdl.dadr[b]): jnp.dot(qrot(Q, mdl.axis[b]), N)}


def inverse_dynamics(mdl, qpos, qvel, qacc, com=None):
    """ID(q, v, a) = M_CRB(q) a + c(q, v) (gravity included, armature excluded): RNE as the engine's smooth phase (which
    evaluates it at a = 0 for the bias force). The adjoint's channel 1 (adjoint.py A4) differentiates lambda' ID."""
    com = jnp.zeros(3) if com is None else com
    K = kinematics(mdl, qpos, qvel, qacc)
    F, N = [jnp.zeros(3)] * mdl.nb, [jnp.zeros(3)] * mdl.nb
    for b in range(1, mdl.nb):
        F[b], N[b] = body_force(mdl, b, K["Q"][b], K["W"][b], K["AL"][b], K["A"][b], body_ipos(mdl, b, com))
    for b in range(mdl.nb - 1, 1, -1):  # children before parents (model order is a topological order)
        p = mdl.parent[b]
        F[p], N[p] = accumulate(K["X"][p], K["X"][b], F[b], N[b], F[p], N[p])
    out = {}
    for b in range(1, mdl.nb):
        out.update(torques(mdl, b, K["Q"][b], F[b], N[b]))
    return jnp.stack([out[i] for i in range(mdl.nv)])


def bias_force(mdl, qpos, qvel, com=None):
    """qfrc_bias = ID(q, v, 0)."""
    return inverse_dynamics(mdl, qpos, qvel, jnp.zeros(mdl.nv), com)


def body_inertia(mdl, b, Q, X, O, ipos):
    """Spatial inertia of body b about the point O, packed as J (xx yy zz xy xz yz), h = m c (3), m; c = com - O."""
    qi = qmul(Q, mdl.iquat[b])
    R = jnp.stack([qrot(qi, e) for e in (EX, EY, EZ)])  # R[k] = column k of the inertia frame
    I = mdl.inert[b]
    J = jnp.stack([I @ (R[:, 0] * R[:, 0]), I @ (R[:, 1] * R[:, 1]), I @ (R[:, 2] * R[:, 2]),
                   I @ (R[:, 0] * R[:, 1]), I @ (R[:, 0] * R[:, 2]), I @ (R[:, 1] * R[:, 2])])
    c = X + qrot(Q, ipos) - O
    m = mdl.mass[b]
    cc = jnp.dot(c, c)
    J = J + m * jnp.stack([cc - c[0] * c[0], cc - c[1] * c[1], cc - c[2] * c[2], -c[0] * c[1], -c[0] * c[2],
                           -c[1] * c[2]])
    return jnp.concatenate([J, m * c, jnp.array([m])])


def apply_inertia(J, w, v):
    """(n, f) = I (w, v) for a packed spatial inertia about O."""
    h = J[6:9]
    n = jnp.stack([J[0] * w[0] + J[3] * w[1] + J[4] * w[2], J[3] * w[0] + J[1] * w[1] + J[5] * w[2],
                   J[4] * w[0] + J[5] * w[1] + J[2] * w[2]]) + cross(h, v)
    return n, J[9] * v - cross(h, w)


def dof_motion(mdl, k, t, Q, X, O):
    """Motion subspace (w, v) of dof t of body k about the point O (root: t = 0..5, hinge: t = 0)."""
    if k == 1:
        e = np.eye(3)[t % 3]
        if t < 3:
            return jnp.zeros(3), jnp.asarray(e)
        return qrot(Q, e), jnp.zeros(3)
    w = qrot(Q, mdl.axis[k])
    return w, cross(w, O - X)


def mass_matrix(mdl, qpos, com=None):
    """Joint-space mass matrix by the composite-rigid-body algorithm about the root origin, with the armature (dense
    NV x NV, the engine's packed lower triangle mirrored)."""
    com = jnp.zeros(3) if com is None else com
    K = kinematics(mdl, qpos)
    O = K["X"][1]
    J = [None] + [body_inertia(mdl, b, K["Q"][b], K["X"][b], O, body_ipos(mdl, b, com)) for b in range(1, mdl.nb)]
    for b in range(mdl.nb - 1, 1, -1):
        J[mdl.parent[b]] = J[mdl.parent[b]] + J[b]
    ent = {}
    for b in range(2, mdl.nb):
        j = int(mdl.dadr[b])
        sw, sv = dof_motion(mdl, b, 0, K["Q"][b], K["X"][b], O)
        n, f = apply_inertia(J[b], sw, sv)
        ent[j, j] = jnp.dot(sw, n) + jnp.dot(sv, f) + mdl.armature[j]
        for k in mdl.anc[b]:
            kw, kv = dof_motion(mdl, k, 0, K["Q"][k], K["X"][k], O)
            ent[j, int(mdl.dadr[k])] = jnp.dot(kw, n) + jnp.dot(kv, f)
        for t in range(6):
            kw, kv = dof_motion(mdl, 1, t, K["Q"][1], K["X"][1], O)
            ent[j, t] = jnp.dot(kw, n) + jnp.dot(kv, f)
    for j in range(6):
        sw, sv = dof_motion(mdl, 1, j, K["Q"][1], K["X"][1], O)
        n, f = apply_inertia(J[1], sw, sv)
        for i in range(j + 1):
            kw, kv = dof_motion(mdl, 1, i, K["Q"][1], K["X"][1], O)
            ent[j, i] = jnp.dot(kw, n) + jnp.dot(kv, f) + (mdl.armature[j] if i == j else 0.0)
    rows, cols = np.array([k[0] for k in ent]), np.array([k[1] for k in ent])
    vals = jnp.stack(list(ent.values()))
    M = jnp.zeros((mdl.nv, mdl.nv)).at[rows, cols].set(vals)
    return M.at[cols, rows].set(vals)


# ---------------------------------------------------------------- actuators, impedance

def actuator_raw(mdl, qpos, qvel, ctrl):
    """Unclamped PD force kp (ctrl - q) - kv qdot on the hinge dofs (zeros on the root dofs)."""
    act = mdl.kp[6:] * (ctrl - qpos[7:]) - mdl.kv[6:] * qvel[6:]
    return jnp.concatenate([jnp.zeros(6), act])


def actuator_force(mdl, qpos, qvel, ctrl, clamped=None):
    """Clamped actuator force (the engine's qfrc_smooth part) and the clamp mask (after clamping at or beyond
    +-frclim, the criterion of MuJoCo Warp's _actuator_lam). clamped: the mask from the forward (tape); its dofs take
    the clamped value with a zero derivative, the others the raw PD law."""
    raw = actuator_raw(mdl, qpos, qvel, ctrl)
    lim = jnp.asarray(mdl.frclim)
    cl = jnp.clip(raw, -lim, lim)
    mask = (cl <= -lim) | (cl >= lim) if clamped is None else jnp.asarray(clamped, bool)
    return jnp.where(mask, jax.lax.stop_gradient(cl), raw), mask


def pow_imp(x, p):
    return x * x if p == 2.0 else (x if p == 1.0 else jnp.power(x, p))


def kbimp(solref, solimp, pos_imp, h):
    """Stiffness k, damping b and impedance imp at pos_imp (refsafe: timeconst >= 2 h), the engine's kbimp."""
    solref, solimp = np.asarray(solref, np.float64), np.asarray(solimp, np.float64)
    tc, dr = max(solref[0], 2.0 * h), solref[1]
    dmin, dmax = min(max(solimp[0], 1e-4), 0.9999), min(max(solimp[1], 1e-4), 0.9999)
    width, mid, power = max(1e-15, solimp[2]), min(max(solimp[3], 1e-4), 0.9999), max(1.0, solimp[4])
    k = 1.0 / max(1e-15, dmax * dmax * tc * tc * dr * dr)
    b = 2.0 / max(1e-15, dmax * tc)
    if dmin == dmax or solimp[2] <= 1e-15:
        return k, b, jnp.full_like(pos_imp, 0.5 * (dmin + dmax))
    x = jnp.abs(pos_imp) / width
    if power == 1.0:
        mid_val = dmin + x * (dmax - dmin)
        lo_val = hi_val = mid_val
    else:
        # each piece on its own range (safe powers); where, not clip: at x == mid a tied clip would pass half the
        # gradient to the selected piece
        xl, xh = jnp.where(x <= mid, x, mid), jnp.where(x >= mid, jnp.minimum(x, 1.0), mid)
        lo_val = jnp.clip(dmin + pow_imp(xl, power) / pow_imp(mid, power - 1.0) * (dmax - dmin), dmin, dmax)
        hi_val = jnp.clip(dmin + (1.0 - pow_imp(1.0 - xh, power) / pow_imp(1.0 - mid, power - 1.0)) * (dmax - dmin),
                          dmin, dmax)
    imp = jnp.where(x <= 0.0, dmin, jnp.where(x >= 1.0, dmax, jnp.where(x <= mid, lo_val, hi_val)))
    return k, b, imp


def efc_D(invw, imp):
    """D = 1 / R, R = invweight (1 - imp) / imp (floored at 1e-15)."""
    return 1.0 / jnp.maximum(invw * (1.0 - imp) / imp, 1e-15)


# ---------------------------------------------------------------- collision (step.STEP_DEVICE)

def geom_poses(mdl, Q, X):
    """World positions (NG, 3) and quaternions (NG, 4) of the collision geoms from the body poses."""
    gx = jnp.stack([X[b] + qrot(Q[b], mdl.gpos[g]) for g, b in enumerate(mdl.gbody)])
    gq = jnp.stack([qmul(Q[b], mdl.gquat[g]) for g, b in enumerate(mdl.gbody)])
    return gx, gq


def make_frame(n):
    """MuJoCo's make_frame(n): rows n, b, c with b from e_y (|n_y| < 0.5) or e_z, made orthogonal to n."""
    e = jnp.where((-0.5 < n[1]) & (n[1] < 0.5), EY, EZ)
    b = normalize3(e - n * jnp.dot(n, e))
    return jnp.stack([n, b, cross(n, b)])


NORMAL_EXACT, NORMAL_FROZEN, NORMAL_REG, NORMAL_CUT = 0, 1, 2, 3  # DK_ADJ_NORMAL (adjoint_contact.header)


def _kappa(nu, mode, eps):
    if mode == NORMAL_EXACT:
        return 1.0 / nu
    if mode == NORMAL_FROZEN:
        return jnp.zeros_like(nu)
    if mode == NORMAL_REG:
        return 1.0 / jnp.maximum(nu, eps)
    return jnp.where(nu < eps, 0.0, 1.0 / nu)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _unit_mod(v, mode, eps):
    return v / jnp.sqrt(jnp.dot(v, v))


def _unit_mod_fwd(v, mode, eps):
    nu = jnp.sqrt(jnp.dot(v, v))
    return v / nu, (v / nu, nu)


def _unit_mod_bwd(mode, eps, res, g):
    n, nu = res
    return (_kappa(nu, mode, eps) * (g - n * jnp.dot(n, g)),)


_unit_mod.defvjp(_unit_mod_fwd, _unit_mod_bwd)


def unit_normal(v, mode=NORMAL_EXACT, eps=1e-3):
    """n = v / |v| (e_x when v = 0, zero derivative there); the derivative (I - n n') / |v| becomes kappa (I - n n') for
    DK_ADJ_NORMAL = mode (the forward value is unchanged)."""
    nu2 = jnp.dot(v, v)
    pos = nu2 > 0.0
    vs = jnp.where(pos, v, EX)
    if mode == NORMAL_EXACT:
        n = vs / jnp.sqrt(jnp.dot(vs, vs))
    else:
        n = _unit_mod(vs, int(mode), float(eps))
    return jnp.where(pos, n, EX)


def sphere_sphere(p1, r1, p2, r2, mode=NORMAL_EXACT, eps=1e-3):
    """(dist, pos, normal) of two spheres; the core of every v / |v| normal (sphere-capsule, capsule-capsule)."""
    d = p2 - p1
    n = unit_normal(d, mode, eps)
    dist = _safe_norm(d) - (r1 + r2)
    return dist, p1 + n * (r1 + 0.5 * dist), n


def plane_sphere(pn, pp, sp, r):
    """Plane (normal pn through pp) against a sphere at sp: (dist, pos, frame)."""
    dist = jnp.dot(sp, pn) - jnp.dot(pp, pn) - r
    return dist, sp - pn * (r + 0.5 * dist), make_frame(pn)


def plane_capsule(pn, pp, cp, cq, r, hl, slot, fallback=True):
    """Plane against capsule end `slot` (0: +axis, 1: -axis): (dist, pos, frame); the frame's first tangent follows the
    capsule axis projected on the plane, or (fallback, MuJoCo Warp and the engine) make_frame's basis when that
    projection is shorter than 0.5. fallback=False keeps the projected axis (MuJoCo C 3.17)."""
    axis = qrot(cq, EZ)
    b = axis - pn * jnp.dot(pn, axis)
    bn = _safe_norm(b)
    b = jnp.where(bn > 0.0, b / jnp.where(bn > 0.0, bn, 1.0), b)
    basis = jnp.where((-0.5 < pn[1]) & (pn[1] < 0.5), EY, EZ)
    b = jnp.where(bn < 0.5, basis, b) if fallback else b
    frame = jnp.stack([pn, b, cross(pn, b)])
    ce = cp + (1.0 if slot == 0 else -1.0) * axis * hl
    dist = jnp.dot(ce, pn) - jnp.dot(pp, pn) - r
    return dist, ce - pn * (r + 0.5 * dist), frame


def sphere_capsule(sp, r1, cp, cq, r2, hl, mode=NORMAL_EXACT, eps=1e-3, reg=1e-6):
    """Sphere at sp against a capsule: closest point of the segment (t regularized by reg = 1e-6 as MuJoCo Warp's
    closest_segment_point), then the sphere core. reg=0 gives the exact closest point (MuJoCo C)."""
    axis = qrot(cq, EZ)
    a = cp - axis * hl
    ab = 2.0 * axis * hl
    t = jnp.clip(jnp.dot(sp - a, ab) / (jnp.dot(ab, ab) + reg), 0.0, 1.0)
    dist, pos, n = sphere_sphere(sp, r1, a + t * ab, r2, mode, eps)
    return dist, pos, make_frame(n)


def capsule_params(p1, q1, hl1, p2, q2, hl2, slot=0, parallel_eps=None):
    """Segment parameters (x1, x2) in [-1, 1] of the capsule-capsule closest points (the engine's clamp cascade), the
    determinant and whether the parallel branch (sin^2 <= 1e-8, endpoint `slot`) was taken. parallel_eps (the engine's
    DK_ADJ_PARALLEL = 1 with DK_ADJ_PARALLEL_EPS; None: exact): where both parameters are free and sin^2(theta) =
    |a1 x a2|^2 / (|a1|^2 |a2|^2) < parallel_eps, x1 and x2 are held constant in the derivative (stop_gradient; the
    values are unchanged)."""
    a1, a2 = qrot(q1, EZ) * hl1, qrot(q2, EZ) * hl2
    dif = p1 - p2
    ma, mb, mc = jnp.dot(a1, a1), -jnp.dot(a1, a2), jnp.dot(a2, a2)
    u, v = -jnp.dot(a1, dif), jnp.dot(a2, dif)
    w = cross(a1, a2)
    det = jnp.dot(w, w)
    par = det <= 1e-8 * ma * mc
    dets = jnp.where(par, 1.0, det)
    first = jnp.dot(w, cross(a2, dif)) / dets
    x1 = jnp.clip(first, -1.0, 1.0)
    second = (v - mb * x1) / mc
    frozen = (~par & (jnp.abs(first) <= 1.0) & (jnp.abs(second) <= 1.0) & (det < parallel_eps * ma * mc)
              if parallel_eps is not None else None)  # both free (no clamp below) and near parallel
    x2 = jnp.clip(second, -1.0, 1.0)
    x1 = jnp.where(jnp.abs(second) > 1.0, jnp.clip((u - mb * x2) / ma, -1.0, 1.0), x1)
    endpoint = -1.0 if slot == 0 else 1.0
    other = (v - mb * endpoint) / mc
    candidate2 = jnp.clip(other, -1.0, 1.0)
    candidate1 = jnp.where(jnp.abs(other) > 1.0, jnp.clip((u - mb * candidate2) / ma, -1.0, 1.0), endpoint)
    x1, x2 = jnp.where(par, candidate1, x1), jnp.where(par, candidate2, x2)
    if frozen is not None:
        x1 = jnp.where(frozen, jax.lax.stop_gradient(x1), x1)
        x2 = jnp.where(frozen, jax.lax.stop_gradient(x2), x2)
    return x1, x2, det, par


def capsule_capsule(p1, q1, r1, hl1, p2, q2, r2, hl2, slot=0, mode=NORMAL_EXACT, eps=1e-3, parallel_eps=None):
    """Two capsules: closest points of the segments (capsule_params, parallel_eps: DK_ADJ_PARALLEL), then the sphere
    core: (dist, pos, frame)."""
    x1, x2, _, _ = capsule_params(p1, q1, hl1, p2, q2, hl2, slot, parallel_eps)
    v1, v2 = p1 + qrot(q1, EZ) * hl1 * x1, p2 + qrot(q2, EZ) * hl2 * x2
    dist, pos, n = sphere_sphere(v1, r1, v2, r2, mode, eps)
    return dist, pos, make_frame(n)


def collide_geoms(t1, t2, x1, q1, s1, x2, q2, s2, slot=0, mode=NORMAL_EXACT, eps=1e-3, reg=1e-6, fallback=True,
                  parallel_eps=None):
    """(dist, pos, frame) of contact `slot` between geoms of engine types t1 <= t2 (static) at world poses (x, q) with
    sizes s (the engine's collide_pair without the culling; the caller decides which contacts exist). reg and fallback:
    see sphere_capsule and plane_capsule (their defaults are MuJoCo Warp's and the engine's); parallel_eps: the
    capsule-capsule option DK_ADJ_PARALLEL (capsule_params)."""
    if t1 == PLANE:
        pn = qrot(q1, EZ)
        if t2 == SPHERE:
            return plane_sphere(pn, x1, x2, s2[0])
        return plane_capsule(pn, x1, x2, q2, s2[0], s2[1], slot, fallback)
    if t1 == SPHERE and t2 == SPHERE:
        dist, pos, n = sphere_sphere(x1, s1[0], x2, s2[0], mode, eps)
        return dist, pos, make_frame(n)
    if t1 == SPHERE:
        return sphere_capsule(x1, s1[0], x2, q2, s2[0], s2[1], mode, eps, reg)
    return capsule_capsule(x1, q1, s1[0], s1[1], x2, q2, s2[0], s2[1], slot, mode, eps, parallel_eps)


def collide(mdl, gx, gq, pair, slot=0, mode=NORMAL_EXACT, eps=1e-3):
    """collide_geoms for contact (pair, slot) from the collision geoms' world poses (geom_poses)."""
    g1, g2 = int(mdl.pg1[pair]), int(mdl.pg2[pair])
    return collide_geoms(int(mdl.gtype[g1]), int(mdl.gtype[g2]), gx[g1], gq[g1], mdl.gsize[g1], gx[g2], gq[g2],
                         mdl.gsize[g2], slot, mode, eps)


def contact_axis_distance(mdl, pair, dist):
    """nu = dist + r1 + r2, the centre / axis distance of a sphere or capsule contact before the radii."""
    return dist + mdl.gsize[mdl.pg1[pair], 0] + mdl.gsize[mdl.pg2[pair], 0]


# ---------------------------------------------------------------- constraint rows

def support(mdl, pair):
    """(dofs, signs) of a contact's Jacobian support: the dofs on exactly one of the two bodies' chains, +1 on body 2's."""
    sgn = support_signs(mdl, pair)
    dofs = np.flatnonzero(sgn)
    return dofs, sgn[dofs]


def support_signs(mdl, pair):
    """(NV,) +1 / -1 on the dofs of body 2's / body 1's chain that are not on both, 0 elsewhere (the support)."""
    b1, b2 = int(mdl.gbody[mdl.pg1[pair]]), int(mdl.gbody[mdl.pg2[pair]])
    sgn = np.zeros(mdl.nv)
    for i in range(mdl.nv):
        bi = int(mdl.dofbody[i])
        in2, in1 = (int(mdl.ancmask[b2]) >> bi) & 1, (int(mdl.ancmask[b1]) >> bi) & 1
        if in1 != in2:
            sgn[i] = 1.0 if in2 else -1.0
    return sgn


def point_jacobian(mdl, Q, X, pos, sgn):
    """(3, NV) translational Jacobian of the point pos moving with body 2 relative to body 1, column i scaled by sgn[i]
    (support_signs; may be traced): the engine's contact-Jacobian columns before the frame projection."""
    O = X[1]
    cols = []
    for i in range(mdl.nv):
        bi = int(mdl.dofbody[i])
        w, v = dof_motion(mdl, bi, i if bi == 1 else 0, Q[bi], X[bi], O)
        cols.append(sgn[i] * (v + cross(w, pos - O)))
    return jnp.stack(cols, axis=1)


def contact_jacobian(mdl, Q, X, pos, frame, sgn):
    """(3, NV) frame rows (normal, two tangents) of J(body 2, pos) - J(body 1, pos)."""
    return frame @ point_jacobian(mdl, Q, X, pos, sgn)


def contact_params(mdl, pair, fric):
    """Static and traced row parameters of a pair: dict dim, solref, solimp (tuples), mu, invweight (body_invweight0
    translation of both bodies; condim 3 times (1 + mu^2) 2 mu^2 / impratio, the engine's formula), sgn."""
    mu = fric if mdl.pfoot[pair] else float(mdl.pmu[pair])
    iw = mdl.binvw[mdl.gbody[mdl.pg1[pair]]] + mdl.binvw[mdl.gbody[mdl.pg2[pair]]]
    dim = int(mdl.pdim[pair])
    if dim == 3:
        iw = (iw + mu * mu * iw) * 2.0 * mu * mu / mdl.impratio
    return dict(dim=dim, solref=tuple(map(float, mdl.psolref[pair])), solimp=tuple(map(float, mdl.psolimp[pair])),
                mu=mu, invweight=iw, sgn=support_signs(mdl, pair))


def edges(dim, mu):
    """(nrow, 3) row directions in frame coordinates: pyramid edges n +- mu t1, n +- mu t2 (condim 3) or n (condim 1)."""
    if dim != 3:
        return jnp.array([[1.0, 0.0, 0.0]])
    one, z = jnp.ones_like(mu), jnp.zeros_like(mu)
    return jnp.stack([jnp.stack([one, mu, z]), jnp.stack([one, -mu, z]), jnp.stack([one, z, mu]),
                      jnp.stack([one, z, -mu])])


def contact_rows(mdl, Q, X, qvel, dist, pos, frame, dim, solref, solimp, mu, invweight, sgn):
    """Rows of one contact (contact_params gives dim, solref, solimp (static), mu, invweight, sgn): J (nrow, NV), D, aref
    (nrow) and the impedance (k, b, imp) at dist."""
    J = edges(dim, mu) @ contact_jacobian(mdl, Q, X, pos, frame, sgn)
    k, b, imp = kbimp(solref, solimp, dist, mdl.timestep)
    D = efc_D(invweight, imp)
    aref = -k * imp * dist - b * (J @ qvel)
    return J, jnp.broadcast_to(D, (J.shape[0],)), aref, (k, b, imp)


def contact_psi(mdl, dist, frame, dim, solref, solimp, mu, invweight, Vv, Va, Vl, active):
    """Psi_c = sum over the contact's active rows of D x_e (u_e . V_lam), x_e = u_e . V_a + k imp(d) d + b u_e . V_v
    (V_* the 3-vectors J_p v, J_p a*, J_p lam of the contact point, u_e the row direction in world coordinates;
    active: (nrow,) the rows in the quadratic zone)."""
    U = edges(dim, mu) @ frame
    k, b, imp = kbimp(solref, solimp, dist, mdl.timestep)
    D = efc_D(invweight, imp)
    x = U @ Va + k * imp * dist + b * (U @ Vv)
    return jnp.sum(jnp.where(jnp.asarray(active, bool), D * x * (U @ Vl), 0.0))


def limit_rows(mdl, qpos, qvel, active=None):
    """Joint-limit rows of all hinge dofs (the active ones by the forward's test, or `active`): dict with dof, sign
    (+1 lower, -1 upper), pos = min(q - lo, hi - q), D, aref, imp; inactive dofs carry the same formulas."""
    i = np.arange(6, mdl.nv)
    q = qpos[i + 1]
    dmin, dmax = q - mdl.jrange[i, 0], mdl.jrange[i, 1] - q
    sign = jnp.where(dmin < dmax, 1.0, -1.0)
    pos = jnp.minimum(dmin, dmax)
    act = (jnp.asarray(mdl.jlim[i]) & (pos < 0.0)) if active is None else jnp.asarray(active, bool)
    k, b, imp = kbimp(mdl.jsolref, mdl.jsolimp, pos, mdl.timestep)
    D = efc_D(mdl.dinvw[i], imp)
    return dict(dof=i, sign=sign, pos=pos, active=act, D=D, aref=-k * imp * pos - b * (sign * qvel[i]), imp=imp)


def limit_psi(mdl, qpos, qvel, qacc, lam, active):
    """sum over the active limit rows of Psi_e = D(pos) x_e (J_e lam), x_e = J_e a + k imp pos + b J_e v."""
    r = limit_rows(mdl, qpos, qvel, active)
    J = r["sign"]
    x = J * qacc[r["dof"]] - r["aref"]
    return jnp.sum(jnp.where(active, r["D"] * x * J * lam[r["dof"]], 0.0))


# ---------------------------------------------------------------- integration, coordinates

def _sinc(x):
    small = x * x < 1e-4
    xs = jnp.where(small, 1.0, x)
    return jnp.where(small, 1.0 - x * x / 6.0 + x ** 4 / 120.0, jnp.sin(xs) / xs)


def quat_integrate(q, w, h):
    """normalize(normalize(q) (cos x, (h/2) sinc(x) w)), x = (h/2)|w| (MuJoCo Warp's smooth form; the same rotation as
    the engine's axis-angle update, differentiable at w = 0)."""
    x = 0.5 * h * _safe_norm(w)
    rho = jnp.concatenate([jnp.cos(x)[None], 0.5 * h * _sinc(x) * w])
    u = qmul(q / jnp.sqrt(jnp.dot(q, q)), rho)
    return u / jnp.sqrt(jnp.dot(u, u))


def integrate_pos(mdl, qpos, qvel, h=None):
    """q' = q (+) h v: root translation, root quaternion, hinge angles (implicitfast passes the new velocity)."""
    h = mdl.timestep if h is None else h
    return jnp.concatenate([qpos[:3] + h * qvel[:3], quat_integrate(qpos[3:7], qvel[3:6], h), qpos[7:] + h * qvel[6:]])


def dof_to_qpos(g, qpos):
    """Lift a dof-tangent cotangent (NV) to qpos form (NQ): (g_p, 2 xi (x) (0, g_theta), g_phi), xi not normalized (MuJoCo
    Warp's _dof_to_qpos)."""
    gq = 2.0 * qmul(qpos[3:7], jnp.concatenate([jnp.zeros(1), g[3:6]]))
    return jnp.concatenate([g[:3], gq, g[6:]])


def qpos_to_dof(gq, qpos):
    """Tangent part of a qpos cotangent (NQ -> NV): g_theta = 1/2 Im(xi* (x) g_xi) / |xi|^2, the left inverse of
    dof_to_qpos (for the 35-dimensional error of qpos gradients)."""
    xi = qpos[3:7]
    gt = 0.5 * qmul(qconj(xi), gq[3:7])[1:] / jnp.dot(xi, xi)
    return jnp.concatenate([gq[:3], gt, gq[7:]])
