"""CUDA source of the G1-specific physics engine (block per world), generated from the MuJoCo model.

The smooth dynamics: forward kinematics, velocity / acceleration propagation, inverse dynamics (RNE) and the
joint-space mass matrix by the composite-rigid-body algorithm, as device functions over shared-memory state with one
warp per world (lane b owns body b; the tree is walked one depth level at a time). Conventions follow MuJoCo: free root
joint (world linear / local angular velocity), hinge joints at the body origin, spatial quantities in the world frame;
the mass matrix is taken about the root body's origin and includes the armature (as mj_crb's qM).

constants(m) packs the model arrays the kernels read (global memory, see DKC; load order = CONST_LAYOUT) and
asserts the model structure the kernels assume, among others: a free root at body 1, then one hinge per body at
its origin, at most 32 bodies, nv <= 64 (the solves give lane l the dof rows l and l + 32; with nv < 32 the lanes
past NV are masked: the Unitree H1, nv 25), position actuators in joint order
(actuator a on dof 6 + a), and a body named torso_link.
"""

import mujoco
import numpy as np

CONST_LAYOUT = (  # (symbol, kind, length in units of the body count or "nv", per-entry width)
    ("c_parent", "i", "nb", 1), ("c_depth", "i", "nb", 1), ("c_qadr", "i", "nb", 1), ("c_dadr", "i", "nb", 1),
    ("c_nchild", "i", "nb", 1), ("c_child", "i", "nb", 3), ("c_pos", "f", "nb", 3), ("c_quat", "f", "nb", 4),
    ("c_axis", "f", "nb", 3), ("c_ipos", "f", "nb", 3), ("c_iquat", "f", "nb", 4), ("c_mass", "f", "nb", 1),
    ("c_inert", "f", "nb", 3), ("c_grav", "f", "one", 3), ("c_armature", "f", "nv", 1), ("c_nanc", "i", "nb", 1),
    ("c_anc", "i", "nb", 12), ("c_kp", "f", "nv", 1), ("c_kv", "f", "nv", 1), ("c_frclim", "f", "nv", 1),
    ("c_misc", "f", "one", 2),  # timestep, torso body id
)


def constants(m):
    """Model arrays in CONST_LAYOUT order (float32), and the tree's maximum depth."""
    nb = m.nbody
    depth, children = [0] * nb, [[] for _ in range(nb)]
    for b in range(1, nb):
        depth[b] = depth[m.body_parentid[b]] + 1
        children[m.body_parentid[b]].append(b)
    assert nb <= 32 and max(len(c) for c in children[1:]) <= 3
    assert m.jnt_type[m.body_jntadr[1]] == mujoco.mjtJoint.mjJNT_FREE and m.body_jntnum[1] == 1
    qadr, dadr, axis = np.zeros(nb), np.zeros(nb), np.zeros((nb, 3))
    for b in range(2, nb):
        j = m.body_jntadr[b]
        assert m.body_jntnum[b] == 1 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and not m.jnt_pos[j].any()
        qadr[b], dadr[b], axis[b] = m.jnt_qposadr[j], m.jnt_dofadr[j], m.jnt_axis[j]
    child = np.zeros((nb, 3))
    for b in range(1, nb):
        child[b, :len(children[b])] = children[b]
    nchild = np.array([len(children[b]) if b else 0 for b in range(nb)])
    anc, nanc = np.zeros((nb, 12)), np.zeros(nb)  # hinge-joint ancestors of each body (root excluded), nearest first
    for b in range(2, nb):
        k, lst = m.body_parentid[b], []
        while k >= 2:
            lst.append(k)
            k = m.body_parentid[k]
        assert len(lst) <= 12
        anc[b, :len(lst)], nanc[b] = lst, len(lst)
    kp, kv, frclim = np.zeros(m.nv), np.zeros(m.nv), np.full(m.nv, 1e30)
    for a in range(m.nu):  # position actuators on hinge joints, gear 1: force = kp (ctrl - q) - kv qdot
        j = m.actuator_trnid[a, 0]
        assert m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT, f"actuator {a}: joint transmissions only"
        assert m.jnt_dofadr[j] == 6 + a, (
            f"actuator {a} drives dof {m.jnt_dofadr[j]}: the kernels read the setpoint of hinge dof i from "
            "ctrl[i - 6], so actuator a must drive dof 6 + a (actuators in joint order)")
        assert m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and m.actuator_gear[a, 0] == 1.0
        assert m.actuator_gainprm[a, 0] == -m.actuator_biasprm[a, 1] and not m.actuator_forcelimited[a]
        dof = m.jnt_dofadr[j]
        kp[dof], kv[dof] = m.actuator_gainprm[a, 0], -m.actuator_biasprm[a, 2]
        if m.jnt_actfrclimited[j]:
            frclim[dof] = m.jnt_actfrcrange[j, 1]
            assert m.jnt_actfrcrange[j, 0] == -m.jnt_actfrcrange[j, 1]
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    assert torso >= 1, 'no body named "torso_link": the per-world COM offset (com) is applied to that body'
    parts = [m.body_parentid, depth, qadr, dadr, nchild, child, m.body_pos, m.body_quat, axis, m.body_ipos,
             m.body_iquat, m.body_mass, m.body_inertia, m.opt.gravity, m.dof_armature, nanc, anc, kp, kv, frclim,
             [m.opt.timestep, torso]]
    return np.concatenate([np.asarray(x, np.float64).ravel() for x in parts]).astype(np.float32), max(depth)


def header(m, maxd, wpb):
    return f"""
#include <cuda_runtime.h>
#define NB {m.nbody}
#define NQ {m.nq}
#define NV {m.nv}
#define MAXD {maxd}
#define WPB {wpb}
#define NTRI (NV * (NV + 1) / 2)
#define TRI(i, j) ((i) * ((i) + 1) / 2 + (j))  // packed lower triangle, j <= i
""" + r"""
#ifndef DKC
#define DKC __device__  // where the model arrays live: global memory (L1-cached), as lanes index them by pair, body or
                       // dof (divergent addresses serialize in __constant__ memory); -DDKC=__constant__ for comparison
#endif
DKC int c_parent[NB], c_depth[NB], c_qadr[NB], c_dadr[NB], c_nchild[NB], c_child[NB * 3], c_nanc[NB],
    c_anc[NB * 12];
DKC float c_pos[NB * 3], c_quat[NB * 4], c_axis[NB * 3], c_ipos[NB * 3], c_iquat[NB * 4], c_mass[NB],
    c_inert[NB * 3], c_grav[3], c_armature[NV], c_kp[NV], c_kv[NV], c_frclim[NV], c_misc[2];
"""


DEVICE = r"""
__device__ __forceinline__ void qmul(const float* a, const float* b, float* o) {
  float w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  float x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  float y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  float z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
  o[0] = w; o[1] = x; o[2] = y; o[3] = z;
}
__device__ __forceinline__ void qrot(const float* q, const float* v, float* o) {  // o = q v q^-1
  float tx = 2.f * (q[2] * v[2] - q[3] * v[1]), ty = 2.f * (q[3] * v[0] - q[1] * v[2]), tz = 2.f * (q[1] * v[1] - q[2] * v[0]);
  o[0] = v[0] + q[0] * tx + q[2] * tz - q[3] * ty;
  o[1] = v[1] + q[0] * ty + q[3] * tx - q[1] * tz;
  o[2] = v[2] + q[0] * tz + q[1] * ty - q[2] * tx;
}
__device__ __forceinline__ void qrot_inv(const float* q, const float* v, float* o) {
  float qc[4] = {q[0], -q[1], -q[2], -q[3]};
  qrot(qc, v, o);
}
__device__ __forceinline__ void cross(const float* a, const float* b, float* o) {
  o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0];
}
__device__ __forceinline__ float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Explicit roundings. The adjoint replays the forward's body poses, collision-geom poses and contact geometry (dist,
// pos, frame) and needs them bitwise (adjoint.py adj_poses / adj_contacts), in another kernel and, in the JAX library,
// another compilation unit. Written as plain C (qmul, qrot, ... above), the mul / add pairs are fused into FMAs by NVVM
// and ptxas depending on the surrounding code (the uses of a product, register pressure, -fmad, the architecture), so
// the same source can give results an ulp apart in two kernels. These intrinsics are never contracted (and __fmaf_rn
// is never split): an expression written with them has its IEEE value, the same in every kernel, compilation unit and
// -fmad setting (on one GPU: rsqrtf and sincosf are the architecture's; IEEE division and sqrtf, nvcc's default
// -prec-div / -prec-sqrt, are correctly rounded and need nothing). Every operation the replay reproduces is written
// with them: the *_rn helpers here and fwd_body (the poses, and with them the twists and accelerations), step.py's
// geom_pose, the contact geometry that collide_pair shares with the adjoint's adj_contact_geom (make_frame,
// sphere_sphere, plane_point, plane_capsule_frame, capsule_end, segment_point, capcap_*), the bounding tests, the limit
// test (limit_dists) and the rows' impedance (kbimp, imp_raw, contact_invw, efc_D). The forms are the ones ptxas chose
// for the plain C on sm_70, so the change left the V100 results bitwise as they were: a b + c d = fma(a, b, c d),
// a b - c d = fma(a, b, -(c d)), x +- a b = fma(+-a, b, x), sums from the left; a product with a second use is rounded
// first (W = Wp + round(qd aw) in fwd_body, p +- round(l a) for the two ends of a capsule). The rest of the engine
// (dynamics, solver, the adjoint's VJPs) keeps the plain C helpers above: it is not replayed. Test: the forward or the
// adjoint unit built with -fmad=false: the replay stays bitwise the forward's
#define RMUL(a, b) __fmul_rn((a), (b))
#define RADD(a, b) __fadd_rn((a), (b))
#define RSUB(a, b) __fsub_rn((a), (b))
#define RFMA(a, b, c) __fmaf_rn((a), (b), (c))   // a b + c, one rounding
#define RFMS(a, b, c) __fmaf_rn(-(a), (b), (c))  // c - a b, one rounding

__device__ __forceinline__ void qmul_rn(const float* a, const float* b, float* o) {
  float w = RFMS(a[3], b[3], RFMS(a[2], b[2], RFMA(a[0], b[0], -RMUL(a[1], b[1]))));
  float x = RFMS(a[3], b[2], RFMA(a[2], b[3], RFMA(a[0], b[1], RMUL(a[1], b[0]))));
  float y = RFMA(a[3], b[1], RFMA(a[2], b[0], RFMA(a[0], b[2], -RMUL(a[1], b[3]))));
  float z = RFMA(a[3], b[0], RFMS(a[2], b[1], RFMA(a[0], b[3], RMUL(a[1], b[2]))));
  o[0] = w; o[1] = x; o[2] = y; o[3] = z;
}
// o = q v q^-1: t = 2 qv x v, o = v + q0 t + qv x t
__device__ __forceinline__ void qrot_rn(const float* q, const float* v, float* o) {
  float tx = RMUL(2.f, RFMA(q[2], v[2], -RMUL(q[3], v[1])));
  float ty = RMUL(2.f, RFMA(q[3], v[0], -RMUL(q[1], v[2])));
  float tz = RMUL(2.f, RFMA(q[1], v[1], -RMUL(q[2], v[0])));
  o[0] = RFMS(q[3], ty, RFMA(q[2], tz, RFMA(q[0], tx, v[0])));
  o[1] = RFMS(q[1], tz, RFMA(q[3], tx, RFMA(q[0], ty, v[1])));
  o[2] = RFMS(q[2], tx, RFMA(q[1], ty, RFMA(q[0], tz, v[2])));
}
__device__ __forceinline__ void cross_rn(const float* a, const float* b, float* o) {
  o[0] = RFMA(a[1], b[2], -RMUL(a[2], b[1]));
  o[1] = RFMA(a[2], b[0], -RMUL(a[0], b[2]));
  o[2] = RFMA(a[0], b[1], -RMUL(a[1], b[0]));
}
__device__ __forceinline__ float dot3_rn(const float* a, const float* b) {
  return RFMA(a[2], b[2], RFMA(a[0], b[0], RMUL(a[1], b[1])));
}
__device__ __forceinline__ void zaxis_rn(const float* q, float* a) {  // a = q e_z: a plane's normal, a capsule's axis
  const float ez[3] = {0.f, 0.f, 1.f};
  qrot_rn(q, ez, a);
}
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// fwd_body's three steps for a hinge body (b >= 2), in its order: the origin X and the parent-composed rotation q0 from
// the parent's pose and the body's constant frame (qb = c_quat + 4 b, pb = c_pos + 3 b); the hinge rotation ha at the
// joint angle (axis ax = c_axis + 3 b), which alone does not depend on the parent; the rest (Q = q0 ha, the twist, the
// accelerations). Split so that a sweep can compute fwd_hinge_rot (and load the constants) for every body before it
// starts (g1_engine.kinematics: the kinematics outputs, the same bits as fwd_body); fwd_body runs them as one
__device__ __forceinline__ void fwd_hinge_pos(const float* Qp, const float* Xp, const float* qb, const float* pb,
                                              float* q0, float* X) {
  float t[3];
  qmul_rn(Qp, qb, q0);
  qrot_rn(Qp, pb, t);
  for (int k = 0; k < 3; k++) X[k] = RADD(Xp[k], t[k]);
}
__device__ __forceinline__ void fwd_hinge_rot(float ang, const float* ax, float* ha) {
  float s, c;
  sincosf(RMUL(0.5f, ang), &s, &c);  // bitwise sinf, cosf with one argument reduction (V100: 4% faster forward)
  ha[0] = c; ha[1] = RMUL(s, ax[0]); ha[2] = RMUL(s, ax[1]); ha[3] = RMUL(s, ax[2]);
}
__device__ __forceinline__ void fwd_hinge_rest(const float* q0, const float* ha, const float* ax, float qd, float qdd,
                                               const float* Xp, const float* Wp, const float* Vp, const float* ALp,
                                               const float* Ap, float* Q, const float* X, float* W, float* V,
                                               float* AL, float* A) {
  qmul_rn(q0, ha, Q);
  float aw[3], awd[3], r[3], u[3], u2[3], wr[3];
  qrot_rn(Q, ax, aw);
  for (int k = 0; k < 3; k++) { r[k] = RSUB(X[k], Xp[k]); awd[k] = RMUL(aw[k], qd); W[k] = RADD(Wp[k], awd[k]); }
  cross_rn(Wp, r, u);
  for (int k = 0; k < 3; k++) V[k] = RADD(Vp[k], u[k]);
  cross_rn(Wp, awd, u);
  for (int k = 0; k < 3; k++) AL[k] = RADD(RFMA(aw[k], qdd, ALp[k]), u[k]);
  cross_rn(ALp, r, u);
  cross_rn(Wp, r, wr);
  cross_rn(Wp, wr, u2);
  for (int k = 0; k < 3; k++) A[k] = RADD(RADD(Ap[k], u[k]), u2[k]);
}

// forward kinematics and velocity / acceleration propagation of body b from its parent's state. Explicit roundings
// (RMUL ...): the forward (substep_pre, with qacc = 0) and the adjoint (adj_poses, at a*) compute the same bits
__device__ __forceinline__ void fwd_body(int b, const float* Qp, const float* Xp, const float* Wp, const float* Vp,
                                         const float* ALp, const float* Ap, const float* qpos, const float* qvel,
                                         const float* qacc, float* Q, float* X, float* W, float* V, float* AL,
                                         float* A) {
  if (b == 1) {  // free root: world position, quaternion, world linear / local angular velocity
    float n = rsqrtf(RFMA(qpos[6], qpos[6], RFMA(qpos[5], qpos[5], RFMA(qpos[3], qpos[3], RMUL(qpos[4], qpos[4])))));
    for (int k = 0; k < 4; k++) Q[k] = RMUL(qpos[3 + k], n);
    for (int k = 0; k < 3; k++) { X[k] = qpos[k]; V[k] = qvel[k]; A[k] = RSUB(qacc[k], c_grav[k]); }
    qrot_rn(Q, qvel + 3, W);
    qrot_rn(Q, qacc + 3, AL);
    return;
  }
  float q0[4], ha[4];
  fwd_hinge_pos(Qp, Xp, c_quat + 4 * b, c_pos + 3 * b, q0, X);
  float ang = qpos[c_qadr[b]], qd = qvel[c_dadr[b]], qdd = qacc[c_dadr[b]];
  fwd_hinge_rot(ang, c_axis + 3 * b, ha);
  fwd_hinge_rest(q0, ha, c_axis + 3 * b, qd, qdd, Xp, Wp, Vp, ALp, Ap, Q, X, W, V, AL, A);
}

// force F and moment N about the body origin that body b alone needs (Newton-Euler at its COM)
__device__ __forceinline__ void body_force(int b, const float* Q, const float* W, const float* AL, const float* A,
                                           const float* ipos, float* F, float* N) {
  float cvec[3], u[3], u2[3], wc[3];
  qrot(Q, ipos, cvec);
  cross(AL, cvec, u);
  cross(W, cvec, wc);
  cross(W, wc, u2);
  float m = c_mass[b];
  for (int k = 0; k < 3; k++) F[k] = m * (A[k] + u[k] + u2[k]);
  float qi[4], l[3], Iw[3], Ia[3], cf[3];
  qmul(Q, c_iquat + 4 * b, qi);
  qrot_inv(qi, W, l);
  for (int k = 0; k < 3; k++) l[k] *= c_inert[3 * b + k];
  qrot(qi, l, Iw);
  qrot_inv(qi, AL, l);
  for (int k = 0; k < 3; k++) l[k] *= c_inert[3 * b + k];
  qrot(qi, l, Ia);
  cross(W, Iw, u);
  cross(cvec, F, cf);
  for (int k = 0; k < 3; k++) N[k] = Ia[k] + u[k] + cf[k];
}

__device__ __forceinline__ void accumulate(const float* Xb, const float* Xc, const float* Fc, const float* Nc, float* F,
                                           float* N) {
  float r[3] = {Xc[0] - Xb[0], Xc[1] - Xb[1], Xc[2] - Xb[2]}, u[3];
  cross(r, Fc, u);
  for (int k = 0; k < 3; k++) { F[k] += Fc[k]; N[k] += Nc[k] + u[k]; }
}

__device__ __forceinline__ void torques(int b, const float* Q, const float* F, const float* N, float* qfrc) {
  if (b == 1) {
    float l[3];
    qrot_inv(Q, N, l);
    for (int k = 0; k < 3; k++) { qfrc[k] = F[k]; qfrc[3 + k] = l[k]; }
  } else {
    float aw[3];
    qrot(Q, c_axis + 3 * b, aw);
    qfrc[c_dadr[b]] = dot3(aw, N);
  }
}

// spatial inertia of body b about the point O, packed as J (xx yy zz xy xz yz), h = m c (3), m; c = com - O
__device__ __forceinline__ void body_inertia(int b, const float* Q, const float* X, const float* O, const float* ipos,
                                             float* J) {
  float qi[4], R[9], c[3], t[3];
  qmul(Q, c_iquat + 4 * b, qi);
  for (int k = 0; k < 3; k++) {
    float e[3] = {0.f, 0.f, 0.f};
    e[k] = 1.f;
    qrot(qi, e, R + 3 * k);  // column k of the inertia frame
  }
  const float* I = c_inert + 3 * b;
  J[0] = I[0] * R[0] * R[0] + I[1] * R[3] * R[3] + I[2] * R[6] * R[6];
  J[1] = I[0] * R[1] * R[1] + I[1] * R[4] * R[4] + I[2] * R[7] * R[7];
  J[2] = I[0] * R[2] * R[2] + I[1] * R[5] * R[5] + I[2] * R[8] * R[8];
  J[3] = I[0] * R[0] * R[1] + I[1] * R[3] * R[4] + I[2] * R[6] * R[7];
  J[4] = I[0] * R[0] * R[2] + I[1] * R[3] * R[5] + I[2] * R[6] * R[8];
  J[5] = I[0] * R[1] * R[2] + I[1] * R[4] * R[5] + I[2] * R[7] * R[8];
  qrot(Q, ipos, t);
  for (int k = 0; k < 3; k++) c[k] = X[k] + t[k] - O[k];
  float m = c_mass[b], cc = dot3(c, c);
  J[0] += m * (cc - c[0] * c[0]); J[1] += m * (cc - c[1] * c[1]); J[2] += m * (cc - c[2] * c[2]);
  J[3] -= m * c[0] * c[1]; J[4] -= m * c[0] * c[2]; J[5] -= m * c[1] * c[2];
  J[6] = m * c[0]; J[7] = m * c[1]; J[8] = m * c[2]; J[9] = m;
}

// (n, f) = I (w, v) for a packed spatial inertia about O; n = J w + h x v, f = m v - h x w
__device__ __forceinline__ void apply_inertia(const float* J, const float* w, const float* v, float* n, float* f) {
  float hv[3], hw[3];
  cross(J + 6, v, hv);
  cross(J + 6, w, hw);
  n[0] = J[0] * w[0] + J[3] * w[1] + J[4] * w[2] + hv[0];
  n[1] = J[3] * w[0] + J[1] * w[1] + J[5] * w[2] + hv[1];
  n[2] = J[4] * w[0] + J[5] * w[1] + J[2] * w[2] + hv[2];
  for (int k = 0; k < 3; k++) f[k] = J[9] * v[k] - hw[k];
}

// motion subspace of dof t of body k about O: w (angular), v (velocity of the point O)
__device__ __forceinline__ void dof_motion(int k, int t, const float* Q, const float* X, const float* O, float* w,
                                           float* v) {
  if (k == 1) {
    float e[3] = {0.f, 0.f, 0.f};
    if (t < 3) { e[t] = 1.f; w[0] = w[1] = w[2] = 0.f; v[0] = e[0]; v[1] = e[1]; v[2] = e[2]; return; }
    e[t - 3] = 1.f;
    qrot(Q, e, w);  // local angular axes about the root origin = O
    v[0] = v[1] = v[2] = 0.f;
    return;
  }
  float r[3] = {O[0] - X[0], O[1] - X[1], O[2] - X[2]};
  qrot(Q, c_axis + 3 * k, w);
  cross(w, r, v);
}

// COM offset of body b in its frame, with the per-world torso offset (domain randomization) applied
__device__ __forceinline__ void body_ipos(int b, const float* com, float* out) {
  for (int k = 0; k < 3; k++) out[k] = c_ipos[3 * b + k] + (b == (int)c_misc[1] ? com[k] : 0.f);
}

// in-place Cholesky A = L L^T (lower triangle of a dense NV x NV shared matrix) by one warp, left-looking (Crout):
// column k from the finished columns to its left; lane l computes rows k+1+l, k+33+l, ... as read-only dot products
// (row stride NV = 35 keeps the lanes on distinct banks; row k is a broadcast)
__device__ __forceinline__ void chol_warp(float* A, int lane) {
  for (int k = 0; k < NV; k++) {
    const float* Ak = A + k * NV;
    float part = 0.f;
    for (int j = lane; j < k; j += 32) part += Ak[j] * Ak[j];
    float d = sqrtf(fmaxf(Ak[k] - warp_sum(part), 1e-30f)), inv = 1.f / d;
    for (int i = k + 1 + lane; i < NV; i += 32) {
      const float* Ai = A + i * NV;
      float a0 = Ai[k], a1 = 0.f, a2 = 0.f, a3 = 0.f;
      int j = 0;
      for (; j + 4 <= k; j += 4) {
        a0 -= Ai[j] * Ak[j]; a1 -= Ai[j + 1] * Ak[j + 1]; a2 -= Ai[j + 2] * Ak[j + 2]; a3 -= Ai[j + 3] * Ak[j + 3];
      }
      for (; j < k; j++) a0 -= Ai[j] * Ak[j];
      A[i * NV + k] = ((a0 + a1) + (a2 + a3)) * inv;
    }
    __syncwarp();
    if (lane == 0) A[k * NV + k] = d;
    __syncwarp();
  }
}

// Cholesky factorization and solve in registers: L L^T = A (A: packed lower triangle TRI(i, j) in shared memory, read
// before anything is written, so L may alias A), then x = A^-1 b in place (b: shared NV vector). Lane l holds rows l and
// l + 32 (NV <= 64) in registers; the factorization is right-looking with column k broadcast by __shfl_sync, all loops
// unrolled at compile time. L (packed lower triangle, with the true diagonal) is written to L for the back substitution
// and for callers that keep it. With NV < 32 the lanes l >= NV hold no row (HAS0 false): their registers stay zero and
// they neither read nor write A, L or b; for NV >= 32 HAS0 is the constant true (the code is that of the NV >= 32 build).
#define HAS0(r0) (NV >= 32 || (r0) < NV)
#ifndef CHOL_INLINE
#define CHOL_INLINE __noinline__  // one copy of the (large, fully unrolled) code: -DCHOL_INLINE=__forceinline__ inlines it
#endif
__device__ CHOL_INLINE void chol_solve_reg(const float* A, float* L, float* b, int lane) {
  static_assert(NV <= 64, "two rows per lane");
  const int r0 = lane, r1 = lane + 32;
  const bool has0 = HAS0(r0), has1 = r1 < NV;
  float a0[NV], a1[NV], dinv[NV];
#pragma unroll
  for (int j = 0; j < NV; j++) {
    a0[j] = has0 && j <= r0 ? A[TRI(r0, j)] : 0.f;
    a1[j] = has1 && j <= r1 ? A[TRI(r1, j)] : 0.f;
  }
  __syncwarp();
#pragma unroll
  for (int k = 0; k < NV; k++) {
    float dkk = fmaxf(__shfl_sync(0xffffffffu, k < 32 ? a0[k] : a1[k], k & 31), 1e-30f);
    float inv = rsqrtf(dkk);
    inv = inv * (1.5f - 0.5f * dkk * inv * inv);  // one Newton step: about full float precision
    float d = dkk * inv;
    dinv[k] = inv;
    if (r0 > k) a0[k] *= inv; else if (r0 == k) a0[k] = d;
    if (r1 > k) a1[k] *= inv; else if (r1 == k) a1[k] = d;
#pragma unroll
    for (int j = k + 1; j < NV; j++) {  // unpredicated: above the diagonal this updates the symmetric Schur complement
      float ljk = __shfl_sync(0xffffffffu, j < 32 ? a0[k] : a1[k], j & 31);  // (bounded, never read as L)
      a0[j] -= a0[k] * ljk;
      a1[j] -= a1[k] * ljk;
    }
  }
#pragma unroll
  for (int j = 0; j < NV; j++) { if (has0 && j <= r0) L[TRI(r0, j)] = a0[j]; if (has1 && j <= r1) L[TRI(r1, j)] = a1[j]; }
  float y0 = has0 ? b[r0] : 0.f, y1 = has1 ? b[r1] : 0.f;
#pragma unroll
  for (int k = 0; k < NV; k++) {  // forward: L y = b, column-wise
    float yk = __shfl_sync(0xffffffffu, k < 32 ? y0 : y1, k & 31) * dinv[k];
    if (r0 == k) y0 = yk; else if (r0 > k) y0 -= a0[k] * yk;
    if (r1 == k) y1 = yk; else if (r1 > k) y1 -= a1[k] * yk;
  }
  __syncwarp();
  // backward: L^T x = y; lane i holds y_i and reads row k of L (contiguous) to update its entries i < k
#pragma unroll
  for (int k = NV - 1; k >= 0; k--) {
    float xk = __shfl_sync(0xffffffffu, k < 32 ? y0 : y1, k & 31) * dinv[k];
    if (r0 == k) y0 = xk; else if (r0 < k) y0 -= L[TRI(k, r0)] * xk;
    if (r1 == k) y1 = xk; else if (r1 < k) y1 -= L[TRI(k, r1)] * xk;
  }
  if (has0) b[r0] = y0;
  if (has1) b[r1] = y1;
  __syncwarp();
}

// Compact variant of chol_solve_reg (same contract): the column loop is not unrolled. Lane registers a0 / a1 hold the
// trailing part of rows lane / lane + 32, shifted left by one entry per step (a0[m] = row(k + m) at step k), so every
// register index stays a compile-time constant; finished columns of L go to shared memory as they are produced. The
// trailing update runs in groups of 8 columns and stops after the last group that holds a column. Small code (no
// instruction-fetch stalls) at the price of more executed instructions than the fully unrolled version.
__device__ __noinline__ void chol_solve_compact(const float* A, float* L, float* b, int lane) {
  static_assert(NV <= 64, "two rows per lane");
  const int r0 = lane, r1 = lane + 32;
  const bool has0 = HAS0(r0), has1 = r1 < NV;
  float a0[NV], a1[NV];
#pragma unroll
  for (int j = 0; j < NV; j++) {
    a0[j] = has0 && j <= r0 ? A[TRI(r0, j)] : 0.f;
    a1[j] = has1 && j <= r1 ? A[TRI(r1, j)] : 0.f;
  }
  __syncwarp();
  for (int k = 0; k < NV; k++) {
    float dkk = fmaxf(__shfl_sync(0xffffffffu, k < 32 ? a0[0] : a1[0], k & 31), 1e-30f);
    float inv = rsqrtf(dkk);
    inv = inv * (1.5f - 0.5f * dkk * inv * inv);
    float d = dkk * inv;
    float l0 = r0 > k ? a0[0] * inv : (r0 == k ? d : 0.f), l1 = r1 > k ? a1[0] * inv : (r1 == k ? d : 0.f);
    if (has0 && r0 >= k) L[TRI(r0, k)] = l0;
    if (has1 && r1 >= k) L[TRI(r1, k)] = l1;
    if (r0 == k) l0 = 0.f;  // the pivot row takes no part in the trailing update
    if (r1 == k) l1 = 0.f;
#pragma unroll
    for (int g = 1; g < NV; g += 8) {
      if (k + g >= NV) break;  // uniform: no column left in this group
#pragma unroll
      for (int m = g; m < g + 8 && m < NV; m++) {
        int j = k + m;
        float ljk = __shfl_sync(0xffffffffu, j < 32 ? l0 : l1, j & 31);
        a0[m - 1] = a0[m] - l0 * ljk;
        a1[m - 1] = a1[m] - l1 * ljk;
      }
    }
  }
  __syncwarp();
  float y0 = has0 ? b[r0] : 0.f, y1 = has1 ? b[r1] : 0.f;
  for (int k = 0; k < NV; k++) {  // forward: L y = b, column-wise
    float yk = __shfl_sync(0xffffffffu, k < 32 ? y0 : y1, k & 31) / L[TRI(k, k)];
    if (r0 == k) y0 = yk; else if (has0 && r0 > k) y0 -= L[TRI(r0, k)] * yk;
    if (r1 == k) y1 = yk; else if (has1 && r1 > k) y1 -= L[TRI(r1, k)] * yk;
  }
  for (int k = NV - 1; k >= 0; k--) {  // backward: L^T x = y
    float xk = __shfl_sync(0xffffffffu, k < 32 ? y0 : y1, k & 31) / L[TRI(k, k)];
    if (r0 == k) y0 = xk; else if (r0 < k) y0 -= L[TRI(k, r0)] * xk;
    if (r1 == k) y1 = xk; else if (has1 && r1 < k) y1 -= L[TRI(k, r1)] * xk;
  }
  if (has0) b[r0] = y0;
  if (has1) b[r1] = y1;
  __syncwarp();
}

// Broadcast variant (same contract, plus cb: 72 floats of 16-byte aligned shared scratch), for 32 < NV <= 35. The
// shuffles of the column broadcast (one per entry of the trailing update in chol_solve_reg / _compact) are the cost of
// those two; here column k goes through shared memory once (one store per lane) and is read back as float4 broadcasts,
// four entries per load. Rows 0..31 are in registers (lane r: row r, shifted left by one entry per step as in
// chol_solve_compact); the NT = NV - 32 tail rows are held transposed (lane j: A[32 + t][j]) and their own NT x NT
// block redundantly on every lane, so a lane updates its row's entries left of column 32 only. The forward solve runs
// inside the factorization (lane k broadcasts y_k with column k), and the pivot chain does not wait for the broadcast:
// lane k + 1 forms its next diagonal from its own entries. The backward solve reads L back by rows (one load per step).
// L is the same factor as chol_solve_compact's (same update order), the solve multiplies by the refined reciprocal
// pivots.
// Cycles per factor + solve of a 35 x 35 matrix, 4096 worlds (a standalone microbenchmark): V100 at 3 per SM 8.2k
// (chol_solve_compact 31.6k, _reg 16.2k), RTX 3090 at 2 per SM 8.9k (51.9k, 19.3k); in the engine (V100) the
// qacc_smooth phase went from 44.8k to 10.8k cycles.
#if NV > 32 && NV <= 35
#define CHOL_NT (NV - 32)  // tail rows
// The arithmetic below is written with explicit roundings (FMS: c - a b in one fma; MUL): ptxas otherwise fuses
// mul / add pairs depending on the calling kernel, and k_substep, k_pre and k_post must agree bitwise
#define FMS(a, b, c) __fmaf_rn(-(a), (b), (c))
#define MUL(a, b) __fmul_rn((a), (b))
// rsqrtf for a normal argument (x >= 2^-126): the bare MUFU.RSQ without rsqrtf's scaling of denormals (same result)
__device__ __forceinline__ float rsqrt_normal(float x) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
// backward substitution L' x = y on rows 0..31 in blocks of 8 (lane r: y_r in, x_r out; the contributions of rows 32..
// already subtracted): a block's own triangle is solved redundantly on every lane (broadcast loads of L; rinv: the
// reciprocal pivots in shared memory), then each lane updates its row: one round of shuffles per block, not per row
__device__ __forceinline__ float chol_back32(const float* L, const float* rinv, float y, int lane) {
#pragma unroll
  for (int kb = 24; kb >= 0; kb -= 8) {
    float xb[8];
#pragma unroll
    for (int j = 0; j < 8; j++) xb[j] = __shfl_sync(0xffffffffu, y, kb + j);
#pragma unroll
    for (int j = 7; j >= 0; j--) {
      float t = xb[j];
#pragma unroll
      for (int i = j + 1; i < 8; i++) t = FMS(L[TRI(kb + i, kb + j)], xb[i], t);
      xb[j] = MUL(t, rinv[kb + j]);
    }
#pragma unroll
    for (int j = 0; j < 8; j++)
      y = lane < kb ? FMS(L[TRI(kb + j, min(lane, kb + j))], xb[j], y) : (lane == kb + j ? xb[j] : y);
  }
  return y;
}

__device__ __noinline__ void chol_solve_bcast(const float* A, float* L, float* b, int lane, float* cb) {
  static_assert(NV > 32 && NV <= 35, "a tail of 1..3 rows");
  const int r = lane;
  float a[32];  // a[m] = A[r][k + m] at step k (entries right of the diagonal: bounded garbage, never read as L)
#pragma unroll
  for (int j = 0; j < 32; j++) a[j] = j <= r ? A[TRI(r, j)] : 0.f;
  float T[CHOL_NT], tb[CHOL_NT * (CHOL_NT + 1) / 2], bt[CHOL_NT];  // A[32 + t][r]; the tail block (packed); b tail
#pragma unroll
  for (int t = 0; t < CHOL_NT; t++) {
    T[t] = A[TRI(32 + t, r)];
    bt[t] = b[32 + t];
#pragma unroll
    for (int u = 0; u <= t; u++) tb[t * (t + 1) / 2 + u] = A[TRI(32 + t, 32 + u)];
  }
  float y = b[r], myinv = 0.f;
  __syncwarp();
  float dn = a[0];  // the next pivot (lane k at step k): the diagonal after the updates so far
  // unrolled by 2 (unroll 1: V100 13.7 against 12.7 ms per control step, RTX 3090 7.59 against 7.46; a test per group
  // of the row update, or phases with fewer groups as the trailing part shrinks, i.e. more code: V100 13.0 / 15.2 ms)
#pragma unroll 2
  for (int k = 0; k < 32; k++) {
    float* buf = cb + (k & 1) * 36;  // double buffered: [0, 4) L[32 + t][k], y_k; 4 + (j - k - 1): L[j][k], k < j < 32
    float dkk = fmaxf(__shfl_sync(0xffffffffu, dn, k), 1e-30f);
    float inv = rsqrt_normal(dkk);
    inv = MUL(inv, FMS(MUL(MUL(0.5f, dkk), inv), inv, 1.5f));  // one Newton step: about full float precision
    float d = MUL(dkk, inv);
    float l = r > k ? MUL(a[0], inv) : (r == k ? d : 0.f);
    if (r >= k) L[TRI(r, k)] = l;
    if (r == k) {
      myinv = inv;
      y = MUL(y, inv);
      float tw[4] = {0.f, 0.f, 0.f, y};
#pragma unroll
      for (int t = 0; t < CHOL_NT; t++) { T[t] = MUL(T[t], inv); L[TRI(32 + t, k)] = T[t]; tw[t] = T[t]; }
      *reinterpret_cast<float4*>(buf) = make_float4(tw[0], tw[1], tw[2], tw[3]);
      l = 0.f;  // the pivot row takes no part in the trailing update
    }
    if (r > k) buf[4 + r - k - 1] = l;
    dn = FMS(l, l, a[1]);  // lane k + 1: its next diagonal (= a[0] after the shift below)
    __syncwarp();
    float4 tv = *reinterpret_cast<const float4*>(buf);
    float tl[3] = {tv.x, tv.y, tv.z};
    y = FMS(l, tv.w, y);
#pragma unroll
    for (int t = 0; t < CHOL_NT; t++) {
      T[t] = FMS(tl[t], l, T[t]);
      bt[t] = FMS(tl[t], tv.w, bt[t]);
#pragma unroll
      for (int u = 0; u <= t; u++) tb[t * (t + 1) / 2 + u] = FMS(tl[t], tl[u], tb[t * (t + 1) / 2 + u]);
    }
#pragma unroll
    for (int g = 1; g < 32; g += 4) {  // all groups every step (skipping the second half once k >= 16, one uniform
      float4 c = *reinterpret_cast<const float4*>(buf + 4 + g - 1);  // branch, was slower: RTX 3090 9.9k against 8.9k)
      a[g - 1] = FMS(l, c.x, a[g]);
      if (g + 1 < 32) a[g] = FMS(l, c.y, a[g + 1]);
      if (g + 2 < 32) a[g + 1] = FMS(l, c.z, a[g + 2]);
      if (g + 3 < 32) a[g + 2] = FMS(l, c.w, a[g + 3]);
    }
  }
  float dt[CHOL_NT], it[CHOL_NT], xt[CHOL_NT];  // the tail block's factor and solve, on every lane
#pragma unroll
  for (int t = 0; t < CHOL_NT; t++) {
    float s = tb[t * (t + 1) / 2 + t];
#pragma unroll
    for (int u = 0; u < t; u++) s = FMS(tb[t * (t + 1) / 2 + u], tb[t * (t + 1) / 2 + u], s);
    float dkk = fmaxf(s, 1e-30f), inv = rsqrt_normal(dkk);
    inv = MUL(inv, FMS(MUL(MUL(0.5f, dkk), inv), inv, 1.5f));
    dt[t] = MUL(dkk, inv); it[t] = inv;
#pragma unroll
    for (int v = t + 1; v < CHOL_NT; v++) {
      float e = tb[v * (v + 1) / 2 + t];
#pragma unroll
      for (int u = 0; u < t; u++) e = FMS(tb[v * (v + 1) / 2 + u], tb[t * (t + 1) / 2 + u], e);
      tb[v * (v + 1) / 2 + t] = MUL(e, inv);
    }
    float yt = bt[t];
#pragma unroll
    for (int u = 0; u < t; u++) yt = FMS(tb[t * (t + 1) / 2 + u], xt[u], yt);
    xt[t] = MUL(yt, inv);  // forward
  }
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < CHOL_NT; t++) {
#pragma unroll
      for (int u = 0; u < t; u++) L[TRI(32 + t, 32 + u)] = tb[t * (t + 1) / 2 + u];
      L[TRI(32 + t, 32 + t)] = dt[t];
    }
  }
#pragma unroll
  for (int t = CHOL_NT - 1; t >= 0; t--) {  // backward on the tail
    float x = xt[t];
#pragma unroll
    for (int v = t + 1; v < CHOL_NT; v++) x = FMS(tb[v * (v + 1) / 2 + t], xt[v], x);
    xt[t] = MUL(x, it[t]);
  }
#pragma unroll
  for (int t = 0; t < CHOL_NT; t++) y = FMS(T[t], xt[t], y);
  __syncwarp();
  // backward on rows 0..31: L' x = y, column by column, unrolled by 4 (fully: V100 12.73 against 12.44 ms, RTX 3090 the
  // same; chol_back32 was slower here)
#pragma unroll 4
  for (int k = 31; k >= 0; k--) {
    float xk = __shfl_sync(0xffffffffu, MUL(y, myinv), k);
    float lk = r < k ? L[TRI(k, r)] : 0.f;
    y = r == k ? xk : FMS(lk, xk, y);
  }
  b[r] = y;
  if (lane == 0) {  // (lane t writing entry t became a dynamically indexed local array)
#pragma unroll
    for (int t = 0; t < CHOL_NT; t++) b[32 + t] = xt[t];
  }
  __syncwarp();
}

// x = (L L')^-1 b in place (b: shared NV vector) with a factor L from CHOL_SOLVE (packed lower triangle in shared
// memory, true diagonal), for a Newton iteration whose Hessian did not change; cb: scratch as for chol_solve_bcast.
// Blocks of 8 rows as in chol_back32 (forward likewise), the tail rows 32.. on every lane. V100: 3.0k cycles (column by
// column with one broadcast per row: 6.3k)
__device__ __noinline__ void chol_resolve(const float* L, float* b, int lane, float* cb) {
  const int r = lane;
  float* rinv = cb;
  rinv[r] = 1.f / L[TRI(r, r)];
  if (r < CHOL_NT) rinv[32 + r] = 1.f / L[TRI(32 + r, 32 + r)];
  const int r1 = min(32 + r, NV - 1);  // lanes 0 .. CHOL_NT - 1: tail row 32 + r (y1); others a valid dummy row
  float y = b[r], y1 = b[r1];
  const int tr = TRI(r, 0), tr1 = TRI(r1, 0);
  __syncwarp();
#pragma unroll
  for (int kb = 0; kb < 32; kb += 8) {  // forward: L y = b on rows 0..31
    float yb[8];
#pragma unroll
    for (int j = 0; j < 8; j++) yb[j] = __shfl_sync(0xffffffffu, y, kb + j);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float t = yb[j];
#pragma unroll
      for (int i = 0; i < j; i++) t = FMS(L[TRI(kb + j, kb + i)], yb[i], t);
      yb[j] = MUL(t, rinv[kb + j]);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
      y = r >= kb + 8 ? FMS(L[tr + min(kb + j, r)], yb[j], y) : (r == kb + j ? yb[j] : y);
      y1 = FMS(L[tr1 + kb + j], yb[j], y1);
    }
  }
  float yt[CHOL_NT];  // the tail rows on every lane: forward, backward, then their part of rows 0..31
#pragma unroll
  for (int t = 0; t < CHOL_NT; t++) {
    float u = __shfl_sync(0xffffffffu, y1, t);
#pragma unroll
    for (int i = 0; i < t; i++) u = FMS(L[TRI(32 + t, 32 + i)], yt[i], u);
    yt[t] = MUL(u, rinv[32 + t]);
  }
#pragma unroll
  for (int t = CHOL_NT - 1; t >= 0; t--) {
    float u = yt[t];
#pragma unroll
    for (int i = t + 1; i < CHOL_NT; i++) u = FMS(L[TRI(32 + i, 32 + t)], yt[i], u);
    yt[t] = MUL(u, rinv[32 + t]);
  }
#pragma unroll
  for (int t = 0; t < CHOL_NT; t++) y = FMS(L[TRI(32 + t, r)], yt[t], y);
  y = chol_back32(L, rinv, y, lane);
  b[r] = y;
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < CHOL_NT; t++) b[32 + t] = yt[t];
  }
  __syncwarp();
}
#undef FMS
#undef MUL
#endif

// which Cholesky the engine uses: CHOL_SOLVE(A, L, b, lane, cb) (cb: scratch for chol_solve_bcast). -DCHOL_BCAST=1
// (the default where NV allows it) selects chol_solve_bcast (V100 20.5 -> 12.4 ms per mjlab control step at 10
// iterations, RTX 3090 11.1 -> 7.5 ms, with NEWTON_REUSE); with -DCHOL_BCAST=0, -DCHOL_COMPACT=1 / =0 forces one of the
// others, by default the compact one on Volta (V100: 26 vs 34 ms) and the unrolled one from Ampere on (RTX 3090: 14.7
// vs 17.9 ms)
#ifndef CHOL_BCAST
#define CHOL_BCAST (NV > 32 && NV <= 35)
#endif
#ifndef CHOL_COMPACT
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define CHOL_COMPACT 1
#else
#define CHOL_COMPACT 0
#endif
#endif
#if CHOL_BCAST
#define CHOL_SOLVE(A, L, b, lane, cb) chol_solve_bcast(A, L, b, lane, cb)
#elif CHOL_COMPACT
#define CHOL_SOLVE(A, L, b, lane, cb) chol_solve_compact(A, L, b, lane)
#else
#define CHOL_SOLVE(A, L, b, lane, cb) chol_solve_reg(A, L, b, lane)
#endif

// solve L L^T x = b in place (b in shared memory) with the factor from chol_warp
__device__ __forceinline__ void chol_solve_warp(const float* L, float* b, int lane) {
  for (int k = 0; k < NV; k++) {  // forward: L y = b
    if (lane == 0) b[k] /= L[k * NV + k];
    __syncwarp();
    for (int i = k + 1 + lane; i < NV; i += 32) b[i] -= L[i * NV + k] * b[k];
    __syncwarp();
  }
  for (int k = NV - 1; k >= 0; k--) {  // backward: L^T x = y
    if (lane == 0) b[k] /= L[k * NV + k];
    __syncwarp();
    for (int i = lane; i < k; i += 32) b[i] -= L[k * NV + i] * b[k];
    __syncwarp();
  }
}
"""
