# The collision-primitive VJPs here differentiate the primitives step.py translated from MuJoCo Warp
# (mujoco_warp/_src/collision_primitive_core.py, math.make_frame; Copyright 2025 The Newton Developers, Apache License
# 2.0), and follow the conventions of MuJoCo Warp's adjoint (google-deepmind/mujoco_warp pull request 1535:
# collision_adjoint.py, adjoint_util.py, constraint_adjoint.py) so that they can be checked against it; the derivations
# are this engine's.
"""CUDA device functions of the contact side of the analytic adjoint: impedance derivatives, the joint-limit row VJP,
the contact row VJP Psi_c, the per-dof pose cotangents of a contact Jacobian column, and the collision-primitive VJPs
of the mjlab model (plane-sphere, plane-capsule, sphere-sphere, sphere-capsule, capsule-capsule) with make_frame, the
normal option DK_ADJ_NORMAL and the near-parallel capsule option DK_ADJ_PARALLEL.

All functions are pure: pointers and scalars in, pointers out (no struct Sim, no struct Adj), meant for one lane per
contact or row; no warp intrinsics, no shared state. They need source.DEVICE (qrot, cross, dot3, the *_rn helpers) and
step.STEP_DEVICE (kbimp, imp_raw, efc_D, contact_invw, sphere_sphere, make_frame and the primitives' geometry that
collide_pair uses: plane_point, plane_capsule_frame, capsule_end, segment_point, capcap_*) before them in the compilation
unit, and header(m) (which asserts the
model: no box contacts). Unit-test labels: U4-U9 (docs/verification.md).

Conventions:
- P = grad Psi with MuJoCo Warp's res_* sign (the adjoint kernel subtracts it: qbar = qbar_dir - P). Every "bar"
  below is a cotangent dL/d(.) of the residual term L = Psi, so dbar, pbar, frbar are MuJoCo Warp's contact_adj_dist
  (the efc_pos cotangent of the contact's first row), res_contact_pos and res_contact_frame, and W_v, W_a, W_lam the
  linear parts of contact_V.grad / contact_A.grad / contact_Z.grad.
- Contact frame fr[9]: rows n (from geom 1 towards geom 2), t1, t2 (step.py make_frame, or the capsule-axis frame of a
  plane-capsule contact). Pyramid rows e = 0..3 of a condim-3 contact: u_e = n + mu t1, n - mu t1, n + mu t2, n - mu t2
  (the forward's row order: sign (e & 1), tangent 1 + (e >> 1)); condim 1: u_0 = n.
- Contact-point motion V (3-vector, world): J_p q' of the contact point moving with body 2 relative to body 1, i.e. the
  linear part of MuJoCo Warp's spatial contact motion (angular first there); the angular part Omega = sum over the
  rotational support dofs of s_i z_i q'_i enters only contact_pbar_motion.
- Pose cotangents (fbar, tbar) of a body or geom: dL = fbar . dX + tbar . dOmega, X the body origin (geom: its centre)
  in world coordinates, dR = [dOmega]x R (world-frame rotation). A rotation-matrix cotangent Rbar converts to
  tbar = sum_k R[:, k] x Rbar[:, k] (MuJoCo Warp's adjoint_util._adj_rotation); a body-fixed vector y = R c gives
  tbar += y x ybar.
- Geom pose cotangents of a contact, gb[12] = (xbar_1, tbar_1, xbar_2, tbar_2) for the pair's geoms g1, g2 in the
  engine's order (lower type first, step.pair_constants); the plane (the static floor) gets zeros.

Use in the adjoint kernel (its phases A5 / A6, adjoint.py), per substep with the record's input state replayed
(body poses Q, X, geom poses gx, gq from geom_poses), h = c_misc[0]:
- contact c of the tape's table (pair, slot), rows r0 .. r0 + nr - 1 (r0 = nlim + rows of the contacts before c,
  nr = 4 for condim 3, else 1), zone = its nr zone bits (bit 0 = row r0):
  1. br = adj_pair_geom(pair, slot, gx, gq, &d, pos, fr, &nu, nullptr, nullptr)   (replay: the forward's d, pos, fr)
  2. mu = PFOOT(pair) ? fric : PMU(pair); invw = contact_invw(pair, mu) (c_binvw[b1] + c_binvw[b2], for condim 3 the
     pyramid scaling: substep_pre's, so D is bitwise the forward's);
     kbimp_d(PSOLREF(pair), PSOLIMP(pair), d, h, &k, &b, &imp, &dimp); efc_D_d(invw, imp, dimp, &D, &dD)
  3. V_v, V_a, V_lam = sum over the support dofs i of s_i col_i (v_i, a*_i, lam_i) (and, for contact_pbar_motion, the
     angular Om_* = sum over the rotational ones of s_i z_i (v_i, a*_i, lam_i))
  4. contact_psi_vjp(V_v, V_a, V_lam, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, W_v, W_a, W_lam, &dbar, frbar)
  5. per support dof: P_v,i += contact_dof_vjp(...) (adds the dof body's fbar, tbar and pbar), or the same G formula in
     a lane-per-dof reduction with pbar from contact_pbar_motion
  6. cot = (dbar, pbar, frbar); adj_pair_geom(pair, slot, gx, gq, &d, pos, fr, &nu, cot, gb); geom_to_body_vjp of
     gb + 0 (geom g1, unless its body is the world) and gb + 6 (g2) onto (fbar, tbar) of their bodies
- joint-limit row of hinge dof i in its zone (pos, sigma as substep_pre): kbimp_d(c_jsolref, c_jsolimp, pos, h, ...),
  efc_D_d(c_dinvw[i], ...), limit_row_vjp(sigma, pos, a*_i, v_i, lam_i, ...) -> P_phi,i, P_v,i
All P terms then enter P_q (the pose sweep, A7) and P_v with MuJoCo Warp's res_* sign.
"""

import mujoco

from nablaengine.g1_engine import step


def check_model(m):
    """The contact adjoint's collision geoms are planes, spheres and capsules (no boxes)."""
    types = {int(m.geom_type[g]) for g in step.collision_geoms(m)}
    from nablaengine.g1_engine import model_hooks
    has_box, box_ok, _ = model_hooks.box_pairs(m)
    assert not has_box or box_ok, "the adjoint has box contacts with planes and boxes only (model_hooks)"
    assert {step.GEOM_TYPE[t] for t in types} <= {0, 1, 2, 3}, "collision geoms other than plane / sphere / capsule / box"
    planes = [g for g in step.collision_geoms(m) if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)]
    assert all(int(m.geom_bodyid[g]) == 0 for g in planes), "a plane not on the world body (it gets no cotangent)"


def header(m):
    """#defines and checks of the contact adjoint for model m (asserts check_model); after step.step_header(m)."""
    check_model(m)
    return r"""
#ifndef DK_ADJ_NORMAL  // default 3: frozen below eps (finite gradients, no 1 / nu singularity)
#define DK_ADJ_NORMAL 3  // normal derivative of the sphere core: 0 exact (MuJoCo Warp), 1 frozen frame,
#endif                   // 2 regularized kappa = 1 / max(nu, eps), 3 frozen below eps (kappa = nu < eps ? 0 : 1 / nu)
#ifndef DK_ADJ_NORMAL_EPS
#define DK_ADJ_NORMAL_EPS 1e-3f  // m, the eps of DK_ADJ_NORMAL 2 and 3 (strict <)
#endif
#ifndef DK_ADJ_PARALLEL  // near-parallel capsule-capsule with both segment parameters free: 1 holds the
#define DK_ADJ_PARALLEL (DK_ADJ_NORMAL != 0)  // parameters constant where sin^2(theta) < DK_ADJ_PARALLEL_EPS (finite);
#endif                                        // 0 exact (MuJoCo Warp). Default: on with a normal option, so the exact
                                              // gradient mode (DK_ADJ_NORMAL 0) stays exact
#ifndef DK_ADJ_PARALLEL_EPS
#define DK_ADJ_PARALLEL_EPS 1.2e-3f  // sin^2(theta) = |a1 x a2|^2 / (|a1|^2 |a2|^2) of DK_ADJ_PARALLEL 1 (strict <):
#endif                               // ~sin^2(2 degrees) (theta < 1.99 degrees)
static_assert(!HAS_BOX, "the contact adjoint has no box collisions");
// adj_contact_geom's branch bits (its return value; -1: a type combination without a VJP)
#define ADJ_BR_FREE1 1      // capsule-capsule: segment parameter x1 free (inside, not clamped)
#define ADJ_BR_FREE2 2      // capsule-capsule x2 / sphere-capsule t free
#define ADJ_BR_PARALLEL 4   // capsule-capsule parallel branch (sin^2 <= 1e-8), the tape's slot = the endpoint
#define ADJ_BR_FALLBACK 8   // plane-capsule: the projected axis is shorter than 0.5, the frame uses make_frame's basis
#define ADJ_BR_FRAME_Z 16   // the frame's basis vector is e_z (|n_y| >= 0.5), else e_y
"""


DEVICE = r"""
// ---------------------------------------------------------------- impedance

// kbimp with the derivative of the impedance: k, b, imp from the forward's kbimp() itself (bitwise the same), and
// dimp = d imp / d pos on the branch kbimp took: sign(pos) imp_x / width, imp_x = (dmax - dmin) p y^(p-1) / m^(p-1)
// with (y, m) = (x, mid) for x <= mid and (1 - x, 1 - mid) above (x = |pos| / width); 0 where imp is constant (x <= 0,
// x >= 1, dmin == dmax) and where the forward's clamp to [dmin, dmax] changes the value (raw outside [dmin, dmax]; it
// only guards rounding). MuJoCo Warp's wp.clamp also drops the gradient when its float32 raw equals dmax, which happens
// for x within about 5e-4 of 1 (mjlab solimp; there imp_x is below 1e-3 of its peak): that float32 artifact is not
// copied. pow_imp: no powf for p = 1, 2
__device__ __forceinline__ void kbimp_d(const float* solref, const float* solimp, float pos, float h, float* k,
                                        float* b, float* imp, float* dimp) {
  kbimp(solref, solimp, pos, h, k, b, imp);
  float dmin = fminf(fmaxf(solimp[0], 1e-4f), 0.9999f), dmax = fminf(fmaxf(solimp[1], 1e-4f), 0.9999f);
  float width = fmaxf(1e-15f, solimp[2]), mid = fminf(fmaxf(solimp[3], 1e-4f), 0.9999f), power = fmaxf(1.f, solimp[4]);
  float x = fabsf(pos) / width, d = 0.f;
  if (!(dmin == dmax || solimp[2] <= 1e-15f) && x > 0.f && x < 1.f) {
    if (power == 1.f) d = dmax - dmin;
    else {
      bool lo = x <= mid;  // the forward's raw (unclamped) value on this branch, for the clamp test
      float raw = imp_raw(x, dmin, dmax, mid, power, lo);
      // 1 - x as (width - |pos|) / width: no cancellation of x's rounding near x = 1 (the derivative only)
      float y = lo ? x : (width - fabsf(pos)) / width, ym = lo ? mid : 1.f - mid;
      if (raw >= dmin && raw <= dmax) d = power * pow_imp(y, power - 1.f) / pow_imp(ym, power - 1.f) * (dmax - dmin);
    }
  }
  *dimp = (pos < 0.f ? -d : d) / width;
}

// row impedance D = 1 / max(R, 1e-15), R = invw (1 - imp) / imp (the forward's expression in substep_pre), and its
// derivative dD = dimp / (invw (1 - imp)^2) (0 where the floor acts): with kbimp_d's dimp, dD = dD / dpos
__device__ __forceinline__ void efc_D_d(float invw, float imp, float dimp, float* D, float* dD) {
  float R = invw * (1.f - imp) / imp, om = 1.f - imp;
  *D = efc_D(invw, imp);  // the forward's (step.py)
  *dD = R > 1e-15f ? dimp / (invw * om * om) : 0.f;
}

// ---------------------------------------------------------------- joint limits

// one joint-limit row of hinge dof i in its quadratic zone (the tape's zone bit; inactive rows contribute nothing).
// sigma = +1 at the lower limit (pos = q_i - lo, J = +e_i), -1 at the upper (pos = hi - q_i, J = -e_i); a, v, lam: the
// dof's a*, qvel and lambda; k, b, imp, dimp from kbimp_d(c_jsolref, c_jsolimp, pos), D, dD from
// efc_D_d(c_dinvw[i], ...). x = J a* - aref = sigma a + k imp pos + b sigma v, Psi_e = D x sigma lam. Out: the row's
// terms of P_phi,i = dPsi_e / dq_i (through pos; MuJoCo Warp's res_dof of the row) and P_v,i = dPsi_e / dv_i (res_qvel)
__device__ __forceinline__ void limit_row_vjp(float sigma, float pos, float a, float v, float lam, float k, float b,
                                              float imp, float dimp, float D, float dD, float* Pphi, float* Pv) {
  float x = sigma * a + k * imp * pos + b * (sigma * v);
  *Pphi = lam * (dD * x + D * (k * (imp + dimp * pos)));
  *Pv = D * b * lam;
}

// ---------------------------------------------------------------- contact rows

// Psi_c VJP of one contact. In: the contact point's motions V_v = J_p v, V_a = J_p a*, V_lam = J_p lam (world, body 2
// relative to body 1), the frame fr (rows n, t1, t2), friction mu, condim dim (1 or 3), zone (bit e: row e on its
// quadratic zone, the tape's bits of the contact's rows), the distance d and its rows' impedance (one D for all rows of
// a contact): k, b, imp, dimp from kbimp_d(PSOLREF, PSOLIMP, d), D, dD from efc_D_d(invw_c, ...).
// Psi_c = sum_{e in zone} D x_e (u_e . V_lam), x_e = u_e . V_a + k imp d + b u_e . V_v.
// Out: W_v, W_a, W_lam = dPsi_c / dV_* (W_v = b W_a),
// dbar = dPsi_c / dd = sum_e (u_e . V_lam) [dD x_e + D k (imp + dimp d)], frbar = dPsi_c / dfr (rows nbar, t1bar,
// t2bar; from ubar_e = D x_e V_lam + D (u_e . V_lam)(V_a + b V_v)). A contact with no row in its zone has Psi_c = 0 and
// all outputs zero (MuJoCo Warp's _contact_any_active).
__device__ __forceinline__ void contact_psi_vjp(const float* Vv, const float* Va, const float* Vl, const float* fr,
                                                float mu, int dim, unsigned zone, float d, float D, float dD, float k,
                                                float b, float imp, float dimp, float* Wv, float* Wa, float* Wl,
                                                float* dbar, float* frbar) {
  float vf[3], af[3], lf[3];  // frame components
#pragma unroll
  for (int j = 0; j < 3; j++) {
    vf[j] = dot3(fr + 3 * j, Vv); af[j] = dot3(fr + 3 * j, Va); lf[j] = dot3(fr + 3 * j, Vl);
  }
  float g = k * imp * d, gd = k * (imp + dimp * d);  // the reference acceleration's position term and its d-derivative
  float cl[3] = {0.f, 0.f, 0.f}, ca[3] = {0.f, 0.f, 0.f}, db = 0.f;  // frame components of W_lam, W_a
#pragma unroll
  for (int e = 0; e < 4; e++) {
    if (!((zone >> e) & 1u) || (e > 0 && dim != 3)) continue;
    const int t = 1 + (e >> 1);
    float sm = (e & 1) ? -mu : mu;  // u_e = n + sm t_t (condim 1: n)
    float uv = vf[0], ua = af[0], ul = lf[0];
    if (dim == 3) { uv += sm * vf[t]; ua += sm * af[t]; ul += sm * lf[t]; }
    float x = ua + g + b * uv, Dx = D * x, Dl = D * ul;
    cl[0] += Dx; ca[0] += Dl;
    if (dim == 3) { cl[t] += sm * Dx; ca[t] += sm * Dl; }
    db += ul * (dD * x + D * gd);
  }
#pragma unroll
  for (int i = 0; i < 3; i++) {
    Wl[i] = cl[0] * fr[i] + cl[1] * fr[3 + i] + cl[2] * fr[6 + i];
    Wa[i] = ca[0] * fr[i] + ca[1] * fr[3 + i] + ca[2] * fr[6 + i];
    Wv[i] = b * Wa[i];
  }
  float ab[3] = {Va[0] + b * Vv[0], Va[1] + b * Vv[1], Va[2] + b * Vv[2]};
#pragma unroll
  for (int j = 0; j < 3; j++)
#pragma unroll
    for (int i = 0; i < 3; i++) frbar[3 * j + i] = cl[j] * Vl[i] + ca[j] * ab[i];
  *dbar = db;
}

// one support dof i of a contact: s = +1 on body 2's chain, -1 on body 1's (the dofs on exactly one chain); col = the
// dof's point-Jacobian column at the contact point p (rotational dof: z x (p - x), z its world axis, x its anchor: the
// origin X_b(i) of its body, X_1 for the three root rotations; root translation k: e_k, rot = false); v, a, lam: the
// dof's qvel, a* and lambda; W_* from contact_psi_vjp. Returns the dof's P_v term s col . W_v. With
// G = s (v W_v + a W_a + lam W_lam) = dPsi_c / dcol (MuJoCo Warp's _contact_scatter G), a rotational dof adds the pose
// cotangents of its body (root rotations: body 1) fbar += z x G, tbar += z x ((p - x) x G), and pbar += G x z (the
// contact point's cotangent through the Jacobian; contact_pbar_motion gives its sum over the support in closed form)
__device__ __forceinline__ float contact_dof_vjp(float s, const float* col, float v, float a, float lam,
                                                 const float* Wv, const float* Wa, const float* Wl, bool rot,
                                                 const float* z, const float* x, const float* p, float* fbar,
                                                 float* tbar, float* pbar) {
  if (rot) {
    float G[3], r[3], u[3], w[3], t[3];
#pragma unroll
    for (int k = 0; k < 3; k++) { G[k] = s * (v * Wv[k] + a * Wa[k] + lam * Wl[k]); r[k] = p[k] - x[k]; }
    cross(z, G, u);
    cross(r, G, w);
    cross(z, w, t);
#pragma unroll
    for (int k = 0; k < 3; k++) { fbar[k] += u[k]; tbar[k] += t[k]; pbar[k] -= u[k]; }
  }
  return s * dot3(col, Wv);
}

// the contact point's cotangent through the Jacobian, summed over the support: sum_i G_i x z_i = W_v x Om_v
// + W_a x Om_a + W_lam x Om_lam with Om_* = sum over the rotational support dofs of s_i z_i (v_i, a*_i, lam_i): the
// angular parts of the contact's relative spatial motions (MuJoCo Warp's contact_V / A / Z top); adds to pbar
__device__ __forceinline__ void contact_pbar_motion(const float* Wv, const float* Wa, const float* Wl, const float* Omv,
                                                    const float* Oma, const float* Oml, float* pbar) {
  float u[3];
  cross(Wv, Omv, u);
  for (int k = 0; k < 3; k++) pbar[k] += u[k];
  cross(Wa, Oma, u);
  for (int k = 0; k < 3; k++) pbar[k] += u[k];
  cross(Wl, Oml, u);
  for (int k = 0; k < 3; k++) pbar[k] += u[k];
}

// ---------------------------------------------------------------- collision geometry

// kappa(nu) of the sphere core's normal derivative kappa (I - n n') (DK_ADJ_NORMAL)
__device__ __forceinline__ float adj_kappa(float nu) {
#if DK_ADJ_NORMAL == 0
  return 1.f / nu;  // exact: (I - n n') / nu, MuJoCo Warp
#elif DK_ADJ_NORMAL == 1
  return 0.f;  // frame frozen in the backward (distance and contact point keep their derivatives)
#elif DK_ADJ_NORMAL == 2
  return 1.f / fmaxf(nu, DK_ADJ_NORMAL_EPS);  // regularized: exact for nu >= eps, continuous in nu
#else
  return nu < DK_ADJ_NORMAL_EPS ? 0.f : 1.f / nu;  // frozen below eps only
#endif
}

// kappa(nu) nu: the factor of the normal's derivative where it is written without 1 / nu (capsule-capsule, both
// parameters free); 1 for the exact derivative
__device__ __forceinline__ float adj_kappa_nu(float nu) {
#if DK_ADJ_NORMAL == 0
  return 1.f;
#elif DK_ADJ_NORMAL == 1
  return 0.f;
#elif DK_ADJ_NORMAL == 2
  return nu / fmaxf(nu, DK_ADJ_NORMAL_EPS);
#else
  return nu < DK_ADJ_NORMAL_EPS ? 0.f : 1.f;
#endif
}

// make_frame VJP: fr = make_frame(n) (rows n, b, c; b = normalize(e - n (n . e)), e = e_y if -0.5 < n_y < 0.5 else e_z,
// c = n x b; n a unit vector), frbar = dL/d(n, b, c) -> nbar = dL/dn (the frame's own n row included):
// nbar = nbar_0 + b x cbar, btot = bbar + cbar x n, bt = (I - b b') btot / beta, nbar -= (n . e) bt + e (n . bt)
__device__ __forceinline__ void make_frame_vjp(const float* fr, const float* frbar, float* nbar) {
  const float *n = fr, *b = fr + 3, *cb = frbar + 6;
  bool ey = -0.5f < n[1] && n[1] < 0.5f;  // the forward's test
  float ne = ey ? n[1] : n[2], bt[3] = {-n[0] * ne, (ey ? 1.f : 0.f) - n[1] * ne, (ey ? 0.f : 1.f) - n[2] * ne};
  float beta = sqrtf(dot3(bt, bt)), u[3], w[3];
  cross(cb, n, u);
  for (int k = 0; k < 3; k++) u[k] += frbar[3 + k];  // btot
  float bu = dot3(b, u);
  for (int k = 0; k < 3; k++) bt[k] = (u[k] - b[k] * bu) / beta;
  cross(b, cb, w);
  float nb = dot3(n, bt);
  for (int k = 0; k < 3; k++) nbar[k] = frbar[k] + w[k] - ne * bt[k];
  nbar[ey ? 1 : 2] -= nb;
}

// sphere core VJP: (dist, pos, n) = sphere_sphere(c1, r1, c2, r2), i.e. nu = |c2 - c1|, n = (c2 - c1) / nu,
// dist = nu - r1 - r2, pos = c1 + n (r1 + dist / 2). In: n, nu, dist, r1 and the cotangents dbar, pbar, nbar (nbar: the
// normal's own, e.g. make_frame_vjp's). Adds c1bar, c2bar: v = c2 - c1, vbar = n dtot + kappa(nu) (I - n n') ntot with
// ntot = nbar + (r1 + dist / 2) pbar, dtot = dbar + (n . pbar) / 2; c2bar += vbar, c1bar += pbar - vbar. nu = 0
// (n = e_x): no derivative through v (MuJoCo Warp's safe_sqrt)
__device__ __forceinline__ void sphere_core_vjp(const float* n, float nu, float dist, float r1, float dbar,
                                                const float* pbar, const float* nbar, float* c1bar, float* c2bar) {
  float hp = r1 + 0.5f * dist, vb[3] = {0.f, 0.f, 0.f};
  if (nu > 0.f) {
    float nt[3], dt = dbar + 0.5f * dot3(n, pbar), kap = adj_kappa(nu);
    for (int k = 0; k < 3; k++) nt[k] = nbar[k] + hp * pbar[k];
    float nn = dot3(n, nt);
    for (int k = 0; k < 3; k++) vb[k] = n[k] * dt + kap * (nt[k] - n[k] * nn);
  }
  for (int k = 0; k < 3; k++) { c2bar[k] += vb[k]; c1bar[k] += pbar[k] - vb[k]; }
}

// One contact's geometry from the two geoms' world poses (geom_poses: s.gx, s.gq): the replay and, with cot, its VJP.
// t1 <= t2: engine geom types of the pair's geoms g1, g2 (0 plane, 1 sphere, 2 capsule; c_gtype[PG1 / PG2]), sz1, sz2
// their sizes (c_gsize + 3 g), slot: the tape's (plane-capsule end 0: +axis, 1: -axis; parallel capsule-capsule
// endpoint 0..1; else 0). Replay (always): dist, pos[3], fr[9] = the forward's (collide_pair) bitwise, and nu = the
// sphere core's centre distance |c2 - c1| before the radii (the tape's min_nu; +inf for plane contacts): by collide_pair's
// own functions (step.py: zaxis_rn, plane_point, plane_capsule_frame, capsule_end, segment_point, capcap_*,
// sphere_sphere, make_frame; explicit roundings, so the same bits in this kernel as in the forward's). The primitive's
// branches are decided again here, on that arithmetic (sphere-capsule t clamp, capsule-capsule clamp cascade with its
// free set, plane-capsule frame fallback, make_frame's basis), as MuJoCo Warp's _narrowphase_recompute does. cot (null:
// replay only) = (dbar, pbar[3], frbar[9]) = dL/d(dist, pos, frame rows); then gb[12] = (xbar1, tbar1,
// xbar2, tbar2): the geoms' pose cotangents (world; torque form, see the module docstring); the plane's are 0. The
// normal derivative of the sphere core follows DK_ADJ_NORMAL. Returns the ADJ_BR_* branch bits, -1 for a type
// combination without a VJP (boxes; the replay outputs are then undefined). One copy of the code: called twice per
// contact by the adjoint (replay before the row VJPs, VJP after)
__device__ __noinline__ int adj_contact_geom(int t1, int t2, int slot, const float* p1, const float* q1,
                                             const float* sz1, const float* p2, const float* q2, const float* sz2,
                                             float* dist, float* pos, float* fr, float* nu, const float* cot,
                                             float* gb) {
  int br = 0;
  *nu = __int_as_float(0x7f800000);
  if (cot)
    for (int k = 0; k < 12; k++) gb[k] = 0.f;
  if (t1 == 0) {  // plane (the floor, static): normal = its z axis; the frame does not depend on the other geom...
    float n[3];
    zaxis_rn(q1, n);
    if (t2 == 1) {  // plane-sphere
      *dist = plane_point(p1, n, p2, sz2[0], pos);
      make_frame(n, fr);
      if (!(-0.5f < n[1] && n[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
      if (cot) {  // xbar_s = pbar + n (dbar - (n . pbar) / 2); the frame is constant
        float pn = dot3(n, cot + 1);
        for (int k = 0; k < 3; k++) gb[6 + k] = cot[1 + k] + n[k] * (cot[0] - 0.5f * pn);
      }
      return br;
    }
    if (t2 != 2) return -1;
    // plane-capsule: ...except here, where t1 follows the capsule axis projected on the plane (fallback when short)
    float axis[3];
    zaxis_rn(q2, axis);
    const float bn = plane_capsule_frame(n, axis, fr), *b = fr + 3;
    if (bn < 0.5f) {
      br |= ADJ_BR_FALLBACK;
      if (!(-0.5f < n[1] && n[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
    }
    float sg = slot == 0 ? 1.f : -1.f, ce[3];  // the end slot (the forward's e)
    capsule_end(p2, axis, sz2[1], slot, ce);
    *dist = plane_point(p1, n, ce, sz2[0], pos);
    if (cot) {  // end ce = x_c + sg l a: cbar = pbar + n (dbar - (n . pbar) / 2); abar = sg l cbar + frame part
      float pn = dot3(n, cot + 1), ab[3], u[3];
      for (int k = 0; k < 3; k++) {
        gb[6 + k] = cot[1 + k] + n[k] * (cot[0] - 0.5f * pn);
        ab[k] = sg * sz2[1] * gb[6 + k];
      }
      if (!(br & ADJ_BR_FALLBACK)) {  // t1 = normalize(a - n (n . a)), t2 = n x t1: btot = t1bar + t2bar x n, then
        cross(cot + 10, n, u);       // abar += (I - n n') (I - t1 t1') btot / beta
        for (int k = 0; k < 3; k++) u[k] += cot[7 + k];
        float bu = dot3(b, u);
        for (int k = 0; k < 3; k++) u[k] -= b[k] * bu;
        float un = dot3(n, u);
        for (int k = 0; k < 3; k++) ab[k] += (u[k] - n[k] * un) / bn;
      }
      cross(axis, ab, gb + 9);
    }
    return br;
  }
  float n[3], c1b[3] = {0.f, 0.f, 0.f}, c2b[3] = {0.f, 0.f, 0.f}, nb[3];
  if (t1 == 1 && t2 == 1) {  // sphere-sphere: the core on the centres
    *nu = sphere_sphere(p1, sz1[0], p2, sz2[0], dist, pos, n);
    make_frame(n, fr);
    if (!(-0.5f < n[1] && n[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
    if (cot) {
      make_frame_vjp(fr, cot + 4, nb);
      sphere_core_vjp(n, *nu, *dist, sz1[0], cot[0], cot + 1, nb, gb, gb + 6);
    }
    return br;
  }
  if (t1 == 1 && t2 == 2) {  // sphere-capsule: closest point A + t B of the segment, then the core
    float axis[3], a[3], ab[3], ap[3], pt[3], gam, t;  // gam: MuJoCo Warp's 1e-6 regularizer (kept)
    zaxis_rn(q2, axis);
    const float raw = segment_point(p1, p2, axis, sz2[1], a, ab, ap, &gam, &t, pt);
    *nu = sphere_sphere(p1, sz1[0], pt, sz2[0], dist, pos, n);
    make_frame(n, fr);
    if (!(-0.5f < n[1] && n[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
    bool free_t = raw >= 0.f && raw < 1.f;  // wp.clamp passes the gradient for lo <= raw < hi
    if (free_t) br |= ADJ_BR_FREE2;
    if (cot) {
      make_frame_vjp(fr, cot + 4, nb);
      sphere_core_vjp(n, *nu, *dist, sz1[0], cot[0], cot + 1, nb, gb, c2b);  // c2b = ptbar
      float Ab[3], Bb[3], tb = dot3(ab, c2b);
      for (int k = 0; k < 3; k++) { Ab[k] = c2b[k]; Bb[k] = t * c2b[k]; }
      if (free_t) {  // t = (x_s - A) . B / gam: alpha bar, gam bar
        float al = tb / gam, gm = -t * tb / gam;
        for (int k = 0; k < 3; k++) {
          gb[k] += al * ab[k]; Ab[k] -= al * ab[k]; Bb[k] += al * ap[k] + 2.f * gm * ab[k];
        }
      }
      float abar[3];  // A = x_c - l a, B = 2 l a
      for (int k = 0; k < 3; k++) { gb[6 + k] = Ab[k]; abar[k] = sz2[1] * (2.f * Bb[k] - Ab[k]); }
      cross(axis, abar, gb + 9);
    }
    return br;
  }
  if (t1 != 2 || t2 != 2) return -1;
  // capsule-capsule: segment parameters (x1, x2) by the forward's clamp cascade (or the parallel candidate `slot`); the
  // free ones solve their rows of G x = h, G = [ma mb; mb mc], h = (u, v), the others are fixed at +-1
  CapCap cc;
  capcap_setup(p1, q1, sz1[1], p2, q2, sz2[1], cc);
  const float *a1 = cc.a1, *a2 = cc.a2, *dif = cc.dif;
  const float ma = cc.ma, mb = cc.mb, mc = cc.mc, det = cc.det;
  float x1, x2, raw;
  bool f1, f2;  // a clamp to [-1, 1] leaves x free for -1 <= raw < 1 (wp.clamp's gradient rule)
  if (capcap_nonparallel(cc)) capcap_params(cc, &x1, &x2, &f1, &f2, &raw);
  else {  // parallel: candidate e = slot of the forward's loop
    br |= ADJ_BR_PARALLEL;
    capcap_candidate(cc, slot, &x1, &x2, &f1, &f2, &raw);
  }
  (void)raw;
  if (f1) br |= ADJ_BR_FREE1;
  if (f2) br |= ADJ_BR_FREE2;
  *nu = capcap_core(p1, p2, cc, x1, x2, sz1[0], sz2[0], dist, pos, n);
  make_frame(n, fr);
  if (!(-0.5f < n[1] && n[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
  if (cot) {
    make_frame_vjp(fr, cot + 4, nb);
    // c_k = x_k + s_k x_k: xbar_k += cbar_k, sbar_k += x_k cbar_k, x_k bar = s_k . cbar_k (free ones only); then
    // z = G_FF^-1 xbar_F (0 off F), hbar = z, Gbar = -z x' (sym.): mabar = -z1 x1, mbbar = -(z1 x2 + z2 x1),
    // mcbar = -z2 x2
    float s1b[3], s2b[3], xb1, xb2, w[3];
    cross(a1, a2, w);
    float ww = dot3(w, w);  // = det = ma mc - mb^2 without its cancellation
    // Below sin^2(theta) = 1e-12 use the general chain with the forward's
    // determinant; otherwise the normal can be differentiated through w.
    bool wok = ww > 1e-12f * ma * mc;
#if DK_ADJ_PARALLEL
    // near parallel with both parameters free (DK_ADJ_PARALLEL): the closest points slide along the axes
    // with derivatives of order nu / sin^2(theta) (a tilt of one axis against the other), the normal n = sg w / |w|
    // turns with 1 / sin(theta). Below the threshold x1, x2 are held constant (no z through G^-1, the general chain
    // below): the contact moves with the two material points p_k + a_k x_k, its frame by the sphere core's
    // kappa(nu) (I - n n'). By the envelope theorem the distance's derivative stays the exact one (d dist / dx = 0 at
    // the interior minimum); what is approximated is the contact point's sliding and the normal's turning
    if (f1 && f2 && ww < DK_ADJ_PARALLEL_EPS * ma * mc) f1 = f2 = false;
#endif
    if (f1 && f2 && *nu > 0.f && wok) {
      // both parameters free (C0): c2 - c1 is normal to both axes, so n = sg w / |w| exactly and the normal's
      // derivative goes to the axes alone, dn = sg (I - n n') dw / |w|. The general chain (sphere_core_vjp, then x
      // through G) gives the same in exact arithmetic but as a sum of terms of order 1/nu that cancel to order 1
      // (relative errors up to 1e-3 at nu = 5 mm in float32); here no 1/nu appears. DK_ADJ_NORMAL scales the normal
      // path by kappa(nu) nu
      const float* pb = cot + 1;
      float hp = sz1[0] + 0.5f * (*dist), dt = cot[0] + 0.5f * dot3(n, pb), t[3], u1[3], u2[3];
      for (int k = 0; k < 3; k++) t[k] = nb[k] + hp * pb[k];
      float nt = dot3(n, t), f = adj_kappa_nu(*nu) * rsqrtf(ww);
      if (dot3(n, w) < 0.f) f = -f;
      for (int k = 0; k < 3; k++) t[k] = f * (t[k] - n[k] * nt);  // sg (I - n n') ntot / |w|
      cross(a2, t, u1);
      cross(t, a1, u2);
      for (int k = 0; k < 3; k++) {  // radial: dnu = n . (dc2 - dc1) with n . s_k dx_k = 0; pos = c1 + n (r1 + d / 2)
        c2b[k] = n[k] * dt; c1b[k] = pb[k] - c2b[k];
        s1b[k] = x1 * c1b[k] + u1[k]; s2b[k] = x2 * c2b[k] + u2[k];
      }
      xb1 = dot3(a1, pb); xb2 = 0.f;  // only pos moves with x1 (c1); the distance and the normal are stationary in x
    } else {
      sphere_core_vjp(n, *nu, *dist, sz1[0], cot[0], cot + 1, nb, c1b, c2b);
      for (int k = 0; k < 3; k++) { s1b[k] = x1 * c1b[k]; s2b[k] = x2 * c2b[k]; }
      xb1 = f1 ? dot3(a1, c1b) : 0.f; xb2 = f2 ? dot3(a2, c2b) : 0.f;
    }
    float z1 = 0.f, z2 = 0.f;
    float gd = wok ? ww : det;
    if (f1 && f2) { z1 = (mc * xb1 - mb * xb2) / gd; z2 = (ma * xb2 - mb * xb1) / gd; }
    else if (f1) z1 = xb1 / ma;
    else if (f2) z2 = xb2 / mc;
    float mab = -z1 * x1, mbb = -(z1 * x2 + z2 * x1), mcb = -z2 * x2, db[3];
    for (int k = 0; k < 3; k++) {  // ma = s1.s1, mb = -s1.s2, mc = s2.s2, u = -s1.dif, v = s2.dif, dif = x_1 - x_2
      s1b[k] += 2.f * mab * a1[k] - mbb * a2[k] - z1 * dif[k];
      s2b[k] += 2.f * mcb * a2[k] - mbb * a1[k] + z2 * dif[k];
      db[k] = -z1 * a1[k] + z2 * a2[k];
      gb[k] = c1b[k] + db[k];
      gb[6 + k] = c2b[k] - db[k];
    }
    cross(a1, s1b, gb + 3);  // s_k = l_k R_k e_z: tbar_k = s_k x sbar_k
    cross(a2, s2b, gb + 9);
  }
  return br;
}

// adj_contact_geom for contact (pair, slot) of the engine's pair table, from the collision geoms' world poses
// gx[NGEOMC][3], gq[NGEOMC][4] (struct Sim's gx / gq layout, or the adjoint's copy)
__device__ __forceinline__ int adj_pair_geom(int pair, int slot, const float (*gx)[3], const float (*gq)[4],
                                             float* dist, float* pos, float* fr, float* nu, const float* cot,
                                             float* gb) {
  int g1 = PG1(pair), g2 = PG2(pair);
  return adj_contact_geom(c_gtype[g1], c_gtype[g2], slot, gx[g1], gq[g1], c_gsize + 3 * g1, gx[g2], gq[g2],
                          c_gsize + 3 * g2, dist, pos, fr, nu, cot, gb);
}

// a geom's pose cotangent (xbar, tbar = gb + 6 j of adj_contact_geom) onto its body's:
// x_g = X_b + R_b g0, R_g = R_b R_g0, so fbar_b += xbar, tbar_b += tbar + (x_g - X_b) x xbar; xg, Xb: the geom's centre
// and the body origin (the world body takes no cotangent: skip it)
__device__ __forceinline__ void geom_to_body_vjp(const float* xg, const float* Xb, const float* xbar, const float* tbar,
                                                 float* fbar_b, float* tbar_b) {
  float r[3] = {xg[0] - Xb[0], xg[1] - Xb[1], xg[2] - Xb[2]}, u[3];
  cross(r, xbar, u);
  for (int k = 0; k < 3; k++) { fbar_b[k] += xbar[k]; tbar_b[k] += tbar[k] + u[k]; }
}
"""



# ---- terrain builds (nablaengine.terrain; adjoint.ADJ_TERRAIN_HOOKS inserts it before adj_pair_geom) ----
TERRAIN_DEVICE = r"""
// A terrain contact (virtual pair NPAIR + t: geom 1 the static terrain, geom 2 the robot geom of type gt, size sz, pose
// p2 / q2) with tape slot k | element << 8 (aux: a native heightfield contact's candidate, the tape's taux): the replay (dist, pos, fr: the forward's bits, by the same __noinline__
// geometry, step.TERRAIN_DEVICE) and, with cot = (dbar, pbar, frbar), gb[6..11] = the robot geom's pose cotangents
// (gb[0..5] = 0). The contact's point on the robot geom is a material point c = p2 + w l axis: the sphere centre (w = 0),
// a capsule_box candidate (w: its segment parameter) or a heightfield sample point (w fixed); w is held constant (exact
// at the segment ends and for the distance at an interior minimum, by the envelope theorem; the candidate's sliding
// along the axis is left out). Box: dist = |c_l - clamp(c_l)| - r with the normal u = (c_l - cl) / |.| (outside, in the
// box frame; its derivative (I - u u') P / |.| on the clamped axes P, scaled as the sphere core's: adj_kappa), or the
// nearest face's constant normal (inside); heightfield: the triangle's plane, dist = n . c - const, constant normal.
// Either way pos = c - n (r + dist / 2) up to the clamped part. Returns the ADJ_BR_* bits (-1: not a contact)
__device__ __noinline__ int adj_terrain_geom(int slot, unsigned aux, int gt, const float* sz, const float* p2,
                                             const float* q2, float* dist, float* pos, float* fr, float* nu,
                                             const float* cot, float* gb) {
  const int k = slot & 255, e = slot >> 8, nbox = g_ter.grid ? int(g_ter.grid[5]) : 0;
  const bool box = e < nbox;  // a terrain box (ter_box)
  *nu = __int_as_float(0x7f800000);
  if (cot)
    for (int t = 0; t < 12; t++) gb[t] = 0.f;
  float n[3], c[3] = {p2[0], p2[1], p2[2]}, axis[3] = {0.f, 0.f, 0.f}, w = 0.f, qp[3];
  bool point_n = false;  // native heightfield HF_POINT: n = (c - qp) / |c - qp|
  if (gt == 2) zaxis_rn(q2, axis);
  if (box) {
    const float* b = ter_box(e);
    if (gt == 2) {
      float segs[2];
      terrain_capbox_segs(p2, axis, sz[1], b, segs);
      w = segs[k];
      capbox_point(p2, axis, sz[1], b, w, c);
    }
    terrain_sphere_box(c, sz[0], b, dist, pos, n);
  } else {
#if TERRAIN_HF_NATIVE
    // the native contact: its candidate (aux: the tape's taux, HfCand.pid) recomputed from its prism alone
    (void)k;
    HfCon con[1];
    if (!hfield_native_one(e - nbox, gt, sz, p2, q2, aux, con)) return -1;
    const int k1 = 0;
    const float* hh = g_ter.hfh + 16 * (e - nbox);
    *dist = con[k1].dist;
    mv3(hh + 3, con[k1].pos, pos);
    for (int t = 0; t < 3; t++) pos[t] += hh[t];
    mv3(hh + 3, con[k1].n, n);
    if (gt == 2) w = 2.f * con[k1].t - 1.f;
    if (con[k1].type == HF_POINT) {  // normal from the fixed prism point q to the core point
      mv3(hh + 3, con[k1].q, qp);
      for (int t = 0; t < 3; t++) qp[t] += hh[t];
      point_n = true;
    }
#else
    if (!terrain_hfield_point(e - nbox, gt, sz, p2, q2, k, dist, pos, n)) return -1;
    if (gt == 2) w = TERRAIN_CAP_PTS > 1 ? -1.f + float(k) * 2.f / float(TERRAIN_CAP_PTS - 1) : 0.f;
#endif
  }
  make_frame(n, fr);
  const int br = (-0.5f < n[1] && n[1] < 0.5f) ? 0 : ADJ_BR_FRAME_Z;
  if (!cot) return br;
  const float* pb = cot + 1;
  float cb[3];  // the material point's cotangent (world)
  bool fixed_n = !box;
  if (box) {
    const float *b = ter_box(e), *R = b + 3, *bs = b + 12;
    float d[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]}, cl[3], u[3], P[3], pbl[3], cbl[3];
    mtv3(R, d, cl);
    for (int t = 0; t < 3; t++) {
      float q = fmaxf(-bs[t], fminf(bs[t], cl[t]));
      u[t] = cl[t] - q;
      P[t] = q != cl[t] ? 1.f : 0.f;
    }
    float dd = sqrtf(dot3(u, u));
    if (dd <= 1e-15f) fixed_n = true;  // inside: the nearest face's normal
    else {
      float nbar[3], nbl[3], ub[3];
      mtv3(R, pb, pbl);
      make_frame_vjp(fr, cot + 4, nbar);
      mtv3(R, nbar, nbl);
      for (int t = 0; t < 3; t++) { u[t] /= dd; ub[t] = nbl[t] - 0.5f * sz[0] * pbl[t]; }
      const float uu = dot3(u, ub), kap = adj_kappa(dd);
      for (int t = 0; t < 3; t++)
        cbl[t] = u[t] * cot[0] + 0.5f * (2.f - P[t]) * pbl[t] + P[t] * kap * (ub[t] - u[t] * uu);
      mv3(R, cbl, cb);
    }
  }
  if (point_n) {  // the sphere core on (qp, c): nu = |c - qp| = dist + r
    float nbar[3], qb[3] = {0.f, 0.f, 0.f};
    make_frame_vjp(fr, cot + 4, nbar);
    for (int t = 0; t < 3; t++) cb[t] = 0.f;
    sphere_core_vjp(n, *dist + sz[0], *dist, 0.f, cot[0], pb, nbar, qb, cb);
    fixed_n = false;
  }
  if (fixed_n) {  // dist = n . c - const, pos = c - n (r + dist / 2): cbar = pbar - n (n . pbar) / 2 + n dbar
    const float pn = dot3(n, pb);
    for (int t = 0; t < 3; t++) cb[t] = pb[t] - 0.5f * n[t] * pn + n[t] * cot[0];
  }
  for (int t = 0; t < 3; t++) gb[6 + t] = cb[t];
  if (gt == 2) {  // c = p2 + w l axis: axis bar = w l cbar, as a torque
    float ab[3];
    for (int t = 0; t < 3; t++) ab[t] = w * sz[1] * cb[t];
    cross(axis, ab, gb + 9);
  }
  return br;
}

"""

def source(m):
    """header(m) + DEVICE: the contact adjoint's device code for model m, to place after step.STEP_DEVICE."""
    return header(m) + DEVICE
