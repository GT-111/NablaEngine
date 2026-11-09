"""State groups of the adjoint checks, labelled from MuJoCo Warp's own forward, and per-group statistics.

A control step's gradient is labelled by what happened in any of its substeps. mjw_substep(mjm, d_in, d_out) reads one
substep of MuJoCo Warp (d_out: the Data that mjw.step(m, d_in, d_out) filled; its contacts, rows and acceleration belong
to the input state d_in) into numpy: contacts with their (pair, slot) identity (step.py's collide_pair: plane-capsule end
0 = +axis, 1 = -axis; capsule-capsule 0, or endpoint 0..1 of the parallel branch; else 0), rows, x_e = J a* - aref,
clamp flags and solver iterations. labels(mjm, subs, ...) turns a list of them into one boolean per world and group:

  G1 axis < 5 mm   min nu < 5 mm over sphere / capsule self contacts (nu = dist + r1 + r2, the centre or axis distance
                   before the radii); G1_lt_1mm likewise
  G2 contact flip  substep 0's contact set differs from the corpus's reference (nablaengine.fixtures.load_contacts,
                   by pair counts), or
                   a contact has |dist| < 1e-6 m: one of MuJoCo Warp's, or a pair of MuJoCo C's float64 collision at the
                   substep's input state with a 1e-5 m margin (catches the contacts float32 rounding left out of this
                   run but that another run, or the engine, may have)
  G3 kink          a row with |x_e| / (|J_e a*| + |aref_e|) < 1e-4
  G4 clamped       a dof's actuator force at or beyond +-frcrange (MuJoCo Warp's _actuator_lam); G4_near: the unclamped
                   PD force within 2% of the bound
  G5 limits        a joint-limit row
  G6 self contact  a contact between two robot geoms
  G7 basis switch  a self contact with ||n_y| - 0.5| < 0.02, or a plane-capsule contact with |beta - 0.5| < 0.02
                   (beta = |capsule axis projected on the plane|)
  G8 near parallel capsule-capsule with both segment parameters inside (-1, 1) and sin(angle) < 0.05
  G9 not converged solver iterations == the limit
  G0 normal        none of G1, G2, G3, G7, G9

Subgroups (SUBGROUPS, group_mask): G4, G5, G6 intersected with G0 ('<G>&G0', as the gradient acceptance gates them), and
G0 without G4 ('G0_normal&~G4_clamped': the finite-difference check's G0, where the clamp option DK_ADJ_CLAMP_KV cannot
matter).
"""

import mujoco
import numpy as np

GROUPS = ("all", "G0_normal", "G1_axis_lt_5mm", "G1_axis_lt_1mm", "G2_contact_flip", "G3_kink", "G4_clamped",
          "G4_near_clamp", "G5_limits", "G6_self_contact", "G7_basis_switch", "G8_near_parallel", "G9_nonconv")
OUTSIDE_G0 = ("G1_axis_lt_5mm", "G2_contact_flip", "G3_kink", "G7_basis_switch", "G9_nonconv")
SUBGROUPS = ("G4_clamped&G0", "G5_limits&G0", "G6_self_contact&G0", "G0_normal&~G4_clamped")  # group_mask
PLANE, SPHERE, CAPSULE = 0, 2, 3  # mujoco.mjtGeom
LIMIT_JOINT = 3  # mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT


def pair_table(mjm):
    """{frozenset(geom1, geom2): pair index} of the model's explicit pairs."""
    return {frozenset((int(mjm.pair_geom1[i]), int(mjm.pair_geom2[i]))): i for i in range(mjm.npair)}


def capsule_segments(p1, a1, hl1, p2, a2, hl2):
    """Segment parameters (x1, x2) of the capsule-capsule closest points (step.py's clamp cascade, float64), det, and
    the four candidates of the parallel branch."""
    s1, s2 = a1 * hl1, a2 * hl2
    dif = p1 - p2
    ma, mb, mc, u, v = s1 @ s1, -(s1 @ s2), s2 @ s2, -(s1 @ dif), s2 @ dif
    det = ma * mc - mb * mb
    cand = [(1.0, np.clip((v - mb) / mc, -1, 1)), (-1.0, np.clip((v + mb) / mc, -1, 1)),
            (np.clip((u - mb) / ma, -1, 1), 1.0), (np.clip((u + mb) / ma, -1, 1), -1.0)]
    if abs(det) < 1e-15:
        return None, None, det, cand
    x1, x2 = (mc * u - mb * v) / det, (ma * v - mb * u) / det
    if x1 > 1:
        x1, x2 = 1.0, (v - mb) / mc
    elif x1 < -1:
        x1, x2 = -1.0, (v + mb) / mc
    if x2 > 1:
        x2, x1 = 1.0, float(np.clip((u - mb) / ma, -1, 1))
    elif x2 < -1:
        x2, x1 = -1.0, float(np.clip((u + mb) / ma, -1, 1))
    return x1, x2, det, cand


def mjw_substep(mjm, d_in, d_out, pairs=None):
    """One substep of MuJoCo Warp as numpy (see the module docstring). Returns dict with per-world lists `contacts`
    (dicts: cid, pair, slot, geom (model ids, MuJoCo Warp's order), dist, pos, frame (3x3), dim, rows (efc rows),
    nu (sphere / capsule self contacts), beta (plane-capsule), sin, x1, x2 (capsule-capsule)) and arrays qpos, qvel, ctrl
    (inputs), qacc (a*), nefc, efc_type, efc_id, efc_pos, efc_vel, efc_D, efc_aref, efc_state, efc_force, jac_a (J a*),
    clamped, near_clamp (W, nu), clamp_margin (W,), niter."""
    pairs = pairs or pair_table(mjm)
    W = d_in.qpos.shape[0]
    g = lambda a: a.numpy().copy()  # noqa: E731  (copies: on the CPU .numpy() is a view)
    out = dict(qpos=g(d_in.qpos).astype(np.float64), qvel=g(d_in.qvel).astype(np.float64), ctrl=g(d_in.ctrl),
               qacc=g(d_out.qacc), nefc=g(d_out.nefc), niter=g(d_out.solver_niter))
    for k in ("type", "id", "pos", "vel", "D", "aref", "state", "force"):
        out[f"efc_{k}"] = g(getattr(d_out.efc, k))
    rownnz, rowadr, colind, Jv = (g(x) for x in (d_out.efc.J_rownnz, d_out.efc.J_rowadr, d_out.efc.J_colind, d_out.efc.J))
    nj = out["efc_pos"].shape[1]
    jac_a = np.zeros((W, nj))
    for w in range(W):
        for r in range(int(out["nefc"][w])):
            a, n = rowadr[w, r], rownnz[w, r]
            jac_a[w, r] = Jv[w, 0, a:a + n] @ out["qacc"][w, colind[w, 0, a:a + n]]
    out["jac_a"] = jac_a
    # actuators: MuJoCo Warp's clamp test on qfrc_actuator; the unclamped PD law for the 2% band
    jid = mjm.actuator_trnid[:, 0]
    dadr, qadr = mjm.jnt_dofadr[jid], mjm.jnt_qposadr[jid]
    lim = np.where(mjm.jnt_actfrclimited[jid], mjm.jnt_actfrcrange[jid, 1], np.inf)
    lo = np.where(mjm.jnt_actfrclimited[jid], mjm.jnt_actfrcrange[jid, 0], -np.inf)
    f = g(d_out.qfrc_actuator)[:, dadr]
    out["clamped"] = (f <= lo) | (f >= lim)
    raw = mjm.actuator_gainprm[:, 0] * (out["ctrl"] - out["qpos"][:, qadr]) + mjm.actuator_biasprm[:, 2] * out["qvel"][:, dadr]
    out["near_clamp"] = np.abs(np.abs(raw) - lim) < 0.02 * lim
    margin = np.where(np.isfinite(lim), np.abs(np.abs(raw) - lim) / np.where(np.isfinite(lim), lim, 1.0), np.inf)
    out["clamp_margin"] = margin.min(axis=1)  # relative distance of the PD force to the nearest bound
    # contacts
    n = int(d_out.nacon.numpy()[0])
    geom, wid, dist, pos, frame, dim, adr = (g(x)[:n] for x in (d_out.contact.geom, d_out.contact.worldid,
                                                               d_out.contact.dist, d_out.contact.pos, d_out.contact.frame,
                                                               d_out.contact.dim, d_out.contact.efc_address))
    gx, gm = g(d_out.geom_xpos).astype(np.float64), g(d_out.geom_xmat).astype(np.float64)
    con = [[] for _ in range(W)]
    for c in range(n):
        g1, g2 = int(geom[c, 0]), int(geom[c, 1])
        t1, t2 = int(mjm.geom_type[g1]), int(mjm.geom_type[g2])
        w = int(wid[c])
        e = dict(cid=c, pair=pairs[frozenset((g1, g2))], slot=0, geom=(g1, g2), types=(t1, t2), dist=float(dist[c]),
                 pos=pos[c].astype(np.float64), frame=frame[c].astype(np.float64).reshape(3, 3), dim=int(dim[c]),
                 rows=[int(r) for r in adr[c] if r >= 0])
        if PLANE in (t1, t2):
            cg = g2 if t1 == PLANE else g1
            if mjm.geom_type[cg] == CAPSULE:
                ax = gm[w, cg].reshape(3, 3)[:, 2]
                e["slot"] = 0 if (e["pos"] - gx[w, cg]) @ ax > 0 else 1
                nrm = e["frame"][0]
                e["beta"] = float(np.linalg.norm(ax - nrm * (nrm @ ax)))
        else:
            e["nu"] = e["dist"] + mjm.geom_size[g1, 0] + mjm.geom_size[g2, 0]
            if t1 == CAPSULE and t2 == CAPSULE:
                ax1, ax2 = gm[w, g1].reshape(3, 3)[:, 2], gm[w, g2].reshape(3, 3)[:, 2]
                x1, x2, _, cand = capsule_segments(gx[w, g1], ax1, mjm.geom_size[g1, 1], gx[w, g2], ax2,
                                                     mjm.geom_size[g2, 1])
                e["sin"] = float(np.linalg.norm(np.cross(ax1, ax2)))
                if x1 is None:  # parallel branch: the candidate whose midpoint is nearest
                    mid = [0.5 * (gx[w, g1] + ax1 * mjm.geom_size[g1, 1] * a + gx[w, g2] + ax2 * mjm.geom_size[g2, 1] * b)
                           for a, b in cand]
                    e["slot"] = int(np.argmin([np.linalg.norm(p - e["pos"]) for p in mid]))
                    x1, x2 = cand[e["slot"]]
                e["x1"], e["x2"] = float(x1), float(x2)
        con[w].append(e)
    for w in range(W):
        con[w].sort(key=lambda e: (e["pair"], e["slot"]))
    out["contacts"] = con
    return out


def near_contact_distance(mjm, qpos, margin=1e-5):
    """Per state (rows of qpos), the smallest |dist| over the model's pairs within `margin` of touching (MuJoCo C,
    float64 collision; inf if none)."""
    saved = mjm.pair_margin.copy()
    try:
        mjm.pair_margin[:] = margin
        d = mujoco.MjData(mjm)
        out = np.full(len(qpos), np.inf)
        for i, q in enumerate(qpos):
            d.qpos[:] = q
            mujoco.mj_kinematics(mjm, d)
            mujoco.mj_collision(mjm, d)
            if d.ncon:
                out[i] = np.min(np.abs(d.contact.dist[:d.ncon]))
        return out
    finally:
        mjm.pair_margin[:] = saved


def labels(mjm, subs, iterations, ref_contacts=None):
    """({group: bool (W,)}, {continuous label: (W,)}) from the substeps `subs` (mjw_substep dicts, in order);
    ref_contacts: the `con` rows of nablaengine.fixtures.load_contacts for the same worlds (substep 0's contact set
    check), or None. The
    continuous labels (minima over the substeps, for attributing outliers): min_nu (m), min_abs_dist (m, as G2), min_kink
    (|x_e| / (|J_e a*| + |aref_e|)), clamp_margin (| |PD force| - bound | / bound), max_niter."""
    W = len(subs[0]["contacts"])
    lab = {k: np.zeros(W, bool) for k in GROUPS}
    cont = {k: np.full(W, np.inf) for k in ("min_nu", "min_abs_dist", "min_kink", "clamp_margin")}
    cont["max_niter"] = np.zeros(W, int)
    min_nu = cont["min_nu"]
    for k, s in enumerate(subs):
        cont["clamp_margin"] = np.minimum(cont["clamp_margin"], s["clamp_margin"])
        cont["min_abs_dist"] = np.minimum(cont["min_abs_dist"], near_contact_distance(mjm, s["qpos"]))
        cont["max_niter"] = np.maximum(cont["max_niter"], s["niter"])
        for w in range(W):
            for e in s["contacts"][w]:
                cont["min_abs_dist"][w] = min(cont["min_abs_dist"][w], abs(e["dist"]))
                if abs(e["dist"]) < 1e-6:
                    lab["G2_contact_flip"][w] = True
                if "nu" in e:
                    lab["G6_self_contact"][w] = True
                    min_nu[w] = min(min_nu[w], e["nu"])
                    if abs(abs(e["frame"][0, 1]) - 0.5) < 0.02:
                        lab["G7_basis_switch"][w] = True
                if "beta" in e and abs(e["beta"] - 0.5) < 0.02:
                    lab["G7_basis_switch"][w] = True
                if "sin" in e and abs(e["x1"]) < 1 and abs(e["x2"]) < 1 and e["sin"] < 0.05:
                    lab["G8_near_parallel"][w] = True
            ne = int(s["nefc"][w])
            x = s["jac_a"][w, :ne] - s["efc_aref"][w, :ne]
            scale = np.abs(s["jac_a"][w, :ne]) + np.abs(s["efc_aref"][w, :ne])
            if ne:
                cont["min_kink"][w] = min(cont["min_kink"][w], float(np.min(np.abs(x) / np.maximum(scale, 1e-30))))
            if ne and (np.abs(x) < 1e-4 * np.maximum(scale, 1e-30)).any():
                lab["G3_kink"][w] = True
            if (s["efc_type"][w, :ne] == LIMIT_JOINT).any():
                lab["G5_limits"][w] = True
        lab["G4_clamped"] |= s["clamped"].any(1)
        lab["G4_near_clamp"] |= s["near_clamp"].any(1)
        lab["G9_nonconv"] |= s["niter"] >= iterations
    lab["G2_contact_flip"] |= cont["min_abs_dist"] < 1e-6
    if ref_contacts is not None:
        for w in range(W):
            ours = sorted(e["pair"] for e in subs[0]["contacts"][w] if e["dist"] < 0)
            ref = sorted(int(r[0]) for r in ref_contacts[w] if r[0] >= 0)
            if ours != ref:
                lab["G2_contact_flip"][w] = True
    lab["G1_axis_lt_5mm"] = min_nu < 5e-3
    lab["G1_axis_lt_1mm"] = min_nu < 1e-3
    lab["all"][:] = True
    lab["G0_normal"] = ~np.any([lab[k] for k in OUTSIDE_G0], axis=0)
    return lab, cont


def qpos_tangent(g, q):
    """qpos-form cotangents (S, NQ) -> dof-tangent form (S, NV) at the states q: the quaternion part as
    1/2 Im(xi* (x) g) / |xi|^2 (jax64.qpos_to_dof; the left inverse of MuJoCo Warp's _dof_to_qpos)."""
    xi, gq = np.asarray(q, np.float64)[:, 3:7], np.asarray(g, np.float64)[:, 3:7]
    w, x, y, z = xi.T
    a, b, c, d = gq.T
    v = np.stack([w * b - x * a - y * d + z * c, w * c + x * d - y * a - z * b, w * d - x * c + y * b - z * a], 1)
    g = np.asarray(g, np.float64)
    return np.concatenate([g[:, :3], 0.5 * v / np.sum(xi * xi, 1, keepdims=True), g[:, 7:]], 1)


def world_errors(g, ref):
    """Per world: relative error |g - ref| / max(|ref|, 1e-9) and cosine (both along axis 1)."""
    g, ref = np.asarray(g, np.float64), np.asarray(ref, np.float64)
    nr = np.linalg.norm(ref, axis=1)
    e = np.linalg.norm(g - ref, axis=1) / np.maximum(nr, 1e-9)
    cos = np.sum(g * ref, axis=1) / np.maximum(np.linalg.norm(g, axis=1) * nr, 1e-30)
    return e, cos


def group_mask(lab, name):
    """The states of a group (lab[name]) or of a subgroup 'A&B' (A and B) / 'A&~B' (A and not B), where G0 stands for
    G0_normal."""
    if "&" not in name:
        return lab[name]
    a, b = name.split("&")
    full = lambda g: "G0_normal" if g == "G0" else g  # noqa: E731
    return lab[full(a)] & (~lab[full(b[1:])] if b.startswith("~") else lab[full(b)])


def stats(e, cos=None):
    e = np.asarray(e)
    if not len(e):
        return dict(n=0)
    out = dict(n=int(len(e)), median=float(np.median(e)), p90=float(np.quantile(e, 0.9)),
               p99=float(np.quantile(e, 0.99)), max=float(e.max()))
    if cos is not None:
        out["cos_p1"] = float(np.quantile(cos, 0.01))
    return out

