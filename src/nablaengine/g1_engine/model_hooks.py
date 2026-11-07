"""Edits of the kernels' text for models outside the G1's structure (nv < 32, more than 3 children per body).

The kernels were written for the mjlab G1 (nv = 35, at most 3 children per body); their text stays as it is for
every model that fits that (the G1's builds are unchanged: scripts/identity_gate.py), and apply() edits it for the
others, before compilation:

  more than 3 children per body (quadrupeds: the trunk's four legs): c_child holds NCH = source.child_width(m)
      entries per body (source.const_layout) and every reader indexes it with NCH;
  nv < 32: the Cholesky and triangular solves give lane l the dof row l and lane l + 32 the row l + 32; lanes
      l >= nv repeat the last row's arithmetic (r0 = min(lane, NV - 1)), so they compute and write the same values
      as lane NV - 1 (no out-of-bounds rows), and no lane reads theirs (the shuffles' sources are rows < NV);
  joint damping (Unitree H1): c_damp (source.const_layout) on every dof, as MuJoCo's passive force -d qvel next to the
      actuator force (not clamped) and as h d in the implicitfast matrix M + h (kv + d); the adjoint adds d lam to
      P_v (the unclamped lam) and h d to the Q solve's correction;
  frictionloss rows (Booster T1): the adjoint's row indices after the NFLOSS frictionloss rows, and their VJP on the quadratic
      zone, P_v += D b lam (adj_limits);
  box-box pairs (the Booster T1's feet): box_box as MuJoCo Warp's native CCD (nablaengine.ccd: GJK, EPA, multicontact) and
      the contact adjoint's exact box-box VJP (ADJ_BOX_BOX, ccd_box_box_vjp on the replayed contact);
  box geoms whose pairs are all with planes (the Go1's trunk): the contact adjoint's plane-box corners
      (adj_contact_geom; the forward's collide_pair has them under HAS_BOX). A corner c = x + R c_l of the box is a
      point of its body: dist = n . (c - x_plane), pos = c - n dist / 2, the plane's frame, so cbar = pbar
      - n (n . pbar) / 2 + n dbar onto the box's position and (c - x) x cbar onto its rotation (exact).

Each hook is (anchor, replacement, count): the anchor must occur count times in the text it is applied to.
"""
from nablaengine.g1_engine import source

ENGINE_CHILD = (
    ("DKC int c_parent[NB], c_depth[NB], c_qadr[NB], c_dadr[NB], c_nchild[NB], c_child[NB * 3], c_nanc[NB],",
     "DKC int c_parent[NB], c_depth[NB], c_qadr[NB], c_dadr[NB], c_nchild[NB], c_child[NB * NCH], c_nanc[NB],", 1),
    ("c_child[3 * b + i]", "c_child[NCH * b + i]", 3),
)
KIN_CHILD = (
    ("  for (int i = 0; i < 3; i++) ch[i] = i < n ? c_child[3 * b + i] : 0;",
     "  for (int i = 0; i < NCH; i++) ch[i] = i < n ? c_child[NCH * b + i] : 0;", 1),
    ("  int ch[3];", "  int ch[NCH];", 2),
)
ENGINE_SMALL_NV = (
    ("  const int r0 = lane, r1 = lane + 32;\n  const bool has1 = r1 < NV;",
     "  const int r0 = min(lane, NV - 1), r1 = lane + 32;  // lanes >= NV repeat row NV - 1 (model_hooks)\n"
     "  const bool has1 = r1 < NV;", 2),  # chol_solve_reg, chol_solve_compact
    ("  const int r0 = lane, r1 = min(lane + 32, NV - 1);",
     "  const int r0 = min(lane, NV - 1), r1 = min(lane + 32, NV - 1);  // (model_hooks)", 1),  # tri_solve_packed
)

# the adjoint's scope asserts, for plane-box contacts and frictionloss rows (the Python checks decide what is supported)
ENGINE_ADJ_SCOPE = (
    ('static_assert(!HAS_BOX && NFLOSS == 0, "the contact adjoint is for the mjlab model: no boxes, no frictionloss rows");',
     '// (model_hooks: plane-box contacts and frictionloss rows in the contact adjoint)', 1),
    ('static_assert(HAS_BOX == 0 && NFLOSS == 0, "the adjoint is for the mjlab model (no boxes, no frictionloss rows)");',
     '// (model_hooks: plane-box contacts and frictionloss rows in the adjoint)', 1),
)
ENGINE_PLANE_BOX = (
    ("    if (t2 != 2) return -1;\n",
     """    if (t2 == 3) {  // plane-box corner `slot` (model_hooks): the forward's collide_pair arithmetic, then its VJP
      float cl[3] = {(slot & 1) ? sz2[0] : -sz2[0], (slot & 2) ? sz2[1] : -sz2[1], (slot & 4) ? sz2[2] : -sz2[2]}, cw[3];
      qrot(q2, cl, cw);
      float cd = dot3(p2, n) - dot3(p1, n);
      *dist = cd + dot3(n, cw);
      for (int k = 0; k < 3; k++) pos[k] = cw[k] + p2[k] - 0.5f * n[k] * (*dist);
      make_frame(n, fr);
      if (!(-0.5f < n[1] && n[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
      if (cot) {  // c = x + cw: xbar = cbar, tbar = cw x cbar; cbar = pbar - n (n . pbar) / 2 + n dbar
        float pn = dot3(n, cot + 1);
        for (int k = 0; k < 3; k++) gb[6 + k] = cot[1 + k] + n[k] * (cot[0] - 0.5f * pn);
        cross(cw, gb + 6, gb + 9);
      }
      return br;
    }
    if (t2 != 2) return -1;\n""", 1),
)

ENGINE_DAMPING = (
    ("    c_inert[NB * 3], c_grav[3], c_armature[NV], c_kp[NV], c_kv[NV], c_frclim[NV], c_misc[2];\n",
     "    c_inert[NB * 3], c_grav[3], c_armature[NV], c_kp[NV], c_kv[NV], c_frclim[NV], c_misc[2];\n"
     "DKC float c_damp[NV];  // joint damping (model_hooks)\n", 1),
    ("      s.qsm[i] = act - s.qsm[i];",
     "      s.qsm[i] = act - c_damp[i] * s.qvel[i] - s.qsm[i];  // the passive damping force (model_hooks)", 1),
    ("{ s.L[TRI(i, i)] += h * c_kv[i]; s.tmp[i] = s.Ma[i]; }",
     "{ s.L[TRI(i, i)] += h * (c_kv[i] + c_damp[i]); s.tmp[i] = s.Ma[i]; }", 1),
    ("{ s.abar[i] -= h * c_kv[i] * s.y[i]; s.ny[i] = -s.y[i]; }",
     "{ s.abar[i] -= h * (c_kv[i] + c_damp[i]) * s.y[i]; s.ny[i] = -s.y[i]; }", 1),
    ("{ float x = s.did[i] - h * c_kv[i] * s.a[i]; acc += x * x; }",
     "{ float x = s.did[i] - h * (c_kv[i] + c_damp[i]) * s.a[i]; acc += x * x; }", 1),
    ("    s.ubar[i - 6] += kl;\n  }\n  __syncwarp();\n",
     "    s.ubar[i - 6] += kl;\n  }\n  __syncwarp();\n"
     "  for (int i = lane; i < NV; i += 32) s.Pv[i] += c_damp[i] * s.lam[i];  // joint damping (model_hooks)\n"
     "  __syncwarp();\n", 1),
    ("    float r = s.did[i] + c_armature[i] * s.a[i] - tau - s.jtf[i];",
     "    float r = s.did[i] + c_armature[i] * s.a[i] - tau + c_damp[i] * s.v[i] - s.jtf[i];", 1),
)

ENGINE_FLOSS = (  # frictionloss rows: the first NFLOSS rows (dof order), then the limit rows, then the contacts'
    ("      const int r = nl + __popc(bal & below);",
     "      const int r = NFLOSS + nl + __popc(bal & below);  // after the frictionloss rows (model_hooks)", 1),
    ("    const int r0 = nlim + c + 3 * (ntri + __popc(b3 & below));  // the contact's first row",
     "    const int r0 = NFLOSS + nlim + c + 3 * (ntri + __popc(b3 & below));  // the contact's first row (model_hooks)",
     1),
    ("""  const unsigned below = (1u << lane) - 1u;
  int nl = 0;
#pragma unroll 1
  for (int i0 = 0; i0 < NV; i0 += 32) {
    const int i = min(i0 + lane, NV - 1);  // clamped so that the loads need no branch (as the forward)""",
     """  const unsigned below = (1u << lane) - 1u;
  // the frictionloss rows (model_hooks): row r = the number of frictionloss dofs before (dof order, the first rows),
  // J = e_i, pos 0, aref = -b v_i (substep_pre), constant D; on its quadratic zone (the tape's bit) the row force
  // -D (a_i + b v_i) gives P_v += D b lam (limit_row_vjp without the position terms); its linear pieces are constant
  int nf = 0;
#pragma unroll 1
  for (int i0 = 0; i0 < NV; i0 += 32) {
    const int i = i0 + lane;
    const bool on = i < NV && c_floss[i] > 0.f;
    const unsigned bal = __ballot_sync(0xffffffffu, on);
    if (on) {
      const int r = nf + __popc(bal & below);
      if ((s.zone[r >> 5] >> (r & 31)) & 1u) {
        float ri[6];
        adj_row_impedance(c_dsolref, c_dsolimp, 0.f, c_dinvw[i], ri);
        s.Pv[i] += ri[4] * ri[1] * s.lam[i];
      }
    }
    nf += __popc(bal);
  }
  int nl = 0;
#pragma unroll 1
  for (int i0 = 0; i0 < NV; i0 += 32) {
    const int i = min(i0 + lane, NV - 1);  // clamped so that the loads need no branch (as the forward)""", 1),
)


def box_box_hook():
    """(the text of step.py's box_box, its replacement): MuJoCo Warp's native CCD for box-box (nablaengine.ccd,
    ccd_box_box: up to 4 contacts by multicontact) through a CSink into the world's contacts."""
    from nablaengine import ccd
    from nablaengine.g1_engine import step
    src = step.STEP_DEVICE
    head = ("template <class S>\n__device__ void box_box(S& s, int pair, const float* p1, const float* R1, "
            "const float* s1, const float* p2,\n                        const float* R2, const float* s2) {\n")
    i = src.index(head)
    k = src.index("#else\n  for (int i = 0; i < n; i++) {\n    if (!(depth[i] < 0.f)) continue;", i)
    wrapper = """
template <class S>
__device__ void box_box(S& s, int pair, const float* p1, const float* R1, const float* s1, const float* p2,
                        const float* R2, const float* s2) {  // (model_hooks: MuJoCo Warp's native CCD, nablaengine.ccd)
  CSink o;
  ccd_box_box(p1, R1, s1, p2, R2, s2, o);
  for (int i = 0; i < o.n; i++) add_contact(s, pair, o.slot[i], o.dist[i], o.pos[i], o.fr[i]);
  if (o.unsupported) atomicOr(&s.unsupported, o.unsupported);
}
"""
    return src[i:src.index("\n}\n", k) + 3], ccd.DEVICE + wrapper


ADJ_BOX_BOX = """  if (t1 == 3 && t2 == 3) {  // box-box (model_hooks): the forward's CCD contact `slot` and its exact VJP
    // (nablaengine.ccd ccd_box_box_vjp: clipped points as plane intersections, EPA vertex-face and edge-edge contacts)
    float R1[9], R2[9];
    quat2mat(q1, R1);
    quat2mat(q2, R2);
    if (!ccd_box_box_vjp(p1, R1, sz1, p2, R2, sz2, slot, cot, gb, dist, pos, fr)) return -1;
    if (!(-0.5f < fr[1] && fr[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
    return br;
  }
"""


def has_floss(m):
    return bool((m.dof_frictionloss > 0).any())


def box_pairs(m):
    """(there is a pair with a box, every such pair is plane-box or box-box: the adjoint has them, there is a box-box
    pair)."""
    import mujoco
    box, plane = int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_PLANE)
    pairs = [(int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]])) for p in range(m.npair)]
    with_box = [t for t in pairs if box in t]
    return bool(with_box), all(plane in t or t == (box, box) for t in with_box), (box, box) in with_box


def needed(m):
    """True where model m needs edits (False for the G1)."""
    return (m.nv < 32 or source.child_width(m) > 3 or box_pairs(m)[0] or source.has_damping(m)
            or has_floss(m))


def _apply(src, hooks):
    for a, b, n in hooks:
        if src.count(a) != n:
            raise RuntimeError(f"model hook anchor found {src.count(a)} times, expected {n}: {a[:70]!r}")
        src = src.replace(a, b)
    return src


def apply(src, m, part="engine"):
    """src with the hooks model m needs; part: "engine" (adjoint.cuda_source's text) or "kinematics"
    (kinematics.KIN_DEVICE). Unchanged for the G1."""
    if not needed(m):
        return src
    hooks = ()
    if source.child_width(m) > 3:
        hooks += ENGINE_CHILD if part == "engine" else KIN_CHILD
    if m.nv < 32 and part == "engine":
        hooks += ENGINE_SMALL_NV
    if source.has_damping(m) and part == "engine":
        hooks += ENGINE_DAMPING
    has_box, box_ok, boxbox = box_pairs(m)
    if (has_box or has_floss(m)) and part == "engine":
        hooks += ENGINE_ADJ_SCOPE
    if has_floss(m) and part == "engine":
        hooks += ENGINE_FLOSS
    if has_box and part == "engine":
        if not box_ok:
            raise ValueError("the adjoint has box contacts with planes and boxes only (no sphere-box / capsule-box)")
        hooks += ENGINE_PLANE_BOX
    if boxbox and part == "engine":
        orig, fn = box_box_hook()
        from nablaengine import ccd
        hooks += ((orig, fn, 1), ("  if (t1 != 2 || t2 != 2) return -1;\n",
                                  ADJ_BOX_BOX + "  if (t1 != 2 || t2 != 2) return -1;\n", 1),
                  ("__device__ __noinline__ int adj_contact_geom(", ccd.ADJ_DEVICE + "__device__ __noinline__ int adj_contact_geom(", 1))
    src = _apply(src, hooks)
    if part == "engine":  # NCH for the readers (the kinematics follow the engine in the same unit)
        extra = ""
        if boxbox:
            from nablaengine import ccd
            extra = ccd.header(m)
        src = src.replace("#define NTRI (NV * (NV + 1) / 2)\n",
                          f"#define NTRI (NV * (NV + 1) / 2)\n#define NCH {source.child_width(m)}  // (model_hooks)\n"
                          + extra, 1)
    return src
