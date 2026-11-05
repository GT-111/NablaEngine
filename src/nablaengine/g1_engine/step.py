# Third-party code: the collision primitives in STEP_DEVICE (plane against sphere / capsule / box, sphere-sphere,
# sphere-capsule, capsule-capsule, sphere-box, capsule-box) are translated to CUDA from MuJoCo Warp,
# mujoco_warp/_src/collision_primitive_core.py, and make_frame from mujoco_warp/_src/math.py (github.com/google-deepmind/
# mujoco_warp; the version used here: etaoxing/mujoco_warp 357a75d, the head of google-deepmind/mujoco_warp pull
# request 1535), Copyright 2025 The Newton Developers, Apache License 2.0; parts of those follow MuJoCo's C
# implementation (Copyright DeepMind Technologies Limited, Apache License 2.0). Modified: float32 with explicit
# roundings where the adjoint replays them, the pairs spread over a warp's lanes, margin 0 only. The constraint rows and
# the Newton solver implement MuJoCo's / MuJoCo Warp's definitions (constraint.py, solver.py) and are this engine's own
# code.
"""CUDA source of one G1 physics substep (block per world): collision, constraints, Newton solver, implicitfast update.

The constraint half of the G1 engine's substep, built on the smooth-dynamics device code of
nablaengine.g1_engine.source. One warp per world, all state in shared memory (struct Sim). Follows MuJoCo Warp's model
for this robot: explicit and eligible mask-filtered contact pairs (plane against sphere / capsule / box, sphere-sphere, sphere-capsule,
capsule-capsule and capsule-box, translated from MuJoCo Warp's primitives; pairs whose bounding spheres do not
overlap are skipped, which is exact at zero margin), pyramidal cones for condim 3, frictionless rows for condim 1, dof
frictionloss and hinge limits, impedance and reference acceleration from solref / solimp with the invweight0
approximation (refsafe on), and the primal Newton solver on cost(qacc) = 0.5 (qacc - qacc_smooth)' M (qacc -
qacc_smooth) + sum_rows s(J qacc - aref). The Newton steps use an exact line search (safeguarded 1-D Newton on the
piecewise-quadratic cost), so a converged solve matches MuJoCo's unique minimizer. implicitfast: qacc' = (M + h kv)^-1
M qacc, qvel += h qacc', qpos integrated with the new qvel.

pair_constants(m, foot_pairs) packs, next to source.constants, the contact-pair table (PAIR_LAYOUT: one word per pair with
geom indices, parameter set, condim and foot flag; the parameter sets in c_pset), the collision-geom table (GEOM_LAYOUT: body, type, size, local pose, bounding radius of every geom
in a pair) and the solver constants (SOLVER_LAYOUT); const_layout(m) gives the symbols and lengths in that order. The
per-pair geometry lives in the geom table so that the 502 pairs of the mjlab model fit in 64 KB of constant memory.

The forward tape (nablaengine.g1_engine.tape): substep(..., rec) writes one tape record when rec is not null (the
code needs tape.header(m) for the OFF_* offsets); the forward outputs are bitwise the same with and without it.

The device functions are templates over the state struct: struct Sim (k_substep, everything in shared memory) or the
split forward's SimP / SimN (host.py k_pre / k_post: each half's own shared state, the rest in struct Hand in global
memory; same member names, same code, bitwise the same results).
"""

import math
import os
import warnings

import mujoco
import numpy as np

PROF_PHASES = ("smooth", "qacc_smooth", "collision", "contact_jac", "rows", "newton_init", "newton_grad",
               "newton_hessian", "newton_chol", "newton_linesearch", "newton_update", "integrate", "ls_jv_Mp",
               "count_ls_steps",
               # sub-phases (SUBPROF(14) ..: cycles since the previous phase or sub-phase mark), parts of the above
               "sm_kin", "sm_force", "sm_accum", "sm_crb",  # smooth: tree passes, mass matrix
               "ni_matvec", "ni_rowsjv",  # newton_init (refresh)
               "ng_rows", "ng_jtf_conf", "ng_jtf_dof", "ng_jtf_scatter",  # newton_grad: forces, J' f
               "ls_rowsjv", "ls_matvec",  # newton_linesearch: J p, M p
               "col_geom", "nh_conC", "nh_copy_dof", "col_broad")  # collision, newton_hessian
PROF_PARTITION = 12  # the first 12 phases partition a substep; then a part of one (cycles) and a count
PAIR_LAYOUT = (  # (symbol, kind, per-pair width): one packed word per pair, see pair_word()
    ("c_pword", "u", 1),
)
PSET_WIDTH = 8  # a contact parameter set in c_pset: friction mu, solref[2], solimp[5]
GEOM_LAYOUT = (  # (symbol, kind, per-geom width)
    ("c_gbody", "i", 1), ("c_gtype", "i", 1), ("c_gsize", "f", 3), ("c_gpos", "f", 3), ("c_gquat", "f", 4),
    ("c_grbound", "f", 1),
)
SOLVER_LAYOUT = (  # (symbol, kind, length in units of the body count, "nv", or a number)
    ("c_binvw", "f", "nb"), ("c_dinvw", "f", "nv"), ("c_floss", "f", "nv"), ("c_jrange", "f", "nv2"),
    ("c_jlim", "f", "nv"), ("c_jsolref", "f", 2), ("c_jsolimp", "f", 5), ("c_dsolref", "f", 2), ("c_dsolimp", "f", 5),
    ("c_sopt", "f", 4))
EQ_LAYOUT = (("c_eqdof1", "i", 1), ("c_eqdof2", "i", 1), ("c_eqpoly", "f", 2),
             ("c_eqsolref", "f", 2), ("c_eqsolimp", "f", 5), ("c_eqinvw", "f", 1))
GEOM_TYPE = {int(mujoco.mjtGeom.mjGEOM_PLANE): 0, int(mujoco.mjtGeom.mjGEOM_SPHERE): 1,
             int(mujoco.mjtGeom.mjGEOM_CAPSULE): 2, int(mujoco.mjtGeom.mjGEOM_BOX): 3,
             int(mujoco.mjtGeom.mjGEOM_MESH): 4}


def contact_pairs(m):
    """Actual explicit and eligible mask-filtered pairs, in the generated contact-table order."""
    from nablaengine.model.capability import filter_pairs
    return tuple((int(a), int(b)) for a, b in zip(m.pair_geom1, m.pair_geom2)) + tuple(filter_pairs(m))


def contact_parameters(m):
    """MuJoCo's explicit parameters and priority/solmix mixing for implicit pairs."""
    pairs = contact_pairs(m)
    dim, friction, solref, solimp, margin, gap = [], [], [], [], [], []
    for p, (g1, g2) in enumerate(pairs):
        if p < m.npair:
            dim.append(m.pair_dim[p])
            friction.append(m.pair_friction[p])
            solref.append(m.pair_solref[p])
            solimp.append(m.pair_solimp[p])
            margin.append(m.pair_margin[p])
            gap.append(m.pair_gap[p])
            continue
        priority = m.geom_priority[[g1, g2]]
        if priority[0] != priority[1]:
            g = g1 if priority[0] > priority[1] else g2
            cd, fr, sr, si = m.geom_condim[g], m.geom_friction[g], m.geom_solref[g], m.geom_solimp[g]
        else:
            cd = max(m.geom_condim[g1], m.geom_condim[g2])
            fr = np.maximum(m.geom_friction[g1], m.geom_friction[g2])
            weight = m.geom_solmix[g1] + m.geom_solmix[g2]
            mix = m.geom_solmix[g1] / weight if weight >= mujoco.mjMINVAL else 0.5
            sr = (mix * m.geom_solref[g1] + (1 - mix) * m.geom_solref[g2]
                  if np.all(m.geom_solref[[g1, g2]] > 0) else np.minimum(m.geom_solref[g1], m.geom_solref[g2]))
            si = mix * m.geom_solimp[g1] + (1 - mix) * m.geom_solimp[g2]
        dim.append(cd)
        friction.append(fr[[0, 0, 1, 2, 2]])
        solref.append(sr)
        solimp.append(si)
        margin.append(max(m.geom_margin[g1], m.geom_margin[g2]))
        gap.append(max(m.geom_gap[g1], m.geom_gap[g2]))
    n = len(pairs)
    return dict(dim=np.asarray(dim, int), friction=np.asarray(friction, float).reshape(n, 5),
                solref=np.asarray(solref, float).reshape(n, 2), solimp=np.asarray(solimp, float).reshape(n, 5),
                margin=np.asarray(margin, float), gap=np.asarray(gap, float))


def contact_multiplicity(m):
    """Maximum emitted points of each generated pair under the model's CCD options."""
    from nablaengine.model.capability import CONTACTS_PER_PAIR

    multi = not (m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD))
    counts = []
    for a, b in contact_pairs(m):
        types = tuple(sorted((GEOM_TYPE.get(int(m.geom_type[a]), -1), GEOM_TYPE.get(int(m.geom_type[b]), -1))))
        counts.append(1 if 4 in types and not multi else CONTACTS_PER_PAIR.get(types, 1))
    return np.asarray(counts, np.int64)


def maxcon(m):
    """Per-world point storage from the authored budget, bounded by the pair graph.

    Automatic MuJoCo arenas have no point declaration (nconmax=-1). Use a
    bounded 256-point buffer there; actual overflow remains an explicit error.
    A warp is the minimum storage tile. $DK_MAXCON is an experimental override.
    """
    budget = int(m.nconmax) if m.nconmax > 0 else 256
    default = max(32, min(budget, int(contact_multiplicity(m).sum())))
    if os.environ.get("DK_MAXCON"):
        warnings.warn(f"DK_MAXCON={os.environ['DK_MAXCON']} overrides the contact capacity (default {default}) of the "
                      "engine builds and the FFI library (an experiment switch)", RuntimeWarning, stacklevel=2)
        return int(os.environ["DK_MAXCON"])
    return default


def nefc(m, max_contacts=None):
    """Rows for every stored point plus every supported noncontact constraint."""
    counts = contact_multiplicity(m)
    dims = contact_parameters(m)["dim"]
    ntri = int(counts[dims == 3].sum())
    capacity = max_contacts or maxcon(m)
    contact_rows = 4 * min(capacity, ntri) + min(max(0, capacity - ntri), int(counts[dims == 1].sum()))
    noncontact = int((m.dof_frictionloss > 0).sum()) + int(m.neq) + int(np.count_nonzero(m.jnt_limited))
    return max(1, int(m.njmax), noncontact + contact_rows)


def max_support(m):
    """Largest number of dofs a contact Jacobian can touch: the dofs on exactly one of the two bodies' chains, over all
    pairs (16 for both G1 models: a hand on the floor, root 6 + waist 3 + arm 7)."""
    dofbody, mask = extra_constants(m)
    return max((sum(1 for i in range(m.nv) if ((mask[b1] >> int(dofbody[i])) ^ (mask[b2] >> int(dofbody[i]))) & 1)
                for b1, b2 in ((m.geom_bodyid[g1], m.geom_bodyid[g2]) for g1, g2 in contact_pairs(m))), default=0)


def collision_geoms(m):
    """The geoms that appear in a contact pair, in model order (index into the geom table)."""
    return sorted({g for pair in contact_pairs(m) for g in pair})


def native_collision(m):
    """Whether a pair needs the native mesh or specialized box-box collider."""
    return any(mujoco.mjtGeom.mjGEOM_MESH in (m.geom_type[a], m.geom_type[b])
               or m.geom_type[a] == m.geom_type[b] == mujoco.mjtGeom.mjGEOM_BOX
               for a, b in contact_pairs(m))


def pair_word(g1, g2, pset, dim, foot, plane=0):
    """A contact pair as one 32-bit word: geom table indices g1, g2 (8 bits each), parameter set (8 bits), condim (4 bits),
    foot-floor flag (1 bit), plane flag (1 bit: g1 is a plane); decoded on the device by PG1 / PG2 / PSET / PDIM /
    PFOOT / PPLANE."""
    assert g1 < 256 and g2 < 256 and pset < 256 and dim < 16
    return g1 | g2 << 8 | pset << 16 | dim << 24 | foot << 28 | plane << 29


def pair_param_sets(m):
    """(unique (mu, solref, solimp) rows as an (n, PSET_WIDTH) array, index of each pair's row): the 502 mjlab pairs
    share a handful of parameter sets, so the pair table is one word per pair."""
    keys, index = {}, []
    params = contact_parameters(m)
    for i in range(len(contact_pairs(m))):
        key = (float(params["friction"][i, 0]), *map(float, params["solref"][i]), *map(float, params["solimp"][i]))
        index.append(keys.setdefault(key, len(keys)))
    return np.asarray(list(keys), np.float64).reshape(len(keys), PSET_WIDTH), index


def const_layout(m):
    """[(symbol, kind, length)] of pair_constants(m, ...) in packing order."""
    ng = len(collision_geoms(m))
    sizes = {"nb": m.nbody, "nv": m.nv, "nv2": 2 * m.nv}
    npset = len(pair_param_sets(m)[0])
    entries = ([(s, k, len(contact_pairs(m)) * w) for s, k, w in PAIR_LAYOUT] + [("c_pset", "f", npset * PSET_WIDTH)]
               + [(s, k, ng * w) for s, k, w in GEOM_LAYOUT]
               + [(s, k, sizes[w] if isinstance(w, str) else w) for s, k, w in SOLVER_LAYOUT]
               + [(s, k, m.neq * w) for s, k, w in EQ_LAYOUT])
    return [(s, k, n) for s, k, n in entries if n]


def pair_constants(m, foot_pairs):
    """Contact-pair arrays (geoms ordered by type, as MuJoCo's collision table expects), the collision-geom table and
    the solver constants, as one float32 array in const_layout(m) order."""
    geoms = collision_geoms(m)
    gid = {g: i for i, g in enumerate(geoms)}
    sets, words = pair_param_sets(m), []
    pairs, params = contact_pairs(m), contact_parameters(m)
    foot = set(int(x) for x in foot_pairs)
    assert foot <= set(range(len(pairs))), "friction overrides must name generated contact-pair indices"
    for i, (g1, g2) in enumerate(pairs):
        if GEOM_TYPE[int(m.geom_type[g1])] > GEOM_TYPE[int(m.geom_type[g2])]:
            g1, g2 = g2, g1
        assert params["margin"][i] == 0 and params["gap"][i] == 0 and params["friction"][i, 0] == params["friction"][i, 1]
        words.append(pair_word(gid[g1], gid[g2], sets[1][i], int(params["dim"][i]), int(i in foot),
                               int(GEOM_TYPE[int(m.geom_type[g1])] == 0)))
    words = np.asarray(words, np.int64)
    out = [np.stack([words & 0xFFFF, words >> 16], 1).ravel(), np.asarray(sets[0], np.float64).ravel()]
    out += [m.geom_bodyid[geoms], [GEOM_TYPE[int(m.geom_type[g])] for g in geoms], m.geom_size[geoms],
            m.geom_pos[geoms], m.geom_quat[geoms], m.geom_rbound[geoms]]
    jr = np.zeros((m.nv, 2))
    jlim = np.zeros(m.nv)
    for j in range(1, m.njnt):
        jr[m.jnt_dofadr[j]], jlim[m.jnt_dofadr[j]] = m.jnt_range[j], m.jnt_limited[j]
    assert not m.jnt_margin.any() and (m.jnt_solref[1:] == m.jnt_solref[1]).all()
    assert (m.jnt_solimp[1:] == m.jnt_solimp[1]).all() and (m.dof_solref == m.dof_solref[0]).all()
    assert (m.dof_solimp == m.dof_solimp[0]).all() and m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
    out += [m.body_invweight0[:, 0], m.dof_invweight0, m.dof_frictionloss, jr, jlim, m.jnt_solref[1], m.jnt_solimp[1],
            m.dof_solref[0], m.dof_solimp[0],
            [m.opt.impratio, m.stat.meaninertia, m.opt.tolerance, len(pairs)]]
    d1, d2 = m.jnt_dofadr[m.eq_obj1id], m.jnt_dofadr[m.eq_obj2id]
    # MuJoCo's joint-equality inverse-weight approximation omits the mimic slope.
    out += [d1, d2, m.eq_data[:, :2], m.eq_solref, m.eq_solimp,
            m.dof_invweight0[d1] + m.dof_invweight0[d2]]
    return np.concatenate([np.asarray(x, np.float64).ravel() for x in out]).astype(np.float32)


def hess_columns(kdof, passes=3):
    """Support pairs (a, b), b <= a < kdof, of the Hessian assembly per lane: one fixed column b per lane and set, so
    that a lane computes G_b = C J_b once per contact. Set k covers the rows lo_k <= a < lo_(k+1), the largest range
    whose columns (each split over ceil(rows / passes) lanes, rows interleaved) fit 32 lanes. Returns [(lo, hi, table)],
    table[pass][lane] = (a, b, thr), the pair valid when a < nd, i.e. nd > thr (thr = 255: none)."""
    sets, lo = [], 0
    while lo < kdof:
        def lanes(hi):
            return sum(math.ceil((hi - max(lo, b)) / passes) for b in range(hi))
        hi = max(h for h in range(lo + 1, kdof + 1) if lanes(h) <= 32)
        tab = [[(0, 0, 255)] * 32 for _ in range(passes)]
        lane = 0
        for b in range(hi):
            rows = math.ceil((hi - max(lo, b)) / passes)
            for j in range(rows):
                for t in range(passes):
                    a = max(lo, b) + j + rows * t
                    tab[t][lane] = (a, b, a) if a < hi else (b, b, 255)
                lane += 1
        sets.append((lo, hi, tab))
        lo = hi
    return sets


def floor_geom(m):
    """Index in the collision-geom table of the model's only plane, or -1 (none or several)."""
    planes = [i for i, g in enumerate(collision_geoms(m)) if GEOM_TYPE[int(m.geom_type[g])] == 0]
    return planes[0] if len(planes) == 1 else -1


def solimp_pow2(m):
    """Whether every solimp power the rows use (contact pairs, joint limits, dof frictionloss) is 2."""
    pw = (list(contact_parameters(m)["solimp"][:, 4]) + list(m.jnt_solimp[1:, 4])
          + list(m.dof_solimp[m.dof_frictionloss > 0, 4]) + list(m.eq_solimp[:, 4]))
    return all(float(x) == 2.0 for x in pw)


def dof_body_root6(m):
    """True when body 1 (the free root) has dofs 0-5 and every later dof i is the only dof of body i - 4 (bodies and
    dofs in the same order): a contact's support mask is then a shift of its bodies' chain mask (support_mask)."""
    d = np.asarray(m.dof_bodyid)
    return m.nv >= 6 and bool((d[:6] == 1).all() and (d[6:] == np.arange(6, m.nv) - 4).all())


def step_header(m, max_contacts=None):
    p, g, kdof = len(contact_pairs(m)), len(collision_geoms(m)), max_support(m)
    nc, nr = max_contacts or maxcon(m), nefc(m, max_contacts)
    hs = hess_columns(kdof)
    hcol = [a | b << 8 | thr << 16 for _, _, tab in hs for row in tab for a, b, thr in row]
    return f"""
#define NPAIR {p}
#define CONTACT_DISABLED {int(bool(m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)))}
#define NGEOMC {g}
#define NPAIR_STORAGE {max(1, p)}
#define NGEOMC_STORAGE {max(1, g)}
#define MAXCON {nc}
#define CON_TILES {(nc + 31) // 32}
#define CON_COUNT_BITS {nc.bit_length()}
#define ROW_COUNT_BITS {nr.bit_length()}
#define KDOF {kdof}
#define KDOF_STORAGE {max(1, kdof)}
#define HSETS {len(hs)}  // Hessian assembly (hess_columns(KDOF)): sets of 3 passes, their first rows, pairs per lane
#define HSETS_STORAGE {max(1, len(hs))}
#define HSET_LO {{{", ".join(str(lo) for lo, _, _ in hs) or "0"}}}
__device__ const unsigned c_hcol[{max(1, len(hcol))}] = {{{", ".join(map(str, hcol)) or "0"}}};  // a | b << 8 | thr << 16
#define HAS_BOX {int(any(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in collision_geoms(m)))}  // compile box code
#define HAS_NATIVE_COLLISION {int(native_collision(m))}
#define HAS_PRIMITIVE_BOX {int(any(mujoco.mjtGeom.mjGEOM_BOX in (m.geom_type[a], m.geom_type[b])
                                 and mujoco.mjtGeom.mjGEOM_MESH not in (m.geom_type[a], m.geom_type[b])
                                 and not m.geom_type[a] == m.geom_type[b] == mujoco.mjtGeom.mjGEOM_BOX
                                 for a, b in contact_pairs(m)))}
#define FLOOR_G {floor_geom(m)}  // the geom table index of the only plane (-1: none or several)
#define NEFC {nr}
#define DOF_BODY_ROOT6 {int(dof_body_root6(m))}  // step.dof_body_root6 (support_mask in closed form)
#define NFLOSS {int((m.dof_frictionloss > 0).sum())}  // dofs with a frictionloss row (the first rows)
#define NEQ {m.neq}  // bilateral soft linear joint equalities, after frictionloss and before limits
#define NEQ_STORAGE {max(1, m.neq)}
#define SOLIMP_POW2 {int(solimp_pow2(m))}  // every solimp power of the model is 2 (kbimp without the general powers)
#define NPROF {len(PROF_PHASES)}
DKC unsigned int c_pword[NPAIR_STORAGE];  // pair_word(): g1 | g2 << 8 | pset << 16 | dim << 24 | foot << 28 | plane << 29
DKC float c_pset[{max(1, len(pair_param_sets(m)[0]) * PSET_WIDTH)}];  // per parameter set: mu, solref[2], solimp[5]
#define PG1(pr) ((int)(c_pword[pr] & 255u))
#define PG2(pr) ((int)((c_pword[pr] >> 8) & 255u))
#define PSET(pr) ((int)((c_pword[pr] >> 16) & 255u))
#define PDIM(pr) ((int)((c_pword[pr] >> 24) & 15u))
#define PFOOT(pr) ((int)((c_pword[pr] >> 28) & 1u))
#define PPLANE(pr) ((int)((c_pword[pr] >> 29) & 1u))
#define PMU(pr) (c_pset[PSET_WIDTH * PSET(pr)])
#define PSOLREF(pr) (c_pset + PSET_WIDTH * PSET(pr) + 1)
#define PSOLIMP(pr) (c_pset + PSET_WIDTH * PSET(pr) + 3)
#define PSET_WIDTH {PSET_WIDTH}
DKC int c_gbody[NGEOMC_STORAGE], c_gtype[NGEOMC_STORAGE];
DKC float c_gsize[NGEOMC_STORAGE * 3], c_gpos[NGEOMC_STORAGE * 3], c_gquat[NGEOMC_STORAGE * 4],
    c_grbound[NGEOMC_STORAGE];
DKC float c_binvw[NB], c_dinvw[NV], c_floss[NV], c_jrange[NV * 2], c_jlim[NV], c_jsolref[2], c_jsolimp[5],
    c_dsolref[2], c_dsolimp[5], c_sopt[4];  // impratio, meaninertia, tolerance, npair
DKC int c_eqdof1[NEQ_STORAGE], c_eqdof2[NEQ_STORAGE];
DKC float c_eqpoly[NEQ_STORAGE * 2], c_eqsolref[NEQ_STORAGE * 2], c_eqsolimp[NEQ_STORAGE * 5],
    c_eqinvw[NEQ_STORAGE];
"""


STEP_DEVICE = r"""
// NEWTON_REUSE: a Newton iteration whose rows all kept their zones (a third of them on a 4096-state corpus) re-uses
// the last factor (chol_resolve). Split forward, ms per control step (that corpus, 10 iterations): RTX 3090 3.07
// against 3.34 without (1.66 / 1.77 at 1787 worlds, the waves of 4096 worlds on the target GPUs); V100 2% slower
// (6.34 / 6.20)
#ifndef NEWTON_REUSE
#define NEWTON_REUSE (NV > 32 && NV <= 35)
#endif
// The Newton reuse's guard (substep_post's zsame): H = M + J' C J changes only when a row enters or leaves its quadratic
// zone (dd != 0; both linear pieces of a frictionloss row have dd = 0 too), so the register-resolve path - bitwise a
// re-factorization's solve - may reuse the factor across a frictionloss row's 2 <-> 3 flip, and a tape record then
// needs no re-factorization at the final zones either (the tape's zone word records the quadratic rows, which did not
// change). The older NEWTON_REUSE resolve differs by roundings instead, so its builds keep the strict zone test: the
// guard only decides which code computes p, never what p is.
#if NR == 2 && !CHOL_BCAST && !CHOL_COMPACT
#define ZONE_CHANGED(z0, z1) (((z0) == 1) != ((z1) == 1))
#else
#define ZONE_CHANGED(z0, z1) ((z0) != (z1))
#endif
struct Sim {
  float Q[NB][4], X[NB][3];  // body poses
  union {  // the smooth phase's body quantities and the constraint rows are never alive at the same time
    struct {
      float W[NB][3], V[NB][3];
      union {  // the collision's candidate pairs (after the mass matrix)
        struct { float AL[NB][3], A[NB][3], F[NB][3], N[NB][3], J[NB][10]; };
        unsigned short cand[NPAIR_STORAGE];
      };
    };
    struct {
      unsigned char efc_kind[NEFC];
      unsigned short efc_a[NEFC];
      signed char efc_b[NEFC];
      float efc_D[NEFC], efc_aref[NEFC], efc_jar[NEFC], efc_jv[NEFC];  // efc_jv: J p in the line search, else forces
    };
  };
  float M[NTRI], L[NTRI];  // mass matrix and Cholesky scratch, packed lower triangles TRI(i, j)
  union { double H[NTRI]; float h_inv[NV]; };  // H dies at factorization; reuse its storage for pivots
  float qpos[NQ], qvel[NV], zero[NV], qsm[NV], qacc0[NV], qacc[NV], Ma[NV], grad[NV], p[NV], Mp[NV], tmp[NV];
  alignas(16) float cbuf[72];  // CHOL_SOLVE's scratch (chol_solve_bcast)
  float com[COM_WIDTH], fric;
  float gx[NGEOMC_STORAGE][3], gq[NGEOMC_STORAGE][4];  // world poses of the collision geoms
#if HAS_NATIVE_COLLISION
  float gaabb[NGEOMC_STORAGE][6];  // conservative native mesh/box bounds, once per geom
#endif
  int ncon, nefc, unsupported;
  int min_nu_bits;  // tape: smallest centre / axis distance (before the radii) of a sphere or capsule self contact, as
                    // float bits (non-negative: the integer order is the float order, so atomicMin works); +inf if none
  int con_pair[MAXCON], con_dim[MAXCON], con_row[MAXCON], con_nrow[MAXCON], nefc_dof;
  union {  // per-contact scratch, each consumed right after it is written
    float con_dot[MAXCON][3];  // J_c,d v (rows_jv)
    float con_f[MAXCON][3];    // a contact's row forces combined per frame direction (rows_jtf)
    float con_C[MAXCON][5];    // Hessian coefficients on (n n, n t1, t1 t1, n t2, t2 t2), see the Newton loop
  };
  // contact Jacobian rows (normal, two tangents) on the contact's support: the con_nd[c] dofs con_dof[c][k] (ascending)
  float con_dist[MAXCON], con_pos[MAXCON][3], con_frame[MAXCON][9], con_mu[MAXCON];
  alignas(16) float con_J[MAXCON][3][KDOF_STORAGE];  // (16-byte aligned: con_jv's 128-bit loads)
  alignas(16) unsigned char con_dof[MAXCON][KDOF_STORAGE];
  unsigned char con_slot[MAXCON];  // which contact of its pair (tape): plane-capsule end, parallel capsule-capsule candidate
  int con_nd[MAXCON];
  unsigned long long con_mask[MAXCON];  // the support as dof bits (bit i: dof i; con_dof[c][k] is the k-th set bit)
#if HAS_NATIVE_COLLISION
  CMScratch cm_storage;
  CMScratch* cm_scratch;
  float* feature_rec;
#endif
#ifdef DK_PROFILE
  long long prof[NPROF], prof_t, prof_u;
#endif
#ifdef DK_TAPE_VERIFY
  float Lv[NTRI];  // debug: the factor of the last Newton assembly, compared with a re-assembly at the final zones
#endif
};

// a condition that is the same on every lane (from warp_sum or the like), as a vote: the compiler then knows the branch
// is uniform. Without it the Newton loop counts as divergent control flow and every shuffle and __syncwarp() in it, and
// in the functions it calls (CHOL_SOLVE), gets a divergence check (BRA.DIV / BRA.CONV, and no scheduling across it):
// chol_solve_bcast 8.2k cycles from a uniform caller, 12.9k from a divergent one (V100). The other source of such
// checks is add_contact's shared atomicAdd, whose result is used in the divergent collision code: in k_substep it costs
// the whole kernel its convergence, in the split forward only k_pre. Split forward, ms per control step (a 4096-state
// corpus, 10 iterations, alternating runs): V100 6.34 against 7.27 without (-DDK_UNIFORM=0), RTX 3090 3.07 against
// 3.14
#ifndef DK_UNIFORM
#define DK_UNIFORM 1
#endif
#if DK_UNIFORM
#define UNIFORM(c) __all_sync(0xffffffffu, (c))
#else
#define UNIFORM(c) (c)
#endif
// the same for an integer that is the same on every lane (a count in shared memory): its bits through votes, for loop
// bounds around shuffles and ballots
__device__ __forceinline__ int uniform_int(int x, int bits) {
#if DK_UNIFORM
  int u = 0;
#pragma unroll
  for (int b = 0; b < bits; b++) u |= (int)__any_sync(0xffffffffu, (x >> b) & 1) << b;
  return u;
#else
  return x;
#endif
}

// phase boundary: a __syncwarp() in every build (the code relies on it), plus per-phase cycle counters of one world
// (lane 0) with -DDK_PROFILE; phases: PROF_PHASES in step.py
#ifdef DK_PROFILE
#define PROF_START() do { __syncwarp(); if (lane == 0) s.prof_t = s.prof_u = clock64(); } while (0)
#define PROF(k) do { __syncwarp(); if (lane == 0) { long long t_ = clock64(); s.prof[k] += t_ - s.prof_t; s.prof_t = s.prof_u = t_; } __syncwarp(); } while (0)
#define SUBPROF(k) do { __syncwarp(); if (lane == 0) { long long t_ = clock64(); s.prof[k] += t_ - s.prof_u; s.prof_u = t_; } __syncwarp(); } while (0)
#else
#define SUBPROF(k)
#define PROF_START() __syncwarp()
#define PROF(k) __syncwarp()
#endif

// inlining of the larger helpers: code size (instruction fetch) against call overhead (an A / B switch, e.g.
// -DDK_INL_MATVEC=__noinline__). newton_grad not inlined (one copy instead of two): RTX 3090 1.3% faster, V100 0.2%;
// matvec, total_cost or refresh not inlined: no gain beyond the noise on the RTX 3090
// DK_JTF_GATHER: rows_jtf's contact part J' f as a per-dof gather over the contacts' support masks (con_mask, written
// by substep_pre), in registers and without a __syncwarp() per contact, instead of a scatter of each contact over its
// support (0). Split forward, ms per control step (a 4096-state corpus, 10 iterations, alternating runs, the other
// switches of this group on):
// V100 6.22 against 6.31 for the scatter (1787 worlds 2.60 / 2.64), RTX 3090 -1.3% (1787 worlds -0.7%)
#ifndef DK_JTF_GATHER
#define DK_JTF_GATHER 1
#endif
// DK_JV_VEC: con_jv reads a contact row's Jacobian entries and support dofs from Hand with 128-bit loads (con_J,
// con_dof 16-byte aligned; RTX 3090 -1.2 to -1.5%, V100 -0.4%); DK_GRAD_COST: the line search's new cost is summed
// in newton_grad's |grad| reduction (one interleaved butterfly; -0.1 to -0.2%); DK_HESS_DOFPAR: the Hessian's dof-row
// curvatures added to L by one lane per row (the dof rows' diagonals are distinct when there are no frictionloss rows:
// one limit row per dof) instead of one row at a time with two shuffles each (RTX 3090 -0.6%, V100 -0.3%)
#ifndef DK_JV_VEC
#define DK_JV_VEC 1
#endif
#ifndef DK_GRAD_COST
#define DK_GRAD_COST 1
#endif
#ifndef DK_HESS_DOFPAR
#define DK_HESS_DOFPAR 1
#endif
#ifndef DK_INL_COLLIDE
#define DK_INL_COLLIDE
#endif
#ifndef DK_INL_ROWSJV
#define DK_INL_ROWSJV
#endif
#ifndef DK_INL_GRAD
#define DK_INL_GRAD __noinline__
#endif
#ifndef DK_INL_MATVEC
#define DK_INL_MATVEC __forceinline__
#endif
#ifndef DK_INL_COST
#define DK_INL_COST
#endif

__device__ __forceinline__ void normalize3_rn(float* v) {
  float n = sqrtf(dot3_rn(v, v));
  if (n > 0.f) { v[0] /= n; v[1] /= n; v[2] /= n; }
}

// MuJoCo's orthogonals(a) / make_frame(a): rows n, b, c (explicit roundings, as all of the contact geometry below:
// the adjoint's replay computes the same bits, see source.DEVICE)
__device__ __forceinline__ void make_frame(const float* n, float* fr) {
  float b[3] = {0.f, 0.f, 0.f};
  if (-0.5f < n[1] && n[1] < 0.5f) b[1] = 1.f; else b[2] = 1.f;
  float d = dot3_rn(n, b);
  for (int k = 0; k < 3; k++) b[k] = RFMS(n[k], d, b[k]);
  normalize3_rn(b);
  float c[3];
  cross_rn(n, b, c);
  for (int k = 0; k < 3; k++) { fr[k] = n[k]; fr[3 + k] = b[k]; fr[6 + k] = c[k]; }
}

// slot: which contact of the pair this is (the tape's contact table, tape.py): the end e of a plane-capsule pair
// (0: +axis, 1: -axis), the candidate e = 0..3 of a parallel capsule-capsule pair, a box corner / clip point; else 0.
// (The atomicAdd's result in divergent code: see UNIFORM)
template <class S>
__device__ __forceinline__ bool add_contact(S& s, int pair, int slot, float dist, const float* pos,
                                            const float* frame
#if HAS_NATIVE_COLLISION
                                            , const CMFeature* feature = nullptr
#endif
                                            ) {
  int c = atomicAdd(&s.ncon, 1);
  if (c >= MAXCON) return false;
  s.con_pair[c] = pair;
  s.con_slot[c] = (unsigned char)slot;
  s.con_dim[c] = PDIM(pair);
  s.con_dist[c] = dist;
  for (int k = 0; k < 3; k++) s.con_pos[c][k] = pos[k];
  for (int k = 0; k < 9; k++) s.con_frame[c][k] = frame[k];
  s.con_mu[c] = PFOOT(pair) ? s.fric : PMU(pair);
#if HAS_NATIVE_COLLISION
  if (s.feature_rec) {
    float* words = s.feature_rec + c * CM_FEATURE_WORDS;
    if (feature) cm_feature_pack(words, *feature);
    else for (int k = 0; k < CM_FEATURE_WORDS; k++) words[k] = 0.f;
  }
#endif
  return true;
}

// a contact from the sphere core (sphere-sphere, sphere-capsule, capsule-capsule): frame from the normal n, and the tape's
// min_nu (nu: the centre or axis distance)
template <class S>
__device__ __forceinline__ void add_sphere_contact(S& s, int pair, int slot, float nu, float dist, const float* pos,
                                                   const float* n) {
  float fr[9];
  make_frame(n, fr);
  if (add_contact(s, pair, slot, dist, pos, fr)) atomicMin(&s.min_nu_bits, __float_as_int(nu));
}

// world pose (centre gx, rotation quaternion gq) of collision geom g on its body's pose (Qb, Xb): the forward's
// geom_poses and the adjoint's replay (adjoint.py adj_contacts)
__device__ __forceinline__ void geom_pose(int g, const float* Qb, const float* Xb, float* gx, float* gq) {
  float t[3];
  qrot_rn(Qb, c_gpos + 3 * g, t);
  for (int i = 0; i < 3; i++) gx[i] = RADD(Xb[i], t[i]);
  qmul_rn(Qb, c_gquat + 4 * g, gq);
}

// world poses of the collision geoms (position, rotation quaternion) from the body poses
template <class S>
__device__ __forceinline__ void geom_poses(S& s, int lane) {
  for (int g = lane; g < NGEOMC; g += 32) {
    int b = c_gbody[g];
    geom_pose(g, s.Q[b], s.X[b], s.gx[g], s.gq[g]);
#if HAS_NATIVE_COLLISION
    if (c_gtype[g] >= 3) cm::world_aabb(g, s.gx[g], s.gq[g], s.gaabb[g]);
#endif
  }
}

// returns the centre distance nu = |p2 - p1| (the tape's min_nu: the contact normal's derivative is of order 1 / nu)
__device__ __forceinline__ float sphere_sphere(const float* p1, float r1, const float* p2, float r2, float* dist,
                                               float* pos, float* n) {
  float dir[3] = {RSUB(p2[0], p1[0]), RSUB(p2[1], p1[1]), RSUB(p2[2], p1[2])};
  float d = sqrtf(dot3_rn(dir, dir));
  if (d == 0.f) { n[0] = 1.f; n[1] = 0.f; n[2] = 0.f; } else { for (int k = 0; k < 3; k++) n[k] = dir[k] / d; }
  *dist = RSUB(d, RADD(r1, r2));
  float hp = RFMA(0.5f, *dist, r1);
  for (int k = 0; k < 3; k++) pos[k] = RFMA(n[k], hp, p1[k]);
  return d;
}

// The contact geometry of the mjlab model's primitives (MuJoCo Warp's collision_primitive_core), shared by the
// forward's collide_pair and the adjoint's replay (adjoint_contact.adj_contact_geom): the intermediates the VJP needs
// are outputs (the forward drops them).

// a sphere of radius r at c on the plane through p1 with normal n (plane-sphere; plane-capsule: an end of the
// capsule): the distance, and the contact point pos, midway in the penetration
__device__ __forceinline__ float plane_point(const float* p1, const float* n, const float* c, float r, float* pos) {
  float dist = RSUB(RSUB(dot3_rn(c, n), dot3_rn(p1, n)), r), hp = RFMA(0.5f, dist, r);
  for (int k = 0; k < 3; k++) pos[k] = RFMS(n[k], hp, c[k]);
  return dist;
}

// the frame of a plane-capsule contact: rows n, b = the capsule axis projected on the plane and normalized (make_frame's
// basis vector where the projection is shorter than 0.5), n x b; returns the projection's length
__device__ __forceinline__ float plane_capsule_frame(const float* n, const float* axis, float* fr) {
  float na = dot3_rn(n, axis), b[3];
  for (int k = 0; k < 3; k++) b[k] = RFMS(n[k], na, axis[k]);
  float bn = sqrtf(dot3_rn(b, b));
  if (bn > 0.f) for (int k = 0; k < 3; k++) b[k] /= bn;
  if (bn < 0.5f) {
    b[0] = 0.f; b[1] = 0.f; b[2] = 0.f;
    if (-0.5f < n[1] && n[1] < 0.5f) b[1] = 1.f; else b[2] = 1.f;
  }
  float c[3];
  cross_rn(n, b, c);
  for (int k = 0; k < 3; k++) { fr[k] = n[k]; fr[3 + k] = b[k]; fr[6 + k] = c[k]; }
  return bn;
}

// end e of a capsule (centre p, unit axis, half length hl): e = 0 the +axis end, 1 the -axis end
__device__ __forceinline__ void capsule_end(const float* p, const float* axis, float hl, int e, float* ce) {
  for (int k = 0; k < 3; k++) {
    float l = RMUL(axis[k], hl);
    ce[k] = e == 0 ? RADD(p[k], l) : RSUB(p[k], l);
  }
}

// sphere-capsule (MuJoCo Warp's sphere_capsule): the closest point pt = a + t ab of the capsule's segment (centre p2,
// unit axis, half length hl; a = p2 - hl axis, ab = 2 hl axis) to the sphere's centre p1, t = clamp(raw, 0, 1),
// raw = ap . ab / gam with ap = p1 - a and MuJoCo Warp's regularizer gam = ab . ab + 1e-6; returns raw
__device__ __forceinline__ float segment_point(const float* p1, const float* p2, const float* axis, float hl, float* a,
                                               float* ab, float* ap, float* gam, float* t, float* pt) {
  for (int k = 0; k < 3; k++) {
    a[k] = RFMS(axis[k], hl, p2[k]); ab[k] = RMUL(RMUL(2.f, axis[k]), hl); ap[k] = RSUB(p1[k], a[k]);
  }
  *gam = RADD(dot3_rn(ab, ab), 1e-6f);
  float raw = dot3_rn(ap, ab) / *gam;
  *t = fminf(fmaxf(raw, 0.f), 1.f);
  for (int k = 0; k < 3; k++) pt[k] = RFMA(*t, ab[k], a[k]);
  return raw;
}

// capsule-capsule (MuJoCo Warp's capsule_capsule): the segments p_k + x_k a_k, x_k in [-1, 1], with the scaled axes
// a_k = hl_k R_k e_z, dif = p1 - p2, the normal equations [ma mb; mb mc] x = (u, v) and det = ma mc - mb^2
struct CapCap { float a1[3], a2[3], dif[3], ma, mb, mc, u, v, det; };
__device__ __forceinline__ void capcap_setup(const float* p1, const float* q1, float hl1, const float* p2,
                                             const float* q2, float hl2, CapCap& c) {
  zaxis_rn(q1, c.a1);
  zaxis_rn(q2, c.a2);
  for (int k = 0; k < 3; k++) { c.a1[k] = RMUL(c.a1[k], hl1); c.a2[k] = RMUL(c.a2[k], hl2); c.dif[k] = RSUB(p1[k], p2[k]); }
  c.ma = dot3_rn(c.a1, c.a1); c.mb = -dot3_rn(c.a1, c.a2); c.mc = dot3_rn(c.a2, c.a2);
  c.u = -dot3_rn(c.a1, c.dif); c.v = dot3_rn(c.a2, c.dif);
  c.det = RFMA(c.ma, c.mc, -RMUL(c.mb, c.mb));
}
// the segment parameters where |det| >= 1e-15: the unconstrained minimum, then the clamp cascade. f1, f2: x1 / x2 free
// (not fixed at +-1; wp.clamp's gradient rule lo <= raw < hi for the clamp of x1 in the cascade's second stage), raw:
// x1 before that clamp (0 if it did not happen)
__device__ __forceinline__ void capcap_params(const CapCap& c, float* x1o, float* x2o, bool* f1, bool* f2, float* raw) {
  float x1 = RFMA(c.mc, c.u, -RMUL(c.mb, c.v)) / c.det, x2 = RFMA(c.ma, c.v, -RMUL(c.mb, c.u)) / c.det;
  *f1 = *f2 = true;
  *raw = 0.f;
  if (x1 > 1.f) { x1 = 1.f; x2 = RSUB(c.v, c.mb) / c.mc; *f1 = false; }
  else if (x1 < -1.f) { x1 = -1.f; x2 = RADD(c.v, c.mb) / c.mc; *f1 = false; }
  if (x2 > 1.f || x2 < -1.f) {
    bool hi = x2 > 1.f;
    x2 = hi ? 1.f : -1.f;
    *raw = hi ? RSUB(c.u, c.mb) / c.ma : RADD(c.u, c.mb) / c.ma;
    x1 = fminf(fmaxf(*raw, -1.f), 1.f);
    *f1 = *raw >= -1.f && *raw < 1.f;
    *f2 = false;
  }
  *x1o = x1; *x2o = x2;
}
// candidate e = 0..3 of the parallel case (|det| < 1e-15): an end of one segment (e = 0, 1: x1 = +1, -1; 2, 3: x2 =
// +1, -1) and the other's clamped closest parameter (raw before its clamp); f1, f2 as capcap_params
__device__ __forceinline__ void capcap_candidate(const CapCap& c, int e, float* x1, float* x2, bool* f1, bool* f2,
                                                 float* raw) {
  if (e < 2) {
    *x1 = e == 0 ? 1.f : -1.f;
    *raw = e == 0 ? RSUB(c.v, c.mb) / c.mc : RADD(c.v, c.mb) / c.mc;
    *x2 = fminf(fmaxf(*raw, -1.f), 1.f);
    *f1 = false;
    *f2 = *raw >= -1.f && *raw < 1.f;
  } else {
    *x2 = e == 2 ? 1.f : -1.f;
    *raw = e == 2 ? RSUB(c.u, c.mb) / c.ma : RADD(c.u, c.mb) / c.ma;
    *x1 = fminf(fmaxf(*raw, -1.f), 1.f);
    *f2 = false;
    *f1 = *raw >= -1.f && *raw < 1.f;
  }
}
// the sphere core on the closest points p_k + x_k a_k (radii r1, r2): returns nu, as sphere_sphere
__device__ __forceinline__ float capcap_core(const float* p1, const float* p2, const CapCap& c, float x1, float x2,
                                             float r1, float r2, float* dist, float* pos, float* n) {
  float v1[3], v2[3];
  for (int k = 0; k < 3; k++) { v1[k] = RFMA(c.a1[k], x1, p1[k]); v2[k] = RFMA(c.a2[k], x2, p2[k]); }
  return sphere_sphere(v1, r1, v2, r2, dist, pos, n);
}

// 3x3 helpers (row-major float[9]) for the box collisions, MuJoCo Warp's quat_to_mat convention
__device__ __forceinline__ void quat2mat(const float* q, float* R) {
  float q00 = q[0] * q[0], q01 = q[0] * q[1], q02 = q[0] * q[2], q03 = q[0] * q[3], q11 = q[1] * q[1];
  float q12 = q[1] * q[2], q13 = q[1] * q[3], q22 = q[2] * q[2], q23 = q[2] * q[3], q33 = q[3] * q[3];
  R[0] = q00 + q11 - q22 - q33; R[1] = 2.f * (q12 - q03); R[2] = 2.f * (q13 + q02);
  R[3] = 2.f * (q12 + q03); R[4] = q00 - q11 + q22 - q33; R[5] = 2.f * (q23 - q01);
  R[6] = 2.f * (q13 - q02); R[7] = 2.f * (q23 + q01); R[8] = q00 - q11 - q22 + q33;
}
__device__ __forceinline__ void mv3(const float* A, const float* v, float* o) {
  for (int i = 0; i < 3; i++) o[i] = A[3 * i] * v[0] + A[3 * i + 1] * v[1] + A[3 * i + 2] * v[2];
}
__device__ __forceinline__ void mtv3(const float* A, const float* v, float* o) {
  for (int i = 0; i < 3; i++) o[i] = A[i] * v[0] + A[3 + i] * v[1] + A[6 + i] * v[2];
}
__device__ __forceinline__ float safe_div(float x, float y) { return x / (y != 0.f ? y : 1e-15f); }

// sphere-box (MuJoCo Warp's sphere_box): distance, position, normal (from the sphere towards the box)
__device__ void sphere_box(const float* sp, float r, const float* bp, const float* R, const float* bs, float* dist,
                           float* pos, float* n) {
  float d[3] = {sp[0] - bp[0], sp[1] - bp[1], sp[2] - bp[2]}, c[3], cl[3], dir[3], pl[3], nl[3] = {0.f, 0.f, 0.f};
  mtv3(R, d, c);
  for (int k = 0; k < 3; k++) { cl[k] = fmaxf(-bs[k], fminf(bs[k], c[k])); dir[k] = cl[k] - c[k]; }
  float dd = sqrtf(dot3(dir, dir));
  if (dd <= 1e-15f) {
    float closest = 2.f * (bs[0] + bs[1] + bs[2]);
    int kk = 0;
    for (int i = 0; i < 6; i++) {
      float fd = fabsf(((i % 2) ? 1.f : -1.f) * bs[i / 2] - c[i / 2]);
      if (closest > fd) { closest = fd; kk = i; }
    }
    nl[kk / 2] = (kk % 2) ? -1.f : 1.f;
    for (int k = 0; k < 3; k++) pl[k] = c[k] + nl[k] * (r - closest) / 2.f;
    *dist = -closest - r;
  } else {
    for (int k = 0; k < 3; k++) { nl[k] = dir[k] / dd; pl[k] = 0.5f * (cl[k] + c[k] + nl[k] * r); }
    *dist = dd - r;
  }
  mv3(R, nl, n);
  float t[3];
  mv3(R, pl, t);
  for (int k = 0; k < 3; k++) pos[k] = bp[k] + t[k];
}

// capsule-box (MuJoCo Warp's capsule_box_witness + capsule_box): up to two sphere-box contacts along the segment
template <class S>
__device__ void capsule_box(S& s, int pair, const float* cp, const float* axisw, float cr, float hl, const float* bp,
                            const float* R, const float* bs) {
  float d[3] = {cp[0] - bp[0], cp[1] - bp[1], cp[2] - bp[2]}, pos[3], axis[3], ha[3];
  mtv3(R, d, pos);
  mtv3(R, axisw, axis);
  for (int k = 0; k < 3; k++) ha[k] = axis[k] * hl;
  int axisdir = int(ha[0] > 0.f) + 2 * int(ha[1] > 0.f) + 4 * int(ha[2] > 0.f);
  float bestdist = 1e32f, bestseg = -12.f, bestbox = 0.f;
  int cltype = -4, clface = -12, clcorner = -123, cledge = -123;
  for (int i = -1; i < 2; i += 2) {  // a capsule end closest to a face
    float tip[3], bpnt[3];
    int nout = 0, axout = -1;
    for (int j = 0; j < 3; j++) {
      tip[j] = pos[j] + float(i) * ha[j];
      bpnt[j] = tip[j];
      if (bpnt[j] < -bs[j]) { nout++; axout = j; bpnt[j] = -bs[j]; }
      else if (bpnt[j] > bs[j]) { nout++; axout = j; bpnt[j] = bs[j]; }
    }
    if (nout > 1) continue;
    float dist = 0.f;
    for (int j = 0; j < 3; j++) dist += (bpnt[j] - tip[j]) * (bpnt[j] - tip[j]);
    if (dist < bestdist) { bestdist = dist; bestseg = float(i); cltype = -2 + i; clface = axout; }
  }
  for (int i = 0; i < 8; i++)  // a box edge closest
    for (int j = 0; j < 3; j++) {
      if (i & (1 << j)) continue;
      float bpt[3] = {((i & 1) ? 1.f : -1.f) * bs[0], ((i & 2) ? 1.f : -1.f) * bs[1], ((i & 4) ? 1.f : -1.f) * bs[2]};
      bpt[j] = 0.f;
      float dif[3] = {bpt[0] - pos[0], bpt[1] - pos[1], bpt[2] - pos[2]};
      float u = -bs[j] * dif[j], v = dot3(ha, dif), ma = bs[j] * bs[j], mb = -bs[j] * ha[j], mc = hl * hl;
      float det = ma * mc - mb * mb;
      if (fabsf(det) < 1e-15f) continue;
      float idet = 1.f / det, x1 = (mc * u - mb * v) * idet, x2 = (ma * v - mb * u) * idet;
      int s1 = 1, s2 = 1;
      if (x1 > 1.f) { x1 = 1.f; s1 = 2; x2 = safe_div(v - mb, mc); }
      else if (x1 < -1.f) { x1 = -1.f; s1 = 0; x2 = safe_div(v + mb, mc); }
      bool over = x2 > 1.f;
      if (over || x2 < -1.f) {
        if (over) { x2 = 1.f; s2 = 2; x1 = safe_div(u - mb, ma); }
        else { x2 = -1.f; s2 = 0; x1 = safe_div(u + mb, ma); }
        if (x1 > 1.f) { x1 = 1.f; s1 = 2; } else if (x1 < -1.f) { x1 = -1.f; s1 = 0; }
      }
      for (int k = 0; k < 3; k++) dif[k] -= ha[k] * x2;
      dif[j] += bs[j] * x1;
      int ct = s1 * 3 + s2;
      float dsq = dot3(dif, dif);
      if (dsq < bestdist - 1e-15f) {
        bestdist = dsq; bestseg = x2; bestbox = x1;
        clcorner = i + (1 << j) * (ct / 6); cledge = j; cltype = ct;
      }
    }
  if (cltype == -4) return;
  float second = -4.f;
  if (cltype >= 0 && cltype / 3 != 1) {  // closest to a box corner
    int c1 = axisdir ^ clcorner;
    if (c1 != 0 && c1 != 7) {
      int mul, ax = 0, ax1 = 1, ax2 = 2;
      if (c1 == 1 || c1 == 2 || c1 == 4) mul = 1; else { mul = -1; c1 = 7 - c1; }
      if (c1 == 1) { ax = 0; ax1 = 1; ax2 = 2; } else if (c1 == 2) { ax = 1; ax1 = 2; ax2 = 0; }
      else if (c1 == 4) { ax = 2; ax1 = 0; ax2 = 1; }
      if (axis[ax] * axis[ax] > 0.5f) {
        float m = 2.f * safe_div(bs[ax], fabsf(ha[ax]));
        second = fminf(1.f - float(mul) * bestseg, m);
      } else {
        float m = 2.f * fminf(safe_div(bs[ax1], fabsf(ha[ax1])), safe_div(bs[ax2], fabsf(ha[ax2])));
        second = -fminf(1.f + float(mul) * bestseg, m);
      }
      second *= float(mul);
    }
  } else if (cltype >= 0 && cltype / 3 == 1) {  // on a box edge
    int c1 = (axisdir ^ clcorner) & (7 - (1 << cledge));
    if (c1 == 1 || c1 == 2 || c1 == 4) {
      int ax1 = 1, ax2 = 2, ax = cledge, mul;
      if (cledge == 0) { ax1 = 1; ax2 = 2; }
      if (cledge == 1) { ax1 = 2; ax2 = 0; }
      if (cledge == 2) { ax1 = 0; ax2 = 1; }
      if (fabsf(axis[ax1]) > fabsf(axis[ax2])) ax1 = ax2;
      ax2 = 3 - ax - ax1;
      if (c1 & (1 << ax2)) { mul = 1; second = 1.f - bestseg; } else { mul = -1; second = 1.f + bestseg; }
      float e1 = 2.f * safe_div(bs[ax2], fabsf(ha[ax2]));
      second = fminf(e1, second);
      float e2 = (((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0)) ? 1.f - bestbox : 1.f + bestbox;
      e1 = bs[ax] * safe_div(e2, fabsf(ha[ax]));
      second = fminf(e1, second);
      second *= float(mul);
    }
  } else if (clface != -1) {  // a capsule end closest to a face
    int mul = cltype == -3 ? 1 : -1;
    second = 2.f;
    float t1[3];
    for (int k = 0; k < 3; k++) t1[k] = pos[k] - ha[k] * float(mul);
    for (int i = 0; i < 3; i++) {
      if (i == clface) continue;
      float har = safe_div(float(mul), ha[i]);
      float e1 = (bs[i] - t1[i]) * har;
      if (0.f < e1 && e1 < second) second = e1;
      e1 = (-bs[i] - t1[i]) * har;
      if (0.f < e1 && e1 < second) second = e1;
    }
    second *= float(mul);
  }
  for (int e = 0; e < 2; e++) {
    if (e == 1 && !(second > -3.f)) break;
    float seg = e == 0 ? bestseg : bestseg + second, ce[3], dist, pc[3], n[3], fr[9];
    for (int k = 0; k < 3; k++) ce[k] = bp[k];
    float lc[3], t[3];
    for (int k = 0; k < 3; k++) lc[k] = pos[k] + ha[k] * seg;
    mv3(R, lc, t);
    for (int k = 0; k < 3; k++) ce[k] += t[k];
    sphere_box(ce, cr, bp, R, bs, &dist, pc, n);
    if (dist < 0.f) { make_frame(n, fr); add_contact(s, pair, e, dist, pc, fr); }
  }
}

template <class S>
__device__ DK_INL_COLLIDE void collide_pair(S& s, int pair) {
  int g1 = PG1(pair), g2 = PG2(pair), t1 = c_gtype[g1], t2 = c_gtype[g2];
  float p1[3], q1[4], p2[3], q2[4];
  for (int k = 0; k < 3; k++) { p1[k] = s.gx[g1][k]; p2[k] = s.gx[g2][k]; }
  for (int k = 0; k < 4; k++) { q1[k] = s.gq[g1][k]; q2[k] = s.gq[g2][k]; }
  const float* sz1 = c_gsize + 3 * g1;
  const float* sz2 = c_gsize + 3 * g2;
  if (t1 == 0) {  // plane (the floor): normal = its z axis
    float n[3];
    zaxis_rn(q1, n);
    float dif[3] = {RSUB(p2[0], p1[0]), RSUB(p2[1], p1[1]), RSUB(p2[2], p1[2])};
    if (dot3_rn(dif, n) > c_grbound[g2]) return;  // the other geom's bounding sphere is above the plane
    if (t2 == 1) {
      float pos[3], fr[9];
      float dist = plane_point(p1, n, p2, sz2[0], pos);
      make_frame(n, fr);
      if (dist < 0.f) add_contact(s, pair, 0, dist, pos, fr);
    } else if (t2 == 2) {
      float axis[3], fr[9];
      zaxis_rn(q2, axis);
      plane_capsule_frame(n, axis, fr);
      for (int e = 0; e < 2; e++) {
        float ce[3], pos[3];
        capsule_end(p2, axis, sz2[1], e, ce);
        float dist = plane_point(p1, n, ce, sz2[0], pos);
        if (dist < 0.f) add_contact(s, pair, e, dist, pos, fr);
      }
    }
#if HAS_BOX
    else if (t2 == 3) {
      float fr[9], cd = dot3(p2, n) - dot3(p1, n);
      make_frame(n, fr);
      for (int i = 0; i < 8; i++) {
        float cl[3] = {(i & 1) ? sz2[0] : -sz2[0], (i & 2) ? sz2[1] : -sz2[1], (i & 4) ? sz2[2] : -sz2[2]}, cw[3];
        qrot(q2, cl, cw);
        float dist = cd + dot3(n, cw), pos[3];
        for (int k = 0; k < 3; k++) pos[k] = cw[k] + p2[k] - 0.5f * n[k] * dist;
        if (dist < 0.f) add_contact(s, pair, i, dist, pos, fr);
      }
    }
#endif
    return;
  }
  {  // bounding spheres (geom_rbound) do not overlap: no contact at zero margin
    float dif[3] = {RSUB(p2[0], p1[0]), RSUB(p2[1], p1[1]), RSUB(p2[2], p1[2])}, rb = RADD(c_grbound[g1], c_grbound[g2]);
    if (dot3_rn(dif, dif) > RMUL(rb, rb)) return;
  }
  if (t1 == 1 && t2 == 1) {  // sphere-sphere
    float dist, pos[3], n[3];
    float nu = sphere_sphere(p1, sz1[0], p2, sz2[0], &dist, pos, n);
    if (dist < 0.f) add_sphere_contact(s, pair, 0, nu, dist, pos, n);
    return;
  }
  if (t1 == 1 && t2 == 2) {  // sphere-capsule (MuJoCo Warp's sphere_capsule: closest point of the segment)
    float axis[3], a[3], ab[3], ap[3], gam, t, pt[3], dist, pos[3], n[3];
    zaxis_rn(q2, axis);
    segment_point(p1, p2, axis, sz2[1], a, ab, ap, &gam, &t, pt);
    float nu = sphere_sphere(p1, sz1[0], pt, sz2[0], &dist, pos, n);
    if (dist < 0.f) add_sphere_contact(s, pair, 0, nu, dist, pos, n);
    return;
  }
  if (t1 == 2 && t2 == 2) {  // capsule-capsule
    CapCap cc;
    capcap_setup(p1, q1, sz1[1], p2, q2, sz2[1], cc);
    float dist, pos[3], n[3], x1, x2, raw;
    bool f1, f2;
    if (fabsf(cc.det) >= 1e-15f) {
      capcap_params(cc, &x1, &x2, &f1, &f2, &raw);
      float nu = capcap_core(p1, p2, cc, x1, x2, sz1[0], sz2[0], &dist, pos, n);
      if (dist < 0.f) add_sphere_contact(s, pair, 0, nu, dist, pos, n);
    } else {
      int cnt = 0;
      for (int e = 0; e < 4 && cnt < 2; e++) {
        capcap_candidate(cc, e, &x1, &x2, &f1, &f2, &raw);
        float nu = capcap_core(p1, p2, cc, x1, x2, sz1[0], sz2[0], &dist, pos, n);
        if (dist <= 0.f) {
          cnt++;
          if (dist < 0.f) add_sphere_contact(s, pair, e, nu, dist, pos, n);
        }
      }
    }
    return;
  }
#if HAS_BOX
  if (t1 == 2 && t2 == 3) {  // capsule-box
    float a1[3], R2[9];
    zaxis_rn(q1, a1);
    quat2mat(q2, R2);
    capsule_box(s, pair, p1, a1, sz1[0], sz1[1], p2, R2, sz2);
    return;
  }
#endif
  atomicOr(&s.unsupported, 1);  // native mesh / box-box pairs use the cooperative candidate path
}

// x^p for the solimp power: the integer powers 1 and 2 (both G1 models use 2) without powf (x * x is the correctly
// rounded square, powf(x, 2) is within 1 ulp of it)
__device__ __forceinline__ float pow_imp(float x, float p) { return p == 2.f ? RMUL(x, x) : (p == 1.f ? x : powf(x, p)); }

// the impedance curve's raw (unclamped) value for 0 < x < 1 and a power p != 1, on the branch lo = (x <= mid): the
// forward's kbimp and the adjoint's kbimp_d (its clamp test) compute the same bits
__device__ __forceinline__ float imp_raw(float x, float dmin, float dmax, float mid, float p, bool lo) {
  return lo ? RFMA(pow_imp(x, p) / pow_imp(mid, p - 1.f), RSUB(dmax, dmin), dmin)
            : RFMA(RSUB(1.f, pow_imp(RSUB(1.f, x), p) / pow_imp(RSUB(1.f, mid), p - 1.f)), RSUB(dmax, dmin), dmin);
}

// impedance (k, b, imp) at pos_imp with refsafe (timeconst >= 2 h); explicit roundings: the adjoint's kbimp_d calls it
__device__ __forceinline__ void kbimp(const float* solref, const float* solimp, float pos_imp, float h, float* k,
                                      float* b, float* imp) {
  float tc = fmaxf(solref[0], RMUL(2.f, h)), dr = solref[1];
  float dmin = fminf(fmaxf(solimp[0], 1e-4f), 0.9999f), dmax = fminf(fmaxf(solimp[1], 1e-4f), 0.9999f);
  float width = fmaxf(1e-15f, solimp[2]), mid = fminf(fmaxf(solimp[3], 1e-4f), 0.9999f);
  const float power = SOLIMP_POW2 ? 2.f : fmaxf(1.f, solimp[4]);  // a constant: pow_imp folds to x * x and mid
  *k = 1.f / fmaxf(1e-15f, RMUL(RMUL(RMUL(RMUL(RMUL(dmax, dmax), tc), tc), dr), dr));
  *b = 2.f / fmaxf(1e-15f, RMUL(dmax, tc));
  float x = fabsf(pos_imp) / width;
  if (dmin == dmax || solimp[2] <= 1e-15f) *imp = RMUL(0.5f, RADD(dmin, dmax));
  else if (x <= 0.f) *imp = dmin;
  else if (x >= 1.f) *imp = dmax;
  else if (power == 1.f) *imp = RFMA(x, RSUB(dmax, dmin), dmin);
  else *imp = fminf(fmaxf(imp_raw(x, dmin, dmax, mid, power, x <= mid), dmin), dmax);
}

// the distances of hinge dof i at angle q to its lower and upper limit (the limit rows' test and pos: one IEEE
// subtraction each; the adjoint's adj_limits replays the test)
__device__ __forceinline__ void limit_dists(int i, float q, float* dmin, float* dmax) {
  *dmin = RSUB(q, c_jrange[2 * i]);
  *dmax = RSUB(c_jrange[2 * i + 1], q);
}

// the invweight of a contact of pair pr (friction mu): body 1's + body 2's, for condim 3 MuJoCo's pyramid scaling
__device__ __forceinline__ float contact_invw(int pr, float mu) {
  float iw = RADD(c_binvw[c_gbody[PG1(pr)]], c_binvw[c_gbody[PG2(pr)]]);
  if (PDIM(pr) == 3) iw = RMUL(RMUL(RMUL(RFMA(RMUL(mu, mu), iw, iw), 2.f), mu), mu) / c_sopt[0];
  return iw;
}

// a row's D = 1 / max(R, 1e-15), R = invw (1 - imp) / imp
__device__ __forceinline__ float efc_D(float iw, float imp) { return 1.f / fmaxf(RMUL(iw, RSUB(1.f, imp)) / imp, 1e-15f); }

template <class T> struct dk_is_ptr { static constexpr bool value = false; };
template <class T> struct dk_is_ptr<T*> { static constexpr bool value = true; };

// con_dot[c][d] = J_c,d v for every contact and frame direction (one lane per pair; v a shared NV vector). DK_JV_VEC's
// 128-bit loads only where con_J is a pointer (to struct Hand: SimP, SimN): in k_substep (struct Sim) they made ptxas
// contract other mul / add pairs of that kernel differently (its mass matrix), and k_substep must stay bitwise the
// split forward
template <class S>
__device__ void con_jv(S& s, const float* v, int lane) {
  int ncon = min(s.ncon, MAXCON);
  for (int q = lane; q < 3 * ncon; q += 32) {
    int c = q / 3, nd = s.con_nd[c];
    const float* J = s.con_J[c][q % 3];
    const unsigned char* dof = s.con_dof[c];
    float a0 = 0.f, a1 = 0.f;  // even / odd support entries (unrolled and predicated: the loads are independent)
#if DK_JV_VEC && KDOF == 16
    if constexpr (dk_is_ptr<decltype(s.con_J)>::value) {
      const uint4 dw = *reinterpret_cast<const uint4*>(dof);  // the KDOF support dofs, 4 per word
      const unsigned dws[4] = {dw.x, dw.y, dw.z, dw.w};
#pragma unroll
      for (int k4 = 0; k4 < KDOF; k4 += 4) {
        float4 j4 = make_float4(0.f, 0.f, 0.f, 0.f);
        if (k4 < nd) j4 = *reinterpret_cast<const float4*>(J + k4);
        const float jk[4] = {j4.x, j4.y, j4.z, j4.w};
#pragma unroll
        for (int t = 0; t < 4; t += 2) {
          const int k = k4 + t;
          if (k < nd) a0 += jk[t] * v[(dws[k4 >> 2] >> (8 * t)) & 255u];
          if (k + 1 < nd) a1 += jk[t + 1] * v[(dws[k4 >> 2] >> (8 * t + 8)) & 255u];
        }
      }
    } else
#endif
    {
#pragma unroll
      for (int k = 0; k < KDOF; k += 2) {
        if (k < nd) a0 += J[k] * v[dof[k]];
        if (k + 1 < nd) a1 += J[k + 1] * v[dof[k + 1]];
      }
    }
    s.con_dot[c][q % 3] = a0 + a1;
  }
  __syncwarp();
}

__device__ __forceinline__ float eq_jv(int e, const float* v) {
  return v[c_eqdof1[e]] - c_eqpoly[2 * e + 1] * v[c_eqdof2[e]];
}
__device__ __forceinline__ float eq_pos(int e, const float* q) {
  return RSUB(q[c_dofqadr[c_eqdof1[e]]],
              RFMA(c_eqpoly[2 * e + 1], q[c_dofqadr[c_eqdof2[e]]], c_eqpoly[2 * e]));
}
template <class S>
__device__ __forceinline__ float dof_row_coef(const S& s, int r, int i) {
  int a = s.efc_a[r];
  if (NEQ && s.efc_kind[r] == 4)
    return (i == c_eqdof1[a] ? 1.f : 0.f) - (i == c_eqdof2[a] ? c_eqpoly[2 * a + 1] : 0.f);
  return i == a ? (s.efc_kind[r] == 0 ? 1.f : (float)s.efc_b[r]) : 0.f;
}

// J v for every row (v a shared NV vector) into out[] (NEFC entries); contact rows from con_jv
template <class S>
__device__ DK_INL_ROWSJV void rows_jv(S& s, const float* v, float* out, int lane) {
  con_jv(s, v, lane);
  for (int r = lane; r < s.nefc; r += 32) {
    int kind = s.efc_kind[r], a = s.efc_a[r], b = s.efc_b[r];
    float val;
    if (kind == 0) val = v[a];
    else if (kind == 1) val = b * v[a];
    else if (kind == 4) val = eq_jv(a, v);
    else if (kind == 2) {
      float mu = s.con_mu[a], sg = (b & 1) ? -1.f : 1.f;
      val = s.con_dot[a][0] + sg * mu * s.con_dot[a][1 + (b >> 1)];
    } else val = s.con_dot[a][0];
    out[r] = val;
  }
  __syncwarp();
}

// The piece of row r's cost at x (0 inactive, 1 quadratic, 2 / 3 the linear frictionloss pieces), with the cost
// derivative s'(x) and the curvature (D on the quadratic piece, else 0); D the row's, F its frictionloss bound. The
// frictionloss rows are the first NFLOSS rows (kind 0), so without them (mjlab model) the test is compiled out
__device__ __forceinline__ int row_piece(int r, float x, float D, float F, float* ds, float* dd) {
  if (NFLOSS > 0 && r < NFLOSS) {
    float R = 1.f / D;
    if (x <= -R * F) { *ds = -F; *dd = 0.f; return 2; }
    if (x >= R * F) { *ds = F; *dd = 0.f; return 3; }
    *ds = D * x; *dd = D; return 1;
  }
  if (r >= NFLOSS && r < NFLOSS + NEQ) { *ds = D * x; *dd = D; return 1; }
  if (x < 0.f) { *ds = D * x; *dd = D; return 1; }
  *ds = 0.f; *dd = 0.f; return 0;
}
__device__ __forceinline__ float row_cost(int z, float x, float D, float F) {  // the row's cost s(x) on piece z
  if (NFLOSS > 0 && z >= 2) { float R = 1.f / D; return (z == 2 ? -F * x : F * x) - 0.5f * R * F * F; }
  return z == 1 ? 0.5f * D * x * x : 0.f;
}
template <class S>
__device__ __forceinline__ float row_floss(const S& s, int r) {  // frictionloss bound of row r (0 for the others)
  return NFLOSS > 0 && r < NFLOSS ? c_floss[s.efc_a[r]] : 0.f;
}

// row cost derivative s'(x), curvature (D on the quadratic zone, else 0) and cost s(x)
template <class S>
__device__ __forceinline__ void row_eval(const S& s, int r, float x, float* ds, float* dd, float* cost) {
  float D = s.efc_D[r], F = row_floss(s, r);
  *cost = row_cost(row_piece(r, x, D, F, ds, dd), x, D, F);
}

// J^T f into out (f per row): a contact's rows combined into one force per frame direction (pyramid rows
// n +- mu t1, n +- mu t2), then per dof the dof rows plus the contacts
template <class S>
__device__ __forceinline__ void con_force(const S& s, const float* f, int c, float* g) {  // contact c's, per direction
  const float* fr = f + s.con_row[c];
  int nr = s.con_nrow[c];
  float f0 = nr > 0 ? fr[0] : 0.f, f1 = nr > 1 ? fr[1] : 0.f, f2 = nr > 2 ? fr[2] : 0.f, f3 = nr > 3 ? fr[3] : 0.f;
  float mu = s.con_dim[c] == 3 ? s.con_mu[c] : 0.f;
  g[0] = f0 + f1 + f2 + f3; g[1] = mu * (f0 - f1); g[2] = mu * (f2 - f3);
}
template <class S>
__device__ void rows_jtf(S& s, const float* f, float* out, int lane) {
  int ncon = uniform_int(min(s.ncon, MAXCON), CON_COUNT_BITS);
#if DK_JTF_GATHER
  // per dof in registers (lane l: dofs l and l + 32): the dof rows, then the contacts in contact order (the sums of the
  // scatter below, bitwise). Lane i of a block of 32 contacts holds contact c0 + i's forces and support mask
  // (con_mask), shuffled to all lanes per contact; dof d is in contact c's support when bit d of the mask is set, at
  // support position k = the set bits below d. No __syncwarp() between the contacts and no shared memory
  float acc[2];
#pragma unroll
  for (int t = 0; t < 2; t++) {
    const int i = lane + 32 * t;
    acc[t] = 0.f;
    if (i < NV)
      for (int r = 0; r < s.nefc_dof; r++) {
        float coef = dof_row_coef(s, r, i);
        if (coef != 0.f) acc[t] += coef * f[r];
      }
  }
  SUBPROF(21);
  SUBPROF(22);
  const unsigned below = (1u << lane) - 1u;
  const auto CJ = s.con_J;  // (SimP / SimN: the pointer into Hand, loaded once)
  for (int c0 = 0; c0 < ncon; c0 += 32) {
    float g[3] = {0.f, 0.f, 0.f};
    unsigned glo = 0u, ghi = 0u;
    if (c0 + lane < ncon) {
      con_force(s, f, c0 + lane, g);
      const unsigned long long m = s.con_mask[c0 + lane];
      glo = (unsigned)m; ghi = (unsigned)(m >> 32);
    }
    const int n = min(32, ncon - c0);
#pragma unroll 1
    for (int i = 0; i < n; i++) {  // the dofs 0-31 (not unrolled: 2 / 4 times were 1.7% / 4.8% slower on V100)
      const unsigned lo = __shfl_sync(0xffffffffu, glo, i);
      const float f0 = __shfl_sync(0xffffffffu, g[0], i), f1 = __shfl_sync(0xffffffffu, g[1], i),
                  f2 = __shfl_sync(0xffffffffu, g[2], i);
      const float* J = CJ[c0 + i][0];
      if ((lo >> lane) & 1u) {
        const int k = __popc(lo & below);
        acc[0] += f0 * J[k] + f1 * J[KDOF + k] + f2 * J[2 * KDOF + k];
      }
    }
    // the dofs 32.. (lanes 0 ..): only the contacts with support there (few: a right wrist), in contact order. In the
    // loop above they cost every contact 3 predicated loads and 7 more instructions (ptxas if-converts a uniform
    // branch around them), and the gather was then 1-2% slower than the scatter on the RTX 3090
    if (NV > 32)
      for (unsigned hm = __ballot_sync(0xffffffffu, ghi != 0u); hm; hm &= hm - 1u) {
        const int i = __ffs(hm) - 1;
        const unsigned lo = __shfl_sync(0xffffffffu, glo, i), hi = __shfl_sync(0xffffffffu, ghi, i);
        const float f0 = __shfl_sync(0xffffffffu, g[0], i), f1 = __shfl_sync(0xffffffffu, g[1], i),
                    f2 = __shfl_sync(0xffffffffu, g[2], i);
        const float* J = CJ[c0 + i][0];
        if ((hi >> lane) & 1u) {
          const int k = __popc(lo) + __popc(hi & below);
          acc[1] += f0 * J[k] + f1 * J[KDOF + k] + f2 * J[2 * KDOF + k];
        }
      }
  }
#pragma unroll
  for (int t = 0; t < 2; t++)
    if (lane + 32 * t < NV) out[lane + 32 * t] = acc[t];
  __syncwarp();
#else
  for (int c = lane; c < ncon; c += 32) con_force(s, f, c, s.con_f[c]);
  __syncwarp();
  SUBPROF(21);
  for (int i = lane; i < NV; i += 32) {
    float acc = 0.f;
    for (int r = 0; r < s.nefc_dof; r++) {
      float coef = dof_row_coef(s, r, i);
      if (coef != 0.f) acc += coef * f[r];
    }
    out[i] = acc;
  }
  __syncwarp();
  SUBPROF(22);
  for (int c = 0; c < ncon; c++) {  // scatter each contact over its support (one dof per lane)
    if (lane < s.con_nd[c])
      out[s.con_dof[c][lane]] += s.con_f[c][0] * s.con_J[c][0][lane] + s.con_f[c][1] * s.con_J[c][1][lane]
                                 + s.con_f[c][2] * s.con_J[c][2][lane];
    __syncwarp();
  }
#endif
  SUBPROF(23);
}

// zone of row r at x (the piece of its cost): 0 inactive, 1 quadratic, 2 / 3 the linear frictionloss pieces
template <class S>
__device__ __forceinline__ int row_zone(const S& s, int r, float x) {
  float ds, dd;
  return row_piece(r, x, s.efc_D[r], row_floss(s, r), &ds, &dd);
}

// The tape helpers are not inlined: the forward without a tape then keeps its compact code (V100, a 4096-state corpus,
// 4096 worlds: tape off 20.49 ms against 20.60 ms with the helpers inlined, 20.41 ms without any tape code; tape on
// 20.84 / 20.88 ms)

// tape: the rows on their quadratic zone at the current jar (row_zone 1, the rows that put D into the Newton Hessian),
// as bits; lane j returns word first + j (0 past nefc).
template <class S>
__device__ __noinline__ unsigned zone_word(const S& s, int lane, int first) {
  unsigned z = 0u;
  const int nefc = uniform_int(s.nefc, ROW_COUNT_BITS);
  for (int r0 = 32 * first; r0 < min(nefc, 32 * (first + 32)); r0 += 32) {
    int r = r0 + lane;
    unsigned b = __ballot_sync(0xffffffffu, r < nefc && row_zone(s, r, s.efc_jar[r]) == 1);
    if (lane + first == r0 >> 5) z = b;
  }
  return z;
}

// tape: n words from shared memory into the record and zeros up to word end (the padding to the next field, so a record
// does not depend on what the buffer held before), lane-strided (each warp store is one coalesced 128-byte line when
// dst is aligned), streaming stores (__stcs: the adjoint reads the tape much later)
__device__ __noinline__ void tape_put(float* dst, const float* src, int n, int end, int lane) {
  for (int i = lane; i < end; i += 32) __stcs(dst + i, i < n ? src[i] : 0.f);
}

// constraint forces at the current jar, J^T f into tmp and grad = M qacc - qfrc_smooth - J^T f; returns |grad| (and
// with DK_GRAD_COST in .y the warp sum of extra, reduced together with it)
template <class S>
// (forces: compute the row forces into efc_jv; false: the line search's update has stored them there)
#if DK_GRAD_COST
__device__ DK_INL_GRAD float2 newton_grad(S& s, int lane, bool forces = true, float extra = 0.f) {
#else
__device__ DK_INL_GRAD float newton_grad(S& s, int lane, bool forces = true) {
#endif
  if (forces)
    for (int r = lane; r < s.nefc; r += 32) {
      float ds, dd, c;
      row_eval(s, r, s.efc_jar[r], &ds, &dd, &c);
      s.efc_jv[r] = -ds;  // the line search's J p is not needed any more
    }
  __syncwarp();
  SUBPROF(20);
  rows_jtf(s, s.efc_jv, s.tmp, lane);
  float gn = 0.f;
  for (int i = lane; i < NV; i += 32) { float g = s.Ma[i] - s.qsm[i] - s.tmp[i]; s.grad[i] = g; gn += g * g; }
  __syncwarp();
#if DK_GRAD_COST
  for (int o = 16; o > 0; o >>= 1) {  // warp_sum of both (the same sums, interleaved)
    gn += __shfl_xor_sync(0xffffffffu, gn, o);
    extra += __shfl_xor_sync(0xffffffffu, extra, o);
  }
  return make_float2(sqrtf(gn), extra);
#else
  return sqrtf(warp_sum(gn));
#endif
}

// out = A v for a symmetric shared NV x NV matrix stored as its packed lower triangle
// (a split of rows 32.. over the warp with shuffle sums, instead of their second trip, was 0.4% / 2.4% slower on the
// V100 / RTX 3090; the element as `j <= i ? A[ri + j] : A[TRI(j, 0) + i]`, meant as two predicated loads at immediate
// offsets, compiles to the same address select; a padded full copy of M does not fit SimN at 11 worlds per SM)
__device__ DK_INL_MATVEC void matvec(const float* A, const float* v, float* out, int lane) {
  for (int i = lane; i < NV; i += 32) {  // one trip count for all lanes: element (i, j) from row max(i, j)
    const int ri = TRI(i, 0);
    float acc = 0.f, acc2 = 0.f;  // even / odd columns
#pragma unroll
    for (int j = 0; j < NV; j++) {  // fully unrolled, no branches: TRI(j, 0) is a constant
      const float* e = A + (j <= i ? ri + j : TRI(j, 0) + i);
      if (j & 1) acc2 += *e * v[j]; else acc += *e * v[j];
    }
    out[i] = acc + acc2;
  }
  __syncwarp();
}

__device__ __forceinline__ float vdot(const float* a, const float* b, int lane) {
  float acc = 0.f;
  for (int i = lane; i < NV; i += 32) acc += a[i] * b[i];
  return warp_sum(acc);
}

// total cost at the current qacc (needs Ma and efc_jar)
template <class S>
__device__ DK_INL_COST float total_cost(S& s, int lane) {
  float g = 0.f;
  for (int i = lane; i < NV; i += 32) g += 0.5f * (s.Ma[i] - s.qsm[i]) * (s.qacc[i] - s.qacc0[i]);
  for (int r = lane; r < s.nefc; r += 32) {
    float ds, dd, c;
    row_eval(s, r, s.efc_jar[r], &ds, &dd, &c);
    g += c;
  }
  return warp_sum(g);
}

template <class S>
__device__ void refresh(S& s, int lane) {  // Ma, jar at the current qacc
  matvec(s.M, s.qacc, s.Ma, lane);
  SUBPROF(18);
  rows_jv(s, s.qacc, s.efc_jar, lane);
  SUBPROF(19);
  for (int r = lane; r < s.nefc; r += 32) s.efc_jar[r] -= s.efc_aref[r];
  __syncwarp();
}

// the free joint (lane 0): position, then the orientation rotated by h times the angular velocity. Not inlined: ptxas
// fuses some of its mul / add pairs depending on the surrounding code, and k_substep and k_post must agree bitwise
__device__ __noinline__ void integrate_root(float* qpos, const float* qvel, float h) {
  for (int k = 0; k < 3; k++) qpos[k] += h * qvel[k];
  float w[3] = {qvel[3], qvel[4], qvel[5]}, nw = sqrtf(dot3(w, w));
  if (nw > 0.f) {
    float ang = h * nw, sa, ca;
    sincosf(0.5f * ang, &sa, &ca);
    float dq[4] = {ca, sa * w[0] / nw, sa * w[1] / nw, sa * w[2] / nw};
    float q[4] = {qpos[3], qpos[4], qpos[5], qpos[6]}, o[4];
    qmul(q, dq, o);
    float on = rsqrtf(o[0] * o[0] + o[1] * o[1] + o[2] * o[2] + o[3] * o[3]);
    for (int k = 0; k < 4; k++) qpos[3 + k] = o[k] * on;
  }
}
"""

STEP_KERNEL = r"""
// a contact's support as dof bits (con_mask) from x = the body mask of its two chains' symmetric difference (bit b:
// body b on exactly one of them): the dofs whose body is in x
__device__ __forceinline__ unsigned long long support_mask(unsigned long long x) {
#if DOF_BODY_ROOT6
  return (unsigned long long)(x >> 2) << 6 | ((x & 2u) ? 0x3full : 0ull);
#else
  unsigned long long m = 0ull;
  for (int i = 0; i < NV; i++)
    if ((x >> c_dofbody[i]) & 1u) m |= 1ull << i;
  return m;
#endif
}
// the Newton Hessian into s.L, accumulated in s.H before its float rounding;
// world and group: tiled contact masks and groups by body
// (substep_post)
template <class S>
__device__ __forceinline__ void hessian(S& s, int ncon, const unsigned* world, const unsigned* group, int lane) {
  // H = M + J' D J on the quadratic rows (lower triangle into L), curvatures D from the rows' zones at jar. The pyramid
  // rows of a contact, u_k = n +- mu t1, n +- mu t2, add C_nn n n' + C_n1 (n t1' + t1 n') + C_11 t1 t1' + (same for t2)
  // = J' C J with C = [C_nn C_n1 C_n2; C_n1 C_11 0; C_n2 0 C_22] (only C_nn for condim 1) and J the frame rows
  float mv[(NTRI + 31) / 32];  // H = M: all loads first (entry e by lane e mod 32)
#pragma unroll
  for (int k = 0; k < (NTRI + 31) / 32; k++) if (lane + 32 * k < NTRI) mv[k] = s.M[lane + 32 * k];
  int ha[HSETS_STORAGE][3], hb[HSETS_STORAGE], hthr[HSETS_STORAGE][3];  // this lane's support pairs (ha, hb) of one column hb per set, the
#pragma unroll                                   // same for every contact, valid when nd > hthr (step.hess_columns)
  for (int k = 0; k < HSETS; k++)
#pragma unroll
    for (int t = 0; t < 3; t++) {
      unsigned e = c_hcol[(3 * k + t) * 32 + lane];
      ha[k][t] = e & 255; hb[k] = (e >> 8) & 255; hthr[k][t] = e >> 16;
    }
  unsigned act[CON_TILES] = {};  // contacts with a quadratic row, one mask per tile
  for (int c0 = 0; c0 < ncon; c0 += 32) {
    int c = c0 + lane;
    bool on = false;
    if (c < ncon) {
      int r0 = s.con_row[c], nr = s.con_nrow[c];
      float d[4];
#pragma unroll
      for (int k = 0; k < 4; k++) {  // curvatures; the loads read a valid row either way (con_row is set only when
        int r = nr > k ? r0 + k : 0;  // nr > 0), so there are no branches
        float jar = s.efc_jar[r], D = s.efc_D[r];
        d[k] = nr > k && jar < 0.f ? D : 0.f;
      }
      float d0 = d[0], d1 = d[1], d2 = d[2], d3 = d[3], mu = s.con_dim[c] == 3 ? s.con_mu[c] : 0.f;
      s.con_C[c][0] = d0 + d1 + d2 + d3;
      s.con_C[c][1] = mu * (d0 - d1); s.con_C[c][2] = mu * mu * (d0 + d1);
      s.con_C[c][3] = mu * (d2 - d3); s.con_C[c][4] = mu * mu * (d2 + d3);
      on = d0 + d1 + d2 + d3 != 0.f;  // C_nn is the sum of the curvatures
    }
    act[c0 >> 5] = __ballot_sync(0xffffffffu, on);
  }
  SUBPROF(27);
#pragma unroll
  for (int k = 0; k < (NTRI + 31) / 32; k++) if (lane + 32 * k < NTRI) s.H[lane + 32 * k] = mv[k];
  const int ndof = uniform_int(s.nefc_dof, 8);
#if NEQ > 0
  __syncwarp();
  for (int i = 0; i < NV; i++)
    for (int j = lane; j <= i; j += 32) {
      double value = s.H[TRI(i, j)];
      for (int r = 0; r < ndof; r++) {
        float ci = dof_row_coef(s, r, i), cj = dof_row_coef(s, r, j);
        if (ci == 0.f || cj == 0.f) continue;
        float ds, dd, cost;
        row_eval(s, r, s.efc_jar[r], &ds, &dd, &cost);
        value += (double)ci * cj * dd;
      }
      s.H[TRI(i, j)] = value;
    }
#else
#if DK_HESS_DOFPAR
  if (NFLOSS == 0) {  // limit rows only: one per dof, distinct diagonal entries, each added by its row's lane
    __syncwarp();     // (after the copy)
    for (int r = lane; r < ndof; r += 32) {
      float ds, dd, c;
      row_eval(s, r, s.efc_jar[r], &ds, &dd, &c);
      s.H[TRI(s.efc_a[r], s.efc_a[r])] += dd;
    }
  } else
#endif
  for (int r0 = 0; r0 < ndof; r0 += 32) {  // dof rows on the diagonal: frictionloss (quadratic zone) and limits, each
    int r = r0 + lane, e = 0;              // added in row order by the lane that copied its entry
    float dd = 0.f;
    if (r < ndof) {
      float ds, c;
      row_eval(s, r, s.efc_jar[r], &ds, &dd, &c);
      e = TRI(s.efc_a[r], s.efc_a[r]);
    }
    for (int k = 0; k < min(32, ndof - r0); k++) {
      int ek = __shfl_sync(0xffffffffu, e, k);
      float dk = __shfl_sync(0xffffffffu, dd, k);
      if ((ek & 31) == lane) s.H[ek] += dk;
    }
  }
#endif
  __syncwarp();
  SUBPROF(28);
  // the contacts on the lower triangles of their supports (dofs ascending: dof[a] >= dof[b]): first the world
  // contacts, a group (one body, one support) at a time, then each contact between two bodies. Per lane and set: the
  // pairs (a, b) of one column b, H_ab = J_a' G_b with G_b = C J_b once per contact, summed over the group before the
  // update of H
  for (int pass = 0; pass < 2; pass++)
  for (int tile = 0; tile < CON_TILES; tile++) {
    bool w = pass == 0;
    unsigned todo = act[tile] & (w ? world[tile] : ~world[tile]);
    for (; todo;) {
      int c0 = 32 * tile, c = __ffs(todo) - 1;
      unsigned gw = __ballot_sync(0xffffffffu, (group[tile] >> c) & 1u), gm = w ? todo & gw : 1u << c;
      todo &= ~gm;
      if (!w) __syncwarp();  // between two bodies: fenced
      int nd = s.con_nd[c0 + c];
      const unsigned char* dof = s.con_dof[c0 + c];
      constexpr int hlo[HSETS_STORAGE] = HSET_LO;
#pragma unroll
      for (int k = 0; k < HSETS; k++) {
        if (nd <= hlo[k]) break;  // no rows of this set in the support
        int db = dof[hb[k]], he[3];
        double hv[3], lv[3];
#pragma unroll
        for (int t = 0; t < 3; t++) {  // the entries of one support are distinct: loads first, stores after the sum
          int da = dof[ha[k][t]];
          he[t] = da * (da + 1) / 2 + db;
          hv[t] = 0.f;
          if (nd > hthr[k][t]) lv[t] = s.H[he[t]];
        }
        unsigned mm = gm;
        int in = __ffs(mm) - 1;  // the next contact, found one iteration ahead
        mm &= mm - 1u;
        for (int i = in; i >= 0; i = in) {
          in = __ffs(mm) - 1;
          mm &= mm - 1u;
          int cc = c0 + i, b = hb[k];
          const float *C = s.con_C[cc], *Jn = s.con_J[cc][0], *Ja = s.con_J[cc][1], *Jb = s.con_J[cc][2];
          float j0 = Jn[b], j1 = Ja[b], j2 = Jb[b];  // G_b (condim 1: C_nn only, the rest adds exact zeros)
          double g0 = (double)C[0] * j0 + (double)C[1] * j1 + (double)C[3] * j2;
          double g1 = (double)C[1] * j0 + (double)C[2] * j1, g2 = (double)C[3] * j0 + (double)C[4] * j2;
#pragma unroll
          for (int t = 0; t < 3; t++) {
            int a = ha[k][t];
            hv[t] = fma((double)Jb[a], g2, fma((double)Ja[a], g1, fma((double)Jn[a], g0, hv[t])));
          }
        }
#pragma unroll
        for (int t = 0; t < 3; t++) if (nd > hthr[k][t]) s.H[he[t]] = lv[t] + hv[t];
      }
    }
  }
  // Support entries can belong to different lanes between contacts. Publish
  // their complete sums before the packed copy consumed by Cholesky.
  __syncwarp();
#pragma unroll
  for (int e = lane; e < NTRI; e += 32) s.L[e] = (float)s.H[e];
}

// The two-root scene has block-diagonal smooth dynamics and one coupled constraint solve. Body work is tiled over
// the warp; every depth finishes before either tile reads the next depth.
template <class S>
__device__ void coupled_smooth(S& s, int lane) {
  for (int L = 1; L <= MAXD; L++) {
    for (int b = lane; b < NB; b += 32)
      if (c_depth[b] == L) {
        int p = c_parent[b];
        float Qp[4], Xp[3], Wp[3], Vp[3], ALp[3], Ap[3], Qb[4], Xb[3], Wb[3], Vb[3], ALb[3], Ab[3];
        for (int k = 0; k < 4; k++) Qp[k] = s.Q[p][k];
        for (int k = 0; k < 3; k++) {
          Xp[k] = s.X[p][k]; Wp[k] = s.W[p][k]; Vp[k] = s.V[p][k]; ALp[k] = s.AL[p][k]; Ap[k] = s.A[p][k];
        }
        fwd_body(b, Qp, Xp, Wp, Vp, ALp, Ap, s.qpos, s.qvel, s.zero, Qb, Xb, Wb, Vb, ALb, Ab);
        for (int k = 0; k < 4; k++) s.Q[b][k] = Qb[k];
        for (int k = 0; k < 3; k++) {
          s.X[b][k] = Xb[k]; s.W[b][k] = Wb[k]; s.V[b][k] = Vb[k]; s.AL[b][k] = ALb[k]; s.A[b][k] = Ab[k];
        }
      }
    __syncwarp();
  }
  SUBPROF(14);
  for (int b = lane; b < NB; b += 32) {
    if (!b) continue;
    int root = c_root[b];
    float ip[3];
    body_ipos(b, s.com, ip);
    body_force(b, s.Q[b], s.W[b], s.AL[b], s.A[b], ip, s.F[b], s.N[b]);
    body_inertia(b, s.Q[b], s.X[b], s.X[root], ip, s.J[b]);
    if (!c_isfree[b]) {
      float w[3], v[3];
      dof_motion(b, 0, s.Q[b], s.X[b], s.X[root], w, v);
      for (int k = 0; k < 3; k++) { s.W[b][k] = w[k]; s.V[b][k] = v[k]; }
    }
  }
  __syncwarp();
  SUBPROF(15);
  for (int L = MAXD - 1; L >= 1; L--) {
    for (int b = lane; b < NB; b += 32)
      if (c_depth[b] == L)
        for (int j = 0; j < c_nchild[b]; j++) {
          int c = c_child[MAX_CHILD * b + j];
          accumulate(s.X[b], s.X[c], s.F[c], s.N[c], s.F[b], s.N[b]);
          for (int k = 0; k < 10; k++) s.J[b][k] += s.J[c][k];
        }
    __syncwarp();
  }
  SUBPROF(16);
  for (int b = lane; b < NB; b += 32) {
    if (!b) continue;
    torques(b, s.Q[b], s.F[b], s.N[b], s.qsm);
    const int root = c_root[b], droot = c_dadr[root], d = c_dadr[b];
    float n[3], f[3], sw[3], sv[3], kw[3], kv[3];
    if (c_isfree[b]) {
      for (int j = 0; j < 6; j++) {
        dof_motion(b, j, s.Q[b], s.X[b], s.X[b], sw, sv);
        apply_inertia(s.J[b], sw, sv, n, f);
        for (int i = 0; i <= j; i++) {
          dof_motion(b, i, s.Q[b], s.X[b], s.X[b], kw, kv);
          s.M[TRI(d + j, d + i)] = dot3(kw, n) + dot3(kv, f) + (i == j ? c_armature[d + j] : 0.f);
        }
      }
    } else {
      apply_inertia(s.J[b], s.W[b], s.V[b], n, f);
      s.M[TRI(d, d)] = dot3(s.W[b], n) + dot3(s.V[b], f) + c_armature[d];
      for (int j = 0; j < c_nanc[b]; j++) {
        int a = c_anc[12 * b + j];
        s.M[TRI(d, c_dadr[a])] = dot3(s.W[a], n) + dot3(s.V[a], f);
      }
      for (int j = 0; j < 6; j++) {
        dof_motion(root, j, s.Q[root], s.X[root], s.X[root], kw, kv);
        s.M[TRI(d, droot + j)] = dot3(kw, n) + dot3(kv, f);
      }
    }
  }
  __syncwarp();
}

// one substep for the world in s (qpos, qvel, qacc warm start, com, fric set); ctrl: this world's NU setpoints.
// Two halves (substep_pre: smooth dynamics, collision, Jacobians, rows; substep_post: Newton, implicitfast) so that they can
// also run as separate kernels with the state in global memory (engine_step_check.py --split).
// rec: this world's tape record of the substep (tape.py), or null (no tape); the same
// pointer goes to both halves. The record is written at five points: the input state (here), the actuator clamp mask,
// the contact table (end of substep_pre), the final zones with their Hessian factor L_H, and after the implicitfast
// solve L_Q, a*, a_adv and the header (substep_post). The forward results do not depend on rec.
template <class S>
__device__ void substep_pre(S& s, const float* ctrl, int lane, float* rec, const float* qfrc = nullptr,
                            float kp_scale = 1.f) {
  const float h = c_misc[0];
  int b = lane, d = (b < NB) ? c_depth[b] : -1, p = (b < NB) ? c_parent[b] : 0;
  if (rec) {  // tape: the substep's input state
    tape_put(rec + OFF_QPOS, s.qpos, NQ, NQ, lane);
    tape_put(rec + OFF_QVEL, s.qvel, NV, NV, lane);
  }
  if (lane == 0) {
    s.Q[0][0] = 1.f; s.Q[0][1] = s.Q[0][2] = s.Q[0][3] = 0.f;
    for (int k = 0; k < 3; k++) s.X[0][k] = s.W[0][k] = s.V[0][k] = s.AL[0][k] = s.A[0][k] = 0.f;
    s.ncon = 0; s.nefc = 0;  // s.unsupported accumulates over substeps; the caller clears it
    s.min_nu_bits = 0x7f800000;  // +inf
#if HAS_NATIVE_COLLISION
    s.feature_rec = rec ? rec + OFF_FEATURE : nullptr;
#endif
  }
  for (int i = lane; i < NV; i += 32) s.zero[i] = 0.f;
  for (int i = lane; i < NTRI; i += 32) s.M[i] = 0.f;
  PROF_START();
#if NR == 1
  // The tree passes below read what they need from shared memory into registers before they store anything: the
  // compiler cannot tell a body's arrays from its parent's or children's (same arrays, variable indices), so a load
  // after a store would wait for it (one shared-memory latency per element). The arithmetic is unchanged
  for (int L = 1; L <= MAXD; L++) {  // kinematics and velocities (zero acceleration: RNE gives the bias forces)
    if (d == L) {
      float Qp[4], Xp[3], Wp[3], Vp[3], ALp[3], Ap[3], Qb[4], Xb[3], Wb[3], Vb[3], ALb[3], Ab[3];
      for (int k = 0; k < 4; k++) Qp[k] = s.Q[p][k];
      for (int k = 0; k < 3; k++) {
        Xp[k] = s.X[p][k]; Wp[k] = s.W[p][k]; Vp[k] = s.V[p][k]; ALp[k] = s.AL[p][k]; Ap[k] = s.A[p][k];
      }
      fwd_body(b, Qp, Xp, Wp, Vp, ALp, Ap, s.qpos, s.qvel, s.zero, Qb, Xb, Wb, Vb, ALb, Ab);
      for (int k = 0; k < 4; k++) s.Q[b][k] = Qb[k];
      for (int k = 0; k < 3; k++) {
        s.X[b][k] = Xb[k]; s.W[b][k] = Wb[k]; s.V[b][k] = Vb[k]; s.AL[b][k] = ALb[k]; s.A[b][k] = Ab[k];
      }
    }
    __syncwarp();
  }
  SUBPROF(14);
  const float O[3] = {s.X[1][0], s.X[1][1], s.X[1][2]};  // the root's origin: the point of the mass matrix
  if (b >= 1 && b < NB) {
    float ip[3], Qb[4], Xb[3], Wb[3], ALb[3], Ab[3], Fb[3], Nb[3], Jb[10];
    for (int k = 0; k < 4; k++) Qb[k] = s.Q[b][k];
    for (int k = 0; k < 3; k++) { Xb[k] = s.X[b][k]; Wb[k] = s.W[b][k]; ALb[k] = s.AL[b][k]; Ab[k] = s.A[b][k]; }
    body_ipos(b, s.com, ip);
    body_force(b, Qb, Wb, ALb, Ab, ip, Fb, Nb);
    body_inertia(b, Qb, Xb, O, ip, Jb);
    for (int k = 0; k < 3; k++) { s.F[b][k] = Fb[k]; s.N[b][k] = Nb[k]; }
    for (int k = 0; k < 10; k++) s.J[b][k] = Jb[k];
    if (b >= 2) {  // the hinge's motion subspace about O into W, V (no longer needed), for the mass matrix and the
      float w[3], v[3];  // contact Jacobians
      dof_motion(b, 0, Qb, Xb, O, w, v);
      for (int k = 0; k < 3; k++) { s.W[b][k] = w[k]; s.V[b][k] = v[k]; }
    }
  }
  __syncwarp();
  SUBPROF(15);
  for (int L = MAXD - 1; L >= 1; L--) {
    if (d == L && c_nchild[b] > 0) {
      int nch = c_nchild[b];
      float Xb[3], Fb[3], Nb[3], Jb[10];
      for (int k = 0; k < 3; k++) { Xb[k] = s.X[b][k]; Fb[k] = s.F[b][k]; Nb[k] = s.N[b][k]; }
      for (int k = 0; k < 10; k++) Jb[k] = s.J[b][k];
      for (int i = 0; i < nch; i++) {
        int c = c_child[MAX_CHILD * b + i];
        float Xc[3], Fc[3], Nc[3];
        for (int k = 0; k < 3; k++) { Xc[k] = s.X[c][k]; Fc[k] = s.F[c][k]; Nc[k] = s.N[c][k]; }
        accumulate(Xb, Xc, Fc, Nc, Fb, Nb);
        for (int k = 0; k < 10; k++) Jb[k] += s.J[c][k];
      }
      for (int k = 0; k < 3; k++) { s.F[b][k] = Fb[k]; s.N[b][k] = Nb[k]; }
      for (int k = 0; k < 10; k++) s.J[b][k] = Jb[k];
    }
    __syncwarp();
  }
  SUBPROF(16);
  if (b >= 1 && b < NB) torques(b, s.Q[b], s.F[b], s.N[b], s.qsm);  // qsm holds the bias for now
  float sw[3], sv[3], kw[3], kv[3], n[3], f[3];
  const float q1[4] = {s.Q[1][0], s.Q[1][1], s.Q[1][2], s.Q[1][3]};  // the root's rotation (its dofs' subspace)
  if (b >= 2 && b < NB) {
    int j = c_dadr[b], na = c_nanc[b], ad[12];
    float Jb[10], aw[12][3], av[12][3];
    for (int k = 0; k < 10; k++) Jb[k] = s.J[b][k];
    for (int k = 0; k < 3; k++) { sw[k] = s.W[b][k]; sv[k] = s.V[b][k]; }
#pragma unroll
    for (int a = 0; a < 12; a++)  // the hinge ancestors' motion subspaces, all loaded before the first store
      if (a < na) {
        int k = c_anc[12 * b + a];
        ad[a] = c_dadr[k];
        for (int t = 0; t < 3; t++) { aw[a][t] = s.W[k][t]; av[a][t] = s.V[k][t]; }
      }
    apply_inertia(Jb, sw, sv, n, f);
    s.M[TRI(j, j)] = dot3(sw, n) + dot3(sv, f) + c_armature[j];
#pragma unroll
    for (int a = 0; a < 12; a++)
      if (a < na) s.M[TRI(j, ad[a])] = dot3(aw[a], n) + dot3(av[a], f);
    for (int t = 0; t < 6; t++) {
      dof_motion(1, t, q1, O, O, kw, kv);
      s.M[TRI(j, t)] = dot3(kw, n) + dot3(kv, f);
    }
  }
  if (b == 1) {
    float J1[10];
    for (int k = 0; k < 10; k++) J1[k] = s.J[1][k];
    for (int j = 0; j < 6; j++) {
      dof_motion(1, j, q1, O, O, sw, sv);
      apply_inertia(J1, sw, sv, n, f);
      for (int i = 0; i <= j; i++) {
        dof_motion(1, i, q1, O, O, kw, kv);
        s.M[TRI(j, i)] = dot3(kw, n) + dot3(kv, f) + (i == j ? c_armature[j] : 0.f);
      }
    }
  }
#else
  coupled_smooth(s, lane);
  const float O[3] = {s.X[1][0], s.X[1][1], s.X[1][2]};
#endif
  __syncwarp();
  SUBPROF(17);
  for (int i0 = 0; i0 < NV; i0 += 32) {  // qfrc_smooth = clamped actuator force - bias
    int i = i0 + lane;
    bool clamp = false;
    if (i < NV) {
      float act = 0.f;
      if (c_actid[i] >= 0) {
        act = kp_scale * c_kp[i] * (ctrl[c_actid[i]] - s.qpos[c_dofqadr[i]]) - c_kv[i] * s.qvel[i];
        act = fminf(fmaxf(act, -c_frclim[i]), c_frclim[i]);
        clamp = act <= -c_frclim[i] || act >= c_frclim[i];  // at the limit: MuJoCo Warp's actuator_lam test
      }
      s.qsm[i] = act - c_damping[i] * s.qvel[i] + (qfrc ? qfrc[i] : 0.f) - s.qsm[i];
    }
    if (rec) {  // tape: the clamp mask (dof bits), stored rather than recomputed: the adjoint's FMA contraction of the
      unsigned bal = __ballot_sync(0xffffffffu, clamp);  // force could differ and flip a dof at the limit
      if (lane == 0) __stcs(rec + OFF_CLAMP + (i0 >> 5), __uint_as_float(bal));
    }
  }
  PROF(0);
  for (int i = lane; i < NV; i += 32) s.qacc0[i] = s.qsm[i];
  __syncwarp();
  CHOL_SOLVE(s.M, s.L, s.qacc0, lane, s.cbuf);
  PROF(1);

  // collision
#if !CONTACT_DISABLED
  geom_poses(s, lane);
  __syncwarp();
  SUBPROF(26);
  {  // broad phase: the pairs whose bounding test passes (collide_pair's first test, here with a relative slack of 1e-5
     // so that it keeps every pair collide_pair keeps), in pair order into s.cand; then collide_pair on them, one per
     // lane (rather than 16 pairs per lane, most of them rejected and the narrow phases of the others diverging)
    static_assert(NPAIR < 65536, "candidate pairs as 16-bit indices");
    const unsigned lanes_below = (1u << lane) - 1u;
#if FLOOR_G >= 0
    float fn[3];  // the only plane's normal
    zaxis_rn(s.gq[FLOOR_G], fn);
#endif
    int ncand = 0;
#pragma unroll 4
    for (int k = 0; k < (NPAIR + 31) / 32; k++) {
      int pr = lane + 32 * k;
      bool keep = false;
      if (pr < NPAIR) {
        int g1 = PG1(pr), g2 = PG2(pr);
        float dif[3] = {RSUB(s.gx[g2][0], s.gx[g1][0]), RSUB(s.gx[g2][1], s.gx[g1][1]), RSUB(s.gx[g2][2], s.gx[g1][2])};
        if (PPLANE(pr)) {
#if FLOOR_G >= 0
          const float* n = fn;
#else
          float n[3];
          zaxis_rn(s.gq[g1], n);
#endif
          keep = dot3(dif, n) <= RFMA(c_grbound[g2], 1.f + 1e-5f, 1e-6f);
        } else {
          float rb = RADD(c_grbound[g1], c_grbound[g2]);
          keep = dot3(dif, dif) <= RFMA(RMUL(rb, rb), 1.f + 1e-5f, 1e-12f);
#if HAS_NATIVE_COLLISION
          // Elongated mesh/box bounds are much tighter than their spheres.
          // Reject only strict separation; native touching contacts are retained.
          if (keep && (c_gtype[g2] == 4 || (c_gtype[g1] == 3 && c_gtype[g2] == 3)))
            keep = cm::aabb_overlap(s.gaabb[g1], s.gaabb[g2]);
#endif
        }
      }
      unsigned bal = __ballot_sync(0xffffffffu, keep);
      if (keep) s.cand[ncand + __popc(bal & lanes_below)] = (unsigned short)pr;
      ncand += __popc(bal);
    }
    __syncwarp();
    SUBPROF(29);
    for (int i = lane; i < ncand; i += 32) {
      int pair = s.cand[i];
#if HAS_NATIVE_COLLISION
      int t1 = c_gtype[PG1(pair)], t2 = c_gtype[PG2(pair)];
      if (t2 == 4 || (t1 == 3 && t2 == 3)) continue;
#endif
      collide_pair(s, pair);
    }
#if HAS_NATIVE_COLLISION
    __syncwarp();
    // A world reuses one scratch buffer for its native candidates. No scratch
    // allocation scales with the number of eligible pairs.
    for (int i = 0; i < ncand; i++) {
      int pair = s.cand[i], g1 = PG1(pair), g2 = PG2(pair);
      int t1 = c_gtype[g1], t2 = c_gtype[g2];
      if (lane == 0 && (t2 == 4 || (t1 == 3 && t2 == 3))) {
        CMManifold manifold;
        if (t2 == 4) convex_manifold(g1, g2, &s.gx[0][0], &s.gq[0][0], s.cm_scratch, &manifold);
        else box_box_manifold(g1, g2, &s.gx[0][0], &s.gq[0][0], s.cm_scratch, &manifold);
        if (manifold.status) s.unsupported |= 32;
        else for (int j = 0; j < manifold.count; j++) {
          const CMContact& contact = manifold.contact[j];
          if (contact.dist < 0.f)
            add_contact(s, pair, j, contact.dist, contact.pos, contact.frame, &contact.feature);
        }
      }
      __syncwarp();
    }
#endif
  }
#endif
  PROF(2);
  if (lane == 0 && s.ncon > MAXCON) s.unsupported |= 4;
  int ncon = uniform_int(min(s.ncon, MAXCON), CON_COUNT_BITS);
  {  // contact Jacobians: frame rows of J(body2, pos) - J(body1, pos) on the dofs on exactly one chain, packed in
     // ascending order. Lane l keeps the motion subspaces (w, v about O) of dofs l and l + 32, and the two bodies'
     // ancestor masks of each contact tile (shuffled to all lanes per contact)
    static_assert(NV <= 64, "two dofs per lane");
    float dw[2][3], dv[2][3];
    int dbody[2];
    unsigned long long am1[CON_TILES], am2[CON_TILES];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      int i = lane + 32 * h;
      dbody[h] = i < NV ? c_dofbody[i] : 0;
      if (i < NV) dof_motion(dbody[h], c_doflocal[i], s.Q[dbody[h]], s.X[dbody[h]], O, dw[h], dv[h]);
    }
#pragma unroll
    for (int h = 0; h < CON_TILES; h++) {
      int c = lane + 32 * h;
      int pr = c < ncon ? s.con_pair[c] : 0;
      am1[h] = c_ancmask[c_gbody[PG1(pr)]];
      am2[h] = c_ancmask[c_gbody[PG2(pr)]];
    }
    for (int c = 0; c < ncon; c++) {
      unsigned long long m1 = __shfl_sync(0xffffffffu, am1[c >> 5], c & 31);
      unsigned long long m2 = __shfl_sync(0xffffffffu, am2[c >> 5], c & 31);
      float pos[3], fr[9];
      for (int t = 0; t < 3; t++) pos[t] = s.con_pos[c][t];
      for (int t = 0; t < 9; t++) fr[t] = s.con_frame[c][t];
      int base = 0;
#pragma unroll
      for (int h = 0; h < 2; h++) {
        int i = lane + 32 * h, bi = dbody[h];
        int in2 = (m2 >> bi) & 1, in1 = (m1 >> bi) & 1;
        bool on = i < NV && in1 != in2;
        unsigned bal = __ballot_sync(0xffffffffu, on);
        if (on) {
          int k = base + __popc(bal & ((1u << lane) - 1u));
          float r[3], u[3], jp[3];
          for (int t = 0; t < 3; t++) r[t] = pos[t] - O[t];
          cross(dw[h], r, u);
          float sg = in2 ? 1.f : -1.f;
          for (int t = 0; t < 3; t++) jp[t] = sg * (dv[h][t] + u[t]);
          for (int dd = 0; dd < 3; dd++) s.con_J[c][dd][k] = dot3(fr + 3 * dd, jp);
          s.con_dof[c][k] = (unsigned char)i;
        }
        base += __popc(bal);
      }
      if (lane == 0) s.con_nd[c] = base;
    }
#pragma unroll
    for (int h = 0; h < CON_TILES; h++)
      if (lane + 32 * h < ncon) s.con_mask[lane + 32 * h] = support_mask(am1[h] ^ am2[h]);  // (the ballots' bits)
  }
  PROF(3);

  // constraint rows: dof frictionloss, soft joint equalities, active hinge limits, then contacts
  // 4 pyramid rows (condim 3) or 1 normal row (condim 1); row indices from ballot prefix counts. First the dof rows'
  // (kind, dof, sign), one lane per dof
  const unsigned below = (1u << lane) - 1u;
  int ndof = 0;
  for (int i0 = 0; i0 < NV; i0 += 32) {
    int i = i0 + lane;
    bool on = i < NV && c_floss[i] > 0.f;
    unsigned bal = __ballot_sync(0xffffffffu, on);
    if (on) { int r = ndof + __popc(bal & below); s.efc_kind[r] = 0; s.efc_a[r] = i; s.efc_b[r] = 1; }
    ndof += __popc(bal);
  }
  for (int e = lane; e < NEQ; e += 32) {
    int r = ndof + e;
    s.efc_kind[r] = 4; s.efc_a[r] = e; s.efc_b[r] = 1;
  }
  ndof += NEQ;
  for (int i0 = 0; i0 < NV; i0 += 32) {
    int i = min(i0 + lane, NV - 1);  // clamped so that the loads need no branch
    float dmin, dmax;
    limit_dists(i, s.qpos[c_dofqadr[i]], &dmin, &dmax);
    bool on = i0 + lane < NV && !c_isfree[c_dofbody[i]] && c_jlim[i] != 0.f && fminf(dmin, dmax) < 0.f;
    unsigned bal = __ballot_sync(0xffffffffu, on);
    if (on) {
      int r = ndof + __popc(bal & below);
      s.efc_kind[r] = 1; s.efc_a[r] = i; s.efc_b[r] = dmin < dmax ? 1 : -1;
    }
    ndof += __popc(bal);
  }
  con_jv(s, s.qvel, lane);  // contact J qvel for the contact rows' velocities (its __syncwarp also publishes the above)
  // then one lane per job, a dof row (job j < ndof) or a contact (c = j - ndof): impedance, D and aref of its rows;
  // contact c's first row is ndof + c + 3 * (the number of condim-3 contacts before c)
  int ntri = 0;
  for (int j0 = 0; j0 < ndof + ncon; j0 += 32) {
    int j = j0 + lane, c = j - ndof;
    bool con = c >= 0 && c < ncon, tri = con && s.con_dim[c] == 3;
    unsigned b3 = __ballot_sync(0xffffffffu, tri);
    int r0 = ndof + c + 3 * (ntri + __popc(b3 & below));
    ntri += __popc(b3);
    if (j < ndof + ncon) {
      const float *sref, *simp;
      float iw, pos = 0.f, mu = 0.f;
      int i = 0;
      if (!con) {  // frictionloss, soft joint equality or limit
        i = s.efc_a[j];
        if (s.efc_kind[j] == 4) {
          iw = c_eqinvw[i]; pos = eq_pos(i, s.qpos);
          sref = c_eqsolref + 2 * i; simp = c_eqsolimp + 5 * i;
        } else {
          bool fl = s.efc_kind[j] == 0;
          float dmin, dmax;
          limit_dists(i, s.qpos[c_dofqadr[i]], &dmin, &dmax);
          iw = c_dinvw[i];
          pos = fl ? 0.f : fminf(dmin, dmax);
          sref = fl ? c_dsolref : c_jsolref; simp = fl ? c_dsolimp : c_jsolimp;
        }
      } else {
        int pr = s.con_pair[c];
        mu = s.con_mu[c];
        iw = contact_invw(pr, mu);
        pos = s.con_dist[c]; sref = PSOLREF(pr); simp = PSOLIMP(pr);
      }
      float k, bb, imp;
      kbimp(sref, simp, pos, h, &k, &bb, &imp);
      float D = efc_D(iw, imp);
      if (!con) {
        float vel = s.efc_kind[j] == 4 ? eq_jv(i, s.qvel) : s.efc_b[j] * s.qvel[i];
        s.efc_D[j] = D; s.efc_aref[j] = -k * imp * pos - bb * vel;
      } else {
        int nr = max(0, min(tri ? 4 : 1, NEFC - r0));  // rows beyond NEFC are dropped
        s.con_row[c] = r0; s.con_nrow[c] = nr;
#pragma unroll
        for (int t = 0; t < 4; t++) {
          if (t >= nr) break;
          float sg = (t & 1) ? -1.f : 1.f;
          float vel = tri ? s.con_dot[c][0] + sg * mu * s.con_dot[c][1 + (t >> 1)] : s.con_dot[c][0];
          s.efc_kind[r0 + t] = tri ? 2 : 3; s.efc_a[r0 + t] = c; s.efc_b[r0 + t] = t;
          s.efc_D[r0 + t] = D; s.efc_aref[r0 + t] = -k * imp * pos - bb * vel;
        }
      }
    }
  }
  if (lane == 0) {
    int need = ndof + ncon + 3 * ntri;
    if (need > NEFC) s.unsupported |= 16;  // constraint rows beyond capacity are dropped
    s.nefc = min(need, NEFC);
    s.nefc_dof = ndof;  // frictionloss, equality and limit rows; contact rows follow
  }
  PROF(4);
  if (rec)  // tape: the contact table in contact (= row) order, two 16-bit entries pair | slot << 9 per word
    for (int i = lane; i < (MAXCON + 1) / 2; i += 32) {
      unsigned e0 = 2 * i < ncon ? (unsigned)s.con_pair[2 * i] | (unsigned)s.con_slot[2 * i] << TAPE_PAIR_BITS : TAPE_CON_NONE;
      unsigned e1 = 2 * i + 1 < ncon ? (unsigned)s.con_pair[2 * i + 1] | (unsigned)s.con_slot[2 * i + 1] << TAPE_PAIR_BITS
                                     : TAPE_CON_NONE;
      __stcs(rec + OFF_CON + i, __uint_as_float(e0 | e1 << 16));
    }
#if HAS_NATIVE_COLLISION
  if (rec)
    for (int i = ncon * CM_FEATURE_WORDS + lane; i < MAXCON * CM_FEATURE_WORDS; i += 32)
      __stcs(rec + OFF_FEATURE + i, 0.f);
#endif
}

template <class S>
__device__ void substep_post(S& s, int iterations, int lane, int* niter_out, float* rec) {
  const float h = c_misc[0];
  int ncon = uniform_int(min(s.ncon, MAXCON), CON_COUNT_BITS);
  PROF_START();
  // Newton solver from the warm start (or qacc_smooth when that is cheaper)
  const float scale = 1.f / (c_sopt[1] * NV), tol = c_sopt[2];
  refresh(s, lane);
  float cost = total_cost(s, lane);
  // the warm start's qacc, Ma and jar kept in tmp, p and efc_jv (all free here; SimN's Mp shares tmp's storage) in
  // case it stays the start
  for (int i = lane; i < NV; i += 32) { s.tmp[i] = s.qacc[i]; s.qacc[i] = s.qacc0[i]; s.p[i] = s.Ma[i]; }
  for (int r = lane; r < s.nefc; r += 32) s.efc_jv[r] = s.efc_jar[r];
  __syncwarp();
  refresh(s, lane);
  float cost0 = total_cost(s, lane);
  if (UNIFORM(cost0 > cost)) {
    for (int i = lane; i < NV; i += 32) { s.qacc[i] = s.tmp[i]; s.Ma[i] = s.p[i]; }
    for (int r = lane; r < s.nefc; r += 32) s.efc_jar[r] = s.efc_jv[r];
    __syncwarp();
  } else cost = cost0;
  PROF(5);
  newton_grad(s, lane);
  PROF(6);
  // contacts against the world, for the Hessian assembly: the support of such a contact is the whole chain of the
  // other body, so the dofs two of them share are their common ancestors' dofs, at the same positions in both supports,
  // and an entry of L they share comes from the same support pair (a, b), which is always on the same lane: no
  // __syncwarp() between them. The world contacts on one body have the same support and are summed in registers before
  // one update of L (group: lane masks of the contacts on the same body in each tile). A contact between two
  // bodies (support: the dofs on exactly one chain) is fenced on both sides.
  static_assert(KDOF <= 32, "hess_columns fits 32 lanes");
  unsigned world[CON_TILES] = {}, group[CON_TILES] = {};
  for (int c0 = 0; c0 < ncon; c0 += 32) {
    int c = c0 + lane, pr = c < ncon ? s.con_pair[c] : 0, b1 = c_gbody[PG1(pr)], b2 = c_gbody[PG2(pr)];
    bool w = c < ncon && (b1 == 0 || b2 == 0);
    world[c0 >> 5] = __ballot_sync(0xffffffffu, w);
    group[c0 >> 5] = __match_any_sync(0xffffffffu, w ? b1 + b2 : -1 - lane);
  }
  // tape (rec): L_H must be the Hessian factor at the final zones. The last factorization was assembled at the jar
  // before the last step; when that step moved a row to another zone (chg, tested where jar is updated), or no step was
  // made, the loop runs once more without a step (refact): assembly and factorization at the final jar, so the tape
  // needs no second copy of the assembly code. Only the scratch s.L, s.p and s.con_C change, so the forward results are
  // the same with and without the tape
  int it = 0;
  bool conv = false, zchanged = rec && iterations <= 0, refact = zchanged, zsame = false;
  while (it < iterations || refact) {
#if NR == 2 && !CHOL_BCAST && !CHOL_COMPACT
    if (zsame && !refact) {  // H and its factor are unchanged; retain the original refined reciprocals
      for (int i = lane; i < NV; i += 32) s.p[i] = -s.grad[i];
      __syncwarp();
      chol_resolve_reg_cached(s.L, s.p, s.h_inv, lane);
    } else
#elif NEWTON_REUSE
    if (zsame && !refact) {  // no row changed its zone in the last step: H and its factor in s.L are the same
      for (int i = lane; i < NV; i += 32) s.p[i] = -s.grad[i];
      __syncwarp();
      chol_resolve(s.L, s.p, lane, s.cbuf);
    } else
#endif
    {
      hessian(s, ncon, world, group, lane);
      PROF(7);
      for (int i = lane; i < NV; i += 32) s.p[i] = -s.grad[i];
      __syncwarp();
#if NR == 2 && !CHOL_BCAST && !CHOL_COMPACT
      chol_solve_reg_save(s.L, s.L, s.p, lane, s.h_inv);
#else
      CHOL_SOLVE(s.L, s.L, s.p, lane, s.cbuf);
#endif
    }
    PROF(8);
    if (refact) break;  // the tape's re-factorization: no step
    // exact line search along p on the piecewise-quadratic cost: 1-D Newton, stopping when the step stays on its piece
    rows_jv(s, s.p, s.efc_jv, lane);
    SUBPROF(24);
    matvec(s.M, s.p, s.Mp, lane);
    SUBPROF(25);
#ifdef DK_PROFILE
    __syncwarp(); if (lane == 0) s.prof[12] += clock64() - s.prof_t;
#endif
    float a1 = 0.f, a2 = 0.f;
    for (int i = lane; i < NV; i += 32) { a1 += (s.Ma[i] - s.qsm[i]) * s.p[i]; a2 += s.p[i] * s.Mp[i]; }
    a1 = warp_sum(a1); a2 = warp_sum(a2);
    // the lane's rows r = lane + 32 t stay in registers for all steps (jar, J p, D, frictionloss bound; zeros past
    // nefc: inactive, no contribution), their pieces as 2-bit codes; a Newton step's test whether it kept every row on
    // its piece (then it landed on the exact minimizer) is done by the next pass over the rows
    constexpr int RT = (NEFC + 31) / 32;
    const int nefc = s.nefc;
    float rx[RT], rj[RT], rD[RT], rF[RT];
#pragma unroll
    for (int t = 0; t < RT; t++) {
      int r = lane + 32 * t;
      bool v = r < nefc;
      rx[t] = v ? s.efc_jar[r] : 0.f; rj[t] = v ? s.efc_jv[r] : 0.f; rD[t] = v ? s.efc_D[r] : 0.f;
      rF[t] = row_floss(s, r);
    }
    float alpha = 0.f, lo = 0.f, hi = 1e30f, d0 = 0.f;
    constexpr int ZWORDS = (RT + 15) / 16;
    unsigned zprev[ZWORDS] = {};
    bool newton = false;
    for (int k = 0; k <= 40; k++) {
      float fd = 0.f, fdd = 0.f;
      unsigned z[ZWORDS] = {};
#pragma unroll
      for (int t = 0; t < RT; t++) {
        if (32 * t >= nefc) break;
        float ds, dd, jv = rj[t];
        z[t >> 4] |= (unsigned)row_piece(lane + 32 * t, rx[t] + alpha * jv, rD[t], rF[t], &ds, &dd) << (2 * (t & 15));
        fd += ds * jv; fdd += dd * jv * jv;
      }
      bool same = newton;
#pragma unroll
      for (int w = 0; w < ZWORDS; w++) same &= z[w] == zprev[w];
      if (__all_sync(0xffffffffu, same)) break;  // the last Newton step kept every row on its piece
      if (k == 40) break;
#ifdef DK_PROFILE
      if (lane == 0) s.prof[13]++;
#endif
      fd = warp_sum(fd) + a1 + alpha * a2;
      fdd = warp_sum(fdd) + a2;
      if (k == 0) { d0 = fd; if (UNIFORM(d0 >= 0.f)) break; }
      if (UNIFORM(fabsf(fd) <= 1e-7f * fabsf(d0))) break;
      if (fd < 0.f) lo = alpha; else hi = alpha;
      float nx = alpha - fd / fdd;
      newton = nx > lo && nx < hi;
      if (!newton) nx = hi < 1e29f ? 0.5f * (lo + hi) : 2.f * fmaxf(alpha, 1e-6f);
#pragma unroll
      for (int w = 0; w < ZWORDS; w++) zprev[w] = z[w];
      alpha = nx;
    }
    // Rebuild the products from the updated float32 acceleration. Incremental
    // products can keep moving after qacc rounds, inventing cost improvement.
    for (int i = lane; i < NV; i += 32) s.qacc[i] += alpha * s.p[i];
    __syncwarp();
    refresh(s, lane);
    float g = 0.f;
    for (int i = lane; i < NV; i += 32)
      g += 0.5f * (s.Ma[i] - s.qsm[i]) * (s.qacc[i] - s.qacc0[i]);
    bool chg = false;  // a row of this lane changes its quadratic zone (ZONE_CHANGED: the Hessian factored above, or the
                       // tape's factor, is then out of date)
#pragma unroll
    for (int t = 0; t < RT; t++) {
      int r = lane + 32 * t;
      if (32 * t >= nefc) break;
      if (r < nefc) {
        float xn = s.efc_jar[r], ds, dd;
        int z0 = row_piece(r, rx[t], rD[t], rF[t], &ds, &dd), z1 = row_piece(r, xn, rD[t], rF[t], &ds, &dd);
        chg |= ZONE_CHANGED(z0, z1);
        g += row_cost(z1, xn, rD[t], rF[t]);
        s.efc_jv[r] = -ds;  // the line search's J p is not needed any more
      }
    }
    zsame = !__any_sync(0xffffffffu, chg);  // then the next H is this one
    PROF(9);
#if DK_GRAD_COST
    PROF(10);
    const float2 gc = newton_grad(s, lane, false, g);  // gradient at the new point (the stopping test and the next
    const float gn = gc.x, newcost = gc.y;             // step), the new cost
    float improvement = scale * (cost - newcost);
    cost = newcost;
#else
    float newcost = warp_sum(g);
    PROF(10);
    float improvement = scale * (cost - newcost);
    cost = newcost;
    float gn = newton_grad(s, lane, false);  // gradient at the new point: the stopping test and the next step
#endif
    PROF(6);
    it++;
    conv = UNIFORM(improvement < tol || scale * gn < tol);
    if (conv || it == iterations) {
      if (rec) {  // tape: did the last step change a zone?
        zchanged = __any_sync(0xffffffffu, chg);
#ifdef DK_TAPE_VERIFY
        for (int e = lane; e < NTRI; e += 32) s.Lv[e] = s.L[e];  // debug: always re-assemble, compare below
        refact = true;
#else
        refact = zchanged;
#endif
      }
      if (!refact) break;
    }
  }
  if (niter_out && lane == 0) *niter_out = it;
  if (rec) {  // tape: the final zones and their Hessian factor, before the implicitfast solve overwrites L
    for (int first = 0; first < (NEFC + 31) / 32; first += 32) {
      unsigned z = zone_word(s, lane, first);
      if (lane + first < (NEFC + 31) / 32) __stcs(rec + OFF_ZONE + first + lane, __uint_as_float(z));
    }
    tape_put(rec + OFF_LH, s.L, NTRI, OFF_LQ - OFF_LH, lane);
  }
#ifdef DK_TAPE_VERIFY
  bool vdiff = false;  // the factor of the last Newton assembly differs from the one at the final zones
  if (rec && it > 0) {
    bool d = false;
    for (int e = lane; e < NTRI; e += 32) d |= __float_as_uint(s.L[e]) != __float_as_uint(s.Lv[e]);
    vdiff = __any_sync(0xffffffffu, d);
  }
#endif

  // implicitfast: (M + h kv) qacc' = M qacc; then qvel, qpos
  for (int e = lane; e < NTRI; e += 32) s.L[e] = s.M[e];
  __syncwarp();
  for (int i = lane; i < NV; i += 32) {
    s.L[TRI(i, i)] += h * (c_kv[i] + c_damping[i]); s.tmp[i] = s.Ma[i];
  }
  __syncwarp();
  CHOL_SOLVE(s.L, s.L, s.tmp, lane, s.cbuf);
  if (rec) {  // tape: L_Q = chol(M + h kv), a* and a_adv (qvel' = qvel + h a_adv), header
    tape_put(rec + OFF_LQ, s.L, NTRI, REC_WORDS - OFF_LQ, lane);
    tape_put(rec + OFF_QACC, s.qacc, NV, NV, lane);
    tape_put(rec + OFF_QADV, s.tmp, NV, OFF_LH - OFF_QADV, lane);
    if (lane == 0) {
      float nu = __int_as_float(s.min_nu_bits);
      unsigned fl = (unsigned)(s.unsupported & 63) | (zchanged ? TAPE_REFACT : 0u) | (conv ? 0u : TAPE_NONCONV)
                    | (nu < TAPE_SINGULAR_NU ? TAPE_SINGULAR : 0u);
#ifdef DK_TAPE_VERIFY
      if (vdiff) fl |= TAPE_VERIFY_DIFF;
#endif
      unsigned hd = (unsigned)ncon | (unsigned)s.nefc << 16;
      unsigned counts = (unsigned)(s.nefc_dof - NFLOSS - NEQ) | (unsigned)min(it, 65535) << 16;
      __stcs(rec + OFF_HDR, __uint_as_float(hd));
      __stcs(rec + OFF_FLAGS, __uint_as_float(fl));
      __stcs(rec + OFF_MIN_NU, nu);
      __stcs(rec + OFF_COUNTS, __uint_as_float(counts));
    }
  }
  for (int i = lane; i < NV; i += 32) s.qvel[i] += h * s.tmp[i];
  __syncwarp();
  for (int r = lane; r < NR; r += 32) {
    int b = c_roots[r];
    integrate_root(s.qpos + c_qadr[b], s.qvel + c_dadr[b], h);
  }
  for (int i = lane; i < NV; i += 32)
    if (!c_isfree[c_dofbody[i]]) s.qpos[c_dofqadr[i]] += h * s.qvel[i];
  __syncwarp();
  PROF(11);
}

template <class S>
__device__ __forceinline__ void substep(S& s, const float* ctrl, int iterations, int lane, int* niter_out,
                                        float* rec, const float* qfrc = nullptr, float kp_scale = 1.f) {
  substep_pre(s, ctrl, lane, rec, qfrc, kp_scale);
  substep_post(s, iterations, lane, niter_out, rec);
}

// The split forward (host.py: k_pre, k_post, dk_forward_split): substep_pre and substep_post as two kernels, each
// with its own state (SimP, SimN: only what that half uses, so more worlds fit an SM, and each kernel's code is about
// half of k_substep's). Their members have struct Sim's names, so the device code above runs unchanged on all three;
// arrays that live in global memory are pointers there (struct Hand, one per world: what substep_pre hands to
// substep_post).
// substep_post runs on SimNT<SPLIT_NCON, SPLIT_NEFC> in shared memory when the world's contacts and rows fit,
// else on a full-capacity SimN in Hand (global memory, the same code).
#ifndef SPLIT_NCON
#if NR == 2 && !defined(DK_TAPE_VERIFY)  // its extra Hessian must also fit the default 48 KiB shared-memory limit
#define SPLIT_NCON (MAXCON < 96 ? MAXCON : 96)  // k_post's shared-memory capacity: contacts
#else
#define SPLIT_NCON (MAXCON < 32 ? MAXCON : 32)  // k_post's shared-memory capacity: contacts
#endif
#endif
#ifndef SPLIT_NEFC
#if NR == 2 && !defined(DK_TAPE_VERIFY)
#define SPLIT_NEFC (NEFC < NFLOSS + NEQ + NV + 4 * SPLIT_NCON ? NEFC : NFLOSS + NEQ + NV + 4 * SPLIT_NCON)
#else
#define SPLIT_NEFC (NEFC < 128 ? NEFC : 128)  // and constraint rows
#endif
#endif
template <int NC, int NROW>
struct SimNT {  // substep_post: Newton, implicitfast; capacity NC contacts, NR rows. Members whose lifetimes do not
  float M[NTRI], L[NTRI];  // overlap share storage (unions), the read-only ones the loop seldom reads stay in Hand
  union { double H[NTRI]; float h_inv[NV]; };  // after assembly, keep the factorization pivots
  float qacc[NV], Ma[NV];
  union { float grad[NV], p[NV]; };  // grad is last read by p = -grad (in place)
  union { float Mp[NV], tmp[NV]; };  // Mp: line search to update; tmp: J'f (newton_grad), start's qacc, implicitfast
  float efc_D[NROW], efc_jar[NROW];
  unsigned char efc_kind[NROW];
  unsigned short efc_a[NROW];
  signed char efc_b[NROW];
  int ncon, nefc, unsupported, min_nu_bits, nefc_dof;
  union {
    float con_C[NC][5];  // the Hessian assembly only (J p, the forces and their per-contact sums are dead there)
    alignas(16) float cbuf[72];  // CHOL_SOLVE's scratch: the factorizations (con_C is dead from there on)
    struct {
      union { float con_dot[NC][3], con_f[NC][3]; };
      float efc_jv[NROW];
    };
  };
  unsigned short con_row[NC];
  unsigned char con_dim[NC], con_nrow[NC], con_nd[NC];
  float *qpos, *qvel;  // the kernel's outputs (implicitfast only)
  const float *qsm, *qacc0, *con_mu, *efc_aref;  // in Hand
  const int* con_pair;
  const float (*con_J)[3][KDOF_STORAGE];
  const unsigned char (*con_dof)[KDOF_STORAGE];
  const unsigned long long* con_mask;
#ifdef DK_PROFILE
  long long prof[NPROF], prof_t, prof_u;
#endif
#ifdef DK_TAPE_VERIFY
  float Lv[NTRI];
#endif
};
typedef SimNT<SPLIT_NCON, SPLIT_NEFC> SimN;  // k_post's shared memory
typedef SimNT<MAXCON, NEFC> SimNfull;  // the worlds beyond it (Hand::big)

struct Hand {
#if HAS_NATIVE_COLLISION
  CMScratch cm_storage;
#endif
  float M[NTRI], qsm[NV], qacc0[NV];
  float efc_D[NEFC], efc_aref[NEFC];
  alignas(16) float con_J[MAXCON][3][KDOF_STORAGE];  // (16-byte aligned: con_jv's 128-bit loads)
  float con_mu[MAXCON];
  int con_pair[MAXCON], con_dim[MAXCON], con_row[MAXCON], con_nrow[MAXCON], con_nd[MAXCON];
  int ncon, nefc, nefc_dof, unsupported, min_nu_bits;
  unsigned char efc_kind[NEFC];
  unsigned short efc_a[NEFC];
  signed char efc_b[NEFC];
  alignas(16) unsigned char con_dof[MAXCON][KDOF_STORAGE];
  unsigned long long con_mask[MAXCON];
  float con_dist[MAXCON], con_pos[MAXCON][3], con_frame[MAXCON][9];  // substep_pre's (collision, Jacobians, rows)
  SimNfull big;  // k_post's state of a world whose contacts or rows exceed SimN's capacity
};

struct SimP {  // substep_pre: smooth dynamics, qacc_smooth, collision, contact Jacobians (into Hand), rows (into Hand)
#if HAS_NATIVE_COLLISION
  CMScratch* cm_scratch;
  float* feature_rec;
#endif
  float Q[NB][4], X[NB][3];
  float W[NB][3], V[NB][3];  // body velocities, then (body forces to contact Jacobians) the hinge motion subspaces
  union {  // the smooth phase's body quantities, then qacc_smooth's factor, then the collision's geoms and candidates
    struct { float AL[NB][3], A[NB][3], F[NB][3], N[NB][3], J[NB][10]; };
    struct { float L[NTRI]; alignas(16) float cbuf[72]; };  // CHOL_SOLVE's scratch
    struct {
      float gx[NGEOMC_STORAGE][3], gq[NGEOMC_STORAGE][4];
#if HAS_NATIVE_COLLISION
      float gaabb[NGEOMC_STORAGE][6];
#endif
      float con_dot[MAXCON][3]; unsigned short cand[NPAIR_STORAGE];
    };
  };
  float* M;  // in Hand (the stores of the CRB, the loads of qacc_smooth's factorization)
  float qpos[NQ], qvel[NV], zero[NV], qsm[NV], qacc0[NV];
  float com[COM_WIDTH], fric;
  int ncon, nefc, unsupported, min_nu_bits, nefc_dof;
  int con_pair[MAXCON], con_dim[MAXCON], con_row[MAXCON], con_nrow[MAXCON], con_nd[MAXCON];
  float con_mu[MAXCON];
  unsigned char con_slot[MAXCON];
  float *con_dist, (*con_pos)[3], (*con_frame)[9];  // in Hand
  float (*con_J)[3][KDOF_STORAGE];
  unsigned char (*con_dof)[KDOF_STORAGE];
  unsigned long long* con_mask;
  unsigned char* efc_kind;
  unsigned short* efc_a;
  signed char* efc_b;
  float *efc_D, *efc_aref;
#ifdef DK_PROFILE
  long long prof[NPROF], prof_t, prof_u;
#endif
};
"""


def extra_constants(m):
    """dof -> body map and per-body ancestor bitmasks (bit k: body k is the body or one of its ancestors)."""
    dofbody = np.asarray(m.dof_bodyid, np.float64)
    mask = np.zeros(m.nbody, np.uint64)
    for b in range(m.nbody):
        k = b
        while k > 0:
            mask[b] |= np.uint64(1) << np.uint64(k)
            k = m.body_parentid[k]
    return dofbody, mask


EXTRA_HEADER = "DKC int c_dofbody[NV];\nDKC unsigned long long c_ancmask[NB];\n"
