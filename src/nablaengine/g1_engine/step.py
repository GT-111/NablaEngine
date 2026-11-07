# Third-party code: the collision primitives in STEP_DEVICE (plane against sphere / capsule / box, sphere-sphere,
# sphere-capsule, capsule-capsule, sphere-box, capsule-box, box-box) are translated to CUDA from MuJoCo Warp,
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
for this robot: explicit contact pairs only (plane against sphere / capsule / box, sphere-sphere, sphere-capsule,
capsule-capsule, capsule-box and box-box, translated from MuJoCo Warp's primitives; pairs whose bounding spheres do not
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

NEFC = 192
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
GEOM_TYPE = {int(mujoco.mjtGeom.mjGEOM_PLANE): 0, int(mujoco.mjtGeom.mjGEOM_SPHERE): 1,
             int(mujoco.mjtGeom.mjGEOM_CAPSULE): 2, int(mujoco.mjtGeom.mjGEOM_BOX): 3}


def maxcon(m):
    """Contact capacity per world: 32 for the Menagerie model (49 pairs, at most 11 contacts seen), 64 for the mjlab
    model (502 pairs; up to 28 foot contacts alone). $DK_MAXCON overrides it (experiments); it
    changes every build, the FFI library included, so it warns."""
    default = 32 if m.npair <= 64 else 64
    if os.environ.get("DK_MAXCON"):
        warnings.warn(f"DK_MAXCON={os.environ['DK_MAXCON']} overrides the contact capacity (default {default}) of the "
                      "engine builds and the FFI library (an experiment switch)", RuntimeWarning, stacklevel=2)
        return int(os.environ["DK_MAXCON"])
    return default


def max_support(m):
    """Largest number of dofs a contact Jacobian can touch: the dofs on exactly one of the two bodies' chains, over all
    pairs (16 for both G1 models: a hand on the floor, root 6 + waist 3 + arm 7)."""
    dofbody, mask = extra_constants(m)
    return max(sum(1 for i in range(m.nv) if ((mask[b1] >> int(dofbody[i])) ^ (mask[b2] >> int(dofbody[i]))) & 1)
               for b1, b2 in ((m.geom_bodyid[m.pair_geom1[p]], m.geom_bodyid[m.pair_geom2[p]]) for p in range(m.npair)))


def collision_geoms(m):
    """The geoms that appear in a contact pair, in model order (index into the geom table)."""
    return sorted(set(int(g) for g in m.pair_geom1) | set(int(g) for g in m.pair_geom2))


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
    for i in range(m.npair):
        key = (float(m.pair_friction[i, 0]), *map(float, m.pair_solref[i]), *map(float, m.pair_solimp[i]))
        index.append(keys.setdefault(key, len(keys)))
    return np.asarray(list(keys), np.float64).reshape(len(keys), PSET_WIDTH), index


def const_layout(m, terrain=None):
    """[(symbol, kind, length)] of pair_constants(m, ..., terrain) in packing order (terrain: (TerrainConfig, the
    robot geoms of terrain.robot_terrain_geoms): NTG more pairs and one more geom, the terrain's pseudo geom)."""
    ng = len(collision_geoms(m)) + (1 if terrain else 0)
    npair = m.npair + (len(terrain[1]) if terrain else 0)
    sizes = {"nb": m.nbody, "nv": m.nv, "nv2": 2 * m.nv}
    npset = len(pair_param_sets(m)[0])
    return ([(s, k, npair * w) for s, k, w in PAIR_LAYOUT] + [("c_pset", "f", npset * PSET_WIDTH)]
            + [(s, k, ng * w) for s, k, w in GEOM_LAYOUT]
            + [(s, k, sizes[w] if isinstance(w, str) else w) for s, k, w in SOLVER_LAYOUT])


def pair_constants(m, foot_pairs, terrain=None):
    """Contact-pair arrays (geoms ordered by type, as MuJoCo's collision table expects), the collision-geom table and
    the solver constants, as one float32 array in const_layout(m, terrain) order. terrain: (TerrainConfig, [(robot
    geom, floor pair, foot)]): after the model's pairs, one virtual pair per such geom (g1 = the terrain's pseudo geom
    on the world body, the last geom table row; the floor pair's parameter set, condim and foot flag)."""
    geoms = collision_geoms(m)
    gid = {g: i for i, g in enumerate(geoms)}
    sets, words = pair_param_sets(m), []
    foot = set(int(x) for x in foot_pairs)
    assert foot <= set(range(m.npair)), f"foot pairs {sorted(foot)}: not all contact pair indices (0..{m.npair - 1})"
    for i in range(m.npair):
        g1, g2 = int(m.pair_geom1[i]), int(m.pair_geom2[i])
        if GEOM_TYPE[int(m.geom_type[g1])] > GEOM_TYPE[int(m.geom_type[g2])]:
            g1, g2 = g2, g1
        assert m.pair_margin[i] == 0 and m.pair_gap[i] == 0 and m.pair_friction[i, 0] == m.pair_friction[i, 1]
        words.append(pair_word(gid[g1], gid[g2], sets[1][i], int(m.pair_dim[i]), int(i in foot),
                               int(GEOM_TYPE[int(m.geom_type[g1])] == 0)))
    if terrain:
        tg = len(geoms)  # the terrain's pseudo geom
        for g, p, ft in terrain[1]:
            words.append(pair_word(tg, gid[g], sets[1][p], int(m.pair_dim[p]), ft, 0))
    words = np.asarray(words, np.int64)
    out = [np.stack([words & 0xFFFF, words >> 16], 1).ravel(), np.asarray(sets[0], np.float64).ravel()]
    gtab = [m.geom_bodyid[geoms], [GEOM_TYPE[int(m.geom_type[g])] for g in geoms], m.geom_size[geoms],
            m.geom_pos[geoms], m.geom_quat[geoms], m.geom_rbound[geoms]]
    if terrain:  # body 0, type 4 (terrain), at the origin
        extra = [[0], [4], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]], [0.0]]
        gtab = [np.concatenate([np.asarray(a, np.float64).reshape(len(geoms), -1),
                                np.asarray(b, np.float64).reshape(1, -1)]) for a, b in zip(gtab, extra)]
    out += gtab
    jr = np.zeros((m.nv, 2))
    jlim = np.zeros(m.nv)
    for j in range(1, m.njnt):
        jr[m.jnt_dofadr[j]], jlim[m.jnt_dofadr[j]] = m.jnt_range[j], m.jnt_limited[j]
    assert not m.jnt_margin.any() and (m.jnt_solref[1:] == m.jnt_solref[1]).all()
    assert (m.jnt_solimp[1:] == m.jnt_solimp[1]).all() and (m.dof_solref == m.dof_solref[0]).all()
    assert (m.dof_solimp == m.dof_solimp[0]).all() and m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
    out += [m.body_invweight0[:, 0], m.dof_invweight0, m.dof_frictionloss, jr, jlim, m.jnt_solref[1], m.jnt_solimp[1],
            m.dof_solref[0], m.dof_solimp[0],
            [m.opt.impratio, m.stat.meaninertia, m.opt.tolerance, m.npair]]
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
    pw = list(m.pair_solimp[:, 4]) + list(m.jnt_solimp[1:, 4]) + list(m.dof_solimp[m.dof_frictionloss > 0, 4])
    return all(float(x) == 2.0 for x in pw)


def dof_body_root6(m):
    """True when body 1 (the free root) has dofs 0-5 and every later dof i is the only dof of body i - 4 (bodies and
    dofs in the same order): a contact's support mask is then a shift of its bodies' chain mask (support_mask)."""
    d = np.asarray(m.dof_bodyid)
    return m.nv >= 6 and bool((d[:6] == 1).all() and (d[6:] == np.arange(6, m.nv) - 4).all())


def step_header(m, max_contacts=None, terrain=None):
    p, g, kdof = m.npair, len(collision_geoms(m)), max_support(m)
    hs = hess_columns(kdof)
    hcol = [a | b << 8 | thr << 16 for _, _, tab in hs for row in tab for a, b, thr in row]
    ter, pt = "", p
    if terrain:  # nablaengine.terrain: NTG virtual pairs after the model's, the pseudo geom as the last geom row
        cfg, tg = terrain
        ter = (f"#define HAS_TERRAIN 1\n#define NTG {len(tg)}  // robot geoms that collide with the terrain\n"
               f"#define TERRAIN_FLOOR {int(cfg.floor)}  // keep the model's floor plane pairs\n"
               f"#define TERRAIN_CAP_PTS {int(cfg.cap_points)}  // heightfield samples per capsule\n"
               f"#define TERRAIN_HF_NATIVE {int(cfg.mode == 'native')}  // heightfield contacts per prism\n")
        pt, g = p + len(tg), g + 1
    return ter + f"""
#define NPAIR {p}
#define NGEOMC {g}
#define MAXCON {max_contacts or maxcon(m)}
#define KDOF {kdof}
#define HSETS {len(hs)}  // Hessian assembly (hess_columns(KDOF)): sets of 3 passes, their first rows, pairs per lane
#define HSET_LO {{{", ".join(str(lo) for lo, _, _ in hs)}}}
__device__ const unsigned c_hcol[{len(hcol)}] = {{{", ".join(map(str, hcol))}}};  // a | b << 8 | thr << 16
#define HAS_BOX {int(any(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in collision_geoms(m)))}  // compile box code
#define FLOOR_G {floor_geom(m)}  // the geom table index of the only plane (-1: none or several)
#define NEFC {NEFC}
#define DOF_BODY_ROOT6 {int(dof_body_root6(m))}  // step.dof_body_root6 (support_mask in closed form)
#define NFLOSS {int((m.dof_frictionloss > 0).sum())}  // dofs with a frictionloss row (the first rows)
#define SOLIMP_POW2 {int(solimp_pow2(m))}  // every solimp power of the model is 2 (kbimp without the general powers)
#define NPROF {len(PROF_PHASES)}
DKC unsigned int c_pword[{pt}];  // pair_word(): g1 | g2 << 8 | pset << 16 | dim << 24 | foot << 28 | plane << 29
DKC float c_pset[{len(pair_param_sets(m)[0]) * PSET_WIDTH}];  // per parameter set: mu, solref[2], solimp[5]
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
DKC int c_gbody[{g}], c_gtype[{g}];
DKC float c_gsize[{g * 3}], c_gpos[{g * 3}], c_gquat[{g * 4}], c_grbound[{g}];
DKC float c_binvw[NB], c_dinvw[NV], c_floss[NV], c_jrange[NV * 2], c_jlim[NV], c_jsolref[2], c_jsolimp[5],
    c_dsolref[2], c_dsolimp[5], c_sopt[4];  // impratio, meaninertia, tolerance, npair
"""


STEP_DEVICE = r"""
// NEWTON_REUSE: a Newton iteration whose rows all kept their zones (a third of them on a 4096-state corpus) re-uses
// the last factor (chol_resolve). Split forward, ms per control step (that corpus, 10 iterations): RTX 3090 3.07
// against 3.34 without (1.66 / 1.77 at 1787 worlds, the waves of 4096 worlds on the target GPUs); V100 2% slower
// (6.34 / 6.20)
#ifndef NEWTON_REUSE
#define NEWTON_REUSE (NV > 32 && NV <= 35)
#endif
struct Sim {
  float Q[NB][4], X[NB][3];  // body poses
  union {  // the smooth phase's body quantities and the constraint rows are never alive at the same time
    struct {
      float W[NB][3], V[NB][3];
      union {  // the collision's candidate pairs (after the mass matrix)
        struct { float AL[NB][3], A[NB][3], F[NB][3], N[NB][3], J[NB][10]; };
        unsigned short cand[NPAIR];
      };
    };
    struct {
      unsigned char efc_kind[NEFC], efc_a[NEFC];
      signed char efc_b[NEFC];
      float efc_D[NEFC], efc_aref[NEFC], efc_jar[NEFC], efc_jv[NEFC];  // efc_jv: J p in the line search, else forces
    };
  };
  float M[NTRI], L[NTRI];  // mass matrix and Cholesky scratch, packed lower triangles TRI(i, j)
  float qpos[NQ], qvel[NV], zero[NV], qsm[NV], qacc0[NV], qacc[NV], Ma[NV], grad[NV], p[NV], Mp[NV], tmp[NV];
  alignas(16) float cbuf[72];  // CHOL_SOLVE's scratch (chol_solve_bcast)
  float com[3], fric;
  float gx[NGEOMC][3], gq[NGEOMC][4];  // world poses of the collision geoms
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
  alignas(16) float con_J[MAXCON][3][KDOF];  // (16-byte aligned: con_jv's 128-bit loads)
  alignas(16) unsigned char con_dof[MAXCON][KDOF];
  unsigned char con_slot[MAXCON];  // which contact of its pair (tape): plane-capsule end, parallel capsule-capsule candidate
  int con_nd[MAXCON];
  unsigned long long con_mask[MAXCON];  // the support as dof bits (bit i: dof i; con_dof[c][k] is the k-th set bit)
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
                                            const float* frame) {
  int c = atomicAdd(&s.ncon, 1);
  if (c >= MAXCON) return false;
  s.con_pair[c] = pair;
  s.con_slot[c] = (unsigned char)slot;
  s.con_dim[c] = PDIM(pair);
  s.con_dist[c] = dist;
  for (int k = 0; k < 3; k++) s.con_pos[c][k] = pos[k];
  for (int k = 0; k < 9; k++) s.con_frame[c][k] = frame[k];
  s.con_mu[c] = PFOOT(pair) ? s.fric : PMU(pair);
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
__device__ __forceinline__ void mm3(const float* A, const float* B, float* C) {  // C = A B
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
}
__device__ __forceinline__ void mmt3(const float* A, const float* B, float* C) {  // C = A B'
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) C[3 * i + j] = A[3 * i] * B[3 * j] + A[3 * i + 1] * B[3 * j + 1] + A[3 * i + 2] * B[3 * j + 2];
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

__device__ __forceinline__ void rotmore_of(int f, float* r) {
  for (int k = 0; k < 9; k++) r[k] = 0.f;
  if (f == 0) { r[2] = -1.f; r[4] = 1.f; r[6] = 1.f; }
  else if (f == 1) { r[0] = 1.f; r[5] = -1.f; r[7] = 1.f; }
  else if (f == 2) { r[0] = 1.f; r[4] = 1.f; r[8] = 1.f; }
  else if (f == 3) { r[2] = 1.f; r[4] = 1.f; r[6] = -1.f; }
  else if (f == 4) { r[0] = 1.f; r[5] = 1.f; r[7] = -1.f; }
  else if (f == 5) { r[0] = -1.f; r[4] = 1.f; r[8] = -1.f; }
}

// box-box (MuJoCo Warp's box_box, margin 0): separating-axis test, then face-vertex or edge-edge clipping
#define BB_MAXPT 24
#ifndef BOX_BOX_CCD
#define BOX_BOX_CCD 1  // 1: MuJoCo Warp default (native CCD); 0: the separating-axis primitive (NATIVECCD disabled)
#endif
template <class S>
__device__ void box_box(S& s, int pair, const float* p1, const float* R1, const float* s1, const float* p2,
                        const float* R2, const float* s2) {
  const float margin = 0.f;
  float d[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]}, dn[3] = {-d[0], -d[1], -d[2]}, pos21[3], pos12[3];
  mtv3(R1, d, pos21);
  mtv3(R2, dn, pos12);
  float rot21[9], rot12[9], a21[9], a12[9], plen2[3], plen1[3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) rot21[3 * i + j] = R1[i] * R2[j] + R1[3 + i] * R2[3 + j] + R1[6 + i] * R2[6 + j];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { rot12[3 * i + j] = rot21[3 * j + i]; a21[3 * i + j] = fabsf(rot21[3 * i + j]); }
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) a12[3 * i + j] = a21[3 * j + i];
  mv3(a21, s2, plen2);
  mv3(a12, s1, plen1);
  float sep = margin + 3.f * (s1[0] + s2[0]) + 3.f * (s1[1] + s2[1]) + 3.f * (s1[2] + s2[2]);
  int code = -1;
  for (int i = 0; i < 3; i++) {
    float c1 = -fabsf(pos21[i]) + s1[i] + plen2[i], c2 = -fabsf(pos12[i]) + s2[i] + plen1[i];
    if (c1 < -margin || c2 < -margin) return;
    if (c1 < sep) { sep = c1; code = i + 3 * int(pos21[i] < 0.f); }
    if (c2 < sep) { sep = c2; code = i + 3 * int(pos12[i] < 0.f) + 6; }
  }
  float clnorm[3] = {0.f, 0.f, 0.f};
  bool inv = false;
  int cle1 = 0, cle2 = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      float ca[3];
      if (i == 0) { ca[0] = 0.f; ca[1] = -rot12[3 * j + 2]; ca[2] = rot12[3 * j + 1]; }
      else if (i == 1) { ca[0] = rot12[3 * j + 2]; ca[1] = 0.f; ca[2] = -rot12[3 * j]; }
      else { ca[0] = -rot12[3 * j + 1]; ca[1] = rot12[3 * j]; ca[2] = 0.f; }
      float cl = sqrtf(dot3(ca, ca));
      if (cl < 1e-15f) continue;
      for (int k = 0; k < 3; k++) ca[k] /= cl;
      float bd = dot3(pos21, ca), c3 = 0.f;
      for (int k = 0; k < 3; k++) {
        if (k != i) c3 += s1[k] * fabsf(ca[k]);
        if (k != j) c3 += s2[k] * a21[3 * i + 3 - k - j] / cl;
      }
      c3 -= fabsf(bd);
      if (c3 < -margin) return;
      if (c3 < sep * (1.f - 1e-12f)) {
        sep = c3; cle1 = 0; cle2 = 0;
        for (int k = 0; k < 3; k++) {
          if (k != i && (int(ca[k] > 0.f) ^ int(bd < 0.f))) cle1 += 1 << k;
          if (k != j && (int(rot21[3 * i + 3 - k - j] > 0.f) ^ int(bd < 0.f) ^ int((k - j + 3) % 3 == 1))) cle2 += 1 << k;
        }
        code = 12 + i * 3 + j;
        for (int k = 0; k < 3; k++) clnorm[k] = ca[k];
        inv = bd < 0.f;
      }
    }
  if (code == -1) return;
  float pts[BB_MAXPT][3], depth[BB_MAXPT], rw[9], pw[3], nrm[3], hz;
  int n = 0;
  if (code < 12) {  // face-vertex
    int face = code % 6, bi = code / 6;
    float rm[9], r[9], p[3], ss[3], sv[3], t[3];
    rotmore_of(face, rm);
    mm3(rm, bi ? rot12 : rot21, r);
    mv3(rm, bi ? pos12 : pos21, p);
    mv3(rm, bi ? s2 : s1, t);
    for (int k = 0; k < 3; k++) { ss[k] = fabsf(t[k]); sv[k] = bi ? s1[k] : s2[k]; }
    float lx = ss[0], ly = ss[1];
    hz = ss[2];
    p[2] -= hz;
    int clc = 0;
    for (int i = 0; i < 3; i++) if (r[6 + i] < 0.f) clc += 1 << i;
    float lp[3] = {p[0], p[1], p[2]};
    for (int i = 0; i < 3; i++)
      for (int k = 0; k < 3; k++) lp[k] += r[3 * k + i] * sv[i] * ((clc & (1 << i)) ? 1.f : -1.f);
    int dirs = 0;
    float cn1[3] = {0.f, 0.f, 0.f}, cn2[3] = {0.f, 0.f, 0.f};
    for (int i = 0; i < 3; i++)
      if (fabsf(r[6 + i]) < 0.5f) {
        float* cn = dirs ? cn2 : cn1;
        for (int k = 0; k < 3; k++) cn[k] = r[3 * k + i] * sv[i] * ((clc & (1 << i)) ? -2.f : 2.f);
        dirs++;
      }
    for (int i = 0; i < dirs * dirs; i++)
      for (int q = 0; q < 2; q++) {
        float la[3], lb[3];
        for (int k = 0; k < 3; k++) {
          la[k] = lp[k] + (i < 2 ? 0.f : (i == 2 ? cn1[k] : cn2[k]));
          lb[k] = (i == 0 || i == 3) ? cn1[k] : cn2[k];
        }
        if (fabsf(lb[q]) > 1e-15f) {
          float br = 1.f / lb[q];
          for (int j = -1; j < 2; j += 2) {
            float l = ss[q] * float(j), c1 = (l - la[q]) * br;
            if (c1 < 0.f || c1 > 1.f) continue;
            float c2 = la[1 - q] + lb[1 - q] * c1;
            if (fabsf(c2) > ss[1 - q]) continue;
            if (n < BB_MAXPT) { for (int k = 0; k < 3; k++) pts[n][k] = la[k] + c1 * lb[k]; }
            n++;
          }
        }
      }
    if (dirs == 2) {
      float ax = cn1[0], bx = cn2[0], ay = cn1[1], by = cn2[1], C = safe_div(1.f, ax * by - bx * ay);
      for (int i = 0; i < 4; i++) {
        float llx = (i / 2) ? lx : -lx, lly = (i % 2) ? ly : -ly, x = llx - lp[0], y = lly - lp[1];
        float u = (x * by - y * bx) * C, v = (y * ax - x * ay) * C;
        if (u > 0.f && v > 0.f && u < 1.f && v < 1.f) {
          if (n < BB_MAXPT) { pts[n][0] = llx; pts[n][1] = lly; pts[n][2] = lp[2] + u * cn1[2] + v * cn2[2]; }
          n++;
        }
      }
    }
    for (int i = 0; i < (1 << dirs); i++) {
      float tv[3];
      for (int k = 0; k < 3; k++) tv[k] = lp[k] + float(i & 1) * cn1[k] + float((i & 2) != 0) * cn2[k];
      if (tv[0] > -lx && tv[0] < lx && tv[1] > -ly && tv[1] < ly) {
        if (n < BB_MAXPT) { for (int k = 0; k < 3; k++) pts[n][k] = tv[k]; }
        n++;
      }
    }
    if (n > BB_MAXPT) atomicOr(&s.unsupported, 2);
    int m = min(n, BB_MAXPT);
    n = 0;
    for (int i = 0; i < m; i++) {
      if (pts[i][2] > margin) continue;
      for (int k = 0; k < 3; k++) pts[n][k] = pts[i][k];
      depth[n] = pts[n][2];
      pts[n][2] *= 0.5f;
      n++;
    }
    mmt3(bi ? R2 : R1, rm, rw);
    for (int k = 0; k < 3; k++) { pw[k] = bi ? p2[k] : p1[k]; nrm[k] = (bi ? -1.f : 1.f) * rw[3 * k + 2]; }
  } else {  // edge-edge
    int e1 = (code - 12) / 3, e2 = (code - 12) % 3;
    int ax1 = 1 - (e2 & 1), ax2 = 2 - (e2 & 2), pax1 = 1 - (e1 & 1), pax2 = 2 - (e1 & 2);
    if (a21[3 * e1 + ax1] < a21[3 * e1 + ax2]) { int t = ax1; ax1 = ax2; ax2 = t; }
    if (a12[3 * e2 + pax1] < a12[3 * e2 + pax2]) { int t = pax1; pax1 = pax2; pax2 = t; }
    float rm[9], p[3], rn[3], r[9], sv[3], t[3];
    rotmore_of((cle1 & (1 << pax2)) ? pax2 : pax2 + 3, rm);
    mv3(rm, pos21, p);
    mv3(rm, clnorm, rn);
    mm3(rm, rot21, r);
    mtv3(rm, s1, t);
    for (int k = 0; k < 3; k++) sv[k] = fabsf(t[k]);
    float lx = sv[0], ly = sv[1];
    hz = sv[2];
    p[2] -= hz;
    float pu[4][3];
    float g1 = (cle2 & (1 << ax1)) ? 1.f : -1.f, g2 = (cle2 & (1 << ax2)) ? 1.f : -1.f;
    for (int k = 0; k < 3; k++) {
      float b0 = p[k] + r[3 * k + ax1] * s2[ax1] * g1 + r[3 * k + ax2] * s2[ax2] * g2;
      float b2 = p[k] - r[3 * k + ax1] * s2[ax1] * g1 + r[3 * k + ax2] * s2[ax2] * g2;
      float de = r[3 * k + e2] * s2[e2];
      pu[0][k] = b0 + de; pu[1][k] = b0 - de; pu[2][k] = b2 + de; pu[3][k] = b2 - de;
    }
    float alp[3], acn1[3], acn2[3];
    for (int k = 0; k < 3; k++) { alp[k] = pu[0][k]; acn1[k] = pu[1][k] - pu[0][k]; acn2[k] = pu[2][k] - pu[0][k]; }
    if (fabsf(rn[2]) < 1e-15f) return;
    float sg = inv ? -1.f : 1.f, innorm = sg / rn[2], pp[4][3];
    for (int i = 0; i < 4; i++) {
      float cs = pu[i][2] * sg * innorm;
      for (int k = 0; k < 3; k++) pp[i][k] = pu[i][k] - rn[k] * cs;
    }
    float plp[3], pcn1[3], pcn2[3];
    for (int k = 0; k < 3; k++) { plp[k] = pp[0][k]; pcn1[k] = pp[1][k] - pp[0][k]; pcn2[k] = pp[2][k] - pp[0][k]; }
    for (int i = 0; i < 4; i++)
      for (int q = 0; q < 2; q++) {
        float la = plp[q] + (i < 2 ? 0.f : (i == 2 ? pcn1[q] : pcn2[q]));
        float lb = (i == 0 || i == 3) ? pcn1[q] : pcn2[q];
        float lc = plp[1 - q] + (i < 2 ? 0.f : (i == 2 ? pcn1[1 - q] : pcn2[1 - q]));
        float ld = (i == 0 || i == 3) ? pcn1[1 - q] : pcn2[1 - q];
        float lua[3], lub[3];
        for (int k = 0; k < 3; k++) {
          lua[k] = alp[k] + (i < 2 ? 0.f : (i == 2 ? acn1[k] : acn2[k]));
          lub[k] = (i == 0 || i == 3) ? acn1[k] : acn2[k];
        }
        if (fabsf(lb) > 1e-15f) {
          float br = 1.f / lb;
          for (int j = -1; j < 2; j += 2) {
            if (n == 8) break;
            float l = sv[q] * float(j), c1 = (l - la) * br;
            if (c1 < 0.f || c1 > 1.f) continue;
            float c2 = lc + ld * c1;
            if (fabsf(c2) > sv[1 - q]) continue;
            if ((lua[2] + lub[2] * c1) * innorm > margin) continue;
            for (int k = 0; k < 3; k++) pts[n][k] = lua[k] * 0.5f + c1 * lub[k] * 0.5f;
            pts[n][q] += 0.5f * l;
            pts[n][1 - q] += 0.5f * c2;
            depth[n] = pts[n][2] * innorm * 2.f;
            n++;
          }
        }
      }
    int nl = n;
    float ax = pcn1[0], bx = pcn2[0], ay = pcn1[1], by = pcn2[1], C = safe_div(1.f, ax * by - bx * ay);
    for (int i = 0; i < 4; i++) {
      if (n == 8) break;
      float llx = (i / 2) ? lx : -lx, lly = (i % 2) ? ly : -ly, x = llx - plp[0], y = lly - plp[1];
      float u = (x * by - y * bx) * C, v = (y * ax - x * ay) * C;
      if (nl == 0) { if ((u < 0.f || u > 1.f) && (v < 0.f || v > 1.f)) continue; }
      else if (u < 0.f || v < 0.f || u > 1.f || v > 1.f) continue;
      u = fminf(fmaxf(u, 0.f), 1.f);
      v = fminf(fmaxf(v, 0.f), 1.f);
      float w = 1.f - u - v, vt[3], q2[3] = {llx, lly, 0.f};
      for (int k = 0; k < 3; k++) vt[k] = pu[0][k] * w + pu[1][k] * u + pu[2][k] * v;
      float tc1 = 0.f;
      for (int k = 0; k < 3; k++) tc1 += (q2[k] - vt[k]) * (q2[k] - vt[k]);
      if (vt[2] > 0.f && tc1 > margin * margin) continue;
      for (int k = 0; k < 3; k++) pts[n][k] = 0.5f * (q2[k] + vt[k]);
      depth[n] = sqrtf(tc1) * (vt[2] < 0.f ? -1.f : 1.f);
      n++;
    }
    int nf = n;
    for (int i = 0; i < 4; i++) {
      if (n >= 8) break;
      float x = pu[i][0], y = pu[i][1];
      if (nl == 0 && nf != 0) { if ((x < -lx || x > lx) && (y < -ly || y > ly)) continue; }
      else if (x < -lx || x > lx || y < -ly || y > ly) continue;
      float c1 = 0.f;
      for (int j = 0; j < 2; j++) {
        if (pu[i][j] < -sv[j]) c1 += (pu[i][j] + sv[j]) * (pu[i][j] + sv[j]);
        else if (pu[i][j] > sv[j]) c1 += (pu[i][j] - sv[j]) * (pu[i][j] - sv[j]);
      }
      c1 += pu[i][2] * innorm * pu[i][2] * innorm;
      if (pu[i][2] > 0.f && c1 > margin * margin) continue;
      float tp[3] = {pu[i][0], pu[i][1], 0.f};
      for (int j = 0; j < 2; j++) {
        if (pu[i][j] < -sv[j]) tp[j] = -sv[j] * 0.5f;
        else if (pu[i][j] > sv[j]) tp[j] = sv[j] * 0.5f;
      }
      for (int k = 0; k < 3; k++) pts[n][k] = (tp[k] + pu[i][k]) * 0.5f;
      depth[n] = sqrtf(c1) * (pu[i][2] < 0.f ? -1.f : 1.f);
      n++;
    }
    mmt3(R1, rm, rw);
    float t2[3];
    mv3(rw, rn, t2);
    for (int k = 0; k < 3; k++) { pw[k] = p1[k]; nrm[k] = sg * t2[k]; }
  }
  if (n > 8) { atomicOr(&s.unsupported, 2); n = 8; }
  float fr[9];
  make_frame(nrm, fr);
#if BOX_BOX_CCD
  // MuJoCo Warp collides box-box with GJK/EPA (native CCD): one contact at the minimum-penetration axis, which is the
  // separating-axis result above, unless multiccd finds two faces within 0.0889 rad of facing each other or an edge
  // within that angle of lying on a face; those near-aligned cases are flagged (bit 8), not emulated
  const float ftol = 0.99605107f, etol = 0.08878297f;  // cos(0.0889), sin(0.0889)
  bool multi = false;
  if (code < 12) {  // face of one box against a vertex of the other: any axis of the vertex box near-perpendicular to n
    const float* Rv = (code / 6) ? R1 : R2;
    for (int k = 0; k < 3; k++) {
      float c = fabsf(Rv[k] * nrm[0] + Rv[3 + k] * nrm[1] + Rv[6 + k] * nrm[2]);
      if (c > ftol || c < etol) multi = true;
    }
    int best = -1;
    for (int i = 0; i < n; i++) if (depth[i] < 0.f && (best < 0 || depth[i] < depth[best])) best = i;
    if (multi) atomicOr(&s.unsupported, 8);
    if (best < 0) return;
    float pl[3] = {pts[best][0], pts[best][1], pts[best][2] + hz}, pc[3];
    mv3(rw, pl, pc);
    for (int k = 0; k < 3; k++) pc[k] += pw[k];
    add_contact(s, pair, 0, depth[best], pc, fr);
  } else {  // edge-edge: the midpoint of the two edges' closest points
    int e1 = (code - 12) / 3, e2 = (code - 12) % 3;
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        float c = fabsf(R1[a] * R2[b] + R1[3 + a] * R2[3 + b] + R1[6 + a] * R2[6 + b]);
        if (a != e1 && b != e2 && c > ftol) multi = true;      // adjacent faces facing each other
        if (a == e1 && b != e2 && c < etol) multi = true;      // edge 1 lying on a face of box 2
        if (b == e2 && a != e1 && c < etol) multi = true;      // edge 2 lying on a face of box 1
      }
    if (multi) atomicOr(&s.unsupported, 8);
    float l1[3], l2[3], c1[3], c2[3], d1[3], d2[3];
    for (int k = 0; k < 3; k++) {
      l1[k] = k == e1 ? 0.f : ((cle1 & (1 << k)) ? s1[k] : -s1[k]);
      l2[k] = k == e2 ? 0.f : ((cle2 & (1 << k)) ? s2[k] : -s2[k]);
    }
    mv3(R1, l1, c1);
    mv3(R2, l2, c2);
    for (int k = 0; k < 3; k++) {
      c1[k] += p1[k]; c2[k] += p2[k];
      d1[k] = R1[3 * k + e1] * s1[e1]; d2[k] = R2[3 * k + e2] * s2[e2];
    }
    // closest points of the segments c1 + t1 d1, c2 + t2 d2 with t in [-1, 1]
    float r[3] = {c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2]};
    float a11 = dot3(d1, d1), a12 = dot3(d1, d2), a22 = dot3(d2, d2), b1 = dot3(d1, r), b2 = dot3(d2, r);
    float det = a11 * a22 - a12 * a12, t1 = 0.f, t2 = 0.f;
    if (det > 1e-12f) t1 = fminf(fmaxf((a12 * b2 - a22 * b1) / det, -1.f), 1.f);
    t2 = fminf(fmaxf((b2 + a12 * t1) / a22, -1.f), 1.f);
    t1 = fminf(fmaxf((a12 * t2 - b1) / a11, -1.f), 1.f);
    float pc[3];
    for (int k = 0; k < 3; k++) pc[k] = 0.5f * (c1[k] + t1 * d1[k] + c2[k] + t2 * d2[k]);
    if (sep > 0.f) add_contact(s, pair, 0, -sep, pc, fr);
  }
#else
  for (int i = 0; i < n; i++) {
    if (!(depth[i] < 0.f)) continue;
    float pl[3] = {pts[i][0], pts[i][1], pts[i][2] + hz}, pc[3];
    mv3(rw, pl, pc);
    for (int k = 0; k < 3; k++) pc[k] += pw[k];
    add_contact(s, pair, i, depth[i], pc, fr);
  }
#endif
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
  if (t1 == 3 && t2 == 3) {  // box-box
    float R1[9], R2[9];
    quat2mat(q1, R1);
    quat2mat(q2, R2);
    box_box(s, pair, p1, R1, sz1, p2, R2, sz2);
    return;
  }
#endif
  atomicOr(&s.unsupported, 1);  // other type combinations do not occur in the G1 model
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

// J v for every row (v a shared NV vector) into out[] (NEFC entries); contact rows from con_jv
template <class S>
__device__ DK_INL_ROWSJV void rows_jv(S& s, const float* v, float* out, int lane) {
  con_jv(s, v, lane);
  for (int r = lane; r < s.nefc; r += 32) {
    int kind = s.efc_kind[r], a = s.efc_a[r], b = s.efc_b[r];
    float val;
    if (kind == 0) val = v[a];
    else if (kind == 1) val = b * v[a];
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
  int ncon = uniform_int(min(s.ncon, MAXCON), 7);
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
        if (s.efc_a[r] != i) continue;
        acc[t] += r < NFLOSS ? f[r] : s.efc_b[r] * f[r];  // frictionloss rows (kind 0) first, then limits (kind 1)
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
      if (s.efc_a[r] != i) continue;
      acc += r < NFLOSS ? f[r] : s.efc_b[r] * f[r];  // frictionloss rows (kind 0) first, then limits (kind 1)
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
// as bits; lane j returns rows 32 j .. 32 j + 31 (0 past nefc and on lanes j >= ceil(nefc / 32))
template <class S>
__device__ __noinline__ unsigned zone_word(const S& s, int lane) {
  unsigned z = 0u;
  const int nefc = uniform_int(s.nefc, 8);
  for (int r0 = 0; r0 < nefc; r0 += 32) {
    int r = r0 + lane;
    unsigned b = __ballot_sync(0xffffffffu, r < nefc && row_zone(s, r, s.efc_jar[r]) == 1);
    if (lane == r0 >> 5) z = b;
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
__device__ __forceinline__ unsigned long long support_mask(unsigned x) {
#if DOF_BODY_ROOT6
  return (unsigned long long)(x >> 2) << 6 | ((x & 2u) ? 0x3full : 0ull);
#else
  unsigned long long m = 0ull;
  for (int i = 0; i < NV; i++)
    if ((x >> c_dofbody[i]) & 1u) m |= 1ull << i;
  return m;
#endif
}
// the Newton Hessian into s.L (see below); world, grp0, grp1: the contacts against the world and their groups by body
// (substep_post)
template <class S>
__device__ __forceinline__ void hessian(S& s, int ncon, unsigned long long world, unsigned grp0, unsigned grp1,
                                        int lane) {
  // H = M + J' D J on the quadratic rows (lower triangle into L), curvatures D from the rows' zones at jar. The pyramid
  // rows of a contact, u_k = n +- mu t1, n +- mu t2, add C_nn n n' + C_n1 (n t1' + t1 n') + C_11 t1 t1' + (same for t2)
  // = J' C J with C = [C_nn C_n1 C_n2; C_n1 C_11 0; C_n2 0 C_22] (only C_nn for condim 1) and J the frame rows
  float mv[(NTRI + 31) / 32];  // L = M: all loads first (entry e by lane e mod 32)
#pragma unroll
  for (int k = 0; k < (NTRI + 31) / 32; k++) if (lane + 32 * k < NTRI) mv[k] = s.M[lane + 32 * k];
  int ha[HSETS][3], hb[HSETS], hthr[HSETS][3];  // this lane's support pairs (ha, hb) of one column hb per set, the
#pragma unroll                                   // same for every contact, valid when nd > hthr (step.hess_columns)
  for (int k = 0; k < HSETS; k++)
#pragma unroll
    for (int t = 0; t < 3; t++) {
      unsigned e = c_hcol[(3 * k + t) * 32 + lane];
      ha[k][t] = e & 255; hb[k] = (e >> 8) & 255; hthr[k][t] = e >> 16;
    }
  unsigned long long act = 0ull;  // contacts with a quadratic row
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
    act |= (unsigned long long)__ballot_sync(0xffffffffu, on) << c0;
  }
  SUBPROF(27);
#pragma unroll
  for (int k = 0; k < (NTRI + 31) / 32; k++) if (lane + 32 * k < NTRI) s.L[lane + 32 * k] = mv[k];
  const int ndof = uniform_int(s.nefc_dof, 8);
#if DK_HESS_DOFPAR
  if (NFLOSS == 0) {  // limit rows only: one per dof, distinct diagonal entries, each added by its row's lane
    __syncwarp();     // (after the copy)
    for (int r = lane; r < ndof; r += 32) {
      float ds, dd, c;
      row_eval(s, r, s.efc_jar[r], &ds, &dd, &c);
      s.L[TRI(s.efc_a[r], s.efc_a[r])] += dd;
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
      if ((ek & 31) == lane) s.L[ek] += dk;
    }
  }
  __syncwarp();
  SUBPROF(28);
  // the contacts on the lower triangles of their supports (dofs ascending: dof[a] >= dof[b]): first the world
  // contacts, a group (one body, one support) at a time, then each contact between two bodies. Per lane and set: the
  // pairs (a, b) of one column b, H_ab = J_a' G_b with G_b = C J_b once per contact, summed over the group before the
  // update of L
  for (unsigned long long todo = act & world, rest = act & ~world; todo | rest;) {
    bool w = todo != 0ull;
    unsigned long long m64 = w ? todo : rest;
    int c0 = (unsigned)m64 ? 0 : 32;  // contacts c0 + i, i in gm (32-bit masks: a cheap __ffs)
    unsigned m = (unsigned)(m64 >> c0);
    int c = __ffs(m) - 1;
    unsigned gw = __ballot_sync(0xffffffffu, ((c0 ? grp1 : grp0) >> c) & 1u), gm = w ? m & gw : 1u << c;  // c's group
    if (w) todo &= ~((unsigned long long)gm << c0);
    else {
      rest &= ~((unsigned long long)gm << c0);
      __syncwarp();  // between two bodies: fenced
    }
    int nd = s.con_nd[c0 + c];
    const unsigned char* dof = s.con_dof[c0 + c];
    constexpr int hlo[HSETS] = HSET_LO;
#pragma unroll
    for (int k = 0; k < HSETS; k++) {
      if (nd <= hlo[k]) break;  // no rows of this set in the support
      int db = dof[hb[k]], he[3];
      float hv[3], lv[3];
#pragma unroll
      for (int t = 0; t < 3; t++) {  // the entries of one support are distinct: loads first, stores after the sum
        int da = dof[ha[k][t]];
        he[t] = da * (da + 1) / 2 + db;
        hv[t] = 0.f;
        if (nd > hthr[k][t]) lv[t] = s.L[he[t]];
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
        float g0 = C[0] * j0 + C[1] * j1 + C[3] * j2, g1 = C[1] * j0 + C[2] * j1, g2 = C[3] * j0 + C[4] * j2;
#pragma unroll
        for (int t = 0; t < 3; t++) {
          int a = ha[k][t];
          hv[t] = fmaf(Jb[a], g2, fmaf(Ja[a], g1, fmaf(Jn[a], g0, hv[t])));
        }
      }
#pragma unroll
      for (int t = 0; t < 3; t++) if (nd > hthr[k][t]) s.L[he[t]] = lv[t] + hv[t];
    }
  }
}

// one substep for the world in s (qpos, qvel, qacc warm start, com, fric set); ctrl: this world's NV-6 setpoints.
// Two halves (substep_pre: smooth dynamics, collision, Jacobians, rows; substep_post: Newton, implicitfast) so that they can
// also run as separate kernels with the state in global memory (engine_step_check.py --split).
// rec: this world's tape record of the substep (tape.py), or null (no tape); the same
// pointer goes to both halves. The record is written at five points: the input state (here), the actuator clamp mask,
// the contact table (end of substep_pre), the final zones with their Hessian factor L_H, and after the implicitfast
// solve L_Q, a*, a_adv and the header (substep_post). The forward results do not depend on rec.
template <class S>
__device__ void substep_pre(S& s, const float* ctrl, int lane, float* rec) {
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
  }
  for (int i = lane; i < NV; i += 32) s.zero[i] = 0.f;
  for (int i = lane; i < NTRI; i += 32) s.M[i] = 0.f;
  PROF_START();
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
        int c = c_child[3 * b + i];
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
  __syncwarp();
  SUBPROF(17);
  for (int i0 = 0; i0 < NV; i0 += 32) {  // qfrc_smooth = clamped actuator force - bias
    int i = i0 + lane;
    bool clamp = false;
    if (i < NV) {
      float act = 0.f;
      if (i >= 6) {
        act = c_kp[i] * (ctrl[i - 6] - s.qpos[i + 1]) - c_kv[i] * s.qvel[i];
        act = fminf(fmaxf(act, -c_frclim[i]), c_frclim[i]);
        clamp = act <= -c_frclim[i] || act >= c_frclim[i];  // at the limit: MuJoCo Warp's actuator_lam test
      }
      s.qsm[i] = act - s.qsm[i];
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
        }
      }
      unsigned bal = __ballot_sync(0xffffffffu, keep);
      if (keep) s.cand[ncand + __popc(bal & lanes_below)] = (unsigned short)pr;
      ncand += __popc(bal);
    }
    __syncwarp();
    SUBPROF(29);
    for (int i = lane; i < ncand; i += 32) collide_pair(s, s.cand[i]);
  }
  PROF(2);
  if (lane == 0 && s.ncon > MAXCON) s.unsupported |= 4;
  int ncon = uniform_int(min(s.ncon, MAXCON), 7);
  {  // contact Jacobians: frame rows of J(body2, pos) - J(body1, pos) on the dofs on exactly one chain, packed in
     // ascending order. Lane l keeps the motion subspaces (w, v about O) of dofs l and l + 32, and the two bodies'
     // ancestor masks of contacts l and l + 32 (shuffled to all lanes per contact)
    static_assert(NV <= 64 && MAXCON <= 64, "two dofs and two contacts per lane");
    float dw[2][3], dv[2][3];
    int dbody[2];
    unsigned am1[2], am2[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      int i = lane + 32 * h, c = lane + 32 * h;
      dbody[h] = i < NV ? c_dofbody[i] : 0;
      if (dbody[h] == 1) dof_motion(1, i, q1, O, O, dw[h], dv[h]);
      else if (dbody[h] >= 2)
        for (int t = 0; t < 3; t++) { dw[h][t] = s.W[dbody[h]][t]; dv[h][t] = s.V[dbody[h]][t]; }
      int pr = c < ncon ? s.con_pair[c] : 0;
      am1[h] = c_ancmask[c_gbody[PG1(pr)]];
      am2[h] = c_ancmask[c_gbody[PG2(pr)]];
    }
    for (int c = 0; c < ncon; c++) {
      unsigned m1 = __shfl_sync(0xffffffffu, c < 32 ? am1[0] : am1[1], c & 31);
      unsigned m2 = __shfl_sync(0xffffffffu, c < 32 ? am2[0] : am2[1], c & 31);
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
    for (int h = 0; h < 2; h++)
      if (lane + 32 * h < ncon) s.con_mask[lane + 32 * h] = support_mask(am1[h] ^ am2[h]);  // (the ballots' bits)
  }
  PROF(3);

  // constraint rows: dof frictionloss rows (dof order), active hinge limits (dof order), then per contact (contact order)
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
  for (int i0 = 0; i0 < NV; i0 += 32) {
    int i = min(i0 + lane, NV - 1);  // clamped so that the loads need no branch
    float dmin, dmax;
    limit_dists(i, s.qpos[i + 1], &dmin, &dmax);
    bool on = i0 + lane < NV && i >= 6 && c_jlim[i] != 0.f && fminf(dmin, dmax) < 0.f;
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
      if (!con) {  // dof row j: frictionloss (at 0) or limit
        i = s.efc_a[j];
        bool fl = s.efc_kind[j] == 0;
        float dmin, dmax;
        limit_dists(i, s.qpos[i + 1], &dmin, &dmax);
        iw = c_dinvw[i];
        pos = fl ? 0.f : fminf(dmin, dmax);
        sref = fl ? c_dsolref : c_jsolref; simp = fl ? c_dsolimp : c_jsolimp;
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
        s.efc_D[j] = D; s.efc_aref[j] = -k * imp * pos - bb * (s.efc_b[j] * s.qvel[i]);
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
    s.nefc_dof = ndof;  // dof rows (frictionloss, limits) come first; contact rows follow
  }
  PROF(4);
  if (rec)  // tape: the contact table in contact (= row) order, two 16-bit entries pair | slot << 9 per word
    for (int i = lane; i < (MAXCON + 1) / 2; i += 32) {
      unsigned e0 = 2 * i < ncon ? (unsigned)s.con_pair[2 * i] | (unsigned)s.con_slot[2 * i] << 9 : TAPE_CON_NONE;
      unsigned e1 = 2 * i + 1 < ncon ? (unsigned)s.con_pair[2 * i + 1] | (unsigned)s.con_slot[2 * i + 1] << 9
                                     : TAPE_CON_NONE;
      __stcs(rec + OFF_CON + i, __uint_as_float(e0 | e1 << 16));
    }
}

template <class S>
__device__ void substep_post(S& s, int iterations, int lane, int* niter_out, float* rec) {
  const float h = c_misc[0];
  int ncon = uniform_int(min(s.ncon, MAXCON), 7);
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
  // one update of L (grp0 / grp1: lane masks of the contacts 0-31 / 32-63 on the same body). A contact between two
  // bodies (support: the dofs on exactly one chain) is fenced on both sides.
  static_assert(MAXCON <= 64 && KDOF <= 32, "contact bit masks; hess_columns fits 32 lanes");
  unsigned long long world = 0ull;
  unsigned grp0 = 0u, grp1 = 0u;
  for (int c0 = 0; c0 < ncon; c0 += 32) {
    int c = c0 + lane, pr = c < ncon ? s.con_pair[c] : 0, b1 = c_gbody[PG1(pr)], b2 = c_gbody[PG2(pr)];
    bool w = c < ncon && (b1 == 0 || b2 == 0);
    world |= (unsigned long long)__ballot_sync(0xffffffffu, w) << c0;
    unsigned g = __match_any_sync(0xffffffffu, w ? b1 + b2 : -1 - lane);
    if (c0 == 0) grp0 = g; else grp1 = g;
  }
  // tape (rec): L_H must be the Hessian factor at the final zones. The last factorization was assembled at the jar
  // before the last step; when that step moved a row to another zone (chg, tested where jar is updated), or no step was
  // made, the loop runs once more without a step (refact): assembly and factorization at the final jar, so the tape
  // needs no second copy of the assembly code. Only the scratch s.L, s.p and s.con_C change, so the forward results are
  // the same with and without the tape
  int it = 0;
  bool conv = false, zchanged = rec && iterations <= 0, refact = zchanged, zsame = false;
  while (it < iterations || refact) {
#if NEWTON_REUSE
    if (zsame && !refact) {  // no row changed its zone in the last step: H and its factor in s.L are the same
      for (int i = lane; i < NV; i += 32) s.p[i] = -s.grad[i];
      __syncwarp();
      chol_resolve(s.L, s.p, lane, s.cbuf);
    } else
#endif
    {
      hessian(s, ncon, world, grp0, grp1, lane);
      PROF(7);
      for (int i = lane; i < NV; i += 32) s.p[i] = -s.grad[i];
      __syncwarp();
      CHOL_SOLVE(s.L, s.L, s.p, lane, s.cbuf);
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
    unsigned zprev = 0u;
    bool newton = false;
    for (int k = 0; k <= 40; k++) {
      float fd = 0.f, fdd = 0.f;
      unsigned z = 0u;
#pragma unroll
      for (int t = 0; t < RT; t++) {
        if (32 * t >= nefc) break;
        float ds, dd, jv = rj[t];
        z |= (unsigned)row_piece(lane + 32 * t, rx[t] + alpha * jv, rD[t], rF[t], &ds, &dd) << (2 * t);
        fd += ds * jv; fdd += dd * jv * jv;
      }
      if (__all_sync(0xffffffffu, newton && z == zprev)) break;  // the last Newton step kept every row on its piece
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
      zprev = z;
      alpha = nx;
    }
    // move along p: qacc, M qacc and J qacc - aref are linear in alpha; with the new cost (as total_cost) and the rows'
    // forces for newton_grad
    float g = 0.f;
    for (int i = lane; i < NV; i += 32) {
      float qa = s.qacc[i] + alpha * s.p[i], ma = s.Ma[i] + alpha * s.Mp[i];
      s.qacc[i] = qa; s.Ma[i] = ma;
      g += 0.5f * (ma - s.qsm[i]) * (qa - s.qacc0[i]);
    }
    bool chg = false;  // a row of this lane changes its zone (tape: the Hessian factored above is then out of date)
#pragma unroll
    for (int t = 0; t < RT; t++) {
      int r = lane + 32 * t;
      if (32 * t >= nefc) break;
      if (r < nefc) {
        float xn = rx[t] + alpha * rj[t], ds, dd;
        int z0 = row_piece(r, rx[t], rD[t], rF[t], &ds, &dd), z1 = row_piece(r, xn, rD[t], rF[t], &ds, &dd);
        chg |= z0 != z1;
        g += row_cost(z1, xn, rD[t], rF[t]);
        s.efc_jar[r] = xn;
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
    unsigned z = zone_word(s, lane);
    if (lane < (NEFC + 31) / 32) __stcs(rec + OFF_ZONE + lane, __uint_as_float(z));
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
  for (int i = lane; i < NV; i += 32) { s.L[TRI(i, i)] += h * c_kv[i]; s.tmp[i] = s.Ma[i]; }
  __syncwarp();
  CHOL_SOLVE(s.L, s.L, s.tmp, lane, s.cbuf);
  if (rec) {  // tape: L_Q = chol(M + h kv), a* and a_adv (qvel' = qvel + h a_adv), header
    tape_put(rec + OFF_LQ, s.L, NTRI, REC_WORDS - OFF_LQ, lane);
    tape_put(rec + OFF_QACC, s.qacc, NV, NV, lane);
    tape_put(rec + OFF_QADV, s.tmp, NV, OFF_LH - OFF_QADV, lane);
    if (lane == 0) {
      float nu = __int_as_float(s.min_nu_bits);
      unsigned fl = (unsigned)(s.unsupported & 31) | (zchanged ? TAPE_REFACT : 0u) | (conv ? 0u : TAPE_NONCONV)
                    | (nu < TAPE_SINGULAR_NU ? TAPE_SINGULAR : 0u);
#ifdef DK_TAPE_VERIFY
      if (vdiff) fl |= TAPE_VERIFY_DIFF;
#endif
      unsigned hd = (unsigned)ncon | (unsigned)s.nefc << 8 | (unsigned)(s.nefc_dof - NFLOSS) << 16
                    | (unsigned)min(it, 255) << 24;
      __stcs(rec + OFF_HDR, __uint_as_float(hd));
      __stcs(rec + OFF_FLAGS, __uint_as_float(fl));
      __stcs(rec + OFF_MIN_NU, nu);
      __stcs(rec + OFF_RESERVED, 0.f);
    }
  }
  for (int i = lane; i < NV; i += 32) s.qvel[i] += h * s.tmp[i];
  __syncwarp();
  if (lane == 0) integrate_root(s.qpos, s.qvel, h);
  for (int i = 6 + lane; i < NV; i += 32) s.qpos[i + 1] += h * s.qvel[i];
  __syncwarp();
  PROF(11);
}

template <class S>
__device__ __forceinline__ void substep(S& s, const float* ctrl, int iterations, int lane, int* niter_out,
                                        float* rec) {
  substep_pre(s, ctrl, lane, rec);
  substep_post(s, iterations, lane, niter_out, rec);
}

// The split forward (host.py: k_pre, k_post, dk_forward_split): substep_pre and substep_post as two kernels, each
// with its own state (SimP, SimN: only what that half uses, so more worlds fit an SM, and each kernel's code is about
// half of k_substep's). Their members have struct Sim's names, so the device code above runs unchanged on all three;
// arrays that live in global memory are pointers there (struct Hand, one per world: what substep_pre hands to
// substep_post).
// substep_post runs on SimNT<SPLIT_NCON, SPLIT_NEFC> in shared memory when the world's contacts and rows fit (corpora:
// at most 31 contacts and 122 rows), else on a full-capacity SimN in Hand (global memory, the same code).
#ifndef SPLIT_NCON
#define SPLIT_NCON (MAXCON < 32 ? MAXCON : 32)  // k_post's shared-memory capacity: contacts
#endif
#ifndef SPLIT_NEFC
#define SPLIT_NEFC (NEFC < 128 ? NEFC : 128)  // and constraint rows
#endif
template <int NC, int NR>
struct SimNT {  // substep_post: Newton, implicitfast; capacity NC contacts, NR rows. Members whose lifetimes do not
  float M[NTRI], L[NTRI];  // overlap share storage (unions), the read-only ones the loop seldom reads stay in Hand
  float qacc[NV], Ma[NV];
  union { float grad[NV], p[NV]; };  // grad is last read by p = -grad (in place)
  union { float Mp[NV], tmp[NV]; };  // Mp: line search to update; tmp: J'f (newton_grad), start's qacc, implicitfast
  float efc_D[NR], efc_jar[NR];
  unsigned char efc_kind[NR], efc_a[NR];
  signed char efc_b[NR];
  int ncon, nefc, unsupported, min_nu_bits, nefc_dof;
  union {
    float con_C[NC][5];  // the Hessian assembly only (J p, the forces and their per-contact sums are dead there)
    alignas(16) float cbuf[72];  // CHOL_SOLVE's scratch: the factorizations (con_C is dead from there on)
    struct {
      union { float con_dot[NC][3], con_f[NC][3]; };
      float efc_jv[NR];
    };
  };
  unsigned short con_row[NC];
  unsigned char con_dim[NC], con_nrow[NC], con_nd[NC];
  float *qpos, *qvel;  // the kernel's outputs (implicitfast only)
  const float *qsm, *qacc0, *con_mu, *efc_aref;  // in Hand
  const int* con_pair;
  const float (*con_J)[3][KDOF];
  const unsigned char (*con_dof)[KDOF];
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
  float M[NTRI], qsm[NV], qacc0[NV];
  float efc_D[NEFC], efc_aref[NEFC];
  alignas(16) float con_J[MAXCON][3][KDOF];  // (16-byte aligned: con_jv's 128-bit loads)
  float con_mu[MAXCON];
  int con_pair[MAXCON], con_dim[MAXCON], con_row[MAXCON], con_nrow[MAXCON], con_nd[MAXCON];
  int ncon, nefc, nefc_dof, unsupported, min_nu_bits;
  unsigned char efc_kind[NEFC], efc_a[NEFC];
  signed char efc_b[NEFC];
  alignas(16) unsigned char con_dof[MAXCON][KDOF];
  unsigned long long con_mask[MAXCON];
  float con_dist[MAXCON], con_pos[MAXCON][3], con_frame[MAXCON][9];  // substep_pre's (collision, Jacobians, rows)
  SimNfull big;  // k_post's state of a world whose contacts or rows exceed SimN's capacity
};

struct SimP {  // substep_pre: smooth dynamics, qacc_smooth, collision, contact Jacobians (into Hand), rows (into Hand)
  float Q[NB][4], X[NB][3];
  float W[NB][3], V[NB][3];  // body velocities, then (body forces to contact Jacobians) the hinge motion subspaces
  union {  // the smooth phase's body quantities, then qacc_smooth's factor, then the collision's geoms and candidates
    struct { float AL[NB][3], A[NB][3], F[NB][3], N[NB][3], J[NB][10]; };
    struct { float L[NTRI]; alignas(16) float cbuf[72]; };  // CHOL_SOLVE's scratch
    struct { float gx[NGEOMC][3], gq[NGEOMC][4], con_dot[MAXCON][3]; unsigned short cand[NPAIR]; };
  };
  float* M;  // in Hand (the stores of the CRB, the loads of qacc_smooth's factorization)
  float qpos[NQ], qvel[NV], zero[NV], qsm[NV], qacc0[NV];
  float com[3], fric;
  int ncon, nefc, unsupported, min_nu_bits, nefc_dof;
  int con_pair[MAXCON], con_dim[MAXCON], con_row[MAXCON], con_nrow[MAXCON], con_nd[MAXCON];
  float con_mu[MAXCON];
  unsigned char con_slot[MAXCON];
  float *con_dist, (*con_pos)[3], (*con_frame)[9];  // in Hand
  float (*con_J)[3][KDOF];
  unsigned char (*con_dof)[KDOF];
  unsigned long long* con_mask;
  unsigned char *efc_kind, *efc_a;
  signed char* efc_b;
  float *efc_D, *efc_aref;
#ifdef DK_PROFILE
  long long prof[NPROF], prof_t, prof_u;
#endif
};
"""



# ---- terrain builds (nablaengine.terrain; work in progress): terrain_source() inserts TERRAIN_DEVICE and the hooks
# below into STEP_DEVICE + STEP_KERNEL. A build without terrain compiles the unchanged text (the G1 identity gate).
TERRAIN_DEVICE = r"""
// ---- terrain (nablaengine.terrain): static boxes and heightfields in global memory, shared by every world ----
// set by <unit>_set_terrain (terrain.pack's arrays on the device); grid == nullptr: no terrain uploaded
struct TerrainPtrs {
  const float* box;    // (nbox, 15): pos, rot (row major), half sizes
  const float* hfh;    // (nhf, 16): pos, rot, MuJoCo hfield_size (x, y radius, z scale, z bottom)
  const int* hfi;      // (nhf, 4): nrow, ncol, data address, 0
  const float* hfd;    // heights, MuJoCo's normalised data (times hfield_size z)
  const float* grid;   // x0, y0, 1 / cell, nx, ny, nbox
  const int* cstart;   // (nx ny + 1) cell offsets into citems
  const int* citems;   // element indices: box i, or nbox + heightfield j
  const int* ecell;    // (n, 2) each element's first cell
};
__device__ TerrainPtrs g_ter;
// box e of the terrain
__device__ __forceinline__ const float* ter_box(int e) { return g_ter.box + 15 * e; }

// height and upward unit normal of the heightfield triangle below local (x, y) (cells split along MuJoCo's
// (c, r+1)-(c+1, r) diagonal); false outside the grid
__device__ __forceinline__ bool hfield_surface(const float* hs, const float* hd, int nrow, int ncol, float x, float y,
                                               float* h, float* n) {
  float dx = 2.f * hs[0] / float(ncol - 1), dy = 2.f * hs[1] / float(nrow - 1);
  float fc = (x + hs[0]) / dx, fr = (y + hs[1]) / dy;
  if (fc < 0.f || fr < 0.f || fc > float(ncol - 1) || fr > float(nrow - 1)) return false;
  int c0 = min(int(floorf(fc)), ncol - 2), r0 = min(int(floorf(fr)), nrow - 2);
  float u = fc - float(c0), v = fr - float(r0);
  float h00 = hd[r0 * ncol + c0] * hs[2], h10 = hd[r0 * ncol + c0 + 1] * hs[2];
  float h01 = hd[(r0 + 1) * ncol + c0] * hs[2], h11 = hd[(r0 + 1) * ncol + c0 + 1] * hs[2];
  float t[3];
  if (u + v <= 1.f) {
    *h = h00 + u * (h10 - h00) + v * (h01 - h00);
    t[0] = -dy * (h10 - h00); t[1] = -dx * (h01 - h00); t[2] = dx * dy;
  } else {
    *h = h11 + (1.f - u) * (h01 - h11) + (1.f - v) * (h10 - h11);
    t[0] = dy * (h01 - h11); t[1] = dx * (h10 - h11); t[2] = dx * dy;
  }
  float il = rsqrtf(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
  for (int k = 0; k < 3; k++) n[k] = t[k] * il;
  return true;
}

// The terrain contacts' geometry, shared by the forward and the adjoint's replay (adjoint_contact.TERRAIN_DEVICE):
// one __noinline__ copy of each, so the replay computes the forward's bits. A contact's tape slot is k | element << 8
// (k: capsule_box's candidate, or the heightfield sample point; element: box i, or nbox + heightfield j)

// sphere (centre sp, radius r) against terrain box b (15 words): sphere_box with the normal from the box towards the
// sphere
__device__ __noinline__ void terrain_sphere_box(const float* sp, float r, const float* b, float* dist, float* pos,
                                                float* n) {
  sphere_box(sp, r, b, b + 3, b + 12, dist, pos, n);
  for (int k = 0; k < 3; k++) n[k] = -n[k];
}

// centre of capsule_box's candidate sphere at segment parameter seg (capbox_segs), as capsule_box computes it
__device__ __noinline__ void capbox_point(const float* cp, const float* axisw, float hl, const float* b, float seg,
                                          float* ce) {
  const float *bp = b, *R = b + 3;
  float d[3] = {cp[0] - bp[0], cp[1] - bp[1], cp[2] - bp[2]}, pos[3], axis[3], lc[3], t[3];
  mtv3(R, d, pos);
  mtv3(R, axisw, axis);
  for (int k = 0; k < 3; k++) lc[k] = pos[k] + axis[k] * hl * seg;
  mv3(R, lc, t);
  for (int k = 0; k < 3; k++) ce[k] = bp[k] + t[k];
}

// capsule_box's candidate spheres for a terrain box b (capbox_segs) with a fast path for the common case, a capsule
// resting on a box's top: both ends above the box's local +z face and inside its footprint by more than 1 mm. There
// capsule_box's search ends with the lower end (the -1 end on a tie) against that face (no edge is nearer: they are
// at least 1 mm away horizontally), and the second candidate is the other end (its exits from the face's footprint
// lie beyond the segment): segs = (lower, other), the same values as capbox_segs without its edge search
__device__ __noinline__ int terrain_capbox_segs(const float* cp, const float* axisw, float hl, const float* b,
                                                float* segs) {
  const float *bp = b, *R = b + 3, *bs = b + 12;
  float d[3] = {cp[0] - bp[0], cp[1] - bp[1], cp[2] - bp[2]}, pos[3], axis[3], dz[2];
  mtv3(R, d, pos);
  mtv3(R, axisw, axis);
  bool fast = true;
  for (int e = 0; e < 2 && fast; e++) {
    float tip[3];
    for (int j = 0; j < 3; j++) tip[j] = pos[j] + float(2 * e - 1) * (axis[j] * hl);
    fast = fabsf(tip[0]) <= bs[0] - 1e-3f && fabsf(tip[1]) <= bs[1] - 1e-3f && tip[2] > bs[2];
    dz[e] = (bs[2] - tip[2]) * (bs[2] - tip[2]);
  }
  if (!fast) return capbox_segs(cp, axisw, hl, bp, R, bs, segs);
  segs[0] = dz[1] < dz[0] ? 1.f : -1.f;
  segs[1] = -segs[0];
  return 2;
}

// robot geom g (sphere or capsule, pose gx / gq) against terrain box e
template <class S>
__device__ void terrain_box(S& s, int pair, int g, int e) {
  const float* b = ter_box(e);
  const float* p = s.gx[g];
  float d[3] = {p[0] - b[0], p[1] - b[1], p[2] - b[2]};
  float rb = c_grbound[g] + sqrtf(b[12] * b[12] + b[13] * b[13] + b[14] * b[14]);
  if (dot3(d, d) > rb * rb) return;
  const float* sz = c_gsize + 3 * g;
  float dist, pos[3], n[3], fr[9];
  if (c_gtype[g] == 1) {
    terrain_sphere_box(p, sz[0], b, &dist, pos, n);
    if (dist < 0.f) { make_frame(n, fr); add_contact(s, pair, e << 8, dist, pos, fr); }
  } else if (c_gtype[g] == 2) {
    float axis[3], segs[2], ce[3];
    zaxis_rn(s.gq[g], axis);
    const int ns = terrain_capbox_segs(p, axis, sz[1], b, segs);
    for (int k = 0; k < ns; k++) {
      capbox_point(p, axis, sz[1], b, segs[k], ce);
      terrain_sphere_box(ce, sz[0], b, &dist, pos, n);
      if (dist < 0.f) { make_frame(n, fr); add_contact(s, pair, k | e << 8, dist, pos, fr); }
    }
  }
}

// sample point k of robot geom g (type gt, size sz, pose gp / gq) against terrain heightfield j: (sphere: the centre;
// capsule: TERRAIN_CAP_PTS points w = -1 .. 1 along the axis) against the plane of the heightfield triangle below it
// (the analytic mode): dist = n_z (q_z - h) - r, pos = q - (r + dist / 2) n, normal nw (world,
// from the terrain towards the geom); false outside the grid
__device__ __noinline__ bool terrain_hfield_point(int j, int gt, const float* sz, const float* gp, const float* gq,
                                                  int k, float* dist, float* pos, float* nw) {
  const float* hh = g_ter.hfh + 16 * j;
  const int* hi = g_ter.hfi + 4 * j;
  const float* R = hh + 3;
  float d[3] = {gp[0] - hh[0], gp[1] - hh[1], gp[2] - hh[2]}, q[3], al[3] = {0.f, 0.f, 0.f}, w = 0.f;
  mtv3(R, d, q);  // the geom centre in the heightfield's frame
  if (gt == 2) {
    float axis[3];
    zaxis_rn(gq, axis);
    mtv3(R, axis, al);
    w = TERRAIN_CAP_PTS > 1 ? -1.f + float(k) * 2.f / float(TERRAIN_CAP_PTS - 1) : 0.f;
  }
  for (int i = 0; i < 3; i++) q[i] += w * sz[1] * al[i];
  float h, n[3];
  if (!hfield_surface(hh + 12, g_ter.hfd + hi[2], hi[0], hi[1], q[0], q[1], &h, n)) return false;
  *dist = (q[2] - h) * n[2] - sz[0];
  float posl[3];
  for (int i = 0; i < 3; i++) posl[i] = q[i] - (sz[0] + 0.5f * (*dist)) * n[i];
  mv3(R, posl, pos);
  for (int i = 0; i < 3; i++) pos[i] += hh[i];
  mv3(R, n, nw);
  return true;
}

#if TERRAIN_HF_NATIVE
// ---- native heightfield contacts (TerrainConfig.mode = "native"): MuJoCo's prisms (mjc_ConvexHField / MuJoCo Warp's
// ccd_hfield_kernel: the cells under the geom's bounding box, two triangular prisms per cell from the triangle down to
// the heightfield's bottom), each measured exactly in closed form instead of GJK / EPA, and MuJoCo's choice of at most 4
// contacts per geom and heightfield (the deepest, the farthest from it, the farthest from their line, the fourth)
// among the penetrating candidates. Per prism, the geom's core (sphere: its centre; capsule: its axis segment):
//  - disjoint from the prism: for every prism face whose face region the core's points project into, the two ends of
//    that part of the core (each one's exact distance to the prism is its height above the face); and the closest
//    pair when it lies on a prism edge or vertex (point-to-point normal);
//  - intersecting the prism: the minimum translation by the separating axes (face normals, axis x edge directions),
//    at the core's deepest point along it (the middle of the core where the core is parallel to the face).
// On a flat heightfield this gives the plane's contacts (a capsule: its two ends). Contact types for the adjoint:
// HF_FACE (constant normal), HF_POINT (normal from a fixed prism point q to the core point), HF_DEEP (constant normal)
#define HF_MAXCAND 24
#define HF_FACE 0
#define HF_POINT 1
#define HF_DEEP 2
#define HF_MIN_NEXT 1e-3f  // MuJoCo Warp's MIN_DIST_TO_NEXT_CONTACT (here a candidate below it is never taken)

struct HfCand {  // a candidate (hfield frame): the core point x = a + t d, its prism point x - (dist + r) n
  float dist, n[3], t;  // t: the core parameter (0 .. 1 from axis end -1 to +1)
  int type;
  unsigned pid;  // ((cell r ncol + c) << 1 | triangle) << 4 | the candidate's index in its prism (hf_prism's sub)
};
struct HfCon {  // a chosen contact (hfield frame): pos midway between the prism point q and the geom's surface
  float dist, pos[3], n[3], q[3], t;
  int type;
  unsigned pid;
};

// segment p1 + s d1 against segment p2 + t d2 (s, t in [0, 1]; Ericson, Real-Time Collision Detection 5.1.9)
__device__ __forceinline__ void hf_segseg(const float* p1, const float* d1, const float* p2, const float* d2, float* s,
                                          float* t) {
  float r[3] = {p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]};
  float a = dot3(d1, d1), e = dot3(d2, d2), f = dot3(d2, r);
  if (a <= 1e-20f) { *s = 0.f; *t = e > 1e-20f ? fminf(fmaxf(f / e, 0.f), 1.f) : 0.f; return; }
  float c = dot3(d1, r);
  if (e <= 1e-20f) { *t = 0.f; *s = fminf(fmaxf(-c / a, 0.f), 1.f); return; }
  float b = dot3(d1, d2), den = a * e - b * b;
  *s = den > 0.f ? fminf(fmaxf((b * f - c * e) / den, 0.f), 1.f) : 0.f;
  *t = (b * (*s) + f) / e;
  if (*t < 0.f) { *t = 0.f; *s = fminf(fmaxf(-c / a, 0.f), 1.f); }
  else if (*t > 1.f) { *t = 1.f; *s = fminf(fmaxf((b - c) / a, 0.f), 1.f); }
}

__device__ __forceinline__ void hf_add(HfCand* cand, int* nc, int type, float dist, const float* n, float t,
                                       int* over, int sub) {
  if (*nc >= HF_MAXCAND) { *over = 1; return; }
  HfCand& c = cand[(*nc)++];
  c.dist = dist; c.t = t; c.type = type; c.pid = (unsigned)sub;
  for (int k = 0; k < 3; k++) c.n[k] = n[k];
}

// a candidate's contact position (hfield frame): midway between its prism point and the geom's surface
__device__ __forceinline__ void hf_pos(const HfCand& c, const float* a, const float* d, float r, float* pos) {
  for (int k = 0; k < 3; k++) pos[k] = a[k] + c.t * d[k] - (r + 0.5f * c.dist) * c.n[k];
}

// the candidates of the core a + t d (t in [0, 1]; d = 0: a sphere) with radius r against the prism of top vertices
// v[0..2] (counterclockwise from above) and bottom zb, hfield frame; each with its index in the prism (pid's sub:
// 2 face + end, 10 the edge / vertex pair, 11 the intersection). One copy (the adjoint's replay calls it as well)
__device__ __noinline__ void hf_prism(const float (*v)[3], float zb, const float* a, const float* d, float r, HfCand* cand, int* nc,
                         int* over) {
  float P[6][3];  // 0..2 top, 3..5 bottom
  for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++) { P[i][k] = v[i][k]; P[3 + i][k] = k == 2 ? zb : v[i][k]; }
  // faces: 0 top (P0 P1 P2), 1 bottom (P3 P5 P4), 2..4 sides (Pi Pj P3+j P3+i), outward normals
  const int fv[5][4] = {{0, 1, 2, -1}, {3, 5, 4, -1}, {0, 3, 4, 1}, {1, 4, 5, 2}, {2, 5, 3, 0}};
  float fn[5][3], hA[5], hd[5];  // normal; h(t) = hA + t hd
  for (int f = 0; f < 5; f++) {
    const float *p0 = P[fv[f][0]], *p1 = P[fv[f][1]], *p2 = P[fv[f][2]];
    float e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]}, e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    cross(e1, e2, fn[f]);
    float il = rsqrtf(dot3(fn[f], fn[f]));
    for (int k = 0; k < 3; k++) fn[f][k] *= il;
    float w[3] = {a[0] - p0[0], a[1] - p0[1], a[2] - p0[2]};
    hA[f] = dot3(fn[f], w); hd[f] = dot3(fn[f], d);
  }
  // inside test of the core's ends and pierced faces: intersection -> separating axes
  bool inA = true, inB = true;
  for (int f = 0; f < 5; f++) { inA = inA && hA[f] < 0.f; inB = inB && hA[f] + hd[f] < 0.f; }
  bool hit = inA || inB;
  float ti0[5], ti1[5];  // each face's region interval of the core
  for (int f = 0; f < 5; f++) {
    float t0 = 0.f, t1 = 1.f;
    const int nv = fv[f][3] < 0 ? 3 : 4;
    for (int k = 0; k < nv && t0 <= t1; k++) {
      const float *q0 = P[fv[f][k]], *q1 = P[fv[f][(k + 1) % nv]];
      float e[3] = {q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]}, m[3], w[3] = {a[0] - q0[0], a[1] - q0[1], a[2] - q0[2]};
      cross(e, fn[f], m);  // outward in the face plane
      float g0 = dot3(m, w), g1 = dot3(m, d);  // inside: g0 + t g1 <= 0
      if (fabsf(g1) <= 1e-20f) { if (g0 > 0.f) t1 = -1.f; }
      else if (g1 > 0.f) t1 = fminf(t1, -g0 / g1);
      else t0 = fmaxf(t0, -g0 / g1);
    }
    ti0[f] = t0; ti1[f] = t1;
    if (t0 <= t1 && (hA[f] + t0 * hd[f]) * (hA[f] + t1 * hd[f]) < 0.f) hit = true;  // the core pierces face f
  }
  if (hit) {  // minimum translation: axes fn (faces) and normalize(d x edge); push the core out along +u
    const int ev[9][2] = {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {0, 3}, {1, 4}, {2, 5}};
    float best = 3.4e38f, bu[3] = {0.f, 0.f, 1.f};
    for (int ax = 0; ax < 14; ax++) {
      float u[3];
      if (ax < 5) { for (int k = 0; k < 3; k++) u[k] = fn[ax][k]; }
      else {
        const float *q0 = P[ev[ax - 5][0]], *q1 = P[ev[ax - 5][1]];
        float e[3] = {q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]};
        cross(d, e, u);
        float l2 = dot3(u, u);
        if (l2 <= 1e-12f * dot3(d, d) * dot3(e, e) || l2 <= 1e-30f) continue;
        float il = rsqrtf(l2);
        for (int k = 0; k < 3; k++) u[k] *= il;
      }
      float pmax = -3.4e38f, pmin = 3.4e38f;
      for (int i = 0; i < 6; i++) { float x = dot3(u, P[i]); pmax = fmaxf(pmax, x); pmin = fminf(pmin, x); }
      float sa = dot3(u, a), sb = sa + dot3(u, d), smin = fminf(sa, sb), smax = fmaxf(sa, sb);
      float pp = pmax - smin, pm = smax - pmin;  // translation along +u, along -u
      if (pp < best) { best = pp; for (int k = 0; k < 3; k++) bu[k] = u[k]; }
      if (ax >= 5 && pm < best) { best = pm; for (int k = 0; k < 3; k++) bu[k] = -u[k]; }
    }
    float ud = dot3(bu, d), t = ud > 1e-7f * sqrtf(dot3(d, d)) ? 0.f : (ud < -1e-7f * sqrtf(dot3(d, d)) ? 1.f : 0.5f);
    hf_add(cand, nc, HF_DEEP, -best - r, bu, t, over, 11);
    return;
  }
  // disjoint: the face-region ends in front of their face (exact distances), then the closest edge pair if it is closer
  float dmin = 3.4e38f;
  for (int f = 0; f < 5; f++) {
    if (ti0[f] > ti1[f]) continue;
    float h0 = hA[f] + ti0[f] * hd[f], h1 = hA[f] + ti1[f] * hd[f];
    if (h0 < 0.f) continue;  // behind the face (then the whole region part is: no pierce)
    for (int e = 0; e < (ti1[f] > ti0[f] ? 2 : 1); e++) {
      float t = e ? ti1[f] : ti0[f], h = e ? h1 : h0;
      dmin = fminf(dmin, h);
      if (h - r >= 0.f) continue;
      hf_add(cand, nc, HF_FACE, h - r, fn[f], t, over, 2 * f + e);
    }
  }
  const int ev[9][2] = {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {0, 3}, {1, 4}, {2, 5}};
  float bd = 3.4e38f, bx[3], bq[3], bt = 0.f;
  for (int i = 0; i < 9; i++) {
    const float *q0 = P[ev[i][0]], *q1 = P[ev[i][1]];
    float e[3] = {q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]}, s, t;
    hf_segseg(a, d, q0, e, &s, &t);
    float x[3], q[3], w[3];
    for (int k = 0; k < 3; k++) { x[k] = a[k] + s * d[k]; q[k] = q0[k] + t * e[k]; w[k] = x[k] - q[k]; }
    float dd = dot3(w, w);
    if (dd < bd) { bd = dd; bt = s; for (int k = 0; k < 3; k++) { bx[k] = x[k]; bq[k] = q[k]; } }
  }
  bd = sqrtf(bd);
  if (bd < dmin && bd - r < 0.f && bd > 0.f) {
    float n[3] = {(bx[0] - bq[0]) / bd, (bx[1] - bq[1]) / bd, (bx[2] - bq[2]) / bd};
    hf_add(cand, nc, HF_POINT, bd - r, n, bt, over, 10);
  }
}

// distance from p to triangle (a, b, c) (Ericson, Real-Time Collision Detection 5.1.5)
__device__ float hf_pt_tri(const float* p, const float* a, const float* b, const float* c);

// distance from p to the heightfield surface where it is below reach: the top triangles of cells rmin .. rmax - 1,
// cmin .. cmax - 1 (cell (r, c) between columns c, c + 1) within reach of p horizontally (a triangle nearer than reach
// is in one of them; the others are not visited), hfield frame; >= reach if none is nearer
__device__ float hf_surf_dist(const float* p, float reach, int rmin, int rmax, int cmin, int cmax, float dx, float dy,
                              const float* hs, const float* hd, int ncol) {
  float best = 3.4e38f;
  const float pad = reach + 1e-4f;
  const int c0 = max(cmin, int(floorf((p[0] - pad + hs[0]) / dx))), c1 = min(cmax - 1, int(floorf((p[0] + pad + hs[0]) / dx)));
  const int r0 = max(rmin, int(floorf((p[1] - pad + hs[1]) / dy))), r1 = min(rmax - 1, int(floorf((p[1] + pad + hs[1]) / dy)));
  for (int rr = r0; rr <= r1; rr++)
    for (int cc = c0; cc <= c1; cc++) {
      float v00[3] = {dx * float(cc) - hs[0], dy * float(rr) - hs[1], hd[rr * ncol + cc] * hs[2]};
      float v01[3] = {v00[0], dy * float(rr + 1) - hs[1], hd[(rr + 1) * ncol + cc] * hs[2]};
      float v10[3] = {dx * float(cc + 1) - hs[0], v00[1], hd[rr * ncol + cc + 1] * hs[2]};
      float v11[3] = {v10[0], v01[1], hd[(rr + 1) * ncol + cc + 1] * hs[2]};
      best = fminf(best, fminf(hf_pt_tri(p, v00, v01, v10), hf_pt_tri(p, v01, v10, v11)));
    }
  return best;
}

__device__ float hf_pt_tri(const float* p, const float* a, const float* b, const float* c) {
  float ab[3], ac[3], ap[3], q[3];
  for (int k = 0; k < 3; k++) { ab[k] = b[k] - a[k]; ac[k] = c[k] - a[k]; ap[k] = p[k] - a[k]; }
  float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
  if (d1 <= 0.f && d2 <= 0.f) { for (int k = 0; k < 3; k++) q[k] = a[k]; }
  else {
    float bp[3] = {p[0] - b[0], p[1] - b[1], p[2] - b[2]}, d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    float cp[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]}, d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    float vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
    if (d3 >= 0.f && d4 <= d3) { for (int k = 0; k < 3; k++) q[k] = b[k]; }
    else if (d6 >= 0.f && d5 <= d6) { for (int k = 0; k < 3; k++) q[k] = c[k]; }
    else if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) { float v = d1 / (d1 - d3); for (int k = 0; k < 3; k++) q[k] = a[k] + v * ab[k]; }
    else if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) { float w = d2 / (d2 - d6); for (int k = 0; k < 3; k++) q[k] = a[k] + w * ac[k]; }
    else if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
      float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      for (int k = 0; k < 3; k++) q[k] = b[k] + w * (c[k] - b[k]);
    } else {
      float den = 1.f / (va + vb + vc), v = vb * den, w = vc * den;
      for (int k = 0; k < 3; k++) q[k] = a[k] + v * ab[k] + w * ac[k];
    }
  }
  float w[3] = {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
  return sqrtf(dot3(w, w));
}

// add_contact (step.py) returning the contact's index (-1 beyond MAXCON): the native contacts tape their candidate
template <class S>
__device__ __forceinline__ int add_contact_aux(S& s, int pair, int slot, float dist, const float* pos,
                                               const float* frame) {
  int c = atomicAdd(&s.ncon, 1);
  if (c >= MAXCON) return -1;
  s.con_pair[c] = pair;
  s.con_slot[c] = (unsigned char)(slot & 255);
  s.con_elem[c] = (unsigned short)(slot >> 8);
  s.con_dim[c] = PDIM(pair);
  s.con_dist[c] = dist;
  for (int k = 0; k < 3; k++) s.con_pos[c][k] = pos[k];
  for (int k = 0; k < 9; k++) s.con_frame[c][k] = frame[k];
  s.con_mu[c] = PFOOT(pair) ? s.fric : PMU(pair);
  return c;
}

// the geom's core in the frame of heightfield j: a + t d (t in [0, 1]), and its bounding box lo .. hi (with the radius)
__device__ __noinline__ void hf_core(int j, int gt, const float* sz, const float* gp, const float* gq, float* a,
                                     float* d, float* lo, float* hi) {
  const float* hh = g_ter.hfh + 16 * j;
  float dw[3] = {gp[0] - hh[0], gp[1] - hh[1], gp[2] - hh[2]}, c[3], ax[3] = {0.f, 0.f, 0.f};
  mtv3(hh + 3, dw, c);
  const float r = sz[0], hl = gt == 2 ? sz[1] : 0.f;
  if (gt == 2) { float aw[3]; zaxis_rn(gq, aw); mtv3(hh + 3, aw, ax); }
  for (int k = 0; k < 3; k++) {
    float e = fabsf(ax[k]) * hl + r;
    lo[k] = c[k] - e; hi[k] = c[k] + e;
    a[k] = c[k] - hl * ax[k]; d[k] = 2.f * hl * ax[k];
  }
}

// top vertices of prism (cell rr, cc: columns cc - 1 .. cc; triangle i) of heightfield j, counterclockwise from above
// (MuJoCo Warp's strip: (cc-1, rr) (cc-1, rr+1) (cc, rr), then (cc-1, rr+1) (cc, rr) (cc, rr+1)); false where the
// prism is below zmin (MuJoCo's prism height test)
__device__ __noinline__ bool hf_prism_verts(int j, int rr, int cc, int i, float zmin, float (*v)[3]) {
  const float* hs = g_ter.hfh + 16 * j + 12;
  const int* hi = g_ter.hfi + 4 * j;
  const float* hd = g_ter.hfd + hi[2];
  const int nrow = hi[0], ncol = hi[1];
  const float dx = 2.f * hs[0] / float(ncol - 1), dy = 2.f * hs[1] / float(nrow - 1);
  const int pc[2][3] = {{cc - 1, cc - 1, cc}, {cc - 1, cc, cc}}, pr[2][3] = {{rr, rr + 1, rr}, {rr + 1, rr, rr + 1}};
  for (int k = 0; k < 3; k++) {
    v[k][0] = dx * float(pc[i][k]) - hs[0];
    v[k][1] = dy * float(pr[i][k]) - hs[1];
    v[k][2] = hd[pr[i][k] * ncol + pc[i][k]] * hs[2];
  }
  if (v[0][2] < zmin && v[1][2] < zmin && v[2][2] < zmin) return false;
  float e1[3] = {v[1][0] - v[0][0], v[1][1] - v[0][1], 0.f}, e2[3] = {v[2][0] - v[0][0], v[2][1] - v[0][1], 0.f};
  if (e1[0] * e2[1] - e1[1] * e2[0] < 0.f)
    for (int k = 0; k < 3; k++) { float t = v[1][k]; v[1][k] = v[2][k]; v[2][k] = t; }
  return true;
}

// a chosen candidate as a contact (hfield frame)
__device__ __noinline__ void hf_con_of(const HfCand& c, const float* a, const float* d, float r, HfCon* o) {
  o->dist = c.dist; o->t = c.t; o->type = c.type; o->pid = c.pid;
  hf_pos(c, a, d, r, o->pos);
  for (int q = 0; q < 3; q++) { o->n[q] = c.n[q]; o->q[q] = a[q] + c.t * d[q] - (c.dist + r) * c.n[q]; }
}

// the native contacts of robot geom (type gt, size sz, pose gp / gq) and heightfield j: out[0..n-1] in MuJoCo's order
// (deepest, farthest, ...; hfield frame), returns n (0..4); *over: candidates beyond HF_MAXCAND were dropped
__device__ __noinline__ int hfield_native(int j, int gt, const float* sz, const float* gp, const float* gq,
                                          HfCon* out, int* over) {
  const float* hh = g_ter.hfh + 16 * j;
  const int* hi = g_ter.hfi + 4 * j;
  const float *hs = hh + 12, *hd = g_ter.hfd + hi[2];
  const int nrow = hi[0], ncol = hi[1];
  const float r = sz[0];
  float a[3], d[3], lo[3], hi3[3];
  hf_core(j, gt, sz, gp, gq, a, d, lo, hi3);
  *over = 0;
  if (lo[0] > hs[0] || hi3[0] < -hs[0] || lo[1] > hs[1] || hi3[1] < -hs[1] || lo[2] > hs[2] || hi3[2] < -hs[3])
    return 0;
  const float xs = 0.5f * float(ncol - 1) / hs[0], ys = 0.5f * float(nrow - 1) / hs[1];
  const int cmin = max(0, int(floorf((lo[0] + hs[0]) * xs))), cmax = min(ncol - 1, int(ceilf((hi3[0] + hs[0]) * xs)));
  const int rmin = max(0, int(floorf((lo[1] + hs[1]) * ys))), rmax = min(nrow - 1, int(ceilf((hi3[1] + hs[1]) * ys)));
  const float dx = 2.f * hs[0] / float(ncol - 1), dy = 2.f * hs[1] / float(nrow - 1);
  HfCand cand[HF_MAXCAND];
  int nc = 0;
  for (int rr = rmin; rr < rmax; rr++)
    for (int cc = cmin + 1; cc <= cmax; cc++)
      for (int i = 0; i < 2; i++) {
        float v[3][3];
        if (!hf_prism_verts(j, rr, cc, i, lo[2], v)) continue;
        const int nc0 = nc;
        hf_prism(v, -hs[3], a, d, r, cand, &nc, over);
        for (int q = nc0; q < nc; q++) cand[q].pid |= (unsigned)((rr * ncol + cc) << 1 | i) << 4;
      }
  // keep the candidates whose prism point is a closest point of the heightfield surface (the prisms' top triangles):
  // a prism's own edge next to a neighbouring triangle (an edge inside a flat or concave part of the surface) gives a
  // farther point than that triangle and is dropped; convex ridges and peaks stay. A candidate inside the core
  // (0 < t < 1: a face-region end at a triangle boundary, an edge point) stays only where the core's distance to the
  // surface has a
  // local minimum: the surface 0.5 mm along the core on either side is not closer (a ridge under the core; on a plane
  // only the core's ends stay, as the plane's contacts). Penetrating cores (HF_DEEP) stay
  const float dl = sqrtf(dot3(d, d));
  int nk = 0;
  for (int i = 0; i < nc; i++) {
    const HfCand& ci = cand[i];
    bool keep = true;
    if (ci.type != HF_DEEP) {
      float x[3], h = ci.dist + r;  // the core point and its distance to the candidate's prism
      for (int k = 0; k < 3; k++) x[k] = a[k] + ci.t * d[k];
      keep = hf_surf_dist(x, h, rmin, rmax, cmin, cmax, dx, dy, hs, hd, ncol) >= h - 1e-6f;
      if (keep && ci.t > 0.f && ci.t < 1.f)  // inside the core: a local minimum of the distance along it
        for (int sg = -1; sg <= 1 && keep; sg += 2) {
          float tb = fminf(fmaxf(ci.t + float(sg) * 5e-4f / fmaxf(dl, 1e-6f), 0.f), 1.f), xb[3];
          for (int k = 0; k < 3; k++) xb[k] = a[k] + tb * d[k];
          keep = hf_surf_dist(xb, h, rmin, rmax, cmin, cmax, dx, dy, hs, hd, ncol) >= h - 1e-8f;
        }
    }
    if (keep) cand[nk++] = cand[i];
  }
  nc = nk;
  if (nc == 0) return 0;
  // MuJoCo's choice (ccd_hfield_kernel): the deepest, the farthest from it, the farthest from their line, the fourth
  int id0 = 0;
  for (int i = 1; i < nc; i++) if (cand[i].dist < cand[id0].dist) id0 = i;
  float p0[3], p1[3], p2[3], pi[3];
  hf_pos(cand[id0], a, d, r, p0);
  int ids[4] = {id0, -1, -1, -1}, n = 1;
  float best = -1.f;
  for (int i = 0; i < nc; i++) {
    if (i == id0) continue;
    hf_pos(cand[i], a, d, r, pi);
    float w[3] = {pi[0] - p0[0], pi[1] - p0[1], pi[2] - p0[2]}, l = sqrtf(dot3(w, w));
    if (l > best) { best = l; ids[1] = i; }
  }
  if (ids[1] >= 0 && best >= HF_MIN_NEXT) {
    n = 2;
    hf_pos(cand[ids[1]], a, d, r, p1);
    const float* n0 = cand[id0].n;
    float l01[3], w01[3] = {p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2]};
    cross(n0, w01, l01);
    best = -1.f;
    for (int i = 0; i < nc; i++) {
      if (i == id0 || i == ids[1]) continue;
      hf_pos(cand[i], a, d, r, pi);
      float w[3] = {pi[0] - p0[0], pi[1] - p0[1], pi[2] - p0[2]}, l = fabsf(dot3(w, l01));
      if (l > best) { best = l; ids[2] = i; }
    }
    if (ids[2] >= 0 && best >= HF_MIN_NEXT) {
      n = 3;
      hf_pos(cand[ids[2]], a, d, r, p2);
      float m2[3], m12[3], w02[3] = {p0[0] - p2[0], p0[1] - p2[1], p0[2] - p2[2]},
                           w12[3] = {p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]};
      cross(n0, w02, m2);
      cross(n0, w12, m12);
      best = -1.f;
      for (int i = 0; i < nc; i++) {
        if (i == id0 || i == ids[1] || i == ids[2]) continue;
        hf_pos(cand[i], a, d, r, pi);
        float w[3] = {pi[0] - p0[0], pi[1] - p0[1], pi[2] - p0[2]};
        float u[3] = {p1[0] - pi[0], p1[1] - pi[1], p1[2] - pi[2]};
        float l = fabsf(dot3(w, m2)) + fabsf(dot3(u, m12));
        if (l > best) { best = l; ids[3] = i; }
      }
      if (ids[3] >= 0 && best >= HF_MIN_NEXT) n = 4;
    }
  }
  for (int k = 0; k < n; k++) hf_con_of(cand[ids[k]], a, d, r, out + k);
  return n;
}

// the adjoint's replay of a native contact: the candidate pid of geom (gt, sz, gp, gq) and heightfield j, recomputed
// from its prism alone (the forward's bits: the same functions)
__device__ __noinline__ bool hfield_native_one(int j, int gt, const float* sz, const float* gp, const float* gq,
                                               unsigned pid, HfCon* o) {
  const float* hh = g_ter.hfh + 16 * j;
  const int ncol = g_ter.hfi[4 * j + 1];
  float a[3], d[3], lo[3], hi3[3], v[3][3];
  hf_core(j, gt, sz, gp, gq, a, d, lo, hi3);
  const int cell = (int)(pid >> 5), rr = cell / ncol, cc = cell - rr * ncol, i = (int)((pid >> 4) & 1u);
  if (!hf_prism_verts(j, rr, cc, i, lo[2], v)) return false;
  HfCand cand[12];
  int nc = 0, over = 0;
  hf_prism(v, -hh[15], a, d, sz[0], cand, &nc, &over);
  for (int q = 0; q < nc; q++)
    if (cand[q].pid == (pid & 15u)) {
      cand[q].pid = pid;
      hf_con_of(cand[q], a, d, sz[0], o);
      return true;
    }
  return false;
}
#endif  // TERRAIN_HF_NATIVE

// ---- height scan (torch_ffi.raycast): rays against the terrain, MuJoCo Warp's ray_box / ray_hfield (ray.py) ----
#define TR_MINVAL 1e-15f  // mjMINVAL
__device__ __forceinline__ float tr_div(float x, float y) { return x / (y != 0.f ? y : TR_MINVAL); }

// _ray_quad: the first non-negative solution of a x^2 + 2 b x + c = 0, else -1
__device__ __forceinline__ float tr_quad(float a, float b, float c) {
  float det = b * b - a * c;
  if (det < TR_MINVAL) return -1.f;
  det = sqrtf(det);
  const float den = tr_div(1.f, a), x0 = (-b - det) * den, x1 = (-b + det) * den;
  return x0 >= 0.f ? x0 : (x1 >= 0.f ? x1 : -1.f);
}

// ray_box: box at pos with rotation R (row major) and half sizes sz; all[6]: per side solutions (or -1), nrm: world
__device__ float tr_box(const float* pos, const float* R, const float* sz, const float* pnt, const float* vec, float* all,
                        float* nrm) {
  for (int i = 0; i < 6; i++) all[i] = -1.f;
  for (int k = 0; k < 3; k++) nrm[k] = 0.f;
  float dif[3] = {pnt[0] - pos[0], pnt[1] - pos[1], pnt[2] - pos[2]};
  if (tr_quad(dot3(vec, vec), dot3(vec, dif), dot3(dif, dif) - dot3(sz, sz)) < 0.f) return -1.f;  // bounding sphere
  float lp[3], lv[3];
  mtv3(R, dif, lp);
  mtv3(R, vec, lv);
  float x = -1.f;
  int fa = -1, fs = -1;
  for (int i = 0; i < 3; i++) {
    if (!(fabsf(lv[i]) > TR_MINVAL)) continue;
    for (int side = -1; side < 2; side += 2) {
      const float sol = (float(side) * sz[i] - lp[i]) / lv[i];
      if (sol >= 0.f) {
        const int i0 = i == 0 ? 1 : 0, i1 = i == 2 ? 1 : 2;
        const float p0 = lp[i0] + sol * lv[i0], p1 = lp[i1] + sol * lv[i1];
        if (fabsf(p0) <= sz[i0] && fabsf(p1) <= sz[i1]) {
          if (x < 0.f || sol < x) { x = sol; fa = i; fs = side; }
          all[2 * i + (side + 1) / 2] = sol;
        }
      }
    }
  }
  if (x >= 0.f) {
    float nl[3] = {0.f, 0.f, 0.f};
    nl[fa] = float(fs);
    mv3(R, nl, nrm);
  }
  return x;
}

// _ray_triangle (local frame; b0, b1: _orthogonal_basis of the ray direction)
__device__ float tr_tri(const float* v0, const float* v1, const float* v2, const float* pnt, const float* vec,
                        const float* b0, const float* b1, float* nrm) {
  float d0[3], d1[3], d2[3];
  for (int k = 0; k < 3; k++) { d0[k] = v0[k] - pnt[k]; d1[k] = v1[k] - pnt[k]; d2[k] = v2[k] - pnt[k]; }
  const float p00 = dot3(d0, b0), p01 = dot3(d0, b1), p10 = dot3(d1, b0), p11 = dot3(d1, b1), p20 = dot3(d2, b0),
              p21 = dot3(d2, b1);
  if ((p00 > 0.f && p10 > 0.f && p20 > 0.f) || (p00 < 0.f && p10 < 0.f && p20 < 0.f) ||
      (p01 > 0.f && p11 > 0.f && p21 > 0.f) || (p01 < 0.f && p11 < 0.f && p21 < 0.f))
    return -1.f;
  const float A00 = p00 - p20, A10 = p10 - p20, A01 = p01 - p21, A11 = p11 - p21, bb0 = -p20, bb1 = -p21;
  const float det = A00 * A11 - A10 * A01;
  if (fabsf(det) < TR_MINVAL) return -1.f;
  const float t0 = (A11 * bb0 - A10 * bb1) / det, t1 = (-A01 * bb0 + A00 * bb1) / det;
  if (t0 < 0.f || t1 < 0.f || t0 + t1 > 1.f) return -1.f;
  float e0[3], e1[3], e2[3], n[3];
  for (int k = 0; k < 3; k++) { e0[k] = v0[k] - v2[k]; e1[k] = v1[k] - v2[k]; e2[k] = pnt[k] - v2[k]; }
  cross(e0, e1, n);
  const float den = dot3(vec, n);
  if (fabsf(den) < TR_MINVAL) return -1.f;
  const float dist = -dot3(e2, n) / den, il = rsqrtf(dot3(n, n));
  for (int k = 0; k < 3; k++) nrm[k] = n[k] * il;
  return dist >= 0.f ? dist : -1.f;
}

// ray_hfield: terrain heightfield j (the base box, the top box, the triangles under the ray's part in the top box,
// split along the (c, r)-(c+1, r+1) diagonal as MuJoCo Warp's rays, and the top box's sides under the boundary heights)
__device__ float tr_hfield(int j, const float* pnt, const float* vec, float* nrm) {
  const float* hh = g_ter.hfh + 16 * j;
  const int* hi = g_ter.hfi + 4 * j;
  const float *pos = hh, *R = hh + 3, *size = hh + 12, *hd = g_ter.hfd + hi[2];
  const int nrow = hi[0], ncol = hi[1];
  const float col[3] = {R[2], R[5], R[8]};
  const float bs = size[3] * 0.5f, ts = size[2] * 0.5f;
  float bsz[3] = {size[0], size[1], bs}, tsz[3] = {size[0], size[1], ts}, bpos[3], tpos[3], allb[6], all[6], nb[3],
        nt[3];
  for (int k = 0; k < 3; k++) { bpos[k] = pos[k] - col[k] * bs; tpos[k] = pos[k] + col[k] * ts; }
  float x = tr_box(bpos, R, bsz, pnt, vec, allb, nb);
  const float top = tr_box(tpos, R, tsz, pnt, vec, all, nt);
  if (top < 0.f) { for (int k = 0; k < 3; k++) nrm[k] = nb[k]; return x; }
  float dif[3] = {pnt[0] - pos[0], pnt[1] - pos[1], pnt[2] - pos[2]}, lp[3], lv[3];
  mtv3(R, dif, lp);
  mtv3(R, vec, lv);
  const float sg = lv[2] >= 0.f ? 1.f : -1.f, a = -1.f / (sg + lv[2]), b = lv[0] * lv[1] * a;
  const float b0[3] = {1.f + sg * lv[0] * lv[0] * a, sg * b, -sg * lv[0]}, b1[3] = {b, sg + lv[1] * lv[1] * a, -lv[1]};
  float s0 = 0.f, s1 = top;
  for (int i = 0; i < 6; i++)
    if (all[i] > s1) { s0 = top; s1 = all[i]; }
  const float dx = tr_div(2.f * size[0], float(ncol - 1)), dy = tr_div(2.f * size[1], float(nrow - 1));
  const float SX0 = tr_div(lp[0] + s0 * lv[0] + size[0], dx), SX1 = tr_div(lp[0] + s1 * lv[0] + size[0], dx);
  const float SY0 = tr_div(lp[1] + s0 * lv[1] + size[1], dy), SY1 = tr_div(lp[1] + s1 * lv[1] + size[1], dy);
  const int cmin = max(0, int(floorf(fminf(SX0, SX1)) - 1.f)), cmax = min(ncol - 1, int(ceilf(fmaxf(SX0, SX1)) + 1.f));
  const int rmin = max(0, int(floorf(fminf(SY0, SY1)) - 1.f)), rmax = min(nrow - 1, int(ceilf(fmaxf(SY0, SY1)) + 1.f));
  float nl[3] = {0.f, 0.f, 0.f};
  if (x >= 0.f) mtv3(R, nb, nl);
  for (int r = rmin; r < rmax; r++)
    for (int c = cmin; c < cmax; c++) {
      const float x0 = dx * float(c) - size[0], x1 = dx * float(c + 1) - size[0];
      const float y0 = dy * float(r) - size[1], y1 = dy * float(r + 1) - size[1];
      const float h00 = hd[r * ncol + c] * size[2], h10 = hd[r * ncol + c + 1] * size[2];
      const float h11 = hd[(r + 1) * ncol + c + 1] * size[2], h01 = hd[(r + 1) * ncol + c] * size[2];
      const float va[3] = {x0, y0, h00}, vb[3] = {x1, y0, h10}, vc[3] = {x1, y1, h11}, vd[3] = {x0, y1, h01};
      float nt2[3];
      float sol = tr_tri(va, vb, vc, lp, lv, b0, b1, nt2);
      if (sol >= 0.f && (x < 0.f || sol < x)) { x = sol; for (int k = 0; k < 3; k++) nl[k] = nt2[k]; }
      sol = tr_tri(va, vc, vd, lp, lv, b0, b1, nt2);
      if (sol >= 0.f && (x < 0.f || sol < x)) { x = sol; for (int k = 0; k < 3; k++) nl[k] = nt2[k]; }
    }
  for (int i = 0; i < 4; i++) {
    if (!(all[i] >= 0.f && (all[i] < x || x < 0.f))) continue;
    const float z = tr_div(lp[2] + all[i] * lv[2], size[2]);
    float y, y0, z0, z1;
    if (i < 2) {
      y = tr_div(lp[1] + all[i] * lv[1] + size[1], dy);
      y0 = fmaxf(0.f, fminf(float(nrow - 2), floorf(y)));
      const int k0 = int(rintf(y0)), k1 = int(rintf(y0 + 1.f));
      z0 = i == 1 ? hd[k0 * ncol + ncol - 1] : hd[k0 * ncol];
      z1 = i == 1 ? hd[k1 * ncol + ncol - 1] : hd[k1 * ncol];
    } else {
      y = tr_div(lp[0] + all[i] * lv[0] + size[0], dx);
      y0 = fmaxf(0.f, fminf(float(ncol - 2), floorf(y)));
      const int k0 = int(rintf(y0)), k1 = int(rintf(y0 + 1.f));
      z0 = i == 3 ? hd[k0 + (nrow - 1) * ncol] : hd[k0];
      z1 = i == 3 ? hd[k1 + (nrow - 1) * ncol] : hd[k1];
    }
    if (z < z0 * (y0 + 1.f - y) + z1 * (y - y0)) {
      x = all[i];
      nl[0] = float(i == 1) - float(i == 0); nl[1] = float(i == 3) - float(i == 2); nl[2] = 0.f;
    }
  }
  if (x >= 0.f) mv3(R, nl, nrm);
  else for (int k = 0; k < 3; k++) nrm[k] = 0.f;
  return x;
}

// n rays (origins pnt, unit directions vec, world, 3 floats each) against the terrain: dist (the nearest hit; -1 for
// none or beyond maxd) and nrm (its world normal, 0 for none); the elements in the grid cells under the segment
// pnt .. pnt + maxd vec
__global__ void k_terrain_rays(int n, const float* pnt, const float* vec, float maxd, float* dist, float* nrm) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float *p = pnt + 3 * i, *v = vec + 3 * i;
  float best = -1.f, bn[3] = {0.f, 0.f, 0.f};
  if (g_ter.grid != nullptr) {
    const float x0 = g_ter.grid[0], y0 = g_ter.grid[1], ic = g_ter.grid[2];
    const int nx = int(g_ter.grid[3]), ny = int(g_ter.grid[4]), nbox = int(g_ter.grid[5]);
    const float ex = p[0] + maxd * v[0], ey = p[1] + maxd * v[1];
    int c0 = max(0, int(floorf((fminf(p[0], ex) - x0) * ic))), c1 = min(nx - 1, int(floorf((fmaxf(p[0], ex) - x0) * ic)));
    int r0 = max(0, int(floorf((fminf(p[1], ey) - y0) * ic))), r1 = min(ny - 1, int(floorf((fmaxf(p[1], ey) - y0) * ic)));
    for (int r = r0; r <= r1; r++)
      for (int c = c0; c <= c1; c++)
        for (int k = g_ter.cstart[r * nx + c]; k < g_ter.cstart[r * nx + c + 1]; k++) {
          const int e = g_ter.citems[k];
          float all[6], nn[3], d;
          if (e < nbox) { const float* b = g_ter.box + 15 * e; d = tr_box(b, b + 3, b + 12, p, v, all, nn); }
          else d = tr_hfield(e - nbox, p, v, nn);
          if (d >= 0.f && (best < 0.f || d < best)) { best = d; for (int q = 0; q < 3; q++) bn[q] = nn[q]; }
        }
  }
  if (best > maxd) best = -1.f;
  dist[i] = best;
  for (int q = 0; q < 3; q++) nrm[3 * i + q] = best >= 0.f ? bn[q] : 0.f;
}

// robot geom g against terrain heightfield j (element nbox + j)
template <class S>
__device__ void terrain_hfield(S& s, int pair, int g, int j, int nbox) {
  const float* hh = g_ter.hfh + 16 * j;
  float d[3] = {s.gx[g][0] - hh[0], s.gx[g][1] - hh[1], s.gx[g][2] - hh[2]}, pl[3];
  mtv3(hh + 3, d, pl);
  if (pl[2] - c_grbound[g] > hh[14]) return;  // above the heightfield's top
#if TERRAIN_HF_NATIVE
  HfCon con[4];
  int over;
  const int nn = hfield_native(j, c_gtype[g], c_gsize + 3 * g, s.gx[g], s.gq[g], con, &over);
  if (over) atomicOr(&s.unsupported, 4);  // candidates dropped (counted as a contact overflow)
  for (int k = 0; k < nn; k++) {
    float pos[3], nw[3], fr[9];
    mv3(hh + 3, con[k].pos, pos);
    for (int i = 0; i < 3; i++) pos[i] += hh[i];
    mv3(hh + 3, con[k].n, nw);
    make_frame(nw, fr);
    const int c = add_contact_aux(s, pair, k | (nbox + j) << 8, con[k].dist, pos, fr);
    if (c >= 0) s.con_aux[c] = con[k].pid;  // the tape's taux: the adjoint replays this prism candidate only
  }
  return;
#endif
  const int gt = c_gtype[g], npts = gt == 2 ? TERRAIN_CAP_PTS : 1;
  for (int k = 0; k < npts; k++) {
    float dist, pos[3], nw[3], fr[9];
    if (!terrain_hfield_point(j, gt, c_gsize + 3 * g, s.gx[g], s.gq[g], k, &dist, pos, nw) || dist >= 0.f) continue;
    make_frame(nw, fr);
    add_contact(s, pair, k | (nbox + j) << 8, dist, pos, fr);
  }
}

// the terrain's contacts of the robot geoms (virtual pairs NPAIR .. NPAIR + NTG - 1), one geom per lane
template <class S>
__device__ void collide_terrain(S& s, int lane) {
  if (g_ter.grid == nullptr) return;
  const float x0 = g_ter.grid[0], y0 = g_ter.grid[1], ic = g_ter.grid[2];
  const int nx = int(g_ter.grid[3]), ny = int(g_ter.grid[4]), nbox = int(g_ter.grid[5]);
  for (int t = lane; t < NTG; t += 32) {
    int pair = NPAIR + t, g = PG2(pair);
    const float* p = s.gx[g];
    float rb = c_grbound[g];
    int c0 = int(floorf((p[0] - rb - x0) * ic)), c1 = int(floorf((p[0] + rb - x0) * ic));
    int r0 = int(floorf((p[1] - rb - y0) * ic)), r1 = int(floorf((p[1] + rb - y0) * ic));
    if (c1 < 0 || r1 < 0 || c0 >= nx || r0 >= ny) continue;
    c0 = max(c0, 0); r0 = max(r0, 0); c1 = min(c1, nx - 1); r1 = min(r1, ny - 1);
    for (int r = r0; r <= r1; r++)
      for (int c = c0; c <= c1; c++)
        for (int k = g_ter.cstart[r * nx + c]; k < g_ter.cstart[r * nx + c + 1]; k++) {
          int e = g_ter.citems[k];
          if (c != max(c0, g_ter.ecell[2 * e]) || r != max(r0, g_ter.ecell[2 * e + 1])) continue;  // once per element
          if (e < nbox) terrain_box(s, pair, g, e);
          else terrain_hfield(s, pair, g, e - nbox, nbox);
        }
  }
}
"""

# (anchor, replacement): each anchor occurs once in STEP_DEVICE + STEP_KERNEL
TERRAIN_HOOKS = (
    # the terrain's device code, before collide_pair
    ("""template <class S>
__device__ DK_INL_COLLIDE void collide_pair(S& s, int pair) {""",
     None),  # TERRAIN_DEVICE with capbox_segs (terrain_source)
    # each contact's terrain element (the tape's telem table): add_contact's slot >> 8
    ("""  unsigned char con_slot[MAXCON];  // which contact of its pair (tape): plane-capsule end, parallel capsule-capsule candidate
""",
     """  unsigned char con_slot[MAXCON];  // which contact of its pair (tape): plane-capsule end, parallel capsule-capsule candidate
  unsigned short con_elem[MAXCON];  // terrain builds: the contact's terrain element (add_contact's slot >> 8)
"""),
    ("""  float con_mu[MAXCON];
  unsigned char con_slot[MAXCON];
  float *con_dist""",
     """  float con_mu[MAXCON];
  unsigned char con_slot[MAXCON];
  unsigned short con_elem[MAXCON];
  float *con_dist"""),
    ("""  s.con_slot[c] = (unsigned char)slot;
""",
     """  s.con_slot[c] = (unsigned char)slot;
  s.con_elem[c] = (unsigned short)(slot >> 8);
"""),
    # the tape's contact tables: pair | slot << 10 (the virtual pairs need 10 bits), and the elements
    ("""      unsigned e0 = 2 * i < ncon ? (unsigned)s.con_pair[2 * i] | (unsigned)s.con_slot[2 * i] << 9 : TAPE_CON_NONE;
      unsigned e1 = 2 * i + 1 < ncon ? (unsigned)s.con_pair[2 * i + 1] | (unsigned)s.con_slot[2 * i + 1] << 9
                                     : TAPE_CON_NONE;
      __stcs(rec + OFF_CON + i, __uint_as_float(e0 | e1 << 16));""",
     """      unsigned e0 = 2 * i < ncon ? (unsigned)s.con_pair[2 * i] | (unsigned)s.con_slot[2 * i] << 10 : TAPE_CON_NONE;
      unsigned e1 = 2 * i + 1 < ncon ? (unsigned)s.con_pair[2 * i + 1] | (unsigned)s.con_slot[2 * i + 1] << 10
                                     : TAPE_CON_NONE;
      __stcs(rec + OFF_CON + i, __uint_as_float(e0 | e1 << 16));
      unsigned x0 = 2 * i < ncon ? (unsigned)s.con_elem[2 * i] : 0u;
      unsigned x1 = 2 * i + 1 < ncon ? (unsigned)s.con_elem[2 * i + 1] : 0u;
      __stcs(rec + OFF_TELEM + i, __uint_as_float(x0 | x1 << 16));"""),
    # broad phase: the terrain replaces the floor plane unless TERRAIN_FLOOR
    ("""        if (PPLANE(pr)) {
#if FLOOR_G >= 0""",
     """        if (PPLANE(pr)) {
#if !TERRAIN_FLOOR
          keep = false;
#else
#if FLOOR_G >= 0"""),
    ("""          keep = dot3(dif, n) <= RFMA(c_grbound[g2], 1.f + 1e-5f, 1e-6f);
        } else {""",
     """          keep = dot3(dif, n) <= RFMA(c_grbound[g2], 1.f + 1e-5f, 1e-6f);
#endif
        } else {"""),
    # narrow phase: the terrain contacts after the model's pairs
    ("""    for (int i = lane; i < ncand; i += 32) collide_pair(s, s.cand[i]);
""",
     """    for (int i = lane; i < ncand; i += 32) collide_pair(s, s.cand[i]);
    collide_terrain(s, lane);
"""),
)


# native heightfield builds (TerrainConfig.mode = "native"), after TERRAIN_HOOKS: each contact's candidate id (the
# tape's taux), so that the adjoint replays one prism
TERRAIN_NATIVE_HOOKS = (
    ("""  unsigned short con_elem[MAXCON];  // terrain builds: the contact's terrain element (add_contact's slot >> 8)
""",
     """  unsigned short con_elem[MAXCON];  // terrain builds: the contact's terrain element (add_contact's slot >> 8)
  unsigned con_aux[MAXCON];  // native heightfield contacts: the candidate id (HfCand.pid), else 0
"""),
    ("""  unsigned short con_elem[MAXCON];
  float *con_dist""",
     """  unsigned short con_elem[MAXCON];
  unsigned con_aux[MAXCON];
  float *con_dist"""),
    ("""  s.con_elem[c] = (unsigned short)(slot >> 8);
  s.con_dim[c] = PDIM(pair);
  s.con_dist[c] = dist;
  for (int k = 0; k < 3; k++) s.con_pos[c][k] = pos[k];
  for (int k = 0; k < 9; k++) s.con_frame[c][k] = frame[k];
  s.con_mu[c] = PFOOT(pair) ? s.fric : PMU(pair);
  return true;""",
     """  s.con_elem[c] = (unsigned short)(slot >> 8);
  s.con_aux[c] = 0u;
  s.con_dim[c] = PDIM(pair);
  s.con_dist[c] = dist;
  for (int k = 0; k < 3; k++) s.con_pos[c][k] = pos[k];
  for (int k = 0; k < 9; k++) s.con_frame[c][k] = frame[k];
  s.con_mu[c] = PFOOT(pair) ? s.fric : PMU(pair);
  return true;"""),
    ("""      __stcs(rec + OFF_TELEM + i, __uint_as_float(x0 | x1 << 16));""",
     """      __stcs(rec + OFF_TELEM + i, __uint_as_float(x0 | x1 << 16));
      __stcs(rec + OFF_TAUX + 2 * i, __uint_as_float(2 * i < ncon ? s.con_aux[2 * i] : 0u));
      if (2 * i + 1 < MAXCON) __stcs(rec + OFF_TAUX + 2 * i + 1, __uint_as_float(2 * i + 1 < ncon ? s.con_aux[2 * i + 1] : 0u));"""),
)


def capbox_segs_source():
    """capsule_box's witness search (STEP_DEVICE) as a function of its own: capbox_segs(...) -> the number of candidate
    spheres (0..2) and their segment parameters segs[] (MuJoCo Warp's capsule_box_witness), for the terrain."""
    i = STEP_DEVICE.index("// capsule-box (MuJoCo Warp's capsule_box_witness")
    j = STEP_DEVICE.index("  for (int e = 0; e < 2; e++) {\n    if (e == 1 && !(second > -3.f)) break;", i)
    src = STEP_DEVICE[i:j]
    reps = (("// capsule-box (MuJoCo Warp's capsule_box_witness + capsule_box): up to two sphere-box contacts along the "
             "segment\ntemplate <class S>\n__device__ void capsule_box(S& s, int pair, const float* cp, const float* "
             "axisw, float cr, float hl, const float* bp,\n                            const float* R, const float* bs) {",
             "// capsule_box's candidate spheres (generated from it by step.capbox_segs_source)\n"
             "__device__ __noinline__ int capbox_segs(const float* cp, const float* axisw, float hl, const float* bp,\n"
             "                                        const float* R, const float* bs, float* segs) {"),
            ("  if (cltype == -4) return;\n", "  if (cltype == -4) return 0;\n"))
    for a, b in reps:
        if src.count(a) != 1:
            raise RuntimeError(f"capbox_segs_source: {a[:60]!r}")
        src = src.replace(a, b)
    if "return;" in src:
        raise RuntimeError("capbox_segs_source: a bare return left")
    return src + "  segs[0] = bestseg;\n  if (!(second > -3.f)) return 1;\n  segs[1] = bestseg + second;\n  return 2;\n}\n"


def apply_hooks(src, hooks):
    """src with each (anchor, replacement) of hooks applied; every anchor must occur exactly once."""
    for a, b in hooks:
        if src.count(a) != 1:
            raise RuntimeError(f"terrain hook anchor found {src.count(a)} times: {a[:60]!r}")
        src = src.replace(a, b)
    return src


def terrain_source(src, terrain=None):
    """src (STEP_DEVICE + STEP_KERNEL, or text containing them) with the terrain hooks for a terrain build; unchanged
    when terrain is None."""
    if terrain is None:
        return src
    dev = capbox_segs_source() + "\n" + TERRAIN_DEVICE.lstrip("\n") + "\n"
    src = apply_hooks(src, [(a, dev + a if b is None else b) for a, b in TERRAIN_HOOKS])
    if terrain[0].mode == "native":
        src = apply_hooks(src, TERRAIN_NATIVE_HOOKS)
    return src


def extra_constants(m):
    """dof -> body map and per-body ancestor bitmasks (bit k: body k is the body or one of its ancestors)."""
    dofbody = np.asarray(m.dof_bodyid, np.float64)
    mask = np.zeros(m.nbody, np.int64)
    for b in range(m.nbody):
        k = b
        while k > 0:
            mask[b] |= 1 << k
            k = m.body_parentid[k]
    return dofbody, mask


EXTRA_HEADER = "DKC int c_dofbody[NV];\nDKC unsigned int c_ancmask[NB];\n"




