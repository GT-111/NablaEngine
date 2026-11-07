"""Which MuJoCo models the engine supports: input validation of a model before anything is built.

The engine's code is generated from the model (nablaengine.g1_engine), but its kernels are written for one class of
robot models (the mjlab G1's): a floating base with one hinge per body, position servos, explicit contact pairs
between planes, spheres, capsules and boxes, the implicitfast integrator and the Newton solver, no passive forces,
tendons or equality constraints. Without these checks many departures from that class would be silent: code generation
would accept the model and the engine would simulate other physics (contype / conaffinity contacts left out, condim 4
treated as frictionless, joint damping ignored, an Euler model integrated as implicitfast, ...). check() turns every
assumption into a finding with a code:

  error  the engine would compute other physics than MuJoCo, or fail: the model is unsupported (UnsupportedModel)
  warn   supported, but on a code path the G1 never exercised on a corpus (untested, may be slower)
  info   a documented semantic difference (H11c: the engine does not clamp ctrl to ctrlrange)

Codes: H1-H31 the requirements on the model's structure and physics (H7b, H11b-e, H14b-d, H26a / b subdivide a code),
U1-U7 the untested code paths, I1 a recorded value, F0 a field this classification does not know (a newer MuJoCo);
every finding states its requirement (what) and the model's value. Besides the explicit checks, every MjModel field of
kind "default" (nablaengine.model.fields: physics the engine does not model, e.g. actuator_damping, actuator_armature,
actuator_delay, dof_dampingpoly, geom_fluid) must equal MuJoCo's compile default, else an error under the field's
code.

Scopes (what a build needs): "forward" (the forward and the check binaries' forward), "engine" (forward and adjoint:
the JAX FFI library), "kinematics" (the kinematics outputs). A finding outside the requested scope is reported with
applies=False and does not count.

  from nablaengine.model import capability
  rep = capability.check(m, foot_pairs=fp)               # CapabilityReport: .errors(), .warnings(), .table()
  capability.require(m, foot_pairs=fp, context="init")   # raises UnsupportedModel (ValueError, AssertionError)
  python -m nablaengine.model.capability --model fixtures/mjlab/g1_train.mjb   # the report (scripts/capability_check.py)

Override: $DK_ALLOW_UNSUPPORTED (default unset) is a comma-separated list of codes (or "all") whose errors require()
reports as a RuntimeWarning instead of raising, for experiments only: the physics is then not MuJoCo's, so results from
such a build are not comparable with the checks'.
"""

import os
import warnings
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from nablaengine.model import fields as F

ALLOW_VAR = "DK_ALLOW_UNSUPPORTED"
COM_BODY = "torso_link"  # the body that takes the per-world COM offset (source.constants, jax_ffi.init)
SCOPES = {"forward": ("forward",), "engine": ("forward", "adjoint"), "kinematics": ("kinematics",)}
J, G, T = mujoco.mjtJoint, mujoco.mjtGeom, mujoco.mjtTrn
GEOM_CODE = {int(G.mjGEOM_PLANE): 0, int(G.mjGEOM_SPHERE): 1, int(G.mjGEOM_CAPSULE): 2, int(G.mjGEOM_BOX): 3}
FORWARD_TYPES = {0, 1, 2, 3}  # step.GEOM_TYPE: plane, sphere, capsule, box
ADJOINT_TYPES = {0, 1, 2}  # adjoint_contact.check_model: no boxes
DISPATCHED = {(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)}  # step.py collide_pair's combinations
CONTACTS_PER_PAIR = {(0, 1): 1, (0, 2): 2, (0, 3): 4, (1, 1): 1, (1, 2): 1, (2, 2): 2, (2, 3): 2, (3, 3): 8}
D, E = mujoco.mjtDisableBit, mujoco.mjtEnableBit
# disable flags that cannot change the engine's results: sensors (the engine computes none), Euler damping (Euler is
# rejected), auto reset, midphase / island bookkeeping of MuJoCo's own collision and solver paths, the parent filter
# (contype contacts are rejected; explicit pairs are not filtered), ctrl clamping (the engine never clamps); the CCD
# flags only without box pairs (box-box follows MuJoCo Warp's native CCD, step.py BOX_BOX_CCD; no other pair uses CCD)
DISABLE_OK = (D.mjDSBL_SENSOR, D.mjDSBL_EULERDAMP, D.mjDSBL_AUTORESET, D.mjDSBL_MIDPHASE, D.mjDSBL_ISLAND,
              D.mjDSBL_FILTERPARENT, D.mjDSBL_CLAMPCTRL)
DISABLE_OK_NO_BOX = (D.mjDSBL_NATIVECCD, D.mjDSBL_MULTICCD)
# disable flags of physics a supported model does not have, so they change nothing: springs and dampers (every
# stiffness and damping field must be at its default 0: H22 and the default-kind fields; no tendons or flexes: H25),
# equality constraints (neq 0: H25); frictionloss and joint limits only where no dof has frictionloss and no joint a limit
DISABLE_NOOP = (D.mjDSBL_SPRING, D.mjDSBL_DAMPER, D.mjDSBL_EQUALITY)
ENABLE_OK = (E.mjENBL_ENERGY, E.mjENBL_FWDINV, E.mjENBL_INVDISCRETE)  # diagnostics / inverse dynamics only
STATUSES = ("error", "warn", "info", "ok")
CTRLSPEC_OK = (0, int(mujoco.mjtCtrlInput.mjINPUT_POS))  # a fixed gain's input: the type default, or a position


def _bits(flags, enum):
    return [k for k, v in enum.__members__.items() if not k.startswith("mjN") and flags & int(v)]


def _mask(members):
    out = 0
    for v in members:
        out |= int(v)
    return out


@dataclass(frozen=True)
class Finding:
    code: str
    status: str  # error / warn / info when the check fails; ok when it passes
    what: str
    value: object = None
    scopes: tuple = ("forward",)
    applies: bool = True  # in the report's scope

    def line(self):
        v = "" if self.value is None else f" [{_short(self.value)}]"
        return f"{self.code:5s} {self.status:5s} {self.what}{v}"


def _short(v, n=160):
    s = str(v)
    return s if len(s) <= n else s[:n - 3] + "..."


class CapabilityReport:
    """The findings of check() for one model and scope."""

    def __init__(self, scope, findings):
        self.scope, self.findings = scope, list(findings)

    def _of(self, status):
        return [f for f in self.findings if f.applies and f.status == status]

    def errors(self):
        return self._of("error")

    def warnings(self):
        return self._of("warn")

    def infos(self):
        return self._of("info")

    def codes(self, status="error"):
        return sorted({f.code for f in self._of(status)})

    def ok(self):
        return not self.errors()

    def table(self, all_rows=False):
        """One line per finding that applies and is not ok (all_rows: every finding)."""
        rows = [f for f in self.findings if all_rows or (f.applies and f.status != "ok")]
        return "\n".join(f.line() + ("" if f.applies else "  (other scope)") for f in rows)

    def summary(self):
        return (f"scope {self.scope}: {len(self.errors())} errors {self.codes('error')}, {len(self.warnings())} "
                f"warnings {self.codes('warn')}, {len(self.infos())} infos {self.codes('info')}")

    def to_dict(self):
        return dict(scope=self.scope, errors=self.codes("error"), warnings=self.codes("warn"), infos=self.codes("info"),
                    findings=[{k: (_jsonable(v)) for k, v in asdict(f).items()} for f in self.findings])


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


class UnsupportedModel(ValueError, AssertionError):
    """A model the engine does not support (capability errors). A ValueError (as jax_ffi.check_model's model
    mismatch) and an AssertionError (as the code generators' asserts, which jax_ffi.init documents)."""

    def __init__(self, report, context=""):
        self.report = report
        lines = "\n  ".join(f.line() for f in report.errors())
        super().__init__(f"nablaengine: the model is not supported by the engine{' (' + context + ')' if context else ''}"
                         f": {report.summary()}\n  {lines}\n(nablaengine.model.capability documents the codes; "
                         f"${ALLOW_VAR}=<codes> overrides them, for experiments only)")


def filter_pairs(m):
    """Geom pairs that MuJoCo's broadphase lets collide besides the explicit <pair>s (contype / conaffinity, not the
    same weld body, the parent filter unless disabled, not excluded, not already an explicit pair; measured on mujoco
    3.17: an explicit pair replaces only its own geom pair, exclude signatures are body1 << 16 | body2): the contacts
    the engine, which generates only explicit pairs, would leave out."""
    ex = {int(s) for s in m.exclude_signature} if m.nexclude else set()
    explicit = {frozenset((int(a), int(b))) for a, b in zip(m.pair_geom1, m.pair_geom2)}
    filt_parent = not (m.opt.disableflags & int(D.mjDSBL_FILTERPARENT))
    g = [i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
    out = []
    for a in range(len(g)):
        for b in range(a + 1, len(g)):
            g1, g2 = g[a], g[b]
            if not ((m.geom_contype[g1] & m.geom_conaffinity[g2]) or (m.geom_contype[g2] & m.geom_conaffinity[g1])):
                continue
            w1, w2 = int(m.body_weldid[m.geom_bodyid[g1]]), int(m.body_weldid[m.geom_bodyid[g2]])
            if w1 == w2:
                continue
            p1, p2 = int(m.body_weldid[m.body_parentid[w1]]), int(m.body_weldid[m.body_parentid[w2]])
            if filt_parent and w1 and w2 and (p1 == w2 or p2 == w1):
                continue
            b1, b2 = sorted((int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])))
            if (b1 << 16) + b2 in ex or frozenset((g1, g2)) in explicit:
                continue
            out.append((g1, g2))
    return out


def collision_geoms(m):
    return sorted({int(g) for g in m.pair_geom1} | {int(g) for g in m.pair_geom2})


def _pair_codes(m, p):
    a, b = int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]])
    return tuple(sorted((GEOM_CODE.get(a, -1), GEOM_CODE.get(b, -1))))


def _name(m, objtype, i):
    return mujoco.mj_id2name(m, objtype, int(i)) or str(int(i))


def check(m, foot_pairs=None, com_body=COM_BODY, scope="engine"):
    """The CapabilityReport of model m for a build scope ("forward", "engine", "kinematics"; module docstring).
    foot_pairs: the contact pairs that take the per-world friction (jax_ffi's foot_pairs; None: not checked, and U4
    counts the plane pairs); com_body: the body that takes the per-world COM offset (None: not checked)."""
    from nablaengine.g1_engine import step  # the generators' own model-derived switches (no import cycle)
    if scope not in SCOPES:
        raise ValueError(f"scope {scope!r}: one of {', '.join(SCOPES)}")
    want = set(SCOPES[scope])
    out = []

    def add(code, ok, status, what, value=None, scopes=("forward",)):
        out.append(Finding(code, "ok" if ok else status, what, _jsonable(value), tuple(scopes),
                           bool(want & set(scopes))))

    nb, nv, nu = int(m.nbody), int(m.nv), int(m.nu)
    parent = [int(x) for x in m.body_parentid]
    children = [[] for _ in range(nb)]
    for b in range(1, nb):
        children[parent[b]].append(b)
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == J.mjJNT_HINGE]

    # ---- tree, joints, dofs
    add("H1", nb <= 32, "error", "at most 32 bodies (lane = body, 32-bit ancestor masks)", nb,
        ("forward", "kinematics"))
    add("H2", nb <= 31, "error", "at most 31 bodies in the adjoint (lanes 0 and 31 carry root rotation dofs)", nb,
        ("adjoint",))
    add("H3", nv >= 32, "warn", "nv >= 32 (the Cholesky / triangular solves give lane l dof row l; with nv < 32 the "
        "lanes past NV are masked, a path checked on the Unitree H1 only, nv 25: scripts/h1_check.py)", nv)
    add("H4", nv <= 64, "error", "nv <= 64 (two rows per lane)", nv)
    add("H5", 33 <= nv <= 35, "warn", "33 <= nv <= 35: the tuned CHOL_BCAST / NEWTON_REUSE / blocked triangular solve "
        "(else the register Cholesky fallback: untested on a corpus, slower)", nv, ("forward", "adjoint"))
    free1 = nb > 1 and m.body_jntnum[1] == 1 and m.jnt_type[m.body_jntadr[1]] == J.mjJNT_FREE
    one_tree = nb > 1 and all(int(m.body_rootid[b]) == 1 for b in range(1, nb))
    add("H6", free1 and children[0] == [1] and one_tree, "error",
        "body 1 has the only joint of its body, a free joint, and is the world's only child (one tree)",
        dict(world_children=children[0], body1_free=bool(free1)), ("forward", "kinematics"))
    bad7 = [b for b in range(2, nb) if not (m.body_jntnum[b] == 1 and m.body_jntadr[b] >= 0
                                            and m.jnt_type[m.body_jntadr[b]] == J.mjJNT_HINGE
                                            and not np.any(m.jnt_pos[m.body_jntadr[b]]))]
    add("H7", not bad7, "error", "every body after the root has one hinge at its origin (no welded, multi-joint, ball "
        "or slide bodies, no hinge anchor offset)", [_name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bad7[:8]],
        ("forward", "kinematics"))
    order = nb > 2 and m.body_dofadr[1] == 0 and m.body_dofnum[1] == 6 and all(
        m.body_dofnum[b] == 1 and m.body_dofadr[b] == b + int(m.body_dofadr[2]) - 2 for b in range(2, nb))
    add("H7b", bool(order), "error", "dofs in body order: the root's 0-5, then body b's hinge at b + const "
        "(adjoint.header's ADJ_HINGE_DOF0)", None, ("adjoint",))
    q0 = [j for j in hinge if m.qpos0[m.jnt_qposadr[j]] != 0]
    add("H8", not q0, "error", "hinge qpos0 = 0 (the engine uses qpos as the angle, MuJoCo qpos - qpos0)",
        {_name(m, mujoco.mjtObj.mjOBJ_JOINT, j): float(m.qpos0[m.jnt_qposadr[j]]) for j in q0[:8]},
        ("forward", "kinematics"))
    nch = max((len(c) for c in children[1:]), default=0)
    add("H9", nch <= 3, "error", "at most 3 children per body (c_child)", nch)

    def nanc(b):
        k, n = parent[b], 0
        while k >= 2:
            n, k = n + 1, parent[k]
        return n
    na_max = max((nanc(b) for b in range(2, nb)), default=0)
    add("H10", na_max <= 12, "error", "at most 12 hinge ancestors per body (c_anc)", na_max)

    # ---- actuators
    bad11 = []
    for a in range(int(m.nactuator)):
        j = int(m.actuator_trnid[a, 0])
        ok = (m.actuator_trntype[a] == T.mjTRN_JOINT and 0 <= j < m.njnt and m.jnt_type[j] == J.mjJNT_HINGE
              and m.jnt_dofadr[j] == 6 + a and m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
              and m.actuator_gear[a, 0] == 1.0 and not np.any(m.actuator_gear[a, 1:])
              and m.actuator_gainprm[a, 0] == -m.actuator_biasprm[a, 1] and not m.actuator_forcelimited[a]
              and m.actuator_ctrladr[a] == a and m.actuator_ctrlnum[a] == 1 and m.actuator_outadr[a] == a
              and m.actuator_outnum[a] == 1)
        if not ok:
            bad11.append(_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a))
    add("H11a", nu == m.nactuator == m.nout == nv - 6 and not bad11, "error",
        "one position servo per hinge in joint order (actuator a on dof 6 + a, one control and force each), joint "
        "transmission, affine bias, gear 1, gainprm[0] = -biasprm[1], not force-limited",
        dict(nu=nu, nactuator=int(m.nactuator), hinges=nv - 6, bad=bad11[:8]))
    gt = [_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(int(m.nactuator))
          if m.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED or m.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE
          or m.actuator_biasprm[a, 0] != 0 or np.any(m.actuator_biasprm[a, 3:]) or np.any(m.actuator_gainprm[a, 1:])
          or int(m.actuator_ctrlspec[a]) not in CTRLSPEC_OK]
    add("H11b", not gt and m.na == 0, "error", "fixed gain with one position input (ctrlspec 0 or mjINPUT_POS), no "
        "activation dynamics (na 0), bias (0, -kp, -kv) only", dict(bad=gt[:8], na=int(m.na)))
    clamped = int(np.asarray(m.actuator_ctrllimited).sum()) if not (m.opt.disableflags & int(D.mjDSBL_CLAMPCTRL)) else 0
    add("H11c", clamped == 0, "info", "ctrl is not clamped to ctrlrange by the engine (MuJoCo and MuJoCo Warp clamp "
        "it; the interface's documented semantics)", dict(ctrllimited=clamped))
    asym = [_name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)
            if m.jnt_actfrclimited[j] and m.jnt_actfrcrange[j, 0] != -m.jnt_actfrcrange[j, 1]]
    add("H12", not asym, "error", "symmetric actuator force ranges (jnt_actfrcrange)", asym[:8])
    if com_body is not None:
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, com_body)
        add("H13", tid >= 1, "error", f"the COM-offset body {com_body!r} exists", tid)

    # ---- contacts
    cg = collision_geoms(m)
    types = {GEOM_CODE.get(int(m.geom_type[g]), -1) for g in cg}
    tnames = sorted({mujoco.mjtGeom(int(m.geom_type[g])).name for g in cg})
    add("H14a", types <= FORWARD_TYPES, "error", "contact pair geoms are planes, spheres, capsules or boxes", tnames)
    planes_off_world = [g for g in cg if m.geom_type[g] == G.mjGEOM_PLANE and m.geom_bodyid[g] != 0]
    add("H14b", types <= ADJOINT_TYPES and not planes_off_world, "error", "contact pair geoms are planes (on the world "
        "body), spheres or capsules in the adjoint", dict(types=tnames, planes_off_world=planes_off_world[:4]),
        ("adjoint",))
    combos = sorted({_pair_codes(m, p) for p in range(m.npair)} - DISPATCHED)
    add("H14c", not combos, "error", "every pair's type combination is one the forward collides (not plane-plane, "
        "sphere-box)", combos)
    # box-box follows MuJoCo Warp's native CCD (GJK / EPA with opt.ccd_iterations and ccd_tolerance) by its converged
    # result (step.py box_box, BOX_BOX_CCD): fewer iterations or a looser tolerance change MuJoCo Warp's contacts, not
    # the engine's (no other pair type of the engine uses CCD)
    d_it, d_tol = F.option_default("ccd_iterations"), F.option_default("ccd_tolerance")
    boxbox = sum(1 for p in range(m.npair) if _pair_codes(m, p) == (3, 3))
    add("H14d", not boxbox or (m.opt.ccd_iterations >= d_it and m.opt.ccd_tolerance <= d_tol), "error",
        f"box-box pairs: native CCD at least as converged as MuJoCo's default (ccd_iterations >= {d_it}, ccd_tolerance "
        f"<= {d_tol:g}), whose converged contact the forward computes",
        dict(boxbox_pairs=boxbox, ccd_iterations=int(m.opt.ccd_iterations), ccd_tolerance=float(m.opt.ccd_tolerance)))
    fp = filter_pairs(m)
    add("H15", not fp, "error", "no contype / conaffinity contacts (the engine generates only the explicit pairs)",
        dict(n=len(fp), first=[(_name(m, mujoco.mjtObj.mjOBJ_GEOM, a), _name(m, mujoco.mjtObj.mjOBJ_GEOM, b))
                               for a, b in fp[:4]]))
    mg = [p for p in range(m.npair) if m.pair_margin[p] or m.pair_gap[p]
          or m.pair_friction[p, 0] != m.pair_friction[p, 1]]
    add("H16", not mg, "error", "pair margin = gap = 0 and friction[0] = friction[1]",
        [_name(m, mujoco.mjtObj.mjOBJ_PAIR, p) for p in mg[:8]])
    cd = sorted({int(x) for x in m.pair_dim} - {1, 3})
    add("H17", not cd, "error", "pair condim 1 or 3 (others would be simulated as frictionless)", cd)
    add("H18", m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL, "error", "pyramidal friction cone",
        mujoco.mjtCone(int(m.opt.cone)).name)
    # MuJoCo Warp (constraint.py) reads each solref entry on its own: solref[0] <= 0 is a direct stiffness, solref[1]
    # <= 0 a direct damping (e.g. (0.02, 0) is the time constant's stiffness without damping); the engine's kbimp has
    # the standard form of both only
    neg = dict(pairs=int((m.pair_solref <= 0).any(axis=1).sum()), joints=int((m.jnt_solref[1:] <= 0).any(axis=1).sum()),
               dofs=int((m.dof_solref <= 0).any(axis=1).sum()))
    add("H19", not any(neg.values()), "error", "solref in the standard form, both entries > 0 (time constant, damping "
        "ratio; a direct, non-positive stiffness or damping entry is unsupported)", neg)
    lim_free = [j for j in range(m.njnt) if m.jnt_limited[j] and m.jnt_type[j] != J.mjJNT_HINGE]
    uni = (m.njnt < 2 or ((m.jnt_solref[1:] == m.jnt_solref[1]).all() and (m.jnt_solimp[1:] == m.jnt_solimp[1]).all()))
    dof_uni = nv == 0 or ((m.dof_solref == m.dof_solref[0]).all() and (m.dof_solimp == m.dof_solimp[0]).all())
    add("H20", bool(uni and not m.jnt_margin.any() and dof_uni and not lim_free), "error",
        "one solref / solimp for every joint limit and every dof, jnt_margin 0, only hinge limits",
        dict(limited_non_hinge=lim_free[:4]))
    add("H21", not (m.dof_frictionloss > 0).any(), "error", "no frictionloss in the adjoint (the forward has it)",
        int((m.dof_frictionloss > 0).sum()), ("adjoint",))

    # ---- passive forces, integrator, solver, options
    medium = dict(density=float(m.opt.density), viscosity=float(m.opt.viscosity), wind=[float(x) for x in m.opt.wind])
    add("H22", m.opt.density == 0 and m.opt.viscosity == 0 and not np.any(m.opt.wind), "error",
        "no fluid forces (opt density, viscosity, wind 0)", medium)
    add("H23", m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST, "error", "implicitfast integrator",
        mujoco.mjtIntegrator(int(m.opt.integrator)).name)
    add("H24a", m.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON, "error", "Newton solver",
        mujoco.mjtSolver(int(m.opt.solver)).name)
    has_box = any(m.geom_type[g] == G.mjGEOM_BOX for g in cg)
    noop = DISABLE_NOOP + ((D.mjDSBL_FRICTIONLOSS,) if not (m.dof_frictionloss > 0).any() else ()) + (
        (D.mjDSBL_LIMIT,) if not np.asarray(m.jnt_limited).any() else ())
    dis = int(m.opt.disableflags) & ~_mask(DISABLE_OK + noop + (() if has_box else DISABLE_OK_NO_BOX))
    en = int(m.opt.enableflags) & ~_mask(ENABLE_OK)
    add("H24b", dis == 0 and en == 0 and m.opt.noslip_iterations == 0 and m.opt.disableactuator == 0, "error",
        "no disable / enable flags that change the physics, noslip off, no disabled actuator groups",
        dict(disable=_bits(dis, D), enable=_bits(en, E), noslip_iterations=int(m.opt.noslip_iterations),
             disableactuator=int(m.opt.disableactuator)))
    add("H25", m.ntendon == 0 and m.neq == 0 and m.nmocap == 0 and m.nflex == 0 and m.nplugin == 0, "error",
        "no tendons, equality constraints, mocap bodies, flexes or plugins",
        dict(ntendon=int(m.ntendon), neq=int(m.neq), nmocap=int(m.nmocap), nflex=int(m.nflex),
             nplugin=int(m.nplugin)), ("forward", "kinematics"))

    # ---- capacities (the generators' own values; they need supported geom types and one hinge per body)
    gen = types <= FORWARD_TYPES and not bad7 and nb > 1
    kdof = step.max_support(m) if m.npair and gen else 0
    maxcon = step.maxcon(m)
    npset = len(step.pair_param_sets(m)[0])
    add("H26a", len(cg) <= 255 and npset <= 255 and kdof <= 32 and maxcon <= 64 and m.npair < 65536, "error",
        "<= 255 collision geoms and parameter sets (the pair word), KDOF <= 32, MAXCON <= 64",
        dict(ngeomc=len(cg), param_sets=npset, kdof=kdof, maxcon=maxcon))
    add("H26b", m.npair <= 512, "error", "at most 512 contact pairs (the tape's 9-bit pair index)", int(m.npair),
        ("adjoint",))
    if foot_pairs is not None:
        fps = [int(p) for p in np.asarray(foot_pairs).ravel()]
        add("H30", set(fps) <= set(range(m.npair)), "error", "the friction (foot) pairs are contact pair indices",
            dict(n=len(fps), outside=sorted(set(fps) - set(range(m.npair)))[:8]))

    # ---- kinematics outputs
    sub = np.array(m.body_mass, np.float64)
    for b in range(nb - 1, 0, -1):
        sub[parent[b]] += sub[b]
    add("H31", not np.any(m.body_pos[0]) and np.array_equal(m.body_quat[0], [1.0, 0.0, 0.0, 0.0])
        and m.body_mass[0] == 0 and (nb < 2 or sub[1:].min() >= mujoco.mjMINVAL), "error",
        "kinematics outputs: the world body at the origin and massless, every subtree mass >= mjMINVAL", None,
        ("kinematics",))

    # ---- untested code paths
    add("U1", not (33 <= nv <= 34), "warn", "CHOL_BCAST's tail of nv - 32 rows: only the G1's 3-row tail (nv 35) ran on "
        "a corpus", nv, ("forward", "adjoint"))
    add("U2", not gen or kdof == 16, "warn", "KDOF = 16 (the G1's): otherwise DK_JV_VEC's 128-bit con_jv loads are off and the "
        "scalar path, never run on a corpus, is used", kdof)
    add("U3", maxcon == 64, "warn", "MAXCON 64 in the adjoint (npair > 64; the adjoint never ran with MAXCON 32)",
        maxcon, ("adjoint",))
    worst = 0
    dr = ([int(p) for p in np.asarray(foot_pairs).ravel()] if foot_pairs is not None
          else [p for p in range(m.npair) if 0 in _pair_codes(m, p)])
    for p in dr:
        if 0 <= p < m.npair:
            worst += CONTACTS_PER_PAIR.get(_pair_codes(m, p), 1)
    add("U4", worst <= maxcon, "warn", "the friction (foot) pairs' worst-case contacts fit MAXCON (else contacts are "
        "dropped and the world flagged while standing)", dict(worst_case=worst, maxcon=maxcon))
    floor = step.floor_geom(m) if gen else None  # (unsupported types: H14a, the switch is not generated)
    add("U5", floor is None or floor >= 0 or not m.npair, "warn", "one plane among the collision geoms (FLOOR_G; else "
        "the general plane path, untested on a corpus)", floor)
    add("U6", bool(step.dof_body_root6(m)), "warn", "bodies and dofs in the same order (DOF_BODY_ROOT6; else the general "
        "support mask, untested on a corpus)", bool(step.dof_body_root6(m)))
    pw = sorted({float(np.float32(x)) for x in list(m.pair_solimp[:, 4]) + list(m.jnt_solimp[1:, 4])})
    add("U7", bool(step.solimp_pow2(m)) and len(pw) <= 1, "warn", "every solimp power 2 (SOLIMP_POW2, ADJ_SOLIMP_POWER; "
        "else the general power paths, untested on a corpus)", pw, ("forward", "adjoint"))
    add("I1", True, "info", "the timestep (baked into the constants)", float(m.opt.timestep))

    # ---- fields the engine does not model: compile defaults only
    defaults = F.default_rows()
    for name in F.default_fields():
        f = F.FIELDS[name]
        bad = F.non_default(m, name, defaults)
        if bad:
            kind = name.split("_", 1)[0]
            obj = dict(body=mujoco.mjtObj.mjOBJ_BODY, jnt=mujoco.mjtObj.mjOBJ_JOINT, dof=None,
                       geom=mujoco.mjtObj.mjOBJ_GEOM, pair=mujoco.mjtObj.mjOBJ_PAIR,
                       actuator=mujoco.mjtObj.mjOBJ_ACTUATOR)[kind]
            rows = [(_name(m, obj, i) if obj is not None else int(i)) for i in bad[:6]]
            add(f.code, False, "error", f"{name} at its default ({f.note})", dict(rows=len(bad), first=rows))
    unknown = F.unclassified(m)
    add("F0", not unknown, "error", "every MjModel field is classified (nablaengine.model.fields; a MuJoCo version "
        "with new fields needs their classification)", unknown[:8], ("forward", "adjoint", "kinematics"))
    return CapabilityReport(scope, out)


def allowed_codes():
    """The codes $DK_ALLOW_UNSUPPORTED lets through (a set; {"all"} for every code)."""
    v = os.environ.get(ALLOW_VAR, "").strip()
    return {c.strip() for c in v.split(",") if c.strip()}


def require(m, foot_pairs=None, com_body=COM_BODY, scope="engine", context=""):
    """check(), then raise UnsupportedModel if it has errors (except the codes $DK_ALLOW_UNSUPPORTED lets through,
    which become a RuntimeWarning). Returns the report."""
    rep = check(m, foot_pairs, com_body, scope)
    errs = rep.errors()
    if errs:
        allow = allowed_codes()
        kept = [f for f in errs if not ("all" in allow or f.code in allow)]
        if kept:
            raise UnsupportedModel(CapabilityReport(scope, [f for f in rep.findings if f not in errs or f in kept]),
                                   context)
        warnings.warn(f"nablaengine: ${ALLOW_VAR}={os.environ.get(ALLOW_VAR)}: building for a model the engine does "
                      f"not support ({', '.join(sorted({f.code for f in errs}))}): its physics is not MuJoCo's"
                      + (f" ({context})" if context else ""), RuntimeWarning, stacklevel=2)
    return rep


def diff(a, b):
    """Codes whose status or value differ between two reports (e.g. of a model and of a modified copy of it): [{code,
    what, a: (status, value), b: (status, value)}] (codes present in one report only too)."""
    def by_code(r):
        out = {}
        for f in r.findings:
            out.setdefault(f.code, []).append(f)
        return out
    ia, ib = by_code(a), by_code(b)
    res = []
    for code in sorted(set(ia) | set(ib), key=lambda c: (c[0], int("".join(ch for ch in c[1:] if ch.isdigit()) or 0), c)):
        fa, fb = ia.get(code, []), ib.get(code, [])
        sa = [(f.status, f.value) for f in fa]
        sb = [(f.status, f.value) for f in fb]
        if sa != sb:
            res.append(dict(code=code, what=(fa or fb)[0].what, a=sa, b=sb))
    return res


def main(argv=None):
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser(description="The engine's capability report of a MuJoCo model (nablaengine.model."
                                             "capability: the input validation of jax_ffi.init and the checks)")
    ap.add_argument("--model", required=True, help="an .mjb or MJCF (.xml) file")
    ap.add_argument("--foot-pairs", default="", help="comma-separated friction (foot) pair indices, or a g1_meta.json")
    ap.add_argument("--com-body", default=COM_BODY)
    ap.add_argument("--scope", default="engine", choices=sorted(SCOPES))
    ap.add_argument("--all", action="store_true", help="print the passing checks too")
    ap.add_argument("--json", default="", help="write the report here")
    args = ap.parse_args(argv)
    p = Path(args.model)
    m = mujoco.MjModel.from_binary_path(str(p)) if p.suffix == ".mjb" else mujoco.MjModel.from_xml_path(str(p))
    fp = None
    if args.foot_pairs.endswith(".json"):
        fp = json.loads(Path(args.foot_pairs).read_text())["foot_pairs"]
    elif args.foot_pairs:
        fp = [int(x) for x in args.foot_pairs.split(",")]
    rep = check(m, fp, args.com_body, args.scope)
    print(rep.table(args.all) or "(no findings)")
    print(rep.summary())
    if args.json:
        Path(args.json).write_text(json.dumps(rep.to_dict(), indent=1) + "\n")
    return 0 if rep.ok() else 1


if __name__ == "__main__":
    raise SystemExit(main())
