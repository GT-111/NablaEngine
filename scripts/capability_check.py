"""The capability checker's acceptance test (nablaengine.model.capability, nablaengine.model.fields): the model
assumptions the generated engine code relies on are checked, and a model that breaks one is rejected before anything
is built.

Sections (--sections, default all):
  fields     every MjModel field of the installed MuJoCo is classified (fields.unclassified is empty), every
             default-kind field has a compile default, the classification's codes are the checker's, and every
             classified name and prefix is one of this MuJoCo's fields (none stale)
  fixtures   the mjlab G1 (fixtures/mjlab/g1_train.mjb) has 0 errors and 0 warnings in every scope (one info: H11c, the
             engine does not clamp ctrl); nablaengine.fixtures.load_model checks it
  variants   the G1 rebuilt as an MjSpec from the fixture's arrays (spec_from_model: bodies, joints, geoms, pairs,
             actuators, options) checks clean, then each deliberately modified unsupported variant (condim 4,
             Euler, negative solref, qpos0 != 0, a tendon, gravcomp, the CG solver;
             actuator-level damping / armature / delay, damping polynomials, a motor actuator, a cylinder pair, a ball
             joint body, the waist roll and pitch welded, a direct damping solref (0.02,
             0), a box-box pair with fewer native-CCD iterations than MuJoCo's default, contacts disabled,
             frictionloss disabled with frictionloss) must raise exactly its own codes (new errors relative to the
             base) in the scope it names, and capability.require and nablaengine.jax_ffi.init(model=...) must reject
             it with UnsupportedModel before anything is built. Supported on purpose (no new error and no new
             warning, require accepts): eligible implicit contype contacts, contype contacts that only duplicate explicit pairs (MuJoCo 3.17 collides
             such a geom pair once, with the pair's parameters: measured), a box-box pair at the CCD defaults
             (forward scope), renamed bodies and joints, visual-only geoms with a camera and a light, sites and
             sensors, another gravity vector, disable flags of physics the model does not have (springs, dampers,
             equality, frictionloss); linear joint damping, a soft linear joint equality and dof frictionloss;
             and the welded-waist G1 with its fused body named torso_link, which must warn U1
             (the nv-33 Cholesky tail) and U2 (KDOF 14: the scalar con_jv path), code paths the G1 does not run
  kinematics g1_engine.kinematics.check_model: the G1 passes; a jointless (welded) body is rejected (the body_jntadr =
             -1 case)

  PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/capability_check.py --out runs/capability_check.json

CPU only (no GPU, no nvcc). Exit status 0 when every section passes, 1 otherwise.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # init(model=...) is called on unsupported models only: no GPU needed

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from nablaengine import fixtures, provenance  # noqa: E402
from nablaengine.g1_engine import kinematics  # noqa: E402
from nablaengine.model import capability, fields  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("fields", "fixtures", "variants", "kinematics")


def spec_from_model(m):
    """An MjSpec of the compiled model m's dynamics (bodies with explicit inertials, joints, collision geoms, explicit
    pairs, actuators, options): a stand-in for the model's source, so that variants can be made as spec edits (what a
    user would write in MJCF). Visual elements, sensors and keyframes are left out."""
    s = mujoco.MjSpec()
    s.compiler.degree = False
    s.compiler.autolimits = False
    s.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
    o = m.opt
    for k in ("timestep", "impratio", "tolerance", "ls_tolerance", "integrator", "cone", "solver", "iterations",
              "ls_iterations", "disableflags", "enableflags", "jacobian"):
        setattr(s.option, k, getattr(o, k))
    s.option.gravity = o.gravity.copy()
    name = lambda t, i: mujoco.mj_id2name(m, t, i)  # noqa: E731
    lim = lambda x: mujoco.mjtLimited.mjLIMITED_TRUE if x else mujoco.mjtLimited.mjLIMITED_FALSE  # noqa: E731
    bodies = {0: s.worldbody}
    for b in range(1, m.nbody):
        sb = bodies[int(m.body_parentid[b])].add_body(name=name(mujoco.mjtObj.mjOBJ_BODY, b), pos=m.body_pos[b],
                                                      quat=m.body_quat[b])
        sb.explicitinertial = True
        sb.mass, sb.ipos, sb.iquat, sb.inertia = m.body_mass[b], m.body_ipos[b], m.body_iquat[b], m.body_inertia[b]
        bodies[b] = sb
    for j in range(m.njnt):
        sb, nm = bodies[int(m.jnt_bodyid[j])], name(mujoco.mjtObj.mjOBJ_JOINT, j)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            sb.add_freejoint(name=nm)
            continue
        d = int(m.jnt_dofadr[j])
        sj = sb.add_joint(name=nm, type=mujoco.mjtJoint(int(m.jnt_type[j])), axis=m.jnt_axis[j], pos=m.jnt_pos[j])
        sj.range, sj.limited, sj.margin = m.jnt_range[j], lim(m.jnt_limited[j]), float(m.jnt_margin[j])
        sj.solref_limit, sj.solimp_limit = m.jnt_solref[j], m.jnt_solimp[j]
        sj.solref_friction, sj.solimp_friction = m.dof_solref[d], m.dof_solimp[d]
        sj.armature, sj.frictionloss = float(m.dof_armature[d]), float(m.dof_frictionloss[d])
        sj.actfrclimited, sj.actfrcrange = lim(m.jnt_actfrclimited[j]), m.jnt_actfrcrange[j]
    for g in range(m.ngeom):
        sg = bodies[int(m.geom_bodyid[g])].add_geom(name=name(mujoco.mjtObj.mjOBJ_GEOM, g),
                                                    type=mujoco.mjtGeom(int(m.geom_type[g])), size=m.geom_size[g],
                                                    pos=m.geom_pos[g], quat=m.geom_quat[g])
        sg.contype, sg.conaffinity, sg.condim = int(m.geom_contype[g]), int(m.geom_conaffinity[g]), int(m.geom_condim[g])
        sg.friction, sg.solref, sg.solimp = m.geom_friction[g], m.geom_solref[g], m.geom_solimp[g]
    for p in range(m.npair):
        sp = s.add_pair(name=name(mujoco.mjtObj.mjOBJ_PAIR, p), geomname1=name(mujoco.mjtObj.mjOBJ_GEOM, m.pair_geom1[p]),
                        geomname2=name(mujoco.mjtObj.mjOBJ_GEOM, m.pair_geom2[p]))
        sp.condim, sp.friction, sp.solref, sp.solimp = int(m.pair_dim[p]), m.pair_friction[p], m.pair_solref[p], \
            m.pair_solimp[p]
        sp.margin, sp.gap = float(m.pair_margin[p]), float(m.pair_gap[p])
    for a in range(m.nu):
        sa = s.add_actuator(name=name(mujoco.mjtObj.mjOBJ_ACTUATOR, a), trntype=mujoco.mjtTrn(int(m.actuator_trntype[a])),
                            target=name(mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0]))
        sa.gaintype, sa.biastype = mujoco.mjtGain(int(m.actuator_gaintype[a])), mujoco.mjtBias(int(m.actuator_biastype[a]))
        sa.gainprm, sa.biasprm, sa.gear = m.actuator_gainprm[a], m.actuator_biasprm[a], m.actuator_gear[a]
        sa.ctrllimited, sa.ctrlrange = lim(m.actuator_ctrllimited[a]), m.actuator_ctrlrange[a]
        sa.forcelimited = lim(m.actuator_forcelimited[a])
    return s


def _joint(s, sub):
    return next(j for j in s.joints if sub in j.name)


def variants():
    """name -> (description, spec edit, scope, the error codes it must add[, an edit of the compiled model])."""
    def contype(s):  # mjlab's raw contact model without its exclude list: pelvis - hip roll contacts
        for g in s.geoms:
            if g.name in ("pelvis_collision", "left_hip_collision", "right_hip_collision"):
                g.contype, g.conaffinity = 1, 1

    def contype_covered(s):  # every filter pair of these geoms is also an explicit pair
        for g in s.geoms:
            if g.name in ("left_hand_collision", "right_hand_collision", "left_wrist_collision"):
                g.contype, g.conaffinity = 1, 1

    def condim4(s):
        p = next(p for p in s.pairs if p.name.startswith("left_foot") and p.name.endswith("_floor"))
        p.condim = 4

    def damping(s):
        for j in s.joints:
            if "knee" in j.name or "ankle" in j.name:
                j.damping = [0.5, 0.0, 0.0]

    def equality(s):
        s.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="left_wrist_yaw_joint", name2="right_wrist_yaw_joint")

    def euler(s):
        s.option.integrator = mujoco.mjtIntegrator.mjINT_EULER

    def negsolref(s):
        for p in s.pairs:
            if p.name.endswith("_floor") and "foot" in p.name:
                p.solref = [-5000.0, -100.0]

    def qpos0(s):
        _joint(s, "left_knee").ref = 0.3

    def tendon(s):
        t = s.add_tendon(name="t_knees", stiffness=50.0)
        t.wrap_joint("left_knee_joint", 1.0)
        t.wrap_joint("right_knee_joint", -1.0)

    def gravcomp(s):
        s.body("torso_link").gravcomp = 1.0

    def cg_solver(s):
        s.option.solver = mujoco.mjtSolver.mjSOL_CG

    def act_damping(s):  # MuJoCo 3.17: lands in actuator_damping, not dof_damping
        for a in s.actuators:
            if "knee" in a.name:
                a.damping = [0.5, 0.0, 0.0]

    def act_armature(s):
        for a in s.actuators:
            if "knee" in a.name:
                a.armature = 0.05

    def act_delay(s):
        a = next(a for a in s.actuators if "knee" in a.name)
        a.delay, a.nsample = 0.01, 4

    def dampingpoly(s):
        _joint(s, "left_ankle_pitch").damping = [0.0, 0.2, 0.0]

    def motor(s):
        a = next(a for a in s.actuators if "left_knee" in a.name)
        a.biastype, a.biasprm = mujoco.mjtBias.mjBIAS_NONE, np.zeros(10)

    def cylinder(s):
        g = next(g for g in s.geoms if g.name == "left_shin_collision")
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER

    def ball(s):
        j = _joint(s, "left_wrist_roll")
        j.type, j.range = mujoco.mjtJoint.mjJNT_BALL, [0.0, 1.0]
        for a in s.actuators:  # (a position shortcut's inheritrange needs a hinge or slide joint)
            if a.target == j.name:
                a.inheritrange = 0.0

    def lockwaist(s):  # the G1 with the waist roll and pitch welded (27 dof), fusestatic
        for k in list(s.keys):  # (keyframes of the full G1 have the wrong qpos size)
            s.delete(k)
        for jn in ("waist_roll_joint", "waist_pitch_joint"):
            for a in [a for a in s.actuators if a.target == jn]:
                s.delete(a)
            s.delete(s.joint(jn))
        s.compiler.fusestatic = True

    def lockwaist_renamed(s):  # ... with the fused body named torso_link (the COM-offset body)
        lockwaist(s)
        s.body("torso_link").name = "torso_link_fused"
        s.body("waist_yaw_link").name = "torso_link"

    def solref_damping(s):  # MuJoCo Warp reads solref[1] <= 0 as a direct damping on its own (here: none)
        for p in s.pairs:
            if p.name.endswith("_floor") and "foot" in p.name:
                p.solref = [0.02, 0.0]

    def boxbox(s):  # a box-box pair (forward only: the adjoint has no boxes), native CCD at MuJoCo's defaults
        for side in ("left", "right"):
            s.body(f"{side}_ankle_roll_link").add_geom(name=f"{side}_ankle_box", type=mujoco.mjtGeom.mjGEOM_BOX,
                                                       size=[0.03, 0.03, 0.01], contype=0, conaffinity=0, mass=0.0)
        s.add_pair(name="ankle_box_box", geomname1="left_ankle_box", geomname2="right_ankle_box")

    def boxbox_ccd(s):  # ... with fewer native-CCD iterations than MuJoCo's default
        boxbox(s)
        s.option.ccd_iterations = 5

    def renamed(s):  # harmless: names of a body and a joint (with its actuator's target), not torso_link
        s.body("left_knee_link").name = "lk_renamed"
        j = _joint(s, "right_elbow")
        for a in [a for a in s.actuators if a.target == j.name]:
            a.target = "re_renamed"
        j.name = "re_renamed"

    def visual(s):  # harmless: visual-only geoms (no collision, no mass), a mesh, a camera, a light
        s.add_mesh(name="vis_cube", uservert=[x * 0.05 for v in range(8) for x in
                                              ((v & 1) * 2 - 1, (v >> 1 & 1) * 2 - 1, (v >> 2 & 1) * 2 - 1)])
        for b in ("pelvis", "torso_link", "left_knee_link"):
            s.body(b).add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="vis_cube", contype=0, conaffinity=0,
                               group=2, mass=0.0)
            s.body(b).add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.05, 0.0], contype=0,
                               conaffinity=0, group=1, mass=0.0)
        s.worldbody.add_camera(name="vis_cam", pos=[2.0, 0.0, 1.0])
        s.worldbody.add_light(name="vis_light", pos=[0.0, 0.0, 3.0])

    def sensors(s):  # harmless: sites and sensors (the engine computes none)
        s.body("torso_link").add_site(name="imu_site")
        s.add_sensor(name="s_gyro", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE,
                     objname="imu_site")
        s.add_sensor(name="s_knee", type=mujoco.mjtSensor.mjSENS_JOINTPOS, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                     objname="left_knee_joint")

    def gravity(s):  # harmless: another gravity vector (packed into the constants)
        s.option.gravity = [0.3, 0.0, -9.6]

    def dsbl_noop(s):  # harmless: disable flags of physics the model does not have
        for f in ("mjDSBL_SPRING", "mjDSBL_DAMPER", "mjDSBL_EQUALITY", "mjDSBL_FRICTIONLOSS"):
            s.option.disableflags |= int(mujoco.mjtDisableBit.__members__[f])

    def dsbl_contact(s):
        s.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

    def dsbl_frictionloss(s):  # (with frictionloss: the compiled-model edit frictionloss below)
        s.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS)

    def frictionloss(m):  # a compiled-model edit: a frictionloss set after compiling
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_joint")
        m.dof_frictionloss[m.jnt_dofadr[j]] = 0.1

    return dict(
        contype=("contype / conaffinity 1 on the pelvis and hip geoms (pelvis - hip roll: generated implicit pair)", contype,
                 "engine", set()),
        contype_covered=("contype / conaffinity 1 on 3 hand / wrist geoms whose filter pairs are all explicit pairs "
                         "(MuJoCo collides a geom pair once, with the pair's parameters): supported", contype_covered,
                         "engine", set()),
        condim4=("condim 4 on one foot-floor pair", condim4, "engine", {"H17"}),
        damping=("dof damping 0.5 on knees and ankles", damping, "engine", set()),
        equality=("soft joint equality: left wrist yaw = right wrist yaw", equality, "engine", set()),
        euler=("Euler integrator", euler, "engine", {"H23"}),
        negsolref=("direct (negative) solref on the foot-floor pairs", negsolref, "engine", {"H19"}),
        qpos0=("left knee ref 0.3 (qpos0 != 0)", qpos0, "engine", {"H8"}),
        tendon=("fixed tendon with stiffness over the knees", tendon, "engine", {"H25"}),
        gravcomp=("gravcomp 1 on the torso", gravcomp, "engine", {"H22"}),
        cg_solver=("CG solver", cg_solver, "engine", {"H24a"}),
        act_damping=("actuator damping 0.5 on the knee actuators (actuator_damping)", act_damping, "engine", {"H22"}),
        act_armature=("actuator armature 0.05 on the knee actuators (actuator_armature)", act_armature, "engine",
                      {"H11d"}),
        act_delay=("a 10 ms delay on one knee actuator (actuator_delay, history)", act_delay, "engine", {"H11e"}),
        dampingpoly=("a quadratic damping term on one ankle (dof_dampingpoly)", dampingpoly, "engine", {"H22"}),
        motor=("one motor actuator (no position bias)", motor, "engine", {"H11a"}),
        cylinder=("a cylinder collision geom in the pairs", cylinder, "engine",
                  {"H14a", "H14b", "H14c"}),
        ball=("a limited ball joint on one wrist body", ball, "engine", {"H7", "H7b", "H11a", "H20"}),
        frictionloss=("frictionloss 0.1 on one knee: supported by forward and adjoint", None, "engine", set(),
                      frictionloss),
        lockwaist=("waist roll and pitch welded (fusestatic, nv 33): torso_link fused away; untested paths", lockwaist,
                   "engine", {"H13"}),
        lockwaist_renamed=("the welded waist with the fused body named torso_link: supported, on the untested nv-33 "
                           "Cholesky tail and the scalar KDOF-14 path (warnings)", lockwaist_renamed, "engine", set()),
        solref_damping=("solref (0.02, 0) on the foot-floor pairs: a direct damping 0 for MuJoCo Warp", solref_damping,
                        "engine", {"H19"}),
        boxbox=("a box-box pair at MuJoCo's native-CCD defaults (forward scope): supported", boxbox, "forward",
                set()),
        boxbox_ccd=("a box-box pair with ccd_iterations 5 (forward scope)", boxbox_ccd, "forward", {"H14d"}),
        renamed=("bodies, joints and geoms renamed (not torso_link): supported", renamed, "engine", set()),
        visual=("visual-only mesh and cylinder geoms, a camera and a light: supported", visual, "engine", set()),
        sensors=("a site, a gyro and a joint-position sensor: supported", sensors, "engine", set()),
        gravity=("gravity (0.3, 0, -9.6): supported (packed into the constants)", gravity, "engine", set()),
        dsbl_noop=("springs, dampers, equality constraints and frictionloss disabled, none in the model: supported",
                   dsbl_noop, "engine", set()),
        dsbl_contact=("contacts disabled (mjDSBL_CONTACT): generated collision pass skipped", dsbl_contact, "engine", set()),
        dsbl_frictionloss=("frictionloss disabled with frictionloss 0.1 on one knee", dsbl_frictionloss, "engine",
                           {"H24b"}, frictionloss),
    )


WARNINGS = {"lockwaist": {"U1", "U2"}, "lockwaist_renamed": {"U1", "U2"}}  # the new warnings a variant must give


def base_findings(rep):
    return {f.code for f in rep.errors()}


def run_variants(make_spec, finish, foot, res, tag):
    """Each variant: its new error codes against the base (made by make_spec, finished by finish) and the entry points'
    verdicts. Fills res, returns the pass."""
    from nablaengine import jax_ffi
    base = finish(make_spec().compile())
    rep0 = capability.check(base, foot)
    res["base"] = dict(errors=rep0.codes("error"), warnings=rep0.codes("warn"), infos=rep0.codes("info"))
    ok = not rep0.errors() and not rep0.warnings()
    res["variants"] = {}
    for name, (desc, edit, scope, want, *model_edit) in variants().items():
        s = make_spec()
        if edit:
            edit(s)
        r = dict(desc=desc, scope=scope, expected=sorted(want))
        try:
            vm = finish(s.compile())
            for e in model_edit:
                e(vm)
        except Exception as e:  # noqa: BLE001
            r.update(compile_error=f"{type(e).__name__}: {e}"[:300], ok=False)
            res["variants"][name] = r
            ok = False
            continue
        rep = capability.check(vm, foot, scope=scope)
        new = set(rep.codes("error")) - set(rep0.codes("error"))
        new_warn = set(rep.codes("warn")) - set(rep0.codes("warn"))
        r.update(new_errors=sorted(new), warnings=rep.codes("warn"), new_warnings=sorted(new_warn),
                 expected_warnings=sorted(WARNINGS.get(name, set())),
                 findings=[f.line() for f in rep.errors()])
        entry = {}
        try:
            capability.require(vm, foot, scope=scope, context=f"{tag} variant {name}")
            entry["require"] = "accepted"
        except capability.UnsupportedModel as e:
            entry["require"] = "UnsupportedModel " + ",".join(e.report.codes("error"))
        if entry["require"] != "accepted":  # (init on a model the checker passes would build the library)
            try:
                jax_ffi.init(model=vm, foot_pairs=foot)
                entry["jax_ffi.init"] = "accepted"
            except capability.UnsupportedModel as e:
                entry["jax_ffi.init"] = "UnsupportedModel " + ",".join(e.report.codes("error"))
            except Exception as e:  # noqa: BLE001
                entry["jax_ffi.init"] = f"{type(e).__name__}: {str(e)[:120]}"
        r["entry_points"] = entry
        r["ok"] = bool(new == want and all(v.startswith("UnsupportedModel") == bool(want) for v in entry.values())
                       and (new_warn == WARNINGS.get(name, set()) if (not want or name in WARNINGS) else True))
        ok &= r["ok"]
        res["variants"][name] = r
        print(f"  {tag} {name:13s} {'ok ' if r['ok'] else 'BAD'} new {sorted(new)} (want {sorted(want)}) "
              f"warnings {sorted(new_warn)} {entry}", flush=True)
    return ok


def section_fields(res):
    m, _ = fixtures.load_model()
    unknown = fields.unclassified(m)
    names = fields.model_fields(m)
    kinds = {}
    for n in names:
        kinds[fields.classify(n).kind if fields.classify(n) else "unclassified"] = kinds.get(
            fields.classify(n).kind if fields.classify(n) else "unclassified", 0) + 1
    d = fields.default_rows()
    codes = {f.code for f in list(fields.FIELDS.values()) + list(fields.SIZES.values()) +
             [f for _, f in fields.PREFIXES] if f.code}
    rep = capability.check(m, None)
    checker_codes = {f.code for f in rep.findings} | {"H11d", "H11e", "H16"}  # (default-only codes)
    stray = sorted(codes - checker_codes)
    stale = ([k for k in list(fields.FIELDS) + list(fields.SIZES) if k not in names]  # classified, not a field here
             + [p + "*" for p, _ in fields.PREFIXES if not any(n.startswith(p) for n in names)])
    res.update(fields=len(names), kinds=kinds, unclassified=unknown, defaults=sorted(d), stale=stale,
               codes_without_check=stray, mujoco=mujoco.__version__)
    return not unknown and len(d) == len(fields.default_fields()) and not stray and not stale


def section_fixtures(res):
    ok = True
    m, mt = fixtures.load_model()  # (loading checks the forward scope)
    for scope in capability.SCOPES:
        rep = capability.check(m, mt["foot_pairs"], scope=scope)
        res[f"mjlab_{scope}"] = dict(errors=rep.codes("error"), warnings=rep.codes("warn"), infos=rep.codes("info"))
        ok &= not rep.errors() and not rep.warnings()
    ok &= res["mjlab_engine"]["infos"] == ["H11c"]
    return ok


def section_variants(res):
    m, mt = fixtures.load_model()
    s0 = spec_from_model(m)
    rebuilt = s0.compile()
    same = {k: bool(np.allclose(getattr(rebuilt, k), getattr(m, k))) for k in (
        "body_parentid", "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat", "jnt_type",
        "jnt_range", "jnt_axis", "dof_armature", "geom_type", "geom_size", "pair_geom1", "pair_geom2", "pair_dim",
        "pair_solref", "pair_friction", "actuator_gainprm", "actuator_biasprm", "qpos0")}
    res["rebuilt_equals_fixture"] = same
    ok = all(same.values())
    ok &= run_variants(lambda: spec_from_model(m), lambda x: x, mt["foot_pairs"], res, "spec")
    return ok


def section_kinematics(res):
    m, _ = fixtures.load_model()
    kinematics.check_model(m)
    xml = """<mujoco><worldbody><body name="root"><freejoint/><geom type="sphere" size="0.1"/>
      <body name="a" pos="0 0 -0.2"><joint name="ja" axis="0 1 0"/><geom type="capsule" size="0.05 0.1"/>
        <body name="welded" pos="0 0 -0.2"><geom type="sphere" size="0.05"/></body>
      </body></body></worldbody></mujoco>"""
    w = mujoco.MjModel.from_xml_string(xml)
    try:
        kinematics.check_model(w)
        verdict = "accepted"
    except AssertionError as e:  # UnsupportedModel is an AssertionError
        verdict = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    res.update(g1="accepted", jointless_body=verdict, body_jntadr=[int(x) for x in w.body_jntadr])
    return verdict != "accepted" and int(w.body_jntadr[3]) == -1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--out", default="", help="result JSON")
    args = ap.parse_args()
    sections = [s for s in args.sections.split(",") if s]
    bad = set(sections) - set(SECTIONS)
    if bad:
        ap.error(f"unknown sections {sorted(bad)}")
    res, ok = dict(sections={}), True
    for s in sections:
        r = {}
        try:
            passed = globals()[f"section_{s}"](r)
        except Exception:  # noqa: BLE001
            r["exception"] = traceback.format_exc()[-3000:]
            passed = False
        r["pass"] = bool(passed)
        res["sections"][s] = r
        ok &= passed
        print(f"{s}: {'PASS' if passed else 'FAIL'}", flush=True)
        if not passed and "exception" in r:
            print(r["exception"], file=sys.stderr)
    res["pass"] = bool(ok)
    res["provenance"] = provenance.provenance()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"capability check: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
