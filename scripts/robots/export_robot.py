"""Export a robot scene as an engine model: an mjlab robot's flat velocity scene (run in an mjlab environment)
or a scene MJCF such as MuJoCo Playground's (robot + floor plane).

mjlab builds the scene (robot entity with its actuators and collision config, the plane terrain) and its MuJoCo
options; here the visual meshes are dropped and mjlab's contype / conaffinity contacts become explicit contact pairs
with MuJoCo's mixed parameters (priority, else max condim / friction and solmix-weighted solref / solimp; margin and
gap are max), so that the engine (explicit pairs only) simulates the same contacts. Writes OUT.xml and OUT.json
(foot pairs by name, the COM body, the control step's substeps).

  python scripts/robots/export_robot.py --robot go1 --out fixtures/robots/go1/go1_flat
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ROBOTS = {  # name: (env cfg factory, foot geom names, COM body)
    "go1": ("mjlab.tasks.velocity.config.go1.env_cfgs:unitree_go1_flat_env_cfg",
            [f"{n}_foot_collision" for n in ("FR", "FL", "RR", "RL")], "trunk"),
}


def mixed(m, g1, g2):
    """MuJoCo's contact parameters of dynamic geom pair (g1, g2): condim, friction (5), solref, solimp, margin, gap."""
    p1, p2 = m.geom_priority[g1], m.geom_priority[g2]
    f1, f2 = m.geom_friction[g1], m.geom_friction[g2]
    if p1 != p2:
        g = g1 if p1 > p2 else g2
        condim, fr = m.geom_condim[g], m.geom_friction[g]
        solref, solimp = m.geom_solref[g], m.geom_solimp[g]
    else:
        condim, fr = max(m.geom_condim[g1], m.geom_condim[g2]), np.maximum(f1, f2)
        s1, s2 = m.geom_solmix[g1], m.geom_solmix[g2]
        mix = 0.5 if s1 + s2 < mujoco.mjMINVAL else s1 / (s1 + s2)
        r1, r2 = m.geom_solref[g1], m.geom_solref[g2]
        solref = mix * r1 + (1 - mix) * r2 if (r1[0] > 0 and r2[0] > 0) else np.minimum(r1, r2)
        solimp = mix * m.geom_solimp[g1] + (1 - mix) * m.geom_solimp[g2]
    margin = max(m.geom_margin[g1], m.geom_margin[g2])
    gap = max(m.geom_gap[g1], m.geom_gap[g2])
    return int(condim), [fr[0], fr[0], fr[1], fr[2], fr[2]], list(solref), list(solimp), float(margin), float(gap)


def convert(spec, out, name, source, foot_geoms, com_body, substeps, init_qpos, feet_only=False, locomotion=False,
            servo=None):
    """The engine model of scene spec (robot + floor) written to OUT.xml / OUT.json: visual meshes dropped, the static
    bodies' geoms onto the world body, actuators in joint order with their force limits on the joints, dynamic
    contacts as explicit pairs (MuJoCo's mixed parameters); init_qpos(m): the initial state as qpos; feet_only: only
    the foot geoms' pairs with the floor (MuJoCo Playground's feet-only collision, for full-collision models);
    locomotion: the usual locomotion contact model (pyramidal cone, condim 3, no margin) for models such as MuJoCo
    Menagerie's (elliptic, condim 6); servo (kp, kd): torque motors as position servos (kd as joint damping)."""
    for g in list(spec.geoms):  # visual meshes (no contacts)
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            assert g.contype == 0 and g.conaffinity == 0, g.name
            spec.delete(g)
    for x in list(spec.meshes):  # (builtin textures stay; file textures go with the visuals)
        spec.delete(x)
    for t in list(spec.textures):
        if t.file:
            for mat in spec.materials:
                mat.textures = ["" if x == t.name else x for x in mat.textures]
            spec.delete(t)
    # the terrain's static body: its geoms onto the world body (the engine wants one tree under the world)
    for body in list(spec.worldbody.bodies):
        if body.joints or body.bodies:
            continue
        assert not np.any(body.pos) and np.allclose(body.quat, [1, 0, 0, 0]), body.name
        for g in list(body.geoms):
            ng = spec.worldbody.add_geom()
            for k in ("type", "size", "pos", "quat", "rgba", "material", "friction", "condim", "solref", "solimp",
                      "solmix", "priority", "margin", "gap", "contype", "conaffinity", "group"):
                setattr(ng, k, getattr(g, k))
            gname = g.name
            spec.delete(g)
            ng.name = gname
        for light in list(body.lights):
            spec.delete(light)
        spec.delete(body)
    # actuators in joint order (the engine: actuator a drives dof 6 + a); mjlab orders them by actuator group
    order = [j.name for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
    acts = {act.target: act for act in spec.actuators}
    assert sorted(acts) == sorted(order), (sorted(acts), order)
    keep = ("name", "trntype", "target", "dyntype", "gaintype", "biastype", "dynprm", "gainprm", "biasprm", "gear",
            "ctrllimited", "ctrlrange", "forcelimited", "forcerange", "actlimited", "actrange", "group")
    saved = [{k: (list(v) if hasattr(v, "__len__") and not isinstance(v, str) else v)
              for k in keep for v in [getattr(acts[n], k)]} for n in order]
    for act in list(spec.actuators):
        spec.delete(act)
    for d in saved:
        if servo and d["biastype"] == mujoco.mjtBias.mjBIAS_NONE:  # a torque motor: a PD position servo
            j = spec.joint(d["target"])
            d["forcelimited"], d["forcerange"] = True, list(d["ctrlrange"])
            d["ctrlrange"], d["ctrllimited"] = list(j.range), True
            d["biastype"] = mujoco.mjtBias.mjBIAS_AFFINE
            d["gainprm"] = [servo[0]] + [0.0] * (len(d["gainprm"]) - 1)
            d["biasprm"] = [0.0, -servo[0]] + [0.0] * (len(d["biasprm"]) - 2)
            j.damping = [j.damping[0] + servo[1], *j.damping[1:]]  # (linear coefficient; higher orders stay 0)
        act = spec.add_actuator()
        for k, v in d.items():
            setattr(act, k, v)
    # actuator force limits as the joints' actuator force ranges (the same clamp for one gear-1 servo per joint)
    for act in spec.actuators:
        if act.forcelimited and act.forcerange[0] < act.forcerange[1]:  # (auto-limited with an empty range: none)
            j = spec.joint(act.target)
            j.actfrcrange, j.actfrclimited = list(act.forcerange), True
            act.forcelimited, act.forcerange = False, [0.0, 0.0]
    for s in list(spec.sensors):
        spec.delete(s)
    m = spec.compile()
    # mjlab's dynamic contacts as explicit pairs (contype / conaffinity filter, no pairs within a body or between a
    # parent and its child body: MuJoCo's filterparent; the world body's geoms collide with every body)
    weld = m.body_weldid
    pairs = []
    for g1 in range(m.ngeom):
        for g2 in range(g1 + 1, m.ngeom):
            if not ((m.geom_contype[g1] & m.geom_conaffinity[g2]) or (m.geom_contype[g2] & m.geom_conaffinity[g1])):
                continue
            b1, b2 = weld[m.geom_bodyid[g1]], weld[m.geom_bodyid[g2]]
            if b1 == b2:
                continue
            if b1 != 0 and b2 != 0 and (m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1):
                continue
            if feet_only and not ((b1 == 0 and m.geom(g2).name in foot_geoms) or
                                  (b2 == 0 and m.geom(g1).name in foot_geoms)):
                continue
            pairs.append((g1, g2))
    names = [m.geom(g).name for g in range(m.ngeom)]
    for i, (g1, g2) in enumerate(pairs):
        condim, fr, solref, solimp, margin, gap = mixed(m, g1, g2)
        if locomotion:
            condim, margin, gap = min(condim, 3), 0.0, 0.0
        spec.add_pair(name=f"p{i}_{names[g1]}__{names[g2]}", geomname1=names[g1], geomname2=names[g2], condim=condim,
                      friction=fr, solref=solref, solimp=solimp, margin=margin, gap=gap)
    for g in spec.geoms:
        g.contype = g.conaffinity = 0
    if locomotion:
        spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m2 = spec.compile()
    spec.to_file(str(out.with_suffix(".xml")))
    xml = out.with_suffix(".xml").read_text()  # newer MuJoCo writes an empty unnamed <default/>, which 3.17 rejects
    out.with_suffix(".xml").write_text("\n".join(l for l in xml.split("\n") if l.strip() != "<default/>"))
    feet = foot_geoms
    is_foot = lambda g: m2.geom(int(g)).name.split("/")[-1] in feet  # noqa: E731  (entity prefixes, e.g. robot/)
    static = lambda g: m2.geom_bodyid[int(g)] == 0  # noqa: E731  (a foot against the floor, not another foot)
    foot_pairs = [p for p in range(m2.npair) if (is_foot(m2.pair_geom1[p]) and static(m2.pair_geom2[p]))
                  or (is_foot(m2.pair_geom2[p]) and static(m2.pair_geom1[p]))]
    q0 = init_qpos(m2)
    meta = dict(robot=name, source=source, feet_only=feet_only, locomotion=locomotion, servo=servo, init_qpos=[float(x) for x in q0], foot_pairs=foot_pairs, foot_geoms=feet,
                com_body=next(m2.body(b).name for b in range(m2.nbody) if m2.body(b).name.split("/")[-1] == com_body), substeps_per_control_step=int(substeps), nq=m2.nq, nv=m2.nv, nu=m2.nu,
                nbody=m2.nbody, npair=m2.npair, timestep=float(m2.opt.timestep), mujoco_version=mujoco.__version__)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


def from_mjlab(robot, out):
    import importlib
    import re
    from mjlab.scene import Scene
    mod, fn = ROBOTS[robot][0].split(":")
    cfg = getattr(importlib.import_module(mod), fn)()
    cfg.scene.num_envs = 1
    scene = Scene(cfg.scene, device="cpu")
    spec = scene.spec
    mo = cfg.sim.mujoco
    o = spec.option
    o.timestep, o.iterations, o.ls_iterations, o.tolerance, o.ls_tolerance = (mo.timestep, mo.iterations,
                                                                             mo.ls_iterations, mo.tolerance,
                                                                             mo.ls_tolerance)
    o.impratio, o.gravity, o.ccd_iterations = mo.impratio, list(mo.gravity), mo.ccd_iterations
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST if mo.integrator == "implicitfast" else o.integrator
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL if mo.cone == "pyramidal" else mujoco.mjtCone.mjCONE_ELLIPTIC
    o.solver = mujoco.mjtSolver.mjSOL_NEWTON
    ent = cfg.scene.entities["robot"]

    def init_qpos(m):  # mjlab's EntityCfg.init_state: root position, joint angles by regex
        q0 = m.qpos0.copy()
        q0[:3] = ent.init_state.pos
        for j in range(m.njnt):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            nm = m.joint(j).name.split("/")[-1]
            for pat, val in ent.init_state.joint_pos.items():
                if re.fullmatch(pat, nm):
                    q0[m.jnt_qposadr[j]] = val
        return q0

    convert(spec, out, robot, ROBOTS[robot][0], ROBOTS[robot][1], ROBOTS[robot][2], cfg.decimation, init_qpos)


def from_mjcf(path, out, name, foot_geoms, com_body, substeps, key=None, feet_only=False, z=None, foot_class=None,
              locomotion=False, servo=None):
    """A scene MJCF (robot + floor plane), e.g. MuJoCo Playground's; key: the keyframe of the initial state; z: the
    root height of the initial state instead (models without a keyframe); foot_class: the foot geoms are those of this
    default class (MuJoCo Menagerie's unnamed foot spheres, named BODY_foot here)."""
    spec = mujoco.MjSpec.from_file(str(path))
    if foot_class:
        foot_geoms = []
        for g in spec.geoms:
            if g.classname and g.classname.name == foot_class:
                g.name = g.name or f"{g.parent.name}_foot"
                foot_geoms.append(g.name)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON

    def init_qpos(m):
        q0 = m.qpos0.copy() if key is None else m.key(key).qpos.copy()
        if z is not None:
            q0[2] = z
        return q0

    convert(spec, out, name, str(path), foot_geoms, com_body, substeps, init_qpos, feet_only, locomotion, servo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=None, help="an mjlab robot (ROBOTS)")
    ap.add_argument("--mjcf", default=None, help="a scene MJCF instead (robot + floor plane)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--foot-geoms", default="", help="comma-separated (with --mjcf)")
    ap.add_argument("--com-body", default="torso_link")
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--key", default=None)
    ap.add_argument("--foot-class", default=None, help="the foot geoms by default class (with --mjcf)")
    ap.add_argument("--feet-only", action="store_true", help="only the foot geoms' pairs with the floor")
    ap.add_argument("--z", type=float, default=None, help="the initial root height (with --mjcf, no keyframe)")
    ap.add_argument("--locomotion", action="store_true", help="pyramidal cone, condim <= 3, no margin (Menagerie)")
    ap.add_argument("--servo", default=None, help="KP,KD: torque motors as position servos (with --mjcf)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mjcf:
        from_mjcf(a.mjcf, a.out, a.name or Path(a.mjcf).stem, [g for g in a.foot_geoms.split(",") if g], a.com_body,
                  a.substeps, a.key, a.feet_only, a.z, a.foot_class, a.locomotion,
                  [float(x) for x in a.servo.split(",")] if a.servo else None)
    else:
        from_mjlab(a.robot, a.out)


if __name__ == "__main__":
    main()
