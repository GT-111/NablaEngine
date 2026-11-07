"""Build fixtures/h1 (the Unitree H1 for the engine) from MuJoCo Menagerie's unitree_h1, after the recipe that turned
Menagerie's G1 into the mjlab G1 (fixtures/mjlab/README.md, "How it was derived"):

  1. collision geometry: the cylinders become capsules of the same radius and half-length, the torso box a capsule
     (radius 0.1 m, half-length 0.12 m); every robot collision geom (32: 26 capsules, 6 of them on the soles, and 6
     spheres) leaves contype / conaffinity, the floor too
  2. contact pairs: every collision geom with the floor (<geom>__floor; condim 3), then every pair of collision geoms
     on different bodies except parent-child bodies and Menagerie's two excludes (torso / shoulder-roll links), in
     geom order (<geom1>__<geom2>; condim 3 with a sole capsule, else 1); sliding friction 0.6 between two sole
     capsules, else 1, torsional 0.005, rolling 0.0001; MuJoCo's default solref / solimp. The pairs whose geoms
     already touch at the home keyframe are left out (3: each hip-yaw / hip-pitch link, the two hip-roll links; the
     capsules reach past the cylinders' ends and would push against the servos forever)
  3. visual content: the mesh geoms, the mesh assets, the materials and the textures (the floor's checker, the
     skybox) are removed
  4. options: timestep 5 ms (Menagerie: 2 ms), implicitfast (Menagerie: Euler), Euler damping disabled
  5. actuators: the torque motors become position servos with mjlab's gains from the armature (Menagerie's 0.1 on
     every joint): kp = armature (2 pi 10 Hz)^2 = 394.8, kd = 2 * 2 * armature (2 pi 10 Hz) = 25.1; the motor torque
     limits (Menagerie's ctrlrange) become the joints' actuator force ranges; the setpoint range is the joint range
     widened on each side by tau_max / kp (mjlab's); no joint damping (Menagerie: 1 N m s) and no friction loss
  6. on the compiled model: joint-limit solref / solimp at MuJoCo's defaults (0.02, 1) and (0.9, 0.95, 0.001, 0.5, 2)

The result is written with mj_saveModel (MJCF's writer rounds numbers) with h1_meta.json; the sole-floor pairs are the
meta's foot_pairs (the per-world friction), torso_link takes the COM offset. The H1 has nv 25: the engine's solves
mask the lanes past NV (capability warning H3). Applied to Menagerie at the commit below with mujoco 3.17.0 it
reproduces fixtures/h1/h1_train.mjb byte for byte (its sha256 is in fixtures/h1/README.md).

  git clone https://github.com/google-deepmind/mujoco_menagerie && git -C mujoco_menagerie checkout 98a70c5
  python scripts/make_h1_model.py --menagerie mujoco_menagerie --out fixtures/h1
"""

import argparse
import hashlib
import itertools
import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

MENAGERIE_COMMIT = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
G = mujoco.mjtGeom
EXCLUDE = {frozenset(("torso_link", "left_shoulder_roll_link")), frozenset(("torso_link", "right_shoulder_roll_link"))}
HZ, ZETA = 10.0, 2.0  # mjlab's PD design: natural frequency and damping ratio


def build(scene):
    spec = mujoco.MjSpec.from_file(str(scene))

    # 1. collision geometry; 3. the visual meshes go
    coll = []
    for g in list(spec.geoms):
        if g.contype == 0 and g.conaffinity == 0:
            spec.delete(g)
            continue
        if g.type == G.mjGEOM_PLANE:
            g.contype = g.conaffinity = 0
            g.material = ""
            continue
        if g.type == G.mjGEOM_CYLINDER:
            g.type = G.mjGEOM_CAPSULE
        elif g.type == G.mjGEOM_BOX:
            g.type, g.size = G.mjGEOM_CAPSULE, [0.1, 0.12, 0]
        assert g.type in (G.mjGEOM_CAPSULE, G.mjGEOM_SPHERE), g.name
        g.contype = g.conaffinity = 0
        coll.append(g)
    for a in list(spec.meshes) + list(spec.materials) + list(spec.textures):
        spec.delete(a)
    for i, g in enumerate(coll):
        g.name = g.name or f"{g.parent.name}_c{i}"

    # 2. contact pairs
    for e in list(spec.excludes):
        spec.delete(e)
    parent = {b.name: b.parent.name for b in spec.bodies if b.name != "world"}
    sole = lambda g: bool(g.classname and g.classname.name.startswith("foot"))  # noqa: E731
    foot_pairs = []
    for g in coll:
        p = spec.add_pair(name=f"{g.name}__floor", geomname1="floor", geomname2=g.name, condim=3)
        p.friction = [1.0, 1.0, 0.005, 0.0001, 0.0001]
        if sole(g):
            foot_pairs.append(p.name)
    for a, b in itertools.combinations(coll, 2):
        ba, bb = a.parent.name, b.parent.name
        if ba == bb or parent[ba] == bb or parent[bb] == ba or frozenset((ba, bb)) in EXCLUDE:
            continue
        p = spec.add_pair(name=f"{a.name}__{b.name}", geomname1=a.name, geomname2=b.name,
                          condim=3 if sole(a) or sole(b) else 1)
        f = 0.6 if sole(a) and sole(b) else 1.0
        p.friction = [f, f, 0.005, 0.0001, 0.0001]

    # 4. options
    spec.option.timestep = 0.005
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP

    # 5. actuators
    w = 2 * np.pi * HZ
    joints = {j.name: j for j in spec.joints}
    for a in spec.actuators:
        j = joints[a.target]
        tau = float(a.ctrlrange[1])
        kp, kd = j.armature * w ** 2, 2 * ZETA * j.armature * w
        a.set_to_position(kp=kp, kv=kd)
        a.ctrllimited, a.forcelimited = True, False
        a.ctrlrange = [j.range[0] - tau / kp, j.range[1] + tau / kp]
        j.actfrclimited, j.actfrcrange = True, [-tau, tau]
        j.damping = [0.0, 0.0, 0.0]
        j.frictionloss = 0.0

    # 2, continued: the robot-robot pairs in contact at the home keyframe
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    touching = {frozenset((m.geom(c.geom1).name, m.geom(c.geom2).name)) for c in d.contact[:d.ncon]
                if floor not in (c.geom1, c.geom2)}
    dropped = [p.name for p in spec.pairs if frozenset((p.geomname1, p.geomname2)) in touching]
    for p in list(spec.pairs):
        if p.name in dropped:
            spec.delete(p)
    m = spec.compile()

    # 6. joint-limit parameters
    m.jnt_solref[:] = [0.02, 1.0]
    m.jnt_solimp[:] = [0.9, 0.95, 0.001, 0.5, 2.0]
    return m, [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_PAIR, n) for n in foot_pairs], dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--menagerie", required=True, help="a MuJoCo Menagerie checkout (commit " + MENAGERIE_COMMIT[:7] + ")")
    ap.add_argument("--out", default="fixtures/h1", help="output directory")
    a = ap.parse_args()
    src = Path(a.menagerie) / "unitree_h1"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    m, foot_pairs, dropped = build(src / "scene.xml")
    mjb = out / "h1_train.mjb"
    mujoco.mj_saveModel(m, str(mjb), None)
    meta = dict(mujoco_version=mujoco.__version__, foot_pairs=foot_pairs, torso_body="torso_link", nq=m.nq, nv=m.nv,
                nu=m.nu, nbody=m.nbody, npair=m.npair, timestep=m.opt.timestep, substeps_per_control_step=4,
                solver="Newton", cone="pyramidal", integrator="implicitfast", robot="h1",
                source=f"MuJoCo Menagerie unitree_h1/scene.xml at {MENAGERIE_COMMIT[:7]} (Unitree Robotics, "
                       "BSD-3-Clause), converted by scripts/make_h1_model.py after the mjlab G1 recipe: capsule and "
                       "sphere collision geoms with explicit pairs, position servos, implicitfast, 5 ms timestep; "
                       "meshes removed. See README.md.")
    (out / "h1_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    shutil.copyfile(src / "LICENSE", out / "LICENSE")
    print(f"{mjb}: nbody {m.nbody} nv {m.nv} nu {m.nu} npair {m.npair} ({len(foot_pairs)} sole-floor pairs; dropped "
          f"{', '.join(dropped)}), {mjb.stat().st_size} bytes, sha256 {hashlib.sha256(mjb.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
