"""A visual check of the engine: one G1 rolled out by nablaengine.jax_ffi and by MuJoCo C (float64), rendered side by
side (left the engine, right MuJoCo) into a video, with the per-step qpos deviation printed.

The world starts at a keyframe of the fixtures model (--keyframe, default knees_bent) at rest, nominal friction (1) and
no torso COM offset; the PD targets are the keyframe's joint positions plus an open-loop squat (0.5 Hz) and arm swing
(0.8 Hz), with no balance control: the robot falls after about 2 s, which exercises the contacts of the fall. MuJoCo
runs the same model with substeps_per_control_step mj_step calls per control step and the engine's Newton iteration cap
(opt.iterations = --iterations). Also printed: the gradient of the final pelvis height with respect to the first 50
control steps' targets (finite, through the analytic adjoint).

The robot's geoms are the collision capsules and spheres of geom group 3 (the fixtures model has no meshes), hidden by
MuJoCo's default visualization and translucent: they are drawn opaque. Rendering is offscreen (MUJOCO_GL=egl unless
set) and the frames are encoded by ffmpeg (on PATH, with the drawtext filter), which labels each half.

  source .dk_env/activate.sh
  python scripts/visualize_rollout.py --out runs/visualize

Writes <out>/rollout.mp4 and <out>/frame_<step>.png at --snapshots. Exit status: 0 when the largest |qpos| deviation
over the rollout is at most --tol, 1 otherwise.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from nablaengine import fixtures, jax_ffi  # noqa: E402

DT = 0.02  # one control step: 4 substeps of 5 ms


def joint_targets(m, base, t):
    """PD targets at time t: base (the keyframe's joint positions) plus the squat and the arm swing, in ctrlrange."""
    names = [m.joint(j).name for j in m.actuator_trnid[:, 0]]
    sel = lambda s: [i for i, n in enumerate(names) if s in n]  # noqa: E731
    c = base.copy()
    s = 0.5 * (1 - np.cos(2 * np.pi * 0.5 * t))
    c[sel("knee")] += 0.6 * s
    c[sel("hip_pitch")] -= 0.3 * s
    c[sel("ankle_pitch")] -= 0.3 * s
    a = np.sin(2 * np.pi * 0.8 * t)
    c[sel("left_shoulder_pitch")] += 0.8 * a
    c[sel("right_shoulder_pitch")] -= 0.8 * a
    c[sel("elbow")] += 0.4 * a
    return np.clip(c, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])


def engine_rollout(q0, ctrls, nv, iterations):
    """qpos (T, nq) of the engine's rollout from (q0, rest) and the gradient of the final pelvis height with respect
    to the first min(50, T) targets."""
    jax_ffi.init()
    qpos = jnp.asarray(q0[None], jnp.float32)
    zero = jnp.zeros((1, nv), jnp.float32)
    fric, com = jnp.ones((1,), jnp.float32), jnp.zeros((1, 3), jnp.float32)

    @jax.jit
    def rollout(u):
        def body(c, uk):
            c = jax_ffi.step(*c, uk[None], fric, com, iterations=iterations)
            return c, c[0][0]
        return jax.lax.scan(body, (qpos, zero, zero), u)[1]

    u = jnp.asarray(ctrls)
    g = jax.jit(jax.grad(lambda u: rollout(u)[-1, 2]))(u[:50])
    return np.asarray(rollout(u)), np.asarray(g)


def mujoco_rollout(m, d, ctrls, substeps):
    qs = np.empty((len(ctrls), m.nq))
    for k, u in enumerate(ctrls):
        d.ctrl[:] = u
        for _ in range(substeps):
            mujoco.mj_step(m, d)
        qs[k] = d.qpos
    return qs


def render(m, qs_left, qs_right, out, size, snapshots):
    """The two trajectories side by side: out/rollout.mp4 and out/frame_<k>.png for k in snapshots."""
    m.geom_rgba[m.geom_group == 3] = [0.85, 0.45, 0.15, 1.0]
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 2.8, -15, 135
    d = mujoco.MjData(m)

    # each half's name at its top left, white on a translucent box
    label = ("drawtext=font=sans:text='{}':x={}+16:y=16:fontsize={}:fontcolor=white:box=1:boxcolor=black@0.5:"
             "boxborderw=8")
    labels = ",".join(label.format(t, x, max(12, size // 20)) for t, x in (("NablaEngine", 0), ("MuJoCo", size)))

    def ffmpeg(*args):
        return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-s", f"{2 * size}x{size}", *args[:-1], "-vf", labels, args[-1]],
                                stdin=subprocess.PIPE)

    with mujoco.Renderer(m, size, size) as r:
        def frame(q):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            cam.lookat[:] = [q[0], q[1], 0.6]
            r.update_scene(d, cam, opt)
            return r.render()

        video = ffmpeg("-r", str(round(1 / DT)), "-i", "-", "-pix_fmt", "yuv420p", str(out / "rollout.mp4"))
        for k in range(len(qs_left)):
            img = np.concatenate([frame(qs_left[k]), frame(qs_right[k])], axis=1).tobytes()
            video.stdin.write(img)
            if k in snapshots:
                png = ffmpeg("-i", "-", "-frames:v", "1", str(out / f"frame_{k:03d}.png"))
                png.communicate(img)
        video.stdin.close()
        if video.wait():
            raise RuntimeError("ffmpeg failed to write the video")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="runs/visualize", help="output directory")
    ap.add_argument("--steps", type=int, default=250, help="control steps of 20 ms (default 5 s)")
    ap.add_argument("--keyframe", default="knees_bent", help="the fixtures model's keyframe to start from")
    ap.add_argument("--iterations", type=int, default=10, help="Newton iterations per substep (both simulators)")
    ap.add_argument("--size", type=int, default=480, help="height and width of each half of the frame, pixels")
    ap.add_argument("--snapshots", default="0,25,50,75,150,249", help="comma-separated steps saved as PNG")
    ap.add_argument("--tol", type=float, default=1e-4, help="largest |qpos| deviation that passes")
    ap.add_argument("--no-render", action="store_true", help="only the rollouts and the deviations")
    a = ap.parse_args()
    if not a.no_render and shutil.which("ffmpeg") is None:
        sys.exit("visualize_rollout.py: no ffmpeg on PATH (or pass --no-render)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    m, mt = fixtures.load_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(a.keyframe).id)
    d.qvel[:] = 0
    q0 = d.qpos.copy()
    base = q0[m.jnt_qposadr[m.actuator_trnid[:, 0]]]
    ctrls = np.stack([joint_targets(m, base, k * DT) for k in range(a.steps)]).astype(np.float32)

    qs_engine, g = engine_rollout(q0, ctrls, m.nv, a.iterations)
    m.opt.iterations = a.iterations
    qs_mj = mujoco_rollout(m, d, ctrls, mt["substeps_per_control_step"])

    err = np.abs(qs_engine - qs_mj).max(axis=1)
    print(f"grad of the final pelvis height wrt {g.shape[0]}x{g.shape[1]} targets: "
          f"finite {bool(np.all(np.isfinite(g)))}, norm {np.linalg.norm(g):.3e}")
    for k in sorted({1, 10, 50, 100, a.steps} & set(range(1, a.steps + 1))):
        print(f"step {k:4d} ({k * DT:5.2f} s): max |qpos diff| {err[k - 1]:.2e}, "
              f"pelvis z engine {qs_engine[k - 1, 2]:.4f} / MuJoCo {qs_mj[k - 1, 2]:.4f}")
    print(f"pelvis height {qs_engine[:, 2].min():.3f} .. {qs_engine[:, 2].max():.3f} m")

    if not a.no_render:
        snapshots = {int(s) for s in a.snapshots.split(",") if s}
        render(m, qs_engine, qs_mj, out, a.size, snapshots)
        print(f"wrote {out / 'rollout.mp4'} (left: NablaEngine, right: MuJoCo)")

    ok = err.max() <= a.tol and np.all(np.isfinite(g))
    print(f"visualize_rollout: {'PASS' if ok else 'FAIL'} (max |qpos diff| {err.max():.2e}, tol {a.tol:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
