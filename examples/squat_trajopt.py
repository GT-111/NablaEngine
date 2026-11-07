"""Trajectory optimization through the engine's gradients: open-loop PD targets that make a humanoid (the G1, or
with --robot h1 the Unitree H1) squat and stand up again without falling, found by gradient descent on a loss over a whole rollout (jax.grad through lax.scan of
nablaengine.jax_ffi.step, the analytic adjoint of every step).

The robot starts at rest at a keyframe of its fixtures model (G1: knees_bent; H1: home, which joint PD alone does not
hold: the H1 tips backwards; nominal friction, no torso COM offset). The decision variables are the (T, nu) joint
targets (G1 29, H1 19) of T control steps of 20 ms (--steps, default 150: 3 s), as offsets
from the naive targets: the keyframe pose with the knees, hip pitch and ankle (pitch) bent along the squat profile, which
alone make the robot fall. The loss (per step, averaged over the rollout):
  pelvis height  against the reference: the keyframe height lowered by --depth (default 0.15 m) along 0.5 (1 - cos),
                 down and back up over the horizon
  upright        the pelvis quaternion's x and y components (roll and pitch)
  drift          the pelvis horizontal position against the start
  rest           qvel over the last 10 steps
  smooth, reg    the targets' step-to-step change and their offset from the naive targets
Adam (--iters, default 400, --lr 3e-3 with cosine decay, the gradient norm clipped at 10); the iterate with the lowest
loss is kept (a rollout that falls is a jump of the loss: the contact sequence changes). The optimized targets are then
replayed open loop in MuJoCo C (float64, the same substeps and Newton iteration cap) as an independent check: the
G1's replay stays within 4e-5 of the engine's rollout; the H1's, balanced on the edge of its heels, drifts to about
2e-2 (float32 against float64 roundings grow in open loop) yet squats and stands the same way.

  source .dk_env/activate.sh
  python examples/squat_trajopt.py --out runs/squat
  python examples/squat_trajopt.py --robot h1 --out runs/squat_h1

Writes <out>/squat.npz (naive and optimized targets and rollouts, the reference, MuJoCo's replay), and unless
--no-render <out>/squat.mp4 (left the naive targets, right the optimized ones; ffmpeg with drawtext on PATH, offscreen
rendering, MUJOCO_GL=egl unless set) and with --gif <out>/squat.gif (smaller, for documentation). About 35 s (G1)
and 26 s (H1) of optimization on an RTX 4060 Laptop GPU.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from nablaengine import fixtures, jax_ffi  # noqa: E402

DT = 0.02  # one control step: 4 substeps of 5 ms
ROBOTS = {  # fixtures directory, start keyframe, the leg joints the naive squat bends, camera (distance, look-at height)
    "g1": dict(fixtures=fixtures.DEFAULT, keyframe="knees_bent", ankle="ankle_pitch", camera=(2.6, 0.55)),
    "h1": dict(fixtures=fixtures.H1, keyframe="home", ankle="ankle", camera=(3.4, 0.75)),
}


def setup(m, robot, steps, depth):
    """(q0, naive targets (T, nu), lower and upper ctrlrange, reference pelvis height (T,))."""
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(robot["keyframe"]).id)
    q0 = d.qpos.copy()
    jid = m.actuator_trnid[:, 0]
    names = [m.joint(j).name for j in jid]
    sel = lambda s: [i for i, n in enumerate(names) if s in n]  # noqa: E731
    lo, hi = (m.actuator_ctrlrange[:, i].astype(np.float32) for i in (0, 1))
    s = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, steps + 1) / steps))[:, None]  # 0 -> 1 -> 0
    naive = np.tile(q0[m.jnt_qposadr[jid]], (steps, 1))
    naive[:, sel("knee")] += 0.7 * s
    naive[:, sel("hip_pitch")] -= 0.35 * s
    naive[:, sel(robot["ankle"])] -= 0.35 * s
    naive = np.clip(naive, lo, hi).astype(np.float32)
    return q0, naive, lo, hi, (q0[2] - depth * s[:, 0]).astype(np.float32)


def optimize(m, mt, q0, naive, lo, hi, z_ref, iters, lr):
    """(rollout function: targets -> (qpos (T, nq), qvel (T, nv)), optimized targets (T, nu))."""
    jax_ffi.init(model=m, foot_pairs=mt["foot_pairs"])
    qpos = jnp.asarray(q0[None], jnp.float32)
    zero = jnp.zeros((1, m.nv), jnp.float32)
    fric, com = jnp.ones((1,), jnp.float32), jnp.zeros((1, 3), jnp.float32)
    naive, z_ref = jnp.asarray(naive), jnp.asarray(z_ref)

    @jax.jit
    def rollout(u):
        def body(c, uk):
            c = jax_ffi.step(*c, uk[None], fric, com)
            return c, (c[0][0], c[1][0])
        return jax.lax.scan(body, (qpos, zero, zero), u)[1]

    def loss(du):
        qs, vs = rollout(jnp.clip(naive + du, lo, hi))
        terms = dict(
            height=jnp.mean((qs[:, 2] - z_ref) ** 2) / 0.01 ** 2,
            upright=jnp.mean(qs[:, 4] ** 2 + qs[:, 5] ** 2) / 0.02 ** 2,
            drift=jnp.mean(jnp.sum((qs[:, :2] - qpos[0, :2]) ** 2, -1)) / 0.03 ** 2,
            rest=jnp.mean(vs[-10:] ** 2),
            smooth=jnp.mean((du[1:] - du[:-1]) ** 2) / 0.01 ** 2,
            reg=jnp.mean(du ** 2) / 0.2 ** 2)
        w = dict(height=1, upright=1, drift=1, rest=0.1, smooth=0.1, reg=0.01)
        return sum(w[k] * v for k, v in terms.items()), terms

    vg = jax.jit(jax.value_and_grad(loss, has_aux=True))
    du = jnp.zeros_like(naive)
    mom, var = jnp.zeros_like(du), jnp.zeros_like(du)
    b1, b2 = 0.9, 0.999
    best, best_du = np.inf, du
    t0 = time.time()
    for i in range(iters):
        (l, terms), g = vg(du)
        g = jnp.nan_to_num(g)
        gn = jnp.linalg.norm(g)
        g = g * jnp.minimum(1.0, 10.0 / (gn + 1e-12))
        if float(l) < best:
            best, best_du = float(l), du
        mom = b1 * mom + (1 - b1) * g
        var = b2 * var + (1 - b2) * g ** 2
        lr_i = lr * 0.5 * (1 + np.cos(np.pi * i / iters))
        du = du - lr_i * (mom / (1 - b1 ** (i + 1))) / (jnp.sqrt(var / (1 - b2 ** (i + 1))) + 1e-8)
        if i % 50 == 0 or i == iters - 1:
            print(f"iter {i:4d}  loss {float(l):9.3f}  " + "  ".join(f"{k} {float(v):.3f}" for k, v in terms.items())
                  + f"  ({time.time() - t0:.0f} s)", flush=True)
    return rollout, np.asarray(jnp.clip(naive + best_du, lo, hi))


def mujoco_replay(m, q0, u, substeps):
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    qs = np.empty((len(u), m.nq))
    for k, uk in enumerate(u):
        d.ctrl[:] = uk
        for _ in range(substeps):
            mujoco.mj_step(m, d)
        qs[k] = d.qpos
    return qs


def render(m, qs_left, qs_right, labels, camera, size, fps_out, path_mp4, path_gif=None):
    """The two trajectories side by side, each half labelled: an MP4 and optionally a palette GIF at half size."""
    m.geom_rgba[m.geom_group == 3] = [0.85, 0.45, 0.15, 1.0]  # the collision geoms, hidden and translucent by default
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth, cam.lookat[:] = camera[0], -10, 120, [0, 0, camera[1]]
    d = mujoco.MjData(m)
    text = ("drawtext=font=sans:text='{}':x={}+16:y=16:fontsize={}:fontcolor=white:box=1:boxcolor=black@0.5:"
            "boxborderw=8")
    vf = ",".join(text.format(t, x, max(12, size // 20)) for t, x in zip(labels, (0, size)))
    raw = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * size}x{size}",
           "-r", str(round(1 / DT)), "-i", "-"]
    outs = [subprocess.Popen([*raw, "-vf", vf, "-pix_fmt", "yuv420p", str(path_mp4)], stdin=subprocess.PIPE)]
    if path_gif:
        gif = (f"{vf},fps={fps_out},scale={size}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];"
               "[b][p]paletteuse=dither=bayer")
        outs.append(subprocess.Popen([*raw, "-vf", gif, "-loop", "0", str(path_gif)], stdin=subprocess.PIPE))
    with mujoco.Renderer(m, size, size) as r:
        def frame(q):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            r.update_scene(d, cam, opt)
            return r.render()

        for ql, qr in zip(qs_left, qs_right):
            img = np.concatenate([frame(ql), frame(qr)], axis=1).tobytes()
            for p in outs:
                p.stdin.write(img)
    for p in outs:
        p.stdin.close()
        if p.wait():
            raise RuntimeError("ffmpeg failed")


def summary(name, qs, q0, z_ref):
    print(f"{name:16s} pelvis z min {qs[:, 2].min():.3f} final {qs[-1, 2]:.3f} m (reference {z_ref[-1]:.3f}), "
          f"max |z - reference| {np.abs(qs[:, 2] - z_ref).max():.3f} m, max tilt {np.abs(qs[:, 4:6]).max():.3f}, "
          f"drift {np.linalg.norm(qs[-1, :2] - q0[:2]):.3f} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robot", default="g1", choices=sorted(ROBOTS), help="the fixtures model (default g1)")
    ap.add_argument("--out", default="runs/squat", help="output directory")
    ap.add_argument("--steps", type=int, default=150, help="control steps of 20 ms (default 3 s)")
    ap.add_argument("--depth", type=float, default=0.15, help="squat depth of the pelvis, m")
    ap.add_argument("--iters", type=int, default=400, help="Adam iterations")
    ap.add_argument("--lr", type=float, default=3e-3, help="Adam's initial learning rate")
    ap.add_argument("--size", type=int, default=480, help="height and width of each half of the video, pixels")
    ap.add_argument("--gif", action="store_true", help="also write squat.gif (half size, 25 fps)")
    ap.add_argument("--no-render", action="store_true", help="no video")
    a = ap.parse_args()
    if not a.no_render and shutil.which("ffmpeg") is None:
        sys.exit("squat_trajopt.py: no ffmpeg on PATH (or pass --no-render)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    robot = ROBOTS[a.robot]
    m, mt = fixtures.load_model(robot["fixtures"])
    q0, naive, lo, hi, z_ref = setup(m, robot, a.steps, a.depth)
    rollout, u_opt = optimize(m, mt, q0, naive, lo, hi, z_ref, a.iters, a.lr)
    qs_naive = np.asarray(rollout(jnp.asarray(naive))[0])
    qs_opt = np.asarray(rollout(jnp.asarray(u_opt))[0])
    m.opt.iterations = 10  # the engine's default Newton iterations per substep
    qs_mj = mujoco_replay(m, q0, u_opt, mt["substeps_per_control_step"])

    summary("naive targets", qs_naive, q0, z_ref)
    summary("optimized", qs_opt, q0, z_ref)
    summary("MuJoCo replay", qs_mj, q0, z_ref)
    print(f"MuJoCo replay of the optimized targets: max |qpos diff| from the engine {np.abs(qs_mj - qs_opt).max():.2e}")
    np.savez(out / "squat.npz", naive=naive, u_opt=u_opt, qs_naive=qs_naive, qs_opt=qs_opt, qs_mujoco=qs_mj,
             z_ref=z_ref)

    if not a.no_render:
        render(m, qs_naive, qs_opt, ("naive targets", "optimized with jax.grad"), robot["camera"], a.size, 25, out / "squat.mp4",
               out / "squat.gif" if a.gif else None)
        print(f"wrote {out / 'squat.mp4'}" + (f" and {out / 'squat.gif'}" if a.gif else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
