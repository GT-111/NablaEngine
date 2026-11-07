"""Engine vs MuJoCo Warp (and float64 MuJoCo) on another robot: MODEL.xml / MODEL.json from export_robot.py.

  corpus (MuJoCo Warp env): states from rollouts: the model's initial state with random tilt, height and velocities,
         PD targets around it that drift smoothly; every world records a few states at random control steps; worlds
         whose base falls below --reset x its initial height, or that diverge, restart
  ref    (MuJoCo Warp env): one control step (substeps of the model's timestep) from the corpus states
  check  (PyTorch): the engine (torch_ffi on this model) on the same inputs: qvel / qpos and contact counts
  grad64 (PyTorch + MuJoCo): the engine's VJP against float64 central differences of MuJoCo C
  time   (PyTorch): forward and forward + backward per control step
$ROBOT_DROP_BOX=1 drops the pairs of box geoms (an experiment switch; results differ from the model's).

  python scripts/robots/robot_check.py corpus --model runs/robots/go1/go1_flat --out runs/robots/go1/corpus.npz
  python scripts/robots/robot_check.py ref    --model runs/robots/go1/go1_flat --corpus runs/robots/go1/corpus.npz --out runs/robots/go1/ref.npz
  python scripts/robots/robot_check.py check  --model runs/robots/go1/go1_flat --ref runs/robots/go1/ref.npz
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def load(model):
    """(MjModel, meta). $ROBOT_DROP_BOX=1: without the pairs of box geoms."""
    import mujoco
    meta = json.loads(Path(model).with_suffix(".json").read_text())
    spec = mujoco.MjSpec.from_file(str(Path(model).with_suffix(".xml")))
    if os.environ.get("ROBOT_DROP_BOX") == "1":
        box = {g.name for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_BOX}
        names = [p.name for p in spec.pairs]
        for p in list(spec.pairs):
            if p.geomname1 in box or p.geomname2 in box:
                spec.delete(p)
        kept = [p.name for p in spec.pairs]
        meta["foot_pairs"] = [kept.index(names[i]) for i in meta["foot_pairs"]]
    return spec.compile(), meta


def mjw_setup(m, B, nconmax=64, njmax=256):
    import mujoco
    import mujoco_warp as mjw
    return mjw.put_model(m), mjw.put_data(m, mujoco.MjData(m), nworld=B, nconmax=nconmax, njmax=njmax)


def cmd_corpus(a):
    import warp as wp
    import mujoco_warp as mjw
    m, meta = load(a.model)
    m.opt.iterations = a.iterations
    rng = np.random.default_rng(a.seed)
    B, sub = a.nworld, meta["substeps_per_control_step"]
    q0 = np.asarray(meta["init_qpos"])

    def init(n):  # the initial state with random tilt, height and velocities
        q = np.tile(q0, (n, 1))
        q[:, 2] += rng.uniform(-0.05, 0.1, n)
        ax = rng.normal(size=(n, 3)) * [1, 1, 0.3]
        ang = rng.uniform(0, a.tilt, n)
        ax /= np.linalg.norm(ax, axis=1, keepdims=True)
        q[:, 3:7] = np.concatenate([np.cos(ang / 2)[:, None], np.sin(ang / 2)[:, None] * ax], 1)
        return q, rng.normal(0, 0.3, (n, m.nv))

    q, v = init(B)
    mw, dw = mjw_setup(m, B)
    wp.copy(dw.qpos, wp.array(q.astype(np.float32)))
    wp.copy(dw.qvel, wp.array(v.astype(np.float32)))
    target = np.tile(q0[7:], (B, 1))
    rec = {k: [] for k in ("qpos", "qvel", "qacc_warmstart", "ctrl")}
    when = rng.integers(5, a.steps, (B, a.per_world))
    for t in range(a.steps):
        target += rng.normal(0, a.drift, target.shape)
        target = q0[7:] + np.clip(target - q0[7:], -a.span, a.span)
        ctrl = target.astype(np.float32)
        wp.copy(dw.ctrl, wp.array(ctrl))
        for _ in range(sub):
            mjw.step(mw, dw)
        sel = np.nonzero((when == t).any(1))[0]
        if len(sel):
            for k in rec:
                rec[k].append((dw.ctrl if k == "ctrl" else getattr(dw, k)).numpy()[sel])
        # fallen worlds restart (feet-only models: a fallen base sinks through the floor, contact-free states), and
        # diverged ones (non-finite)
        qn = dw.qpos.numpy()
        fall = np.nonzero(~(qn[:, 2] >= a.reset * q0[2]) | ~np.isfinite(dw.qvel.numpy()).all(1))[0]
        if len(fall):
            vn, wn = dw.qvel.numpy(), dw.qacc_warmstart.numpy()
            qn[fall], vn[fall] = init(len(fall))
            wn[fall] = 0
            target[fall] = q0[7:]
            for x, y in ((dw.qpos, qn), (dw.qvel, vn), (dw.qacc_warmstart, wn)):
                wp.copy(x, wp.array(y.astype(np.float32)))
    out = {k: np.concatenate(v).astype(np.float32) for k, v in rec.items()}
    ok = np.all([np.isfinite(x).all(1) for x in out.values()], 0)  # (states recorded as their world diverged)
    out = {k: x[ok] for k, x in out.items()}
    if not ok.all():
        print(f"corpus: dropped {np.sum(~ok)} non-finite states")
    # the engine's per-world foot friction replaces the foot pairs' friction: the model's value here
    mu = {float(m.pair_friction[p, 0]) for p in meta["foot_pairs"]}
    assert len(mu) == 1, mu
    out["friction"] = np.full(len(out["qpos"]), mu.pop(), np.float32)
    out["com"] = np.zeros((len(out["qpos"]), 3), np.float32)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    z = out["qpos"][:, 2]
    print(f"corpus: {len(z)} states, base height {np.quantile(z, 0.05):.3f} .. {np.quantile(z, 0.95):.3f}; {a.out}")


def cmd_ref(a):
    import warp as wp
    import mujoco_warp as mjw
    m, meta = load(a.model)
    m.opt.iterations = a.iterations
    z = np.load(a.corpus)
    B = min(a.nworld, len(z["qpos"]))
    mw, dw = mjw_setup(m, B)
    for k in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
        wp.copy(getattr(dw, k), wp.array(np.ascontiguousarray(z[k][:B])))
    for i in range(meta["substeps_per_control_step"]):
        mjw.step(mw, dw)
        if i == meta["substeps_per_control_step"] - 1:
            nacon = int(dw.nacon.numpy()[0])
            ncon = np.bincount(dw.contact.worldid.numpy()[:nacon], minlength=B)
    np.savez(a.out, **{k: z[k][:B] for k in z.files}, out_qpos=dw.qpos.numpy(), out_qvel=dw.qvel.numpy(), ncon=ncon,
             iterations=a.iterations)
    print(f"ref: {B} worlds, mean contacts {ncon.mean():.2f}, max {ncon.max()}; {a.out}")


def cmd_check(a):
    import torch
    from nablaengine import torch_ffi
    m, meta = load(a.model)
    r = np.load(a.ref)
    torch_ffi.init(model=m, foot_pairs=meta["foot_pairs"], com_body=meta["com_body"])
    t = lambda k: torch.from_numpy(np.ascontiguousarray(r[k], np.float32)).cuda()  # noqa: E731
    q1, v1, _, info = torch_ffi.forward(t("qpos"), t("qvel"), t("qacc_warmstart"), t("ctrl"), t("friction"), t("com"),
                                        iterations=int(r["iterations"]), substeps=meta["substeps_per_control_step"])
    it_max = int(info[:, 0].max().item())
    q1, v1, info = q1.cpu().numpy(), v1.cpu().numpy(), info.cpu().numpy()
    ev = np.linalg.norm(v1 - r["out_qvel"], axis=1) / np.maximum(np.linalg.norm(r["out_qvel"], axis=1), 1e-6)
    eq = np.abs(q1 - r["out_qpos"]).max(1)
    if a.dump:
        np.savez(a.dump, q1=q1, v1=v1, info=info, ev=ev)
    print(json.dumps(dict(nworld=len(ev), qvel_rel_median=float(np.median(ev)), qvel_rel_p99=float(np.quantile(ev, 0.99)),
                          qvel_rel_max=float(ev.max()), qpos_abs_median=float(np.median(eq)), qpos_abs_max=float(eq.max()),
                          ncon_equal=float((info[:, 1] == r["ncon"]).mean()), ncon_mean=float(info[:, 1].mean()),
                          flags=int((info[:, 3] != 0).sum()), newton_iterations_max=it_max,
                          at_iteration_limit=float((info[:, 0] >= int(r["iterations"])).mean())), indent=1))



def cmd_grad(a):
    """The engine's VJP against its own float32 central differences, per world along a random direction of qpos (not
    the root quaternion), qvel and ctrl with a random output cotangent (as scripts/terrain_check.py grad); --g1: the
    G1 fixture on its corpus (the differences' noise level)."""
    import torch
    from nablaengine import fixtures, torch_ffi
    if a.g1:
        z = fixtures.load_corpus("mf_s0")
        r = {k: z[k][:a.nworld] for k in ("qpos", "qvel", "ctrl", "qacc_warmstart", "friction", "com")}
        sub = 4
        torch_ffi.init()
    else:
        m, meta = load(a.model)
        r = dict(np.load(a.ref))
        sub = meta["substeps_per_control_step"]
        torch_ffi.init(model=m, foot_pairs=meta["foot_pairs"], com_body=meta["com_body"])
    B = min(a.nworld, len(r["qpos"]))
    t = lambda k: torch.from_numpy(np.ascontiguousarray(r[k][:B], np.float32)).cuda()  # noqa: E731
    qpos, qvel, ctrl, warm, fr, com = t("qpos"), t("qvel"), t("ctrl"), t("qacc_warmstart"), t("friction"), t("com")
    gen = torch.Generator(device="cuda").manual_seed(0)
    yq = torch.randn(B, qpos.shape[1], device="cuda", generator=gen)
    yv = torch.randn(B, qvel.shape[1], device="cuda", generator=gen)

    def out(q, v, u):
        q1, v1, _, info = torch_ffi.forward(q, v, warm, u, fr, com, iterations=a.iterations, substeps=sub)
        return (yq * q1).sum(1).double() + (yv * v1).sum(1).double(), info[:, 1]

    xs = [qpos.clone().requires_grad_(), qvel.clone().requires_grad_(), ctrl.clone().requires_grad_()]
    q1, v1, _ = torch_ffi.step(xs[0], xs[1], warm, xs[2], fr, com, iterations=a.iterations, substeps=sub)
    ((yq * q1).sum() + (yv * v1).sum()).backward()
    res = dict(nworld=B, model="g1" if a.g1 else a.model)
    for k, (name, x, eps) in enumerate((("qpos", qpos, a.eps_q), ("qvel", qvel, a.eps_v), ("ctrl", ctrl, a.eps_u))):
        d = torch.randn(x.shape, device="cuda", generator=gen)
        if name == "qpos":
            d[:, 3:7] = 0.0
        d /= d.norm(dim=1, keepdim=True)
        ap, am = [qpos, qvel, ctrl], [qpos, qvel, ctrl]
        ap[k], am[k] = x + eps * d, x - eps * d
        lp, np_ = out(*ap)
        lm, nm = out(*am)
        fd = ((lp - lm) / (2 * eps)).cpu().numpy()
        an = (xs[k].grad * d).sum(1).double().cpu().numpy()
        rel = np.abs(fd - an) / np.maximum(np.maximum(np.abs(fd), np.abs(an)), 1e-12)
        same = (np_ == nm).cpu().numpy()
        res[name] = dict(rel_median=float(np.median(rel[same])), rel_p90=float(np.quantile(rel[same], 0.9)),
                         same_contacts=float(same.mean()), nonfinite=int((~np.isfinite(an)).sum()))
    print(json.dumps(res, indent=1))


def cmd_grad64(a):
    """The engine's VJP (--iterations, default 50) against float64 central differences of MuJoCo C (mj_step, the same
    model with tolerance 1e-12 and the same iterations) along a random direction of qpos (not the root quaternion),
    qvel and ctrl per world with a random output cotangent; --g1: the G1 fixture on its corpus."""
    import mujoco
    import torch
    from nablaengine import fixtures, torch_ffi
    if a.g1:
        m, mt = fixtures.load_model()
        z = fixtures.load_corpus("mf_s0")
        r = {k: z[k] for k in ("qpos", "qvel", "ctrl", "qacc_warmstart", "friction", "com")}
        sub, fp, com = 4, mt["foot_pairs"], mt.get("torso_body", "torso_link")
        torch_ffi.init()
    else:
        m, meta = load(a.model)
        r = dict(np.load(a.ref))
        sub, fp, com = meta["substeps_per_control_step"], meta["foot_pairs"], meta["com_body"]
        torch_ffi.init(model=m, foot_pairs=fp, com_body=com)
    rng = np.random.default_rng(1)
    idx = rng.choice(len(r["qpos"]), min(a.nworld, len(r["qpos"])), replace=False)
    B = len(idx)
    f32 = lambda k: np.ascontiguousarray(r[k][idx], np.float32)  # noqa: E731
    yq, yv = rng.standard_normal((B, m.nq)), rng.standard_normal((B, m.nv))
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32)).cuda()  # noqa: E731
    xs = [t(f32("qpos")).requires_grad_(), t(f32("qvel")).requires_grad_(), t(f32("ctrl")).requires_grad_()]
    q1, v1, _ = torch_ffi.step(xs[0], xs[1], t(f32("qacc_warmstart")), xs[2], t(f32("friction")), t(f32("com")),
                               iterations=a.iterations, substeps=sub, clamp_kv=a.clamp_kv, normal=a.normal)
    ((t(yq) * q1).sum() + (t(yv) * v1).sum()).backward()
    m.opt.iterations, m.opt.tolerance, m.opt.ls_tolerance = a.iterations, 1e-12, 1e-12
    d = mujoco.MjData(m)
    fpairs = set(int(p) for p in fp)

    def run(q, v, u, w, fr):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = q, v, u, w
        mu = m.pair_friction.copy()
        for p in fpairs:  # the engine's per-world foot friction
            m.pair_friction[p, :2] = fr
        for _ in range(sub):
            mujoco.mj_step(m, d)
        m.pair_friction[:] = mu
        return d.qpos.copy(), d.qvel.copy()

    res = dict(nworld=B, model="g1" if a.g1 else a.model, iterations=a.iterations, clamp_kv=a.clamp_kv, normal=a.normal)
    q1n, v1n = q1.detach().cpu().numpy().astype(np.float64), v1.detach().cpu().numpy().astype(np.float64)
    fq, fv = [], []
    for i in range(B):  # the forward against MuJoCo C (float64, converged): qvel relative, qpos absolute
        qc, vc = run(*(np.asarray(r[x][idx[i]], np.float64) for x in ("qpos", "qvel", "ctrl", "qacc_warmstart")),
                     float(r["friction"][idx[i]]))
        fv.append(np.linalg.norm(v1n[i] - vc) / max(np.linalg.norm(vc), 1e-6))
        fq.append(np.abs(q1n[i] - qc).max())
    res["forward"] = dict(qvel_rel_median=float(np.median(fv)), qvel_rel_p99=float(np.quantile(fv, 0.99)),
                          qpos_abs_median=float(np.median(fq)), qpos_abs_max=float(np.max(fq)))
    agree = np.asarray(fv) < 1e-4  # worlds where the engine's step is MuJoCo C's (the gradients' fair comparison)
    res["forward"]["agree_share"] = float(agree.mean())
    rel = {k: [] for k in ("qpos", "qvel", "ctrl")}
    for k, name in enumerate(("qpos", "qvel", "ctrl")):
        an = (xs[k].grad.cpu().numpy().astype(np.float64))
        for i in range(B):
            q, v, u = (np.asarray(r[x][idx[i]], np.float64) for x in ("qpos", "qvel", "ctrl"))
            w, fr = np.asarray(r["qacc_warmstart"][idx[i]], np.float64), float(r["friction"][idx[i]])
            dd = rng.standard_normal([m.nq, m.nv, m.nu][k])
            if k == 0:
                dd[3:7] = 0.0
            dd /= np.linalg.norm(dd)
            eps = a.eps
            args_p, args_m = [q.copy(), v.copy(), u.copy()], [q.copy(), v.copy(), u.copy()]
            args_p[k] += eps * dd
            args_m[k] -= eps * dd
            qp, vp = run(*args_p, w, fr)
            qm, vm = run(*args_m, w, fr)
            fd = (yq[i] @ (qp - qm) + yv[i] @ (vp - vm)) / (2 * eps)
            g = an[i] @ dd
            rel[name].append(abs(fd - g) / max(abs(fd), abs(g), 1e-12))
        x = np.asarray(rel[name])
        res[name] = dict(rel_median=float(np.median(x)), rel_p90=float(np.quantile(x, 0.9)),
                         rel_p99=float(np.quantile(x, 0.99)), agree_rel_median=float(np.median(x[agree])),
                         agree_rel_p90=float(np.quantile(x[agree], 0.9)))
    print(json.dumps(res, indent=1))


def cmd_time(a):
    """Time per control step at --nworld worlds (the reference's states tiled; --g1: the G1 fixture's corpus): the
    forward and the forward with the backward, CUDA events, median of --reps."""
    import torch
    from nablaengine import fixtures, torch_ffi
    if a.g1:
        z = fixtures.load_corpus("mf_s0")
        r = {k: z[k] for k in ("qpos", "qvel", "ctrl", "qacc_warmstart", "friction", "com")}
        sub = 4
        torch_ffi.init()
        name = "g1"
    else:
        m, meta = load(a.model)
        r = dict(np.load(a.ref))
        sub = meta["substeps_per_control_step"]
        torch_ffi.init(model=m, foot_pairs=meta["foot_pairs"], com_body=meta["com_body"])
        name = meta["robot"]
    idx = np.arange(a.nworld) % len(r["qpos"])
    t = lambda k: torch.from_numpy(np.ascontiguousarray(r[k][idx], np.float32)).cuda()  # noqa: E731
    qpos, qvel, ctrl, warm, fr, com = t("qpos"), t("qvel"), t("ctrl"), t("qacc_warmstart"), t("friction"), t("com")
    u = ctrl.clone().requires_grad_()

    def fwd():
        torch_ffi.forward(qpos, qvel, warm, ctrl, fr, com, iterations=a.iterations, substeps=sub)

    def both():
        q1, v1, _ = torch_ffi.step(qpos, qvel, warm, u, fr, com, iterations=a.iterations, substeps=sub)
        (q1.sum() + v1.sum()).backward()

    res = dict(robot=name, nworld=a.nworld, substeps=sub, iterations=a.iterations, gpu=torch.cuda.get_device_name())
    for key, f in (("forward_ms", fwd), ("forward_backward_ms", both)):
        for _ in range(3):
            f()
        ts = []
        for _ in range(a.reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            f()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        res[key] = round(float(np.median(ts)), 3)
    print(json.dumps(res))

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("corpus")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nworld", type=int, default=512)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--per-world", type=int, default=8)
    p.add_argument("--tilt", type=float, default=0.6)
    p.add_argument("--drift", type=float, default=0.05)
    p.add_argument("--span", type=float, default=0.6)
    p.add_argument("--reset", type=float, default=0.5, help="restart worlds whose base is below this x initial height")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=10)
    p = sub.add_parser("ref")
    p.add_argument("--model", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=10)
    p = sub.add_parser("check")
    p.add_argument("--model", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--dump", default=None)
    p = sub.add_parser("grad")
    p.add_argument("--model", default=None)
    p.add_argument("--ref", default=None)
    p.add_argument("--g1", action="store_true")
    p.add_argument("--nworld", type=int, default=2048)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--eps-q", type=float, default=1e-4)
    p.add_argument("--eps-v", type=float, default=1e-3)
    p.add_argument("--eps-u", type=float, default=1e-3)
    p = sub.add_parser("grad64")
    p.add_argument("--model", default=None)
    p.add_argument("--ref", default=None)
    p.add_argument("--g1", action="store_true")
    p.add_argument("--nworld", type=int, default=256)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--clamp-kv", type=int, default=1, help="the gradient mode (1, normal 0: the exact derivative)")
    p.add_argument("--normal", type=int, default=0)
    p = sub.add_parser("time")
    p.add_argument("--model", default=None)
    p.add_argument("--ref", default=None)
    p.add_argument("--g1", action="store_true")
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    {"corpus": cmd_corpus, "ref": cmd_ref, "check": cmd_check, "grad": cmd_grad, "grad64": cmd_grad64,
     "time": cmd_time}[a.cmd](a)


if __name__ == "__main__":
    main()
