"""Terrain check: the engine's terrain contacts against MuJoCo Warp on the same scene.

Two steps, two environments:
  ref    (MuJoCo Warp): the scene SCENE.mjb (the fixture's G1 with its floor replaced by a terrain of boxes and a
         heightfield, explicit pairs between the terrain and the robot geoms with their floor pairs' parameters:
         fixtures/mjlab/g1_terrain_feet.mjb, _body.mjb), steps the corpus states placed on the terrain, writes inputs,
         outputs and contacts to REF.npz and the scene to REF.mjb
  check  (PyTorch): the engine (torch_ffi, a terrain build on the fixture model) on the same inputs with the terrain
         read from REF.mjb's world body; compares qpos / qvel and the contact counts

MuJoCo Warp's heightfield contacts are its native ones (convex collision per prism); the engine's "native" mode follows
MuJoCo's prism rules, its "analytic" mode is its own sample-point model (nablaengine.terrain), so heightfield worlds
differ from MuJoCo Warp by construction; boxes are the same closed forms.

  python scripts/terrain_check.py ref   --scene fixtures/mjlab/g1_terrain_feet.mjb --out runs/terrain/ref.npz
  python scripts/terrain_check.py check --ref runs/terrain/ref.npz [--mode analytic|native]
  python scripts/terrain_check.py grad  --ref runs/terrain/ref.npz | --flat
"""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def place(z, B, origins=None, coltype=None):
    """Corpus states placed on the terrain: base xy in a region over the stairs, the obstacle or the heightfield, or
    (origins: (rows, cols, 3) sub-terrain spawn points, mjlab's TerrainGenerator.terrain_origins) at a random sub-terrain
    each, 0.5 m around its origin (region = its column, or coltype[column]: its sub-terrain type); z is lifted
    afterwards (terrain_lift)."""
    rng = np.random.default_rng(1)
    q = np.ascontiguousarray(z["qpos"][:B], np.float64).copy()
    if origins is not None:
        o = origins.reshape(-1, 3)
        k = rng.integers(0, len(o), B)
        q[:, :2] += o[k, :2] - q[:, :2].mean(0) + rng.uniform(-0.5, 0.5, (B, 2))
        col = k % origins.shape[1]
        return q, (coltype[col] if coltype is not None else col)
    region = rng.integers(0, 3, B)
    centre = np.array([[1.0, 0.0], [-0.4, 0.5], [0.0, -3.0]])[region]
    q[:, :2] += centre + rng.uniform(-0.5, 0.5, (B, 2))
    return q, region


def cmd_ref(a):
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    from nablaengine import fixtures
    from nablaengine import terrain as T
    m = mujoco.MjModel.from_binary_path(str(a.scene))
    tnames = [m.geom(g).name for g in range(m.ngeom) if m.geom_bodyid[g] == 0]
    geoms = "body" if "_body" in Path(a.scene).stem else "feet"
    m.opt.iterations = a.iterations
    if a.fric_scale != 1.0:  # the terrain pairs' friction (a domain-randomization yardstick for eval)
        tid = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, t) for t in tnames}
        for p in range(m.npair):
            if int(m.pair_geom2[p]) in tid or int(m.pair_geom1[p]) in tid:
                m.pair_friction[p, :2] *= a.fric_scale
    z = fixtures.load_corpus("mf_s0")
    B = min(a.nworld, len(z["qpos"]))
    q, region = place(z, B, np.load(a.origins) if a.origins else None, np.load(a.coltype) if a.coltype else None)
    d = mujoco.MjData(m)
    for i in range(B):  # lift each state so that its lowest robot geom sits on the terrain (no deep penetration)
        d.qpos[:] = q[i]
        mujoco.mj_kinematics(m, d)
        q[i, 2] += terrain_lift(m, d)
    vel = np.ascontiguousarray(z["qvel"][:B], np.float64)
    ctrl = np.ascontiguousarray(z["ctrl"][:B], np.float64)
    warm = np.ascontiguousarray(z["qacc_warmstart"][:B], np.float64)
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, mujoco.MjData(m), nworld=B, nconmax=96, njmax=400)
    wp.copy(dw.qpos, wp.array(q.astype(np.float32)))
    wp.copy(dw.qvel, wp.array(vel.astype(np.float32)))
    wp.copy(dw.ctrl, wp.array(ctrl.astype(np.float32)))
    wp.copy(dw.qacc_warmstart, wp.array(warm.astype(np.float32)))
    base = []  # the base position after each control step (4 substeps), a.steps control steps with ctrl held
    for i in range(4 * a.steps):
        mjw.step(mw, dw)
        if i == 3:
            q1, v1 = dw.qpos.numpy().copy(), dw.qvel.numpy().copy()
            nacon = int(dw.nacon.numpy()[0])
            cw = dw.contact.worldid.numpy()[:nacon]
        if i % 4 == 3:
            base.append(dw.qpos.numpy()[:, :3].copy())
    ncon = np.bincount(cw, minlength=B)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(mujoco.MjModel.from_binary_path(str(a.scene)), str(Path(a.out).with_suffix(".mjb")), None)
    T.Terrain.from_model(m).save(Path(a.out).with_suffix(".terrain.npz"))
    np.savez(a.out, qpos=q.astype(np.float32), qvel=vel.astype(np.float32), ctrl=ctrl.astype(np.float32),
             warm=warm.astype(np.float32), out_qpos=q1, out_qvel=v1, ncon=ncon,
             region=region, geoms=geoms, iterations=a.iterations, base=np.stack(base, 1),
             mujoco_warp=importlib.metadata.version("mujoco_warp"), fric_scale=a.fric_scale)
    print(f"ref: {B} worlds, mean contacts (last substep) {ncon.mean():.1f}, max {ncon.max()}; {a.out}")


def terrain_lift(m, d):
    """How far to lift a corpus state (made on a flat floor at z = 0): the highest terrain point below its robot
    collision geoms, so that the geom above it keeps its contact with the floor as contact with the terrain."""
    import mujoco
    best = 0.0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 or m.geom_group[g] != 3:
            continue
        p = d.geom_xpos[g]
        gid = np.array([-1], np.int32)
        dist = mujoco.mj_ray(m, d, np.array([p[0], p[1], 5.0]), np.array([0.0, 0.0, -1.0]),
                             np.array([1, 0, 0, 0, 0, 0], np.uint8), 1, -1, gid)  # terrain geoms: group 0
        if dist >= 0:
            best = max(best, 5.0 - dist)
    return best



def ref_terrain(ref):
    """The terrain of reference REF.npz: REF.terrain.npz when present (written here on first use), else read from
    REF.mjb's world body."""
    import mujoco
    from nablaengine import terrain as T
    tp = Path(ref).with_suffix(".terrain.npz")
    if tp.exists():
        return T.Terrain.load(tp)
    ter = T.Terrain.from_model(mujoco.MjModel.from_binary_path(str(Path(ref).with_suffix(".mjb"))))
    ter.save(tp)
    return ter

def cmd_check(a):
    import torch
    from nablaengine import terrain as T, torch_ffi
    r = np.load(a.ref)
    ter = ref_terrain(a.ref)
    cfg = T.TerrainConfig(geoms=str(r["geoms"]), maxcon=a.maxcon, mode=a.mode)
    torch_ffi.init(terrain=cfg)
    torch_ffi.set_terrain(ter)
    B = len(r["qpos"])
    t = lambda k: torch.from_numpy(np.ascontiguousarray(r[k], np.float32)).cuda()  # noqa: E731
    fr = torch.ones(B, device="cuda")
    com = torch.zeros(B, 3, device="cuda")
    q1, v1, w1, info_ = torch_ffi.forward(t("qpos"), t("qvel"), t("warm"), t("ctrl"), fr, com,
                                           iterations=int(r["iterations"]))
    q1, v1, info_ = q1.cpu().numpy(), v1.cpu().numpy(), info_.cpu().numpy()
    ev = np.linalg.norm(v1 - r["out_qvel"], axis=1) / np.maximum(np.linalg.norm(r["out_qvel"], axis=1), 1e-6)
    eq = np.abs(q1 - r["out_qpos"]).max(1)
    res = dict(nworld=B, geoms=str(r["geoms"]), boxes=len(ter.boxes), hfields=len(ter.hf_head),
               qvel_rel_median=float(np.median(ev)), qvel_rel_p99=float(np.quantile(ev, 0.99)),
               qvel_rel_max=float(ev.max()), qpos_abs_median=float(np.median(eq)), qpos_abs_max=float(eq.max()),
               ncon_equal=float((info_[:, 1] == r["ncon"]).mean()), ncon_mean=float(info_[:, 1].mean()),
               overflow=int((info_[:, 3] != 0).sum()))
    for k in np.unique(r["region"]):
        sel = r["region"] == k
        res[f"region{k}_qvel_rel_median"] = float(np.median(ev[sel]))
        res[f"region{k}_ncon_equal"] = float((info_[sel, 1] == r["ncon"][sel]).mean())
    zero = r["ncon"] == 0
    res["nocontact_worlds"] = int(zero.sum())
    if zero.any():
        res["nocontact_qvel_rel_median"] = float(np.median(ev[zero]))
    if a.dump:
        np.savez(a.dump, q1=q1, v1=v1, info=info_)
    print(json.dumps(res, indent=1))



def cmd_grad(a):
    """Gradients against central differences of the engine itself (float32), per world along a random direction of
    qpos (not the root quaternion), qvel and ctrl with a random output cotangent: rel = |fd - an| / max(|fd|, |an|).
    --flat: the flat build on the corpus states (the finite differences' own noise on the floor)."""
    import torch
    from nablaengine import fixtures, terrain as T, torch_ffi
    if a.flat:
        z = fixtures.load_corpus("mf_s0")
        B = a.nworld
        r = dict(qpos=z["qpos"][:B], qvel=z["qvel"][:B], ctrl=z["ctrl"][:B], warm=z["qacc_warmstart"][:B],
                 region=np.zeros(B, int), iterations=a.iterations)
        torch_ffi.init()
    else:
        r = dict(np.load(a.ref))
        B = min(a.nworld, len(r["qpos"]))
        torch_ffi.init(terrain=T.TerrainConfig(geoms=str(r["geoms"]), mode=a.mode))
        torch_ffi.set_terrain(ref_terrain(a.ref))
    t = lambda k: torch.from_numpy(np.ascontiguousarray(r[k][:B], np.float32)).cuda()  # noqa: E731
    qpos, qvel, ctrl, warm = t("qpos"), t("qvel"), t("ctrl"), t("warm")
    fr, com = torch.ones(B, device="cuda"), torch.zeros(B, 3, device="cuda")
    it = int(r["iterations"])
    gen = torch.Generator(device="cuda").manual_seed(0)
    yq = torch.randn(B, qpos.shape[1], device="cuda", generator=gen)
    yv = torch.randn(B, qvel.shape[1], device="cuda", generator=gen)

    def out(q, v, u):
        q1, v1, _, info = torch_ffi.forward(q, v, warm, u, fr, com, iterations=it)
        return (yq * q1).sum(1).double() + (yv * v1).sum(1).double(), info[:, 1]

    xs = [qpos.clone().requires_grad_(), qvel.clone().requires_grad_(), ctrl.clone().requires_grad_()]
    q1, v1, _ = torch_ffi.step(xs[0], xs[1], warm, xs[2], fr, com, iterations=it)
    ((yq * q1).sum() + (yv * v1).sum()).backward()
    res = dict(nworld=B, flat=bool(a.flat), geoms=None if a.flat else str(r["geoms"]))
    region = np.asarray(r["region"][:B])
    for name, x, eps in (("qpos", qpos, a.eps_q), ("qvel", qvel, a.eps_v), ("ctrl", ctrl, a.eps_u)):
        d = torch.randn(x.shape, device="cuda", generator=gen)
        if name == "qpos":
            d[:, 3:7] = 0.0
        d /= d.norm(dim=1, keepdim=True)
        args = [qpos, qvel, ctrl]
        k = ("qpos", "qvel", "ctrl").index(name)
        ap, am = list(args), list(args)
        ap[k], am[k] = x + eps * d, x - eps * d
        lp, np_ = out(*ap)
        lm, nm = out(*am)
        fd = ((lp - lm) / (2 * eps)).cpu().numpy()
        an = (xs[k].grad * d).sum(1).double().cpu().numpy()
        rel = np.abs(fd - an) / np.maximum(np.maximum(np.abs(fd), np.abs(an)), 1e-12)
        same = (np_ == nm).cpu().numpy()  # the contact count unchanged between the two sides
        res[name] = dict(rel_median=float(np.median(rel[same])), rel_p90=float(np.quantile(rel[same], 0.9)),
                         same_contacts=float(same.mean()))
        if not a.flat:
            for g in np.unique(region):
                sel = same & (region == g)
                res[name][f"region{g}_rel_median"] = float(np.median(rel[sel]))
    print(json.dumps(res, indent=1))


def cmd_eval(a):
    """Two MuJoCo Warp references of the same states (e.g. a friction scale): per region the one-step qvel difference relative to |qvel| (median) and the base position
    divergence after each control step (median, mm)."""
    r0 = np.load(a.ref)
    res = {}
    for o in a.other:
        r1 = np.load(o)
        ev = np.linalg.norm(r1["out_qvel"] - r0["out_qvel"], axis=1) / np.maximum(
            np.linalg.norm(r0["out_qvel"], axis=1), 1e-6)
        dv = np.linalg.norm(r1["base"] - r0["base"], axis=2) * 1e3  # (B, steps) mm
        out = {}
        for k, name in enumerate(("stairs", "box", "hfield")):
            sel = r0["region"] == k
            out[name] = dict(qvel_rel_median=round(float(np.median(ev[sel])), 4),
                             base_mm_median={s + 1: round(float(np.median(dv[sel, s])), 2)
                                             for s in sorted({0, 4, 9, 24, dv.shape[1] - 1}) if s < dv.shape[1]})
        res[Path(o).stem] = out
    print(json.dumps(res, indent=1))


def cmd_flathf(a):
    """The engine on the corpus states (one control step): --build flat (the floor plane) or a terrain build whose
    terrain is one flat heightfield at z = 0 (--mode analytic / native) without the floor; writes q1, v1, info."""
    import torch
    from nablaengine import fixtures, terrain as T, torch_ffi
    z = fixtures.load_corpus("mf_s0")
    B = a.nworld
    if a.build == "flat":
        torch_ffi.init()
    else:
        torch_ffi.init(terrain=T.TerrainConfig(geoms=a.geoms, mode=a.build, cap_points=a.cap_points))
        n = 201
        ter = T.Terrain()
        ter.hf_head = np.concatenate([[0.0, 0.0, 0.0], np.eye(3).ravel(), [10.0, 10.0, 0.1, 0.1]])[None]
        ter.hf_int = np.array([[n, n, 0, 0]])
        ter.hf_data = np.zeros(n * n)
        torch_ffi.set_terrain(ter)
    t = lambda k: torch.from_numpy(np.ascontiguousarray(z[k][:B], np.float32)).cuda()  # noqa: E731
    fr = torch.ones(B, device="cuda") if a.unit_friction else t("friction")
    q1, v1, _, info = torch_ffi.forward(t("qpos"), t("qvel"), t("qacc_warmstart"), t("ctrl"), fr, t("com"),
                                        iterations=a.iterations)
    np.savez(a.out, q1=q1.cpu().numpy(), v1=v1.cpu().numpy(), info=info.cpu().numpy())
    print(f"{a.build}: mean contacts {info[:, 1].float().mean().item():.2f}, flags {int((info[:, 3] != 0).sum())}")


def cmd_time(a):
    """Time per control step (CUDA events, median of --reps) of the forward and of the forward with the backward, at
    --nworld worlds: the reference's states tiled; --mode flat (the floor build on the same states, no terrain),
    analytic or native."""
    import torch
    from nablaengine import terrain as T, torch_ffi
    r = np.load(a.ref)
    if a.mode == "flat":
        torch_ffi.init()
    else:
        torch_ffi.init(terrain=T.TerrainConfig(geoms=str(r["geoms"]), mode=a.mode))
        torch_ffi.set_terrain(ref_terrain(a.ref))
    B = a.nworld
    idx = np.arange(B) % len(r["qpos"])
    t = lambda k: torch.from_numpy(np.ascontiguousarray(r[k][idx], np.float32)).cuda()  # noqa: E731
    qpos, qvel, ctrl, warm = t("qpos"), t("qvel"), t("ctrl"), t("warm")
    fr, com = torch.ones(B, device="cuda"), torch.zeros(B, 3, device="cuda")

    def fwd():
        torch_ffi.forward(qpos, qvel, warm, ctrl, fr, com, iterations=int(r["iterations"]))

    u = ctrl.clone().requires_grad_()

    def both():
        q1, v1, _ = torch_ffi.step(qpos, qvel, warm, u, fr, com, iterations=int(r["iterations"]))
        (q1.sum() + v1.sum()).backward()

    res = dict(mode=a.mode, nworld=B, gpu=torch.cuda.get_device_name())
    for name, f in (("forward_ms", fwd), ("forward_backward_ms", both)):
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
        res[name] = round(float(np.median(ts)), 3)
    print(json.dumps(res))


RAY_PATTERNS = (  # (name, size, resolution, direction, alignment)
    ("down_yaw", (1.6, 1.0), 0.1, (0.0, 0.0, -1.0), "yaw"),
    ("slant_base", (1.0, 1.0), 0.1, (0.5, 0.0, -1.0), "base"),
)


def _grid_np(size, res, direction):
    """mjlab's GridPatternCfg rays in NumPy float32 (the reference side; torch_ffi.grid_pattern on the engine side)."""
    x = np.arange(-size[0] / 2, size[0] / 2 + res * 0.5, res, dtype=np.float32)
    y = np.arange(-size[1] / 2, size[1] / 2 + res * 0.5, res, dtype=np.float32)
    gx, gy = np.meshgrid(x, y, indexing="xy")
    off = np.zeros((gx.size, 3), np.float32)
    off[:, 0], off[:, 1] = gx.ravel(), gy.ravel()
    d = np.asarray(direction, np.float32)
    return off, np.broadcast_to(d / np.linalg.norm(d), off.shape).copy()


def _align_np(R, alignment):
    if alignment == "base":
        return R
    if alignment == "world":
        return np.broadcast_to(np.eye(3), R.shape).copy()
    x = R[:, :, 0].copy()
    x[:, 2] = 0
    xn = np.linalg.norm(x, axis=1)
    sing = xn < 0.1
    y = R[:, :, 1].copy()
    y[:, 2] = 0
    y /= np.maximum(np.linalg.norm(y, axis=1), 1e-6)[:, None]
    x[sing] = np.stack([y[:, 1], -y[:, 0], np.zeros(len(y))], 1)[sing]
    xn[sing] = 1.0
    x /= np.maximum(xn, 1e-6)[:, None]
    out = np.zeros_like(R)
    out[:, 0, 0], out[:, 1, 0], out[:, 0, 1], out[:, 1, 1], out[:, 2, 2] = x[:, 0], x[:, 1], -x[:, 1], x[:, 0], 1
    return out


def cmd_raysref(a):
    """MuJoCo Warp's rays (mjlab's RayCastSensor) from the pelvis of REF's input states against the terrain geoms
    (group 0), per RAY_PATTERNS; writes the frame poses and the distances / normals to --out."""
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src.types import vec6
    r = np.load(a.ref)
    m = mujoco.MjModel.from_binary_path(str(Path(a.ref).with_suffix(".mjb")))
    B = min(a.nworld, len(r["qpos"]))
    d = mujoco.MjData(m)
    pos, rot = np.zeros((B, 3)), np.zeros((B, 3, 3))
    for i in range(B):
        d.qpos[:] = r["qpos"][i]
        mujoco.mj_kinematics(m, d)
        pos[i], rot[i] = d.xpos[1], d.xmat[1].reshape(3, 3)
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, mujoco.MjData(m), nworld=B)
    wp.copy(dw.qpos, wp.array(np.ascontiguousarray(r["qpos"][:B], np.float32)))
    mjw.kinematics(mw, dw)
    out = dict(pos=pos.astype(np.float32), rot=rot.astype(np.float32), region=r["region"][:B])
    for name, size, res, direction, align in RAY_PATTERNS:
        off, dirs = _grid_np(size, res, direction)
        R = _align_np(rot.astype(np.float32), align).astype(np.float32)
        org = (pos.astype(np.float32)[:, None] + np.einsum("bij,nj->bni", R, off)).astype(np.float32)
        vec = np.einsum("bij,nj->bni", R, dirs).astype(np.float32)
        N = off.shape[0]
        dist = wp.zeros((B, N), dtype=float)
        gid = wp.zeros((B, N), dtype=int)
        nrm = wp.zeros((B, N), dtype=wp.vec3)
        mjw.rays(mw, dw, wp.array(org, dtype=wp.vec3), wp.array(vec, dtype=wp.vec3), vec6(-1, 0, 0, 0, 0, 0), True,
                 wp.array(np.full(N, -1, np.int32), dtype=int), dist, gid, nrm)
        dd = dist.numpy()
        dd[dd > 10.0] = -1.0
        out[f"{name}_dist"], out[f"{name}_normal"] = dd, nrm.numpy()
    np.savez(a.out, **out)
    print(f"raysref: {B} worlds, {a.out}")


def cmd_rays(a):
    """torch_ffi.raycast against raysref's MuJoCo Warp rays: distance differences and hit agreement per pattern."""
    import torch
    from nablaengine import terrain as T, torch_ffi
    z = np.load(a.rays)
    torch_ffi.init(terrain=T.TerrainConfig(geoms="feet", mode=a.mode))
    torch_ffi.set_terrain(ref_terrain(a.ref))
    pos, rot = torch.from_numpy(z["pos"]).cuda(), torch.from_numpy(z["rot"]).cuda()
    res = {}
    for name, size, res_, direction, align in RAY_PATTERNS:
        off, dirs = torch_ffi.grid_pattern(size, res_, direction, device="cuda")
        o = torch_ffi.raycast(pos, rot, off, dirs, alignment=align)
        dist, nrm = o["distances"].cpu().numpy(), o["normals_w"].cpu().numpy()
        rd, rn = z[f"{name}_dist"], z[f"{name}_normal"]
        both = (dist >= 0) & (rd >= 0)
        res[name] = dict(rays=int(dist.size), hit_agree=float(((dist >= 0) == (rd >= 0)).mean()),
                         hit_share=float((rd >= 0).mean()),
                         dist_abs_median=float(np.median(np.abs(dist - rd)[both])),
                         dist_abs_max=float(np.abs(dist - rd)[both].max()),
                         normal_abs_max=float(np.abs(nrm - rn)[both].max()),
                         dist_abs_gt_1e4=int((np.abs(dist - rd)[both] > 1e-4).sum()),
                         normal_abs_gt_1e3=int((np.abs(nrm - rn).max(-1)[both] > 1e-3).sum()))
    print(json.dumps(res, indent=1))


def cmd_vjp(a):
    """One control step and its ctrl VJP (cotangent: fixed random weights on qpos', qvel') on REF's inputs with the
    terrain, through --api torch or jax (the same kernels: the two outputs must be bitwise equal); writes --out."""
    from nablaengine import terrain as T
    r = np.load(a.ref)
    B = len(r["qpos"])
    rng = np.random.default_rng(0)
    yq = rng.standard_normal((B, r["qpos"].shape[1])).astype(np.float32)
    yv = rng.standard_normal((B, r["qvel"].shape[1])).astype(np.float32)
    cfg = T.TerrainConfig(geoms=str(r["geoms"]), mode=a.mode)
    f = lambda k: np.ascontiguousarray(r[k], np.float32)  # noqa: E731
    fr, com = np.ones(B, np.float32), np.zeros((B, 3), np.float32)
    it = int(r["iterations"])
    if a.api == "torch":
        import torch
        from nablaengine import torch_ffi
        torch_ffi.init(terrain=cfg)
        torch_ffi.set_terrain(ref_terrain(a.ref))
        t = lambda x: torch.from_numpy(x).cuda()  # noqa: E731
        u = t(f("ctrl")).requires_grad_()
        q1, v1, _ = torch_ffi.step(t(f("qpos")), t(f("qvel")), t(f("warm")), u, t(fr), t(com), iterations=it)
        ((t(yq) * q1).sum() + (t(yv) * v1).sum()).backward()
        out = dict(q1=q1.detach().cpu().numpy(), v1=v1.detach().cpu().numpy(), g=u.grad.cpu().numpy())
        if a.rays:
            z = np.load(a.rays)
            off, dirs = torch_ffi.grid_pattern((1.6, 1.0), 0.1, (0.0, 0.0, -1.0), device="cuda")
            o = torch_ffi.raycast(t(z["pos"]), t(z["rot"]), off, dirs, alignment="yaw")
            out["ray_dist"], out["ray_normal"] = o["distances"].cpu().numpy(), o["normals_w"].cpu().numpy()
    else:
        import jax
        import jax.numpy as jnp
        from nablaengine import jax_ffi
        jax_ffi.init(terrain=cfg)
        jax_ffi.set_terrain(ref_terrain(a.ref))

        def loss(u):
            q1, v1, _ = jax_ffi.step(f("qpos"), f("qvel"), f("warm"), u, fr, com, iterations=it)
            return jnp.sum(yq * q1) + jnp.sum(yv * v1), (q1, v1)

        (_, (q1, v1)), g = jax.jit(jax.value_and_grad(loss, has_aux=True))(jnp.asarray(f("ctrl")))
        out = dict(q1=np.asarray(q1), v1=np.asarray(v1), g=np.asarray(g))
        if a.rays:
            z = np.load(a.rays)
            off, dirs = jax_ffi.grid_pattern((1.6, 1.0), 0.1, (0.0, 0.0, -1.0))
            o = jax.jit(lambda p, R: jax_ffi.raycast(p, R, off, dirs, alignment="yaw"))(z["pos"], z["rot"])
            out["ray_dist"], out["ray_normal"] = np.asarray(o["distances"]), np.asarray(o["normals_w"])
    np.savez(a.out, **out)
    print(f"vjp ({a.api}): {a.out}")


def cmd_vjpcmp(a):
    x, y = np.load(a.a), np.load(a.b)
    print(json.dumps({k: dict(bitwise=bool(np.array_equal(x[k], y[k])), max_abs=float(np.abs(x[k] - y[k]).max()))
                      for k in x.files if k in y.files}, indent=1))

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ref")
    p.add_argument("--scene", required=True, help="the scene's binary model (robot + terrain), e.g. "
                   "fixtures/mjlab/g1_terrain_feet.mjb")
    p.add_argument("--out", required=True)
    p.add_argument("--nworld", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--origins", default=None)
    p.add_argument("--coltype", default=None)
    p.add_argument("--fric-scale", type=float, default=1.0)
    p = sub.add_parser("check")
    p.add_argument("--ref", required=True)
    p.add_argument("--maxcon", type=int, default=None)
    p.add_argument("--mode", default="analytic")
    p.add_argument("--dump", default=None)
    p = sub.add_parser("grad")
    p.add_argument("--ref", default=None)
    p.add_argument("--flat", action="store_true")
    p.add_argument("--mode", default="analytic")
    p.add_argument("--nworld", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--eps-q", type=float, default=1e-4)
    p.add_argument("--eps-v", type=float, default=1e-3)
    p.add_argument("--eps-u", type=float, default=1e-3)
    p = sub.add_parser("flathf")
    p.add_argument("--build", default="native")
    p.add_argument("--cap-points", type=int, default=3)
    p.add_argument("--unit-friction", action="store_true")
    p.add_argument("--geoms", default="feet")
    p.add_argument("--out", required=True)
    p.add_argument("--nworld", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=10)
    p = sub.add_parser("time")
    p.add_argument("--ref", required=True)
    p.add_argument("--mode", default="analytic")
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--reps", type=int, default=20)
    p = sub.add_parser("raysref")
    p.add_argument("--ref", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nworld", type=int, default=1024)
    p = sub.add_parser("rays")
    p.add_argument("--ref", required=True)
    p.add_argument("--rays", required=True)
    p.add_argument("--mode", default="analytic")
    p = sub.add_parser("vjp")
    p.add_argument("--ref", required=True)
    p.add_argument("--api", default="torch")
    p.add_argument("--mode", default="analytic")
    p.add_argument("--rays", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("vjpcmp")
    p.add_argument("a")
    p.add_argument("b")
    p = sub.add_parser("eval")
    p.add_argument("--ref", required=True)
    p.add_argument("other", nargs="+")
    a = ap.parse_args()
    {"ref": cmd_ref, "check": cmd_check, "grad": cmd_grad, "eval": cmd_eval, "flathf": cmd_flathf,
     "time": cmd_time, "raysref": cmd_raysref, "rays": cmd_rays,
     "vjp": cmd_vjp, "vjpcmp": cmd_vjpcmp}[a.cmd](a)


if __name__ == "__main__":
    main()
