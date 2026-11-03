"""Finite CUDA/MuJoCo checks for the first coupled two-FREE core snapshot.

python scripts/coupled_physics_check.py --source-root /path/NablaEngine --cache /scratch/nabla-cache \
    --out /scratch/coupled-check.json --cases coupled_contact,coupled_large,coupled_equalities,g1

Runs only on the explicitly selected CUDA device. No remote execution or allocation is performed.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time


def primitive_model(mujoco):
    return mujoco.MjModel.from_xml_string('''<mujoco><option integrator="implicitfast" solver="Newton"
        cone="pyramidal" timestep="0.005" tolerance="1e-9"/><default><joint damping="0.1" armature="0.001"/></default>
      <worldbody>
        <body name="left" pos="0 0 0"><freejoint/><geom name="left_contact" type="sphere" size="0.1"/>
          <body pos="0 0 0.12"><joint name="left_hinge" range="-1 1"/>
            <geom type="sphere" size="0.02" contype="0" conaffinity="0"/></body></body>
        <body name="right" pos="0.15 0 0"><freejoint/><geom name="right_contact" type="sphere" size="0.1"/>
          <body pos="0 0 0.12"><joint name="right_hinge" range="-1 1"/>
            <geom type="sphere" size="0.02" contype="0" conaffinity="0"/></body></body>
      </worldbody><actuator><position joint="left_hinge" kp="20" kv="0.2"/>
        <position joint="right_hinge" kp="20" kv="0.2"/></actuator></mujoco>''')


def check_case(name, model, foot_pairs, com_ids, options, modules):
    mujoco, np, torch, ffi, step = modules
    rng = np.random.default_rng(731)
    model.opt.tolerance = 1e-9
    model.opt.iterations = options.iterations
    model.opt.ls_iterations = 50
    owner = ffi.create_context(device=options.device, model=model, foot_pairs=foot_pairs, com_body_id=com_ids)
    qpos = model.qpos0.copy()
    tangent = rng.normal(size=model.nv) * 0.005
    mujoco.mj_integratePos(model, qpos, tangent, 1.0)
    qvel = rng.normal(size=model.nv) * 0.05
    ctrl = qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]].copy()
    applied = rng.normal(size=model.nv) * 0.02
    friction, com, kp_scale = 0.65, rng.normal(size=len(com_ids) * 3) * 0.002, 1.1
    base = dict(ipos=model.body_ipos.copy(), pair_friction=model.pair_friction.copy(),
                geom_friction=model.geom_friction.copy(), gain=model.actuator_gainprm.copy(),
                bias=model.actuator_biasprm.copy())
    geoms = step.collision_geoms(model)

    def oracle(q, v, u, force):
        model.body_ipos[:] = base["ipos"]
        for r, body in enumerate(com_ids):
            model.body_ipos[body] += com[3*r:3*r+3]
        model.actuator_gainprm[:] = base["gain"]
        model.actuator_biasprm[:] = base["bias"]
        model.actuator_gainprm[:, 0] *= kp_scale
        model.actuator_biasprm[:, 1] *= kp_scale
        model.pair_friction[:] = base["pair_friction"]
        model.geom_friction[:] = base["geom_friction"]
        if name == "g1":
            model.pair_friction[np.asarray(foot_pairs), :2] = friction
        elif geoms:
            model.geom_friction[geoms, 0] = friction
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:], data.ctrl[:], data.qfrc_applied[:] = q, v, u, force
        for _ in range(options.substeps):
            mujoco.mj_step(model, data)
        return data.qpos.copy(), data.qvel.copy(), int(data.ncon)

    def tensor(value, grad=False):
        return torch.tensor(np.asarray(value)[None], dtype=torch.float32, device=options.device, requires_grad=grad)

    inputs = [tensor(qpos, True), tensor(qvel, True), tensor(np.zeros(model.nv)), tensor(ctrl, True),
              tensor(np.asarray(friction)).reshape(1), tensor(com), tensor(applied, True),
              tensor(np.asarray(kp_scale)).reshape(1)]
    keywords = dict(iterations=options.iterations, substeps=options.substeps,
                    clamp_kv=1, normal=0, grad_decay=1.0)
    qout, vout, _, forward_info = owner.step(*inputs, return_info=True, **keywords)
    torch.cuda.synchronize(options.device)
    observed = qout.detach().cpu().numpy()[0], vout.detach().cpu().numpy()[0]
    reference = oracle(qpos, qvel, ctrl, applied)
    forward_error = [float(np.linalg.norm(a-b) / max(1.0, np.linalg.norm(b)))
                     for a, b in zip(observed, reference[:2])]
    cot_q, cot_v = rng.normal(size=model.nq), rng.normal(size=model.nv)
    loss = (qout * tensor(cot_q)).sum() + (vout * tensor(cot_v)).sum()
    gradients = torch.autograd.grad(loss, [inputs[k] for k in (0, 1, 3, 6)],
                                    retain_graph=name == "coupled_contact")
    gradients = [value.detach().cpu().numpy()[0].astype(float) for value in gradients]

    def scalar(q, v, u, force):
        q1, v1, _ = oracle(q, v, u, force)
        return float(q1 @ cot_q + v1 @ cot_v)

    errors = {}
    for block, width in (("qpos", model.nv), ("qvel", model.nv), ("ctrl", model.nu), ("wrench", model.nv)):
        rows = []
        for _ in range(options.directions):
            direction = rng.normal(size=width)
            direction /= np.linalg.norm(direction)
            plus, minus = [qpos.copy(), qvel.copy(), ctrl.copy(), applied.copy()], [qpos.copy(), qvel.copy(), ctrl.copy(), applied.copy()]
            index = ("qpos", "qvel", "ctrl", "wrench").index(block)
            if block == "qpos":
                mujoco.mj_integratePos(model, plus[0], direction, options.epsilon)
                mujoco.mj_integratePos(model, minus[0], direction, -options.epsilon)
                input_direction = (plus[0] - minus[0]) / (2*options.epsilon)
            else:
                plus[index] += options.epsilon * direction
                minus[index] -= options.epsilon * direction
                input_direction = direction
            fd = (scalar(*plus) - scalar(*minus)) / (2*options.epsilon)
            analytic = float(gradients[index] @ input_direction)
            error = abs(analytic-fd) / max(1.0, abs(fd))
            rows.append(dict(analytic=analytic, finite_difference=fd, error=error,
                             passed=abs(analytic-fd) <= 2e-5 + 2e-3*abs(fd)))
        errors[block] = rows
    info = forward_info.detach().cpu().numpy().tolist()
    cross_root = None
    if name == "coupled_contact":
        right_dof = int(model.body_dofadr[com_ids[1]])
        cross_grad = torch.autograd.grad(vout[0, right_dof], inputs[6])[0]
        cross_root = float(cross_grad[0, 0].detach().cpu())
    diagnostics = dict(zip(ffi.DIAG, (int(value) for value in owner.diagnostics().cpu())))
    fatal_diagnostics = {key: diagnostics[key] for key in
                         ("fwd_unsupported_worlds", "nonconv_records", "singular_records",
                          "replay_mismatch_worlds", "nonfinite_rows")}
    result = dict(name=name, dimensions=dict(nbody=model.nbody, nq=model.nq, nv=model.nv, nu=model.nu, neq=model.neq),
                  context=owner.info, forward_relative_error=forward_error, directional_vjp=errors,
                  reference_contacts=reference[2], info=info, diagnostics=diagnostics,
                  fatal_diagnostics=fatal_diagnostics,
                  cross_root_wrench_to_velocity=cross_root,
                  mode=dict(clamp_kv=1, normal=0, grad_decay=1, kp_scale=kp_scale),
                  passed=bool(max(forward_error) < 5e-4 and all(row[3] == 0 for row in info)
                              and not any(fatal_diagnostics.values())
                              and all(row["passed"] for rows in errors.values() for row in rows)
                              and (cross_root is None or abs(cross_root) > 1e-9)))
    for key, target in (("ipos", "body_ipos"), ("pair_friction", "pair_friction"), ("geom_friction", "geom_friction"),
                        ("gain", "actuator_gainprm"), ("bias", "actuator_biasprm")):
        getattr(model, target)[:] = base[key]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cases", default="coupled_contact,coupled_large,coupled_equalities,g1")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--directions", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve() / "src"))
    os.environ["DK_CACHE_DIR"] = str(args.cache.resolve())
    import mujoco
    import numpy as np
    import torch
    from nablaengine import fixtures, torch_ffi
    from nablaengine.g1_engine import source, step
    from model_generation_check import hand_model
    torch.cuda.set_device(args.device)
    result = dict(source_root=str(args.source_root.resolve()), cache=str(args.cache.resolve()),
                  device=torch.cuda.get_device_name(args.device), mujoco=mujoco.__version__,
                  requested_cases=args.cases.split(","), cases=[])
    start = time.monotonic()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name in args.cases.split(","):
        try:
            if name == "g1":
                model, metadata = fixtures.load_model(args.source_root / "fixtures" / "mjlab")
                pairs, bodies = metadata["foot_pairs"], (fixtures.torso_id(model, metadata),)
            elif name == "coupled_contact":
                model = primitive_model(mujoco)
                pairs, bodies = list(range(len(step.contact_pairs(model)))), tuple(source.free_roots(model))
            elif name in ("coupled_large", "coupled_equalities"):
                eq = name == "coupled_equalities"
                model = hand_model(12 if eq else 22, 5, roots=2, passive=eq, equality=eq,
                                   damping=0.1, frictionloss=0.01 if eq else 0.0)
                model.opt.timestep = 0.005
                pairs, bodies = [], tuple(source.free_roots(model))
            else:
                raise ValueError(f"unknown finite case {name!r}")
            item = check_case(name, model, pairs, bodies, args, (mujoco, np, torch, torch_ffi, step))
        except Exception as error:
            import traceback
            item = dict(name=name, passed=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        result["cases"].append(item)
        result["elapsed_seconds"] = time.monotonic() - start
        result["complete"] = len(result["cases"]) == len(result["requested_cases"])
        result["passed"] = result["complete"] and all(case["passed"] for case in result["cases"])
        args.out.write_text(json.dumps(result, indent=2, default=lambda value: value.item()) + "\n")
        print(f"{name}: {'PASS' if item['passed'] else 'FAIL'}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
