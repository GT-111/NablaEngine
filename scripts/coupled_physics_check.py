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

VJP_GATES = {"qpos": 2e-3, "qvel": 5e-4, "ctrl": 1e-4, "wrench": 1e-4}
COSINE_GATE = 0.99999


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


def native_contact_model(mujoco, kind):
    mesh = 'type="mesh" mesh="hull"'
    box = 'type="box" size=".1 .12 .08"'
    first, second = ((box, box) if kind == "box_box" else (mesh, box if kind == "box_mesh" else mesh))
    return mujoco.MjModel.from_xml_string(f'''<mujoco><option integrator="implicitfast" solver="Newton"
        cone="pyramidal" timestep=".005" tolerance="1e-9" ccd_iterations="35" ccd_tolerance="1e-6"/>
      <asset><mesh name="hull" vertex="-.1 -.12 -.08 .1 -.12 -.08 -.1 .12 -.08 .1 .12 -.08
        -.1 -.12 .08 .1 -.12 .08 -.1 .12 .08 .1 .12 .08"/></asset>
      <worldbody>
        <body name="left"><freejoint/><geom {first}/>
          <body pos="0 0 .12"><joint name="left_hinge" damping=".1" armature=".001" range="-1 1"/>
            <geom type="sphere" size=".02" contype="0" conaffinity="0"/></body></body>
        <body name="right" pos=".17 .013 .009"><freejoint/><geom {second}/>
          <body pos="0 0 .12"><joint name="right_hinge" damping=".1" armature=".001" range="-1 1"/>
            <geom type="sphere" size=".02" contype="0" conaffinity="0"/></body></body>
      </worldbody><actuator><position joint="left_hinge" kp="20" kv=".2"/>
        <position joint="right_hinge" kp="20" kv=".2"/></actuator></mujoco>''')


def set_reference_com(model, base_ipos, base_sameframe, body_ids, offsets):
    """Apply physical COM offsets without the compiler's old inertial-frame shortcut."""
    import mujoco

    none = int(mujoco.mjtSameFrame.mjSAMEFRAME_NONE)
    inertia = int(mujoco.mjtSameFrame.mjSAMEFRAME_INERTIA)
    model.body_ipos[:] = base_ipos
    model.body_sameframe[:] = base_sameframe
    for r, body in enumerate(body_ids):
        model.body_ipos[body] += offsets[3*r:3*r+3]
        model.body_sameframe[body] = none
        # These compiled aliases would move authored collision/site poses with COM.
        model.geom_sameframe[(model.geom_bodyid == body) & (model.geom_sameframe == inertia)] = none
        model.site_sameframe[(model.site_bodyid == body) & (model.site_sameframe == inertia)] = none


def check_case(name, model, foot_pairs, com_ids, options, modules):
    mujoco, np, torch, ffi, step = modules
    from nablaengine.g1_engine import tape as tape_module
    rng = np.random.default_rng(731)
    model.opt.tolerance = 1e-9
    model.opt.iterations = options.iterations
    model.opt.ls_iterations = 50
    owner = ffi.create_context(device=options.device, model=model, foot_pairs=foot_pairs, com_body_id=com_ids)
    qpos = model.qpos0.copy()
    tangent = rng.normal(size=model.nv) * 0.005
    mujoco.mj_integratePos(model, qpos, tangent, 1.0)
    if name == "coupled_smooth":
        right_joint = int(model.body_jntadr[com_ids[1]])
        qpos[model.jnt_qposadr[right_joint]] += 1.0  # separate the same enabled-contact model; reuse its build
    qvel = rng.normal(size=model.nv) * 0.05
    ctrl = qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]].copy()
    applied = rng.normal(size=model.nv) * 0.02
    friction, com, kp_scale = options.friction, rng.normal(size=len(com_ids) * 3) * options.com_magnitude, options.kp_scale
    base = dict(ipos=model.body_ipos.copy(), sameframe=model.body_sameframe.copy(),
                geom_sameframe=model.geom_sameframe.copy(), site_sameframe=model.site_sameframe.copy(),
                pair_friction=model.pair_friction.copy(),
                geom_friction=model.geom_friction.copy(), gain=model.actuator_gainprm.copy(),
                bias=model.actuator_biasprm.copy())
    geoms = step.collision_geoms(model)

    def oracle(q, v, u, force, capture=False):
        set_reference_com(model, base["ipos"], base["sameframe"], com_ids, com)
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
        records = []
        for _ in range(options.substeps):
            state_q, state_v = data.qpos.copy(), data.qvel.copy()
            mujoco.mj_step(model, data)
            if capture:
                mass = np.empty((model.nv, model.nv))
                mujoco.mj_fullM(model, data, mass)
                jacobian = data.efc_J.reshape(data.nefc, model.nv).copy()
                active = data.efc_state == int(mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC)
                hessian = mass + jacobian.T @ ((data.efc_D * active)[:, None] * jacobian)
                records.append(dict(qpos=state_q.tolist(), qvel=state_v.tolist(), mass=mass.tolist(),
                                    hessian=hessian.tolist(), qacc=data.qacc.copy().tolist(),
                                    qacc_smooth=data.qacc_smooth.copy().tolist(),
                                    qfrc_bias=data.qfrc_bias.copy().tolist(), qfrc_passive=data.qfrc_passive.copy().tolist(),
                                    qfrc_actuator=data.qfrc_actuator.copy().tolist(),
                                    qfrc_smooth=data.qfrc_smooth.copy().tolist(), jacobian=jacobian.tolist(),
                                    efc_D=data.efc_D.copy().tolist(), efc_aref=data.efc_aref.copy().tolist(),
                                    efc_force=data.efc_force.copy().tolist(), efc_state=data.efc_state.copy().tolist()))
        return data.qpos.copy(), data.qvel.copy(), int(data.ncon), records

    def tensor(value, grad=False):
        return torch.tensor(np.asarray(value)[None], dtype=torch.float32, device=options.device, requires_grad=grad)

    inputs = [tensor(qpos, True), tensor(qvel, True), tensor(np.zeros(model.nv)), tensor(ctrl, True),
              tensor(np.asarray(friction)).reshape(1), tensor(com), tensor(applied, True),
              tensor(np.asarray(kp_scale)).reshape(1)]
    keywords = dict(iterations=options.iterations, substeps=options.substeps,
                    clamp_kv=1, normal=0, grad_decay=1.0)
    saved_tape = []

    def save_tensor(value):
        if tuple(value.shape) == (1, options.substeps * owner.info["rec_words"]):
            saved_tape.append(value.detach().clone())
        return value

    with torch.autograd.graph.saved_tensors_hooks(save_tensor, lambda value: value):
        qout, vout, _, forward_info = owner.step(*inputs, return_info=True, **keywords)
    torch.cuda.synchronize(options.device)
    observed = qout.detach().cpu().numpy()[0], vout.detach().cpu().numpy()[0]
    reference = oracle(qpos, qvel, ctrl, applied, capture=True)
    decoded = tape_module.decode(saved_tape[0].cpu().numpy().reshape(1, options.substeps, -1), model)
    comparisons = []
    for k, native in enumerate(reference[3]):
        factor_q = tape_module.tri_unpack(decoded["L_Q"][0, k], model.nv)
        mass = factor_q @ factor_q.T - model.opt.timestep * np.diag(model.dof_damping
                  + np.bincount(model.jnt_dofadr[model.actuator_trnid[:, 0]],
                                weights=-base["bias"][:, 2], minlength=model.nv))
        factor_h = tape_module.tri_unpack(decoded["L_H"][0, k], model.nv)
        hessian = factor_h @ factor_h.T
        native_mass, native_hessian = np.asarray(native["mass"]), np.asarray(native["hessian"])
        comparisons.append(dict(engine_L_Q=factor_q.tolist(), engine_L_H=factor_h.tolist(), engine_mass=mass.tolist(),
                                mass_relative_error=float(np.linalg.norm(mass-native_mass)/np.linalg.norm(native_mass)),
                                engine_hessian=hessian.tolist(),
                                hessian_relative_error=float(np.linalg.norm(hessian-native_hessian)/np.linalg.norm(native_hessian)),
                                engine_qacc=decoded["qacc"][0, k].tolist(),
                                engine_qacc_adv=decoded["qacc_adv"][0, k].tolist(), reference=native))
    forward_error = [float(np.linalg.norm(a-b) / max(1.0, np.linalg.norm(b)))
                     for a, b in zip(observed, reference[:2])]
    cot_q, cot_v = rng.normal(size=model.nq), rng.normal(size=model.nv)
    loss = (qout * tensor(cot_q)).sum() + (vout * tensor(cot_v)).sum()
    contact_case = name in ("coupled_contact", "coupled_mesh", "coupled_box_mesh", "coupled_box_box")
    gradients = torch.autograd.grad(loss, [inputs[k] for k in (0, 1, 3, 6)], retain_graph=contact_case)
    gradients = [value.detach().cpu().numpy()[0].astype(float) for value in gradients]

    def scalar(q, v, u, force):
        q1, v1, _, _ = oracle(q, v, u, force)
        return float(q1 @ cot_q + v1 @ cot_v)

    scalar_base = scalar(qpos, qvel, ctrl, applied)

    def finite_difference(block, index, direction):
        fd, projection = [], None
        for multiplier in (1, 2):
            epsilon = multiplier * options.epsilon
            plus = [qpos.copy(), qvel.copy(), ctrl.copy(), applied.copy()]
            minus = [qpos.copy(), qvel.copy(), ctrl.copy(), applied.copy()]
            if block == "qpos":
                mujoco.mj_integratePos(model, plus[0], direction, epsilon)
                mujoco.mj_integratePos(model, minus[0], direction, -epsilon)
                projection = (plus[0] - minus[0]) / (2*epsilon)
            else:
                plus[index] += epsilon * direction
                minus[index] -= epsilon * direction
                projection = direction
            fd.append((scalar(*plus) - scalar(*minus)) / (2*epsilon))
        analytic = float(gradients[index] @ projection)
        # The observed change across h and 2h estimates central-difference
        # uncertainty. The second term records float64 scalar cancellation.
        floor = max(abs(fd[0] - fd[1]), 8*np.finfo(float).eps*max(1.0, abs(scalar_base))/options.epsilon)
        absolute = abs(analytic-fd[0])
        relative = absolute / max(abs(fd[0]), floor)
        floored = max(0.0, absolute-floor) / max(abs(fd[0]), floor)
        return dict(analytic=analytic, finite_difference=fd[0], finite_difference_2h=fd[1],
                    finite_difference_floor=floor, absolute_error=absolute, relative_error=relative,
                    floored_relative_error=floored, passed=floored <= VJP_GATES[block])

    errors, full = {}, {}
    for index, (block, width) in enumerate((("qpos", model.nv), ("qvel", model.nv),
                                            ("ctrl", model.nu), ("wrench", model.nv))):
        rows = [finite_difference(block, index, direction) for direction in np.eye(width)]
        analytic = np.asarray([row["analytic"] for row in rows])
        fd = np.asarray([row["finite_difference"] for row in rows])
        scale = np.linalg.norm(analytic) * np.linalg.norm(fd)
        cosine = float(analytic @ fd / scale) if scale else float(np.array_equal(analytic, fd))
        full[block] = dict(relative_gate=VJP_GATES[block], cosine_gate=COSINE_GATE, cosine=cosine, coordinates=rows,
                           max_absolute_error=max((row["absolute_error"] for row in rows), default=0.0),
                           max_relative_error=max((row["relative_error"] for row in rows), default=0.0),
                           max_floored_relative_error=max((row["floored_relative_error"] for row in rows), default=0.0),
                           passed=all(row["passed"] for row in rows) and cosine >= COSINE_GATE)
        directions = []
        for _ in range(options.directions):
            direction = rng.normal(size=width)
            direction /= np.linalg.norm(direction)
            directions.append(finite_difference(block, index, direction))
        errors[block] = directions
    info = forward_info.detach().cpu().numpy().tolist()
    cross_root = None
    if contact_case:
        right_dof = int(model.body_dofadr[com_ids[1]])
        cross_grad = torch.autograd.grad(vout[0, right_dof], inputs[6])[0]
        cross_root = float(cross_grad[0, 0].detach().cpu())
    diagnostics = dict(zip(ffi.DIAG, (int(value) for value in owner.diagnostics().cpu())))
    fatal_diagnostics = {key: diagnostics[key] for key in
                         ("fwd_unsupported_worlds", "nonconv_records", "singular_records",
                          "replay_mismatch_worlds", "nonfinite_rows")}
    result = dict(name=name, dimensions=dict(nbody=model.nbody, nq=model.nq, nv=model.nv, nu=model.nu, neq=model.neq),
                  context=owner.info, forward_relative_error=forward_error, directional_vjp=errors, full_gradient_vjp=full,
                  physics_inputs=dict(qpos=qpos.tolist(), qvel=qvel.tolist(), ctrl=ctrl.tolist(),
                                      applied_force=applied.tolist(), friction=friction, com=com.tolist(), kp_scale=kp_scale),
                  observed=dict(qpos=observed[0].tolist(), qvel=observed[1].tolist()),
                  reference=dict(qpos=reference[0].tolist(), qvel=reference[1].tolist()),
                  tape_reference_comparisons=comparisons,
                  reference_contacts=reference[2], info=info, diagnostics=diagnostics,
                  fatal_diagnostics=fatal_diagnostics,
                  cross_root_wrench_to_velocity=cross_root,
                  mode=dict(clamp_kv=1, normal=0, grad_decay=1, kp_scale=kp_scale),
                  passed=bool(max(forward_error) < 5e-4 and all(row[3] == 0 for row in info)
                              and not any(fatal_diagnostics.values())
                              and all(block["passed"] for block in full.values())
                              and all(row["passed"] for rows in errors.values() for row in rows)
                              and (cross_root is None or abs(cross_root) > 1e-9)))
    for key, target in (("ipos", "body_ipos"), ("sameframe", "body_sameframe"),
                        ("geom_sameframe", "geom_sameframe"), ("site_sameframe", "site_sameframe"),
                        ("pair_friction", "pair_friction"), ("geom_friction", "geom_friction"),
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
    parser.add_argument("--com-magnitude", type=float, default=0.002)
    parser.add_argument("--kp-scale", type=float, default=1.1)
    parser.add_argument("--friction", type=float, default=0.65)
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
            elif name in ("coupled_contact", "coupled_smooth"):
                model = primitive_model(mujoco)
                pairs, bodies = list(range(len(step.contact_pairs(model)))), tuple(source.free_roots(model))
            elif name in ("coupled_mesh", "coupled_box_mesh", "coupled_box_box"):
                model = native_contact_model(mujoco, name.removeprefix("coupled_"))
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
