"""CPU context/typed-shape/custom-VJP checks with explicit analytic FFI doubles.

No CUDA solver executes here. Real hand/contact/PD qualification remains a
separate allocated-GPU check of the same context API.
"""

from contextlib import contextmanager
import ctypes
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from nablaengine import jax_ffi as F


def bilateral_model(mass=1.):
    import mujoco
    return mujoco.MjModel.from_xml_string(f"""<mujoco>
      <option timestep="0.005" integrator="implicitfast" cone="pyramidal" iterations="100"/>
      <worldbody>
        <body name="left" pos="-.2 0 .5"><freejoint/>
          <geom type="sphere" size=".04" mass="{mass}"/>
          <body pos="0 0 .1"><joint name="left_finger" type="hinge" range="-1 1"/>
            <geom type="capsule" size=".01 .04" mass=".1"/></body></body>
        <body name="right" pos=".2 0 .5"><freejoint/>
          <geom type="sphere" size=".04" mass="1"/>
          <body pos="0 0 .1"><joint name="right_finger" type="hinge" range="-1 1"/>
            <geom type="capsule" size=".01 .04" mass=".1"/></body></body>
      </worldbody>
      <actuator><position joint="left_finger" kp="20" kv="1"/>
        <position joint="right_finger" kp="20" kv="1"/></actuator>
    </mujoco>""")


def kin(q, v, com, info):
    nb = info["nbody"]
    position = jnp.broadcast_to(q[:, None, :3], (q.shape[0], nb, 3))
    position = position + jax.lax.stop_gradient(com[:, None, :3])
    quaternion = jnp.broadcast_to(q[:, None, :4], (q.shape[0], nb, 4))
    velocity = jnp.broadcast_to(v[:, None, :6], (q.shape[0], nb, 6))
    return q, position, quaternion, velocity, position


def state(q, v, w, u, force, kp, info):
    gain = np.float32(info["gain"])
    return gain*q, gain*v + u.sum(-1, keepdims=True) + force*kp[:, None], jax.lax.stop_gradient(w)


def inputs(owner):
    i, B = owner.info, 2
    shapes = ((B, i["nq"]), (B, i["nv"]), (B, i["nv"]), (B, i["nu"]), (B,),
              (B, i["com_width"]), (B, i["nv"]), (B,))
    arrays = [jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)/64 for shape in shapes]
    arrays[7] = jnp.array([.5, 1.5], jnp.float32)
    return tuple(arrays)


@contextmanager
def contexts():
    owners, targets, calls = [], {}, []
    for index in range(3):
        owner = F.ModelContext(index)
        # Models 0 and 2 deliberately have equal shapes and different constants.
        extra = int(index == 1)
        owner._state["info"] = dict(key=f"jax-context-check-{index}", nq=8+extra, nv=7+extra,
            nu=2+extra, nbody=3+extra, rec_words=9+extra, maxcon=2, com_width=3*(1+extra),
            com_body_ids=tuple(range(1, 2+extra)), kinematics_error=None, gain=index+1.)
        owner._state["targets"] = {}
        for kind in ("step", "step_kin", "fwd", "fwd_kin", "bwd", "bwd_soft", "bwd_kin", "bwd_soft_kin",
                     "kin", "kin_bwd"):
            name = f"jax-context-check-{index}/{kind}"
            owner._state["targets"][kind] = name
            targets[name] = owner, kind
        owners.append(owner)

    def ffi_call(target, outputs, **options):
        owner, kind = targets[target]
        info = owner.info

        def call(*args, **attrs):
            calls.append((owner.context_id, kind, tuple(value.shape for value in args)))
            if kind in ("step", "step_kin", "fwd", "fwd_kin"):
                if len(args) != 8:
                    raise AssertionError("native forward must receive force and Kp")
                q, v, w, u, _fr, com, force, kp = args
                q1, v1, w1 = state(q, v, w, u, force, kp, info)
                flags = jnp.broadcast_to(jnp.array([attrs["iterations"], 17, 23, F.OVERFLOW_ROWS], jnp.int32),
                                         (q.shape[0], 4))
                result = [q1, v1, w1, flags, jnp.zeros(outputs[4].shape, jnp.float32)]
                if kind.startswith("fwd"):
                    result.append(jnp.zeros(outputs[5].shape, jnp.float32))
                if kind.endswith("_kin"):
                    result.extend(kin(q1, v1, com, info))
            elif kind.startswith("bwd"):
                tp, _fr, com, force, kp, *cot = args
                if kind.startswith("bwd_soft"):
                    cot = cot[1:]  # The soft branch also receives its original ctrl.
                if kind.endswith("_kin"):
                    q1, v1, gq, gv, *gkin = cot
                    _, pull = jax.vjp(lambda q, v: kin(q, v, com, info), q1, v1)
                    kq, kv = pull(tuple(gkin))
                    gq, gv = gq+kq, gv+kv
                else:
                    gq, gv = cot
                B = tp.shape[0]
                gu = jnp.broadcast_to(gv.sum(-1, keepdims=True), (B, info["nu"]))
                result = [info["gain"]*gq, info["gain"]*gv, gu, jnp.zeros((B,), jnp.int32), gv*kp[:, None]]
            elif kind == "kin":
                result = kin(*args, info)
            else:
                q, v, com, *cot = args
                _, pull = jax.vjp(lambda a, b: kin(a, b, com, info), q, v)
                result = pull(tuple(cot))
            if len(result) != len(outputs):
                raise AssertionError((kind, len(result), len(outputs)))
            for value, spec in zip(result, outputs, strict=True):
                if value.shape != spec.shape or value.dtype != spec.dtype:
                    raise AssertionError((kind, value.shape, value.dtype, spec))
            return result
        return call

    with patch.object(jax.ffi, "ffi_call", ffi_call), patch.object(F, "init", side_effect=AssertionError("global model used")):
        yield owners, calls
    F._make_step.cache_clear()
    F._kinematics_fn.cache_clear()


class ContextChecks(unittest.TestCase):
    def assert_tree_equal(self, actual, expected):
        aa, a_tree = jax.tree.flatten(actual)
        ee, e_tree = jax.tree.flatten(expected)
        self.assertEqual(a_tree, e_tree)
        for a, e in zip(aa, ee, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(e))

    def test_factory_retains_two_free_com_facts_and_reuses_only_the_same_model(self):
        # Library loading/registration and capability admission are explicit
        # doubles. Constant packing and the bilateral native chart are real.
        models = (bilateral_model(), bilateral_model(1.5))
        libraries = []
        for model in models:
            step, tape = F.engines.modules(model).step, F.engines.modules(model).tape
            maximum = step.maxcon(model)
            rec = tape.rec_words(model, maximum, tape.kind(None))
            rows = np.array([[1, 1, rec, model.nq, model.nv, maximum, kv, normal,
                              tape.layout(model)["flags"][0]] for kv, normal in F.VARIANTS], np.int64)

            def sizes(pointer, values=rows):
                target = np.ctypeslib.as_array((ctypes.c_int64 * values.size).from_address(pointer.value))
                target[:] = values.reshape(-1)

            lib = SimpleNamespace(dk_set_constants=lambda pointer, count: 0, dk_sizes=sizes,
                                  dk_launch_info=lambda *args: 0, dk_last_slots=lambda *args: None)
            for symbol in ("DkStep", "DkStepFwd", "DkStepBwd", "DkStepBwdSoft", "DkDiag", "DkLast",
                           "DkStepKin", "DkStepFwdKin", "DkStepBwdKin", "DkStepBwdSoftKin", "DkKin", "DkKinBwd"):
                setattr(lib, symbol, object())
            libraries.append(lib)
        builds = [(Path("model-A.so"), False, "model-A-key"), (Path("model-B.so"), False, "model-B-key"),
                  (Path("model-A.so"), False, "model-A-key")]
        with patch.dict(F._CONTEXTS, {}, clear=True), patch.dict(F._MODEL_CONTEXTS, {}, clear=True), \
                patch.object(F, "_arch", return_value="sm_90"), patch.object(F, "_build", side_effect=builds), \
                patch.object(F.ctypes.cdll, "LoadLibrary", side_effect=libraries), \
                patch.object(F.capability, "require") as admission, \
                patch.object(jax.ffi, "pycapsule", side_effect=lambda symbol: symbol), \
                patch.object(jax.ffi, "register_ffi_target") as register, patch("atexit.register"):
            first = F.create_context(model=models[0], foot_pairs=[], com_body_id=(1, 3))
            second = F.create_context(model=models[1], foot_pairs=[], com_body_id=(1, 3))
            again = F.create_context(model=models[0], foot_pairs=[], com_body_id=(1, 3))
            self.assertIs(first, again)
            self.assertIsNot(first, second)
            self.assertEqual((first.info["com_width"], first.info["com_body_ids"], first.info["torso_id"]),
                             (6, (1, 3), 1))
            self.assertEqual(len(register.call_args_list), 24)
            native_admissions = [call for call in admission.call_args_list if call.args[-1] == "jax_ffi.init"]
            self.assertEqual([call.args[2] for call in native_admissions], [(1, 3)]*3)
            self.assertEqual(first.check_model(models[0])["key"], "model-A-key")
            self.assertEqual(second.check_model(models[1])["key"], "model-B-key")
            with self.assertRaises(ValueError):
                first.check_model(models[1])
            with self.assertRaises(ValueError):
                first.check_model(models[0], com_body_id=(1, 2))

    def test_model_owned_shapes_info_and_equal_shape_cache(self):
        with contexts() as (owners, calls):
            for owner in (*owners, owners[0]):
                args = inputs(owner)
                for use_kin in (False, True):
                    program = jax.jit(lambda *x: owner.step(*x, substeps=1, iterations=100,
                                                          kinematics=use_kin, return_info=True))
                    result = program(*args)
                    expected = state(args[0], args[1], args[2], args[3], args[6], args[7], owner.info)
                    self.assert_tree_equal(result[:3], expected)
                    np.testing.assert_array_equal(np.asarray(result[3]), [[100, 17, 23, F.OVERFLOW_ROWS]]*2)
                    if use_kin:
                        self.assert_tree_equal(result[4:], kin(expected[0], expected[1], args[5], owner.info))
                    self.assertEqual(owner.tape_bytes(2, substeps=1), 8*owner.info["rec_words"])
            self.assertEqual({item[0] for item in calls}, {0, 1, 2})

    def test_force_vjp_stopped_parameters_and_successor_kinematics(self):
        with contexts() as (owners, _calls):
            for owner in owners:
                args = inputs(owner)
                for use_kin in (False, True):
                    for return_info in (False, True):
                        for clamp in (0, 2):
                            decay = .5 if clamp == 2 else 1.
                            def loss(*values):
                                out = owner.step(*values, substeps=1, iterations=100, clamp_kv=clamp,
                                                 kinematics=use_kin, return_info=return_info, grad_decay=decay)
                                outputs = out[:3] + out[4:] if return_info else out
                                return sum(value.sum() for value in outputs)

                            def reference(*values):
                                q, v, w, u, fr, com, force, kp = values
                                q, v = F.scale_grad((q, v), decay)
                                out = state(q, v, w, u, force, jax.lax.stop_gradient(kp), owner.info)
                                extra = kin(out[0], out[1], com, owner.info) if use_kin else ()
                                return sum(value.sum() for value in (*out, *extra))

                            expected = jax.jit(jax.value_and_grad(reference, argnums=tuple(range(8))))(*args)
                            actual = jax.jit(jax.value_and_grad(loss, argnums=tuple(range(8))))(*args)
                            self.assert_tree_equal(actual, expected)
                            self.assertTrue(np.asarray(actual[1][6]).any())
                            for index in (2, 4, 5, 7):
                                self.assertFalse(np.asarray(actual[1][index]).any())

    def test_default_force_and_kp_and_separate_kinematics(self):
        with contexts() as (owners, _calls):
            for owner in owners:
                q, v, w, u, fr, com, force, kp = inputs(owner)
                default = owner.make_step(substeps=1)
                out = jax.jit(default)(q, v, w, u, fr, com)
                self.assert_tree_equal(out, owner.step(q, v, w, u, fr, com, jnp.zeros_like(force), jnp.ones_like(kp),
                                                      substeps=1))
                grad = jax.jit(jax.grad(lambda q, v, u: sum(x.sum() for x in default(q, v, w, u, fr, com)),
                                        argnums=(0, 1, 2)))(q, v, u)
                self.assertTrue(all(np.isfinite(np.asarray(value)).all() for value in grad))
                direct = jax.jit(owner.kinematics)(q, v, com)
                self.assert_tree_equal(direct, kin(q, v, com, owner.info))
                actual = jax.jit(jax.grad(lambda q, v: sum(x.sum() for x in owner.kinematics(q, v, com)),
                                          argnums=(0, 1)))(q, v)
                expected = jax.jit(jax.grad(lambda q, v: sum(x.sum() for x in kin(q, v, com, owner.info)),
                                            argnums=(0, 1)))(q, v)
                self.assert_tree_equal(actual, expected)

    def test_bad_force_and_kp_are_rejected_before_native_call(self):
        with contexts() as (owners, calls):
            owner = owners[0]
            args = list(inputs(owner))
            for index, bad in ((6, jnp.zeros((2, 1))), (7, jnp.ones((2, 1)))):
                changed = args.copy()
                changed[index] = bad
                with self.assertRaises(TypeError):
                    owner.step(*changed)
            self.assertFalse(calls)

    def test_model_group_routes_empty_batches_and_the_same_vjp_and_adam(self):
        from nablaengine.jax_model_group import ModelGroup

        with contexts() as (owners, _calls):
            group = ModelGroup.__new__(ModelGroup)
            group.contexts, group.iterations = tuple(owners), (100,) * 3
            group.grad_decay, group.clamp_soft = (1., .5, 1.), None
            group.group_id, group._targets = 17, {"step": "cpu-model-group"}
            serial_call = jax.ffi.ffi_call

            def ffi_call(target, outputs, **options):
                if target != "cpu-model-group":
                    return serial_call(target, outputs, **options)

                def call(*values, **attrs):
                    result, offset = [], 0
                    for index, owner in enumerate(owners):
                        item = values[index * 8:index * 8 + 8]
                        backward = bool(attrs["backward"])
                        kind = "bwd" if backward else ("fwd" if attrs["taped"] else "step")
                        stride = 5 if backward or not attrs["taped"] else 6
                        if backward:
                            tape, friction, com, force, kp, _ctrl, cq, cv = item
                            item = (tape, friction, com, force, kp, cq, cv)
                        result.extend(serial_call(owner._state["targets"][kind], outputs[offset:offset + stride])(
                            *item, iterations=np.int32(100), substeps=attrs["substeps"]))
                        offset += stride
                    return result

                return call

            values = tuple(inputs(owner) for owner in owners)
            # A real H sampler can leave a plant empty; no padded fake world.
            values = values[:1] + (tuple(x[:0] for x in values[1]),) + values[2:]
            with patch.object(jax.ffi, "ffi_call", ffi_call):
                expected = tuple(owner.step(*item, iterations=100, substeps=1, return_info=True,
                                            grad_decay=decay)
                                 for owner, item, decay in zip(owners, values, group.grad_decay, strict=True))
                self.assert_tree_equal(jax.jit(group.step)(values), expected)

                def loss(items, grouped):
                    result = group.step(items) if grouped else tuple(owner.step(*item, iterations=100,
                        substeps=1, return_info=True, grad_decay=decay)
                        for owner, item, decay in zip(owners, items, group.grad_decay, strict=True))
                    return sum(jnp.sum(item[0]) + jnp.sum(item[1]) for item in result)

                oracle = jax.jit(jax.value_and_grad(lambda x: loss(x, False)))(values)
                actual = jax.jit(jax.value_and_grad(lambda x: loss(x, True)))(values)
                self.assert_tree_equal(actual, oracle)
                programs = [jax.jit(jax.value_and_grad(lambda force, flag=flag: loss(
                    tuple(item[:6] + (f,) + item[7:] for item, f in zip(values, force, strict=True)), flag)))
                    for flag in (False, True)]
                force = tuple(item[6] for item in values)
                zeros = jax.tree.map(jnp.zeros_like, force)
                states = [(force, zeros, zeros), (force, zeros, zeros)]
                for visit in range(1, 4):
                    results = []
                    for (parameter, moment, variance), program in zip(states, programs, strict=True):
                        value, gradient = program(parameter)
                        moment = jax.tree.map(lambda a, b: .9*a+.1*b, moment, gradient)
                        variance = jax.tree.map(lambda a, b: .95*a+.05*b*b, variance, gradient)
                        parameter = jax.tree.map(lambda p, m, v: p-.001*(m/(1-.9**visit))
                            /(jnp.sqrt(v/(1-.95**visit))+1.e-8), parameter, moment, variance)
                        results.append((value, gradient, parameter, moment, variance))
                    self.assert_tree_equal(results[0], results[1])
                    states = [(item[2], item[3], item[4]) for item in results]

    def test_h64_four_substeps_force_gradient_and_three_actual_adam_updates(self):
        with contexts() as (owners, _calls):
            owner = owners[0]  # Unit-gain analytic double remains finite for 256 native calls.
            args = inputs(owner)
            def objective(theta, native):
                def body(carry, _):
                    q, v, w = carry
                    if native:
                        out = owner.step(q, v, w, args[3], args[4], args[5], theta, args[7], substeps=1,
                                         iterations=100, return_info=True)
                        return out[:3], None
                    return state(q, v, w, args[3], theta, jax.lax.stop_gradient(args[7]), owner.info), None
                out, _ = jax.lax.scan(body, args[:3], None, length=4*64)
                return out[0].sum()+out[1].sum()
            gradients = [jax.jit(jax.value_and_grad(lambda theta: objective(theta, native))) for native in (False, True)]
            states = [(args[6], jnp.zeros_like(args[6]), jnp.zeros_like(args[6])) for _ in range(2)]
            for visit in range(1, 4):
                results = []
                for (theta, moment, variance), value_and_grad in zip(states, gradients, strict=True):
                    loss, grad = value_and_grad(theta)
                    moment = .9*moment+.1*grad
                    variance = .95*variance+.05*jnp.square(grad)
                    theta = theta - .001*(moment/(1-.9**visit))/(jnp.sqrt(variance/(1-.95**visit))+1e-8)
                    results.append((loss, grad, theta, moment, variance))
                self.assert_tree_equal(results[0], results[1])
                states = [(result[2], result[3], result[4]) for result in results]


if __name__ == "__main__":
    unittest.main()
