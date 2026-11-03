"""CPU regression checks for model-specific CUDA source and packed constants.

PYTHONPATH=src python scripts/model_generation_check.py
These checks generate source; numerical CUDA qualification is a separate check.
"""

import shutil
import subprocess
import unittest

import mujoco
import numpy as np

from nablaengine.g1_engine import adjoint, source, step
from nablaengine.model import capability


def hand_model(nhinge=12, fingers=5, *, passive=False, damping=0.0, frictionloss=0.0, equality=False, roots=1):
    """One or two real FREE palms and hinge chains, with no collision geometry."""
    bodies, acts, equalities = [], [], []
    for root in range(roots):
        chains, index = [], root * nhinge
        for finger in range(fingers):
            chain = ""
            length = nhinge // fingers + (finger < nhinge % fingers)
            for _ in range(length):
                chain += (f'<body name="b{index}" pos="0.02 0 0"><joint name="j{index}" range="-1 1" '
                          f'damping="{damping}" frictionloss="{frictionloss}"/>'
                          '<inertial mass="0.1" pos="0.01 0 0" diaginertia="0.001 0.001 0.001"/>')
                index += 1
            chains.append(chain + "</body>" * length)
        bodies.append(f'<body name="palm{root}" pos="0 {root * 0.2} 0"><freejoint/>'
                      '<inertial mass="0.5" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>'
                      + "".join(chains) + '</body>')
        joints = list(range(nhinge)) if not passive else list(range(0, nhinge, 2))[::-1]
        acts += [f'<position joint="j{root * nhinge + i}" kp="20" kv="0.2"/>' for i in joints]
        if equality:
            equalities += [f'<joint joint1="j{root * nhinge + i}" joint2="j{root * nhinge + i - 1}" '
                           f'polycoef="0 {ratio} 0 0 0"/>'
                           for i, ratio in zip(range(1, nhinge, 2), (0.6, 0.8, 1.05, 1.05, 1.05, 1.18))]
    return mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><option integrator="implicitfast" solver="Newton"/>'
        '<worldbody>' + ''.join(bodies) + '</worldbody><actuator>' + ''.join(acts) + '</actuator>'
        + ('<equality>' + ''.join(equalities) + '</equality>' if equalities else '') + '</mujoco>')


class GenerationCheck(unittest.TestCase):
    @unittest.skipUnless(shutil.which("c++"), "state layout check requires a C++ compiler")
    def test_state_layout_compiles_with_root_count_macro(self):
        model = hand_model(22, roots=2)
        header = source.header(model, maxd=8, wpb=1).replace("#include <cuda_runtime.h>", "")
        sim = step.STEP_DEVICE.split("struct Sim {", 1)[1].split("// a condition that", 1)[0]
        split = step.STEP_KERNEL.split("// The split forward (host.py:", 1)[1]
        code = ("#define __device__\n" + header + step.step_header(model) + "struct Sim {" + sim
                + "// The split forward (host.py:" + split + "\nstatic_assert(sizeof(Hand) > 0);\n")
        result = subprocess.run(["c++", "-std=c++17", "-fsyntax-only", "-x", "c++", "-"],
                                input=code, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_implicit_mesh_contacts_remain_guarded(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <option integrator="implicitfast" solver="Newton" cone="pyramidal"/>
          <asset><mesh name="hull" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/></asset>
          <worldbody>
            <body name="left"><freejoint/><geom type="mesh" mesh="hull"/></body>
            <body name="right" pos="0.05 0 0"><freejoint/><geom type="mesh" mesh="hull"/></body>
          </worldbody></mujoco>''')
        self.assertEqual(model.npair, 0)
        self.assertEqual(step.contact_pairs(model), ((0, 1),))
        errors = capability.check(model, [], source.free_roots(model)).codes("error")
        self.assertTrue({"H14a", "H14b", "H14c"} <= set(errors))

    def test_native_cross_root_pairs_and_material_mix(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <option integrator="implicitfast" solver="Newton" cone="pyramidal"/>
          <worldbody>
            <body name="left" pos="0 0 0"><freejoint/>
              <geom name="g1" type="sphere" size="0.1" friction="0.3 0.01 0.002" solmix="1"
                    solref="0.02 1" solimp="0.8 0.9 0.01 0.5 2"/>
              <body pos="0 0 0.1"><joint name="j1"/><geom type="sphere" size="0.01" contype="0" conaffinity="0"/></body>
            </body>
            <body name="right" pos="0.15 0 0"><freejoint/>
              <geom name="g2" type="sphere" size="0.1" friction="0.7 0.02 0.003" solmix="3"
                    solref="0.04 2" solimp="0.9 0.95 0.02 0.5 2"/>
              <body pos="0 0 0.1"><joint name="j2"/><geom type="sphere" size="0.01" contype="0" conaffinity="0"/></body>
            </body>
          </worldbody><actuator><position joint="j1" kp="20"/><position joint="j2" kp="20"/></actuator>
        </mujoco>''')
        roots = source.free_roots(model)
        capability.require(model, [0], roots)
        self.assertEqual(model.npair, 0)
        self.assertEqual(step.contact_pairs(model), ((0, 2),))
        self.assertEqual(step.max_support(model), 12)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertEqual(data.ncon, 1)
        parameters = step.contact_parameters(model)
        np.testing.assert_allclose(parameters["friction"][0], data.contact[0].friction)
        np.testing.assert_allclose(parameters["solref"][0], data.contact[0].solref)
        np.testing.assert_allclose(parameters["solimp"][0], data.contact[0].solimp)
        jac = data.efc_J.reshape(-1, model.nv)
        hessian = jac.T @ (data.efc_D[:, None] * jac)
        split = int(model.body_dofadr[roots[1]])
        self.assertGreater(np.linalg.norm(hessian[:split, split:]), 0)
        packed, _, _ = adjoint.engine_constants(model, [0], roots)
        self.assertTrue(np.all(np.isfinite(packed)))
        self.assertIn("#define NPAIR 1", adjoint.cuda_source(model, com_body_id=roots))

    def test_two_roots_full_state_and_64bit_masks(self):
        for hinges, fingers, passive in ((22, 5, False), (12, 5, True)):
            model = hand_model(hinges, fingers, roots=2, passive=passive, equality=passive, damping=0.1)
            roots = source.free_roots(model)
            capability.require(model, [], roots)
            packed, layouts, sizes = adjoint.engine_constants(model, [], com_body_id=roots)
            offset, arrays = 0, {}
            for name, kind, unit, width in layouts:
                count = (sizes[unit] if isinstance(unit, str) else unit) * width * (4 if kind == "q" else 2 if kind == "u" else 1)
                arrays[name], offset = packed[offset:offset + count], offset + count
            self.assertEqual(offset, len(packed))
            np.testing.assert_array_equal(arrays["c_roots"], roots)
            np.testing.assert_array_equal(arrays["c_root"], model.body_rootid)
            np.testing.assert_array_equal(arrays["c_combody"], roots)
            for joint in range(model.njnt):
                if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
                    self.assertEqual(arrays["c_dofqadr"][model.jnt_dofadr[joint]], model.jnt_qposadr[joint])
            chunks = arrays["c_ancmask"].astype(np.uint64).reshape(-1, 4)
            masks = sum(chunks[:, k] << np.uint64(16 * k) for k in range(4))
            for body in range(1, model.nbody):
                self.assertTrue((masks[body] >> np.uint64(body)) & np.uint64(1))
                self.assertTrue((masks[body] >> np.uint64(model.body_rootid[body])) & np.uint64(1))
            code = adjoint.cuda_source(model, com_body_id=roots)
            self.assertIn("#define NR 2", code)
            self.assertEqual(model.nq, model.nv + 2)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            mass = np.empty((model.nv, model.nv))
            mujoco.mj_fullM(model, data, mass)
            split = int(model.body_dofadr[roots[1]])
            np.testing.assert_array_equal(mass[:split, split:], 0)

    def test_soft_equalities_and_frictionloss(self):
        model = hand_model(passive=True, damping=0.1, frictionloss=0.05, equality=True)
        capability.require(model, [], 1)
        self.assertEqual(model.neq, 6)
        packed, layouts, sizes = adjoint.engine_constants(model, [], com_body_id=1)
        offset, arrays = 0, {}
        for name, kind, unit, width in layouts:
            count = (sizes[unit] if isinstance(unit, str) else unit) * width * (4 if kind == "q" else 2 if kind == "u" else 1)
            arrays[name], offset = packed[offset:offset + count], offset + count
        data = mujoco.MjData(model)
        data.qpos[7:] = np.linspace(0.1, 0.4, model.nv - 6)
        data.qvel[6:] = np.linspace(-0.2, 0.3, model.nv - 6)
        mujoco.mj_forward(model, data)
        d1, d2 = arrays["c_eqdof1"].astype(int), arrays["c_eqdof2"].astype(int)
        coef = arrays["c_eqpoly"].reshape(-1, 2)
        expected_pos = data.qpos[d1 + 1] - coef[:, 0] - coef[:, 1] * data.qpos[d2 + 1]
        expected_vel = data.qvel[d1] - coef[:, 1] * data.qvel[d2]
        np.testing.assert_allclose(expected_pos, data.efc_pos[:model.neq], atol=2e-8)
        np.testing.assert_allclose(expected_vel, data.efc_vel[:model.neq], atol=2e-8)
        imp = model.eq_solimp[:, 1]  # the chosen states are outside the impedance transition
        timeconst = np.maximum(model.eq_solref[:, 0], 2 * model.opt.timestep)
        stiffness = 1 / (imp * timeconst * model.eq_solref[:, 1]) ** 2
        damping = 2 / (imp * timeconst)
        np.testing.assert_allclose(-stiffness * imp * data.efc_pos[:model.neq]
                                   - damping * data.efc_vel[:model.neq], data.efc_aref[:model.neq], rtol=1e-12)
        expected_D = 1 / (arrays["c_eqinvw"] * (1 - imp) / imp)
        np.testing.assert_allclose(expected_D, data.efc_D[:model.neq], rtol=2e-7)
        jac = np.zeros((model.neq, model.nv))
        jac[np.arange(model.neq), d1] = 1
        jac[np.arange(model.neq), d2] = -coef[:, 1]
        np.testing.assert_allclose(jac, data.efc_J.reshape(-1, model.nv)[:model.neq], atol=1e-7)
        code = adjoint.cuda_source(model, com_body_id=1)
        self.assertIn("#define NEQ 6", code)
        self.assertIn("#define NFLOSS 12", code)

    def test_unsupported_constraint_flags_and_polynomials(self):
        model = hand_model(passive=True, damping=0.1, equality=True)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        capability.require(model, [], 1)
        self.assertIn("#define CONTACT_DISABLED 1", step.step_header(model))
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EQUALITY)
        self.assertIn("H24b", capability.check(model, [], 1).codes("error"))
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_EQUALITY)
        model.eq_data[0, 2] = 0.1
        self.assertIn("H25b", capability.check(model, [], 1).codes("error"))

    def test_passive_actuators_and_damping(self):
        model = hand_model(passive=True, damping=0.1)
        capability.require(model, [], 1)
        self.assertEqual(model.nu, 6)
        packed, layouts, sizes = adjoint.engine_constants(model, [], com_body_id=1)
        offset, arrays = 0, {}
        for name, kind, unit, width in layouts:
            count = (sizes[unit] if isinstance(unit, str) else unit) * width * (4 if kind == "q" else 2 if kind == "u" else 1)
            arrays[name], offset = packed[offset:offset + count], offset + count
        driven = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        np.testing.assert_array_equal(arrays["c_actid"][driven], np.arange(model.nu))
        np.testing.assert_array_equal(arrays["c_actid"][[i for i in range(model.nv) if i not in driven]], -1)
        np.testing.assert_array_equal(arrays["c_damping"], model.dof_damping.astype(np.float32))
        self.assertIn("#define NU 6\n", adjoint.cuda_source(model, com_body_id=1))

    def test_no_contact_support(self):
        model = hand_model()
        self.assertEqual(model.npair, 0)
        self.assertEqual(step.max_support(model), 0)
        header = step.step_header(model)
        self.assertIn("#define NPAIR 0\n", header)
        self.assertIn("#define KDOF 0\n", header)
        self.assertNotIn("[0]", header)
        self.assertTrue(all(n > 0 for _, _, n in step.const_layout(model)))

    def test_low_nv_children_and_com(self):
        for nv, fingers in ((13, 3), (18, 5), (22, 4), (28, 5)):
            with self.subTest(nv=nv, fingers=fingers):
                model = hand_model(nv - 6, fingers)
                capability.require(model, [], 1)
                packed, layouts, sizes = adjoint.engine_constants(model, [], com_body_id=1)
                counts = [(sizes[n] if isinstance(n, str) else n) * w for _, _, n, w in layouts]
                self.assertEqual(len(packed), sum(4 * n if k == "q" else 2 * n if k == "u" else n
                                                  for (_, k, _, _), n in zip(layouts, counts)))
                offset = 0
                arrays = {}
                for (name, kind, _, _), count in zip(layouts, counts):
                    width = 4 * count if kind == "q" else 2 * count if kind == "u" else count
                    arrays[name] = packed[offset:offset + width]
                    offset += width
                np.testing.assert_equal(arrays["c_misc"][1], 1)
                self.assertEqual(arrays["c_child"].size, model.nbody * fingers)
                code = adjoint.cuda_source(model, com_body_id=1)
                self.assertIn(f"#define NV {nv}\n", code)
                self.assertIn(f"#define MAX_CHILD {fingers}\n", code)


if __name__ == "__main__":
    unittest.main()
