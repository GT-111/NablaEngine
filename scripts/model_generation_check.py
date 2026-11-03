"""CPU regression checks for model-specific CUDA source and packed constants.

PYTHONPATH=src python scripts/model_generation_check.py
These checks generate source; numerical CUDA qualification is a separate check.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import mujoco
import numpy as np

from nablaengine.g1_engine import adjoint, source, step, tape
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


def dense_contact_model(nconmax=320, njmax=1400):
    geoms = '<geom type="sphere" size=".1"/>' * 17
    bodies = []
    for index, x in enumerate((0, .15)):
        bodies.append(f'<body pos="{x} 0 0"><freejoint/>' + geoms
                      + f'<body pos="0 0 .1"><joint name="j{index}" range="-1 1"/>'
                      '<inertial mass=".1" pos=".02 0 0" diaginertia=".001 .001 .001"/></body></body>')
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><size nconmax="{nconmax}" njmax="{njmax}"/>'
        '<option timestep=".005" integrator="implicitfast" solver="Newton" cone="pyramidal"/>'
        '<worldbody>' + ''.join(bodies) + '</worldbody><actuator>'
        '<position joint="j0" kp="20" kv=".2"/><position joint="j1" kp="20" kv=".2"/></actuator></mujoco>')


class GenerationCheck(unittest.TestCase):
    def test_authored_contact_budget_retains_large_native_manifold_and_row_counts(self):
        # Distinct geoms on two bodies produce 289 genuine cross-body contacts.
        # This exercises point counts above 255 and row/zone words above 1024.
        model = dense_contact_model()
        data = mujoco.MjData(model)
        mujoco.mj_fwdPosition(model, data)
        self.assertEqual((data.ncon, data.nefc), (289, 1156))
        capability.require(model, [], source.free_roots(model))
        self.assertGreaterEqual(step.maxcon(model), data.ncon)
        self.assertEqual(step.nefc(model), 1400)
        record = np.zeros(tape.rec_words(model), np.float32)
        words, layout = record.view(np.uint32), tape.layout(model)
        words[layout["hdr"][0]] = data.ncon | data.nefc << 16
        words[layout["counts"][0]] = 300 << 16
        last = data.nefc - 1
        words[layout["zone"][0] + (last >> 5)] = 1 << (last & 31)
        decoded = tape.decode(record, model)
        self.assertEqual((decoded["ncon"], decoded["nefc"], decoded["nlim"], decoded["niter"]), (289, 1156, 0, 300))
        self.assertEqual(np.flatnonzero(decoded["zone"]).tolist(), [last])
        # A smaller explicit point budget must remain visible as an overflow
        # boundary, while its allocated rows still cover every retained point.
        model = dense_contact_model(64, 64)
        self.assertEqual(step.maxcon(model), 64)
        self.assertEqual(step.nefc(model), 258)

    def test_reference_com_invalidates_compiled_sameframe_shortcut(self):
        from coupled_physics_check import set_reference_com

        model = hand_model(7, 3, roots=2)
        roots = source.free_roots(model)
        before = mujoco.MjData(model)
        mujoco.mj_forward(model, before)
        offsets = np.asarray((.001, -.002, .003, -.004, .005, -.006))
        set_reference_com(model, model.body_ipos.copy(), model.body_sameframe.copy(), roots, offsets)
        after = mujoco.MjData(model)
        mujoco.mj_forward(model, after)
        for r, body in enumerate(roots):
            offset = offsets[3*r:3*r+3]
            np.testing.assert_allclose(after.xipos[body], after.xpos[body] + offset, atol=1e-15)
            dof = int(model.body_dofadr[body])
            expected = model.body_mass[body] * np.cross(offset, -model.opt.gravity)
            np.testing.assert_allclose(after.qfrc_bias[dof+3:dof+6] - before.qfrc_bias[dof+3:dof+6],
                                       expected, atol=1e-14)

    def test_reference_com_preserves_authored_geom_and_site_poses(self):
        from coupled_physics_check import native_contact_model, set_reference_com

        models = [native_contact_model(mujoco, "mesh"), mujoco.MjModel.from_xml_string('''<mujoco>
          <worldbody><body><freejoint/>
            <inertial pos="0 0 0" quat="0.7071067811865476 0 0 0.7071067811865476"
                      mass="1" diaginertia=".03 .02 .01"/>
            <geom type="box" size=".1 .12 .08" quat="0.7071067811865476 0 0 0.7071067811865476"/>
            <site quat="0.7071067811865476 0 0 0.7071067811865476"/>
          </body></worldbody></mujoco>''')]
        for model in models:
            roots = source.free_roots(model)
            before = mujoco.MjData(model)
            mujoco.mj_forward(model, before)
            self.assertIn(int(mujoco.mjtSameFrame.mjSAMEFRAME_INERTIA), model.geom_sameframe)
            if model.nsite:
                self.assertIn(int(mujoco.mjtSameFrame.mjSAMEFRAME_INERTIA), model.site_sameframe)
            offsets = np.tile((.001, -.002, .003), len(roots))
            set_reference_com(model, model.body_ipos.copy(), model.body_sameframe.copy(), roots, offsets)
            after = mujoco.MjData(model)
            mujoco.mj_forward(model, after)
            for field in ("geom_xpos", "geom_xmat", "site_xpos", "site_xmat"):
                np.testing.assert_allclose(getattr(after, field), getattr(before, field), atol=1e-15)

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

    @unittest.skipUnless(shutil.which("c++"), "Hessian accumulation check requires a C++ compiler")
    def test_hessian_retains_small_contact_curvature_across_tiles(self):
        model = dense_contact_model()
        header = source.header(model, maxd=2, wpb=1).replace("#include <cuda_runtime.h>", "")
        sim = step.STEP_DEVICE.split("struct Sim {", 1)[1].split("// a condition that", 1)[0]
        hessian = step.STEP_KERNEL.split("template <class S>\n__device__ __forceinline__ void hessian", 1)[1]
        hessian = "template <class S>\nvoid hessian" + hessian.split("// The two-root scene", 1)[0]
        # All contacts have the same scalar curvature. This executes the real
        # Hessian entry owned by lane zero; uniform ballots stand in for its
        # warp, and the other lanes' identical con_C values are prefilled.
        harness = r'''
#include <cmath>
#include <cstdio>
#define __device__
#define SUBPROF(x)
inline void __syncwarp() {}
inline unsigned __ballot_sync(unsigned, bool on) { return on ? ~0u : 0u; }
template<class T> T __shfl_sync(unsigned, T value, int) { return value; }
inline int __ffs(unsigned value) { return __builtin_ffs(value); }
inline int uniform_int(int value, int) { return value; }
template<class T> T min(T a, T b) { return a < b ? a : b; }
template<class S> void row_eval(S&, int, float, float* ds, float* dd, float* cost) {
  *ds=*dd=*cost=0;
}
'''
        main = r'''
// Negative control: per-contact float storage loses these valid contributions.
struct FloatAccumulation : Sim { float H[NTRI]; };
template<class S> float assembled_entry(S& s) {
  s.M[0]=100000000.f;
  for (int c=0;c<288;c++) {
    s.con_nd[c]=KDOF; s.con_dim[c]=1; s.con_row[c]=c; s.con_nrow[c]=1;
    s.con_C[c][0]=1; s.efc_jar[c]=-1; s.efc_D[c]=1;
    s.con_J[c][0][0]=1;
    for (int k=0;k<KDOF;k++) s.con_dof[c][k]=k;
  }
  unsigned world[CON_TILES]={}, group[CON_TILES]={};
  hessian(s,288,world,group,0);
  return s.L[0];
}
int main() {
  static Sim real{}; static FloatAccumulation rounded{};
  std::printf("%.0f %.0f\n", assembled_entry(real), assembled_entry(rounded));
}
'''
        code = harness + header + step.step_header(model) + "struct Sim {" + sim + hessian + main
        with tempfile.TemporaryDirectory(prefix="nabla-hessian-cpu-") as directory:
            path = Path(directory)
            (path / "hessian.cc").write_text(code)
            compiled = subprocess.run(["c++", "-std=c++17", "-O2", str(path / "hessian.cc"),
                                       "-o", str(path / "hessian")], text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(path / "hessian")], check=True, text=True, capture_output=True)
        self.assertEqual([float(value) for value in result.stdout.split()], [100000288., 100000000.])

    def test_native_mesh_feature_tape_and_pair_mapping(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <option integrator="implicitfast" solver="Newton" cone="pyramidal"/>
          <asset><mesh name="hull" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/></asset>
          <worldbody>
            <body name="left"><freejoint/><geom type="mesh" mesh="hull"/></body>
            <body name="right" pos="0.05 0 0"><freejoint/><geom type="mesh" mesh="hull"/></body>
          </worldbody></mujoco>''')
        self.assertEqual(model.npair, 0)
        self.assertEqual(step.contact_pairs(model), ((0, 1),))
        roots = source.free_roots(model)
        capability.require(model, [0], roots)
        self.assertEqual(tape.layout(model)["feature"][1], 12 * step.maxcon(model))
        packed, layout, sizes = adjoint.engine_constants(model, [0], roots)
        expected = sum((sizes[count] if isinstance(count, str) else count) * width
                       * (4 if kind == "q" else 2 if kind == "u" else 1)
                       for _, kind, count, width in layout)
        self.assertEqual(len(packed), expected)
        self.assertIn("cm_vert", {row[0] for row in layout})
        self.assertIn("#define HAS_NATIVE_COLLISION 1", adjoint.cuda_source(model, com_body_id=roots))
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        self.assertIn("H24b", capability.check(model, [0], roots).codes("error"))
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        model.geom_type[1] = mujoco.mjtGeom.mjGEOM_SPHERE
        self.assertIn("H14c", capability.check(model, [0], roots).codes("error"))

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
        single = hand_model(31, 5)
        self.assertIn("H1", capability.check(single, [], 1, scope="forward").codes("error"))

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
