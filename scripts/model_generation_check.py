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


def hand_model(nhinge=12, fingers=5, *, passive=False, damping=0.0, frictionloss=0.0, equality=False, roots=1,
               collision=False):
    """One or two real FREE palms and hinge chains, optionally with dense contacts."""
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
                      + ('<geom type="sphere" size=".1"/>' * 12 if collision else '')
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


def dense_contact_model(nconmax=320, njmax=1400, *, frictionloss=0.0, equality=False):
    geoms = '<geom type="sphere" size=".1"/>' * 17
    bodies = []
    for index, x in enumerate((0, .15)):
        bodies.append(f'<body pos="{x} 0 0"><freejoint/>' + geoms
                      + f'<body pos="0 0 .1"><joint name="j{index}" range="-1 1" frictionloss="{frictionloss}"/>'
                      '<inertial mass=".1" pos=".02 0 0" diaginertia=".001 .001 .001"/></body></body>')
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><size nconmax="{nconmax}" njmax="{njmax}"/>'
        '<option timestep=".005" integrator="implicitfast" solver="Newton" cone="pyramidal"/>'
        '<worldbody>' + ''.join(bodies) + '</worldbody><actuator>'
        '<position joint="j0" kp="20" kv=".2"/><position joint="j1" kp="20" kv=".2"/></actuator>'
        + ('<equality><joint joint1="j0" joint2="j1" polycoef="0 1 0 0 0"/></equality>' if equality else '')
        + '</mujoco>')


class GenerationCheck(unittest.TestCase):
    @unittest.skipUnless(shutil.which("c++"), "capsule geometry check requires a C++ compiler")
    def test_parallel_capsules_keep_the_two_contact_branch(self):
        # Execute the actual collision/replay helper with IEEE float operations.
        # At hand-proxy sizes, the subtractive Gram determinant contains enough
        # rounding noise to select one contact for perfectly parallel capsules.
        rounded = "#define RMUL" + source.DEVICE.split("#define RMUL", 1)[1].split(
            "__device__ __forceinline__ float warp_sum", 1)[0]
        capsule = "struct CapCap" + step.STEP_DEVICE.split("struct CapCap", 1)[1].split(
            "// the segment parameters", 1)[0]
        harness = r'''
#include <cmath>
#include <cstdio>
#define __device__
#define __forceinline__ inline
inline float __fmul_rn(float a, float b) { return a*b; }
inline float __fadd_rn(float a, float b) { return a+b; }
inline float __fsub_rn(float a, float b) { return a-b; }
inline float __fmaf_rn(float a, float b, float c) { return std::fma(a,b,c); }
'''
        main = r'''
int main() {
  float p1[3]={}, p2[3]={.01f,0,.005f}, q[4]={1,0,0,0}, opposite[4]={0,1,0,0};
  CapCap cc;
  capcap_setup(p1,q,.03f,p2,q,.04f,cc);
  float old = RFMA(cc.ma,cc.mc,-RMUL(cc.mb,cc.mb));
  std::printf("%.10e %.10e ",cc.det,old);
  capcap_setup(p1,q,.03f,p2,opposite,.04f,cc);
  std::printf("%.10e\n",cc.det);
}
'''
        with tempfile.TemporaryDirectory(prefix="nabla-capsule-cpu-") as directory:
            path = Path(directory)
            (path / "capsule.cc").write_text(harness + rounded + capsule + main)
            compiled = subprocess.run(["c++", "-std=c++17", "-O2", "-ffp-contract=off",
                                       str(path / "capsule.cc"), "-o", str(path / "capsule")],
                                      text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(path / "capsule")], check=True, text=True, capture_output=True)
        parallel, cancellation, antiparallel = map(float, result.stdout.split())
        self.assertGreaterEqual(abs(cancellation), 1e-15, "the regression must exercise the old wrong branch")
        self.assertLess(abs(parallel), 1e-15)
        self.assertLess(abs(antiparallel), 1e-15)

    def test_upstream_020_terrain_and_coupled_build_plans(self):
        from coupled_physics_check import native_contact_model
        from nablaengine import fixtures, jax_ffi
        from nablaengine.terrain import TerrainConfig

        # Native hand layouts must coexist with 0.2's generator routing and
        # terrain hooks; no CUDA device is needed to detect a stale text anchor.
        model, meta = fixtures.load_model()
        for mode in ("analytic", "native"):
            with self.subTest(terrain=mode):
                plan = jax_ffi.build_plan(model, meta["foot_pairs"], "sm_90", TerrainConfig(mode=mode))
                self.assertEqual(len(plan["units"]), 13)
                self.assertTrue(np.isfinite(plan["consts"]).all())
        model = hand_model(12, 5, passive=True, damping=.1, frictionloss=.02, equality=True, roots=2)
        plan = jax_ffi.build_plan(model, [], "sm_90", com_body_id=source.free_roots(model))
        self.assertEqual(len(plan["units"]), 13)
        self.assertTrue(np.isfinite(plan["consts"]).all())
        model = native_contact_model(mujoco, "box_box")
        plan = jax_ffi.build_plan(model, [], "sm_90", com_body_id=source.free_roots(model))
        self.assertEqual(len(plan["units"]), 13)

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
        sim = step.STEP_DEVICE.split("struct Sim {", 1)[1].split("// a condition that", 1)[0]
        split = step.STEP_KERNEL.split("// The split forward (host.py:", 1)[1]
        g1 = mujoco.MjModel.from_binary_path(str(Path(__file__).resolve().parents[1] / "fixtures/mjlab/g1_train.mjb"))
        dense = dense_contact_model(frictionloss=.1, equality=True)
        large = hand_model(26, roots=2, frictionloss=.1, equality=True, collision=True)
        large.dof_frictionloss[:] = .1  # FREE dofs may have frictionloss too, so NFLOSS can equal NV.
        capability.require(large, [], source.free_roots(large))
        cases = (
            ("one root", g1, (), 32, 128),
            ("two roots limited", hand_model(22, roots=2), (), 32, 44),
            ("two roots full", dense, (), 96, 401),
            ("two roots capped", dense_contact_model(64, 64), (), 64, 258),
            ("contact override", dense, ("-DSPLIT_NCON=16",), 16, 81),
            ("row override", dense, ("-DSPLIT_NEFC=79",), 96, 79),
            ("large two roots", large, (), 96, 524),
            ("large tape verification", large, ("-DDK_TAPE_VERIFY",), 32, 128),
        )
        for name, model, defines, ncon, nefc in cases:
            with self.subTest(name=name):
                header = source.header(model, maxd=8, wpb=1).replace("#include <cuda_runtime.h>", "")
                code = ("#define __device__\n" + header + step.step_header(model) + "struct Sim {" + sim
                        + "// The split forward (host.py:" + split
                        + f"\nstatic_assert(SPLIT_NCON == {ncon} && SPLIT_NEFC == {nefc});\n"
                        + f"static_assert(sizeof(((SimN*)0)->con_dim) == {ncon});\n"
                        + f"static_assert(sizeof(((SimN*)0)->efc_kind) == {nefc});\n"
                        + "static_assert(sizeof(SimN) <= sizeof(SimNfull) && sizeof(SimN) <= 48 * 1024);\n"
                        + "static_assert(sizeof(Hand) > 0);\n")
                result = subprocess.run(["c++", "-std=c++17", "-fsyntax-only", *defines, "-x", "c++", "-"],
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

    @unittest.skipUnless(shutil.which("c++"), "adjoint scaling check requires a C++ compiler")
    def test_adjoint_preconditions_large_cotangents_without_hiding_output_overflow(self):
        # Execute the actual native entry/return code with one active lane. Its
        # local linear VJP has a representable result but an overflowing float
        # intermediate at the magnitude of the first captured H64 failure.
        kernel = adjoint.ADJ_KERNEL.split("#ifndef DK_ADJ_WORLDS_PER_SM", 1)[0]
        kernel = kernel.replace("extern __shared__ __align__(16) unsigned char dk_smem[];",
                                "alignas(16) static unsigned char dk_smem[sizeof(Adj)];")
        harness = r'''
#include <cmath>
#include <cstddef>
#include <limits>
using std::isfinite;
#define __global__
#define NQ 1
#define NV 1
#define NU 1
#define COM_WIDTH 1
#define REC_WORDS 1
#define DK_ADJ_CLAMP_KV 0
#define DK_ADJ_DEBUG
#define ADJ_SOFT_PARAM
#define APROF_START()
#define APROF(x)
#define DBG_W_DIR 2
#define DBG_ADJ_QPOS_M 3
#define DBG_RESID 7
#define DBG_CON_DIST_BAR 8
#define DBG_CON_X 10
#define DBG_WORDS 12
struct { int x=0; } threadIdx, blockIdx;
inline void __syncwarp() {}
inline float __shfl_xor_sync(unsigned, float, int) { return 0.f; }
struct Adj {
  float qbar[1],vbar[1],ubar[1],qfrcbar[1],com[1],fric,kp_scale;
  const float* applied;
  int flags;
};
int mode=0;
void adj_init(Adj& s,int) { s.flags=0; s.qfrcbar[0]=0; }
void adj_substep(Adj& s,const float*,int,float* dbg,const float*) {
  // Both modes are linear. The first has an avoidable intermediate
  // overflow; the second's true result exceeds the output representation.
  volatile float intermediate=s.qbar[0]*32.f;
  s.qbar[0]=mode ? intermediate : intermediate/32.f;
  s.vbar[0]*=2.f;
  s.ubar[0]+=s.qbar[0]*.125f;
  s.qfrcbar[0]+=s.vbar[0]*.25f;
  if(dbg) {
    dbg[0]=s.qbar[0]; dbg[1]=s.vbar[0];
    dbg[3]=s.qbar[0]; dbg[4]=s.vbar[0]; dbg[5]=s.ubar[0]; dbg[6]=s.qfrcbar[0];
    dbg[8]=s.qbar[0]*.25f; dbg[9]=s.vbar[0]*.5f;
    dbg[2]=7.f; dbg[7]=11.f; dbg[10]=13.f;
  }
}
'''
        main = r'''
bool check(float q,float v) {
  float record=0,friction=0,com=0,force=0,kp=1,gq,gv,gu,gf,dbg[DBG_WORDS]={}; int info;
  k_adjoint(1,1,&record,&friction,&com,&q,&v,&gq,&gv,&gu,&info,dbg,nullptr,&force,&kp,&gf);
  return gq==q && gv==2.f*v && gu==q*.125f && gf==v*.5f && info==0
    // Lane zero restores the first entry of each cotangent debug range;
    // the other lanes own the remaining entries in a real CUDA warp.
    && dbg[0]==q && dbg[3]==q && dbg[8]==q*.25f
    && dbg[2]==7.f && dbg[7]==11.f && dbg[10]==13.f;
}
int main() {
  const float q=2.5991347605514315e37f,v=6.176665182989242e34f;
  volatile float large=q, intermediate=large*32.f;
  if(!std::isinf(intermediate/32.f)) return 1;  // negative control
  if(!check(q,v) || !check(.75f,-.5f) || !check(2.75f,-.125f) || !check(0.f,0.f)) return 2;
  // frexpf(FLT_MAX) returns exponent128, whose inverse scale cannot be
  // materialized as a finite float. scalbnf must still restore this output.
  if(!check(std::numeric_limits<float>::max(),1.f)) return 3;
  mode=1;
  float record=0,friction=0,com=0,gq,gv,gu,gf; int info;
  k_adjoint(1,1,&record,&friction,&com,&q,&v,&gq,&gv,&gu,&info,nullptr,nullptr,nullptr,nullptr,&gf);
  return std::isinf(gq) ? 0 : 4;
}
'''
        with tempfile.TemporaryDirectory(prefix="nabla-adjoint-scale-cpu-") as directory:
            path = Path(directory)
            (path / "adjoint.cc").write_text(harness + kernel + main)
            compiled = subprocess.run(["c++", "-std=c++17", "-O2", str(path / "adjoint.cc"),
                                       "-o", str(path / "adjoint")], text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(path / "adjoint")], text=True, capture_output=True)
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
