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

    @unittest.skipUnless(shutil.which("c++"), "root-axis reuse check requires a C++ compiler")
    def test_contact_root_axes_reuse_exact_rotations_for_one_and_two_free_trees(self):
        rotation = source.DEVICE.split("__device__ __forceinline__ void qrot(", 1)[1]
        rotation = "void qrot(" + rotation.split("__device__ __forceinline__ void qrot_inv", 1)[0]
        axes = adjoint.ADJ_DEVICE.split("__device__ __forceinline__ void adj_contact_root_axes", 1)[1]
        axes = "void adj_contact_root_axes" + axes.split("// A5 / A6 of contact c", 1)[0]
        harness = r'''
#include <cmath>
#include <cstring>
#define __device__
#define __forceinline__ inline
#define NB 35
int c_roots[NR]={1
#if NR == 2
  ,18
#endif
};
struct Adj {
  float Q[NB][4];
  unsigned before;
  float zr[3 * NR][3];
  unsigned after;
};
'''
        main = r'''
int main() {
  Adj s={}; s.before=0x12345678u; s.after=0x87654321u;
  for(int pass=0;pass<50;pass++) {
    for(int r=0;r<NR;r++) {
      float norm=0.f;
      for(int k=0;k<4;k++) {
        float value=std::sin(float(1+pass+7*r+3*k));
        s.Q[c_roots[r]][k]=value; norm+=value*value;
      }
      for(int k=0;k<4;k++)s.Q[c_roots[r]][k]/=std::sqrt(norm);
    }
    for(int lane=0;lane<32;lane++)adj_contact_root_axes(s,lane);
    for(int r=0;r<NR;r++)for(int axis=0;axis<3;axis++) {
      float e[3]={0.f,0.f,0.f},direct[3],cached[3];e[axis]=1.f;
      qrot(s.Q[c_roots[r]],e,direct);
      // Repeated contact slots read the same world axis without recomputing.
      for(int contact=0;contact<127;contact++) {
        adj_contact_root_axis(s,c_roots[r],axis,cached);
        if(std::memcmp(direct,cached,sizeof(direct)))return 1;
      }
    }
    if(s.before!=0x12345678u || s.after!=0x87654321u)return 2;
  }
}
'''
        with tempfile.TemporaryDirectory(prefix="nabla-root-axes-cpu-") as directory:
            path = Path(directory)
            (path / "axes.cc").write_text(harness + rotation + axes + main)
            for roots in (1, 2):
                compiled = subprocess.run(["c++", "-std=c++17", "-O2", "-ffp-contract=off", f"-DNR={roots}",
                    str(path / "axes.cc"), "-o", str(path / "axes")], text=True, capture_output=True)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                result = subprocess.run([str(path / "axes")], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("c++"), "equality Hessian check requires a C++ compiler")
    def test_equality_hessian_matches_dense_row_order(self):
        coef = step.STEP_DEVICE.split("template <class S>\n__device__ __forceinline__ float dof_row_coef", 1)[1]
        coef = "template <class S>\n__device__ __forceinline__ float dof_row_coef" + coef.split("// J v for every row", 1)[0]
        pieces = step.STEP_DEVICE.split("__device__ __forceinline__ int row_piece", 1)[1]
        pieces = "__device__ __forceinline__ int row_piece" + pieces.split("// J^T f into out", 1)[0]
        dof_hessian = step.STEP_KERNEL.split("const int ndof = uniform_int(s.nefc_dof, 8);\n#if NEQ > 0", 1)[1]
        dof_hessian = dof_hessian.split("\n#else\n#if DK_HESS_DOFPAR", 1)[0]
        main = r'''
struct State {
  double H[NTRI]; int nefc_dof;
  unsigned char efc_kind[2*NV+NEQ]; unsigned short efc_a[2*NV+NEQ]; signed char efc_b[2*NV+NEQ];
  float efc_D[2*NV+NEQ], efc_jar[2*NV+NEQ];
};
void add_dof_hessian(State& s, int lane) { int ndof=s.nefc_dof;
''' + dof_hessian + r'''
}
float reference_coef(const State& s, int r, int i) {
  int a=s.efc_a[r];
  if(s.efc_kind[r]==4)
    return (i==c_eqdof1[a] ? 1.f : 0.f) - (i==c_eqdof2[a] ? c_eqpoly[2*a+1] : 0.f);
  return i==a ? (s.efc_kind[r]==0 ? 1.f : (float)s.efc_b[r]) : 0.f;
}
int main() {
  for(int pass=0;pass<8;pass++) {
    State s={}; int n=0;
    for(int i=0;i<NV;i++)if(c_floss[i]>0.f) {
      s.efc_kind[n]=0;s.efc_a[n]=i;s.efc_b[n++]=1;
    }
    if(n!=NFLOSS)return 1;
    for(int e=0;e<NEQ;e++) { s.efc_kind[n]=4;s.efc_a[n]=e;s.efc_b[n++]=1; }
    for(int i=0;i<NV;i++)if(c_hinge[i] && (i+pass)%3!=0) {
      s.efc_kind[n]=1;s.efc_a[n]=i;s.efc_b[n++]=i%2 ? 1 : -1;
    }
    s.nefc_dof=n;
    for(int r=0;r<n;r++) {
      s.efc_D[r]=(1+(r*7)%23)/31.f;
      s.efc_jar[r]=((r+pass)%5-2)*1.25f;
    }
    double expected[NTRI];
    for(int e=0;e<NTRI;e++)s.H[e]=expected[e]=std::sin((e+3)*.17)*1e5;
    for(int i=0;i<NV;i++)for(int j=0;j<=i;j++) {
      double value=expected[TRI(i,j)];
      for(int r=0;r<n;r++) {
        float ci=reference_coef(s,r,i),cj=reference_coef(s,r,j);
        if(ci==0.f || cj==0.f)continue;
        float D=s.efc_D[r],x=s.efc_jar[r],dd;
        if(r<NFLOSS) {
          float bound=(1.f/D)*c_floss[s.efc_a[r]];
          dd=(x<=-bound || x>=bound) ? 0.f : D;
        } else dd=r<NFLOSS+NEQ || x<0.f ? D : 0.f;
        value+=(double)ci*cj*dd;
      }
      expected[TRI(i,j)]=value;
    }
    for(int lane=0;lane<32;lane++)add_dof_hessian(s,lane);
    for(int e=0;e<NTRI;e++)if(std::memcmp(&expected[e],&s.H[e],sizeof(double))) {
      std::fprintf(stderr,"pass=%d entry=%d expected=%.17g actual=%.17g\n",pass,e,expected[e],s.H[e]);return 2;
    }
  }
}
'''
        models = (hand_model(12, roots=1, frictionloss=.05, equality=True),
                  hand_model(12, roots=2, equality=True),
                  hand_model(16, roots=2, equality=True),
                  hand_model(16, roots=2, frictionloss=.05, equality=True),
                  hand_model(22, roots=2, frictionloss=.05, equality=True))
        for model in models:
            # A supported zero slope and a reversed, overlapping equality exercise all three entries and row order.
            model.eq_data[0, 1] = 0
            model.eq_obj1id[1], model.eq_obj2id[1] = model.eq_obj2id[0], model.eq_obj1id[0]
            capability.require(model, [], source.free_roots(model))
            eq1, eq2 = model.jnt_dofadr[model.eq_obj1id], model.jnt_dofadr[model.eq_obj2id]
            hinge_dofs = model.jnt_dofadr[model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE]
            def floats(values):
                return ",".join(f"static_cast<float>({float(value):.17g})" for value in values)
            header = f'''
#include <cmath>
#include <cstdio>
#include <cstring>
#define __device__
#define __forceinline__ inline
#define __syncwarp() ((void)0)
#define NV {model.nv}
#define NEQ {model.neq}
#define NFLOSS {int((model.dof_frictionloss > 0).sum())}
#define NTRI (NV*(NV+1)/2)
#define TRI(i,j) ((i)*((i)+1)/2+(j))
using std::min; using std::max;
float c_floss[NV]={{{floats(model.dof_frictionloss)}}};
int c_hinge[NV]={{{','.join('1' if i in hinge_dofs else '0' for i in range(model.nv))}}};
int c_eqdof1[NEQ]={{{','.join(map(str, eq1))}}};
int c_eqdof2[NEQ]={{{','.join(map(str, eq2))}}};
float c_eqpoly[2*NEQ]={{{floats(model.eq_data[:, :2].flat)}}};
'''
            with tempfile.TemporaryDirectory(prefix="nabla-equality-hessian-cpu-") as directory:
                path = Path(directory)
                (path / "hessian.cc").write_text(header + coef + pieces + main)
                for contraction in ("off", "fast"):
                    compiled = subprocess.run(["c++", "-std=c++17", "-O2", f"-ffp-contract={contraction}",
                        str(path / "hessian.cc"), "-o", str(path / "hessian")], text=True, capture_output=True)
                    self.assertEqual(compiled.returncode, 0, compiled.stderr)
                    result = subprocess.run([str(path / "hessian")], text=True, capture_output=True)
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

    @unittest.skipUnless(shutil.which("c++"), "contact support check requires a C++ compiler")
    def test_contact_jacobian_support_matches_dofs_for_two_roots(self):
        model = hand_model(26, roots=2)
        dofbody, ancestors = step.extra_constants(model)
        pairs = [(1 + (7 * c) % (model.nbody - 1), 1 + (13 * c + 27) % (model.nbody - 1)) for c in range(80)]
        jac = step.STEP_KERNEL.split("  {  // contact Jacobians:", 1)[1].split("  PROF(3);", 1)[0]
        support = step.STEP_KERNEL.split("__device__ __forceinline__ unsigned long long support_mask", 1)[1]
        support = "__device__ __forceinline__ unsigned long long support_mask" + support.split("// the Newton Hessian", 1)[0]
        declarations = f'''
#include <cstdlib>
#define __device__
#define __forceinline__ inline
#define NR 2
#define NV {model.nv}
#define NB {model.nbody}
#define CON_TILES 3
#define DOF_BODY_ROOT6 0
#define PG1(p) (2*(p))
#define PG2(p) (2*(p)+1)
int c_dofbody[NV] = {{{','.join(str(int(v)) for v in dofbody)}}};
int c_doflocal[NV] = {{}};
unsigned long long c_ancmask[NB] = {{{','.join(str(int(v)) + 'ull' for v in ancestors)}}};
int c_gbody[160] = {{{','.join(str(v) for pair in pairs for v in pair)}}};
'''
        harness = r'''
int lane_id, shuffles, ballots;
unsigned long long __shfl_sync(unsigned, unsigned long long, int src) {
  int c=shuffles/2, side=shuffles++%2;
  if(src!=(c&31))std::abort();
  return c_ancmask[c_gbody[2*c+side]];
}
unsigned __ballot_sync(unsigned, bool on) {
  int c=ballots/2, h=ballots++%2;
  auto x=c_ancmask[c_gbody[2*c]]^c_ancmask[c_gbody[2*c+1]];
  unsigned result=0;
  for(int j=0;j<32;j++)if(32*h+j<NV && ((x>>c_dofbody[32*h+j])&1ull))result|=1u<<j;
  if(on!=bool((result>>lane_id)&1u))std::abort();
  return result;
}
int __popc(unsigned x) { return __builtin_popcount(x); }
void dof_motion(int,int,const float*,const float*,const float*,float* w,float* v) {
  for(int i=0;i<3;i++)w[i]=v[i]=0.f;
}
void cross(const float*,const float*,float* out) { for(int i=0;i<3;i++)out[i]=0.f; }
float dot3(const float*,const float*) { return 0.f; }
struct State {
  float Q[NB][4],X[NB][3],con_pos[80][3],con_frame[80][9],con_J[80][3][NV];
  int con_pair[80],con_nd[80]; unsigned char con_dof[80][NV]; unsigned long long con_mask[80];
};
'''
        main = r'''
int main() {
  static State s{};
  for(int c=0;c<80;c++)s.con_pair[c]=c;
  for(int lane=0;lane<32;lane++){lane_id=lane;shuffles=ballots=0;run(s,80,lane);}
  for(int c=0;c<80;c++) {
    auto x=c_ancmask[c_gbody[2*c]]^c_ancmask[c_gbody[2*c+1]], expected=0ull;
    int k=0;
    for(int i=0;i<NV;i++)if((x>>c_dofbody[i])&1ull){
      expected|=1ull<<i;
      if(s.con_dof[c][k++]!=i)return 1;
    }
    if(s.con_mask[c]!=expected || s.con_nd[c]!=k)return 2;
  }
}
'''
        code = declarations + harness + support + "\nvoid run(State& s, int ncon, int lane) {\n" \
            + "const float O[3]={};\n{  // contact Jacobians:" + jac + "}\n" + main
        with tempfile.TemporaryDirectory(prefix="nabla-contact-support-cpu-") as directory:
            binary = str(Path(directory) / "support")
            compiled = subprocess.run(["c++", "-std=c++17", "-O2", "-x", "c++", "-", "-o", binary],
                                      input=code, text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([binary], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

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
