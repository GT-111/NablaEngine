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
        sim = step.STEP_DEVICE.split("struct Sim {", 1)[1].split("// a condition that", 1)[0]
        split = step.STEP_KERNEL.split("// The split forward (host.py:", 1)[1]
        row_tiles = next(line for line in step.STEP_KERNEL.splitlines() if "constexpr int RT =" in line)
        instances = ("\ntemplate<class S> constexpr int row_tiles(const S& s) {\n" + row_tiles
                     + "\nreturn RT;\n}\n"
                     "static_assert(row_tiles(Sim{}) == (NEFC + 31) / 32);\n"
                     "static_assert(row_tiles(SimN{}) == (SPLIT_NEFC + 31) / 32);\n"
                     "static_assert(row_tiles(SimNfull{}) == (NEFC + 31) / 32);\n"
                     "static_assert(sizeof(Hand) > 0);\n")
        for model in (hand_model(22, roots=2), dense_contact_model()):
            header = source.header(model, maxd=8, wpb=1).replace("#include <cuda_runtime.h>", "")
            code = ("#define __device__\n" + header + step.step_header(model) + "struct Sim {" + sim
                    + "// The split forward (host.py:" + split + instances)
            result = subprocess.run(["c++", "-std=c++17", "-fsyntax-only", "-x", "c++", "-"],
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

    @unittest.skipUnless(shutil.which("c++"), "contact-motion check requires a C++ compiler")
    def test_contact_support_walk_preserves_dense_motion_sums_with_passive_dofs(self):
        motion = adjoint.ADJ_DEVICE.split("  float Vv[3] =", 1)[1].split("  float Wv[3]", 1)[0]
        motion = "  float Vv[3] =" + motion
        # Keep the arithmetic fixed: the reference visits the same dofs by a dense scan.
        contribution = "float sg =" + motion.split("      float sg =", 1)[1].rsplit("\n    }\n  }", 1)[0]
        dense = motion.split("  // Body and local-dof order", 1)[0] + """
  for (int i = 0; i < NV; i++) {
    int bi = c_dofbody[i], local = c_doflocal[i];
    if (!((sup >> bi) & 1ull)) continue;
""" + contribution + "\n  }\n"
        store = """
  const float values[18] = {Vv[0],Vv[1],Vv[2],Va[0],Va[1],Va[2],Vl[0],Vl[1],Vl[2],
                           Ov[0],Ov[1],Ov[2],Oa[0],Oa[1],Oa[2],Ol[0],Ol[1],Ol[2]};
  std::memcpy(output, values, sizeof(values));
}
"""
        harness = r'''
#include <cmath>
#include <cstdio>
#include <cstring>
struct Adj { float v[NV],a[NV],lam[NV],X[NB][3],z[NB][3],zr[3*NR][3]; };
inline int __ffsll(unsigned long long value) { return __builtin_ffsll(value); }
inline void cross(const float* a,const float* b,float* out) {
  out[0]=a[1]*b[2]-a[2]*b[1];out[1]=a[2]*b[0]-a[0]*b[2];out[2]=a[0]*b[1]-a[1]*b[0];
}
inline void adj_contact_root_axis(const Adj& s,int body,int axis,float* out) {
  for(int k=0;k<3;k++)out[k]=s.zr[(body==c_roots[0]?0:3)+axis][k];
}
'''
        main = r'''
int main() {
  Adj s;
  for(int pass=0;pass<8;pass++) {
    float* state=reinterpret_cast<float*>(&s);
    for(unsigned k=0;k<sizeof(s)/sizeof(float);k++)state[k]=std::sin(float(1+pass+7*k));
    for(int b1=0;b1<NB;b1++)for(int b2=0;b2<NB;b2++) {
      float p[3]={.13f*b1,-.17f*b2,.03f*(b1-b2)}, actual[18],expected[18];
      sparse(s,b1,b2,p,actual);dense(s,b1,b2,p,expected);
      if(std::memcmp(actual,expected,sizeof(actual))) {
        std::fprintf(stderr,"motion mismatch: pass=%d bodies=%d,%d\n",pass,b1,b2);return 1;
      }
    }
  }
}
'''
        with tempfile.TemporaryDirectory(prefix="nabla-contact-support-cpu-") as directory:
            path = Path(directory)
            for roots, hinges in ((1, 58), (2, 26)):
                model = hand_model(hinges, roots=roots, passive=True)
                source.constants(model, source.free_roots(model))
                _, masks = step.extra_constants(model)
                local = np.arange(model.nv) - model.body_dofadr[model.dof_bodyid]
                arrays = {"c_dadr": model.body_dofadr, "c_isfree": model.body_dofnum == 6,
                          "c_dofbody": model.dof_bodyid, "c_doflocal": local, "c_roots": source.free_roots(model)}
                constants = "".join(f'int {name}[]={{' + ",".join(str(int(x)) for x in values) + "};\n"
                                    for name, values in arrays.items())
                constants += "unsigned long long c_ancmask[]={" + ",".join(f"{int(x)}ull" for x in masks) + "};\n"
                code = constants + harness
                for name, body in (("sparse", motion), ("dense", dense)):
                    code += f'void {name}(const Adj& s,int b1,int b2,const float* p,float* output) {{\n' + body + store
                (path / "motion.cc").write_text(code + main)
                compiled = subprocess.run(["c++", "-std=c++17", "-O2", "-ffp-contract=off",
                    f"-DNB={model.nbody}", f"-DNV={model.nv}", f"-DNR={roots}",
                    str(path / "motion.cc"), "-o", str(path / "motion")], text=True, capture_output=True)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                result = subprocess.run([str(path / "motion")], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("c++"), "friction-row preprocessing check requires a C++ compiler")
    def test_adjoint_omits_friction_row_scan_only_when_model_has_none(self):
        limits = adjoint.ADJ_DEVICE.split("void adj_limits(", 1)[1].split("// Root rotation axes", 1)[0]
        for count in (0, 3):
            result = subprocess.run(["c++", "-E", "-P", "-x", "c++", f"-DNFLOSS={count}", "-"],
                                    input="void adj_limits(" + limits, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual("c_floss[i]" in result.stdout, count > 0)

    @unittest.skipUnless(shutil.which("c++"), "contact-group check requires a C++ compiler")
    def test_native_contact_groups_keep_pair_cache_and_per_slot_results(self):
        group = "    bool active = use && (zone || dbg);" + adjoint.ADJ_DEVICE.split(
            "    bool active = use && (zone || dbg);", 1)[1].split("#else", 1)[0]
        flags = "    unsigned skip = __ballot_sync" + adjoint.ADJ_DEVICE.split(
            "    unsigned skip = __ballot_sync", 1)[1].split("    for (int tile", 1)[0]
        # Execute the real group/state logic; CPU barriers do not qualify CUDA memory ordering.
        harness = r'''
#include <atomic>
#include <barrier>
#include <cstdio>
#include <thread>
#define CS_N 24
#define CM_FEATURE_WORDS 1
#define ADJ_ROWS_SKIPPED 4096
#define ADJ_REPLAY_MISMATCH 8192
#define PG1(p) (2*(p))
#define PG2(p) (2*(p)+1)
thread_local int test_lane;
std::barrier warp(32);
int exchange[32],c_gtype[68],c_gbody[68],pairs[64],slots[64],prepared[34];
unsigned zones[64];bool enabled[64],bad_out[64];float features[64],words_out[64][CS_N];
int bodies_out[64],fail_pair,distance_error;std::atomic<int> failures{0},called[64];
void __syncwarp() { warp.arrive_and_wait(); }
int __ffs(unsigned value) { return __builtin_ffs(value); }
int __shfl_sync(unsigned,int value,int from) {
  exchange[test_lane]=value;__syncwarp();int result=exchange[from];__syncwarp();return result;
}
unsigned __ballot_sync(unsigned,bool value) {
  exchange[test_lane]=value;__syncwarp();unsigned result=0;
  for(int k=0;k<32;k++)if(exchange[k])result|=1u<<k;
  __syncwarp();return result;
}
void atomicOr(int* value,int bits) { *value|=bits; }
struct CMFeature { int contact; };
CMFeature cm_feature_unpack(const float* value) { return {int(*value)}; }
struct Adj {
  int cm_cached_pair=-1,flags=0,cm_scratch=0,cm_linearization=-1;
  float gx[1][3]{},gq[1][4]{};const float* feature_rec=features;
};
int linearize(int g1,int,const float*,const float*,CMFeature feature,int*,int* result) {
  int pair=g1/2;
  if(test_lane!=0 || pairs[feature.contact]!=pair)failures++;
  prepared[pair]++;
  if(pair==fail_pair)return 16;
  *result=pair;return 0;
}
#define convex_manifold_linearize linearize
#define box_box_manifold_linearize linearize
float adj_contact_one(Adj& s,int c,int pair,int slot,unsigned zone,float* words,float*) {
  bool native=c_gtype[PG2(pair)]==4 || (c_gtype[PG1(pair)]==3 && c_gtype[PG2(pair)]==3);
  if((native && (s.cm_cached_pair!=pair || s.cm_linearization!=pair)) ||
     pair!=pairs[c] || slot!=slots[c] || zone!=zones[c])failures++;
  called[c]++;
  for(int k=0;k<CS_N;k++)words[k]=float(1+100*c+k);
  return c==distance_error ? 0.f : -1.f;
}
void run_lane(Adj& s,int lane,bool debug) {
  test_lane=lane;
  for(int c0=0;c0<50;c0+=32) {
    int c=c0+lane;bool on=c<50,use=on && enabled[c];
    int pair=on?pairs[c]:0,slot=slots[c],cb=-1;unsigned zone=zones[c];
    float cw[CS_N]={},debug_value=0.f,*dbg=debug?&debug_value:nullptr;bool bad=false;
'''
        main = r'''
    for(int k=0;k<CS_N;k++)words_out[c][k]=cw[k];bad_out[c]=bad;bodies_out[c]=cb;
    __syncwarp();
  }
}
int main() {
  for(int p=0;p<34;p++) {
    c_gtype[2*p]=c_gtype[2*p+1]=p<30?1:(p==31?3:4);
    c_gbody[2*p]=1;c_gbody[2*p+1]=2;
  }
  for(int c=0;c<64;c++) {
    pairs[c]=c<30?c:(c<34?30:(c<42?31:(c<46?32:33)));
    slots[c]=c<30?0:(c<34?c-30:(c<42?c-34:(c<46?c-42:c-46)));
    zones[c]=c<46 && c%2 ? 1u:0u;enabled[c]=c!=1 && c!=39;features[c]=float(c);
  }
  for(bool debug:{false,true})for(int error_case=0;error_case<3;error_case++) {
    fail_pair=error_case==1?32:-1;distance_error=error_case==2?7:-1;Adj state;
    for(auto& count:called)count=0;for(int& count:prepared)count=0;
    std::thread lanes[32];
    for(int lane=0;lane<32;lane++)lanes[lane]=std::thread(run_lane,std::ref(state),lane,debug);
    for(auto& lane:lanes)lane.join();
    for(int c=0;c<50;c++) {
      bool active=enabled[c] && (zones[c] || debug),failed=active && pairs[c]==fail_pair;
      if(called[c]!=(active && !failed) || bad_out[c]!=(failed || (active && c==distance_error)) ||
         bodies_out[c]!=(enabled[c] && zones[c] ? 1|(2<<8):-1))failures++;
      for(int k=0;k<CS_N;k++)if(words_out[c][k]!=(active && !failed?float(1+100*c+k):0.f))failures++;
    }
    for(int p=0;p<34;p++)if(prepared[p]!=(p>=30 && (p!=33 || debug)))failures++;
    int expected=ADJ_ROWS_SKIPPED | (error_case ? ADJ_REPLAY_MISMATCH:0);
    if(state.flags!=expected)failures++;
  }
  if(failures)std::fprintf(stderr,"contact-group failures: %d\n",int(failures));
  return failures?1:0;
}
'''
        with tempfile.TemporaryDirectory(prefix="nabla-contact-groups-cpu-") as directory:
            path = Path(directory)
            (path / "groups.cc").write_text(harness + group + flags + main)
            compiled = subprocess.run(["c++", "-std=c++20", "-O2", "-pthread", str(path / "groups.cc"),
                                       "-o", str(path / "groups")], text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(path / "groups")], text=True, capture_output=True, timeout=20)
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
