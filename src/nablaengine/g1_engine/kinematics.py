"""CUDA source of the G1 engine's kinematics outputs and their VJP: body poses, COM velocities and subtree COMs of a
state (qpos, qvel) with the per-world torso COM offset com, as MuJoCo MJX's mjx.kinematics + smooth.com_pos +
smooth.com_vel (mujoco 3.17.0, mujoco/mjx/_src/smooth.py) compute them (qpos, xpos, xquat, cvel, subtree_com), and the
VJP of that map. nablaengine.jax_ffi runs it after the step's last
substep (step(..., kinematics=True): the output state's) and on its own (jax_ffi.kinematics).

Semantics (MJX source line numbers of mujoco 3.17.0; the mjlab G1: a free root body 1, hinge joints at the body
origins, no mocap bodies; check_model asserts what is used):
  qpos_n  qpos with the free joint's quaternion normalized and written back (smooth.py 49, 67-70: "also normalize
          qpos while we're at it", qpos.at[3:7].set(normalize(qpos[3:7])); math.normalize = x / |x| (math.py 79-108)).
          Its derivative removes the quaternion cotangent's radial part: g_q = (g - u (u.g)) / |q|, u = q / |q|
  xpos, xquat  (B, nbody, 3 / 4), the body frames, world body 0 included (smooth.py 47-108, scan.body_tree root to
          leaves): body 0 = (body_pos[0], body_quat[0]) = (0, (1, 0, 0, 0)); the root = (qpos[0:3], qpos_n[3:7]) (60-70);
          a hinge body b: pos = xpos[p] + rotate(body_pos[b], xquat[p]), quat = xquat[p] * body_quat[b] (51-54), then
          quat = quat * axis_angle(jnt_axis, qpos - qpos0) (78-83; jnt_pos = 0 and qpos0 = 0 on the hinges: the
          anchor is the body origin and the angle qpos). Here the engine's fwd_body (source.DEVICE, the explicit
          roundings of the forward's substep_pre): the bits of the poses and twists that the next substep computes at
          its start for this state. The quaternions of the hinge bodies are products, not renormalized (as MJX)
  subtree_com  (B, nbody, 3) (smooth.py 150-165): xipos = xpos + rotate(body_ipos, xquat) (121, support.py 208-216),
          body_ipos of the torso plus com (the per-world COM offset: body_ipos.at[torso].add(com); the engine's
          body_ipos(b, com)), subtree sums of mass * xipos and mass leaf to root, their quotient (mass < mjMINVAL: none
          in this model, asserted). The world body's subtree is body 1's (scan.body_tree's reverse scan sums body 1
          into body 0; body 0 has mass 0): subtree_com[0] = subtree_com[1] (bitwise in MJX too)
  cvel    (B, nbody, 6) = [angular; linear] (smooth.py 415-457 with the cdof of com_pos 182-219): the twist of each body
          at the reference point subtree_com[body_rootid[b]] = subtree_com[1] for b >= 1 (body 0: zeros): angular
          W_b = world angular velocity (root: R_1 qvel[3:6]; child: W_parent + xaxis qd), linear V_b + W_b x
          (subtree_com[1] - xpos[b]) with V_b the velocity of the body origin (root: qvel[0:3]; child: V_parent +
          W_parent x (xpos[b] - xpos[parent])). MJX sums cdof * qvel down the tree, cdof = [xaxis; xaxis x (root_com -
          xanchor)] (hinge), the free joint's translations [0; e_k] and rotations [R e_k; R e_k x (root_com -
          qpos[0:3])]: the same twist; a caller that wants the velocity of the body origin undoes the shift (lin -
          (xpos - subtree_com[root]) x ang)

VJP (cotangents of qpos_n, xpos, xquat, cvel, subtree_com -> of qpos, qvel; com gets none): one warp per world, lane b =
body b, with the forward recomputed from (qpos, qvel, com) (no residuals besides the inputs). The pose cotangents are
kept per body as a world force f bar (position) and torque tau bar (rotation; a quaternion cotangent g_Q gives
tau bar = 1/2 vec(g_Q Q*)), as the adjoint's pose sweep (adjoint.py adj_tree_up): cvel's -> the twists' and
the reference point's (the root's subtree COM), subtree_com's -> the subtree sums' (a root-to-leaf sweep of the path
sums, times the mass: the xipos cotangents -> f bar, tau bar += c x g), the twist recursion reversed leaf to root (V_c =
V_b + W_b x r_c: W bar_b += r_c x V bar_c, r bar_c = V bar_c x W_b; W_c = W_b + z_c qd: qd bar = z_c . W bar_c, tau bar_c
+= z_c x qd W bar_c) in the same pass as the pose sweep F_b = f bar_b + sum (F_c - r bar_c), N_b = tau bar_b + sum (N_c
+ r_c x F_c): hinge qpos bar = z_b . N_b, the root's position F_1, its quaternion through the normalization: a = 2 [0;
N_1] Q_1 + g_xquat[1] + g_qpos_n[3:7], g_q = (a - u (u.a)) / |q|; qpos_n's other entries pass their cotangent through.
The step's combined backward adds the incoming cotangents of (qpos', qvel') last (one rounding: bitwise what JAX's sum of
the two VJPs gives).
"""

import mujoco
import numpy as np

KIN_WPB = 4  # worlds (warps) per block of k_kin / k_kin_vjp
OUTPUTS = ("qpos_n", "xpos", "xquat", "cvel", "subtree_com")


def shapes(m, B):
    """{output: shape} of one call for batch size B."""
    nb = m.nbody
    return dict(qpos_n=(B, m.nq), xpos=(B, nb, 3), xquat=(B, nb, 4), cvel=(B, nb, 6), subtree_com=(B, nb, 3))


def check_model(m):
    """Assert what the kernels assume of the model (beyond source.constants: free root body 1, hinges at the body
    origins): MJX's world body at the origin, massless, with body 1 its only child; one hinge per body after the root,
    with qpos0 = 0 (MJX's angle is qpos - qpos0, the engine's qpos); no mocap bodies; every subtree mass >= mjMINVAL
    (MJX's fallback to xipos unused); one kinematic tree (body_rootid 1 for b >= 1). The capability checker's
    kinematics scope (nablaengine.model.capability: H1, H6, H7, H8, H25, H31) runs first and raises its UnsupportedModel
    (an AssertionError) with the codes; the asserts below are the same conditions. A jointless (welded) body has
    body_jntadr -1 (which would index the last joint): body_jntnum is checked first."""
    from nablaengine.model import capability
    capability.require(m, None, None, "kinematics", "kinematics outputs")
    assert m.nbody <= 64 and m.nmocap == 0
    assert not np.any(m.body_pos[0]) and np.array_equal(m.body_quat[0], [1.0, 0.0, 0.0, 0.0]) and m.body_mass[0] == 0
    roots = [b for b in range(1, m.nbody) if m.body_parentid[b] == 0]
    assert len(roots) in (1, 2) and int(m.body_rootid[0]) == 0
    for b in range(1, m.nbody):
        assert m.body_jntnum[b] == 1 and m.body_jntadr[b] >= 0, f"body {b}: one joint (a jointless body has none)"
        j = m.body_jntadr[b]
        assert m.jnt_type[j] == (mujoco.mjtJoint.mjJNT_FREE if b in roots else mujoco.mjtJoint.mjJNT_HINGE)
        assert b in roots or m.qpos0[m.jnt_qposadr[j]] == 0.0
    sub = np.array(m.body_mass, np.float64)
    for b in range(m.nbody - 1, 0, -1):
        sub[m.body_parentid[b]] += sub[b]
    assert sub.min() >= mujoco.mjMINVAL, "a subtree mass below mjMINVAL (MJX would use xipos)"


KIN_DEVICE = r"""
// ---- kinematics outputs (nablaengine.g1_engine.kinematics; MJX kinematics + com_pos + com_vel) and their VJP.
// One warp per world (lane b = body b, the tree one depth level at a time, as substep_pre), KIN_WPB worlds per block,
// each world's state in its own slice of static shared memory (kept small: the kernels are latency bound, so the
// worlds per SM count). Inputs and outputs whole per world, lane-strided (coalesced); k_kin_vjp reads its cotangents
// by body rows (lane b: body b's few floats; the warp's loads cover one contiguous segment, cached)
#ifndef KIN_WPB
#define KIN_WPB 4
#endif
#define KIN_QOUT (NB * 3)  // floats per world of xpos and subtree_com; xquat NB * 4, cvel NB * 6, qpos_n NQ

struct KinF {  // the forward's per-world state (2.4 KB)
  float Q[NB][4], X[NB][3];  // fwd_body's poses
  union {
    struct { float W[NB][3], V[NB][3]; };  // fwd_body's twists (W angular, V the velocity of the body origin)
    float cv[NB][6];  // then cvel (staged for the copy-out)
  };
  float C[NB][3], Mt[NB];  // the subtree sums of m xipos, then the subtree COMs; the subtree masses
  float qpos[NQ], qvel[NV], com[COM_WIDTH];
};
struct KinB {  // the VJP's per-world state (4.5 KB): the forward's, the sweeps' cotangents, the gradients
  float Q[NB][4], X[NB][3], W[NB][3], V[NB][3], Mt[NB];
  union {
    float C[NB][3];  // the forward's subtree sums / COMs (C_1 is kept in registers before G overwrites them)
    float G[NB][3];  // sum over the path root .. b of the subtree sums' cotangents (the down sweep)
  };
  float qpos[NQ], qvel[NV], com[COM_WIDTH];
  float bW[NB][3], bV[NB][3], F[NB][3], N[NB][3], rb[NB][3];  // twists' cotangents, pose sweep F_b, N_b, r bar_b
  float gq[NQ], gv[NV];
};

__device__ __forceinline__ void kin_copy(float* dst, const float* src, int n, int lane) {
  for (int i = lane; i < n; i += 32) dst[i] = src[i];
}

// body b's children (source.constants) into registers before a sweep: their count
__device__ __forceinline__ int kin_children(int b, int* ch) {
  const int n = b >= 1 && b < NB ? c_nchild[b] : 0;
#pragma unroll
  for (int i = 0; i < MAX_CHILD; i++) ch[i] = i < n ? c_child[MAX_CHILD * b + i] : 0;
  return n;
}

// one world's state from (qpos, qvel, com) already in s: poses and twists by the forward's fwd_body steps (the root:
// fwd_body itself; a hinge body: fwd_hinge_rot for every body before the sweep, off the levels' chain, then
// fwd_hinge_pos and fwd_hinge_rest at its level: the bits of fwd_body, whose acceleration channel is dead code here),
// then the subtree COMs C and masses Mt (MJX com_pos: leaf to root, own m xipos plus the children's sums in child
// order; C_0 = C_1, body 0 being massless with body 1 its only child, at depth 1)
template <class S>
__device__ __forceinline__ void kin_state_single(S& s, int lane) {
  const int b = lane, d = b < NB ? c_depth[b] : -1, p = b < NB ? c_parent[b] : 0;
  float qb[4] = {1.f, 0.f, 0.f, 0.f}, pb[3] = {0.f, 0.f, 0.f}, ax[3] = {0.f, 0.f, 1.f}, ha[4] = {1.f, 0.f, 0.f, 0.f};
  float qd = 0.f;
  if (b >= 2 && b < NB) {
    for (int k = 0; k < 4; k++) qb[k] = c_quat[4 * b + k];
    for (int k = 0; k < 3; k++) { pb[k] = c_pos[3 * b + k]; ax[k] = c_axis[3 * b + k]; }
    qd = s.qvel[c_dadr[b]];
    fwd_hinge_rot(s.qpos[c_qadr[b]], ax, ha);
  }
  if (lane == 0) {
    s.Q[0][0] = 1.f; s.Q[0][1] = s.Q[0][2] = s.Q[0][3] = 0.f;
    for (int k = 0; k < 3; k++) s.X[0][k] = s.W[0][k] = s.V[0][k] = 0.f;
  }
  __syncwarp();
  for (int L = 1; L <= MAXD; L++) {
    if (d == L) {
      float Qp[4], Xp[3], Wp[3], Vp[3], Qb[4], Xb[3], Wb[3], Vb[3], ALb[3], Ab[3], q0[4];
      const float z3[3] = {0.f, 0.f, 0.f};
      for (int k = 0; k < 4; k++) Qp[k] = s.Q[p][k];
      for (int k = 0; k < 3; k++) { Xp[k] = s.X[p][k]; Wp[k] = s.W[p][k]; Vp[k] = s.V[p][k]; }
      if (L == 1) {  // body 1, the free root
        fwd_body(1, Qp, Xp, Wp, Vp, z3, z3, s.qpos, s.qvel, s.qvel, Qb, Xb, Wb, Vb, ALb, Ab);
      } else {
        fwd_hinge_pos(Qp, Xp, qb, pb, q0, Xb);
        fwd_hinge_rest(q0, ha, ax, qd, 0.f, Xp, Wp, Vp, z3, z3, Qb, Xb, Wb, Vb, ALb, Ab);
      }
      for (int k = 0; k < 4; k++) s.Q[b][k] = Qb[k];
      for (int k = 0; k < 3; k++) { s.X[b][k] = Xb[k]; s.W[b][k] = Wb[k]; s.V[b][k] = Vb[k]; }
    }
    __syncwarp();
  }
  float S3[3] = {0.f, 0.f, 0.f}, Mb = 0.f;
  int ch[MAX_CHILD];
  const int nch = kin_children(b, ch);
  if (b >= 1 && b < NB) {
    float ip[3], c[3];
    body_ipos(b, s.com, ip);
    qrot(s.Q[b], ip, c);
    Mb = c_mass[b];
    for (int k = 0; k < 3; k++) { S3[k] = (s.X[b][k] + c[k]) * Mb; s.C[b][k] = S3[k]; }
    s.Mt[b] = Mb;
  }
  __syncwarp();
  for (int L = MAXD - 1; L >= 1; L--) {
    if (d == L && nch > 0) {
#pragma unroll
      for (int i = 0; i < MAX_CHILD; i++)
        if (i < nch) {
          for (int k = 0; k < 3; k++) S3[k] += s.C[ch[i]][k];
          Mb += s.Mt[ch[i]];
        }
      for (int k = 0; k < 3; k++) s.C[b][k] = S3[k];
      s.Mt[b] = Mb;
    }
    __syncwarp();
  }
  if (b >= 1 && b < NB)
    for (int k = 0; k < 3; k++) s.C[b][k] = S3[k] / Mb;
  __syncwarp();
  if (lane == 0) {
    for (int k = 0; k < 3; k++) s.C[0][k] = s.C[1][k];
    s.Mt[0] = s.Mt[1];
  }
  __syncwarp();
}

template <class S>
__device__ void kin_state_coupled(S& s, int lane) {
  if (lane == 0) {
    s.Q[0][0] = 1.f; s.Q[0][1] = s.Q[0][2] = s.Q[0][3] = 0.f;
    for (int k = 0; k < 3; k++) s.X[0][k] = s.W[0][k] = s.V[0][k] = 0.f;
  }
  __syncwarp();
  for (int L = 1; L <= MAXD; L++) {
    for (int b = lane; b < NB; b += 32)
      if (c_depth[b] == L) {
        int p = c_parent[b];
        const float z[3] = {0.f, 0.f, 0.f};
        float al[3], a[3];
        fwd_body(b, s.Q[p], s.X[p], s.W[p], s.V[p], z, z, s.qpos, s.qvel, s.qvel,
                 s.Q[b], s.X[b], s.W[b], s.V[b], al, a);
      }
    __syncwarp();
  }
  for (int b = lane; b < NB; b += 32) {
    float ip[3], c[3];
    body_ipos(b, s.com, ip); qrot(s.Q[b], ip, c);
    s.Mt[b] = c_mass[b];
    for (int k = 0; k < 3; k++) s.C[b][k] = c_mass[b] * (s.X[b][k] + c[k]);
  }
  __syncwarp();
  for (int L = MAXD - 1; L >= 0; L--) {
    for (int b = lane; b < NB; b += 32)
      if (c_depth[b] == L) {
        int nc = b ? c_nchild[b] : NR;
        for (int j = 0; j < nc; j++) {
          int c = b ? c_child[MAX_CHILD * b + j] : c_roots[j];
          s.Mt[b] += s.Mt[c];
          for (int k = 0; k < 3; k++) s.C[b][k] += s.C[c][k];
        }
      }
    __syncwarp();
  }
  for (int b = lane; b < NB; b += 32)
    for (int k = 0; k < 3; k++) s.C[b][k] /= s.Mt[b];
  __syncwarp();
}

template <class S>
__device__ __forceinline__ void kin_state(S& s, int lane) {
#if NR == 2
  kin_state_coupled(s, lane);
#else
  kin_state_single(s, lane);
#endif
}

template <class S>
__device__ __forceinline__ void kin_load(S& s, int w, const float* qpos, const float* qvel, const float* com,
                                         int lane) {
  kin_copy(s.qpos, qpos + (size_t)w * NQ, NQ, lane);
  kin_copy(s.qvel, qvel + (size_t)w * NV, NV, lane);
  if (lane < COM_WIDTH) s.com[lane] = com[(size_t)w * COM_WIDTH + lane];
  __syncwarp();
}

// the kinematics outputs of nw worlds: qn (nw, NQ), xpos (nw, NB, 3), xquat (nw, NB, 4), cvel (nw, NB, 6), sc (nw, NB,
// 3); block of KIN_WPB warps, warp = world
__global__ void __launch_bounds__(32 * KIN_WPB) k_kin(int nw, const float* qpos, const float* qvel, const float* com,
                                                      float* qn, float* xpos, float* xquat, float* cvel, float* sc) {
  __shared__ KinF sm[KIN_WPB];
  const int lane = threadIdx.x & 31, w = blockIdx.x * KIN_WPB + (threadIdx.x >> 5);
  if (w >= nw) return;  // a whole warp
  KinF& s = sm[threadIdx.x >> 5];
  kin_load(s, w, qpos, qvel, com, lane);
  kin_state(s, lane);
  float o[(NB + 31) / 32][6] = {};
  for (int b = lane; b < NB; b += 32)
    if (b) {
      float r[3], u[3];
      for (int k = 0; k < 3; k++) r[k] = s.C[c_root[b]][k] - s.X[b][k];
      cross(s.W[b], r, u);
      for (int k = 0; k < 3; k++) { o[b / 32][k] = s.W[b][k]; o[b / 32][3 + k] = s.V[b][k] + u[k]; }
    }
  __syncwarp();
  for (int b = lane; b < NB; b += 32)
    for (int k = 0; k < 6; k++) s.cv[b][k] = o[b / 32][k];
  __syncwarp();
  float* oq = qn + (size_t)w * NQ;
  for (int i = lane; i < NQ; i += 32) oq[i] = s.qpos[i];
  __syncwarp();
  for (int r = lane; r < NR; r += 32) {
    int b = c_roots[r];
    for (int k = 0; k < 4; k++) oq[c_qadr[b] + 3 + k] = s.Q[b][k];
  }
  kin_copy(xpos + (size_t)w * KIN_QOUT, &s.X[0][0], KIN_QOUT, lane);
  kin_copy(xquat + (size_t)w * NB * 4, &s.Q[0][0], NB * 4, lane);
  kin_copy(cvel + (size_t)w * NB * 6, &s.cv[0][0], NB * 6, lane);
  kin_copy(sc + (size_t)w * KIN_QOUT, &s.C[0][0], KIN_QOUT, lane);
}

__device__ void kin_vjp_coupled(KinB& s, int w, const float* gqn, const float* gxpos, const float* gxquat,
                                const float* gcvel, const float* gsc, const float* gq_in, const float* gv_in,
                                float* gq, float* gv, int lane) {
  for (int b = lane; b < NB; b += 32) {
    if (!b) continue;
    size_t row = (size_t)w * NB + b;
    float vb[3], r[3], t[3];
    for (int k = 0; k < 3; k++) { vb[k] = gcvel[row * 6 + 3 + k]; r[k] = s.C[c_root[b]][k] - s.X[b][k]; }
    cross(r, vb, t);
    for (int k = 0; k < 3; k++) { s.bW[b][k] = gcvel[row * 6 + k] + t[k]; s.bV[b][k] = vb[k]; }
    cross(vb, s.W[b], s.rb[b]);
    for (int k = 0; k < 3; k++) { s.F[b][k] = gxpos[row * 3 + k] - s.rb[b][k]; s.N[b][k] = 0.f; }
    if (!c_isfree[b]) {
      const float* q = s.Q[b], *g = gxquat + row * 4;
      s.N[b][0] = 0.5f * (-g[0] * q[1] + g[1] * q[0] - g[2] * q[3] + g[3] * q[2]);
      s.N[b][1] = 0.5f * (-g[0] * q[2] + g[1] * q[3] + g[2] * q[0] - g[3] * q[1]);
      s.N[b][2] = 0.5f * (-g[0] * q[3] - g[1] * q[2] + g[2] * q[1] + g[3] * q[0]);
    }
  }
  __syncwarp();
  // C has been consumed by every body before its union is reused for the subtree-sum cotangents G.
  for (int b = lane; b < NB; b += 32) {
    for (int k = 0; k < 3; k++) s.G[b][k] = gsc[((size_t)w * NB + b) * 3 + k] / s.Mt[b];
    if (b && c_isfree[b])
      for (int j = 1; j < NB; j++)
        if (c_root[j] == b)
          for (int k = 0; k < 3; k++) s.G[b][k] += s.rb[j][k] / s.Mt[b];
  }
  __syncwarp();
  for (int L = 1; L <= MAXD; L++) {
    for (int b = lane; b < NB; b += 32)
      if (c_depth[b] == L)
        for (int k = 0; k < 3; k++) s.G[b][k] += s.G[c_parent[b]][k];
    __syncwarp();
  }
  for (int b = lane; b < NB; b += 32)
    if (b) {
      float ip[3], c[3], g[3], t[3];
      body_ipos(b, s.com, ip); qrot(s.Q[b], ip, c);
      for (int k = 0; k < 3; k++) { g[k] = c_mass[b] * s.G[b][k]; s.F[b][k] += g[k]; }
      cross(c, g, t);
      for (int k = 0; k < 3; k++) s.N[b][k] += t[k];
    }
  __syncwarp();
  for (int L = MAXD; L >= 1; L--) {
    for (int b = lane; b < NB; b += 32)
      if (c_depth[b] == L) {
        int p = c_parent[b], d = c_dadr[b], q = c_qadr[b];
        float bw[3], bv[3], f[3], n[3], t[3], u[3];
        for (int k = 0; k < 3; k++) { bw[k] = s.bW[b][k]; bv[k] = s.bV[b][k]; f[k] = s.F[b][k]; n[k] = s.N[b][k]; }
        for (int j = 0; j < c_nchild[b]; j++) {
          int c = c_child[MAX_CHILD * b + j];
          float r[3];
          for (int k = 0; k < 3; k++) r[k] = s.X[c][k] - s.X[b][k];
          cross(r, s.bV[c], t); cross(r, s.F[c], u);
          for (int k = 0; k < 3; k++) {
            bw[k] += s.bW[c][k] + t[k]; bv[k] += s.bV[c][k];
            f[k] += s.F[c][k] - s.rb[c][k]; n[k] += s.N[c][k] + u[k];
          }
        }
        if (!c_isfree[b]) {
          float z[3], rb[3];
          qrot(s.Q[b], c_axis + 3 * b, z);
          s.gv[d] = dot3(z, bw);
          for (int k = 0; k < 3; k++) u[k] = s.qvel[d] * bw[k];
          cross(z, u, t); cross(bv, s.W[p], rb);
          for (int k = 0; k < 3; k++) { n[k] += t[k]; f[k] += rb[k]; s.rb[b][k] = rb[k]; }
          s.gq[q] = dot3(z, n);
        } else {
          qrot_inv(s.Q[b], bw, u);
          for (int k = 0; k < 3; k++) { s.gv[d + k] = bv[k]; s.gv[d + 3 + k] = u[k]; s.gq[q + k] = f[k]; }
          cross(s.W[b], bw, t);
          for (int k = 0; k < 3; k++) n[k] += t[k];
          float n0[4] = {0.f, n[0], n[1], n[2]}, a[4];
          qmul(n0, s.Q[b], a);
          size_t row = (size_t)w * NB + b;
          for (int k = 0; k < 4; k++) a[k] = 2.f * a[k] + gxquat[row * 4 + k] + gqn[(size_t)w * NQ + q + 3 + k];
          const float* input = s.qpos + q + 3;
          float norm = sqrtf(dot4(input, input)), radial = dot4(s.Q[b], a);
          for (int k = 0; k < 4; k++) s.gq[q + 3 + k] = (a[k] - s.Q[b][k] * radial) / norm;
        }
        for (int k = 0; k < 3; k++) { s.bW[b][k] = bw[k]; s.bV[b][k] = bv[k]; s.F[b][k] = f[k]; s.N[b][k] = n[k]; }
      }
    __syncwarp();
  }
  for (int i = lane; i < NQ; i += 32) {
    bool quaternion = false;
    for (int r = 0; r < NR; r++) { int q = c_qadr[c_roots[r]]; quaternion |= i >= q + 3 && i < q + 7; }
    float value = s.gq[i] + (quaternion ? 0.f : gqn[(size_t)w * NQ + i]);
    gq[(size_t)w * NQ + i] = value + (gq_in ? gq_in[(size_t)w * NQ + i] : 0.f);
  }
  for (int i = lane; i < NV; i += 32)
    gv[(size_t)w * NV + i] = s.gv[i] + (gv_in ? gv_in[(size_t)w * NV + i] : 0.f);
}

// the VJP: cotangents of k_kin's outputs (gqn, gxpos, gxquat, gcvel, gsc, same layouts) -> gq (nw, NQ), gv (nw, NV)
// of (qpos, qvel); gq_in / gv_in (may be null: zero) are added last (the step's combined backward: its outputs'
// direct cotangents)
__global__ void __launch_bounds__(32 * KIN_WPB) k_kin_vjp(int nw, const float* qpos, const float* qvel,
                                                          const float* com, const float* gqn, const float* gxpos,
                                                          const float* gxquat, const float* gcvel, const float* gsc,
                                                          const float* gq_in, const float* gv_in, float* gq,
                                                          float* gv) {
  __shared__ KinB sm[KIN_WPB];
  const int lane = threadIdx.x & 31, w = blockIdx.x * KIN_WPB + (threadIdx.x >> 5);
  if (w >= nw) return;
  KinB& s = sm[threadIdx.x >> 5];
  kin_load(s, w, qpos, qvel, com, lane);
  kin_state(s, lane);
#if NR == 2
  kin_vjp_coupled(s, w, gqn, gxpos, gxquat, gcvel, gsc, gq_in, gv_in, gq, gv, lane);
#else
  const int b = lane, d = b < NB ? c_depth[b] : -1, p = b < NB ? c_parent[b] : 0;
  const bool body = b >= 1 && b < NB;
  const size_t row = (size_t)w * NB + (body ? b : 0);  // this lane's body row of the cotangents
  const float C1[3] = {s.C[1][0], s.C[1][1], s.C[1][2]};
  // cvel_b = [W_b; V_b + W_b x r_b], r_b = C_1 - X_b: W bar, V bar; r bar = nu bar x W_b -> C_1 bar, -X_b bar.
  // xquat (b >= 2; the root's goes into its quaternion directly): tau bar = 1/2 vec(g_Q Q*)
  float fb[3] = {0.f, 0.f, 0.f}, tb[3] = {0.f, 0.f, 0.f}, gr[3] = {0.f, 0.f, 0.f}, gS[3] = {0.f, 0.f, 0.f};
  if (body) {
    float wb[3], vb[3], r[3], t[3];
    for (int k = 0; k < 3; k++) { wb[k] = gcvel[row * 6 + k]; vb[k] = gcvel[row * 6 + 3 + k]; r[k] = C1[k] - s.X[b][k]; }
    cross(r, vb, t);
    for (int k = 0; k < 3; k++) { s.bW[b][k] = wb[k] + t[k]; s.bV[b][k] = vb[k]; }
    cross(vb, s.W[b], gr);
    for (int k = 0; k < 3; k++) { fb[k] = gxpos[row * 3 + k] - gr[k]; gS[k] = gsc[row * 3 + k]; }
    if (b >= 2) {
      const float* q = s.Q[b];  // vec(g q*)
      const float g0 = gxquat[row * 4], g1 = gxquat[row * 4 + 1], g2 = gxquat[row * 4 + 2], g3 = gxquat[row * 4 + 3];
      tb[0] = 0.5f * (-g0 * q[1] + g1 * q[0] - g2 * q[3] + g3 * q[2]);
      tb[1] = 0.5f * (-g0 * q[2] + g1 * q[3] + g2 * q[0] - g3 * q[1]);
      tb[2] = 0.5f * (-g0 * q[3] - g1 * q[2] + g2 * q[1] + g3 * q[0]);
    }
  }
  float gc1[3];  // C_1 bar from the cvel reference point (every lane: the same sum)
  for (int k = 0; k < 3; k++) gc1[k] = warp_sum(gr[k]);
  // subtree_com: C_b = S_b / M_b (C_0 = C_1): S bar_b = C bar_b / M_b; xipos_j bar = m_j sum over the path root .. j
  if (body)
    for (int k = 0; k < 3; k++)
      gS[k] = (b == 1 ? gS[k] + gsc[(size_t)w * NB * 3 + k] + gc1[k] : gS[k]) / s.Mt[b];
  __syncwarp();  // (every lane has read C_1: G overwrites C)
  for (int L = 1; L <= MAXD; L++) {
    if (d == L)
      for (int k = 0; k < 3; k++) { gS[k] += b >= 2 ? s.G[p][k] : 0.f; s.G[b][k] = gS[k]; }
    __syncwarp();
  }
  if (body) {  // xipos_b = X_b + c_b, c_b = R_b ipos_b: f bar += g, tau bar += c x g
    float ip[3], c[3], g[3], t[3];
    body_ipos(b, s.com, ip);
    qrot(s.Q[b], ip, c);
    const float m = c_mass[b];
    for (int k = 0; k < 3; k++) { g[k] = m * gS[k]; fb[k] += g[k]; }
    cross(c, g, t);
    for (int k = 0; k < 3; k++) { s.F[b][k] = fb[k]; s.N[b][k] = tb[k] + t[k]; }
  }
  __syncwarp();
  // leaf to root: the twists' recursion reversed with the pose sweep (adjoint.py adj_tree_up's pattern: a parent pulls
  // its children's finished cotangents, deterministic). Off the levels' chain, before: the children, the hinge axis z_b
  // = R_b axis_b (W_b = W_p + z qd), its rate and addresses
  int ch[MAX_CHILD];
  const int nch = kin_children(b, ch), iv = b >= 2 && b < NB ? c_dadr[b] : 0, iq = b >= 2 && b < NB ? c_qadr[b] : 0;
  float z[3] = {0.f, 0.f, 0.f};
  if (b >= 2 && b < NB) qrot(s.Q[b], c_axis + 3 * b, z);
  const float qd = s.qvel[iv];
  for (int L = MAXD; L >= 1; L--) {
    if (d == L) {
      float bW[3], bV[3], F[3], N[3], t[3], u[3];
      for (int k = 0; k < 3; k++) { bW[k] = s.bW[b][k]; bV[k] = s.bV[b][k]; F[k] = s.F[b][k]; N[k] = s.N[b][k]; }
#pragma unroll
      for (int i = 0; i < MAX_CHILD; i++) {
        if (i >= nch) break;
        const int c = ch[i];
        float r[3];
        for (int k = 0; k < 3; k++) r[k] = s.X[c][k] - s.X[b][k];
        cross(r, s.bV[c], t);  // V_c = V_b + W_b x r_c
        cross(r, s.F[c], u);
        for (int k = 0; k < 3; k++) {
          bW[k] += s.bW[c][k] + t[k]; bV[k] += s.bV[c][k];
          F[k] += s.F[c][k] - s.rb[c][k]; N[k] += s.N[c][k] + u[k];
        }
      }
      if (b >= 2) {
        const int i = iv;
        float rb[3];
        s.gv[i] = dot3(z, bW);
        for (int k = 0; k < 3; k++) u[k] = qd * bW[k];
        cross(z, u, t);  // z = R_b axis rotates with the body
        cross(bV, s.W[p], rb);  // V_b = V_p + W_p x r_b
        for (int k = 0; k < 3; k++) { N[k] += t[k]; F[k] += rb[k]; s.rb[b][k] = rb[k]; }
        s.gq[iq] = dot3(z, N);
      } else {  // the root: V_1 = qvel[0:3], W_1 = R_1 qvel[3:6]; X_1 = qpos[0:3], Q_1 = normalize(qpos[3:7])
        float l[3];
        qrot_inv(s.Q[1], bW, l);
        for (int k = 0; k < 3; k++) { s.gv[k] = bV[k]; s.gv[3 + k] = l[k]; s.gq[k] = F[k]; }
        cross(s.W[1], bW, t);
        for (int k = 0; k < 3; k++) N[k] += t[k];
        const float n0[4] = {0.f, N[0], N[1], N[2]}, *u4 = s.Q[1], *q = s.qpos + 3;
        float a[4];
        qmul(n0, u4, a);  // dL/dQ_1 (tangential) = 2 [0; N] Q_1, plus the direct cotangents of Q_1
        for (int k = 0; k < 4; k++) a[k] = 2.f * a[k] + gxquat[row * 4 + k] + gqn[(size_t)w * NQ + 3 + k];
        const float nq = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        const float ua = u4[0] * a[0] + u4[1] * a[1] + u4[2] * a[2] + u4[3] * a[3];
        for (int k = 0; k < 4; k++) s.gq[3 + k] = (a[k] - u4[k] * ua) / nq;  // normalize's VJP
      }
      for (int k = 0; k < 3; k++) { s.bW[b][k] = bW[k]; s.bV[b][k] = bV[k]; s.F[b][k] = F[k]; s.N[b][k] = N[k]; }
    }
    __syncwarp();
  }
  const float* cq = gqn + (size_t)w * NQ;
  float* oq = gq + (size_t)w * NQ;
  float* ov = gv + (size_t)w * NV;
  for (int i = lane; i < NQ; i += 32) {
    const float g = i >= 3 && i < 7 ? s.gq[i] : s.gq[i] + cq[i];  // qpos_n passes the other entries through
    oq[i] = gq_in ? gq_in[(size_t)w * NQ + i] + g : g;
  }
  for (int i = lane; i < NV; i += 32) ov[i] = gv_in ? gv_in[(size_t)w * NV + i] + s.gv[i] : s.gv[i];
#endif
}

static inline unsigned kin_grid(int nw) { return (unsigned)((nw + KIN_WPB - 1) / KIN_WPB); }
"""
