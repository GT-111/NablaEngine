"""Host-side glue of the G1 engine shared by the check binaries (scripts/engine_step_check.py, the adjoint check binary
of nablaengine.checks, ...) and the JAX FFI library (nablaengine.jax_ffi): KERNEL, the forward kernel k_substep (one
launch per control step of nsub substeps, one block of 32 threads per world, optionally writing the tape), the split
forward (k_pre / k_post, two launches per substep, dk_forward_split: the production forward when DK_SPLIT is 1, the
default; bitwise the same results), and loader(), the host code that copies the packed model constants into the
kernels' DKC arrays (scripts/engine_step_check.py re-exports both).
"""


KERNEL = r"""
#ifndef DK_SPLIT
#define DK_SPLIT 1  // engine_step_check.py's and the FFI's forward: 1 the split kernels (dk_forward_split), 0 k_substep
#endif
#ifdef DK_PROFILE
static_assert(NPROF <= 32, "one profile counter per lane (step.PROF_PHASES: at most 32 phases)");
__device__ unsigned long long g_prof[NPROF];
#endif
// tape: null, or nw * nsub records of REC_WORDS (tape.py), record w * nsub + k for world w and substep k
__global__ void k_substep(int nw, const float* qpos, const float* qvel, const float* warm, const float* ctrl,
                          const float* fric, const float* com, int iterations, int nsub, float* oqpos, float* oqvel,
                          float* owarm, int* info, float* ocon, float* tape, const float* qfrc = nullptr,
                          const float* kp_scale = nullptr) {
  extern __shared__ __align__(16) unsigned char dk_smem[];  // dynamic: struct Sim can exceed the 48 KB static limit
  Sim& s = *reinterpret_cast<Sim*>(dk_smem);
  // lane = threadIdx.x & 31, not threadIdx.x (also in k_pre, k_post, k_adjoint): the block is one warp, but the
  // compiler bounds threadIdx.x only by 1024 (even under __launch_bounds__(32, ...)), so every lane-strided loop of the
  // inlined code (for i = lane; i < n; i += 32) has an unknown trip count and is unrolled by 4 with a remainder loop;
  // with the mask it has at most ceil(n / 32) trips and stays a short loop (a __noinline__ callee's lane parameter has
  // no bound: adjoint.py LANE_BOUND). Results bitwise the same. SASS (sm_86): k_substep 10949 -> 9672 instructions,
  // k_pre 6375 -> 5771, k_post 11983 -> 9303, k_adjoint 2882 -> 1810; k_post's spills on sm_120 / sm_90a (16 / 20 B,
  // ptxas 12.9 on nvcc 12.4's PTX) gone. RTX 3090 (a 4096-state corpus, 10 iterations): the forward 2.5% / 2.2%
  // faster at 1787 / 4096 worlds, k_adjoint 3.0% / 3.1% (the mask alone)
  int lane = threadIdx.x & 31, w = blockIdx.x;
  if (w >= nw) return;
  for (int i = lane; i < NQ; i += 32) s.qpos[i] = qpos[(size_t)w * NQ + i];
  for (int i = lane; i < NV; i += 32) { s.qvel[i] = qvel[(size_t)w * NV + i]; s.qacc[i] = warm[(size_t)w * NV + i]; }
  if (lane < COM_WIDTH) s.com[lane] = com[w * COM_WIDTH + lane];
  if (lane == 0) { s.fric = fric[w]; s.unsupported = 0; }
#ifdef DK_PROFILE
  if (lane < NPROF) s.prof[lane] = 0;
#endif
  __syncwarp();
  int nit = 0;
  for (int k = 0; k < nsub; k++)
    substep(s, ctrl + (size_t)w * NU, iterations, lane, &nit,
            tape ? tape + ((size_t)w * nsub + k) * REC_WORDS : nullptr,
            qfrc ? qfrc + (size_t)w * NV : nullptr, kp_scale ? kp_scale[w] : 1.f);
  for (int i = lane; i < NQ; i += 32) oqpos[(size_t)w * NQ + i] = s.qpos[i];
  for (int i = lane; i < NV; i += 32) { oqvel[(size_t)w * NV + i] = s.qvel[i]; owarm[(size_t)w * NV + i] = s.qacc[i]; }
#ifdef DK_PROFILE
  __syncwarp();
  if (lane < NPROF) atomicAdd(&g_prof[lane], (unsigned long long)s.prof[lane]);
#endif
  if (lane == 0) { info[w * 4] = nit; info[w * 4 + 1] = s.ncon; info[w * 4 + 2] = s.nefc; info[w * 4 + 3] = s.unsupported; }
  for (int c = lane; c < MAXCON; c += 32) {  // contacts of the last substep's collision: pair, dist, pos, normal
    float* o = ocon + ((size_t)w * MAXCON + c) * 8;
    bool v = c < min(s.ncon, MAXCON);
    o[0] = v ? (float)s.con_pair[c] : -1.f; o[1] = v ? s.con_dist[c] : 0.f;
    for (int k = 0; k < 3; k++) { o[2 + k] = v ? s.con_pos[c][k] : 0.f; o[5 + k] = v ? s.con_frame[c][k] : 0.f; }
  }
}

// The split forward: every substep as two halves, substep_pre on struct SimP and substep_post on struct SimN (step.py),
// what crosses between them in struct Hand (global memory, work[w] for world w). The pre half reads the substep's input
// state (the inputs for substep 0, then the outputs), the post half the warm start (warm, then owarm) and writes the
// state after the substep to the outputs, so no other buffer is needed. Each half writes its part of the substep's tape
// record (substep_pre / substep_post, as in k_substep). Results, tape and contacts are bitwise those of k_substep.
// Two kernels per substep, k_pre and k_post (2 nsub launches, dk_forward_split). Every launch ends on a tail: per world
// 3.67 / 3.57 / 3.51 us at 1280 / 4096 / 8192 worlds on V100, 1.55 / 1.21 us at 1804 / 4096 on RTX 3090 (2 and 4.5
// waves; 4096 worlds on the target GPUs are about 2 waves). A persistent single-launch variant (blocks taking (half,
// world) tasks from a counter, per-world release / acquire waits) was 20% slower on V100 and was dropped.
// DK_SPLIT_ORDER 1 (default): k_post runs the worlds with the most constraint rows first (SPLIT_NBUCKET buckets of
// 2^SPLIT_ROW_SHIFT rows, counted by k_pre), so the launch does not end on a late expensive world (rows correlate 0.6
// with the Newton iterations); the results do not depend on the order (nor on the two bucket parameters). At about 2
// waves (4096 worlds on the target GPUs, 1787 on the RTX 3090) k_post's makespan is then the duration of its slowest
// world, which starts in the first wave. RTX 3090 (a 4096-state corpus, 10 iterations, alternating runs): 64 buckets
// of 2 rows with the warp-parallel lookup (dk_order_world) against 16 buckets of 8 rows with a serial lookup on
// lane 0: forward -4.1% at 1787 worlds (a second corpus -2.1%), -2.3% at 4096 (cycles -3.0%); the faster lookup alone
// -0.5%;
// 32 buckets of 4 rows -3.3%, 128 of 1 row -3.8%. Measured-cost keys lose: the world's k_post(k - 1) cycles +5..6%
// (Spearman 0.79 with k_post(k)'s cycles, the rows 0.89: a world's cycles depend on its SM's load), its k_post(k - 1)
// Newton iterations x rows +3.2%; ordering k_pre(k) by the world's k_pre(k - 1) cycles -0.4 / 0.0% (not kept).
// Worlds per SM: the register file is split over the SM's 4 schedulers (16K registers each), so 1-warp blocks fit
// 8 / 12 / 16 per SM at <= 255 / 168 / 128 registers; SPLIT_PRE_MINB / SPLIT_POST_MINB (__launch_bounds__ minimum
// blocks per SM, 0: none) make ptxas fit a kernel's registers to that many blocks. Default 12 (RTX 3090: k_pre 12,
// k_post 11 worlds per SM, 5.61 ms against 6.32 ms uncapped at 8 / 8). V100 (uncapped: k_pre 151, k_post 190
// registers, 12 / 8 per SM; capped: 159 / 168, 12 / 12 per SM): 6.05 against 6.35 ms (a 4096-state corpus, 10
// iterations; 1787 worlds 2.55 / 2.65, a second corpus 6.18 / 6.55, with the tape 6.26 / 6.54), the same results. In
// the current code k_post has no spills at 168 registers (20 / 52 bytes in an earlier version) and the cap is worth
// 5.0% on V100 (5.89 against 6.20 ms);
// SPLIT_POST_MINB 10 / 11 or a runtime cap of 11 / 10 per SM: +0.5% / +0.7% / 0% / +16%. On the RTX 3090 SimN's
// shared memory allows 11 k_post blocks; SPLIT_POST_MINB 11 compiles to the same 168 registers.
// SPLIT_POST_MAXNREG (defined and > 0): k_post's register cap as __maxnreg__(N) (nvcc 12.4+) in place of
// __launch_bounds__(32, SPLIT_POST_MINB). On sm_90 / sm_90a the build gives the forward unit -DSPLIT_PRE_MINB=16
// -DSPLIT_POST_MAXNREG=128 (nablaengine.cuda.unit_defines, $DK_SM90_REG_CAPS; decided by the build target in Python,
// not by __CUDA_ARCH__, which is 900 in sm_120's compute_90 forward unit too): H100 SXM5 (nvcc 12.8) k_pre 158 ->
// 128 registers without spills, k_post 168 -> 128 with spills (76 / 176 B stored / loaded; an earlier version of the
// code with nvcc 12.4: 32 / 88 B), 12 -> 16 worlds per SM each, forward -10.8% at 4096 worlds
// (k_pre's cap alone -3.6%, k_post's -7.1%), bitwise unchanged; on the RTX PRO 6000 the same two settings +0.6%
// (shared memory allows fewer worlds per SM there).
#ifndef DK_SPLIT_ORDER
#define DK_SPLIT_ORDER 1
#endif
#ifndef SPLIT_NBUCKET
#define SPLIT_NBUCKET 64
#endif
#ifndef SPLIT_ROW_SHIFT
#define SPLIT_ROW_SHIFT 1  // the bucket width: 2 rows (bucket 0, the first, also takes the worlds above the range)
#endif
#define SPLIT_MAXSUB 32  // substeps per call with DK_SPLIT_ORDER (the bucket counters)
#define SPLIT_MINB_DEFAULT 12
#ifndef SPLIT_PRE_MINB
#define SPLIT_PRE_MINB SPLIT_MINB_DEFAULT
#endif
#ifndef SPLIT_POST_MINB
#define SPLIT_POST_MINB SPLIT_MINB_DEFAULT
#endif
#if SPLIT_PRE_MINB > 0
#define SPLIT_PRE_BOUNDS __launch_bounds__(32, SPLIT_PRE_MINB)
#else
#define SPLIT_PRE_BOUNDS
#endif
#if defined(SPLIT_POST_MAXNREG) && SPLIT_POST_MAXNREG > 0
#if __CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ < 1204
#error "SPLIT_POST_MAXNREG: __maxnreg__ needs nvcc 12.4 or newer"
#endif
#define SPLIT_POST_BOUNDS __maxnreg__(SPLIT_POST_MAXNREG)
#elif SPLIT_POST_MINB > 0
#define SPLIT_POST_BOUNDS __launch_bounds__(32, SPLIT_POST_MINB)
#else
#define SPLIT_POST_BOUNDS
#endif

// the pre half of substep k for world w
__device__ __forceinline__ void pre_world(SimP& s, Hand& g, int w, int lane, const float* qpos, const float* qvel,
                                          const float* ctrl, const float* fric, const float* com, int k, int nsub,
                                          float* ocon, float* tape, const float* qfrc = nullptr,
                                          const float* kp_scale = nullptr) {
  for (int i = lane; i < NQ; i += 32) s.qpos[i] = qpos[(size_t)w * NQ + i];
  for (int i = lane; i < NV; i += 32) s.qvel[i] = qvel[(size_t)w * NV + i];
  if (lane < COM_WIDTH) s.com[lane] = com[w * COM_WIDTH + lane];
  if (lane == 0) {
    s.fric = fric[w];
    s.unsupported = k ? g.unsupported : 0;  // accumulated over the substeps (the post half does not change it)
    s.M = g.M; s.con_dist = g.con_dist; s.con_pos = g.con_pos; s.con_frame = g.con_frame;
    s.con_J = g.con_J; s.con_dof = g.con_dof; s.con_mask = g.con_mask;
    s.efc_kind = g.efc_kind; s.efc_a = g.efc_a; s.efc_b = g.efc_b; s.efc_D = g.efc_D; s.efc_aref = g.efc_aref;
  }
#ifdef DK_PROFILE
  if (lane < NPROF) s.prof[lane] = 0;
#endif
  __syncwarp();
  substep_pre(s, ctrl + (size_t)w * NU, lane, tape ? tape + ((size_t)w * nsub + k) * REC_WORDS : nullptr,
              qfrc ? qfrc + (size_t)w * NV : nullptr, kp_scale ? kp_scale[w] : 1.f);
  int ncon = min(s.ncon, MAXCON);
  for (int i = lane; i < NV; i += 32) { g.qsm[i] = s.qsm[i]; g.qacc0[i] = s.qacc0[i]; }
  for (int c = lane; c < ncon; c += 32) {
    g.con_pair[c] = s.con_pair[c]; g.con_dim[c] = s.con_dim[c]; g.con_row[c] = s.con_row[c];
    g.con_nrow[c] = s.con_nrow[c]; g.con_nd[c] = s.con_nd[c]; g.con_mu[c] = s.con_mu[c];
  }
  if (lane == 0) {
    g.ncon = s.ncon; g.nefc = s.nefc; g.nefc_dof = s.nefc_dof; g.unsupported = s.unsupported;
    g.min_nu_bits = s.min_nu_bits;
  }
  if (k == nsub - 1)  // contacts of the last substep's collision: pair, dist, pos, normal
    for (int c = lane; c < MAXCON; c += 32) {
      float* o = ocon + ((size_t)w * MAXCON + c) * 8;
      bool v = c < ncon;
      o[0] = v ? (float)s.con_pair[c] : -1.f; o[1] = v ? s.con_dist[c] : 0.f;
      for (int t = 0; t < 3; t++) { o[2 + t] = v ? s.con_pos[c][t] : 0.f; o[5 + t] = v ? s.con_frame[c][t] : 0.f; }
    }
#ifdef DK_PROFILE
  __syncwarp();
  if (lane < NPROF) atomicAdd(&g_prof[lane], (unsigned long long)s.prof[lane]);
#endif
  __syncwarp();
}

template <class S>  // the post half of substep k for world w on its state s (SimN in shared memory, or Hand::big)
__device__ __forceinline__ void post_world_on(S& s, const Hand& g, int w, int lane, const float* qpos,
                                              const float* qvel, const float* warm, int iterations, int k, int nsub,
                                              float* oqpos, float* oqvel, float* owarm, int* info, float* tape) {
  const int ncon = min(g.ncon, MAXCON), nefc = g.nefc;
  float *q = oqpos + (size_t)w * NQ, *v = oqvel + (size_t)w * NV;  // implicitfast updates the outputs in place
  if (k == 0) {  // (from substep 1 on, the inputs are the outputs)
    for (int i = lane; i < NQ; i += 32) q[i] = qpos[(size_t)w * NQ + i];
    for (int i = lane; i < NV; i += 32) v[i] = qvel[(size_t)w * NV + i];
  }
  for (int i = lane; i < NV; i += 32) s.qacc[i] = warm[(size_t)w * NV + i];
  for (int i = lane; i < NTRI; i += 32) s.M[i] = g.M[i];
  for (int r = lane; r < nefc; r += 32) {
    s.efc_kind[r] = g.efc_kind[r]; s.efc_a[r] = g.efc_a[r]; s.efc_b[r] = g.efc_b[r]; s.efc_D[r] = g.efc_D[r];
  }
  for (int c = lane; c < ncon; c += 32) {
    s.con_dim[c] = g.con_dim[c]; s.con_row[c] = g.con_row[c]; s.con_nrow[c] = g.con_nrow[c]; s.con_nd[c] = g.con_nd[c];
  }
  if (lane == 0) {
    s.ncon = g.ncon; s.nefc = nefc; s.nefc_dof = g.nefc_dof; s.unsupported = g.unsupported;
    s.min_nu_bits = g.min_nu_bits;
    s.qpos = q; s.qvel = v; s.qsm = g.qsm; s.qacc0 = g.qacc0; s.con_mu = g.con_mu; s.efc_aref = g.efc_aref;
    s.con_pair = g.con_pair; s.con_J = g.con_J; s.con_dof = g.con_dof; s.con_mask = g.con_mask;
  }
#ifdef DK_PROFILE
  if (lane < NPROF) s.prof[lane] = 0;
#endif
  __syncwarp();
  int nit = 0;
  substep_post(s, iterations, lane, &nit, tape ? tape + ((size_t)w * nsub + k) * REC_WORDS : nullptr);
  for (int i = lane; i < NV; i += 32) owarm[(size_t)w * NV + i] = s.qacc[i];
  if (k == nsub - 1 && lane == 0) {
    info[w * 4] = nit; info[w * 4 + 1] = s.ncon; info[w * 4 + 2] = s.nefc; info[w * 4 + 3] = s.unsupported;
  }
#ifdef DK_PROFILE
  __syncwarp();
  if (lane < NPROF) atomicAdd(&g_prof[lane], (unsigned long long)s.prof[lane]);
#endif
  __syncwarp();
}

// the post half: on SimN in shared memory (smem) when the world's contacts and rows fit, else on Hand::big
__device__ __forceinline__ void post_world(unsigned char* smem, Hand& g, int w, int lane, const float* qpos,
                                           const float* qvel, const float* warm, int iterations, int k, int nsub,
                                           float* oqpos, float* oqvel, float* owarm, int* info, float* tape) {
#ifndef DK_SPLIT_ALL_BIG  // (debug: every world on the full-capacity state in global memory)
  if (min(g.ncon, MAXCON) <= SPLIT_NCON && g.nefc <= SPLIT_NEFC)
    post_world_on(*reinterpret_cast<SimN*>(smem), g, w, lane, qpos, qvel, warm, iterations, k, nsub, oqpos, oqvel,
                  owarm, info, tape);
  else
#endif
    post_world_on(g.big, g, w, lane, qpos, qvel, warm, iterations, k, nsub, oqpos, oqvel, owarm, info, tape);
}

// DK_SPLIT_ORDER's ordering memory (after the Hand records): counters cnt[SPLIT_MAXSUB][SPLIT_NBUCKET] (zero at the
// call), then order[2][SPLIT_NBUCKET][nw]: substep k's worlds of bucket b (most rows first) at order[k & 1][b]
#if DK_SPLIT_ORDER
// block i's world in bucket order (cnt: one substep's counters, order: its lists; all of k_pre(k) has run), on the
// whole warp: lane l sums the counters of buckets l * PER .. l * PER + PER - 1, a shuffle scan gives the lane whose
// range holds i and that lane the bucket (a serial walk over 64 buckets on lane 0 would cost each block up to 64
// dependent loads)
__device__ __forceinline__ int dk_order_world(const int* cnt, const int* order, int nw, int i, int lane) {
  constexpr int PER = (SPLIT_NBUCKET + 31) / 32;
  int c[PER], s = 0;
#pragma unroll
  for (int j = 0; j < PER; j++) {
    const int b = lane * PER + j;
    c[j] = b < SPLIT_NBUCKET ? cnt[b] : 0;
    s += c[j];
  }
  int incl = s;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int t = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += t;
  }
  const int excl = incl - s;
  const unsigned m = __ballot_sync(0xffffffffu, i >= excl && i < incl);  // one lane (the counters sum to nw > i)
  const int src = __ffs(m) - 1;
  int b = lane * PER + PER - 1, r = i - excl;
  bool found = false;
#pragma unroll
  for (int j = 0; j < PER; j++) {  // the lane's first bucket j with r < c[0] + ... + c[j]
    if (!found && r < c[j]) { b = lane * PER + j; found = true; }
    else if (!found) r -= c[j];
  }
  b = __shfl_sync(0xffffffffu, b, src);
  r = __shfl_sync(0xffffffffu, r, src);
  return order[(size_t)b * nw + r];
}
#endif
__global__ void SPLIT_PRE_BOUNDS k_pre(int nw, const float* qpos, const float* qvel, const float* ctrl,
                                       const float* fric, const float* com, int k, int nsub, Hand* work, float* ocon,
                                       float* tape, const float* qfrc = nullptr, const float* kp_scale = nullptr) {
  extern __shared__ __align__(16) unsigned char dk_smem[];
  const int lane = threadIdx.x & 31, w = blockIdx.x;  // (& 31: see k_substep)
  if (w >= nw) return;
  pre_world(*reinterpret_cast<SimP*>(dk_smem), work[w], w, lane, qpos, qvel, ctrl, fric, com, k, nsub, ocon, tape,
            qfrc, kp_scale);
#if DK_SPLIT_ORDER
  if (lane == 0) {
    int* cnt = reinterpret_cast<int*>(work + nw);
    int* order = cnt + SPLIT_MAXSUB * SPLIT_NBUCKET + (size_t)(k & 1) * SPLIT_NBUCKET * nw;
    int b = SPLIT_NBUCKET - 1 - min(work[w].nefc >> SPLIT_ROW_SHIFT, SPLIT_NBUCKET - 1);
    order[(size_t)b * nw + atomicAdd(cnt + k * SPLIT_NBUCKET + b, 1)] = w;
  }
#endif
}

__global__ void SPLIT_POST_BOUNDS k_post(int nw, const float* qpos, const float* qvel, const float* warm,
                                         int iterations, int k, int nsub, Hand* work, float* oqpos, float* oqvel,
                                         float* owarm, int* info, float* tape) {
  extern __shared__ __align__(16) unsigned char dk_smem[];
  const int lane = threadIdx.x & 31;  // (& 31: see k_substep)
  int w = blockIdx.x;
  if (w >= nw) return;
#if DK_SPLIT_ORDER
  {  // block w takes the w-th world in bucket order
    const int* cnt = reinterpret_cast<const int*>(work + nw);
    w = dk_order_world(cnt + k * SPLIT_NBUCKET,
                       cnt + SPLIT_MAXSUB * SPLIT_NBUCKET + (size_t)(k & 1) * SPLIT_NBUCKET * nw, nw, w, lane);
  }
#endif
  post_world(dk_smem, work[w], w, lane, qpos, qvel, warm, iterations, k, nsub, oqpos, oqvel, owarm, info, tape);
}

// work memory of dk_forward_split for nw worlds: nw records of struct Hand (and DK_SPLIT_ORDER's ordering memory)
static inline size_t dk_work_bytes(int nw) {
  return (size_t)nw * sizeof(Hand)
         + (DK_SPLIT_ORDER ? (SPLIT_MAXSUB + (size_t)2 * nw) * SPLIT_NBUCKET * sizeof(int) : 0);
}
static int dk_split_only = 0;  // timing experiments (engine_step_check.py): 1 / 2 launches only k_pre / k_post, on the
                              // state the previous launches left (the results are not a forward then)
// one control step (k_substep's arguments) on stream st; work: dk_work_bytes(nw) bytes (16-byte aligned); smem_pre /
// smem_post: the kernels' dynamic shared memory (at least sizeof(SimP) / sizeof(SimN); more caps the worlds per SM,
// and above 48 KB needs the cudaFuncAttributeMaxDynamicSharedMemorySize opt-in)
static inline cudaError_t dk_forward_split(cudaStream_t st, int nw, const float* qpos, const float* qvel,
                                           const float* warm, const float* ctrl, const float* fric, const float* com,
                                           int iterations, int nsub, float* oqpos, float* oqvel, float* owarm,
                                           int* info, float* ocon, float* tape, void* work,
                                           size_t smem_pre = sizeof(SimP), size_t smem_post = sizeof(SimN),
                                           const float* qfrc = nullptr, const float* kp_scale = nullptr) {
  Hand* hand = reinterpret_cast<Hand*>(work);
#if DK_SPLIT_ORDER
  if (nsub > SPLIT_MAXSUB) return cudaErrorInvalidValue;
  if (dk_split_only != 2)  // (k_post alone reuses the previous launches' order)
    cudaMemsetAsync(hand + nw, 0, SPLIT_MAXSUB * SPLIT_NBUCKET * sizeof(int), st);
#endif
  for (int k = 0; k < nsub; k++) {
    if (dk_split_only != 2)
      k_pre<<<nw, 32, smem_pre, st>>>(nw, k ? oqpos : qpos, k ? oqvel : qvel, ctrl, fric, com, k, nsub, hand, ocon,
                                      tape, qfrc, kp_scale);
    if (dk_split_only != 1)
      k_post<<<nw, 32, smem_post, st>>>(nw, k ? oqpos : qpos, k ? oqvel : qvel, k ? owarm : warm, iterations, k, nsub,
                                        hand, oqpos, oqvel, owarm, info, tape);
  }
  return cudaGetLastError();
}
"""


def loader(layouts, sizes):
    """C++ statements copying the packed constants (float32 array at the host pointer p, advanced as they are read)
    into the DKC symbols of layouts [(symbol, kind, count, width)] with sizes {unit: n}: kind "i" int, "u" unsigned 32-bit
    as two 16-bit halves, "q" unsigned 64-bit as four 16-bit words, else float."""
    lines, tmp = [], 0
    for sym, kind, cnt, width in layouts:
        size = (sizes[cnt] if isinstance(cnt, str) else cnt) * width
        if kind == "i":
            lines.append(f"  {{ static int t{tmp}[{size}]; for (int i = 0; i < {size}; i++) t{tmp}[i] = (int)p[i]; "
                         f"cudaMemcpyToSymbol({sym}, t{tmp}, {size * 4}); p += {size}; }}")
            tmp += 1
        elif kind == "u":  # unsigned 32-bit values packed as (low 16 bits, high 16 bits)
            n = size
            lines.append(f"  {{ static unsigned int t{tmp}[{n}]; for (int i = 0; i < {n}; i++) "
                         f"t{tmp}[i] = (unsigned int)p[2 * i] | ((unsigned int)p[2 * i + 1] << 16); "
                         f"cudaMemcpyToSymbol({sym}, t{tmp}, {n * 4}); p += {2 * n}; }}")
            tmp += 1
        elif kind == "q":
            n = size
            value = " | ".join(f"((unsigned long long)p[4 * i + {k}] << {16*k})" for k in range(4))
            lines.append(f"  {{ static unsigned long long t{tmp}[{n}]; for (int i = 0; i < {n}; i++) "
                         f"t{tmp}[i] = {value}; cudaMemcpyToSymbol({sym}, t{tmp}, {n * 8}); p += {4 * n}; }}")
            tmp += 1
        else:
            lines.append(f"  cudaMemcpyToSymbol({sym}, p, {size * 4}); p += {size};")
    return "\n".join(lines)
