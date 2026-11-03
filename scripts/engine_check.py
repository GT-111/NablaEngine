"""Check (and time) the G1 engine's smooth-dynamics stage against MuJoCo on corpus states.

One warp per world (nablaengine.g1_engine.source): forward kinematics, inverse dynamics qfrc = M(q) qacc + c(q, qvel), and
the dense joint-space mass matrix M (composite rigid bodies, with armature). Compared with MuJoCo's mj_rne(flg_acc=1)
and mj_crb + mj_fullM on --check worlds; timed against MuJoCo Warp's kinematics + com_pos + crb + com_vel +
rne(flg_acc) on the same states. Timings are only indicative when the GPU is shared with another job.

  PYTHONPATH=src python scripts/engine_check.py --out runs/engine_smooth.json
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from nablaengine import fixtures, provenance, workdir
from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine.source import DEVICE, const_layout, constants, header


KERNEL = r"""
__global__ void k_smooth(int nw, const float* __restrict__ qpos, const float* __restrict__ qvel,
                         const float* __restrict__ ctrl, const float* __restrict__ com, float* qfrc, float* Mout,
                         float* qacc_out) {
  int lane = threadIdx.x & 31, g = threadIdx.x >> 5, w = blockIdx.x * WPB + g;
  __shared__ float sQ[WPB][NB][4], sX[WPB][NB][3], sW[WPB][NB][3], sV[WPB][NB][3], sAL[WPB][NB][3], sA[WPB][NB][3],
      sF[WPB][NB][3], sN[WPB][NB][3], sJ[WPB][NB][10], sM[WPB][NV * NV], sq[WPB][NQ + 2 * NV], sb[WPB][NV];
  if (w >= nw) return;
  for (int i = lane; i < NQ; i += 32) sq[g][i] = qpos[w * NQ + i];
  for (int i = lane; i < NV; i += 32) { sq[g][NQ + i] = qvel[w * NV + i]; sq[g][NQ + NV + i] = 0.f; }
  for (int i = lane; i < NV * NV; i += 32) sM[g][i] = 0.f;
  if (lane == 0) {
    sQ[g][0][0] = 1.f; sQ[g][0][1] = sQ[g][0][2] = sQ[g][0][3] = 0.f;
    for (int k = 0; k < 3; k++) sX[g][0][k] = sW[g][0][k] = sV[g][0][k] = sAL[g][0][k] = sA[g][0][k] = 0.f;
  }
  __syncwarp();
  int b = lane, d = (b < NB) ? c_depth[b] : -1, p = (b < NB) ? c_parent[b] : 0;
  for (int L = 1; L <= MAXD; L++) {
    if (d == L)
      fwd_body(b, sQ[g][p], sX[g][p], sW[g][p], sV[g][p], sAL[g][p], sA[g][p], sq[g], sq[g] + NQ, sq[g] + NQ + NV,
               sQ[g][b], sX[g][b], sW[g][b], sV[g][b], sAL[g][b], sA[g][b]);
    __syncwarp();
  }
  const float* O = sX[g][1];
  if (b >= 1 && b < NB) {
    float ip[3];
    body_ipos(b, com + 3 * w, ip);
    body_force(b, sQ[g][b], sW[g][b], sAL[g][b], sA[g][b], ip, sF[g][b], sN[g][b]);
    body_inertia(b, sQ[g][b], sX[g][b], O, ip, sJ[g][b]);
  }
  __syncwarp();
  for (int L = MAXD - 1; L >= 1; L--) {
    if (d == L)
      for (int i = 0; i < c_nchild[b]; i++) {
        int c = c_child[MAX_CHILD * b + i];
        accumulate(sX[g][b], sX[g][c], sF[g][c], sN[g][c], sF[g][b], sN[g][b]);
        for (int k = 0; k < 10; k++) sJ[g][b][k] += sJ[g][c][k];
      }
    __syncwarp();
  }
  if (b >= 1 && b < NB) torques(b, sQ[g][b], sF[g][b], sN[g][b], sb[g]);
  float sw[3], sv[3], kw[3], kv[3], n[3], f[3];
  if (b >= 2 && b < NB) {  // row j of this body's dof, up to the diagonal: every ancestor dof and its own
    int j = c_dadr[b];
    dof_motion(b, 0, sQ[g][b], sX[g][b], O, sw, sv);
    apply_inertia(sJ[g][b], sw, sv, n, f);
    sM[g][j * NV + j] = dot3(sw, n) + dot3(sv, f) + c_armature[j];
    for (int a = 0; a < c_nanc[b]; a++) {
      int k = c_anc[12 * b + a];
      dof_motion(k, 0, sQ[g][k], sX[g][k], O, kw, kv);
      sM[g][j * NV + c_dadr[k]] = dot3(kw, n) + dot3(kv, f);
    }
    for (int t = 0; t < 6; t++) {
      dof_motion(1, t, sQ[g][1], sX[g][1], O, kw, kv);
      sM[g][j * NV + t] = dot3(kw, n) + dot3(kv, f);
    }
  }
  if (b == 1)
    for (int j = 0; j < 6; j++) {
      dof_motion(1, j, sQ[g][1], sX[g][1], O, sw, sv);
      apply_inertia(sJ[g][1], sw, sv, n, f);
      for (int i = 0; i <= j; i++) {
        dof_motion(1, i, sQ[g][1], sX[g][1], O, kw, kv);
        sM[g][j * NV + i] = dot3(kw, n) + dot3(kv, f) + (i == j ? c_armature[j] : 0.f);
      }
    }
  __syncwarp();
  for (int e = lane; e < NV * NV; e += 32) {  // mirror the lower triangle (row j, columns < j) to the upper
    int r = e / NV, c = e % NV;
    if (c > r) sM[g][e] = sM[g][c * NV + r];
  }
  __syncwarp();
  for (int i = lane; i < NV * NV; i += 32) Mout[(size_t)w * NV * NV + i] = sM[g][i];
  for (int i = lane; i < NV; i += 32) {  // qfrc_smooth = clamped actuator force - bias
    float bias = sb[g][i];
    qfrc[w * NV + i] = bias;
    float act = 0.f;
    if (c_actid[i] >= 0) {
      int j = c_actid[i];
      act = c_kp[i] * (ctrl[w * (NU) + j] - sq[g][i + 1]) - c_kv[i] * sq[g][NQ + i];
      act = fminf(fmaxf(act, -c_frclim[i]), c_frclim[i]);
    }
    sb[g][i] = act - c_damping[i] * sq[g][NQ + i] - bias;
  }
  __syncwarp();
  chol_warp(sM[g], lane);
  chol_solve_warp(sM[g], sb[g], lane);
  for (int i = lane; i < NV; i += 32) qacc_out[w * NV + i] = sb[g][i];
}
"""

HOST = r"""
#include <cstdio>
#include <cstdlib>
#include <vector>
static std::vector<float> readf(const char* path, size_t n) {
  std::vector<float> v(n);
  FILE* f = fopen(path, "rb");
  if (!f || fread(v.data(), 4, n, f) != n) { fprintf(stderr, "read %s\n", path); exit(1); }
  fclose(f);
  return v;
}
static void writef(const char* path, const float* v, size_t n) { FILE* f = fopen(path, "wb"); fwrite(v, 4, n, f); fclose(f); }
int main(int argc, char** argv) {  // dir nworld reps
  const char* dir = argv[1];
  int nw = atoi(argv[2]), reps = atoi(argv[3]);
  char path[1024];
  snprintf(path, sizeof path, "%s/consts.bin", dir);
  std::vector<float> cf = readf(path, NCONST);
  const float* p = cf.data();
LOAD_CONSTS
  snprintf(path, sizeof path, "%s/qpos.bin", dir); std::vector<float> hq = readf(path, (size_t)nw * NQ);
  snprintf(path, sizeof path, "%s/qvel.bin", dir); std::vector<float> hv = readf(path, (size_t)nw * NV);
  snprintf(path, sizeof path, "%s/ctrl.bin", dir); std::vector<float> hc = readf(path, (size_t)nw * (NU));
  snprintf(path, sizeof path, "%s/com.bin", dir); std::vector<float> ho = readf(path, (size_t)nw * 3);
  float *dq, *dv, *dc, *dcom, *df, *dM, *da;
  cudaMalloc(&dq, hq.size() * 4); cudaMalloc(&dv, hv.size() * 4); cudaMalloc(&dc, hc.size() * 4);
  cudaMalloc(&dcom, ho.size() * 4); cudaMalloc(&df, (size_t)nw * NV * 4); cudaMalloc(&dM, (size_t)nw * NV * NV * 4);
  cudaMalloc(&da, (size_t)nw * NV * 4);
  cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dc, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dcom, ho.data(), ho.size() * 4, cudaMemcpyHostToDevice);
  int grid = (nw + WPB - 1) / WPB;
  for (int i = 0; i < 3; i++) k_smooth<<<grid, WPB * 32>>>(nw, dq, dv, dc, dcom, df, dM, da);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < reps; i++) k_smooth<<<grid, WPB * 32>>>(nw, dq, dv, dc, dcom, df, dM, da);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  printf("{\"nworld\": %d, \"ms\": %.5f, \"err\": \"%s\"}\n", nw, ms / reps, cudaGetErrorString(cudaGetLastError()));
  std::vector<float> of((size_t)nw * NV), oM((size_t)nw * NV * NV), oa((size_t)nw * NV);
  cudaMemcpy(of.data(), df, of.size() * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(oM.data(), dM, oM.size() * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(oa.data(), da, oa.size() * 4, cudaMemcpyDeviceToHost);
  snprintf(path, sizeof path, "%s/qfrc.bin", dir); writef(path, of.data(), of.size());
  snprintf(path, sizeof path, "%s/M.bin", dir); writef(path, oM.data(), oM.size());
  snprintf(path, sizeof path, "%s/qacc.bin", dir); writef(path, oa.data(), oa.size());
  return 0;
}
"""


def load_consts_code(m):
    n = {"nb": m.nbody, "nv": m.nv, "one": 1}
    lines, tmp = [], 0
    for sym, kind, cnt, width in const_layout(m):
        size = n[cnt] * width
        if kind == "i":
            lines.append(f"  {{ static int t{tmp}[{size}]; for (int i = 0; i < {size}; i++) t{tmp}[i] = (int)p[i]; "
                         f"cudaMemcpyToSymbol({sym}, t{tmp}, {size * 4}); p += {size}; }}")
            tmp += 1
        else:
            lines.append(f"  cudaMemcpyToSymbol({sym}, p, {size * 4}); p += {size};")
    return "\n".join(lines)


def mjw_baseline(m, qpos, qvel, qacc, reps):
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src import smooth
    wp.init()
    mw, dw = mjw.put_model(m), mjw.make_data(m, nworld=len(qpos))
    dw.qpos.assign(qpos.astype(np.float32))
    dw.qvel.assign(qvel.astype(np.float32))
    dw.qacc.assign(qacc.astype(np.float32))

    def step():
        mjw.kinematics(mw, dw)
        mjw.com_pos(mw, dw)
        smooth.crb(mw, dw)
        mjw.com_vel(mw, dw)
        mjw.rne(mw, dw, flg_acc=True)

    step()
    with wp.ScopedCapture() as cap:
        step()
    for _ in range(3):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None, help="corpus name in the fixtures directory (fixtures.corpora())")
    ap.add_argument("--nworld", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--wpb", type=int, default=4)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--check", type=int, default=256)
    ap.add_argument("--out", required=True)
    workdir.add_argument(ap)
    args = ap.parse_args()
    workdir.configure(args)
    args.corpus = args.corpus or fixtures.default_corpus()
    m, mt = fixtures.load_model()
    consts, maxd = constants(m)
    work = workdir.work_dir("engine_check_")
    src = (header(m, maxd, args.wpb) + DEVICE + KERNEL
           + HOST.replace("LOAD_CONSTS", load_consts_code(m)).replace("NCONST", str(len(consts))))
    (work / "engine.cu").write_text(src)
    compile_cuda(work / "engine.cu", work / "engine")
    consts.tofile(work / "consts.bin")
    z = fixtures.load_corpus(args.corpus)
    res = {"corpus": args.corpus, "wpb": args.wpb, "runs": []}
    for nw in args.nworld:
        rep = -(-nw // len(z["qpos"]))
        qpos, qvel, ctrl, com = (np.tile(z[k], (rep, 1))[:nw] for k in ("qpos", "qvel", "ctrl", "com"))
        qacc = np.tile(z["qacc_warmstart"], (rep, 1))[:nw]
        for name, a in (("qpos", qpos), ("qvel", qvel), ("ctrl", ctrl), ("com", com)):
            a.astype(np.float32).tofile(work / f"{name}.bin")
        out = subprocess.run([str(work / "engine"), str(work), str(nw), str(args.reps)], check=True,
                             stdout=subprocess.PIPE, universal_newlines=True).stdout
        entry = json.loads(next(line for line in out.splitlines() if line.startswith("{")))
        if nw == args.nworld[0]:
            d = mujoco.MjData(m)
            K = args.check
            qf = np.fromfile(work / "qfrc.bin", np.float32).reshape(nw, m.nv)[:K]
            Me = np.fromfile(work / "M.bin", np.float32).reshape(nw, m.nv, m.nv)[:K]
            qa = np.fromfile(work / "qacc.bin", np.float32).reshape(nw, m.nv)[:K]
            rf, rM, ra = np.zeros((K, m.nv)), np.zeros((K, m.nv, m.nv)), np.zeros((K, m.nv))
            torso = fixtures.torso_id(m, mt)
            ipos0 = m.body_ipos[torso].copy()
            for i in range(K):
                m.body_ipos[torso] = ipos0 + com[i]
                d.qpos[:], d.qvel[:], d.ctrl[:] = qpos[i], qvel[i], ctrl[i]
                mujoco.mj_forward(m, d)
                mujoco.mj_fullM(m, d, rM[i])
                rf[i], ra[i] = d.qfrc_bias, d.qacc_smooth
            m.body_ipos[torso] = ipos0
            entry["bias_rel_err"] = float(np.linalg.norm(qf - rf) / np.linalg.norm(rf))
            entry["qacc_smooth_rel_err"] = float(np.linalg.norm(qa - ra) / np.linalg.norm(ra))
            entry["qacc_smooth_p99_rel_row"] = float(np.quantile(np.linalg.norm(qa - ra, axis=1)
                                                                / np.linalg.norm(ra, axis=1), 0.99))
            entry["M_rel_err"] = float(np.linalg.norm(Me - rM) / np.linalg.norm(rM))
            entry["M_max_abs"] = float(np.abs(Me - rM).max())
            entry["M_worst_entry"] = [int(x) for x in np.unravel_index(np.abs(Me - rM).max(0).argmax(), (m.nv, m.nv))]
        try:
            entry["mujoco_warp_ms"] = mjw_baseline(m, qpos, qvel, qacc, args.reps)
        except Exception as e:  # noqa: BLE001 - the baseline is optional
            entry["mujoco_warp_error"] = repr(e)[:300]
        res["runs"].append(entry)
        print(json.dumps(entry), flush=True)
        if "provenance" not in res:
            res["provenance"] = provenance.provenance()  # commit, GPU, driver, nvcc, packages, MuJoCo Warp
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
