"""Shared code of the verification scripts: the adjoint check binary (the forward kernel k_substep, the tape and the
analytic adjoint k_adjoint with a small host program) and the digest manifest of the forward outputs.

Adjoint check binary (scripts/jax_ffi_check.py, scripts/identity_gate.py, scripts/paper_bench/ours_bare.py):
  build(m, mt, work, tag, defines)   compile it with -D options (e.g. DK_ADJ_CLAMP_KV=1, DK_ADJ_NORMAL=0); where the
                                     forward unit has architecture flags of its own (sm_120: from compute_90 PTX) the
                                     same source is also built as the forward unit and run() takes the forward and
                                     the tape from that build
  write_inputs(work, corpus_inputs(z))   the corpus's inputs and its three output cotangents as the binary's files
  run(binary, work, m, nw, iterations, nsub, ncot, reps, debug, timeout)   (its JSON line, outputs: tape, info, con,
                                     qpos, qvel, grad{c}_qpos / qvel / ctrl, adj_info{c})

Forward digests: a committed manifest (DIGESTS, fixtures/mjlab/engine_digests.json; $DK_DIGESTS names a further
manifest, e.g. a local one for a toolchain without committed entries) of sha256 digests of the forward outputs of
scripts/engine_step_check.py, keyed by the corpus (name and input digest), the model file, the run (substeps,
iterations, worlds, world selection, tape, fused or split forward) and the build (target architecture, nvcc release,
the forward unit's architecture flags and -D options). digest_check: "match" (an entry with this key, every digest
equal), "mismatch" (an entry with this key, a digest different: the forward changed) or "unanchored" (no entry for
this key, e.g. another GPU architecture or nvcc release: bit identity with the released forward is not checked there,
the tolerance gates against MuJoCo Warp and MuJoCo C still apply). Comments in the generated CUDA source do not
change the outputs, so the source itself is not part of the key (its sha256 is recorded)."""

import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from nablaengine import cuda, provenance
from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine import adjoint, step, tape
from nablaengine.g1_engine.host import KERNEL, loader

CHECKOUT = Path(__file__).resolve().parents[2]
DIGESTS = CHECKOUT / "fixtures" / "mjlab" / "engine_digests.json"
DIGESTS_ENV = "DK_DIGESTS"
DIGESTS_FORMAT = "dk-engine-digests/1"

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
template <class T> static void writef(const char* path, const T* v, size_t n) { FILE* f = fopen(path, "wb"); fwrite(v, sizeof(T), n, f); fclose(f); }
static float* up(const std::vector<float>& h) { float* d; cudaMalloc(&d, h.size() * 4); cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice); return d; }
template <class T> static void down(const char* dir, const char* name, const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  char path[1024];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  writef(path, h.data(), n);
}
int main(int argc, char** argv) {  // dir nworld iterations substeps ncot reps debug [worlds per SM [carveout [delta]]]
  const char* dir = argv[1];
  int nw = atoi(argv[2]), iters = atoi(argv[3]), nsub = atoi(argv[4]), ncot = atoi(argv[5]), reps = atoi(argv[6]);
  bool dbg_on = atoi(argv[7]) > 0;
  // k_adjoint's launch (adjoint.py adj_launch_smem): the production choice for this GPU (-1), or at most argv[8] > 0
  // worlds per SM (more shared memory than it needs); 0 is a failure
  cudaError_t err_a;
  const size_t smem_a = adj_launch_smem(argc > 8 && atoi(argv[8]) > 0 ? atoi(argv[8]) : -1, &err_a);
  if (!smem_a) { fprintf(stderr, "adj_launch_smem: %s\n", cudaGetErrorString(err_a)); return 1; }
  char path[1024], name[256];
  snprintf(path, sizeof path, "%s/consts.bin", dir);
  std::vector<float> cf = readf(path, NCONST);
  const float* p = cf.data();
LOAD_CONSTS
#define IN(name, n) snprintf(path, sizeof path, "%s/" #name ".bin", dir); float* d_##name = up(readf(path, (size_t)nw * (n)));
  IN(qpos, NQ) IN(qvel, NV) IN(warm, NV) IN(ctrl, NU) IN(fric, 1) IN(com, 3)
#if DK_ADJ_CLAMP_KV == 2
  // the soft clamp (adjoint.py): its delta (argv[10], default 0: the gradients of DK_ADJ_CLAMP_KV=1) and the ctrl the
  // adjoint then always needs
  const float clamp_soft = argc > 10 ? (float)atof(argv[10]) : 0.f;
#define ADJ_CTRL(on) d_ctrl
#else
#define ADJ_CTRL(on) ((on) ? d_ctrl : nullptr)
#endif
  float *oq, *ov, *ow, *oc, *tp; int* oi;
  cudaMalloc(&oc, (size_t)nw * MAXCON * 8 * 4);
  cudaMalloc(&oq, (size_t)nw * NQ * 4); cudaMalloc(&ov, (size_t)nw * NV * 4); cudaMalloc(&ow, (size_t)nw * NV * 4);
  cudaMalloc(&oi, (size_t)nw * 16);
  size_t ntape = (size_t)nw * nsub * REC_WORDS;
  cudaMalloc(&tp, ntape * 4);
  cudaFuncSetAttribute(k_substep, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Sim));
  if (argc > 9 && atoi(argv[9]) >= 0)  // shared-memory carveout preference of k_adjoint, percent (the rest is L1)
    cudaFuncSetAttribute(k_adjoint, cudaFuncAttributePreferredSharedMemoryCarveout, atoi(argv[9]));
  int nblk_f = 0, nblk_a = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk_f, k_substep, 32, sizeof(Sim));
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk_a, k_adjoint, 32, smem_a);
  int nprof_f = 0, nprof_a = 0;  // launches since g_prof / g_aprof were zeroed (-DDK_PROFILE: printed with them)
  auto fwd = [&]() {
    k_substep<<<nw, 32, sizeof(Sim)>>>(nw, d_qpos, d_qvel, d_warm, d_ctrl, d_fric, d_com, iters, nsub, oq, ov, ow, oi, oc, tp);
    nprof_f++;
  };
#ifdef DK_PROFILE
  // g_prof and g_aprof sum over every launch since they were last zeroed: each is zeroed before the launch it profiles
  // (the first forward, the first adjoint) and read right after, with the launches since printed (run() requires 1:
  // a reading moved after the timing loop would sum over its launches)
  static const unsigned long long prof_zero[NPROF > NAPROF ? NPROF : NAPROF] = {};
  cudaMemcpyToSymbol(g_prof, prof_zero, sizeof(unsigned long long) * NPROF);
  nprof_f = 0;
#endif
  fwd();
  cudaDeviceSynchronize();
  if (const char* tin = getenv("DK_TAPE_IN")) {  // the adjoint on another binary's tape (run(): the forward unit's build, as on sm_120)
    std::vector<float> ht = readf(tin, ntape);
    cudaMemcpy(tp, ht.data(), ntape * 4, cudaMemcpyHostToDevice);
  }
  down(dir, "out_qpos.bin", oq, (size_t)nw * NQ); down(dir, "out_qvel.bin", ov, (size_t)nw * NV);
  down(dir, "out_warm.bin", ow, (size_t)nw * NV); down(dir, "out_info.bin", oi, (size_t)nw * 4);
  down(dir, "out_con.bin", oc, (size_t)nw * MAXCON * 8);
  down(dir, "tape.bin", tp, ntape);
  float *gq, *gv, *gu, *dbg = nullptr; int* ai;
  cudaMalloc(&gq, (size_t)nw * NQ * 4); cudaMalloc(&gv, (size_t)nw * NV * 4); cudaMalloc(&gu, (size_t)nw * (NU) * 4);
  cudaMalloc(&ai, (size_t)nw * 4);
  void* adj_work = nullptr;
  if (const size_t bytes = adj_work_bytes(nw)) cudaMalloc(&adj_work, bytes);
  size_t ndbg = (size_t)nw * nsub * DBG_WORDS;
  if (dbg_on) cudaMalloc(&dbg, ndbg * 4);
  float *cq = nullptr, *cv = nullptr;
  for (int c = 0; c < ncot; c++) {
    snprintf(path, sizeof path, "%s/cot%d_qpos.bin", dir, c);
    std::vector<float> hq = readf(path, (size_t)nw * NQ);
    snprintf(path, sizeof path, "%s/cot%d_qvel.bin", dir, c);
    std::vector<float> hv = readf(path, (size_t)nw * NV);
    if (cq) { cudaFree(cq); cudaFree(cv); }
    cq = up(hq); cv = up(hv);
    if (dbg) cudaMemset(dbg, 0, ndbg * 4);
#ifdef DK_PROFILE
    if (c == 0) { cudaMemcpyToSymbol(g_aprof, prof_zero, sizeof(unsigned long long) * NAPROF); nprof_a = 0; }
#endif
    k_adjoint<<<nw, 32, smem_a>>>(nw, nsub, tp, d_fric, d_com, cq, cv, gq, gv, gu, ai, dbg, ADJ_CTRL(dbg)
                                  ADJ_SOFT_ARG(clamp_soft), nullptr, nullptr, nullptr, adj_work);
    nprof_a++;
    cudaDeviceSynchronize();
#ifdef DK_PROFILE
    if (c == 0) {  // -DDK_PROFILE: the phase cycles of the first forward launch and of this first adjoint launch
      unsigned long long hp[NPROF], ha[NAPROF];
      cudaMemcpyFromSymbol(hp, g_prof, sizeof hp);
      cudaMemcpyFromSymbol(ha, g_aprof, sizeof ha);
      printf("{\"prof_launches\": %d, \"adj_prof_launches\": %d, \"prof_cycles\": [", nprof_f, nprof_a);
      for (int k = 0; k < NPROF; k++) printf("%s%llu", k ? ", " : "", hp[k]);
      printf("], \"adj_prof_cycles\": [");
      for (int k = 0; k < NAPROF; k++) printf("%s%llu", k ? ", " : "", ha[k]);
      printf("]}\n");
    }
#endif
    snprintf(name, sizeof name, "grad%d_qpos.bin", c); down(dir, name, gq, (size_t)nw * NQ);
    snprintf(name, sizeof name, "grad%d_qvel.bin", c); down(dir, name, gv, (size_t)nw * NV);
    snprintf(name, sizeof name, "grad%d_ctrl.bin", c); down(dir, name, gu, (size_t)nw * (NU));
    snprintf(name, sizeof name, "adj_info%d.bin", c); down(dir, name, ai, (size_t)nw);
    if (dbg && c == 0) down(dir, "dbg0.bin", dbg, ndbg);  // the intermediates of cotangent 0 (as the dump's)
  }
  float ms_f = 0.f, ms_a = 0.f;
  if (reps > 0) {  // forward with the tape, then the adjoint (cotangent ncot - 1; debug output off), after warm-up launches
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) fwd();
    cudaEventRecord(e0);
    for (int i = 0; i < reps; i++) fwd();
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms_f, e0, e1);
    auto adj = [&]() {
      k_adjoint<<<nw, 32, smem_a>>>(nw, nsub, tp, d_fric, d_com, cq, cv, gq, gv, gu, ai, nullptr, ADJ_CTRL(false)
                                    ADJ_SOFT_ARG(clamp_soft), nullptr, nullptr, nullptr, adj_work);
      nprof_a++;
    };
    for (int i = 0; i < 3; i++) adj();
    cudaEventRecord(e0);
    for (int i = 0; i < reps; i++) adj();
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms_a, e0, e1);
    ms_f /= reps; ms_a /= reps;
  }
  printf("{\"nworld\": %d, \"ms_forward_tape\": %.5f, \"ms_adjoint\": %.5f, \"err\": \"%s\", \"sizeof_sim\": %zu, "
         "\"sizeof_adj\": %zu, \"blocks_per_sm_forward\": %d, \"blocks_per_sm_adjoint\": %d}\n", nw, ms_f, ms_a,
         cudaGetErrorString(cudaGetLastError()), sizeof(Sim), sizeof(Adj), nblk_f, nblk_a);
  return 0;
}
"""


FORWARD_OF = {}  # build(): an adjoint binary -> the same source built as the forward unit, where its flags differ


def binary_source(m, mt):
    """(CUDA source, packed constants) of build()'s binary for model m (the forward and adjoint device code,
    nablaengine.g1_engine.host's KERNEL and HOST's main; scripts/identity_gate.py hashes it)."""
    consts, lays, sizes = adjoint.engine_constants(m, mt["foot_pairs"])
    src = (adjoint.cuda_source(m) + KERNEL
           + HOST.replace("LOAD_CONSTS", loader(lays, sizes)).replace("NCONST", str(len(consts))))
    return src, consts


def binary_builds(target=None):
    """build()'s compilations of one source for build target target (default cuda.arch()): [(file suffix, unit of
    cuda.compile_cuda)]: one as the forward unit where the forward / kinematics unit has the adjoint units' flags,
    else [("", None), ("_fwd", "forward")] (run() takes the forward and tape from the _fwd binary)."""
    split = cuda.arch_flags("forward", target) != cuda.arch_flags(None, target)
    return [("", None), ("_fwd", "forward")] if split else [("", "forward")]


def build(m, mt, work, tag, defines, flags=()):
    """Compile the forward + adjoint binary with the -D definitions (and extra nvcc flags); returns its path. Where the
    forward / kinematics unit has other architecture flags than the adjoint units (nablaengine.cuda.arch_flags: NVVM 7
    on sm_120), the same source is also built with the forward unit's flags (<tag>_fwd, FORWARD_OF): run() takes the
    forward outputs and the tape from it and runs this binary's k_adjoint on that tape, as the FFI library does. The
    binary that runs the forward (this one where the flags are the same, else <tag>_fwd) is compiled as the forward
    unit, so it also gets the target's forward-unit -D options (nablaengine.cuda.unit_defines: the register caps on
    sm_90), as the FFI library's unit with the forward kernels."""
    src, consts = binary_source(m, mt)
    (work / f"{tag}.cu").write_text(src)
    consts.tofile(work / "consts.bin")
    extra = [f"-D{d}" for d in defines] + list(flags)
    for suffix, unit in binary_builds():
        compile_cuda(work / f"{tag}.cu", work / f"{tag}{suffix}", extra, unit=unit)
        if suffix:
            FORWARD_OF[work / tag] = work / f"{tag}{suffix}"
    return work / tag


def run(binary, work, m, nw, iterations, nsub, ncot, reps, debug, timeout, blocks_per_sm=0, carveout=-1,
        clamp_soft=None, tape_in=None):
    """One run: (its JSON line, outputs: tape (nw, nsub, REC_WORDS), info, con (the last substep's contacts), qpos, qvel,
    grad{c}_*, adj_info{c}, and with debug the decoded records of cotangent 0, dbg0). blocks_per_sm > 0 caps the adjoint's
    worlds per SM (it requests more shared memory than it needs), 0 is the production launch (adjoint.py
    adj_launch_smem: on V100 at most 4 of the 6 worlds that fit an SM); carveout >= 0 sets the adjoint's preferred
    shared-memory carveout (percent of the SM's shared memory, the rest is L1).
    clamp_soft: the soft clamp's delta of a DK_ADJ_CLAMP_KV=2 build (float32; the others ignore it; default 0).
    tape_in: a tape file (float32, nw * nsub records) the adjoint runs on instead of the binary's own forward's (which
    still runs: info, con, qpos, qvel are its own; the returned tape is tape_in's). Without tape_in, a binary with a
    forward-unit build (FORWARD_OF) runs on that build's tape, and the forward outputs (the files in work too) and
    ms_forward_tape are that build's."""
    fwd = FORWARD_OF.get(Path(binary))
    if fwd is not None and tape_in is None:
        rf, fo = run(fwd, work, m, nw, iterations, nsub, min(ncot, 1) if reps else 0, reps, False, timeout,
                     clamp_soft=clamp_soft)
        names = ("out_qpos.bin", "out_qvel.bin", "out_warm.bin", "out_info.bin", "out_con.bin")
        keep = {n: (work / n).read_bytes() for n in names}
        tin = work / f"{Path(binary).name}_tape_in.bin"
        os.replace(work / "tape.bin", tin)
        try:
            r, out = run(binary, work, m, nw, iterations, nsub, ncot, reps, debug, timeout, blocks_per_sm, carveout,
                         clamp_soft, tape_in=tin)
        finally:
            tin.unlink(missing_ok=True)
        for n, b in keep.items():  # the forward unit's outputs, as callers also read them from work
            (work / n).write_bytes(b)
        out.update({k: fo[k] for k in ("info", "con", "qpos", "qvel")})
        r.update(ms_forward_tape=rf["ms_forward_tape"], blocks_per_sm_forward=rf["blocks_per_sm_forward"],
                 forward_build=Path(fwd).name)
        return r, out
    soft = [] if clamp_soft is None else [repr(float(np.float32(clamp_soft)))]
    env = dict(os.environ, DK_TAPE_IN=str(tape_in)) if tape_in is not None else None
    o = subprocess.run([str(binary), str(work), str(nw), str(iterations), str(nsub), str(ncot), str(reps),
                        str(int(debug)), str(int(blocks_per_sm)), str(int(carveout))] + soft, check=True,
                       timeout=timeout, stdout=subprocess.PIPE, universal_newlines=True, env=env).stdout
    r = json.loads(next(line for line in o.splitlines() if line.startswith('{"nworld"')))
    if r["err"] != "no error":
        raise RuntimeError(f"CUDA error: {r['err']}")
    prof = [json.loads(line) for line in o.splitlines() if line.startswith('{"prof_launches"')]
    if prof:  # a -DDK_PROFILE build: the phase cycles of the first forward and adjoint launches, each of one launch
        if (prof[0]["prof_launches"], prof[0]["adj_prof_launches"]) != (1, 1):
            raise RuntimeError(f"profile counters summed over several launches: {prof[0]['prof_launches']} forward, "
                               f"{prof[0]['adj_prof_launches']} adjoint")
        r.update(prof[0])
    f = lambda name, dt=np.float32: np.fromfile(work / name, dt)  # noqa: E731
    out = dict(tape=f("tape.bin").reshape(nw, nsub, tape.rec_words(m)), info=f("out_info.bin", np.int32).reshape(nw, 4),
               qpos=f("out_qpos.bin").reshape(nw, m.nq), qvel=f("out_qvel.bin").reshape(nw, m.nv),
               con=f("out_con.bin").reshape(nw, step.maxcon(m), 8))
    for c in range(ncot):
        out[f"grad{c}_qpos"] = f(f"grad{c}_qpos.bin").reshape(nw, m.nq)
        out[f"grad{c}_qvel"] = f(f"grad{c}_qvel.bin").reshape(nw, m.nv)
        out[f"grad{c}_ctrl"] = f(f"grad{c}_ctrl.bin").reshape(nw, m.nu)
        out[f"adj_info{c}"] = f(f"adj_info{c}.bin", np.int32)
        if debug and c == 0:
            out["dbg0"] = adjoint.decode_debug(f("dbg0.bin").reshape(nw, nsub, -1), m)
    return r, out


def write_inputs(work, arrs):
    for k, a in arrs.items():
        np.ascontiguousarray(a, np.float32).tofile(work / f"{k}.bin")


def corpus_inputs(z, q=None):
    """The binary's input arrays from a corpus (qpos replaced by q when given), with its three cotangents."""
    arrs = dict(qpos=z["qpos"] if q is None else q, qvel=z["qvel"], warm=z["qacc_warmstart"], ctrl=z["ctrl"],
                fric=z["friction"], com=z["com"])
    for c in range(3):
        arrs[f"cot{c}_qpos"], arrs[f"cot{c}_qvel"] = z[f"cot{c}_qpos"], z[f"cot{c}_qvel"]
    return arrs


# ---------------------------------------------------------------- forward digests

def array_digest(a):
    """{shape, dtype, sha256} of an array (sha256 of its C-order bytes)."""
    a = np.ascontiguousarray(a)
    return dict(shape=list(a.shape), dtype=a.dtype.str, sha256=hashlib.sha256(a.tobytes()).hexdigest())


def inputs_digest(arrays, names=("qpos", "qvel", "warm", "ctrl", "fric", "com")):
    """sha256 of the float32 inputs as the binaries read them, each preceded by its name."""
    h = hashlib.sha256()
    for k in names:
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrays[k], np.float32).tobytes())
    return h.hexdigest()


def forward_key(corpus, inputs, model_file, run, extra=()):
    """The manifest key of a forward run of scripts/engine_step_check.py: corpus (name, input digest), the model
    file's sha256, run (dict: substeps, iterations, nworld, select, tape, fused) and the build of this machine
    (cuda.arch(), the nvcc release, cuda.arch_flags('forward'), the extra nvcc options and cuda.unit_defines)."""
    extra = list(extra)
    return dict(unit="check_forward", corpus=corpus, inputs_sha256=inputs_digest(inputs),
                model_sha256=hashlib.sha256(Path(model_file).read_bytes()).hexdigest(),
                run={k: run[k] for k in sorted(run)},
                build=dict(arch=cuda.arch(), nvcc=provenance.nvcc_version(), forward_flags=cuda.arch_flags("forward"),
                           options=extra + cuda.unit_defines("forward", None, extra)))


def digest_files():
    """The manifests read by digest_check: DIGESTS, then $DK_DIGESTS when set."""
    out = [DIGESTS]
    if os.environ.get(DIGESTS_ENV):
        out.append(Path(os.environ[DIGESTS_ENV]))
    return out


def _load(path):
    path = Path(path)
    if not path.exists():
        return dict(format=DIGESTS_FORMAT, entries=[])
    d = json.loads(path.read_text())
    if d.get("format") != DIGESTS_FORMAT:
        raise ValueError(f"{path}: format {d.get('format')!r}, expected {DIGESTS_FORMAT}")
    return d


def _same_key(a, b):
    return json.loads(json.dumps(a, sort_keys=True)) == json.loads(json.dumps(b, sort_keys=True))


def digest_check(arrays, key, files=None):
    """The outputs (dict name -> array) against the manifests' entry of this key (module docstring): {status: match /
    mismatch / unanchored, arrays: {name: equal} over the names both have, manifest, note}."""
    for f in files or digest_files():
        for e in _load(f)["entries"]:
            if not _same_key(e["key"], key):
                continue
            common = sorted(set(e["outputs"]) & set(arrays))
            same = {k: e["outputs"][k] == array_digest(arrays[k]) for k in common}
            status = "match" if common and all(same.values()) else "mismatch"
            return dict(status=status, arrays=same, manifest=provenance.rel(f), made=e.get("made"),
                        note=None if common else "no output in common with the entry")
    return dict(status="unanchored", arrays={}, manifest=None,
                note=f"no entry for this corpus, run and build ({key['build']['arch']}, nvcc {key['build']['nvcc']}) "
                     f"in {', '.join(provenance.rel(f) for f in (files or digest_files()))}")


def digest_write(path, arrays, key, source_sha256=None, replace=False, about=None):
    """Add the entry of this key with the outputs' digests to the manifest at path (created if missing); an entry
    with this key and other digests is replaced only with replace=True (ValueError otherwise). Returns the status."""
    path = Path(path)
    d = _load(path)
    d.setdefault("about", about or "sha256 digests of scripts/engine_step_check.py's forward outputs per corpus, run "
                 "and build (nablaengine.checks)")
    outputs = {k: array_digest(v) for k, v in sorted(arrays.items())}
    made = dict(date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"), gpu=provenance.gpu(),
                source_sha256=source_sha256)
    for i, e in enumerate(d["entries"]):
        if _same_key(e["key"], key):
            if e["outputs"] == outputs:
                return "unchanged"
            if not replace:
                raise ValueError(f"{path}: an entry with this key has other digests (the forward changed); "
                                 "replace it explicitly")
            d["entries"][i] = dict(key=key, outputs=outputs, made=made)
            break
    else:
        d["entries"].append(dict(key=key, outputs=outputs, made=made))
    d["entries"].sort(key=lambda e: json.dumps(e["key"], sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    return "written"
