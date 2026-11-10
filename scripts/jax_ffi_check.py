"""The JAX interface of the G1 engine (nablaengine.jax_ffi): acceptance tests and the cost of the FFI calls.

Corpora: --corpora (default mf_s0, mf_s1: benchmarks/data, nablaengine.fixtures), 4096 states each; the sections
that use one corpus take the first.

Sections (--sections, default all; each runs in its own process, so its device memory and timing are its own):
  bitwise    the FFI against the adjoint check binary (nablaengine.checks: build / run, the same engine source) on the
             same inputs, every corpus, 4 substeps, --bitwise-iterations (default 50 and 10), the adjoint variants
             --variants (default (clamp_kv, normal) (1, 0), (0, 3), (1, 3), (0, 0), and the soft clamp (2, 0), (2, 3)
             at clamp_soft CHECK_SOFT (0.2; the FFI's dk_step_bwd_soft with the step's ctrl, the binary's delta
             argument)): the forward (qpos', qvel', warm', info, tape) and the three cotangents' gradients and adjoint
             flags bitwise; dk_step (no tape) equal to dk_step_fwd; the backward repeated: bitwise the same
             (determinism). At 50 iterations the forward also against the forward reference (--forward-ref): by
             default "self", this tree's own scripts/engine_step_check.py (the standalone forward unit, --tape
             --save-outputs) on the same inputs, bitwise, with that run's digest check against the committed manifest
             of forward digests (nablaengine.checks: match / unanchored; a mismatch fails); or an npz of a reference
             made with engine_step_check.py --save-outputs ({corpus} in the path is the corpus name); or "off"
  jit        jax.jit of step and of the gradient of a one-step loss for B = 4096, 160, 40 (each twice, then again):
             Python traces, XLA compiles (jax.monitoring's backend-compile events) and library builds / loads counted
  window     a differentiated rollout: lax.scan over H = 64 control steps of jax_ffi.step (interface defaults) from
             corpus states, ctrl_t = ctrl_corpus + theta_t, two losses: quadratic sum_t (z_t - 0.75)^2 + 1e-3 |qvel_t|^2
             (mean over worlds) and linear sum_t <c_t, (qpos_t, qvel_t)> (fixed random c_t: dL/dx_t is exact);
             jax.grad with respect to (theta, qpos_0, qvel_0) against a manual chain of the per-step VJPs
             (jax_ffi.forward(tape=True) / backward, the per-step loss's jax.vjp, cotangents summed as dL/dx_t + the
             next step's VJP): bitwise for the linear loss, the quadratic one reported (XLA rounds its fused dL/dx_t
             differently, BPTT amplifies the ulps); time per window and peak device memory against the tape
             residuals (5888 B x 4 x H x B); the tape's copy into the stacked residual buffer: forward-only windows
             that stack the tapes, drop them, or run without the tape, and the tape-sized ops in the compiled
             gradient's HLO. Remat (each mode in its own process, before the rest): the scan body as is,
             jax.checkpoint(body), the policy save_only_these_names("dk_tape") and jax_ffi.remat_policy() (the tape
             and the step's outputs): forward / backward calls per window (the counters), time and peak memory; gate:
             with remat_policy() H forward and H backward calls and both losses' gradients bitwise the plain scan's
             (the other two rerun the forward: 2 H, reported)
  zeros      the cotangents of warm, friction and com are exactly zero, the warm' cotangent changes nothing
  nonfinite  the non-finite counter: 0 rows on every corpus (interface defaults and the exact gradient mode, 10 and 50
             iterations); NaN / inf injected into k rows of the output cotangents gives exactly k non-finite gradient
             rows and k non-finite cotangent rows, inside jit and inside lax.scan (read through diagnostics(*deps) in
             the same jit and through read_diagnostics() after it); the forward counters against the tape flags
  options    the gradient options (H = 16, B = 1024 states, a feedback ctrl_t = ctrl_corpus + theta_t - 0.5 (q_t - q_0)
             on the hinge angles so that the state reaches the loss also through the controls; the quadratic loss of
             the window section): gradient decay, grad_decay gamma of jax_ffi.step, against scale_grad((qpos, qvel),
             gamma) on the physics call's state arguments: jax.grad with respect to (theta, qpos_0, qvel_0) bitwise
             equal, also under jax.checkpoint(body, policy=remat_policy()) (and there H forward calls), for
             (clamp_kv, gamma) (0, 0.95), (1, 0.98), (2, 0.95); gamma = 1 is the default function itself (make_step,
             scale_grad's identity); the decay law (open loop, the linear loss of one step j only): d/dtheta_0 with
             gamma equals gamma^j times without it (relative 1e-4: each step rounds gamma's product). The closed-loop
             law (closed_loop_law, gated): the same feedback window (interface defaults; the set point a constant)
             whose carried state is a pytree with int32, bool and typed PRNG-key leaves, the linear loss of step j
             (4, 15) alone: in the full-state form, scale_grad(state, gamma) at the top of the scan body (gamma 0.9),
             d/dtheta_t equals gamma^(j - t) times without it for every t <= j and d/d(qpos_0, qvel_0) gamma^(j + 1)
             times (relative 1e-4), zero for t > j, and with gamma 0.5 bitwise (a power of 2 scales without
             rounding); the physics-only form (grad_decay 0.9) deviates from the law through the feedback path (its
             size reported; gated: at least 100 x the full-state form's deviation, so the check resolves the two). The
             soft clamp, clamp_kv=2: jax.vjp through step equal to backward(ctrl=, clamp_soft=) bitwise,
             clamp_soft=0 equal to clamp_kv=1 bitwise, under remat_policy() H forward calls (ctrl, a residual of the
             soft clamp, is recomputed with the body, not the physics); argument errors raise ValueError (clamp_soft
             with clamp_kv 0 / 1, clamp_kv 2 without ctrl, a negative or non-finite clamp_soft)
  api        EnginePhysics (the interface of nablaengine.reference.warp_physics.WarpPhysics): nconmax / njmax, stats()
             with WarpPhysics.stats()'s keys equal to the last forward call's info of its batch size (4096 through
             jax.vjp, 160 through step) and unchanged by reading it twice, diagnostics(); the model check raises on
             another timestep, foot-pair order, torso id, body mass or pair friction; negative shape tests through
             jax.ffi.ffi_call (wrong info, tape, contact or gradient output shapes, rank-0 qpos, rank-1 tape) raise
             the library's InvalidArgument, count no call, and the valid calls before and after are bitwise the same;
             k_adjoint's launch (launch_info: adj_launch_smem, at least each variant's sizeof(Adj); 4 worlds per SM on
             V100)
  timing     per call at B = 4096 (4 substeps, 10 and 50 iterations) of jax_ffi.step (dk_step), forward(tape=True)
             (dk_step_fwd), backward (dk_step_bwd) and jax.vjp (both), under jax.jit, against the bare kernels: the
             split forward (k_pre / k_post, the FFI's forward) of engine_step_check.py --tape, its T7 runs alternating
             the tape off and on with one binary (forward_ms, forward_tape_ms), and the adjoint check binary for the
             adjoint (adjoint_ms; --reps launches). That binary's own forward is the fused k_substep at its occupancy,
             not the FFI's path: it is reported as fused_forward_tape_ms and enters no overhead

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \\
      python scripts/jax_ffi_check.py --out runs/jax_ffi_check.json
  python scripts/jax_ffi_check.py --sections bitwise --corpora mf_s0 --variants 0,3 --bitwise-iterations 50 \\
      --out runs/jax_ffi_bitwise.json   # the quick form

Exit status: 0 when every section passes, 1 otherwise (the JSON is written either way).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nablaengine import checks, fixtures, provenance, workdir  # noqa: E402

SECTIONS = ("bitwise", "jit", "window", "zeros", "nonfinite", "options", "api", "timing")
CORPORA = list(fixtures.MAIN_CORPORA)  # --corpora (set in main)
# (clamp_kv, normal): the exact mode, the interface default, the other two, the soft clamp at CHECK_SOFT
CHECK_VARIANTS = ((1, 0), (0, 3), (1, 3), (0, 0), (2, 0), (2, 3))
CHECK_SOFT = 0.2
INPUTS = ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction", "com")


def gpu_name():
    """The GPU's name (nvidia-smi, the first visible GPU)."""
    return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i",
                           os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                          stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()


def jnp_inputs(z, B=None, rows=None):
    import jax.numpy as jnp
    sel = slice(0, B) if rows is None else rows
    return [jnp.asarray(np.asarray(z[k][sel], np.float32)) for k in INPUTS]


def same_bits(a, b):
    a, b = np.ascontiguousarray(np.asarray(a)), np.ascontiguousarray(np.asarray(b))
    return bool(a.shape == b.shape and a.dtype.itemsize == b.dtype.itemsize
                and np.array_equal(a.view(np.uint32), b.view(np.uint32)))


def fns(iterations, substeps=4, ctrl=None):
    """Jitted forward (tape) / backward per variant (the soft clamp's with ctrl, at CHECK_SOFT)."""
    import jax
    from nablaengine import jax_ffi as F
    fwd = jax.jit(lambda *a: F.forward(*a, iterations=iterations, substeps=substeps, tape=True))
    step = jax.jit(lambda *a: F.forward(*a, iterations=iterations, substeps=substeps, tape=False))
    bwd = {v: jax.jit(lambda tp, fr, c, gq, gv, v=v: F.backward(
        tp, fr, c, gq, gv, substeps=substeps, clamp_kv=v[0], normal=v[1], ctrl=ctrl if v[0] == 2 else None,
        clamp_soft=CHECK_SOFT if v[0] == 2 else None)) for v in CHECK_VARIANTS}
    return fwd, step, bwd


# ---------------------------------------------------------------- bitwise

def forward_reference(args, corpus, nw, work):
    """(reference arrays or None, record) of the forward reference (--forward-ref, module docstring) at 50 iterations."""
    if args.forward_ref == "off":
        return None, "off"
    if args.forward_ref != "self":
        path = Path(args.forward_ref.format(corpus=corpus))
        if not path.exists():
            return None, f"skipped ({provenance.rel(path)} missing)"
        return np.load(path), dict(reference=provenance.rel(path))
    out, ref = work / f"forward_ref_{corpus}.json", work / f"forward_ref_{corpus}_outputs.npz"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "engine_step_check.py"), "--corpus", corpus,
                        "--substeps", "4", "--iterations", "50", "--nworld", str(nw), "--check", "8", "--reps", "0",
                        "--tape", "--save-outputs", str(ref), "--out", str(out)], timeout=args.timeout,
                       stdout=subprocess.DEVNULL)
    if not ref.exists():
        return None, dict(reference="self", error=f"engine_step_check.py exit code {p.returncode}")
    r = json.loads(out.read_text())
    dg = r.get("digests", {})
    return np.load(ref), dict(reference="self: scripts/engine_step_check.py --tape --save-outputs (this tree)",
                              engine_step_check_failures=r.get("failures"),
                              digests=dict(status=dg.get("status"), manifest=dg.get("manifest"),
                                           arrays=dg.get("arrays"), note=dg.get("note")))


def sec_bitwise(args):
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    m, mt = fixtures.load_model()
    work = workdir.work_dir("jax_ffi_bitwise_")
    variants = [v for v in CHECK_VARIANTS if f"{v[0]},{v[1]}" in args.variants.split(";")] if args.variants else \
        list(CHECK_VARIANTS)
    bins = {v: checks.build(m, mt, work, f"adj_kv{v[0]}_n{v[1]}", [f"DK_ADJ_CLAMP_KV={v[0]}", "DK_ADJ_STAGE=3",
                                                                   f"DK_ADJ_NORMAL={v[1]}"]) for v in variants}
    info = F.init()
    res, ok = dict(library=dict(info, path=provenance.rel(info["path"])), variants=[list(v) for v in variants],
                   runs=[]), True
    for corpus in CORPORA:
        z = fixtures.load_corpus(corpus, adjoint=True)
        nw = len(z["qpos"])
        checks.write_inputs(work, checks.corpus_inputs(z))
        a = jnp_inputs(z)
        cots = [(jnp.asarray(z[f"cot{c}_qpos"]), jnp.asarray(z[f"cot{c}_qvel"])) for c in range(3)]
        for iters in (int(x) for x in args.bitwise_iterations.split(",")):
            fwd, step, bwd = fns(iters, ctrl=a[3])
            q, v, w, inf, tp = fwd(*a)
            qs, vs, ws, infs = step(*a)
            tp_np = np.asarray(tp)
            row = dict(corpus=corpus, iterations=iters, nworld=nw,
                       step_vs_step_fwd=dict(qpos=same_bits(qs, q), qvel=same_bits(vs, v), warm=same_bits(ws, w),
                                             info=same_bits(infs, inf)), variants={})
            ok &= all(row["step_vs_step_fwd"].values())
            if iters == 50 and args.forward_ref != "off":  # the library's target, nvcc, forward-unit flags
                fr, rec = forward_reference(args, corpus, nw, work)
                if fr is None:
                    row["forward_vs_reference"] = rec
                    ok &= not isinstance(rec, dict)  # a self reference that failed to run fails
                else:
                    same = dict(qpos=same_bits(q, fr["qpos"]), qvel=same_bits(v, fr["qvel"]),
                                warm=same_bits(w, fr["warm"]), info=same_bits(inf, fr["info"]),
                                tape=same_bits(tp_np.reshape(fr["tape"].shape), fr["tape"]))
                    row["forward_vs_reference"] = dict(rec, **same)
                    ok &= all(same.values()) and rec.get("digests", {}).get("status") != "mismatch"
            for var in variants:
                _, out = checks.run(bins[var], work, m, nw, iters, 4, 3, 0, False, args.timeout,
                                    clamp_soft=CHECK_SOFT if var[0] == 2 else None)
                out_warm = np.fromfile(work / "out_warm.bin", np.float32).reshape(nw, m.nv)
                r = dict(forward=dict(qpos=same_bits(q, out["qpos"]), qvel=same_bits(v, out["qvel"]),
                                      warm=same_bits(w, out_warm), info=same_bits(inf, out["info"]),
                                      tape=same_bits(tp_np, out["tape"].reshape(nw, -1))), gradients={})
                for c in range(3):
                    gq, gv, gu, ai = bwd[var](tp, a[4], a[5], *cots[c])
                    ai = np.asarray(ai)
                    r["gradients"][f"cot{c}"] = dict(
                        qpos=same_bits(gq, out[f"grad{c}_qpos"]), qvel=same_bits(gv, out[f"grad{c}_qvel"]),
                        ctrl=same_bits(gu, out[f"grad{c}_ctrl"]),
                        adj_info=bool(np.array_equal(ai & ((1 << F.NONFINITE_SHIFT) - 1), out[f"adj_info{c}"])),
                        nonfinite_bits_zero=bool(((ai >> F.NONFINITE_SHIFT) == 0).all()))
                    if c == 0:  # determinism: the same call again
                        gq2, gv2, gu2, _ = bwd[var](tp, a[4], a[5], *cots[c])
                        r["gradients"]["cot0"]["repeat_bitwise"] = bool(same_bits(gq2, gq) and same_bits(gv2, gv)
                                                                        and same_bits(gu2, gu))
                r["pass"] = bool(all(r["forward"].values())
                                 and all(all(g.values()) for g in r["gradients"].values()))
                ok &= r["pass"]
                row["variants"][f"kv{var[0]}_n{var[1]}"] = r
            print("bitwise", corpus, iters, {k: x["pass"] for k, x in row["variants"].items()},
                  row.get("forward_vs_reference"), flush=True)
            res["runs"].append(row)
    res["pass"] = bool(ok)
    return res


# ---------------------------------------------------------------- jit

def sec_jit(args):
    import jax
    import jax.numpy as jnp
    from jax import monitoring
    from nablaengine import jax_ffi as F
    compiles = []
    monitoring.register_event_duration_secs_listener(
        lambda name, dur, **kw: compiles.append(name) if name == "/jax/core/compile/backend_compile_duration" else None)
    z = fixtures.load_corpus(CORPORA[0], adjoint=True)
    F.init()
    st0 = dict(F.STATS)
    traces = {"step": [], "grad": []}

    def f_step(*a):
        traces["step"].append(a[0].shape[0])
        return F.step(*a)

    def loss(qpos, qvel, ctrl, warm, fr, com, cq, cv):
        q, v, _ = F.step(qpos, qvel, warm, ctrl, fr, com)
        traces["grad"].append(qpos.shape[0])
        return jnp.sum(q * cq) + jnp.sum(v * cv)

    j_step = jax.jit(f_step)
    j_grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    inputs = {}
    for B in (4096, 160, 40):
        a = jnp_inputs(z, B)
        inputs[B] = (a, jnp.asarray(z["cot0_qpos"][:B]), jnp.asarray(z["cot0_qvel"][:B]))
    jax.block_until_ready(inputs)
    n0 = len(compiles)
    order = (4096, 160, 40, 4096, 160, 40, 40, 160, 4096)
    calls = []
    for B in order:
        a, cq, cv = inputs[B]
        c0 = len(compiles)
        t = time.perf_counter()
        out = j_step(*a)
        g = j_grad(a[0], a[1], a[3], a[2], a[4], a[5], cq, cv)
        jax.block_until_ready((out, g))
        calls.append(dict(B=B, seconds=time.perf_counter() - t, compiles=len(compiles) - c0,
                          finite=bool(all(np.isfinite(np.asarray(x)).all() for x in g))))
    st1 = dict(F.STATS)
    res = dict(order=list(order), calls=calls, traces_step=traces["step"], traces_grad=traces["grad"],
               compiles_total=len(compiles) - n0, library_builds_during=st1["builds"] - st0["builds"],
               library_loads_during=st1["loads"] - st0["loads"], library_stats=st1)
    res["pass"] = bool(sorted(traces["step"]) == [40, 160, 4096] and sorted(traces["grad"]) == [40, 160, 4096]
                       and res["compiles_total"] == 6 and res["library_builds_during"] == 0
                       and res["library_loads_during"] == 0 and all(c["finite"] for c in calls)
                       and all(c["compiles"] == 0 for c in calls[3:]))
    print("jit", json.dumps({k: v for k, v in res.items() if k != "calls"}), flush=True)
    return res


# ---------------------------------------------------------------- window

def _hlo_tape_ops(text, B, cols):
    """Counts of HLO instructions whose result is a tape-sized f32 array (one step's tape, or a stack of them)."""
    import re
    one, out = f"f32[{B},{cols}]", {}
    for line in text.splitlines():
        mt = re.match(r"\s*(?:ROOT )?%?[\w.\-]+ = (\S+?)(?:\{[^}]*\})? ([\w\-]+)\(", line)
        if not mt:
            continue
        shape, op = mt.group(1), mt.group(2)
        if shape == one or (shape.startswith("f32[") and shape.endswith(f",{B},{cols}]")):
            key = f"{op} {'one' if shape == one else 'stack'}"
            out[key] = out.get(key, 0) + 1
    return out


REMAT_MODES = ("none", "default", "tape", "named")  # the window's scan body: as is, jax.checkpoint(body), with the
# policy save_only_these_names("dk_tape"), with jax_ffi.remat_policy() (the tape and the step's outputs)


def _window_setup(args):
    """The rollout window of the window section: its inputs and (step loss name -> function), window_loss(theta, q0, v0,
    step_loss, remat) with the scan body wrapped per REMAT_MODES."""
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    H, B = args.horizon, args.window_worlds
    z = fixtures.load_corpus(CORPORA[0], adjoint=True)
    q0, v0, w0, u0, fr, com = jnp_inputs(z, B)
    theta = jnp.zeros((H, B, u0.shape[1]), jnp.float32)
    rng = np.random.default_rng(0)  # the linear loss's fixed weights (exact cotangents: no rounding in dL/dx_t)
    cq_lin = jnp.asarray(rng.standard_normal((H, B, q0.shape[1])).astype(np.float32) / B)
    cv_lin = jnp.asarray(rng.standard_normal((H, B, v0.shape[1])).astype(np.float32) / B)

    def quad_loss(q, v, k):  # a quadratic loss: root height and velocity
        return (jnp.sum((q[:, 2] - 0.75) ** 2) + 1e-3 * jnp.sum(v * v)) / B

    def lin_loss(q, v, k):
        return jnp.sum(q * cq_lin[k]) + jnp.sum(v * cv_lin[k])

    def window_loss(theta, q0, v0, step_loss, remat="none"):
        def body(carry, x):
            q, v, w = carry
            th, k = x
            q, v, w = F.step(q, v, w, u0 + th, fr, com)
            return (q, v, w), step_loss(q, v, k)
        if remat == "default":
            body = jax.checkpoint(body)
        elif remat == "tape":
            body = jax.checkpoint(body, policy=jax.checkpoint_policies.save_only_these_names(F.TAPE_NAME))
        elif remat == "named":
            body = jax.checkpoint(body, policy=F.remat_policy())
        _, ls = jax.lax.scan(body, (q0, v0, w0), (theta, jnp.arange(H)))
        return jnp.sum(ls)

    return dict(H=H, B=B, q0=q0, v0=v0, w0=w0, u0=u0, fr=fr, com=com, theta=theta,
                losses=dict(quadratic=quad_loss, linear=lin_loss), window_loss=window_loss)


def sec_window_remat(args):
    """(internal, one process per --remat-mode: its own peak memory) the window's gradient with the scan body wrapped
    per the mode: forward / backward calls of one gradient, time per window, peak device memory; the gradients to
    --remat-grads."""
    import jax
    from nablaengine import jax_ffi as F
    S = _window_setup(args)
    H, B, theta, q0, v0 = S["H"], S["B"], S["theta"], S["q0"], S["v0"]
    F.init()
    dev = jax.local_devices()[0]
    base = dev.memory_stats()["bytes_in_use"]
    res, grads = dict(mode=args.remat_mode, H=H, B=B), {}
    for name, f in S["losses"].items():  # quadratic first: the peak memory is its own
        fn = jax.jit(jax.value_and_grad(lambda th, q, v, f=f: S["window_loss"](th, q, v, f, args.remat_mode),
                                        argnums=(0, 1, 2)))
        t = time.perf_counter()
        compiled = fn.lower(theta, q0, v0).compile()
        t_compile = time.perf_counter() - t
        ma = compiled.memory_analysis()
        F.read_diagnostics(reset=True)
        t = time.perf_counter()
        val, g = fn(theta, q0, v0)
        jax.block_until_ready(g)
        t_first = time.perf_counter() - t
        d = F.read_diagnostics(reset=True)
        r = dict(fwd_calls=d["fwd_calls"], bwd_calls=d["bwd_calls"], tape_records=d["tape_records"],
                 nonfinite_rows=d["nonfinite_rows"], loss=float(val), compile_seconds=t_compile,
                 first_call_seconds=t_first,
                 xla_temp_bytes=int(getattr(ma, "temp_size_in_bytes", -1)) if ma is not None else None)
        if name == "quadratic":
            r["peak_minus_before"] = int(dev.memory_stats()["peak_bytes_in_use"] - base)
            r["peak_over_tape"] = r["peak_minus_before"] / F.tape_bytes(B, 4, H)
            times = []
            for _ in range(args.window_reps):
                t = time.perf_counter()
                jax.block_until_ready(fn(theta, q0, v0))
                times.append(time.perf_counter() - t)
            r.update(window_seconds=float(np.median(times)), window_seconds_all=times,
                     ms_per_control_step=1e3 * float(np.median(times)) / H)
        for k, x in zip(("theta", "qpos0", "qvel0"), g):
            grads[f"{name}_{k}"] = np.asarray(x)
        res[name] = r
        del fn, compiled, g
    np.savez(args.remat_grads, **grads)
    print("window_remat", json.dumps(res), flush=True)
    res["pass"] = True  # judged by sec_window
    return res


def _remat_runs(args):
    """The window section's remat cases, each mode in its own process (one after the other, before that section itself
    touches the GPU), the gradients compared bitwise with the plain scan's ("none"). Gate: with jax_ffi.remat_policy()
    ("named") one dk_step_fwd and one dk_step_bwd per control step and the gradients bitwise those without remat;
    reported: jax.checkpoint(body) and the tape-only policy (both rerun the forward: 2 H forward calls)."""
    work = workdir.work_dir("jax_ffi_remat_")
    runs = {}
    for mode in REMAT_MODES:
        out, gpath = work / f"{mode}.json", work / f"{mode}.npz"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--section", "window_remat", "--remat-mode", mode,
               "--remat-grads", str(gpath), "--out", str(out)]
        for k in ("horizon", "window_worlds", "window_reps", "timeout"):
            cmd += [f"--{k.replace('_', '-')}", str(getattr(args, k))]
        p = subprocess.run(cmd, timeout=args.timeout)
        runs[mode] = json.loads(out.read_text()) if p.returncode == 0 and out.exists() else dict(
            error=f"exit code {p.returncode}")
    H = args.horizon
    ref = np.load(work / "none.npz") if (work / "none.npz").exists() else None
    for mode in REMAT_MODES[1:]:
        if ref is None or not (work / f"{mode}.npz").exists():
            continue
        g = np.load(work / f"{mode}.npz")
        runs[mode]["bitwise_vs_no_remat"] = {k: same_bits(g[k], ref[k]) for k in ref.files}
    expect = dict(none=H, default=2 * H, tape=2 * H, named=H)
    calls = {m: {loss: (r.get(loss, {}).get("fwd_calls"), r.get(loss, {}).get("bwd_calls")) for loss in
                 ("quadratic", "linear")} for m, r in runs.items()}
    named = runs.get("named", {})
    from nablaengine import jax_ffi as F
    res = dict(modes=runs, forward_backward_calls=calls, expected_forward_calls=expect,
               as_expected={m: all(c[0] == expect[m] for c in calls[m].values()) for m in calls},
               policy="jax.checkpoint(body, policy=jax_ffi.remat_policy()) = "
                      f"save_only_these_names{tuple(F.REMAT_NAMES)}")
    res["pass"] = bool("error" not in named and all(named[loss]["fwd_calls"] == H and named[loss]["bwd_calls"] == H
                                                     for loss in ("quadratic", "linear"))
                       and named.get("bitwise_vs_no_remat") and all(named["bitwise_vs_no_remat"].values()))
    print("window remat", json.dumps(dict(calls=calls, pass_=res["pass"],
                                          times={m: r.get("quadratic", {}).get("window_seconds") for m, r in runs.items()},
                                          peak={m: r.get("quadratic", {}).get("peak_minus_before")
                                                for m, r in runs.items()})), flush=True)
    return res


def sec_window(args):
    remat = _remat_runs(args)  # first: its processes need the GPU's memory
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    S = _window_setup(args)
    H, B = S["H"], S["B"]
    q0, v0, w0, u0, fr, com, theta = (S[k] for k in ("q0", "v0", "w0", "u0", "fr", "com", "theta"))
    quad_loss, lin_loss, window_loss = S["losses"]["quadratic"], S["losses"]["linear"], S["window_loss"]
    info = F.init()
    dev = jax.local_devices()[0]
    base = dev.memory_stats()["bytes_in_use"]

    grad_fns = {name: jax.jit(jax.value_and_grad(lambda th, q, v, f=f: window_loss(th, q, v, f), argnums=(0, 1, 2)))
                for name, f in (("quadratic", quad_loss), ("linear", lin_loss))}
    t = time.perf_counter()
    compiled = grad_fns["quadratic"].lower(theta, q0, v0).compile()
    t_compile = time.perf_counter() - t
    tape_ops = _hlo_tape_ops(compiled.as_text(), B, 4 * info["rec_words"])
    t = time.perf_counter()
    val, grads = grad_fns["quadratic"](theta, q0, v0)
    jax.block_until_ready(grads)
    t_first = time.perf_counter() - t
    peak = dev.memory_stats()["peak_bytes_in_use"]
    times = []
    for _ in range(args.window_reps):
        t = time.perf_counter()
        jax.block_until_ready(grad_fns["quadratic"](theta, q0, v0))
        times.append(time.perf_counter() - t)
    tape_res = F.tape_bytes(B, 4, H)
    results = {"quadratic": (val, grads)}
    results["linear"] = grad_fns["linear"](theta, q0, v0)
    jax.block_until_ready(results["linear"])

    # forward-only windows: tapes stacked as scan outputs / computed and dropped / not written
    def fwd_window(mode):
        def body(carry, th):
            q, v, w = carry
            if mode == "notape":
                q, v, w, _ = F.forward(q, v, w, u0 + th, fr, com)
                return (q, v, w), None
            q, v, w, _, tp = F.forward(q, v, w, u0 + th, fr, com, tape=True)
            return (q, v, w), (tp if mode == "stacked" else None)
        return jax.jit(lambda theta: jax.lax.scan(body, (q0, v0, w0), theta))

    fwd_times = {}
    for mode in ("stacked", "dropped", "notape"):
        f = fwd_window(mode)
        jax.block_until_ready(f(theta))
        ts = []
        for _ in range(args.window_reps):
            t = time.perf_counter()
            jax.block_until_ready(f(theta))
            ts.append(time.perf_counter() - t)
        fwd_times[mode] = float(np.median(ts))
        del f

    # the manual chain of per-step VJPs (the same FFI calls, one control step at a time, cotangents accumulated
    # as dL/dx_t + the next step's VJP)
    fwd1 = jax.jit(lambda q, v, w, th: F.forward(q, v, w, u0 + th, fr, com, tape=True))
    bwd1 = jax.jit(lambda tp, gq, gv: F.backward(tp, fr, com, gq, gv))
    t = time.perf_counter()
    q, v, w = q0, v0, w0
    tapes, states = [], []
    for k in range(H):
        q, v, w, _, tp = fwd1(q, v, w, theta[k])
        tapes.append(tp)
        states.append((q, v))
    jax.block_until_ready(q)
    t_manual_fwd = time.perf_counter() - t

    def rel(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))

    cons, t_manual_bwd = {}, {}
    for name, f in (("quadratic", quad_loss), ("linear", lin_loss)):
        lgrad = jax.jit(lambda q, v, k, f=f: jax.vjp(lambda a, b: f(a, b, k), q, v)[1](jnp.float32(1.0)))
        t = time.perf_counter()
        gq, gv = jnp.zeros_like(q0), jnp.zeros_like(v0)
        g_th_m = [None] * H
        for k in range(H - 1, -1, -1):
            lq, lv = lgrad(*states[k], k)
            gq, gv = gq + lq, gv + lv
            gq, gv, gu, _ = bwd1(tapes[k], gq, gv)
            g_th_m[k] = gu
        jax.block_until_ready((gq, gv))
        t_manual_bwd[name] = time.perf_counter() - t
        g_th, g_q0, g_v0 = results[name][1]
        g_th_m = jnp.stack(g_th_m)
        cons[name] = dict(theta=dict(rel=rel(g_th, g_th_m), bitwise=same_bits(g_th, g_th_m)),
                          qpos0=dict(rel=rel(g_q0, gq), bitwise=same_bits(g_q0, gq)),
                          qvel0=dict(rel=rel(g_v0, gv), bitwise=same_bits(g_v0, gv)),
                          gradient_finite=bool(all(np.isfinite(np.asarray(x)).all() for x in (g_th, g_q0, g_v0))),
                          loss=float(results[name][0]))
    del tapes
    diag = F.read_diagnostics()
    t_win = float(np.median(times))
    res = dict(H=H, B=B, options=dict(iterations=F.ITERATIONS, substeps=F.SUBSTEPS, clamp_kv=F.CLAMP_KV,
                                      normal=F.NORMAL),
               consistency_vs_manual_chain=cons, compile_seconds=t_compile, first_call_seconds=t_first,
               window_seconds=t_win, window_seconds_all=times, ms_per_control_step=1e3 * t_win / H,
               forward_part_ms_per_step=1e3 * fwd_times["stacked"] / H,
               backward_part_ms_per_step=1e3 * (t_win - fwd_times["stacked"]) / H,
               manual_chain_seconds=dict(forward=t_manual_fwd, backward=t_manual_bwd),
               memory=dict(bytes_in_use_before=int(base), peak_bytes_in_use=int(peak), tape_residual_bytes=int(tape_res),
                           peak_minus_before=int(peak - base), peak_over_tape=float((peak - base) / tape_res),
                           device_limit=int(dev.memory_stats().get("bytes_limit", 0))),
               forward_window_seconds=fwd_times,
               tape_copy=dict(stacked_minus_dropped_ms_per_step=1e3 * (fwd_times["stacked"] - fwd_times["dropped"]) / H,
                              tape_write_ms_per_step=1e3 * (fwd_times["dropped"] - fwd_times["notape"]) / H,
                              stacked_share_of_window=(fwd_times["stacked"] - fwd_times["dropped"]) / t_win,
                              hlo_tape_sized_ops=tape_ops),
               diagnostics=diag, remat=remat)
    # gated: the linear loss (exact cotangents) bitwise; the quadratic loss's dL/dx_t is rounded differently by XLA's
    # fused scan body than by the separate per-step jax.vjp, and BPTT through 64 steps amplifies those ulps (reported);
    # with remat_policy() one forward per control step and the plain scan's gradients bitwise
    res["pass"] = bool(all(cons["linear"][k]["bitwise"] for k in ("theta", "qpos0", "qvel0"))
                       and cons["linear"]["gradient_finite"] and cons["quadratic"]["gradient_finite"]
                       and all(cons["quadratic"][k]["rel"] <= 1e-3 for k in ("theta", "qpos0", "qvel0"))
                       and diag["nonfinite_rows"] == 0 and remat["pass"])
    print("window", json.dumps({k: v for k, v in res.items() if k not in ("diagnostics", "remat")}), flush=True)
    return res


# ---------------------------------------------------------------- zeros

def sec_zeros(args):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    z = fixtures.load_corpus(CORPORA[-1], adjoint=True)
    a = jnp_inputs(z)
    res, ok = {}, True
    for opts in (dict(), dict(iterations=50, clamp_kv=1, normal=0)):
        @jax.jit
        def run(*a, opts=opts):
            outs, f = jax.vjp(lambda *x: F.step(*x, **opts), *a)
            cq, cv = jnp.asarray(z["cot0_qpos"]), jnp.asarray(z["cot0_qvel"])
            g0 = f((cq, cv, jnp.zeros_like(outs[2])))
            g1 = f((cq, cv, jnp.asarray(z["cot1_qvel"])))  # a random warm' cotangent: must change nothing
            return g0, g1
        g0, g1 = run(*a)
        r = dict(warm_zero=bool((np.asarray(g0[2]) == 0).all()), friction_zero=bool((np.asarray(g0[4]) == 0).all()),
                 com_zero=bool((np.asarray(g0[5]) == 0).all()),
                 warm_cotangent_ignored=bool(all(same_bits(x, y) for x, y in zip(g0, g1))),
                 shapes=[list(x.shape) for x in g0], dtypes=[str(x.dtype) for x in g0])
        r["pass"] = bool(r["warm_zero"] and r["friction_zero"] and r["com_zero"] and r["warm_cotangent_ignored"])
        ok &= r["pass"]
        res["default" if not opts else "exact_options"] = r
    res["pass"] = bool(ok)
    print("zeros", json.dumps(res), flush=True)
    return res


# ---------------------------------------------------------------- nonfinite

def sec_nonfinite(args):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    from nablaengine.g1_engine import tape
    m, _ = fixtures.load_model()
    res, ok = dict(corpus={}), True
    F.read_diagnostics(reset=True)
    for corpus in CORPORA:
        z = fixtures.load_corpus(corpus, adjoint=True)
        a = jnp_inputs(z)
        nw = len(z["qpos"])
        for iters, var in ((10, (0, 3)), (50, (1, 0)), (10, (1, 0)), (50, (0, 3))):
            fwd, _, bwd = fns(iters)
            F.read_diagnostics(reset=True)
            _, _, _, inf, tp = fwd(*a)
            gr = [bwd[var](tp, a[4], a[5], jnp.asarray(z[f"cot{c}_qpos"]), jnp.asarray(z[f"cot{c}_qvel"]))
                  for c in range(3)]
            d = F.read_diagnostics(reset=True)
            rec = tape.decode(np.asarray(tp).reshape(nw, 4, -1), m)
            inf = np.asarray(inf)
            nonfinite_np = sum(int((~np.isfinite(np.concatenate([np.asarray(g[i]) for i in range(3)], 1))).any(1).sum())
                               for g in gr)
            expect = dict(fwd_calls=1, fwd_worlds=nw, tape_records=nw * 4, fwd_unsupported_worlds=int((inf[:, 3] != 0).sum()),
                          fwd_iteration_limit_worlds=int((inf[:, 0] >= iters).sum()),
                          refact_records=int(rec["refact"].sum()), nonconv_records=int(rec["nonconv"].sum()),
                          singular_records=int(rec["singular"].sum()), bwd_calls=3, bwd_rows=3 * nw,
                          nonfinite_rows=nonfinite_np, nonfinite_cotangent_rows=0, replay_mismatch_worlds=0)
            match = {k: d[k] == v for k, v in expect.items()}
            r = dict(counters=d, match=match, zero_nonfinite=d["nonfinite_rows"] == 0)
            r["pass"] = bool(all(match.values()) and r["zero_nonfinite"])
            ok &= r["pass"]
            res["corpus"][f"{corpus}/it{iters}/kv{var[0]}_n{var[1]}"] = r
            print("nonfinite corpus", corpus, iters, var, d["nonfinite_rows"], all(match.values()), flush=True)
    # injected NaN / inf in k rows of the output cotangents, inside jit (diagnostics(*deps) in the same jit)
    z = fixtures.load_corpus(CORPORA[0], adjoint=True)
    B = 160
    a = jnp_inputs(z, B)
    rows = np.array([3, 17, 42, 99, 150])
    cq = np.asarray(z["cot0_qpos"][:B]).copy()
    cv = np.asarray(z["cot0_qvel"][:B]).copy()
    cq[rows[:3], 5] = np.nan
    cv[rows[3], 7] = np.inf
    cv[rows[4], 0] = -np.inf
    cq, cv = jnp.asarray(cq), jnp.asarray(cv)

    @jax.jit
    def inject(*a):
        outs, f = jax.vjp(lambda *x: F.step(*x), *a)
        g = f((cq, cv, jnp.zeros_like(outs[2])))
        return g, F.diagnostics(*g, reset=True)

    F.read_diagnostics(reset=True)
    g, dj = inject(*a)
    dj = F.decode_diagnostics(dj)
    gbad = np.flatnonzero(~np.isfinite(np.concatenate([np.asarray(g[i]) for i in (0, 1, 3)], 1)).all(1))
    r = dict(injected_rows=rows.tolist(), counters_in_jit=dj, gradient_rows_nonfinite=gbad.tolist())
    r["pass"] = bool(dj["nonfinite_rows"] == len(rows) and dj["nonfinite_cotangent_rows"] == len(rows)
                     and set(gbad.tolist()) == set(rows.tolist()) and dj["bwd_rows"] == B)
    # inside lax.scan over 8 steps (the cotangents of the last step's output carry the NaNs back through all of them),
    # read after the jit by read_diagnostics()
    H = 8

    def loss(th):
        def body(c, t):
            q, v, w = c
            q, v, w = F.step(q, v, w, a[3] + t, a[4], a[5])
            return (q, v, w), None
        (q, v, _), _ = jax.lax.scan(body, (a[0], a[1], a[2]), th)
        return jnp.sum(q * cq) + jnp.sum(v * cv)

    F.read_diagnostics(reset=True)
    gth = jax.jit(jax.grad(loss))(jnp.zeros((H, B, a[3].shape[1]), jnp.float32))
    ds = F.read_diagnostics(reset=True)
    rs = dict(H=H, counters_after_scan=ds,
              theta_rows_nonfinite=int((~np.isfinite(np.asarray(gth))).any((0, 2)).sum()))
    # every step's backward sees the 5 worlds' non-finite cotangents (the NaN propagates backwards in each world)
    rs["pass"] = bool(ds["bwd_calls"] == H and ds["nonfinite_rows"] == H * len(rows)
                      and ds["nonfinite_cotangent_rows"] == H * len(rows) and rs["theta_rows_nonfinite"] == len(rows))
    res["injected_jit"], res["injected_scan"] = r, rs
    ok &= r["pass"] and rs["pass"]
    res["pass"] = bool(ok)
    print("nonfinite injected", json.dumps(r["pass"]), json.dumps(rs), flush=True)
    return res


# ---------------------------------------------------------------- options

def sec_options(args):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    H, B = 16, 1024
    z = fixtures.load_corpus(CORPORA[0], adjoint=True)
    q0, v0, w0, u0, fr, com = jnp_inputs(z, B)
    theta = jnp.zeros((H, B, u0.shape[1]), jnp.float32)
    F.init()
    res, ok = dict(H=H, B=B), True

    def loss_step(q, v):
        return (jnp.sum((q[:, 2] - 0.75) ** 2) + 1e-3 * jnp.sum(v * v)) / B

    def window(theta, q0, v0, form, opts, gamma, remat=False, feedback=True, only=None):
        """form "decay": step(grad_decay=gamma); "scale": step on scale_grad((q, v), gamma); only: the loss of that
        step alone (linear in its state, open loop)."""
        def body(carry, x):
            q, v, w = carry
            th, k = x
            u = u0 + th - (0.5 * (q[:, 7:] - q0[:, 7:]) if feedback else 0.0)  # the state also reaches the controls
            if form == "decay":
                q2, v2, w2 = F.step(q, v, w, u, fr, com, grad_decay=gamma, **opts)
            else:
                qs, vs = F.scale_grad((q, v), gamma)
                q2, v2, w2 = F.step(qs, vs, w, u, fr, com, **opts)
            if only is None:
                lk = loss_step(q2, v2)
            else:
                lk = jnp.where(k == only, jnp.sum(q2[:, 2]) + jnp.sum(v2[:, 6]), 0.0) / B
            return (q2, v2, w2), lk
        if remat:
            body = jax.checkpoint(body, policy=F.remat_policy())
        _, ls = jax.lax.scan(body, (q0, v0, w0), (theta, jnp.arange(H)))
        return jnp.sum(ls)

    def grads(form, opts, gamma, **kw):
        f = jax.jit(jax.grad(lambda th, q, v: window(th, q, v, form, opts, gamma, **kw), argnums=(0, 1, 2)))
        F.read_diagnostics(reset=True)
        g = f(theta, q0, v0)
        jax.block_until_ready(g)
        d = F.read_diagnostics(reset=True)
        return [np.asarray(x) for x in g], d

    # gradient decay: grad_decay against scale_grad, bitwise (plain scan and remat_policy())
    res["decay_vs_scale_grad"] = {}
    for kv, gamma in ((0, 0.95), (1, 0.98), (2, 0.95)):
        opts = dict(clamp_kv=kv)
        r = {}
        for remat in (False, True):
            gd, dd = grads("decay", opts, gamma, remat=remat)
            gs, ds = grads("scale", opts, gamma, remat=remat)
            r["remat" if remat else "plain"] = dict(
                bitwise={k: same_bits(x, y) for k, x, y in zip(("theta", "qpos0", "qvel0"), gd, gs)},
                fwd_calls=(dd["fwd_calls"], ds["fwd_calls"]), bwd_calls=(dd["bwd_calls"], ds["bwd_calls"]),
                finite=bool(all(np.isfinite(x).all() for x in gd + gs)),
                theta_norm=float(np.linalg.norm(gd[0])))
            if remat:
                r["remat_vs_plain_bitwise"] = all(same_bits(x, y) for x, y in zip(gd, g_plain))  # noqa: F821
            g_plain = gd  # noqa: F841 (read in the next iteration)
        g1, _ = grads("decay", opts, 1.0)
        r["theta_norm_gamma1"] = float(np.linalg.norm(g1[0]))
        r["pass"] = bool(all(all(x["bitwise"].values()) and x["finite"] for x in (r["plain"], r["remat"]))
                         and r["remat"]["fwd_calls"] == (H, H) and r["plain"]["fwd_calls"] == (H, H)
                         and r["remat_vs_plain_bitwise"])
        ok &= r["pass"]
        res["decay_vs_scale_grad"][f"kv{kv}_gamma{gamma:g}"] = r
    # gamma = 1: the default function and the identity
    x = (q0, v0)
    res["gamma1_is_default"] = bool(F.make_step(grad_decay=1.0) is F.make_step() and F.scale_grad(x, 1.0) is x
                                    and F.make_step(10, 4, 0, 3, None, 1.0) is F.make_step())
    ok &= res["gamma1_is_default"]
    # the decay law: open loop, the loss of step j only
    law = {}
    for j in (1, 4, 15):
        g1, _ = grads("decay", dict(clamp_kv=1), 1.0, feedback=False, only=j)
        gg, _ = grads("decay", dict(clamp_kv=1), 0.9, feedback=False, only=j)
        a, b = gg[0][0].astype(np.float64), 0.9 ** j * g1[0][0].astype(np.float64)
        law[f"j{j}"] = dict(rel=float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)),
                            norm_gamma1=float(np.linalg.norm(g1[0][0])), expected_factor=0.9 ** j)
    res["decay_law_theta0"] = law
    res["decay_law_pass"] = bool(all(v["rel"] <= 1e-4 and v["norm_gamma1"] > 0 for v in law.values()))
    ok &= res["decay_law_pass"]
    # the closed-loop law (jax_ffi's module docstring, the two forms of gradient decay): a window with feedback whose
    # carried state is a pytree with int, bool and PRNG-key leaves; the cost of step j alone (weights wk, one jit per
    # form); full: scale_grad(state, gamma) at the top of the scan body, physics: step(grad_decay=gamma)
    qref = jnp.asarray(z["qpos"][:B])  # the feedback's set point: a constant, not the differentiated q0

    def window_cl(th_all, q0_, v0_, wk, form, gamma):
        def body(st, x):
            if form == "full":
                st = F.scale_grad(st, gamma)  # every inexact leaf; n, alive and key pass through
            th, wj = x
            key, _ = jax.random.split(st["key"])
            u = u0 + th - 0.5 * (st["q"][:, 7:] - qref[:, 7:])  # the state reaches the next step through ctrl
            q2, v2, w2 = F.step(st["q"], st["v"], st["w"], u, fr, com,
                                grad_decay=gamma if form == "physics" else 1.0)
            lk = wj * (jnp.sum(q2[:, 2]) + jnp.sum(v2[:, 6])) / B
            return dict(q=q2, v=v2, w=w2, n=st["n"] + 1, alive=st["alive"] & (q2[:, 2] > 0.3), key=key), lk
        st0 = dict(q=q0_, v=v0_, w=w0, n=jnp.zeros(B, jnp.int32), alive=jnp.ones(B, bool), key=jax.random.key(0))
        _, ls = jax.lax.scan(body, st0, (th_all, wk))
        return jnp.sum(ls)

    cl = {(form, g): jax.jit(jax.grad(lambda th, q, v, wk, form=form, g=g: window_cl(th, q, v, wk, form, g),
                                      argnums=(0, 1, 2)))
          for form, g in (("none", 1.0), ("full", 0.9), ("physics", 0.9), ("full", 0.5))}

    def rel(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))

    closed = {}
    for j in (4, 15):
        wk = jnp.asarray(np.eye(H, dtype=np.float32)[j])
        g = {k: [np.asarray(x) for x in f(theta, q0, v0, wk)] for k, f in cl.items()}
        g1 = g[("none", 1.0)]
        r = dict(norm_theta_gamma1=[float(np.linalg.norm(g1[0][t])) for t in range(H)])
        for form, gamma in (("full", 0.9), ("physics", 0.9)):
            gd = g[(form, gamma)]
            per_t = [rel(gd[0][t], gamma ** (j - t) * g1[0][t].astype(np.float64)) for t in range(j + 1)]
            r[form] = dict(gamma=gamma, rel_theta_t=per_t, rel_theta_max=max(per_t),
                           rel_theta_median=float(np.median(per_t)),
                           rel_qpos0=rel(gd[1], gamma ** (j + 1) * g1[1].astype(np.float64)),
                           rel_qvel0=rel(gd[2], gamma ** (j + 1) * g1[2].astype(np.float64)),
                           zero_after_j=bool(not np.any(gd[0][j + 1:]) and not np.any(g1[0][j + 1:])))
        # gamma a power of 2: the scaled cotangents round like the unscaled ones, so the law holds bitwise
        g5 = g[("full", 0.5)]
        r["full_gamma0.5_bitwise_t"] = int(sum(same_bits(g5[0][t], (g1[0][t] * np.float32(0.5 ** (j - t))))
                                               for t in range(j + 1)))
        r["full_gamma0.5_bitwise_qpos0_qvel0"] = bool(same_bits(g5[1], g1[1] * np.float32(0.5 ** (j + 1)))
                                                      and same_bits(g5[2], g1[2] * np.float32(0.5 ** (j + 1))))
        closed[f"j{j}"] = r
    res["closed_loop_law"] = dict(
        H=H, B=B, feedback="ctrl = ctrl_corpus + theta_t - 0.5 (q_t - q_ref) on the hinges", loss="the cost of step j "
        "alone (sum of root z and qvel[6] of its output)", carried_leaves="q, v, w (float32), n (int32), alive (bool), "
        "key (typed PRNG key)", law="full: d cost_j / d theta_t = gamma^(j - t) x (gamma 1), d / d(qpos_0, qvel_0) = "
        "gamma^(j + 1) x (gamma 1); physics: not this law under feedback (reported)", gate_rel=1e-4, per_j=closed)
    for k, v in closed.items():  # the check resolves the two forms: the physics-only one 100x off the law at least
        v["physics_resolved"] = bool(v["physics"]["rel_theta_max"] >= 100 * max(v["full"]["rel_theta_max"], 1e-7))
    res["closed_loop_pass"] = bool(all(
        v["full"]["rel_theta_max"] <= 1e-4 and v["full"]["rel_qpos0"] <= 1e-4 and v["full"]["rel_qvel0"] <= 1e-4
        and v["full"]["zero_after_j"] and min(v["norm_theta_gamma1"][:int(k[1:]) + 1]) > 0
        and v["full_gamma0.5_bitwise_t"] == int(k[1:]) + 1 and v["full_gamma0.5_bitwise_qpos0_qvel0"]
        and v["physics_resolved"] for k, v in closed.items()))
    ok &= res["closed_loop_pass"]
    print("options closed loop", json.dumps({k: (v["full"]["rel_theta_max"], v["full"]["rel_qpos0"],
                                                 v["physics"]["rel_theta_max"], v["physics"]["rel_theta_median"],
                                                 v["full_gamma0.5_bitwise_t"]) for k, v in closed.items()}),
          flush=True)
    # the soft clamp through the interface
    a = jnp_inputs(z, B)
    cq, cv = jnp.asarray(z["cot0_qpos"][:B]), jnp.asarray(z["cot0_qvel"][:B])

    def vjp_step(a, soft, kv):
        outs, f = jax.vjp(lambda *x: F.step(*x, iterations=50, clamp_kv=kv, normal=0,
                                            clamp_soft=soft if kv == 2 else None), *a)
        return f((cq, cv, jnp.zeros_like(outs[2])))

    @jax.jit
    def manual(*a):
        _, _, _, _, tp = F.forward(*a, iterations=50, tape=True)
        return F.backward(tp, a[4], a[5], cq, cv, clamp_kv=2, normal=0, ctrl=a[3], clamp_soft=CHECK_SOFT)

    gv_, gm = jax.jit(lambda *a: vjp_step(a, CHECK_SOFT, 2))(*a), manual(*a)
    r = dict(vjp_vs_backward=bool(same_bits(gv_[0], gm[0]) and same_bits(gv_[1], gm[1]) and same_bits(gv_[3], gm[2])))
    g0 = jax.jit(lambda *a: vjp_step(a, 0.0, 2))(*a)
    g1 = jax.jit(lambda *a: vjp_step(a, None, 1))(*a)
    r["soft0_vs_kv1"] = bool(all(same_bits(x, y) for x, y in zip(g0, g1)))
    _, dr = grads("decay", dict(clamp_kv=2), 1.0, remat=True)
    r["remat_fwd_calls"] = dr["fwd_calls"]
    errs = {}
    for name, f in (("clamp_soft_with_kv0", lambda: F.make_step(clamp_kv=0, clamp_soft=0.2)),
                    ("negative_clamp_soft", lambda: F.make_step(clamp_kv=2, clamp_soft=-0.1)),
                    ("nan_clamp_soft", lambda: F.make_step(clamp_kv=2, clamp_soft=float("nan"))),
                    ("nan_grad_decay", lambda: F.make_step(grad_decay=float("nan"))),
                    ("kv2_without_ctrl", lambda: F.backward(jnp.zeros((B, 4 * F.rec_words()), jnp.float32), a[4], a[5],
                                                            cq, cv, clamp_kv=2))):
        try:
            f()
            errs[name] = "no exception"
        except ValueError as e:
            errs[name] = f"ValueError: {e}"
    r["argument_errors"] = errs
    r["pass"] = bool(r["vjp_vs_backward"] and r["soft0_vs_kv1"] and r["remat_fwd_calls"] == H
                     and all(v.startswith("ValueError") for v in errs.values()))
    ok &= r["pass"]
    res["soft_clamp"] = r
    res["pass"] = bool(ok)
    print("options", json.dumps(res), flush=True)
    return res


# ---------------------------------------------------------------- api

def _raises(f):
    """(raised, exception type and message) of calling f and waiting for its result."""
    import jax
    try:
        jax.block_until_ready(f())
    except Exception as e:  # noqa: BLE001  (XLA's runtime error type differs between versions)
        return True, f"{type(e).__name__}: {str(e)[:300]}"
    return False, "no exception"


def sec_api(args):
    import copy
    import jax
    import jax.numpy as jnp
    import mujoco
    from nablaengine import jax_ffi as F
    m, mt = fixtures.load_model()
    fp, torso = np.asarray(mt["foot_pairs"]), fixtures.torso_id(m, mt)
    z = fixtures.load_corpus(CORPORA[0], adjoint=True)
    info = F.init()
    res, ok = {}, True

    # EnginePhysics as WarpPhysics: attributes, stats() (non-destructive, the last call of its batch size), diagnostics
    m10 = copy.deepcopy(m)
    m10.opt.iterations, m10.opt.ls_iterations = 10, 20  # the interface's solver options are not model constants
    m10.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    phys = {B: F.EnginePhysics(m10, B, fp, torso, graph_mode="warp_staged", nconmax=64, njmax=192)
            for B in (4096, 160)}
    p4 = phys[4096]
    r = dict(nconmax=p4.nconmax, njmax=p4.njmax, iterations=p4.iterations, stats_before=p4.stats())
    expect = {}
    for B, p in phys.items():
        a = jnp_inputs(z, B)
        if B == 4096:  # the differentiated path: jax.vjp (dk_step_fwd), then the backward
            outs, f = jax.jit(lambda *x: jax.vjp(p.step, *x))(*a)
            jax.block_until_ready(f((jnp.asarray(z["cot0_qpos"][:B]), jnp.asarray(z["cot0_qvel"][:B]),
                                     jnp.zeros_like(outs[2]))))
        else:  # the forward only: dk_step
            jax.block_until_ready(jax.jit(p.step)(*a))
        _, _, _, inf = jax.jit(lambda *x: F.forward(*x, iterations=10))(*a)  # the same forward's info
        inf = np.asarray(inf)
        expect[B] = {"nacon": int(inf[:, 1].sum()), "naconmax": 64 * B, "nefc_max": int(inf[:, 2].max()),
                     "njmax": 192}
    st = {B: p.stats() for B, p in phys.items()}
    st2 = {B: p.stats() for B, p in phys.items()}  # a read: the same again
    r.update(stats=st, expected_from_forward_info=expect, stats_repeat_equal=st == st2,
             keys_as_warp_physics=all(set(s) == {"nacon", "naconmax", "nefc_max", "njmax"} for s in st.values()),
             json=json.dumps(st[4096]), last_forward_any=F.last_forward(), diagnostics=p4.diagnostics(reset=True))
    r["pass"] = bool(r["nconmax"] == info["maxcon"] == 64 and r["njmax"] == 192 and r["iterations"] == 10
                     and r["keys_as_warp_physics"] and r["stats_repeat_equal"] and all(st[B] == expect[B] for B in st)
                     and r["diagnostics"]["fwd_calls"] >= 2)
    res["engine_physics"] = r
    ok &= r["pass"]

    # the model check: another timestep, foot-pair order, torso id or physics constant raises ValueError
    def variant(change):
        mm = copy.deepcopy(m)
        args_ = [mm, 4096, fp.copy(), torso]
        change(mm, args_)
        return args_

    cases = dict(timestep=lambda mm, a: setattr(mm.opt, "timestep", 0.004),
                 foot_pair_order=lambda mm, a: a.__setitem__(2, fp[::-1].copy()),
                 torso_id=lambda mm, a: a.__setitem__(3, torso + 1),
                 body_mass=lambda mm, a: mm.body_mass.__setitem__(5, mm.body_mass[5] * 1.01),
                 pair_friction=lambda mm, a: mm.pair_friction.__setitem__((0, slice(0, 2)), 0.5))
    mc = {}
    for name, change in cases.items():
        try:
            F.EnginePhysics(*variant(change))
            mc[name] = dict(raised=False)
        except ValueError as e:
            mc[name] = dict(raised=True, message=str(e)[:400])
    mc["pass"] = all(v["raised"] for v in mc.values())
    res["model_check"] = mc
    ok &= mc["pass"]

    # negative shape tests through jax.ffi.ffi_call: wrong output or input shapes raise (before any kernel runs) and
    # leave the library working: the valid calls before and after are bitwise the same, the counters count them only
    B = 160
    a = jnp_inputs(z, B)
    cq, cv = jnp.asarray(z["cot0_qpos"][:B]), jnp.asarray(z["cot0_qvel"][:B])
    f32, i32 = jnp.float32, jnp.int32
    nq, nv, nu, rw, mc8 = info["nq"], info["nv"], info["nu"], info["rec_words"], info["maxcon"] * 8
    S = jax.ShapeDtypeStruct
    fwd_outs = lambda info_shape=(B, 4), tape_cols=4 * rw, con=(B, mc8): [  # noqa: E731
        S((B, nq), f32), S((B, nv), f32), S((B, nv), f32), S(info_shape, i32), S(con, f32), S((B, tape_cols), f32)]
    bwd_outs = lambda info_shape=(B,), gq=(B, nq): [S(gq, f32), S((B, nv), f32), S((B, nu), f32),  # noqa: E731
                                                    S(info_shape, i32), S((B, nv), f32)]
    attrs = dict(iterations=np.int32(10), substeps=np.int32(4))
    battrs = dict(substeps=np.int32(4), clamp_kv=np.int32(0), normal=np.int32(3))
    T = F._STATE["targets"]

    def valid():
        q, v, w, inf, tp = jax.jit(lambda *x: F.forward(*x, tape=True))(*a)
        g = jax.jit(lambda tp: F.backward(tp, a[4], a[5], cq, cv))(tp)
        return [np.asarray(x) for x in (q, v, w, inf, tp) + tuple(g)]

    before = valid()
    F.read_diagnostics(reset=True)
    neg = {}
    force, kp = jnp.zeros((B, nv), f32), jnp.ones((B,), f32)
    neg["step_info_B3"] = _raises(lambda: jax.jit(lambda *x: jax.ffi.ffi_call(T["step"], fwd_outs((B, 3))[:5])(
        *x, force, kp, **attrs))(*a))
    neg["step_info_rank1"] = _raises(lambda: jax.jit(lambda *x: jax.ffi.ffi_call(T["step"], fwd_outs((B,))[:5])(
        *x, force, kp, **attrs))(*a))
    neg["fwd_tape_short"] = _raises(lambda: jax.jit(lambda *x: jax.ffi.ffi_call(T["fwd"], fwd_outs(tape_cols=4 * rw - 1))(
        *x, force, kp, **attrs))(*a))
    neg["fwd_contacts_short"] = _raises(lambda: jax.jit(lambda *x: jax.ffi.ffi_call(T["fwd"], fwd_outs(con=(B, 8)))(
        *x, force, kp, **attrs))(*a))
    neg["fwd_qpos_rank0"] = _raises(lambda: jax.jit(lambda *x: jax.ffi.ffi_call(T["fwd"], fwd_outs())(
        jnp.zeros((), f32), *x[1:], force, kp, **attrs))(*a))
    tp0 = jnp.asarray(before[4])
    neg["bwd_info_B2"] = _raises(lambda: jax.jit(lambda tp: jax.ffi.ffi_call(T["bwd"], bwd_outs((B, 2)))(
        tp, a[4], a[5], force, kp, cq, cv, **battrs))(tp0))
    neg["bwd_gqpos_wide"] = _raises(lambda: jax.jit(lambda tp: jax.ffi.ffi_call(T["bwd"], bwd_outs(gq=(B, nq + 1)))(
        tp, a[4], a[5], force, kp, cq, cv, **battrs))(tp0))
    neg["bwd_tape_rank1"] = _raises(lambda: jax.jit(lambda tp: jax.ffi.ffi_call(T["bwd"], bwd_outs())(
        tp.reshape(-1), a[4], a[5], force, kp, cq, cv, **battrs))(tp0))
    d = F.read_diagnostics(reset=True)
    after = valid()
    res["negative_shapes"] = dict(
        cases={k: dict(raised=v[0], error=v[1]) for k, v in neg.items()},
        counters_after_failed_calls=dict(fwd_calls=d["fwd_calls"], bwd_calls=d["bwd_calls"]),
        valid_calls_bitwise_before_after=all(same_bits(x, y) for x, y in zip(before, after)))
    res["negative_shapes"]["pass"] = bool(all(v[0] and "nablaengine.jax_ffi" in v[1] for v in neg.values())
                                          and d["fwd_calls"] == 0 and d["bwd_calls"] == 0
                                          and res["negative_shapes"]["valid_calls_bitwise_before_after"])
    ok &= res["negative_shapes"]["pass"]

    # k_adjoint's launch (adjoint.adj_launch_smem): on V100 at most 4 worlds per SM
    li = F.launch_info()
    cc = jax.devices()[0].compute_capability
    res["adjoint_launch"] = dict(compute_capability=str(cc), variants=li, sizeof_adj=info["sizeof_adj"])
    res["adjoint_launch"]["pass"] = bool(all(v["smem"] >= info["sizeof_adj"][k] and v["worlds_per_sm"] > 0
                                             for k, v in li.items())
                                         and (str(cc) != "7.0" or all(v["worlds_per_sm"] == 4 for v in li.values())))
    ok &= res["adjoint_launch"]["pass"]
    res["pass"] = bool(ok)
    print("api", json.dumps({k: v.get("pass") if isinstance(v, dict) else v for k, v in res.items()}),
          json.dumps(res["negative_shapes"]["cases"])[:2000], flush=True)
    return res


# ---------------------------------------------------------------- timing

def _time(f, reps):
    import jax
    jax.block_until_ready(f())
    jax.block_until_ready(f())
    t = time.perf_counter()
    out = None
    for _ in range(reps):
        out = f()
    jax.block_until_ready(out)
    thr = (time.perf_counter() - t) / reps
    lat = []
    for _ in range(min(reps, 10)):
        t = time.perf_counter()
        jax.block_until_ready(f())
        lat.append(time.perf_counter() - t)
    return 1e3 * thr, 1e3 * float(np.median(lat))


def sec_timing(args):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    m, mt = fixtures.load_model()
    z = fixtures.load_corpus(CORPORA[0], adjoint=True)
    nw = len(z["qpos"])
    a = jnp_inputs(z)
    cq, cv = jnp.asarray(z["cot0_qpos"]), jnp.asarray(z["cot0_qvel"])
    work = workdir.work_dir("jax_ffi_timing_")
    binary = checks.build(m, mt, work, "adj_kv0_n3", ["DK_ADJ_CLAMP_KV=0", "DK_ADJ_STAGE=3", "DK_ADJ_NORMAL=3"])
    checks.write_inputs(work, checks.corpus_inputs(z))
    res = dict(nworld=nw, reps=args.reps, repeats=args.timing_repeats, options=dict(substeps=4, clamp_kv=0, normal=3),
               note="medians over the repeats (bare kernels and FFI calls interleaved); V100 clocks vary by ~1 ms "
                    "between processes", iterations={})
    for iters in (10, 50):
        step = jax.jit(lambda *x, it=iters: F.step(*x, iterations=it))
        fwd = jax.jit(lambda *x, it=iters: F.forward(*x, iterations=it, tape=True))
        _, _, _, _, tp = fwd(*a)
        bwd = jax.jit(lambda tp: F.backward(tp, a[4], a[5], cq, cv))

        def vjp_fn(*x, it=iters):
            outs, f = jax.vjp(lambda *y: F.step(*y, iterations=it), *x)
            return f((cq, cv, jnp.zeros_like(outs[2])))
        vjp = jax.jit(vjp_fn)
        calls = (("step", lambda: step(*a)), ("forward_tape", lambda: fwd(*a)), ("backward", lambda: bwd(tp)),
                 ("vjp", lambda: vjp(*a)))
        bare_all = dict(forward_ms=[], forward_tape_ms=[], adjoint_ms=[], fused_forward_tape_ms=[])
        ffi_all = {name: dict(ms_per_call=[], ms_latency_median=[]) for name, _ in calls}
        for rep in range(args.timing_repeats):
            r, _ = checks.run(binary, work, m, nw, iters, 4, 1, args.reps, False, args.timeout)  # forward + tape, adjoint
            bare_all["fused_forward_tape_ms"].append(r["ms_forward_tape"])  # (k_substep: not the FFI's forward)
            bare_all["adjoint_ms"].append(r["ms_adjoint"])
            out = work / f"esc_it{iters}_{rep}.json"  # the split forward without / with the tape (T7, one binary)
            subprocess.run([sys.executable, str(ROOT / "scripts" / "engine_step_check.py"), "--corpus", CORPORA[0],
                            "--substeps", "4", "--iterations", str(iters), "--check", "8", "--reps", str(args.reps),
                            "--tape", "--digests", "off", "--out", str(out)], check=True, timeout=args.timeout,
                           stdout=subprocess.DEVNULL)
            t7 = json.loads(out.read_text())["tape"]["T7_timing"]
            bare_all["forward_ms"].append(float(np.median(t7["ms_tape_off"])))
            bare_all["forward_tape_ms"].append(float(np.median(t7["ms_tape_on"])))
            for name, f in calls:
                thr, lat = _time(f, args.reps)
                ffi_all[name]["ms_per_call"].append(thr)
                ffi_all[name]["ms_latency_median"].append(lat)
        med = lambda x: float(np.median(x))  # noqa: E731
        bare = {k: med(v) for k, v in bare_all.items()}
        ffi = {name: {k: med(v) for k, v in d.items()} for name, d in ffi_all.items()}
        over = dict(step_minus_bare_forward_ms=ffi["step"]["ms_per_call"] - bare["forward_ms"],
                    forward_tape_minus_bare_ms=ffi["forward_tape"]["ms_per_call"] - bare["forward_tape_ms"],
                    backward_minus_bare_ms=ffi["backward"]["ms_per_call"] - bare["adjoint_ms"],
                    vjp_minus_bare_ms=ffi["vjp"]["ms_per_call"] - bare["forward_tape_ms"] - bare["adjoint_ms"])
        res["iterations"][str(iters)] = dict(bare_kernels=bare, ffi=ffi, overhead=over, bare_kernels_all=bare_all,
                                             ffi_all=ffi_all)
        print("timing", iters, json.dumps(dict(bare_kernels=bare, ffi=ffi, overhead=over)), flush=True)
    res["adjoint_launch"] = F.launch_info()  # the FFI's k_adjoint launch (adj_launch_smem), as the binary's
    res["pass"] = True  # reported, not gated
    return res


RUN = dict(bitwise=sec_bitwise, jit=sec_jit, window=sec_window, zeros=sec_zeros, nonfinite=sec_nonfinite,
           options=sec_options,
           api=sec_api, timing=sec_timing, window_remat=sec_window_remat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--section", default="", help="internal: run one section in this process, JSON to --out")
    ap.add_argument("--corpora", default=",".join(fixtures.MAIN_CORPORA), help="comma-separated corpus names")
    ap.add_argument("--forward-ref", default="self",
                    help="bitwise: the forward reference at 50 iterations: self (this tree's engine_step_check.py), "
                         "an npz of engine_step_check.py --save-outputs ({corpus}: the corpus name), or off")
    ap.add_argument("--variants", default="", help="bitwise: the adjoint variants 'clamp_kv,normal' separated by ';' "
                    "(default all of CHECK_VARIANTS)")
    ap.add_argument("--bitwise-iterations", default="50,10", help="bitwise: the iteration counts")
    ap.add_argument("--remat-mode", default="none", choices=REMAT_MODES, help="internal (window_remat)")
    ap.add_argument("--remat-grads", default="", help="internal (window_remat): npz of the gradients")
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--window-worlds", type=int, default=4096)
    ap.add_argument("--window-reps", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20, help="timing: calls / launches")
    ap.add_argument("--timing-repeats", type=int, default=3, help="timing: repeats (medians)")
    ap.add_argument("--update", action="store_true",
                    help="replace only the run sections in an existing --out JSON (the others are kept)")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", required=True)
    workdir.add_argument(ap)
    args = ap.parse_args()
    workdir.configure(args)
    CORPORA[:] = [c for c in args.corpora.split(",") if c]
    if args.section:
        r = RUN[args.section](args)
        Path(args.out).write_text(json.dumps(r, indent=1))
        return
    import jax  # noqa: F401  (the environment is recorded by the parent)
    res = dict(gpu=gpu_name(), jax=jax.__version__, corpora=CORPORA, sections={})
    if args.update and Path(args.out).exists():
        res = json.loads(Path(args.out).read_text())
        assert res["gpu"] == gpu_name(), "--update: the existing results are from another GPU"
        provenance.keep_sections(res, [(res, [s for s in res["sections"] if s not in args.sections.split(",")])])
    from nablaengine import jax_ffi as F
    t = time.perf_counter()
    lib = F.init()  # its path absolute; recorded relative (provenance.rel)
    lib = dict(lib, path=provenance.rel(lib["path"]), init_seconds=time.perf_counter() - t, stats=dict(F.STATS))
    res["library_last_update" if args.update and "library" in res else "library"] = lib
    work = workdir.work_dir("jax_ffi_check_")
    for sec in args.sections.split(","):
        part = work / f"{sec}.json"
        t = time.perf_counter()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--section", sec, "--out", str(part)]
        for k in ("corpora", "forward_ref", "variants", "bitwise_iterations", "horizon", "window_worlds",
                  "window_reps", "reps", "timing_repeats", "timeout"):
            cmd += [f"--{k.replace('_', '-')}", str(getattr(args, k))]
        p = subprocess.run(cmd, timeout=args.timeout * 4)
        if p.returncode or not part.exists():
            res["sections"][sec] = dict(error=f"exit code {p.returncode}")
            res["sections"][sec]["pass"] = False
            continue
        r = json.loads(part.read_text())
        r["seconds"] = time.perf_counter() - t
        res["sections"][sec] = r
        print(f"== {sec}: pass {r.get('pass')} ({r['seconds']:.0f} s)", flush=True)
    res["pass"] = bool(all(v.get("pass") for v in res["sections"].values()))
    res["provenance"] = provenance.provenance()  # commit, GPU, driver, nvcc, packages, MuJoCo Warp
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(pass_=res["pass"], sections={k: v.get("pass") for k, v in res["sections"].items()})))
    return 0 if res["pass"] else 1  # the exit status (a --section child exits 0: its parent reads the JSON)


if __name__ == "__main__":
    sys.exit(main())
