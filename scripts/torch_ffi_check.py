"""Check of nablaengine.torch_ffi against nablaengine.jax_ffi on the same GPU: bitwise equality of every output and
gradient, plus CUDA-graph capture and torch.compile of the torch entries.

Sections (all by default; --sections a,b,...):
  forward    forward(..., tape=True) and forward(...) without tape: qpos', qvel', warm', info, tape (and the forward
             without tape equals the taped one's outputs)
  backward   backward() of the tape for the corpus's 3 stored cotangents in every adjoint variant (clamp_kv 0 / 1 / 2,
             normal 0..3; clamp_kv 2 at the default delta): g_qpos, g_qvel, g_ctrl, info
  step       torch.autograd through step() (default options, and clamp_kv=1 normal=0 grad_decay=0.98) against jax.vjp
             of jax_ffi.step for the same cotangent
  kinematics kinematics() outputs and VJP, and step(..., kinematics=True) with all eight cotangents, against jax_ffi
  rollout    a 3-step rollout (ctrl per step requires grad, the state carried, scale_grad 0.98 on the state at the top
             of each step, loss = cotangents . final state): gradients of the 3 controls and the initial state
  diag       the device counters after one forward with tape and one backward: equal to jax_ffi.read_diagnostics()
  graph      forward with tape + backward captured in one CUDA graph and replayed on new inputs: equal to eager calls
  compile    torch.compile(fullgraph=True) of a function calling step and summing a loss: no graph break, values equal
  multigpu   (two or more visible GPUs of one architecture) forward with tape + backward + kinematics on the last GPU
             while the first is current, and the same calls on both GPUs alternately: bitwise equal to the first GPU's
             results; per-device diagnostic counters; jax_ffi.step and its VJP on the last JAX device equal to the first
             (the JAX interface's multi-device path); skipped (recorded) with one GPU
  artifact   the module's interface constants equal jax_ffi's; a fresh process that cannot import mujoco or jax and
             has no nvcc on PATH loads the current artifact and computes forward with tape + backward bitwise equal to
             this process; an artifact whose recorded source hash differs is refused
  overflow   states pushed 0.6 m into the floor with scrambled joints (contacts beyond 64 / rows beyond 192 per world):
             torch and jax_ffi forward bitwise equal (info's overflow bits included), their overflow counters equal and
             equal to the info bits; no warning by default, and read_diagnostics() warns in both with
             set_overflow_warnings(True)
  timing     ms per control step at --nworld (interface defaults), torch against jax_ffi in the same process: the
             forward (torch step without grad / jax.jit(step)) and forward + backward (torch autograd of a cotangent
             loss / jax.jit(jax.vjp)), wall time of --reps back-to-back calls ending in one synchronisation, --rounds
             rounds (the first dropped), median and min; recorded, never gated (run it on an idle GPU)

  PYTHONPATH=src python scripts/torch_ffi_check.py --out runs/torch_ffi_check.json [--nworld 1024]
Exit 1 if any comparison fails. JAX runs with XLA_PYTHON_CLIENT_PREALLOCATE=false (both frameworks on one GPU).
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

SECTIONS = ("forward", "backward", "step", "kinematics", "rollout", "diag", "graph", "compile", "multigpu", "artifact", "overflow", "timing")


def bits_equal(a, b):
    """Bitwise equality of two arrays (NaN payloads included)."""
    a, b = np.ascontiguousarray(np.asarray(a)), np.ascontiguousarray(np.asarray(b))
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="mf_s0")
    ap.add_argument("--nworld", type=int, default=1024)
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=6)
    a = ap.parse_args()
    sections = [s for s in a.sections.split(",") if s]
    bad = [s for s in sections if s not in SECTIONS]
    if bad:
        ap.error(f"unknown sections {bad}")
    out = os.path.abspath(a.out)
    if os.path.exists(out) and os.path.relpath(out).startswith("docs"):
        ap.error("--out must not overwrite a committed result")

    import torch
    import jax
    import jax.numpy as jnp
    from nablaengine import fixtures, jax_ffi, provenance, torch_ffi

    info = torch_ffi.init()
    z = fixtures.load_corpus(a.corpus, adjoint=True)
    B = min(a.nworld, len(z["qpos"]))
    names = ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction", "com")
    host = [np.ascontiguousarray(z[k][:B], np.float32) for k in names]
    cots = [(np.ascontiguousarray(z[f"cot{c}_qpos"][:B], np.float32),
             np.ascontiguousarray(z[f"cot{c}_qvel"][:B], np.float32)) for c in range(3)]
    dev = torch.device("cuda", torch.cuda.current_device())
    T = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)  # noqa: E731
    J = jnp.asarray
    tin = [T(x) for x in host]
    res = dict(corpus=a.corpus, nworld=B, sections={}, provenance=provenance.provenance(),
               torch=torch.__version__, torch_cuda=torch.version.cuda, library=provenance.rel(info["path"]))
    fails = []

    def record(sec, key, ok, **extra):
        res["sections"].setdefault(sec, {})[key] = dict(equal=bool(ok), **extra)
        if not ok:
            fails.append(f"{sec}.{key}")

    # the CUDA graph first, before JAX's GPU runtime starts in this process (with several GPUs, JAX's runtime makes
    # torch's allocations during a capture fail; a process without JAX is not affected)
    if "graph" in sections:
        cq, cv = cots[0]
        st = [t.clone() for t in tin]
        sc = [T(cq), T(cv)]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up on the capture stream
            for _ in range(2):
                f = torch_ffi.forward(*st, tape=True)
                torch_ffi.backward(f[4], st[4], st[5], *sc)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gf = torch_ffi.forward(*st, tape=True)
            gb = torch_ffi.backward(gf[4], st[4], st[5], *sc)
        shift = [np.roll(x, 1, axis=0) for x in host]  # new inputs into the static buffers
        for t, x in zip(st, shift):
            t.copy_(T(x))
        g.replay()
        torch.cuda.synchronize()
        ef = torch_ffi.forward(*[T(x) for x in shift], tape=True)
        eb = torch_ffi.backward(ef[4], T(shift[4]), T(shift[5]), *sc)
        ok = all(bits_equal(x.cpu().numpy(), y.cpu().numpy()) for x, y in zip(list(gf) + list(gb), list(ef) + list(eb)))
        record("graph", "replay_equals_eager", ok)
    jax_ffi.init()
    jin = [J(x) for x in host]
    def cmp(sec, key, t, j):
        record(sec, key, bits_equal(t.detach().cpu().numpy(), np.asarray(j)))

    t0 = time.time()
    if "forward" in sections or "backward" in sections or "diag" in sections or "graph" in sections:
        tf = torch_ffi.forward(*tin, tape=True)
        jf = jax_ffi.forward(*jin, tape=True)
    if "forward" in sections:
        for k, t, j in zip(("qpos1", "qvel1", "warm1", "info", "tape"), tf, jf):
            cmp("forward", k, t, j)
        tn = torch_ffi.forward(*tin)
        for k, t, j in zip(("qpos1", "qvel1", "warm1", "info"), tn, tf[:4]):
            record("forward", f"no_tape_{k}", bits_equal(t.cpu().numpy(), j.cpu().numpy()))
    if "backward" in sections:
        for kv, n in jax_ffi.VARIANTS:
            for c, (cq, cv) in enumerate(cots):
                kw = dict(clamp_kv=kv, normal=n)
                tb = torch_ffi.backward(tf[4], tin[4], tin[5], T(cq), T(cv), ctrl=tin[3] if kv == 2 else None, **kw)
                jb = jax_ffi.backward(jf[4], jin[4], jin[5], J(cq), J(cv), ctrl=jin[3] if kv == 2 else None, **kw)
                for k, t, j in zip(("g_qpos", "g_qvel", "g_ctrl", "info"), tb, jb):
                    cmp("backward", f"kv{kv}_n{n}_cot{c}_{k}", t, j)
    if "step" in sections:
        cq, cv = cots[0]
        for tag, kw in (("default", {}), ("kv1_n0_decay0.98", dict(clamp_kv=1, normal=0, grad_decay=0.98))):
            x = [t.clone().requires_grad_(i in (0, 1, 3)) for i, t in enumerate(tin)]
            q1, v1, w1 = torch_ffi.step(*x, **kw)
            (q1 * T(cq)).sum().add_((v1 * T(cv)).sum()).backward()  # d/dx of cq . q1 + cv . v1
            tg = [x[i].grad for i in (0, 1, 3)]
            jo, vjp = jax.vjp(lambda q, v, u: jax_ffi.step(q, v, jin[2], u, jin[4], jin[5], **kw)[:2], jin[0], jin[1],
                              jin[3])
            jg = vjp((J(cq), J(cv)))
            cmp("step", f"{tag}_qpos1", q1, jo[0])
            for k, t, j in zip(("g_qpos", "g_qvel", "g_ctrl"), tg, jg):
                cmp("step", f"{tag}_{k}", t, j)
    if "kinematics" in sections:
        rng = np.random.default_rng(0)
        tk = torch_ffi.kinematics(tin[0], tin[1], tin[5])
        jk = jax_ffi.kinematics(jin[0], jin[1], jin[5])
        kn = ("qpos_n", "xpos", "xquat", "cvel", "subtree_com")
        for k, t, j in zip(kn, tk, jk):
            cmp("kinematics", k, t, j)
        g = [rng.standard_normal(t.shape).astype(np.float32) for t in tk]
        x = [tin[0].clone().requires_grad_(), tin[1].clone().requires_grad_()]
        y = torch_ffi.kinematics(x[0], x[1], tin[5])
        sum((yi * T(gi)).sum() for yi, gi in zip(y, g)).backward()
        _, vjp = jax.vjp(lambda q, v: jax_ffi.kinematics(q, v, jin[5]), jin[0], jin[1])
        jg = vjp(tuple(J(gi) for gi in g))
        cmp("kinematics", "vjp_g_qpos", x[0].grad, jg[0])
        cmp("kinematics", "vjp_g_qvel", x[1].grad, jg[1])
        # step with kinematics outputs, a cotangent for each of the eight outputs (warm' has none)
        cq, cv = cots[1]
        x = [t.clone().requires_grad_(i in (0, 1, 3)) for i, t in enumerate(tin)]
        r = torch_ffi.step(*x, kinematics=True)
        loss = (r[0] * T(cq)).sum() + (r[1] * T(cv)).sum() + sum((ri * T(gi)).sum() for ri, gi in zip(r[3:], g))
        loss.backward()
        jo, vjp = jax.vjp(lambda q, v, u: (lambda o: (o[0], o[1]) + tuple(o[3:]))(
            jax_ffi.step(q, v, jin[2], u, jin[4], jin[5], kinematics=True)), jin[0], jin[1], jin[3])
        jg = vjp((J(cq), J(cv)) + tuple(J(gi) for gi in g))
        for k, t, j in zip(kn, r[3:], jo[2:]):
            cmp("kinematics", f"step_{k}", t, j)
        for k, t, j in zip(("g_qpos", "g_qvel", "g_ctrl"), (x[0].grad, x[1].grad, x[3].grad), jg):
            cmp("kinematics", f"step_{k}", t, j)
    if "rollout" in sections:
        cq, cv = cots[2]
        ctrls = [np.ascontiguousarray(z["ctrl"][(np.arange(B) + 97 * s) % len(z["ctrl"])], np.float32)
                 for s in range(3)]
        xs = [tin[0].clone().requires_grad_(), tin[1].clone().requires_grad_()]
        us = [T(u).requires_grad_() for u in ctrls]
        q, v, w = xs[0], xs[1], tin[2]
        for u in us:
            q, v, w = torch_ffi.scale_grad((q, v, w), 0.98)
            q, v, w = torch_ffi.step(q, v, w, u, tin[4], tin[5])
        ((q * T(cq)).sum() + (v * T(cv)).sum()).backward()

        def jroll(q, v, *u):
            w = jin[2]
            for ui in u:
                q, v, w = jax_ffi.scale_grad((q, v, w), 0.98)
                q, v, w = jax_ffi.step(q, v, w, ui, jin[4], jin[5])
            return q, v

        jo, vjp = jax.vjp(jroll, jin[0], jin[1], *[J(u) for u in ctrls])
        jg = vjp((J(cq), J(cv)))
        cmp("rollout", "qpos3", q, jo[0])
        cmp("rollout", "qvel3", v, jo[1])
        for k, t, j in zip(("g_qpos0", "g_qvel0", "g_ctrl0", "g_ctrl1", "g_ctrl2"), xs + us, jg):
            cmp("rollout", k, t.grad, j)
    if "diag" in sections:
        torch.cuda.synchronize()
        # the counters are off by default ($DK_DIAGNOSTICS): this section compares them with jax_ffi's
        record("diag", "off_by_default", torch_ffi.set_diagnostics(True) is False)
        torch_ffi.read_diagnostics(reset=True)
        jax_ffi.read_diagnostics(reset=True)
        cq, cv = cots[0]
        tf2 = torch_ffi.forward(*tin, tape=True)
        torch_ffi.backward(tf2[4], tin[4], tin[5], T(cq), T(cv))
        jf2 = jax_ffi.forward(*jin, tape=True)
        jax.block_until_ready(jax_ffi.backward(jf2[4], jin[4], jin[5], J(cq), J(cv)))
        td, jd = torch_ffi.read_diagnostics(), jax_ffi.read_diagnostics()
        jd = {k: int(jd[k]) for k in torch_ffi.DIAG}
        record("diag", "counters", td == jd, torch=td, jax=jd)
    if "compile" in sections:
        cq, cv = T(cots[0][0]), T(cots[0][1])

        def f(q, v, w, u, fr, c):
            q1, v1, _ = torch_ffi.step(q, v, w, u, fr, c)
            return (q1 * cq).sum() + (v1 * cv).sum()

        x = [t.clone().requires_grad_(i in (0, 1, 3)) for i, t in enumerate(tin)]
        ex = torch._dynamo.explain(f)(*x)
        cf = torch.compile(f, fullgraph=True)
        lc = cf(*x)
        lc.backward()
        gc = [x[i].grad.clone() for i in (0, 1, 3)]
        x2 = [t.clone().requires_grad_(i in (0, 1, 3)) for i, t in enumerate(tin)]
        le = f(*x2)
        le.backward()
        ge = [x2[i].grad for i in (0, 1, 3)]
        record("compile", "graph_breaks", ex.graph_break_count == 0, graph_break_count=int(ex.graph_break_count))
        record("compile", "gradients_equal_eager",
               all(bits_equal(p.cpu().numpy(), q.cpu().numpy()) for p, q in zip(gc, ge)))
    if "multigpu" in sections:
        ng = torch.cuda.device_count()
        caps = {torch.cuda.get_device_capability(i) for i in range(ng)}
        if ng < 2 or len(caps) != 1:
            res["sections"]["multigpu"] = dict(skipped=f"{ng} visible GPU(s), architectures {sorted(caps)}")
        else:
            cq, cv = cots[0]
            d0, d1 = torch.device("cuda", 0), torch.device("cuda", ng - 1)

            def run(d):
                x = [t.to(d) for t in tin]
                f = torch_ffi.forward(*x, tape=True)
                b = torch_ffi.backward(f[4], x[4], x[5], T(cq).to(d), T(cv).to(d))
                k = torch_ffi.kinematics(f[0], f[1], x[5])
                return [t.cpu().numpy() for t in list(f) + list(b) + list(k)]

            torch.cuda.set_device(d0)
            torch_ffi.read_diagnostics(d0, reset=True)
            torch_ffi.init(d1)
            torch_ffi.read_diagnostics(d1, reset=True)
            r0 = run(d0)
            r1 = run(d1)  # d0 current, inputs on d1
            record("multigpu", "last_gpu_equals_first", all(bits_equal(p, q) for p, q in zip(r0, r1)),
                   devices=[0, ng - 1])
            r = [run(d) for d in (d1, d0, d1)]  # alternating
            record("multigpu", "alternating_equal", all(bits_equal(p, q) for x in r for p, q in zip(r0, x)))
            c0, c1 = torch_ffi.read_diagnostics(d0), torch_ffi.read_diagnostics(d1)
            record("multigpu", "per_device_counters", c0["fwd_calls"] == 2 and c1["fwd_calls"] == 3
                   and c0["bwd_calls"] == 2 and c1["bwd_calls"] == 3, cuda0=c0, cuda_last=c1)
            jd = jax.devices()
            j0, j1 = jd[0], jd[-1]
            outs = []
            for jdv in (j0, j1):
                xs = [jax.device_put(J(x), jdv) for x in host]
                o, vjp = jax.vjp(lambda q, v, u: jax_ffi.step(q, v, xs[2], u, xs[4], xs[5])[:2], xs[0], xs[1], xs[3])
                g = vjp((jax.device_put(J(cq), jdv), jax.device_put(J(cv), jdv)))
                outs.append([np.asarray(t) for t in list(o) + list(g)])
            record("multigpu", "jax_last_device_equals_first", all(bits_equal(p, q) for p, q in zip(*outs)),
                   devices=[str(j0), str(j1)])
    if "artifact" in sections:
        names = ("ITERATIONS", "SUBSTEPS", "CLAMP_KV", "NORMAL", "CLAMP_SOFT", "GRAD_DECAY", "DIAG", "OVERFLOW_CONTACTS",
                 "OVERFLOW_ROWS", "NONFINITE_QPOS",
                 "NONFINITE_QVEL", "NONFINITE_CTRL", "NONFINITE_COT", "NONFINITE_SHIFT", "VARIANTS", "FWD_UNIT")
        record("artifact", "constants_equal_jax_ffi",
               all(getattr(torch_ffi, n) == getattr(jax_ffi, n) for n in names))
        import subprocess
        import tempfile
        cq, cv = cots[0]
        tmp = tempfile.mkdtemp(prefix="torch_ffi_artifact_", dir=os.environ.get("TMPDIR"))
        try:
            np.savez(os.path.join(tmp, "in.npz"), *host, cq, cv)
            ref = torch_ffi.forward(*tin, tape=True)
            refb = torch_ffi.backward(ref[4], tin[4], tin[5], T(cq), T(cv))
            child = (
                "import sys, json, numpy as np\n"
                "class _Block:\n"
                "    def find_spec(self, name, path=None, target=None):\n"
                "        if name.split('.')[0] in ('mujoco', 'jax', 'jaxlib'): raise ImportError('blocked: ' + name)\n"
                "sys.meta_path.insert(0, _Block())\n"
                "import torch\n"
                "from nablaengine import torch_ffi\n"
                "art = sys.argv[3] if len(sys.argv) > 3 else None\n"
                "try:\n"
                "    torch_ffi.init(artifact=art)\n"
                "except RuntimeError as e:\n"
                "    print(json.dumps(dict(error=str(e)))); sys.exit(0)\n"
                "z = np.load(sys.argv[1]); a = [torch.from_numpy(z[f'arr_{i}']).cuda() for i in range(8)]\n"
                "f = torch_ffi.forward(*a[:6], tape=True)\n"
                "b = torch_ffi.backward(f[4], a[4], a[5], a[6], a[7])\n"
                "np.savez(sys.argv[2], *[t.cpu().numpy() for t in list(f) + list(b)])\n"
                "print(json.dumps(dict(mods=sorted(m for m in ('mujoco', 'jax', 'jaxlib', 'nablaengine.jax_ffi') "
                "if m in sys.modules))))\n")
            env = dict(os.environ, PATH="/usr/bin:/bin")  # no nvcc
            env.pop("NVCC", None)
            r = subprocess.run([sys.executable, "-c", child, os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                               timeout=600)
            last = (r.stdout.strip().splitlines() or ["{}"])[-1]
            try:
                msg = json.loads(last)
            except ValueError:
                msg = dict(error=r.stdout[-2000:])
            ok = r.returncode == 0 and "error" not in msg and msg.get("mods") == []
            if ok:
                z2 = np.load(os.path.join(tmp, "out.npz"))
                ok = all(bits_equal(t.cpu().numpy(), z2[f"arr_{i}"]) for i, t in enumerate(list(ref) + list(refb)))
            record("artifact", "load_without_mujoco_jax_nvcc_bitwise", ok, child=msg)
            js = json.loads(open(torch_ffi.init()["artifact"]).read())
            js["sources_sha256"] = "0" * 64
            bad = os.path.join(os.path.dirname(torch_ffi.init()["artifact"]), "torch_ffi_check_stale.json")
            with open(bad, "w") as fh:
                json.dump(js, fh)
            try:
                r = subprocess.run([sys.executable, "-c", child, os.path.join(tmp, "in.npz"),
                                    os.path.join(tmp, "out2.npz"), bad], env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
            finally:
                os.remove(bad)
            last = (r.stdout.strip().splitlines() or ["{}"])[-1]
            msg = json.loads(last) if last.startswith("{") else dict(error=r.stdout[-500:])
            err = msg.get("error", "")
            record("artifact", "stale_artifact_refused", "other package sources" in err,
                   child_error=err.rsplit(": ", 2)[-2:] if err else None)  # without the artifact's local path
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    if "overflow" in sections:
        import warnings
        rng = np.random.default_rng(0)
        q = host[0].copy()
        q[:, 2] -= 0.6
        q[:, 7:] += rng.uniform(-1.5, 1.5, q[:, 7:].shape).astype(np.float32)
        ov = [q] + host[1:]
        torch.cuda.synchronize()
        torch_ffi.set_diagnostics(True)  # the counters are off by default ($DK_DIAGNOSTICS)
        torch_ffi.read_diagnostics(reset=True)
        jax_ffi.read_diagnostics(reset=True)
        to = torch_ffi.forward(*[T(x) for x in ov])
        jo = jax_ffi.forward(*[J(x) for x in ov])
        for k, t, j in zip(("qpos1", "qvel1", "warm1", "info"), to, jo):
            cmp("overflow", k, t, j)
        u = to[3][:, 3].cpu().numpy()
        nc, nr = int(((u & torch_ffi.OVERFLOW_CONTACTS) != 0).sum()), int(((u & torch_ffi.OVERFLOW_ROWS) != 0).sum())
        with warnings.catch_warnings(record=True) as w0:  # off by default: counted, not warned
            warnings.simplefilter("always")
            td = torch_ffi.read_diagnostics(reset=False)
            jd = jax_ffi.read_diagnostics(reset=False)
        record("overflow", "no_warning_by_default", not any("capacity" in str(w.message) for w in w0))
        pt, pj = torch_ffi.set_overflow_warnings(True), jax_ffi.set_overflow_warnings(True)
        try:
            with warnings.catch_warnings(record=True) as wt:
                warnings.simplefilter("always")
                td = torch_ffi.read_diagnostics()
            with warnings.catch_warnings(record=True) as wj:
                warnings.simplefilter("always")
                jd = jax_ffi.read_diagnostics()
        finally:
            torch_ffi.set_overflow_warnings(pt)
            jax_ffi.set_overflow_warnings(pj)
        keys = ("fwd_contact_overflow_worlds", "fwd_row_overflow_worlds")
        record("overflow", "counters_equal_info_bits", nc > 0 and nr > 0 and
               (td[keys[0]], td[keys[1]]) == (jd[keys[0]], jd[keys[1]]) == (nc, nr),
               contact_overflow_worlds=nc, row_overflow_worlds=nr, nworld=B)
        warned = lambda ws: any(issubclass(w.category, RuntimeWarning) and "capacity" in str(w.message)  # noqa: E731
                                for w in ws)
        record("overflow", "warnings", warned(wt) and warned(wj))
    if "timing" in sections:
        cq, cv = cots[0]
        tcq, tcv, jcq, jcv = T(cq), T(cv), J(cq), J(cv)
        x = [t.clone().requires_grad_(i in (0, 1, 3)) for i, t in enumerate(tin)]

        def t_fwd():
            with torch.no_grad():
                return torch_ffi.step(*tin)[1]

        def t_vjp():
            q1, v1, _ = torch_ffi.step(*x)
            return torch.autograd.grad((q1 * tcq).sum() + (v1 * tcv).sum(), [x[0], x[1], x[3]])[2]

        jf = jax.jit(lambda *z: jax_ffi.step(*z)[1])
        jv = jax.jit(lambda q, v, u: jax.vjp(lambda q, v, u: jax_ffi.step(q, v, jin[2], u, jin[4], jin[5])[:2],
                                             q, v, u)[1]((jcq, jcv))[2])

        def clock(f, sync):
            out = []
            for _ in range(a.rounds + 1):
                sync(f())
                t = time.perf_counter()
                for _ in range(a.reps):
                    r = f()
                sync(r)
                out.append((time.perf_counter() - t) * 1e3 / a.reps)
            k = sorted(out[1:])
            return dict(median=round(k[len(k) // 2], 4), min=round(k[0], 4), rounds=[round(v, 4) for v in out[1:]])

        tsync = lambda r: torch.cuda.synchronize()  # noqa: E731
        jsync = jax.block_until_ready
        tim = dict(torch_forward=clock(t_fwd, tsync), jax_forward=clock(lambda: jf(*jin), jsync),
                   torch_forward_backward=clock(t_vjp, tsync),
                   jax_forward_backward=clock(lambda: jv(jin[0], jin[1], jin[3]), jsync))
        tim["ratio_forward"] = round(tim["torch_forward"]["median"] / tim["jax_forward"]["median"], 4)
        tim["ratio_forward_backward"] = round(tim["torch_forward_backward"]["median"]
                                              / tim["jax_forward_backward"]["median"], 4)
        res["timing"] = dict(nworld=B, reps=a.reps, rounds=a.rounds, unit="ms per control step", **tim)
    res["seconds"] = round(time.time() - t0, 1)
    res["failures"] = fails
    res["pass"] = not fails
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
        fh.write("\n")
    n = sum(len(v) for v in res["sections"].values())
    print(f"torch_ffi check ({a.corpus}, {B} worlds): {n - len(fails)}/{n} equal; "
          f"{'PASS' if not fails else 'FAIL: ' + ', '.join(fails[:10])}")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
