"""The Unitree H1 (fixtures/h1, nv 25: the solves' masked lanes past NV) against MuJoCo C in float64.

States: --states independent MuJoCo C rollouts of the H1 from the home keyframe (seed --seed), each with an initial
push qvel ~ N(0, 0.3) and PD setpoints at the keyframe plus N(0, 0.3) rad re-drawn every 10 control steps, sampled
after 5 .. 150 control steps: standing, staggering, falling and lying states. The step's setpoints are a new draw,
clipped to ctrlrange (MuJoCo clamps ctrl, the engine does not: H11c). Both simulators take the same float32 inputs;
one control step is 4 substeps at --iterations Newton iterations (MuJoCo: tolerance 1e-12), nominal friction, no COM
offset.

Sections (--sections, default all):
  capability  the model loads (nablaengine.fixtures.load_model(fixtures.H1), scope engine) with no error and exactly
              the warnings H3 (nv < 32), H5 (not the tuned nv 33-35 solves) and U2 (KDOF 11)
  forward     qvel' relative error per state (|d| / |ref|) and qpos' max abs error against MuJoCo; outputs finite and
              bitwise the same on a second call. Gate: median relative error <= 1e-5 (the tail is the contact-rich
              states, where float32 and float64 part ways: the same as for the G1, reported)
  gradient    the exact adjoint (clamp_kv=1, normal=0) of <c_v, qvel'> + <c_q, hinge qpos'> with random N(0, 1)
              cotangents, with respect to ctrl and qvel, against MuJoCo C central differences (eps 1e-6), relative
              error per state, on --grad-states states of two kinds: free flight (raised 3 m, hinges 0.15 rad inside
              their limits: no floor contact or limit rows; nablaengine.fixtures.free_flight_states) with gates
              median <= 1e-5 and p90 <= 1e-4, and the corpus states as drawn (contacts, limits) with median <= 1e-4.
              The tails of both are states at a kink, where central differences straddle it: an actuator force at
              its limit (jnt_actfrcrange: about one state in six is within 2 N m of one) or a self contact (also in
              free flight)

  PYTHONPATH=src python scripts/h1_check.py --out runs/h1_check.json

Exit status: 0 when every section passes, 1 otherwise (the JSON is written either way).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from nablaengine import fixtures, jax_ffi

SECTIONS = ("capability", "forward", "gradient")
GATES = dict(forward_median=1e-5, free_median=1e-5, free_p90=1e-4, contact_median=1e-4)
WARNINGS = {"H3", "H5", "U2"}


def corpus(m, n, seed, substeps):
    """n states (qpos, qvel, warm, ctrl), float32 values in float64 arrays."""
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    jq = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    out = {k: [] for k in ("qpos", "qvel", "warm", "ctrl")}
    for _ in range(n):
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        d.qvel[:] = rng.normal(0, 0.3, m.nv)
        base = d.qpos[jq].copy()
        for k in range(rng.integers(5, 150)):
            if k % 10 == 0:
                d.ctrl[:] = base + rng.normal(0, 0.3, m.nu)
            for _ in range(substeps):
                mujoco.mj_step(m, d)
        out["qpos"].append(d.qpos.copy())
        out["qvel"].append(d.qvel.copy())
        out["warm"].append(d.qacc_warmstart.copy())
        out["ctrl"].append(np.clip(base + rng.normal(0, 0.3, m.nu), *m.actuator_ctrlrange.T))
    return {k: np.array(v).astype(np.float32).astype(np.float64) for k, v in out.items()}


class Reference:
    """One control step of MuJoCo C (float64) from a corpus state."""

    def __init__(self, m, substeps):
        self.m, self.d, self.substeps = m, mujoco.MjData(m), substeps

    def __call__(self, qpos, qvel, warm, ctrl):
        d = self.d
        d.qpos[:], d.qvel[:], d.qacc_warmstart[:], d.ctrl[:], d.time = qpos, qvel, warm, ctrl, 0.0
        for _ in range(self.substeps):
            mujoco.mj_step(self.m, d)
        return d.qpos.copy(), d.qvel.copy(), d.ncon


def stats(x):
    return {k: float(f(x)) for k, f in (("median", np.median), ("p90", lambda v: np.percentile(v, 90)),
                                        ("p99", lambda v: np.percentile(v, 99)), ("max", np.max))}


def fmt(s):
    return "  ".join(f"{k} {v:.2e}" for k, v in s.items())


def section_capability(res, m, mt):
    from nablaengine.model import capability
    rep = capability.check(m, mt["foot_pairs"], mt["torso_body"], "engine")
    errors, warnings = sorted({f.code for f in rep.errors()}), sorted({f.code for f in rep.warnings()})
    print(f"capability: errors {errors}, warnings {warnings} (expected {sorted(WARNINGS)})")
    res["capability"] = dict(errors=errors, warnings=warnings)
    return not errors and set(warnings) == WARNINGS


def section_forward(res, m, S, ref, step, n):
    rq, rv, ncon = map(np.array, zip(*(ref(*(S[k][i] for k in ("qpos", "qvel", "warm", "ctrl"))) for i in range(n))))
    q1, v1 = step(S)
    q2, v2 = step(S)
    rel = np.linalg.norm(v1 - rv, axis=1) / np.maximum(np.linalg.norm(rv, axis=1), 1e-6)
    absq = np.abs(q1 - rq).max(axis=1)
    finite = bool(np.isfinite(q1).all() and np.isfinite(v1).all())
    bitwise = bool((q1 == q2).all() and (v1 == v2).all())
    print(f"forward ({n} states, contacts per state median {np.median(ncon):.0f} max {ncon.max()}):")
    print(f"  qvel' relative error  {fmt(stats(rel))}")
    print(f"  qpos' max abs error   {fmt(stats(absq))}")
    print(f"  finite {finite}, bitwise deterministic {bitwise}")
    res["forward"] = dict(qvel_rel=stats(rel), qpos_abs=stats(absq), finite=finite, bitwise=bitwise,
                          ncon_median=float(np.median(ncon)), ncon_max=int(ncon.max()))
    return finite and bitwise and np.median(rel) <= GATES["forward_median"]


def section_gradient(res, m, S, ref, iterations, ng, seed):
    rng = np.random.default_rng(seed + 1)
    jq = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    idx = rng.choice(len(S["qpos"]), ng, replace=False)
    cv, cq = rng.normal(size=(ng, m.nv)), rng.normal(size=(ng, m.nu))
    f32 = lambda x: jnp.asarray(x, jnp.float32)  # noqa: E731
    fric, com = jnp.ones(ng, jnp.float32), jnp.zeros((ng, 3), jnp.float32)
    ok, out = True, {}
    for kind in ("free", "contact"):
        qpos = S["qpos"][idx]
        if kind == "free":
            qpos = fixtures.free_flight_states(m, qpos, lift=3.0, joint_margin=0.15).astype(np.float64)
        qvel, warm, ctrl = S["qvel"][idx], S["warm"][idx], S["ctrl"][idx]

        def loss_engine(v, u):
            q1, v1, _ = jax_ffi.step(f32(qpos), v, f32(warm), u, fric, com, iterations=iterations, clamp_kv=1,
                                     normal=0)
            return jnp.sum(f32(cv) * v1) + jnp.sum(f32(cq) * q1[:, jq])

        gv_e, gu_e = jax.grad(loss_engine, argnums=(0, 1))(f32(qvel), f32(ctrl))
        gu_e, gv_e = np.asarray(gu_e, np.float64), np.asarray(gv_e, np.float64)

        def loss_ref(j, v, u):
            q1, v1, _ = ref(qpos[j], v, warm[j], u)
            return cv[j] @ v1 + cq[j] @ q1[jq]

        eps = 1e-6
        gu_r, gv_r = np.zeros_like(gu_e), np.zeros_like(gv_e)
        for j in range(ng):
            for a in range(m.nu):
                e = np.zeros(m.nu)
                e[a] = eps
                gu_r[j, a] = (loss_ref(j, qvel[j], ctrl[j] + e) - loss_ref(j, qvel[j], ctrl[j] - e)) / (2 * eps)
            for a in range(m.nv):
                e = np.zeros(m.nv)
                e[a] = eps
                gv_r[j, a] = (loss_ref(j, qvel[j] + e, ctrl[j]) - loss_ref(j, qvel[j] - e, ctrl[j])) / (2 * eps)
        out[kind] = {}
        for name, ge, gr in (("ctrl", gu_e, gu_r), ("qvel", gv_e, gv_r)):
            rel = np.linalg.norm(ge - gr, axis=1) / np.maximum(np.linalg.norm(gr, axis=1), 1e-9)
            s = stats(rel)
            out[kind][name] = s
            finite = bool(np.isfinite(ge).all())
            if kind == "free":
                passed = finite and s["median"] <= GATES["free_median"] and s["p90"] <= GATES["free_p90"]
            else:
                passed = finite and s["median"] <= GATES["contact_median"]
            ok &= passed
            print(f"gradient {kind:7s} wrt {name}: {fmt(s)}  {'ok' if passed else 'FAIL'}")
    res["gradient"] = out
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--states", type=int, default=512, help="corpus states (the forward section uses all)")
    ap.add_argument("--grad-states", type=int, default=64, help="states of each gradient kind")
    ap.add_argument("--iterations", type=int, default=50, help="Newton iterations per substep (both simulators)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="result JSON")
    a = ap.parse_args()
    sections = [s for s in a.sections.split(",") if s]
    t0 = time.time()

    m, mt = fixtures.load_model(fixtures.H1, check="engine")
    substeps = mt["substeps_per_control_step"]
    res = dict(model=str(fixtures.H1 / "h1_train.mjb"), nv=m.nv, iterations=a.iterations, seed=a.seed, gates=GATES)
    passed = {}
    if "capability" in sections:
        passed["capability"] = section_capability(res, m, mt)
    if {"forward", "gradient"} & set(sections):
        S = corpus(m, a.states, a.seed, substeps)
        m.opt.iterations, m.opt.tolerance = a.iterations, 1e-12
        ref = Reference(m, substeps)
        jax_ffi.init(model=m, foot_pairs=mt["foot_pairs"])
        n = a.states
        fric, com = jnp.ones(n, jnp.float32), jnp.zeros((n, 3), jnp.float32)

        def step(S):
            q, v, _ = jax_ffi.step(*(jnp.asarray(S[k], jnp.float32) for k in ("qpos", "qvel", "warm", "ctrl")),
                                   fric, com, iterations=a.iterations)
            return np.asarray(q, np.float64), np.asarray(v, np.float64)

        if "forward" in sections:
            passed["forward"] = section_forward(res, m, S, ref, step, n)
        if "gradient" in sections:
            passed["gradient"] = section_gradient(res, m, S, ref, a.iterations, a.grad_states, a.seed)
    passed = {k: bool(v) for k, v in passed.items()}
    res["passed"] = passed
    ok = all(passed.values())
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"h1 check: {'PASS' if ok else 'FAIL'} ({', '.join(f'{k} {v}' for k, v in passed.items())}; "
          f"{time.time() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
