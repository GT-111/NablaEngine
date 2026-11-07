"""The G1 model and the reference corpora of the checks.

Fixtures directory (the model, and the references of the checks): $DK_FIXTURES when set (and not empty), else DEFAULT,
<checkout>/fixtures/mjlab, for this module, nablaengine.jax_ffi and every script. It holds the mjlab G1
(g1_train.mjb, g1_meta.json: foot-floor pairs, torso body, sizes and options) and, per benchmark corpus, the MuJoCo
Warp reference
mjw317_<corpus>.npz (scripts/make_mjw_reference.py: native mujoco-warp 3.17.0, one control step at 50 iterations, and
the contacts at the input states).

Corpora. A name is looked up in this order:
  1. <fixtures directory>/<name>_fwd.npz (+ <name>_adj.npz): a corpus in the fixture format (inputs, MuJoCo Warp's
     outputs, and with the adjoint file the cotangents cot{c}_qpos / cot{c}_qvel and reference gradients
     grad{c}_qpos / qvel / ctrl);
  2. a benchmark corpus, <bench data>/<name>.npz with <bench data> = $DK_BENCH_DATA (default <checkout>/benchmarks/data):
     the corpora mf_s0 .. mf_s3 (benchmarks/bench/corpus/gen_mf.py: MuJoCo C rollouts of the G1 under random joint
     setpoints and pushes; the inputs qpos, qvel, qacc_warmstart, ctrl, friction, com of 4096 states,
     their stored cotangents cot_qpos (S, 3, nq) / cot_qvel (S, 3, nv), labels label_*). load_corpus maps them to the
     fixture keys (cot{c}_* = cot_*[:, c], prefall / rollout_step from the labels) and merges the corpus's MuJoCo Warp
     reference from the fixtures directory when it exists (out_qpos, out_qvel, ncon, nefc, solver_niter per substep,
     iterations 50). Benchmark corpora have no reference gradients: the gradient checks score against the MuJoCo C
     float64 finite-difference truth, <bench data>/truth/<name>.npz (load_truth).
The public defaults are MAIN_CORPORA = (mf_s0, mf_s1)."""

import json
import os
from pathlib import Path

import mujoco
import numpy as np

CHECKOUT = Path(__file__).resolve().parents[2]
ROOT = CHECKOUT / "fixtures"
DEFAULT = ROOT / "mjlab"  # the default fixtures directory: the mjlab G1 and the references of the public corpora
BENCH_DATA = CHECKOUT / "benchmarks" / "data"  # $DK_BENCH_DATA: the benchmark corpora and their truth files
BENCH_PATTERN = "mf_s*.npz"  # the benchmark corpora of the mjlab G1 that corpora() lists
MJW_REFERENCE = "mjw317_{name}.npz"  # a corpus's MuJoCo Warp reference in the fixtures directory
MAIN_CORPORA = ("mf_s0", "mf_s1")  # the public defaults, in order of preference
NCOT = 3


def fixtures_dir():
    """The fixtures directory: $DK_FIXTURES (relative to the current directory) when set and not empty, else DEFAULT."""
    return Path(os.environ.get("DK_FIXTURES") or DEFAULT)


def bench_dir():
    """The benchmark corpora's directory: $DK_BENCH_DATA when set and not empty, else BENCH_DATA."""
    return Path(os.environ.get("DK_BENCH_DATA") or BENCH_DATA)


def bench_path(name):
    return bench_dir() / f"{name}.npz"


def reference_path(name):
    """The MuJoCo Warp reference of a benchmark corpus (it may not exist)."""
    return fixtures_dir() / MJW_REFERENCE.format(name=name)


def truth_path(name):
    return bench_dir() / "truth" / f"{name}.npz"


def is_fixture(name):
    """True for a corpus in the fixture format (<fixtures directory>/<name>_fwd.npz)."""
    return (fixtures_dir() / f"{name}_fwd.npz").exists()


def corpora():
    """Names of the corpora: those of the fixtures directory (<name>_fwd.npz), then the benchmark corpora of the
    mjlab G1 (<bench data>/mf_s*.npz)."""
    names = sorted(p.name[:-len("_fwd.npz")] for p in fixtures_dir().glob("*_fwd.npz"))
    return names + sorted(p.stem for p in bench_dir().glob(BENCH_PATTERN) if p.stem not in names)


def meta(d=None):
    path = Path(d or fixtures_dir()) / "g1_meta.json"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"no fixtures model at {path.parent} ({path.name} missing). The default, fixtures/mjlab next to src/, "
            "exists only in a source checkout (PYTHONPATH=src, a .pth file or pip install -e): set "
            "DK_FIXTURES=<checkout>/fixtures/mjlab, or give the JAX interface the model itself, "
            "nablaengine.jax_ffi.init(model=<MjModel>, foot_pairs=<its foot-floor pair indices>)") from None


def load_model(d=None, check="forward"):
    """(MjModel, meta): the model of the fixtures directory d (default fixtures_dir()); the binary format needs the
    MuJoCo version it was saved with. check: the capability scope it must pass (nablaengine.model.capability.require
    with the meta's foot pairs and torso body; default "forward", the adjoint asserts the rest; "engine",
    "kinematics", or None: no check), else UnsupportedModel: every script that loads the fixtures model checks it
    this way."""
    d = Path(d or fixtures_dir())
    mt = meta(d)
    if mujoco.__version__ != mt["mujoco_version"]:
        raise RuntimeError(f"g1_train.mjb was saved by MuJoCo {mt['mujoco_version']}, this is {mujoco.__version__}: "
                           f"pip install mujoco=={mt['mujoco_version']}")
    m = mujoco.MjModel.from_binary_path(str(d / "g1_train.mjb"))
    if check:
        from nablaengine.model import capability
        capability.require(m, mt["foot_pairs"], mt.get("torso_body", capability.COM_BODY), check,
                           f"fixtures model {d.name}/g1_train.mjb")
    return m, mt


def torso_id(m, mt):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, mt["torso_body"])


def default_corpus():
    """The default corpus: the first of MAIN_CORPORA present, else the first fixture corpus with adjoint references,
    else the first corpus."""
    names = corpora()
    if not names:
        raise FileNotFoundError(f"no corpus: neither <name>_fwd.npz in {fixtures_dir()} nor {BENCH_PATTERN} in "
                                f"{bench_dir()} (set DK_FIXTURES / DK_BENCH_DATA)")
    for n in MAIN_CORPORA:
        if n in names:
            return n
    with_adj = [n for n in names if (fixtures_dir() / f"{n}_adj.npz").exists()]
    return (with_adj or names)[0]


def _bench_corpus(name, adjoint):
    path = bench_path(name)
    if not path.exists():
        raise FileNotFoundError(f"corpus {name!r}: neither {fixtures_dir() / (name + '_fwd.npz')} nor {path}")
    with np.load(path, allow_pickle=False) as f:
        raw = {k: f[k] for k in f.files}
    z = {k: raw[k] for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction", "com")}
    S = len(z["qpos"])
    z["prefall"] = raw.get("label_prefall", np.zeros(S, bool))
    z["rollout_step"] = raw.get("label_rollout_step", np.zeros(S, np.int32))
    z["source"] = np.array(f"benchmark corpus {name}")
    if adjoint:
        if "cot_qpos" not in raw:
            raise KeyError(f"corpus {name}: {path.name} has no stored cotangents (cot_qpos, cot_qvel)")
        for c in range(NCOT):
            z[f"cot{c}_qpos"] = np.ascontiguousarray(raw["cot_qpos"][:, c])
            z[f"cot{c}_qvel"] = np.ascontiguousarray(raw["cot_qvel"][:, c])
    ref = reference_path(name)
    if ref.exists():
        with np.load(ref, allow_pickle=False) as r:
            if r["out_qpos"].shape[0] != S:
                raise ValueError(f"{ref.name}: {r['out_qpos'].shape[0]} states, the corpus has {S}")
            z.update(out_qpos=r["out_qpos"], out_qvel=r["out_qvel"], ncon=r["ncon"], nefc=r["nefc"],
                     solver_niter=r["niter"], iterations=np.array(json.loads(str(r["meta"]))["iterations"]))
    return z


def load_corpus(name=None, adjoint=False):
    """Dict of arrays: qpos, qvel, qacc_warmstart, ctrl, friction, com (inputs of one control step), prefall,
    rollout_step; where a MuJoCo Warp reference exists out_qpos, out_qvel (50 iterations), ncon / nefc / solver_niter,
    iterations; with adjoint=True also the cotangents cot{c}_qpos / qvel, c = 0..2, and for a fixture-format corpus
    its reference gradients grad{c}_qpos / qvel / ctrl (module docstring)."""
    d = fixtures_dir()
    name = name or default_corpus()
    if not is_fixture(name):
        return _bench_corpus(name, adjoint)
    z = dict(np.load(d / f"{name}_fwd.npz"))
    if adjoint:
        z.update(np.load(d / f"{name}_adj.npz"))
    return z


def has_contacts(name=None):
    """Whether MuJoCo Warp's contacts at the corpus states exist (load_contacts)."""
    name = name or default_corpus()
    if (fixtures_dir() / f"mjw_contacts_{name}.npz").exists():
        return True
    p = reference_path(name)
    if not p.exists():
        return False
    with np.load(p, allow_pickle=False) as r:
        return "con" in r.files


def load_contacts(name=None):
    """con (nworld, maxcon, 8): MuJoCo Warp's active contacts at the corpus states, rows (pair, dist, pos[3], normal[3])
    sorted by pair and dist, padded with pair -1; ncon (nworld,), which can exceed maxcon. From
    mjw_contacts_<name>.npz in the fixtures directory, else from the corpus's MuJoCo Warp reference."""
    name = name or default_corpus()
    p = fixtures_dir() / f"mjw_contacts_{name}.npz"
    if p.exists():
        return dict(np.load(p))
    with np.load(reference_path(name), allow_pickle=False) as r:
        return dict(con=r["con"], ncon=r["con_count"].astype(np.int64))


def load_truth(name=None):
    """The MuJoCo C float64 truth of a benchmark corpus (<bench data>/truth/<name>.npz, benchmarks/bench/truth/store.py),
    as a dict: index (N,) the sampled states, out (N, nq + nv) their forward truth (qpos' then qvel'), model_flags (S,)
    (0: MuJoCo C and native MuJoCo Warp agree on the contacts, zones and frames of the state), labels (S,), the
    gradient truth fd_vjp / fd_vjp_floor / bad / group_<G>, and meta (dict)."""
    name = name or default_corpus()
    with np.load(truth_path(name), allow_pickle=False) as t:
        z = {k: t[k] for k in t.files}
    z["meta"] = json.loads(str(z["meta"]))
    return z


def _segment_distance(p0, p1, q0, q1):
    """Closest distance between segments p0-p1 and q0-q1 (Ericson, Real-Time Collision Detection 5.1.9)."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-15 and e <= 1e-15:
        return float(np.linalg.norm(r))
    if a <= 1e-15:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= 1e-15:
            s, t = np.clip(-c / a, 0.0, 1.0), 0.0
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0.0, 1.0) if den > 1e-15 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                s, t = np.clip(-c / a, 0.0, 1.0), 0.0
            elif t > 1.0:
                s, t = np.clip((b - c) / a, 0.0, 1.0), 1.0
    return float(np.linalg.norm(p0 + d1 * s - (q0 + d2 * t)))


def self_contact_axis_distance(m, qpos):
    """Per state, the smallest distance between the axes (segments; a sphere is a point) of two robot geoms in contact
    at qpos (MuJoCo C collision), inf without self contacts. Below a few millimetres the contact normal v / |v| of
    capsule-capsule, sphere-capsule and sphere-sphere contacts has a derivative of order 1 / |v| that blows up
    (crossed feet), so checks report those states separately."""
    d = mujoco.MjData(m)
    plane = m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    out = np.full(len(qpos), np.inf)
    for i, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        for c in d.contact[:d.ncon]:
            g = (int(c.geom1), int(c.geom2))
            if plane[g[0]] or plane[g[1]]:
                continue
            seg = []
            for k in g:
                ctr, ax = d.geom_xpos[k], d.geom_xmat[k].reshape(3, 3)[:, 2]
                h = m.geom_size[k, 1] if m.geom_type[k] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
                seg += [ctr - ax * h, ctr + ax * h]
            out[i] = min(out[i], _segment_distance(*seg))
    return out


def free_flight_states(m, qpos, lift=0.0, joint_margin=0.0):
    """Corpus states moved away from the floor and the joint limits (a test without contact and limit rows): the root
    raised by `lift` metres and every limited hinge angle clipped to at least `joint_margin` rad inside its range
    (joint_margin <= 0: unchanged). float64 arithmetic, returned as float32 (the input of both engines)."""
    q = np.array(qpos, np.float64)
    q[:, 2] += lift
    if joint_margin > 0:
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.jnt_limited[j]:
                a = m.jnt_qposadr[j]
                q[:, a] = np.clip(q[:, a], m.jnt_range[j, 0] + joint_margin, m.jnt_range[j, 1] - joint_margin)
    return q.astype(np.float32)
