"""The forward tape of the G1 engine: one record per world and substep, the residuals of the analytic adjoint.

The forward kernel writes, when given a tape pointer (null: off), tape[w][k][REC_WORDS] for world w and substep k: the
substep's input state, the final Newton acceleration a* and the implicitfast acceleration a_adv, the discrete decisions
(contact table as (pair, slot), quadratic-zone bit of every constraint row, actuator clamp mask) and the two Cholesky
factors the adjoint needs, L_H (Newton Hessian at the final zones) and L_Q = chol(M + h diag(kv)). Words are float32;
integer fields are stored bitwise (decode() reinterprets them). Records and their large blocks start on 128-byte
boundaries.

layout(m) gives {name: (offset, length)} in words, rec_words(m) the record length; header(m) emits the same table as
#define OFF_* / REC_WORDS for the CUDA code (step.py writes the record), decode(arr, m) splits records into arrays.
"""

import numpy as np

from nablaengine.g1_engine import step

ALIGN = 32  # words (128 bytes): record and block alignment
FLAG_UNSUPPORTED = 31  # flags bits 0-4: s.unsupported, accumulated over the substeps so far (as out_info)
FLAG_REFACT = 1 << 8  # the zones changed after the last Newton factorization: H re-assembled and re-factored for the tape
FLAG_NONCONV = 1 << 9  # the Newton iterations ran out before the stopping test held
FLAG_SINGULAR = 1 << 10  # min_nu < SINGULAR_NU: a self contact whose geom axes (or centres) nearly meet
FLAG_VERIFY_DIFF = 1 << 11  # -DDK_TAPE_VERIFY: L_H differs from a re-assembly at the final zones (expected only with REFACT)
SINGULAR_NU = 5e-3  # m
CON_NONE = 0xFFFF  # contact table entry past ncon

# (name, #define, length: callable of (nq, nv, ntri, maxcon) or an int); a None name pads to the next ALIGN boundary
_FIELDS = (
    ("hdr", "OFF_HDR", 1),  # ncon | nefc << 8 | nlim << 16 | niter << 24 (u8 each)
    ("flags", "OFF_FLAGS", 1),  # FLAG_*
    ("min_nu", "OFF_MIN_NU", 1),  # float: smallest centre / axis distance of a sphere or capsule self contact, else +inf
    ("reserved", "OFF_RESERVED", 1),
    ("clamp", "OFF_CLAMP", lambda nq, nv, ntri, nc: -(-nv // 32)),  # dof bits: actuator force at its limit
    ("zone", "OFF_ZONE", lambda nq, nv, ntri, nc: -(-step.NEFC // 32)),  # row bits: quadratic zone (row in the Hessian)
    ("con", "OFF_CON", lambda nq, nv, ntri, nc: -(-nc // 2)),  # u16 per contact (row order): pair | slot << 9
    ("qpos", "OFF_QPOS", lambda nq, nv, ntri, nc: nq),  # substep input
    ("qvel", "OFF_QVEL", lambda nq, nv, ntri, nc: nv),
    ("qacc", "OFF_QACC", lambda nq, nv, ntri, nc: nv),  # final Newton acceleration a* (the warm start of the next one)
    ("qacc_adv", "OFF_QADV", lambda nq, nv, ntri, nc: nv),  # (M + h kv)^-1 M a*: qvel' = qvel + h qacc_adv
    (None, None, ALIGN),
    ("L_H", "OFF_LH", lambda nq, nv, ntri, nc: ntri),  # packed lower triangle TRI(i, j)
    (None, None, ALIGN),
    ("L_Q", "OFF_LQ", lambda nq, nv, ntri, nc: ntri),
    (None, None, ALIGN),
)


# terrain builds (nablaengine.terrain): the contact table's entries are pair | slot << 10 (the terrain's virtual pairs
# follow the model's), and a second table holds each contact's terrain element (u16; 0 for the model's pairs)
_TERRAIN_FIELDS = (("telem", "OFF_TELEM", lambda nq, nv, ntri, nc: -(-nc // 2)),)
# native heightfield builds: also each contact's candidate id (u32; step.py HfCand.pid, 0 for other contacts)
_NATIVE_FIELDS = (("taux", "OFF_TAUX", lambda nq, nv, ntri, nc: nc),)


def kind(terrain):
    """The record kind of a build: None (no terrain), "analytic" or "native" (terrain: (TerrainConfig, geoms), a
    TerrainConfig, or None)."""
    if terrain is None:
        return None
    cfg = terrain[0] if isinstance(terrain, tuple) else terrain
    return cfg.mode


def _table(m, max_contacts=None, terrain=False):
    nc = max_contacts or step.maxcon(m)
    nv, ntri = m.nv, m.nv * (m.nv + 1) // 2
    npmax = 1024 if terrain else 512
    assert nv <= 64 and m.npair < npmax and nc <= 255 and step.NEFC <= 255, "field widths (clamp u64, pair bits, u8)"
    fields = _FIELDS
    if terrain:
        i = [f[0] for f in _FIELDS].index("con") + 1
        fields = _FIELDS[:i] + _TERRAIN_FIELDS + (_NATIVE_FIELDS if terrain == "native" else ()) + _FIELDS[i:]
    out, off = [], 0
    for name, define, n in fields:
        if name is None:
            off = -(-off // n) * n
            continue
        n = n(m.nq, nv, ntri, nc) if callable(n) else n
        out.append((name, define, off, n))
        off += n
    return out, off


def layout(m, max_contacts=None, terrain=False):
    """{name: (offset, length)} of a record in 32-bit words (mjlab model: L_H at 192, L_Q at 832, 1472 words).
    terrain: a terrain build's record kind (kind(): "analytic" or "native"; _TERRAIN_FIELDS, _NATIVE_FIELDS)."""
    return {name: (off, n) for name, _, off, n in _table(m, max_contacts, terrain)[0]}


def rec_words(m, max_contacts=None, terrain=False):
    """Record length in words (a multiple of ALIGN)."""
    return _table(m, max_contacts, terrain)[1]


def header(m, max_contacts=None, terrain=False):
    """#define OFF_* / REC_WORDS / TAPE_* for the engine's CUDA code (step.py)."""
    tab, total = _table(m, max_contacts, terrain)
    lines = [f"#define REC_WORDS {total}  // tape record (tape.py): words per world and substep"]
    lines += [f"#define {define} {off}  // {name}, {n} words" for name, define, off, n in tab]
    lines += [f"#define TAPE_REFACT {FLAG_REFACT}u", f"#define TAPE_NONCONV {FLAG_NONCONV}u",
              f"#define TAPE_SINGULAR {FLAG_SINGULAR}u", f"#define TAPE_VERIFY_DIFF {FLAG_VERIFY_DIFF}u",
              f"#define TAPE_SINGULAR_NU {SINGULAR_NU}f", f"#define TAPE_CON_NONE {CON_NONE}u"]
    return "\n" + "\n".join(lines) + "\n"


def tri_unpack(packed, n):
    """(..., n (n + 1) / 2) packed lower triangles TRI(i, j) -> (..., n, n) lower-triangular matrices."""
    packed = np.asarray(packed)
    i, j = np.tril_indices(n)
    out = np.zeros(packed.shape[:-1] + (n, n), packed.dtype)
    out[..., i, j] = packed[..., i * (i + 1) // 2 + j]
    return out


def decode(arr, m, max_contacts=None, terrain=False):
    """Records (..., REC_WORDS) float32 -> dict of arrays (leading shape kept): ncon, nefc, nlim, niter, flags (and
    unsupported / refact / nonconv / singular / verify_diff), min_nu, clamp (u64 dof mask) and clamp_dof (bool, nv),
    zone (bool, NEFC), pair / slot (int, maxcon; pair -1 past ncon), qpos, qvel, qacc, qacc_adv, L_H, L_Q (packed);
    terrain: a terrain build's record, with elem (the contact's terrain element)."""
    a = np.ascontiguousarray(arr, np.float32)
    nw = rec_words(m, max_contacts, terrain)
    assert a.shape[-1] == nw, (a.shape, nw)
    u = a.view(np.uint32)
    lay = layout(m, max_contacts, terrain)
    sl = lambda name: slice(lay[name][0], lay[name][0] + lay[name][1])  # noqa: E731
    hdr, flags = u[..., lay["hdr"][0]].astype(np.int64), u[..., lay["flags"][0]].astype(np.int64)
    out = dict(ncon=hdr & 255, nefc=(hdr >> 8) & 255, nlim=(hdr >> 16) & 255, niter=hdr >> 24, flags=flags,
               unsupported=flags & FLAG_UNSUPPORTED, refact=(flags & FLAG_REFACT) != 0,
               nonconv=(flags & FLAG_NONCONV) != 0, singular=(flags & FLAG_SINGULAR) != 0,
               verify_diff=(flags & FLAG_VERIFY_DIFF) != 0, min_nu=a[..., lay["min_nu"][0]])
    cw = u[..., sl("clamp")].astype(np.uint64)
    out["clamp"] = sum(cw[..., k] << np.uint64(32 * k) for k in range(cw.shape[-1]))
    out["clamp_dof"] = ((cw[..., :, None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)).astype(bool).reshape(
        cw.shape[:-1] + (-1,))[..., :m.nv]
    zw = u[..., sl("zone")].astype(np.uint64)
    out["zone"] = ((zw[..., :, None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)).astype(bool).reshape(
        zw.shape[:-1] + (-1,))[..., :step.NEFC]
    nc = max_contacts or step.maxcon(m)
    e = np.ascontiguousarray(u[..., sl("con")]).view(np.uint16)[..., :nc].astype(np.int64)  # little endian: low half first
    past = np.arange(nc) >= out["ncon"][..., None]
    pb = 10 if terrain else 9
    out["pair"], out["slot"] = np.where(past, -1, e & ((1 << pb) - 1)), np.where(past, -1, e >> pb)
    if terrain:
        x = np.ascontiguousarray(u[..., sl("telem")]).view(np.uint16)[..., :nc].astype(np.int64)
        out["elem"] = np.where(past, -1, x)
    for name in ("qpos", "qvel", "qacc", "qacc_adv", "L_H", "L_Q"):
        out[name] = a[..., sl(name)]
    return out
