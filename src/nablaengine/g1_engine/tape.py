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
FLAG_UNSUPPORTED = 63  # flags bits 0-5: unsupported geometry, capacity or native narrowphase failure (as out_info)
FLAG_REFACT = 1 << 8  # the zones changed after the last Newton factorization: H re-assembled and re-factored for the tape
FLAG_NONCONV = 1 << 9  # the Newton iterations ran out before the stopping test held
FLAG_SINGULAR = 1 << 10  # min_nu < SINGULAR_NU: a self contact whose geom axes (or centres) nearly meet
FLAG_VERIFY_DIFF = 1 << 11  # -DDK_TAPE_VERIFY: L_H differs from a re-assembly at the final zones (expected only with REFACT)
SINGULAR_NU = 5e-3  # m
CON_NONE = 0xFFFF  # contact table entry past ncon


def pair_bits(m):
    return max(9, (len(step.contact_pairs(m)) - 1).bit_length())

# (name, #define, length: callable of (nq, nv, ntri, maxcon) or an int); a None name pads to the next ALIGN boundary
_FIELDS = (
    ("hdr", "OFF_HDR", 1),  # ncon | nefc << 16 (u16 each)
    ("flags", "OFF_FLAGS", 1),  # FLAG_*
    ("min_nu", "OFF_MIN_NU", 1),  # float: smallest centre / axis distance of a sphere or capsule self contact, else +inf
    ("counts", "OFF_COUNTS", 1),  # nlim | niter << 16 (u16 each)
    ("clamp", "OFF_CLAMP", lambda nq, nv, ntri, nc: -(-nv // 32)),  # dof bits: actuator force at its limit
    ("zone", "OFF_ZONE", None),  # model-derived row bits: quadratic zone (row in the Hessian)
    ("con", "OFF_CON", lambda nq, nv, ntri, nc: -(-nc // 2)),  # u16 per contact (row order): pair | slot << 9
    ("feature", "OFF_FEATURE", lambda nq, nv, ntri, nc: 12 * nc),  # raw native CMFeature words per contact
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


def _table(m, max_contacts=None):
    nc = max_contacts or step.maxcon(m)
    nv, ntri = m.nv, m.nv * (m.nv + 1) // 2
    assert nv <= 64 and len(step.contact_pairs(m)) <= 2048 and nc <= 65535 and step.nefc(m, nc) <= 65535, "tape field widths"
    out, off = [], 0
    for name, define, n in _FIELDS:
        if name == "feature" and not step.native_collision(m):
            continue
        if name is None:
            off = -(-off // n) * n
            continue
        n = -(-step.nefc(m, nc) // 32) if name == "zone" else n(m.nq, nv, ntri, nc) if callable(n) else n
        out.append((name, define, off, n))
        off += n
    return out, off


def layout(m, max_contacts=None):
    """{name: (offset, length)} of a record in 32-bit words (mjlab model: L_H at 192, L_Q at 832, 1472 words)."""
    return {name: (off, n) for name, _, off, n in _table(m, max_contacts)[0]}


def rec_words(m, max_contacts=None):
    """Record length in words (a multiple of ALIGN)."""
    return _table(m, max_contacts)[1]


def header(m, max_contacts=None):
    """#define OFF_* / REC_WORDS / TAPE_* for the engine's CUDA code (step.py)."""
    tab, total = _table(m, max_contacts)
    lines = [f"#define REC_WORDS {total}  // tape record (tape.py): words per world and substep"]
    lines += [f"#define {define} {off}  // {name}, {n} words" for name, define, off, n in tab]
    lines += [f"#define TAPE_REFACT {FLAG_REFACT}u", f"#define TAPE_NONCONV {FLAG_NONCONV}u",
              f"#define TAPE_SINGULAR {FLAG_SINGULAR}u", f"#define TAPE_VERIFY_DIFF {FLAG_VERIFY_DIFF}u",
              f"#define TAPE_SINGULAR_NU {SINGULAR_NU}f", f"#define TAPE_CON_NONE {CON_NONE}u",
              f"#define TAPE_PAIR_BITS {pair_bits(m)}", f"#define TAPE_PAIR_MASK {(1 << pair_bits(m)) - 1}u"]
    return "\n" + "\n".join(lines) + "\n"


def tri_unpack(packed, n):
    """(..., n (n + 1) / 2) packed lower triangles TRI(i, j) -> (..., n, n) lower-triangular matrices."""
    packed = np.asarray(packed)
    i, j = np.tril_indices(n)
    out = np.zeros(packed.shape[:-1] + (n, n), packed.dtype)
    out[..., i, j] = packed[..., i * (i + 1) // 2 + j]
    return out


def decode(arr, m, max_contacts=None):
    """Records (..., REC_WORDS) float32 -> dict of arrays (leading shape kept): ncon, nefc, nlim, niter, flags (and
    unsupported / refact / nonconv / singular / verify_diff), min_nu, clamp (u64 dof mask) and clamp_dof (bool, nv),
    zone (bool, NEFC), pair / slot (int, maxcon; pair -1 past ncon), qpos, qvel, qacc, qacc_adv, L_H, L_Q (packed)."""
    a = np.ascontiguousarray(arr, np.float32)
    nw = rec_words(m, max_contacts)
    assert a.shape[-1] == nw, (a.shape, nw)
    u = a.view(np.uint32)
    lay = layout(m, max_contacts)
    sl = lambda name: slice(lay[name][0], lay[name][0] + lay[name][1])  # noqa: E731
    hdr, flags = u[..., lay["hdr"][0]].astype(np.int64), u[..., lay["flags"][0]].astype(np.int64)
    counts = u[..., lay["counts"][0]].astype(np.int64)
    out = dict(ncon=hdr & 65535, nefc=hdr >> 16, nlim=counts & 65535, niter=counts >> 16, flags=flags,
               unsupported=flags & FLAG_UNSUPPORTED, refact=(flags & FLAG_REFACT) != 0,
               nonconv=(flags & FLAG_NONCONV) != 0, singular=(flags & FLAG_SINGULAR) != 0,
               verify_diff=(flags & FLAG_VERIFY_DIFF) != 0, min_nu=a[..., lay["min_nu"][0]])
    cw = u[..., sl("clamp")].astype(np.uint64)
    out["clamp"] = sum(cw[..., k] << np.uint64(32 * k) for k in range(cw.shape[-1]))
    out["clamp_dof"] = ((cw[..., :, None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)).astype(bool).reshape(
        cw.shape[:-1] + (-1,))[..., :m.nv]
    zw = u[..., sl("zone")].astype(np.uint64)
    out["zone"] = ((zw[..., :, None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)).astype(bool).reshape(
        zw.shape[:-1] + (-1,))[..., :step.nefc(m, max_contacts)]
    nc = max_contacts or step.maxcon(m)
    e = np.ascontiguousarray(u[..., sl("con")]).view(np.uint16)[..., :nc].astype(np.int64)  # little endian: low half first
    past = np.arange(nc) >= out["ncon"][..., None]
    out["pair"], out["slot"] = np.where(past, -1, e & ((1 << pair_bits(m)) - 1)), np.where(past, -1, e >> pair_bits(m))
    if "feature" in lay:
        out["feature"] = u[..., sl("feature")].reshape(a.shape[:-1] + (nc, 12))
    for name in ("qpos", "qvel", "qacc", "qacc_adv", "L_H", "L_Q"):
        out[name] = a[..., sl(name)]
    return out
