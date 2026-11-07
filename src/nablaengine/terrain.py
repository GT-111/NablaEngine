"""Terrain for the engine: static boxes and heightfields shared by every world.

A terrain is the static geometry of a scene's world body: MuJoCo `box` and `hfield` geoms, as mjlab's terrain
generator (`mjlab.terrains.TerrainGenerator`) or legged_gym build them (one large terrain of sub-terrains;
every world starts on its own sub-terrain). The robot model stays the engine's; the terrain comes as data:

  terrain = Terrain.from_model(m_scene)     # the box / hfield geoms of an MjModel's world body
  cfg = TerrainConfig(geoms="feet")         # which robot geoms collide with it (compile time)

On the device the terrain is global memory, uploaded once per device: no recompilation when it changes, and terrains
of any size (thousands of boxes, millions of heightfield samples).

Contacts (TerrainConfig.mode = "analytic"): a robot sphere / capsule against a box is MuJoCo's sphere_box /
capsule_box (closed form, as MuJoCo Warp); against a heightfield, fixed sample points (sphere: the centre; capsule:
cap_points points along the axis) are measured against the plane of the heightfield triangle below them (an
approximation of MuJoCo's prism contacts): dist = n_z (q_z - h) - r. TerrainConfig.mode = "native": MuJoCo's prisms
(two per cell), each measured exactly, at most 4 contacts per geom and heightfield. Every robot geom g that collides with
the terrain has a virtual contact pair (index NPAIR + t) with the parameters of g's pair with the floor plane (the
foot pairs keep the per-world friction).

Broad phase: a uniform 2-D grid over the terrain (cell size `cell`); each cell lists the elements (boxes, heightfields)
whose xy bounding box overlaps it; a robot geom visits the cells its bounding sphere overlaps, and an element is
handled in the first of those cells it also covers (no duplicates).
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

BOX_WORDS = 15  # pos 3, rot 9 (row major), half sizes 3
HF_WORDS = 16  # pos 3, rot 9, size 4 (MuJoCo hfield_size: x / y radius, z top scale, z bottom)
HF_INTS = 4  # nrow, ncol, data address, 0
MAX_ELEMENTS = 0xFF00  # terrain elements (boxes + heightfields): the tape's 16-bit element index


@dataclass
class TerrainConfig:
    """Compile-time terrain options. geoms: "feet" (the robot geoms of the foot-floor pairs) or "body" (every robot
    geom with a floor pair); floor: keep the model's floor plane pairs too (a terrain usually has its own ground);
    cap_points: heightfield samples per capsule; mode: heightfield contacts, "analytic"
    (sample points against the triangle below) or "native" (MuJoCo's prisms, each measured
    exactly, at most 4 contacts per geom and heightfield: step.py, TERRAIN_HF_NATIVE); maxcon: contact capacity per
    world (None: the model's, step.maxcon)."""
    geoms: str = "feet"
    floor: bool = False
    cap_points: int = 3
    mode: str = "analytic"
    maxcon: int = None

    def __post_init__(self):
        if self.geoms not in ("feet", "body"):
            raise ValueError(f"TerrainConfig.geoms {self.geoms!r}: 'feet' or 'body'")
        if self.mode not in ("analytic", "native"):
            raise ValueError(f"TerrainConfig.mode {self.mode!r}: 'analytic' (sample points) or 'native' (prisms)")

    def key(self):
        k = dict(geoms=self.geoms, floor=self.floor, cap_points=self.cap_points, mode=self.mode,
                 maxcon=self.maxcon)
        return k


@dataclass
class Terrain:
    """Static terrain data in world coordinates (float64 here, float32 on the device)."""
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, BOX_WORDS)))  # (nbox, BOX_WORDS)
    hf_head: np.ndarray = field(default_factory=lambda: np.zeros((0, HF_WORDS)))  # (nhf, HF_WORDS)
    hf_int: np.ndarray = field(default_factory=lambda: np.zeros((0, HF_INTS), np.int64))  # (nhf, HF_INTS)
    hf_data: np.ndarray = field(default_factory=lambda: np.zeros(0))  # heights in [0, 1] (MuJoCo's normalised data)

    @staticmethod
    def from_model(m, d=None):
        """The static box and hfield geoms of MjModel m: those of the world body and of bodies welded to it (mjlab's
        "terrain" body; poses from mj_kinematics on d, or a fresh MjData)."""
        if d is None:
            d = mujoco.MjData(m)
            mujoco.mj_kinematics(m, d)
        boxes, head, ints, data = [], [], [], []
        adr = 0
        for g in range(m.ngeom):
            if m.body_weldid[m.geom_bodyid[g]] != 0:
                continue
            t = int(m.geom_type[g])
            pos, rot = d.geom_xpos[g].copy(), d.geom_xmat[g].copy()
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                boxes.append(np.concatenate([pos, rot, m.geom_size[g]]))
            elif t == mujoco.mjtGeom.mjGEOM_HFIELD:
                h = int(m.geom_dataid[g])
                nrow, ncol = int(m.hfield_nrow[h]), int(m.hfield_ncol[h])
                a = int(m.hfield_adr[h])
                head.append(np.concatenate([pos, rot, m.hfield_size[h]]))
                ints.append([nrow, ncol, adr, 0])
                data.append(m.hfield_data[a:a + nrow * ncol])
                adr += nrow * ncol
        t = Terrain()
        if boxes:
            t.boxes = np.asarray(boxes, np.float64)
        if head:
            t.hf_head, t.hf_int = np.asarray(head, np.float64), np.asarray(ints, np.int64)
            t.hf_data = np.concatenate(data).astype(np.float64)
        return t

    def save(self, path):
        """Write the terrain to an .npz file (load() reads it back)."""
        np.savez(path, boxes=self.boxes, hf_head=self.hf_head, hf_int=self.hf_int, hf_data=self.hf_data)

    @staticmethod
    def load(path):
        z = np.load(path)
        return Terrain(boxes=z["boxes"], hf_head=z["hf_head"], hf_int=z["hf_int"], hf_data=z["hf_data"])

    def add_to_spec(self, spec, prefix="ter"):
        """Add the terrain to an MjSpec's world body as box and hfield geoms (the reverse of from_model, e.g. for a
        MuJoCo / MuJoCo Warp reference scene); returns the geom names."""
        import mujoco
        names = []
        wb = spec.worldbody
        for i, b in enumerate(self.boxes):
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, np.asarray(b[3:12], np.float64))
            n = f"{prefix}_box{i}"
            wb.add_geom(name=n, type=mujoco.mjtGeom.mjGEOM_BOX, size=b[12:15].tolist(), pos=b[:3].tolist(), quat=q.tolist())
            names.append(n)
        for j, (h, hi) in enumerate(zip(self.hf_head, self.hf_int)):
            nrow, ncol, adr = int(hi[0]), int(hi[1]), int(hi[2])
            n = f"{prefix}_hf{j}"
            spec.add_hfield(name=n, size=h[12:16].tolist(), nrow=nrow, ncol=ncol,
                            userdata=self.hf_data[adr:adr + nrow * ncol].tolist())
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, np.asarray(h[3:12], np.float64))
            wb.add_geom(name=n, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=n, pos=h[:3].tolist(), quat=q.tolist())
            names.append(n)
        return names

    def element_aabbs(self):
        """(n, 4) xmin, ymin, xmax, ymax of the elements: the boxes, then the heightfields."""
        out = []
        for b in self.boxes:
            R, hs = b[3:12].reshape(3, 3), b[12:15]
            ext = np.abs(R) @ hs
            out.append([b[0] - ext[0], b[1] - ext[1], b[0] + ext[0], b[1] + ext[1]])
        for h in self.hf_head:
            R, sz = h[3:12].reshape(3, 3), h[12:16]
            ext = np.abs(R[:2, :2]) @ sz[:2]
            out.append([h[0] - ext[0], h[1] - ext[1], h[0] + ext[0], h[1] + ext[1]])
        return np.asarray(out, np.float64).reshape(-1, 4)

    def pack(self, cell=1.0):
        """Device arrays: boxes float32 (nbox, 15), heightfield headers float32 (nhf, 16) and ints int32 (nhf, 4), data
        float32, and the broad-phase grid: grid float32 [x0, y0, 1 / cell, nx, ny, nbox], cell_start int32 (nx ny + 1),
        cell_items int32 (element index: box i, or nbox + heightfield j), elem_cell int32 (n, 2): the element's first
        cell (cx, cy)."""
        aabb = self.element_aabbs()
        nbox = len(self.boxes)
        if len(aabb) >= MAX_ELEMENTS:
            raise ValueError(f"{len(aabb)} terrain elements: at most {MAX_ELEMENTS - 1} (the tape's 16-bit element "
                             "index)")
        if len(aabb):
            x0, y0 = aabb[:, 0].min() - 1e-3, aabb[:, 1].min() - 1e-3
            nx = max(1, int(np.ceil((aabb[:, 2].max() - x0) / cell)) + 1)
            ny = max(1, int(np.ceil((aabb[:, 3].max() - y0) / cell)) + 1)
        else:
            x0 = y0 = 0.0
            nx = ny = 1
        lists = [[] for _ in range(nx * ny)]
        first = np.zeros((len(aabb), 2), np.int32)
        for e, (a0, b0, a1, b1) in enumerate(aabb):
            c0, c1 = int((a0 - x0) / cell), min(nx - 1, int((a1 - x0) / cell))
            r0, r1 = int((b0 - y0) / cell), min(ny - 1, int((b1 - y0) / cell))
            first[e] = (c0, r0)
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    lists[r * nx + c].append(e)
        start = np.zeros(nx * ny + 1, np.int32)
        start[1:] = np.cumsum([len(x) for x in lists])
        items = np.asarray([e for x in lists for e in x] or [0], np.int32)
        return dict(boxes=np.asarray(self.boxes, np.float32).reshape(-1, BOX_WORDS),
                    hf_head=np.asarray(self.hf_head, np.float32).reshape(-1, HF_WORDS),
                    hf_int=np.asarray(self.hf_int, np.int32).reshape(-1, HF_INTS),
                    hf_data=np.asarray(self.hf_data, np.float32).reshape(-1),
                    grid=np.asarray([x0, y0, 1.0 / cell, nx, ny, nbox], np.float32),
                    cell_start=start, cell_items=items, elem_cell=first.reshape(-1, 2))


def robot_terrain_geoms(m, foot_pairs, which):
    """[(robot geom, its floor pair index, foot flag)] of the geoms that collide with the terrain: which = "feet" (the
    foot-floor pairs' robot geoms) or "body" (every robot geom with a floor pair), in floor-pair order."""
    foot = set(int(p) for p in foot_pairs)
    out = []
    for p in range(m.npair):
        g1, g2 = int(m.pair_geom1[p]), int(m.pair_geom2[p])
        t1, t2 = int(m.geom_type[g1]), int(m.geom_type[g2])
        if mujoco.mjtGeom.mjGEOM_PLANE not in (t1, t2):
            continue
        g = g2 if t1 == mujoco.mjtGeom.mjGEOM_PLANE else g1
        if which == "feet" and p not in foot:
            continue
        if which not in ("feet", "body"):
            raise ValueError(f"terrain geoms {which!r}: 'feet' or 'body'")
        if int(m.geom_type[g]) not in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE):
            raise ValueError(f"terrain contacts of robot geom {m.geom(g).name or g} "
                             f"({mujoco.mjtGeom(int(m.geom_type[g])).name}): only spheres and capsules collide with "
                             "the terrain (TerrainConfig.geoms='feet' when only the feet are spheres / capsules)")
        out.append((g, p, int(p in foot)))
    return out
