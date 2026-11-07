# Terrain

A terrain build adds static terrain, boxes and heightfields shared by every world, to a model of the engine's class
(the G1, or another robot from [Other robots](api.md#other-robots)). Typical uses: stairs, steps, obstacles and rough
heightfields like mjlab's sub-terrain grids. Every world starts on its own sub-terrain.

The terrain is data, not part of the compiled model. It lives in GPU global memory, is uploaded once per device, can be
replaced at any time without recompiling, and can be large (thousands of boxes, millions of heightfield samples).

## Usage

```python
import mujoco
from nablaengine import terrain as T, torch_ffi

scene = mujoco.MjModel.from_xml_path("scene_with_terrain.xml")  # box / hfield geoms on the world body
ter = T.Terrain.from_model(scene)            # or Terrain.load("terrain.npz"); Terrain.save() writes one
cfg = T.TerrainConfig(geoms="feet")          # which robot geoms collide with it (compile time)

torch_ffi.init(terrain=cfg)                  # a terrain build of the G1 (or init(model=m, foot_pairs=..., terrain=cfg))
torch_ffi.set_terrain(ter)                   # upload; call again to replace the terrain
qpos1, qvel1, warm1 = torch_ffi.step(qpos, qvel, warm, ctrl, friction, com)   # as without terrain
```

`jax_ffi` has the same entries: `init(terrain=cfg)`, `set_terrain(ter)`, `raycast`, `grid_pattern`. Both interfaces
give bitwise the same steps and gradients.

`Terrain.from_model(m)` reads the box and hfield geoms of the world body, and of bodies welded to it (mjlab's
`terrain` body), in world coordinates. `Terrain.add_to_spec(spec)` does the reverse and writes the terrain into an
`MjSpec`, e.g. to build a MuJoCo or MuJoCo Warp reference scene.

## Options (`TerrainConfig`)

| option | default | meaning |
|---|---|---|
| `geoms` | `"feet"` | `"feet"`: the robot geoms of the foot-floor pairs collide with the terrain; `"body"`: every robot geom with a floor pair. Only spheres and capsules collide with the terrain (other geom types raise `ValueError`). |
| `floor` | `False` | keep the model's floor plane pairs as well. A terrain usually has its own ground (mjlab's does). |
| `mode` | `"analytic"` | heightfield contacts: `"analytic"` or `"native"` (below) |
| `cap_points` | `3` | heightfield sample points per capsule (analytic mode) |
| `maxcon` | the model's | contact capacity per world. Worlds beyond it are flagged and counted, as without terrain. |

Each robot geom that collides with the terrain gets one terrain contact pair. Its parameters (condim, friction, solref,
solimp) are those of the geom's pair with the floor, and foot pairs keep the per-world `friction` input.

`set_terrain(ter, cell=1.0)` also builds the broad phase: a uniform 2-D grid of `cell`-metre cells, each listing the
boxes and heightfields that overlap it.

## Contacts

- **Boxes:** MuJoCo's closed-form sphere-box and capsule-box collisions, the same as MuJoCo Warp's.
- **Heightfields, `analytic`:** fixed sample points (a sphere's centre; `cap_points` points along a capsule's axis)
  against the plane of the heightfield triangle below each point (cells split along MuJoCo's diagonal). This is a
  smooth approximation of MuJoCo's prism contacts.
- **Heightfields, `native`:** MuJoCo's heightfield rule. Each cell under the geom has two triangular prisms, each prism
  is measured exactly against the sphere or capsule, and at most 4 contacts are kept per geom and heightfield, chosen by
  MuJoCo's rule (the deepest, the farthest from it, the farthest from the line between them, and a fourth).

MuJoCo Warp computes heightfield contacts with its general convex collision (GJK / EPA per prism) and its own candidate
selection. On heightfields, both modes therefore differ from it by construction (numbers below).

## Height scan

`grid_pattern(size, resolution, direction)` makes the rays of mjlab's `GridPatternCfg`.
`raycast(pos, rot, offsets, directions, alignment="base", max_distance=10.0)` casts them from a frame (`pos (B, 3)`,
`rot (B, 3, 3)`, e.g. a body's pose from `kinematics()`), with `alignment` `"base"`, `"yaw"` or `"world"`, as mjlab's
`RayCastSensor`.

It returns `distances (B, N)` (-1 when nothing is hit within `max_distance`), `normals_w (B, N, 3)` and
`hit_pos_w (B, N, 3)`. Only the terrain is hit, not the robot. The rays are forward only (no gradient); the
intersections are MuJoCo Warp's `ray_box` / `ray_hfield`.

## Accuracy

Source: `scripts/terrain_check.py`, results in `docs/results/terrain_check_v100.json`.

Setup: V100, MuJoCo Warp 3.17.0, one control step (4 substeps, 10 iterations) of 1024 G1 states placed on the test
terrain of `fixtures/mjlab/g1_terrain_feet.mjb`. The test terrain has stairs of six 5 cm steps, a rotated box obstacle,
and a 41 x 41 heightfield with heights up to 6 cm.

Columns: the median relative error of qvel after the step, and the share of worlds with the same contact count, against
MuJoCo Warp.

| region | analytic | native |
|---|---|---|
| stairs (boxes) | 8e-7, 100 % | 8e-7, 100 % |
| box obstacle | 1.2e-6, 100 % | 1.2e-6, 100 % |
| heightfield | 6.4e-2, 44 % | 5.8e-2, 44 % |
| mjlab `ROUGH_TERRAINS_CFG` (2319 boxes, 69 heightfields; generated with mjlab, not committed) | 2.0e-6, 85 % | 2.0e-6, 86 % |

With every robot geom that has a floor pair colliding (`geoms="body"`, `fixtures/mjlab/g1_terrain_body.mjb`), the
numbers are the same to the digits shown (9e-7 on the stairs).

**Gradients.** The engine's VJP against float32 central differences of the engine itself. Median relative error,
qpos / qvel / ctrl:

| build | qpos | qvel | ctrl |
|---|---|---|---|
| test terrain, analytic | 0.49 % | 0.62 % | 0.054 % |
| test terrain, native | 0.67 % | 0.74 % | 0.067 % |
| no terrain, the floor (the differences' own noise) | 0.47 % | 0.48 % | 0.063 % |

In the native mode, a heightfield contact point is differentiated as a point fixed to the geom. This is exact for a
capsule's end points; the sliding term along the capsule is left out.

**Height scan** against MuJoCo Warp's `rays` (two ray grids from the pelvis, about 315,000 rays): hits agree for 100 %
of the rays, and distances agree within 1e-5. The PyTorch and JAX interfaces give bitwise the same steps and
gradients; their ray distances agree within 3e-7.

## Timing

Time per 20 ms control step of the G1 at 4096 worlds on one RTX PRO 6000 (10 iterations, medians of 30;
`geoms="feet"`; `docs/results/terrain_time_rtx6000.json`):

| scene | build | forward ms | forward + backward ms |
|---|---|---|---|
| test terrain | no terrain (the floor) | 0.986 | 1.627 |
| test terrain | analytic | 1.539 | 2.294 |
| test terrain | native | 2.183 | 3.105 |
| mjlab `ROUGH_TERRAINS_CFG` | no terrain (the floor) | 1.088 | 1.743 |
| mjlab `ROUGH_TERRAINS_CFG` | analytic | 1.513 | 2.253 |
| mjlab `ROUGH_TERRAINS_CFG` | native | 2.091 | 2.963 |

The floor rows use the same states without the terrain, so the difference to them is the terrain's cost.
