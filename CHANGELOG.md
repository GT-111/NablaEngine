# Changelog

## 0.2.0

- **Other robots.** Models of the engine's class beyond the G1:
  - 7 to 64 dofs (the G1 has 35) and up to 8 children per body (quadrupeds);
  - joint damping and frictionloss;
  - box-box contacts following MuJoCo Warp's general convex collision (GJK, EPA, multi-contact clipping), with an
    exact adjoint;
  - `init(com_body=...)` names the body of the per-world COM offset.

  Validated against MuJoCo Warp and float64 MuJoCo on the Unitree Go1, Go2, A1 and H1, ANYbotics ANYmal C, Boston
  Dynamics Spot, Booster T1, ROBOTIS OP3 and Berkeley Humanoid (`docs/robots.md`, `docs/api.md` Other robots).
  `scripts/robots/` exports and checks a model. The G1's builds are unchanged (bitwise).
  The Unitree H1 was first brought to the engine by @ShaoxiongYao in #1; 0.2.0's implementation supersedes it.
- **Terrain.** Static boxes and heightfields shared by every world, uploaded at run time without recompiling:
  - MuJoCo's closed-form box contacts;
  - two heightfield contact modes (analytic sample points, or MuJoCo's prisms measured exactly);
  - the adjoint;
  - a height scan (`raycast`, as mjlab's `RayCastSensor`).

  Available in PyTorch and JAX (`init(terrain=TerrainConfig(...))`, `set_terrain`; `docs/terrain.md`). Checked against
  MuJoCo Warp with `scripts/terrain_check.py` on the committed scenes `fixtures/mjlab/g1_terrain_*.mjb`.

## 0.1.0

First release.

- The fused CUDA engine for the Unitree G1 (mjlab model): one control step for thousands of worlds per call, with
  MuJoCo's physics and an analytic adjoint of the step.
- JAX interface (`nablaengine.jax_ffi`, `docs/api.md`, extra `nablaengine[jax]`) and PyTorch interface
  (`nablaengine.torch_ffi`, `docs/api_torch.md`, extra `nablaengine[torch]`), with bitwise the same results; the
  PyTorch one needs no JAX and runs a library built once per GPU architecture.
- Diagnostics on the device, including worlds beyond the contact (64) or constraint-row (192) capacity, whose extra
  contacts / rows are dropped: counted and flagged, with optional warnings (`set_overflow_warnings`).
- Verification against MuJoCo Warp, float64 MuJoCo and MJX (`scripts/verify.sh`, `docs/verification.md`) and the
  benchmark harness (`benchmarks/`).
