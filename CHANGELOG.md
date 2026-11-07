# Changelog

## Unreleased

- The Unitree H1 (`fixtures/h1`, nv 25), converted from MuJoCo Menagerie by `scripts/make_h1_model.py` after the mjlab
  G1 recipe and checked against MuJoCo C in float64 by `scripts/h1_check.py` (in `scripts/verify.sh`).
- Models with nv < 32: the Cholesky and triangular solves mask the lanes past NV; capability H3 is now a warning
  (checked on the H1 only). The G1's builds are unchanged (identical PTX and SASS on sm_70, sm_86, sm_89, sm_90).
- `nablaengine.fixtures.load_model(fixtures.H1)`: a fixtures directory holds `<prefix>_train.mjb` and
  `<prefix>_meta.json`, prefix `g1` or `h1`.
- `scripts/visualize_rollout.py` (a visual check against MuJoCo) and `examples/squat_trajopt.py` (trajectory
  optimization through the engine's gradients; `--robot h1` for the H1).

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
