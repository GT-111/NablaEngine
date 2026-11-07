# Third-party notices

NablaEngine is licensed under the Apache License 2.0 (`LICENSE`). This file lists the third-party material that the
repository contains or is derived from, with its source, its licence and the changes made. Section 5 lists the
third-party software that the checks compare with but that the repository does not include.

| Material | Licence | Copyright | Used in |
|---|---|---|---|
| MuJoCo Warp | Apache-2.0 | 2025 The Newton Developers | `src/nablaengine/g1_engine/step.py` (translated code); `src/nablaengine/g1_engine/adjoint_contact.py`, `src/nablaengine/reference/jax64.py` (built on those translations); conventions in `src/nablaengine/g1_engine/adjoint.py` and `benchmarks/bench/adapters/mjw_pr.py` |
| MuJoCo | Apache-2.0 | 2021 DeepMind Technologies Limited | through the MuJoCo Warp primitives in `step.py`; the model semantics the engine implements |
| MuJoCo MJX | Apache-2.0 | 2023 DeepMind Technologies Limited | `src/nablaengine/g1_engine/kinematics.py` (reproduced semantics, no code) |
| MuJoCo Menagerie, Unitree G1 | BSD-3-Clause | 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics") | `fixtures/mjlab/g1_train.mjb` |
| mjlab | Apache-2.0 | 2025, The mjlab Developers | `fixtures/mjlab/g1_train.mjb` (parameter values) |

## 1. MuJoCo Warp

- Source: <https://github.com/google-deepmind/mujoco_warp>.
- Licence: Apache License 2.0, the same text as `LICENSE`.
- Copyright 2025 The Newton Developers. The project's `AUTHORS` file lists Google LLC and NVIDIA Corporation.
- Version used: the head of the adjoint pull request google-deepmind/mujoco_warp#1535 ("hybrid analytic
  differentiability"), etaoxing/mujoco_warp@357a75d60a56d67d476942a1b6e54b3045ee8e87 (MuJoCo Warp 3.12 with an
  analytic adjoint).

Files:

- `src/nablaengine/g1_engine/step.py`, CUDA source `STEP_DEVICE`. It contains CUDA C++ translations of these
  functions:
  - from `mujoco_warp/_src/collision_primitive_core.py`: `plane_sphere`, `plane_capsule`, `plane_box`,
    `sphere_sphere`, `sphere_capsule`, `capsule_capsule`, `sphere_box`, `capsule_box` (with `capsule_box_witness`)
    and `box_box` (with `_compute_rotmore`);
  - from `mujoco_warp/_src/math.py`: `orthogonals` and `make_frame`, and the quaternion-to-matrix convention of the
    box primitives.

  Changes made:
  - rewritten as CUDA C++ device functions for one warp per simulated world, with the contact pairs spread over the
    lanes of the warp;
  - float32 arithmetic with explicit roundings, so that the analytic adjoint can replay the contacts bit for bit;
  - zero contact margin only, and pairs whose bounding spheres do not overlap are skipped;
  - intermediate values are recorded on the forward tape for the adjoint.

  The box primitives are compiled only for models with box geoms; the G1 model in `fixtures/mjlab` has none. The
  file header states the source, the copyright, the licence and the changes.
- `src/nablaengine/g1_engine/adjoint_contact.py`: new code, the vector-Jacobian products of the translated
  primitives of the G1 model (plane-sphere, plane-capsule, sphere-sphere, sphere-capsule, capsule-capsule) and of
  `make_frame`. So that its results can be compared with the pull request's adjoint, it follows that adjoint's
  conventions (`collision_adjoint.py`, `adjoint_util.py`, `constraint_adjoint.py`): the signs of the residual terms
  and the names of the cotangents. The file header states this.
- `src/nablaengine/reference/jax64.py`: float64 JAX ports of the five primitives above and of `make_frame`, written
  after the translations in `step.py`, so they derive from the same MuJoCo Warp functions. They are the reference of
  the unit checks of the adjoint.
- `src/nablaengine/g1_engine/adjoint.py`: an independent implementation of the step's adjoint; no code is copied.
  So that the two can be compared, it uses the conventions of the pull request's adjoint: the quaternion-integration
  VJP (`_adj_quat_integrate`), `actuator_lam`, `_dof_to_qpos` and `_contact_any_active`.
- `benchmarks/bench/adapters/mjw_pr.py`: the backward of this benchmark adapter follows the adjoint-graph pattern of
  the pull request's `contrib/diffsim/_rollout.py` (`Rollout._backward_step` and `_capture`; Copyright 2026 The Newton
  Developers, Apache License 2.0), applied to one control step.

## 2. MuJoCo

- Source: <https://github.com/google-deepmind/mujoco>.
- Licence: Apache License 2.0.
- Copyright 2021 DeepMind Technologies Limited.

Parts of the MuJoCo Warp collision primitives of section 1 follow MuJoCo's C implementation, and the translations in
`step.py` carry those parts over. The rest of the engine is its own code. It implements MuJoCo's documented model
semantics:

- in `src/nablaengine/g1_engine/source.py`: the composite-rigid-body mass matrix and the recursive Newton-Euler
  algorithm;
- in `src/nablaengine/g1_engine/step.py`: the constraint impedance from `solref` / `solimp`, pyramidal friction
  cones, the primal Newton solver and the `implicitfast` integrator.

No other MuJoCo source code is included. The package and the checks use the MuJoCo Python bindings (`mujoco` 3.17.0)
at run time.

## 3. MuJoCo MJX

- Source: `mujoco-mjx` 3.17.0, `mujoco/mjx/_src/smooth.py` (<https://github.com/google-deepmind/mujoco>).
- Licence: Apache License 2.0.
- Copyright 2023 DeepMind Technologies Limited.

`src/nablaengine/g1_engine/kinematics.py` is an independent CUDA implementation that reproduces the outputs of
`mjx.kinematics`, `com_pos` and `com_vel`. Its docstring cites MJX source line numbers and quotes one statement; no
MJX code is included. `scripts/kinematics_check.py` compares with MJX at run time.

## 4. The G1 model: MuJoCo Menagerie (Unitree G1) and mjlab

`fixtures/mjlab/g1_train.mjb` is a compiled MuJoCo model in the binary format of MuJoCo 3.17.0. It is derived from two
sources; `fixtures/mjlab/README.md` lists every change.

- **MuJoCo Menagerie**, `unitree_g1/scene_mjx.xml` (which includes `unitree_g1/g1_mjx.xml`) at commit
  f3b6013629c7b94f8ee93fd977b161fa8350f9bc:
  <https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/unitree_g1>.
  - Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics").
  - BSD 3-Clause License. `fixtures/mjlab/LICENSE` is a verbatim copy of Menagerie's `unitree_g1/LICENSE`; its text
    is reproduced below.
  - Changes made: the foot collision geoms and the contact pairs replaced, contact, joint and actuator parameters
    changed, the timestep set to 5 ms, the meshes and textures removed.
- **mjlab**, <https://github.com/mujocolab/mjlab>.
  - Apache License 2.0. Copyright 2025, The mjlab Developers.
  - The model takes over values of mjlab's G1 configuration (`mjlab.asset_zoo.robots.unitree_g1`): the sole
    capsules, the contact set, the contact and joint parameters, the PD gains and the setpoint range. No mjlab code
    is included.

Unitree Robotics, Google DeepMind and the mjlab developers do not endorse this project.

The licence of the Unitree G1 description (`unitree_g1/LICENSE` of MuJoCo Menagerie):

```
Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 5. Used for comparison, not included

- **MuJoCo Warp 3.17.0** (PyPI `mujoco-warp`, Apache-2.0): the released reference for the forward step.
- **The adjoint pull request google-deepmind/mujoco_warp#1535**, the reference for the gradients, in two trees:
  - etaoxing/mujoco_warp@357a75d60a56d67d476942a1b6e54b3045ee8e87, the head of the pull request, unmodified;
  - etaoxing/mujoco_warp@4af093f02a3f48f0009ca7c916674b928136f310, the same tree with etaoxing/mujoco_warp#1: two
    public fixes of the pull request's adjoint by Ruslan Rakhimov (commits c8c8890d0cf431c6e2b92f035cfb235d924e73c8 and
    4af093f02a3f48f0009ca7c916674b928136f310).

  Both trees are fetched from GitHub at these commits; this repository includes none of their code.
- **NVIDIA Warp 1.18.0** (`warp-lang`, Apache-2.0): the runtime of MuJoCo Warp.
- **MuJoCo 3.17.0** (the C engine, through its Python bindings, in float64): the finite-difference reference.
- **jaxlib's XLA FFI headers** (Apache-2.0): the FFI library is compiled against the headers that the installed jaxlib
  provides (`jax.ffi.include_dir()`). No header is copied into this repository.

## 6. Cited algorithms

- `src/nablaengine/fixtures.py`: the closest distance between two segments, reimplemented in numpy after C. Ericson,
  *Real-Time Collision Detection* (2005), section 5.1.9.
