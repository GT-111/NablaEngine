#!/usr/bin/env bash
# One-command verification of the G1 engine on the public corpora (benchmarks/data/mf_s0, mf_s1):
#   scripts/verify.sh [--quick] [OUTDIR]          (default OUTDIR: runs/verify)
#
# --quick (a few minutes on one GPU):
#   env              scripts/env_check.py (versions, nvcc, the fixture model, a test kernel, JAX's devices)
#   step_mf_s0       one control step of 256 worlds of mf_s0, the truth file's sampled states first, against the
#                    MuJoCo Warp reference (native mujoco-warp 3.17.0) and the MuJoCo C float64 truth, and the
#                    outputs' digests (scripts/engine_step_check.py --select truth --gate)
#   gradient_G0      the G0 group of mf_s0: the JAX interface's VJPs against the finite-difference truth
#                    (scripts/gradient_check.py --groups G0)
#   ffi_bitwise      the JAX interface bitwise against the check binaries for the interface default (clamp_kv 0,
#                    normal 3) at 50 iterations, and its forward bitwise against this tree's engine_step_check.py
#                    (scripts/jax_ffi_check.py --sections bitwise)
#   capability       the model checks (scripts/capability_check.py, CPU)
# Without --quick (about 15 minutes on one V100): env; per corpus mf_s0 and mf_s1 the control step on all 4096 states
# with the forward-tape checks, on the truth's states, and the contact sets of one substep; every gradient group of
# both corpora; every section of scripts/jax_ffi_check.py and scripts/kinematics_check.py; capability.
#
# Each check prints one line, PASS or FAIL, with its log (and JSON) in OUTDIR; the exit status is 1 when any check
# failed, else 0. The pass criteria are the scripts' own (their GATES / THRESHOLDS, documented in each script).
# Needs: the packages of requirements.txt, nvcc of CUDA 12 or newer ($NVCC, CUDA_HOME or on PATH), one GPU (the first
# visible one), PYTHONPATH=src (the default here) or an installed nablaengine; $PYTHON names the interpreter.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
QUICK=0
if [ "${1:-}" = "--quick" ]; then QUICK=1; shift; fi
O=${1:-runs/verify}
mkdir -p "$O" || exit 1
export PYTHONPATH=${PYTHONPATH:-src}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
PY=${PYTHON:-python}
FAILED=()

check() {  # name command...: run it, log to OUTDIR/name.log, print PASS / FAIL
  local name=$1 t0=$SECONDS
  shift
  rm -f "$O/$name.json"  # a result left by an earlier run must not stand in for this one
  if "$@" > "$O/$name.log" 2>&1; then
    echo "PASS  $name ($((SECONDS - t0)) s)"
  else
    echo "FAIL  $name ($((SECONDS - t0)) s; $O/$name.log)"
    FAILED+=("$name")
  fi
}

check env "$PY" scripts/env_check.py
if [ "$QUICK" = 1 ]; then
  check step_mf_s0 "$PY" scripts/engine_step_check.py --corpus mf_s0 --substeps 4 --nworld 256 --check 256 \
    --select truth --reps 0 --gate --out "$O/step_mf_s0.json"
  check gradient_G0 "$PY" scripts/gradient_check.py --corpora mf_s0 --groups G0 --out "$O/gradient_G0.json"
  check ffi_bitwise "$PY" scripts/jax_ffi_check.py --sections bitwise --corpora mf_s0 --variants 0,3 \
    --bitwise-iterations 50 --out "$O/ffi_bitwise.json"
else
  for c in mf_s0 mf_s1; do
    check "step_$c" "$PY" scripts/engine_step_check.py --corpus "$c" --substeps 4 --check 4096 --reps 0 --tape \
      --gate --out "$O/step_$c.json"
    check "step_truth_$c" "$PY" scripts/engine_step_check.py --corpus "$c" --substeps 4 --nworld 256 --check 256 \
      --select truth --reps 0 --gate --out "$O/step_truth_$c.json"
    check "contacts_$c" "$PY" scripts/engine_step_check.py --corpus "$c" --substeps 1 --check 4096 --reps 0 --gate \
      --out "$O/contacts_$c.json"
  done
  check gradient "$PY" scripts/gradient_check.py --corpora mf_s0,mf_s1 --out "$O/gradient.json"
  check ffi "$PY" scripts/jax_ffi_check.py --corpora mf_s0,mf_s1 --out "$O/ffi.json"
  check kinematics "$PY" scripts/kinematics_check.py --corpora mf_s0,mf_s1 --out "$O/kinematics.json"
fi
check capability env JAX_PLATFORMS=cpu "$PY" scripts/capability_check.py --out "$O/capability.json"

if [ ${#FAILED[@]} -eq 0 ]; then
  echo "verify: PASS"
  exit 0
fi
echo "verify: FAIL (${FAILED[*]})"
exit 1
