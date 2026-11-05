"""The G1 identity gate: is every production build unit of the mjlab G1 still what the committed manifest says?

A refactor, a feature behind a generated switch that is off for the G1, or a new model check must leave the G1's builds
unchanged (a G1-neutral change). This script regenerates every production build unit for the G1 fixture (fixtures/mjlab/g1_train.mjb) per build target with the production build functions, hashes it at the levels
of nablaengine.identity (L0 generated source, -D options, packed constants and nvcc command; L1 preprocessed source of
the device and host passes; L2 normalized PTX, SASS, resource usage and host translation) and compares the hashes with
the manifest docs/results/identity_g1_mjlab.json.

Units per target (sm_70, sm_86, sm_90, sm_120):
  ffi/<unit>.cu    the JAX FFI library's translation units, from nablaengine.jax_ffi.build_plan (the 12 adjoint variants
                   dk_kv<clamp_kv>_n<normal>.cu, one of which holds the forward and kinematics kernels, the FFI unit
                   dk_ffi.cu with the XLA handlers; with $DK_SM120_FRONTEND=split also dk_fwd.cu), each with
                   jax_ffi.unit_command's nvcc arguments
  check/step.cu    scripts/engine_step_check.py's binary (binary_source; split forward, the default), compiled as its
                   main() does (cuda.compile_command, unit "forward")
  check/step_fused.cu  the same with --fused (-DDK_SPLIT=0)
  check/adj.cu     the adjoint check binary (nablaengine.checks.binary_source) with the exact gradient mode's defines
                   (DK_ADJ_CLAMP_KV=1 DK_ADJ_STAGE=3 DK_ADJ_NORMAL=0 DK_ADJ_PARALLEL=0), built as checks.build does
                   (binary_builds: one build, or a forward-unit build check/adj_fwd.cu besides where the forward unit
                   has flags of its own)
A target nvcc cannot build itself (sm_120 with nvcc 12.4) is built from its virtual architecture's PTX with a separate
ptxas ($PTXAS, else the pip nvidia.cuda_nvcc package's): see nablaengine.identity.

Checking (default): per unit L0 first; where it differs L1, where that differs L2 (--level caps the deepest level).
A unit passes at the first identical level that can clear its L0 difference (nablaengine.identity.clearing_level):
identical L0 means nothing changed, identical L1 a change of the source or the -D options that the G1 build compiles
out, identical L2 a change of dead code, comments or nvcc options without effect. Two L0 differences are never cleared
by L1: different packed constants fail at once (the units are uploaded with them at run time; no compiler output
contains them), and a change of an nvcc option other than -D needs an identical L2 (it acts after preprocessing).
The default mode is the one for a G1-neutral feature (code behind a generated switch that is off for the G1); --full
computes every level of every unit and requires all of them identical, L0 included (the acceptance form after a pure
refactor, which leaves the generated source as it was). L1 and L2 entries are per toolchain (target, nvcc, host
compiler, separate ptxas, cuobjdump: SASS depends on all of them); without an entry for this machine's toolchain only
L0 is compared (exit 2 unless --level 0). A unit that differs at every level computed changes what the G1 runs: it
needs the full procedure (engine_step_check --tape against the forward digests, jax_ffi_check, gradient_check, then a
new manifest and new digest entries).

  PYTHONPATH=src python scripts/identity_gate.py                                # check (escalating)
  PYTHONPATH=src python scripts/identity_gate.py --full --json runs/identity_check.json
  PYTHONPATH=src python scripts/identity_gate.py --write --out runs/identity_new.json   # a manifest of this tree
  PYTHONPATH=src python scripts/identity_gate.py --write --out docs/results/identity_g1_mjlab.json --force
      # (re)write the committed manifest's entries of this machine's toolchains (other entries are kept)

cuobjdump: $CUOBJDUMP, else next to nvcc, on PATH or in a conda package cache (nablaengine.identity.find_tool); it must
know every target it disassembles (sm_120: CUDA 12.8 or newer). Build switches that change units ($DK_SM120_FRONTEND,
$DK_SM90_REG_CAPS, $DK_SM120_PRE_CAP, $DK_MAXCON, $DK_TEST_*) must be unset (--allow-env to run anyway; not for --write).
Exit status: 0 identical, 1 a unit differs (or is missing / new), 2 not comparable (no entry for this toolchain, a tool
missing). Temporary build directories are removed at exit (--keep-work keeps them).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import engine_step_check as ESC  # noqa: E402  (binary_source: the forward check binary)

from nablaengine import checks, cuda, fixtures, identity, jax_ffi, provenance, workdir  # noqa: E402

MANIFEST = ROOT / "docs" / "results" / "identity_g1_mjlab.json"
TARGETS = ("sm_70", "sm_86", "sm_90", "sm_120")
INCLUDE = "jax_ffi_include"  # placeholder of the XLA FFI headers' directory (dk_ffi.cu's -I)
EAC_DEFINES = ("DK_ADJ_CLAMP_KV=1", "DK_ADJ_STAGE=3", "DK_ADJ_NORMAL=0", "DK_ADJ_PARALLEL=0")  # the exact mode
BUILD_ENV = ("DK_SM120_FRONTEND", "DK_SM120_FORWARD_FRONTEND", "DK_SM90_REG_CAPS", "DK_SM120_PRE_CAP", "DK_MAXCON",
             "DK_TEST_FORWARD_VIRTUAL_ARCH", "DK_TEST_VIRTUAL_ARCH", "DK_FORWARD_SPLIT", "DK_SPLIT_ORDER",
             "DK_SPLIT_PRE_MINB", "DK_SPLIT_POST_MAXNREG", "DK_SPLIT_POST_MINB")
ABOUT = ("Identity manifest of the mjlab G1's production build units (scripts/identity_gate.py, nablaengine.identity): "
         "per target and toolchain the L0 / L1 / L2 hashes of every FFI library unit and check binary")


def ffi_units(m, foot, target, include):
    """The FFI library's units for target (jax_ffi.build_plan, jax_ffi.unit_command)."""
    plan = jax_ffi.build_plan(m, foot, target)
    out = []
    for name in sorted(plan["units"]):
        cmd = jax_ffi.unit_command(name, target, plan["defines"], plan["unit_arch"], "{" + INCLUDE + "}")
        out.append(identity.Unit(name=f"ffi/{name}", file=name, source=plan["units"][name],
                                 command=[*cmd, "-c", "-o", "{out}", "{src}"], defines=list(plan["defines"][name]),
                                 consts=plan["consts"], include={INCLUDE: include} if name == "dk_ffi.cu" else {}))
    return out


def _check_command(extra, unit, target):
    cmd = cuda.compile_command("{src}", "{out}", extra, unit=unit, target=target)
    return cmd[1:]  # without the nvcc path


def check_units(m, mt, target):
    """The check binaries' units for target (engine_step_check.binary_source, nablaengine.checks.binary_source and
    binary_builds, cuda.compile_command)."""
    src, consts = ESC.binary_source(m, mt["foot_pairs"])
    out = []
    for name, extra in (("check/step.cu", []), ("check/step_fused.cu", ["-DDK_SPLIT=0"])):
        cmd = _check_command(extra, "forward", target)
        out.append(identity.Unit(name=name, file="step.cu", source=src, command=cmd,
                                 defines=[a for a in cmd if a.startswith("-D")], consts=consts))
    src, consts = checks.binary_source(m, mt)
    extra = [f"-D{d}" for d in EAC_DEFINES]
    for suffix, unit in checks.binary_builds(target):
        cmd = _check_command(extra, unit, target)
        out.append(identity.Unit(name=f"check/adj{suffix}.cu", file="adj.cu", source=src, command=cmd,
                                 defines=[a for a in cmd if a.startswith("-D")], consts=consts))
    return out


def model_record(d):
    mjb = Path(d) / "g1_train.mjb"
    m, mt = fixtures.load_model(d)
    return m, mt, dict(mjb=provenance.rel(mjb), sha256=identity.sha256(mjb.read_bytes()),
                       mujoco=mt["mujoco_version"], foot_pairs=[int(p) for p in mt["foot_pairs"]])


def compute(tc, unit, levels, work_root):
    """{L0: ..., L1: ..., L2: ...} of the levels asked for (L0 always)."""
    out = dict(defines=unit.defines, command=" ".join([identity.NVCC_PLACEHOLDER, *unit.command]),
               L0=identity.level0(unit))
    for lv in (1, 2):
        if lv in levels:
            d = workdir.work_dir(f"id_L{lv}_", dir=work_root)
            out[f"L{lv}"] = getattr(tc, f"level{lv}")(unit, d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=str(MANIFEST), help="the manifest to check against (default: the committed one)")
    ap.add_argument("--arch", default=",".join(TARGETS), help="comma-separated build targets")
    ap.add_argument("--units", default="ffi,check", help="ffi, check or both")
    ap.add_argument("--only", default="", help="only the units whose name contains one of these comma-separated "
                    "strings (not with --write)")
    ap.add_argument("--level", type=int, default=2, choices=identity.LEVELS, help="the deepest level computed")
    ap.add_argument("--full", action="store_true", help="every level of every unit, all required identical")
    ap.add_argument("--write", action="store_true", help="compute every level and write a manifest to --out")
    ap.add_argument("--out", default="", help="--write: the manifest to write (entries of other toolchains kept)")
    ap.add_argument("--force", action="store_true", help="--write: allow --out to be the committed manifest")
    ap.add_argument("--json", default="", help="write the comparison report here")
    ap.add_argument("--jobs", type=int, default=min(8, len(os.sched_getaffinity(0))), help="parallel compiles")
    ap.add_argument("--fixtures", default=str(fixtures.DEFAULT), help="the G1 fixtures directory (not $DK_FIXTURES)")
    ap.add_argument("--allow-env", action="store_true", help="run although build switches are set (not with --write)")
    workdir.add_argument(ap)
    args = ap.parse_args(argv)
    workdir.configure(args)
    env = {k: os.environ[k] for k in BUILD_ENV if os.environ.get(k)}
    if env and (args.write or not args.allow_env):
        ap.error(f"build switches set ({env}): unset them (--allow-env to compare anyway; never with --write)")
    if args.write and args.only:
        ap.error("--only is for checks; a manifest holds every unit")
    if args.write:
        if not args.out:
            ap.error("--write needs --out")
        if Path(args.out).resolve() == MANIFEST.resolve() and not args.force:
            ap.error(f"--out is the committed manifest {provenance.rel(MANIFEST)}: add --force to rewrite its entries")
    targets = [t.strip() for t in args.arch.split(",") if t.strip()]
    kinds = set(args.units.split(","))
    levels = set(identity.LEVELS) if (args.full or args.write) else set(range(args.level + 1))
    import jax.ffi
    include = jax.ffi.include_dir()
    m, mt, model = model_record(args.fixtures)
    tc = identity.Toolchain()
    old = None if args.write and not Path(args.out).exists() else json.loads(
        Path(args.out if args.write else args.manifest).read_text())
    if old is not None and old.get("model", {}).get("sha256") != model["sha256"] and not args.write:
        print(f"manifest model {old.get('model')} is not this fixture {model}", file=sys.stderr)
        return 1
    work_root = workdir.work_dir("identity_gate_")
    t0 = time.time()
    report, status = dict(targets={}), 0
    new_targets = {}
    for target in targets:
        units = ((ffi_units(m, mt["foot_pairs"], target, include) if "ffi" in kinds else [])
                 + (check_units(m, mt, target) if "check" in kinds else []))
        only = [x for x in args.only.split(",") if x]
        if only:
            units = [u for u in units if any(x in u.name for x in only)]
        route = tc.route(target)
        key = tc.key(target) if route else f"{target} (no compiler)"
        ref = None if old is None else old["targets"].get(key)
        same_arch = [] if old is None else [v for v in old["targets"].values() if v["arch"] == target]
        rep = dict(toolchain_key=key, route=route, units={})
        report["targets"][target] = rep
        if args.write:
            if route is None:
                print(f"{target}: no nvcc or ptxas targets it: skipped", file=sys.stderr)
                status = max(status, 2)
                continue
            with ThreadPoolExecutor(args.jobs) as ex:
                res = list(ex.map(lambda u: compute(tc, u, levels, work_root), units))
            new_targets[key] = dict(arch=target, toolchain=tc.record(target), route=route,
                                    units={u.name: r for u, r in zip(units, res)})
            rep["units"] = {u.name: "written" for u in units}
            continue
        l0_ref = ref or (same_arch[0] if same_arch else None)
        if l0_ref is None:
            rep["error"] = "no manifest entry for this target"
            status = max(status, 2)
            continue
        if ref is None:
            other = next(k for k, v in old["targets"].items() if v is l0_ref)
            rep["note"] = f"no entry for this toolchain ({key}); L0 compared with {other}"
        names = {u.name for u in units}
        for n in sorted(set(l0_ref["units"]) - names) if not only and kinds == {"ffi", "check"} else []:
            rep["units"][n] = dict(verdict="missing", detail="in the manifest, not generated")
            status = 1
        lv_ok = levels if ref is not None else {0}
        if ref is None and levels - {0}:
            status = max(status, 2)

        def check(u):
            entry = l0_ref["units"].get(u.name)
            if entry is None:
                return dict(verdict="new", detail="generated, not in the manifest")
            cur = compute(tc, u, {0}, work_root)
            diff = {0: identity.compare(entry, cur)[0]}
            need = identity.clearing_level(entry, cur)  # the first level that may clear L0's difference (None: none)
            todo = [lv for lv in (1, 2) if lv in lv_ok]
            for lv in todo:
                if not args.full and (need is None or (not diff[lv - 1] and lv - 1 >= need)):
                    break
                d = workdir.work_dir(f"id_L{lv}_", dir=work_root)
                cur[f"L{lv}"] = getattr(tc, f"level{lv}")(u, d)
                diff[lv] = identity.compare(ref["units"][u.name], cur)[lv]
            reason = None
            if args.full:
                ok = all(not v for v in diff.values()) and set(diff) >= lv_ok
                verdict = "identical" if ok else "differs"
            else:
                first = next((lv for lv in sorted(diff) if not diff[lv] and need is not None and lv >= need), None)
                verdict = f"identical at L{first}" if first is not None else "differs"
                if first is None and need is None:
                    reason = "the packed constants differ (run-time data: no compiler level can clear it)"
                elif first is None and need == 2:
                    reason = ("nvcc options other than -D differ: only an identical L2 clears it"
                              + ("" if 2 in diff else " (L2 not computed)"))
            return dict(verdict=verdict, differences={f"L{k}": v for k, v in diff.items()}, defines=u.defines,
                        **({"reason": reason} if reason else {}))

        with ThreadPoolExecutor(args.jobs) as ex:
            for u, r in zip(units, ex.map(check, units)):
                rep["units"][u.name] = r
                if r["verdict"] in ("differs", "new"):
                    status = 1
        if ref is None and lv_ok == {0} and levels - {0}:
            rep["levels_not_comparable"] = sorted(levels - {0})
    if args.write:
        man = old if old is not None else identity.manifest_skeleton(ABOUT, model)
        man["model"] = model
        man["targets"].update(new_targets)
        man["targets"] = dict(sorted(man["targets"].items()))
        man["provenance"] = provenance.provenance()
        for k in ("gpu", "mujoco_warp", "environment"):  # compile only: no GPU; build switches are unset (checked)
            man["provenance"].pop(k, None)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(man, indent=1) + "\n")
        print(f"wrote {provenance.rel(args.out)}: {sum(len(v['units']) for v in new_targets.values())} units, "
              f"{len(new_targets)} toolchain entries", flush=True)
    report.update(status=status, levels=sorted(levels), full=args.full, wall_s=round(time.time() - t0, 1),
                  manifest=provenance.rel(args.out if args.write else args.manifest), model=model,
                  provenance=provenance.provenance())
    for t, rep in report["targets"].items():
        verdicts = {}
        for r in rep["units"].values():
            v = r if isinstance(r, str) else r["verdict"]
            verdicts[v] = verdicts.get(v, 0) + 1
        print(f"{t} [{rep['toolchain_key']}]: {verdicts} {rep.get('note', '') or rep.get('error', '')}", flush=True)
        for n, r in rep["units"].items():
            if isinstance(r, dict) and r["verdict"] in ("differs", "new", "missing"):
                print(f"  {n}: {r['verdict']} {r.get('differences', r.get('detail'))}"
                      + (f": {r['reason']}" if r.get("reason") else ""), flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    print(f"identity gate: {'PASS' if status == 0 else 'FAIL' if status == 1 else 'NOT COMPARABLE'} "
          f"({report['wall_s']} s)", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
