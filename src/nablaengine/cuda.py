"""nvcc and target architecture for the hand-written CUDA kernels.

nvcc (nvcc()): $NVCC, else nvcc on PATH, else $CUDA_HOME/bin/nvcc, else a pip package nvidia.cuda_nvcc that has one (the
nvidia-cuda-nvcc-cu12 wheel of the JAX stack ships only ptxas, not nvcc), else /usr/local/cuda/bin/nvcc. Architecture:
$DK_ARCH (e.g. sm_70 for V100, sm_120 for RTX PRO 6000 Blackwell), else the compute capability of the first visible GPU
(the first entry of $CUDA_VISIBLE_DEVICES, else GPU 0) from nvidia-smi. CUDA 13 dropped Volta (sm_70), and sm_120 needs
CUDA 12.8 or newer: CUDA 12.8 / 12.9 covers every GPU tested.

Architecture flags per compilation unit (arch_flags, virtual_arch): every unit is built with -arch=<arch>, except on
sm_120. There nvcc 12.8 / 12.9 compile compute_120 with the NVVM 20 front end, and both the forward and the adjoint
are faster built from compute_90 PTX (the NVVM 7 front end, assembled by ptxas for sm_120: -arch=compute_90
-code=sm_120). RTX PRO 6000 Server Edition, 4096 worlds: the NVVM 7 forward 0.955 against the native one (the native
forward 1.0475 against it), the NVVM 7 adjoint 0.893 against the native one (cycles -11.1%): with 10108 B of shared
memory per world (struct Adj) the native k_adjoint (238 registers) is register-limited to 8 worlds per SM, the NVVM 7
one (about 127 registers) fits the 9 that shared memory allows. So on sm_120 every unit (forward, kinematics, the
adjoint variants, the FFI unit) takes the NVVM 7 path by default. Switches:

  DK_SM120_FRONTEND          n7 (default) | split | native: on sm_120, n7 builds every unit from compute_90 PTX; split
                             only the forward / kinematics unit (the adjoint units -arch=sm_120); native every unit
                             -arch=sm_120. It changes bits on sm_120: the forward's between native and n7 / split (n7
                             and split have the same forward bits, so their forward references share the tag
                             +fwd-compute_90), the gradients' between n7 and split / native; other architectures are not
                             affected. n7 / split take effect only where the PTX headers confirm them (nvvm_version:
                             compute_90 is NVVM 7 and the native compute_120 is not); otherwise every unit keeps the
                             native build, with a note on stderr.
  DK_SM120_FORWARD_FRONTEND  deprecated alias, read only where DK_SM120_FRONTEND is unset (with a DeprecationWarning):
                             n7 -> n7, native -> native.
  DK_TEST_FORWARD_VIRTUAL_ARCH  testing only (default unset): on any GPU, build the forward unit from this virtual
                             architecture's PTX (e.g. compute_70 on a V100: -arch=compute_70 -code=sm_70): the split
                             structure (a forward unit of its own, check binaries in pairs) without an sm_120 GPU.
  DK_TEST_VIRTUAL_ARCH       testing only (default unset): every unit from this virtual architecture's PTX (sm_120's
                             default structure: one set of flags, the forward reference tagged +fwd-<virtual>). Both are
                             bitwise the native build where that PTX is the native one (compute_70 on sm_70).

Defines per compilation unit (unit_defines): the forward unit (whatever holds the split forward kernels k_pre / k_post:
the FFI library's and the check binaries') gets register caps on some targets (REG_CAPS), decided by the build target,
not by __CUDA_ARCH__ in the CUDA source (sm_120's units are compiled from compute_90 PTX, where __CUDA_ARCH__ is 900 as
well). All of them are bitwise unchanged (outputs, contacts and tape):
  sm_90 / sm_90a  -DSPLIT_PRE_MINB=16 -DSPLIT_POST_MAXNREG=128 (g1_engine/host.py: k_pre at 16 blocks per SM, i.e. 128
                  registers, and k_post __maxnreg__(128)): on the H100 SXM5 16 worlds per SM for both kernels and a
                  10.8% faster forward (k_pre's cap alone -3.6%, k_post's -7.1%; 4096 worlds, nvcc 12.8). Switch
                  DK_SM90_REG_CAPS 1 (default) | 0. Needs nvcc 12.4+ (__maxnreg__): an older nvcc builds without the
                  caps, with a note on stderr.
  sm_120          -DSPLIT_PRE_MINB=16 with the NVVM 7 forward (DK_SM120_FRONTEND n7 or split; not native): on the RTX
                  PRO 6000 k_pre at 128 registers is 1.3% faster (0.987 against the uncapped build, no occupancy change
                  there, shared memory allows 12 k_pre blocks: code generation only). k_post's __maxnreg__(128) is not
                  taken there (1.004 alone, 0.992 with k_pre's cap). Switch DK_SM120_PRE_CAP 1 (default) | 0.
  A caller's own -DSPLIT_PRE_MINB (k_pre), -DSPLIT_POST_MAXNREG or -DSPLIT_POST_MINB (k_post) replaces the cap of that
  kernel. sm_86 / sm_70 get none (the V100: forced caps 4% slower, shared memory allows 12 k_post blocks).

Forward build switches (forward_split, split_order, cap_overrides: $DK_FORWARD_SPLIT, $DK_SPLIT_ORDER, $DK_SPLIT_*):
a build of the forward kernels (jax_ffi.library_sources, both interfaces and the check binaries that go through it) can
select the fused forward and its launch scheduling, all of them arithmetic-neutral (the source and its operations are
host.py's; the DK_SPLIT and DK_SPLIT_ORDER options are checked bitwise, a changed register cap is a code-generation
change: check it with --tape --save-outputs):
  DK_FORWARD_SPLIT  1 (default) builds the split forward (k_pre / k_post, host.DK_SPLIT 1: 2 nsub launches per control
                    step, the per-world Hand in global memory); 0 the fused k_substep (host.DK_SPLIT 0: one launch for
                    all nsub substeps, the state stays in struct Sim, no Hand traffic and no DK_SPLIT_ORDER memory).
                    k_substep's dynamic shared memory is sizeof(Sim) (its cudaFuncSetAttribute opt-in is in the unit's
                    upload), i.e. more than sizeof(SimP) / sizeof(SimN) but with no __maxnreg__ / launch-bound cap, and a
                    small batch (a few hundred worlds, less than one resident wave) does not pay the split kernels' two
                    launches per substep. Which is faster depends on the model's NV and the worlds per rank.
  DK_SPLIT_ORDER    1 (default) / 0: k_post dispatches the worlds by constraint-row count (host.DK_SPLIT_ORDER):
                    scheduling only, but at about one resident wave the memset and the per-world bucket atomics are a
                    fixed cost per launch with no makespan to gain.
  DK_SPLIT_PRE_MINB, DK_SPLIT_POST_MAXNREG, DK_SPLIT_POST_MINB  a cap of that kernel in place of the target's
                    (cap_overrides; MAXNREG and MINB are alternatives: MINB replaces __maxnreg__, 0 bounds nothing).
  These are build options (in the library cache key through `defines`); the check binaries' own -D options still win.
"""

import functools
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

FRONTEND_VAR = "DK_SM120_FRONTEND"
LEGACY_FRONTEND_VAR = "DK_SM120_FORWARD_FRONTEND"  # deprecated alias (n7 / native)
FRONTENDS = ("n7", "split", "native")
TEST_FORWARD_VAR = "DK_TEST_FORWARD_VIRTUAL_ARCH"
TEST_VAR = "DK_TEST_VIRTUAL_ARCH"
N7_VIRTUAL = {"sm_120": "compute_90"}  # architecture -> the virtual architecture of its NVVM 7 units
SM90_CAPS = ("SPLIT_PRE_MINB=16", "SPLIT_POST_MAXNREG=128")
# build target -> (its switch, the forward unit's register caps, only with the NVVM 7 forward); module docstring
REG_CAPS = {"sm_90": ("DK_SM90_REG_CAPS", SM90_CAPS, False),
            "sm_90a": ("DK_SM90_REG_CAPS", SM90_CAPS, False),
            "sm_120": ("DK_SM120_PRE_CAP", ("SPLIT_PRE_MINB=16",), True)}
# a cap's macro -> the macros by which a caller already bounds that kernel
CAP_BOUNDS = {"SPLIT_PRE_MINB": ("SPLIT_PRE_MINB",), "SPLIT_POST_MAXNREG": ("SPLIT_POST_MAXNREG", "SPLIT_POST_MINB"),
              "SPLIT_POST_MINB": ("SPLIT_POST_MINB", "SPLIT_POST_MAXNREG")}
MAXNREG_NVCC = (12, 4)  # __maxnreg__ (SPLIT_POST_MAXNREG): nvcc 12.4 or newer
FORWARD_SPLIT_VAR = "DK_FORWARD_SPLIT"  # 1 (default): k_pre / k_post; 0: the fused k_substep (module docstring)
SPLIT_ORDER_VAR = "DK_SPLIT_ORDER"  # 1 (default): k_post's worlds by row count; 0: the world order
CAP_VARS = tuple(f"DK_{name}" for name in ("SPLIT_PRE_MINB", "SPLIT_POST_MAXNREG", "SPLIT_POST_MINB"))  # cap_overrides


def nvcc():
    """The nvcc to build with: $NVCC, nvcc on PATH, $CUDA_HOME/bin/nvcc, a pip nvidia.cuda_nvcc package's nvcc, then
    /usr/local/cuda/bin/nvcc (the first that exists); FileNotFoundError if none does."""
    cands = [os.environ.get("NVCC"), shutil.which("nvcc")]
    if os.environ.get("CUDA_HOME"):
        cands.append(str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc"))
    try:
        import nvidia.cuda_nvcc
        cands.append(str(Path(list(nvidia.cuda_nvcc.__path__)[0]) / "bin" / "nvcc"))
    except ImportError:
        pass
    cands.append("/usr/local/cuda/bin/nvcc")
    for c in cands:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError(
        "no nvcc found: set NVCC to a CUDA 12.x nvcc (or CUDA_HOME, or put nvcc on PATH). The pip wheel "
        "nvidia-cuda-nvcc-cu12 has no nvcc (only ptxas); install a toolkit, e.g. conda's cuda-nvcc "
        "(scripts/setup_env.sh, README.md)")


def visible_gpu():
    """The first visible GPU as nvidia-smi names it (index or UUID): the first entry of $CUDA_VISIBLE_DEVICES, else
    None (nvidia-smi's GPU 0). nvidia-smi itself ignores CUDA_VISIBLE_DEVICES and numbers the GPUs in PCI order, CUDA
    fastest first unless CUDA_DEVICE_ORDER=PCI_BUS_ID: the same GPU on a machine with one GPU type."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return v if v and not v.startswith("-") else None


def device_arch():
    """sm_XY of the first visible GPU (nvidia-smi), None without nvidia-smi or a GPU."""
    q = ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    for cmd in ([*q, "-i", visible_gpu()] if visible_gpu() else None, q):  # an id nvidia-smi rejects: GPU 0
        if cmd is None:
            continue
        try:
            out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 universal_newlines=True).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            continue
        if out:
            return "sm_" + out[0].replace(".", "")
    return None


def arch():
    """$DK_ARCH, else the first visible GPU's architecture (device_arch). A $DK_ARCH that is not the GPU's warns (once
    per process): code built for an older target runs that target's code paths (CHOL_COMPACT follows __CUDA_ARCH__)
    from its SASS within the same major version, else from the driver's compilation of its PTX, so the results are not
    bitwise those of a build for the GPU; code built for a newer target does not load. No warning without a GPU (a
    build for another machine)."""
    if os.environ.get("DK_ARCH"):
        a = os.environ["DK_ARCH"]
        dev = device_arch()
        if dev is not None and dev != a:
            warnings.warn(f"DK_ARCH={a} but the GPU is {dev}: the CUDA code is built for {a} (unset DK_ARCH to build "
                          f"for the GPU)", RuntimeWarning, stacklevel=2)
        return a
    a = device_arch()
    if a is None:
        raise RuntimeError("no GPU found by nvidia-smi: set DK_ARCH (e.g. sm_70, sm_86, sm_90, sm_120) to build "
                           "without one")
    return a


def frontend():
    """$DK_SM120_FRONTEND: 'n7' (the default), 'split' or 'native'. The deprecated $DK_SM120_FORWARD_FRONTEND (n7 / native)
    is read where $DK_SM120_FRONTEND is unset, with a DeprecationWarning (n7 -> n7, native -> native)."""
    v = (os.environ.get(FRONTEND_VAR) or "").strip().lower()
    old = (os.environ.get(LEGACY_FRONTEND_VAR) or "").strip().lower()
    if old:
        _warn_legacy(old, bool(v))
        if not v:
            if old not in ("n7", "native"):
                raise ValueError(f"${LEGACY_FRONTEND_VAR}={old!r}: must be n7 or native (deprecated: use ${FRONTEND_VAR})")
            v = old
    v = v or "n7"
    if v not in FRONTENDS:
        raise ValueError(f"${FRONTEND_VAR}={v!r}: must be one of {', '.join(FRONTENDS)}")
    return v


@functools.lru_cache(maxsize=None)
def _warn_legacy(old, shadowed):
    warnings.warn(f"${LEGACY_FRONTEND_VAR} is deprecated: use ${FRONTEND_VAR} (n7 / split / native)"
                  + (f"; ${FRONTEND_VAR} is set and wins" if shadowed else f"; {old} is taken as ${FRONTEND_VAR}={old}"),
                  DeprecationWarning, stacklevel=4)


def nvvm_version(virt, nvcc_path=None):
    """The front end nvcc uses for a virtual architecture (compute_XY), from the header of the PTX it makes for a tiny
    kernel ('NVVM 7.0.1', 'NVVM 20.0.0'), or None if nvcc cannot make it."""
    return _nvvm_version(virt, nvcc_path or nvcc())


@functools.lru_cache(maxsize=None)
def _nvvm_version(virt, nvcc_path):
    with tempfile.TemporaryDirectory(prefix="dk_nvvm_") as d:
        src, ptx = Path(d) / "k.cu", Path(d) / "k.ptx"
        src.write_text("__global__ void k(float* a) { a[threadIdx.x] *= 2.f; }\n")
        r = subprocess.run([nvcc_path, f"-arch={virt}", "-ptx", "-o", str(ptx), str(src)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode or not ptx.exists():
            return None
        m = re.search(r"Based on NVVM ([\d.]+)", ptx.read_text())
        return f"NVVM {m.group(1)}" if m else None


@functools.lru_cache(maxsize=None)
def _n7_applies(target, nvcc_path):
    virt = N7_VIRTUAL[target]
    native, n7 = nvvm_version("compute_" + target[3:], nvcc_path), nvvm_version(virt, nvcc_path)
    if n7 and n7.startswith("NVVM 7.") and not (native or "").startswith("NVVM 7."):
        return virt
    print(f"nablaengine.cuda: {FRONTEND_VAR}={frontend()} not applied on {target} (front ends: compute_{target[3:]} "
          f"{native}, {virt} {n7}): every unit keeps the native build", file=sys.stderr)
    return None


@functools.lru_cache(maxsize=None)
def _warn_test(var, virt):
    warnings.warn(f"${var}={virt}: {'the forward / kinematics unit is' if var == TEST_FORWARD_VAR else 'every unit is'} "
                  f"built from {virt} PTX (a testing switch, not the default build)", RuntimeWarning, stacklevel=4)


def virtual_arch(unit=None, target=None, nvcc_path=None):
    """The virtual architecture whose PTX a compilation unit is built from, or None for the native build
    (-arch=<target>). unit 'forward': the forward / kinematics unit; anything else: the adjoint units, the FFI unit and
    test programs. sm_120 (see the module docstring): compute_90 for every unit with $DK_SM120_FRONTEND n7 (the
    default), for the forward unit only with split, none with native (n7 / split only where the PTX headers confirm
    it). The testing switches take precedence on any target: $DK_TEST_FORWARD_VIRTUAL_ARCH for the forward unit,
    $DK_TEST_VIRTUAL_ARCH for every unit."""
    target = target or arch()
    fwd = unit == "forward"
    for var, applies in ((TEST_FORWARD_VAR, fwd), (TEST_VAR, True)):
        test = (os.environ.get(var) or "").strip()
        if test and applies:
            _warn_test(var, test)
            return test
    if target not in N7_VIRTUAL:
        return None
    mode = frontend()
    if mode == "native" or (mode == "split" and not fwd):
        return None
    return _n7_applies(target, nvcc_path or nvcc())


def forward_virtual_arch(target=None, nvcc_path=None):
    """virtual_arch of the forward / kinematics unit (compute_90 on sm_120 with n7 or split)."""
    return virtual_arch("forward", target, nvcc_path)


def arch_flags(unit=None, target=None, nvcc_path=None):
    """nvcc's architecture flags for a compilation unit (unit 'forward': the forward / kinematics kernels; anything
    else: the adjoint units, the FFI unit, test programs): ['-arch=<virtual>', '-code=<target>'] where virtual_arch gives
    one (sm_120: every unit by default, see the module docstring), else ['-arch=<target>']."""
    target = target or arch()
    virt = virtual_arch(unit, target, nvcc_path)
    return [f"-arch={virt}", f"-code={target}"] if virt else [f"-arch={target}"]


def forward_tag(target=None, nvcc_path=None):
    """'' where the forward unit has the native flags, else 'fwd-<virtual architecture>' (names forward references:
    the forward's bits depend only on the forward unit's flags, so sm_120's n7 and split share 'fwd-compute_90')."""
    virt = forward_virtual_arch(target, nvcc_path)
    return f"fwd-{virt}" if virt else ""


def reg_caps(target=None):
    """The register-cap switch of the build target (REG_CAPS: $DK_SM90_REG_CAPS on sm_90 / sm_90a, $DK_SM120_PRE_CAP on
    sm_120): True for 1 (the default), False for 0; None for a target without caps."""
    target = target or arch()
    if target not in REG_CAPS:
        return None
    var = REG_CAPS[target][0]
    v = (os.environ.get(var) or "1").strip()
    if v not in ("0", "1"):
        raise ValueError(f"${var}={v!r}: must be 1 or 0")
    return v == "1"


def _switch(var, default):
    """$var as a 1 / 0 build switch (default '1' or '0'); module docstring."""
    v = (os.environ.get(var) or default).strip()
    if v not in ("0", "1"):
        raise ValueError(f"${var}={v!r}: must be 1 or 0")
    return v == "1"


def forward_split():
    """$DK_FORWARD_SPLIT: True (the default) builds the split forward (k_pre / k_post, host.DK_SPLIT 1), False the fused
    k_substep (host.DK_SPLIT 0): one launch per control step instead of 2 nsub, no struct Hand traffic and no
    DK_SPLIT_ORDER memory; the results, tape and contacts are bitwise the split forward's (host.py). A build option."""
    return _switch(FORWARD_SPLIT_VAR, "1")


def split_order():
    """$DK_SPLIT_ORDER: True (the default) dispatches k_post's worlds by constraint-row count (host.DK_SPLIT_ORDER 1),
    False in world order (host.DK_SPLIT_ORDER 0: no counter memset, no per-world bucket atomics, no warp scan per
    launch). Scheduling only (the results do not depend on the order). A build option; the split forward only."""
    return _switch(SPLIT_ORDER_VAR, "1")


def forward_build_defines():
    """The -D options of forward_split() and split_order() that differ from host.KERNEL's defaults: [] for the default
    build, -DDK_SPLIT=0 for the fused forward, -DDK_SPLIT_ORDER=0 for the world order."""
    return ([] if forward_split() else ["-DDK_SPLIT=0"]) + ([] if split_order() else ["-DDK_SPLIT_ORDER=0"])


def cap_overrides():
    """$DK_SPLIT_PRE_MINB / $DK_SPLIT_POST_MAXNREG / $DK_SPLIT_POST_MINB as {macro: value}: a forward-unit register cap
    in place of the build target's (reg_caps_defines; e.g. a hand model's NV 44 / 56 Cholesky keeps 3 NV floats per lane
    in chol_solve_reg_save, so the G1's 128-register k_post cap spills). $DK_SPLIT_POST_MINB replaces __maxnreg__ with
    the launch bound; a value of 0 bounds that kernel not at all. Unset: the target's caps (module docstring)."""
    out = {}
    for var in CAP_VARS:
        v = (os.environ.get(var) or "").strip()
        if v:
            if not v.isdigit():
                raise ValueError(f"${var}={v!r}: must be a non-negative integer")
            out[var[3:]] = v
    return out


def defined_macros(options):
    """The macro names an nvcc option list defines (-DX, -DX=Y, -D X, --define-macro X, --define-macro=X)."""
    out, it = set(), iter(str(a) for a in options)
    for a in it:
        if a in ("-D", "--define-macro"):
            a = "-D" + next(it, "")
        elif a.startswith("--define-macro="):
            a = "-D" + a.split("=", 1)[1]
        if a.startswith("-D") and len(a) > 2:
            out.add(a[2:].split("=", 1)[0])
    return out


def reg_caps_defines(options=(), caps=SM90_CAPS):
    """The caps (e.g. ('SPLIT_PRE_MINB=16', ...)) as -D options, less each cap whose kernel the caller's nvcc options
    already bound (defined_macros; CAP_BOUNDS), with cap_overrides()'s macros in place of the target's: an override of 0
    drops that macro, and $DK_SPLIT_POST_MINB replaces __maxnreg__ with its launch bound (host.py: MAXNREG > 0 wins over
    MINB)."""
    have = defined_macros(options)
    over = cap_overrides()
    chosen = []
    for d in caps:
        name = d.split("=", 1)[0]
        if name == "SPLIT_POST_MAXNREG" and "SPLIT_POST_MINB" in over:
            continue
        chosen.append(f"{name}={over[name]}" if name in over else d)
    if "SPLIT_POST_MINB" in over:
        chosen.append(f"SPLIT_POST_MINB={over['SPLIT_POST_MINB']}")
    # MAXNREG 0 is "undefined" (the launch bound applies); MINB 0 is a value of host.py's (no bounds at all)
    return [f"-D{d}" for d in chosen
            if not (d.startswith("SPLIT_POST_MAXNREG=") and d == "SPLIT_POST_MAXNREG=0")
            and not have.intersection(CAP_BOUNDS[d.split("=", 1)[0]])]


@functools.lru_cache(maxsize=None)
def _maxnreg_ok(nvcc_path, var):
    r = subprocess.run([nvcc_path, "--version"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                       universal_newlines=True)
    m = re.search(r"release (\d+)\.(\d+)", r.stdout)
    rel = (int(m.group(1)), int(m.group(2))) if m else None
    if rel and rel >= MAXNREG_NVCC:
        return True
    print(f"nablaengine.cuda: {var}=1 not applied (nvcc release {'.'.join(map(str, rel)) if rel else '?'}; "
          "__maxnreg__ needs 12.4 or newer): the forward unit keeps the launch bounds of the other targets",
          file=sys.stderr)
    return False


def unit_defines(unit=None, target=None, options=(), nvcc_path=None):
    """The -D options a compilation unit gets for its build target besides the caller's nvcc options: unit 'forward'
    (the split forward kernels' unit; see the module docstring) gets the target's register caps (REG_CAPS: sm_90 /
    sm_90a ['-DSPLIT_PRE_MINB=16', '-DSPLIT_POST_MAXNREG=128'], sm_120 with the NVVM 7 forward ['-DSPLIT_PRE_MINB=16'])
    where its switch is 1 (the default), each unless `options` already bound that kernel; every other unit and target:
    none."""
    if unit != "forward":
        return []
    target = target or arch()
    if not reg_caps(target):
        return []
    var, caps, needs_n7 = REG_CAPS[target]
    if needs_n7 and not forward_virtual_arch(target, nvcc_path):
        return []
    if any(c.startswith("SPLIT_POST_MAXNREG=") for c in caps) and not _maxnreg_ok(nvcc_path or nvcc(), var):
        return []
    return reg_caps_defines(options, caps)


def compile_command(src, out, extra=(), unit=None, defines=True, target=None, nvcc_path=None):
    """compile_cuda's command line for build target target (default arch()): [nvcc, -O3, <arch_flags(unit)>,
    -std=c++17, [extra], [unit_defines(unit)], -o, out, src] (also scripts/identity_gate.py's)."""
    extra = list(extra)
    return [nvcc_path or nvcc(), "-O3", *arch_flags(unit, target, nvcc_path), "-std=c++17", *extra,
            *(unit_defines(unit, target, extra, nvcc_path) if defines else []), "-o", str(out), str(src)]


def compile_cuda(src, out, extra=(), unit=None, defines=True):
    """nvcc -O3 <arch_flags(unit)> -std=c++17 [extra] [unit_defines(unit)] -o out src (unit 'forward': the forward /
    kinematics kernels' unit, which gets the target's register caps; on sm_120 every unit is built from compute_90 PTX
    by default; default unit: the adjoint units' / test programs' flags). defines=False: without unit_defines (the
    caller passes every -D option itself). The command is compile_command's."""
    subprocess.run(compile_command(src, out, extra, unit, defines), check=True)
