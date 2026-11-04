"""CPU checks of context routing; --cuda also checks real A/B/A kernels and graphs.

CPU checks deliberately replace launches: they establish binding ownership and
fake shapes, not CUDA numerical correctness. CUDA uses two real model artifacts.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from nablaengine import torch_ffi as F


@contextmanager
def contexts():
    previous = dict(F._CONTEXTS)
    ids = [max(previous, default=-1) + i + 1 for i in range(10)]
    values = []
    for index, cid in enumerate(ids):
        nq, nv, nu, nb, rec = 8+index, 7+index, 1+index, 3+index, 9+index*8
        context = F.ModelContext(cid)
        nr = 1+index%2
        context._state["info"] = dict(nq=nq, nv=nv, nu=nu, nbody=nb, nr=nr, com_width=3*nr, rec_words=rec)
        F._CONTEXTS[cid] = context
        values.append(context)
    try:
        yield values
    finally:
        F._CONTEXTS.clear()
        F._CONTEXTS.update(previous)


def inputs(context, device="cpu", requires_grad=False):
    i, B = context.info, 2
    values = [torch.zeros(shape, device=device) for shape in
              ((B, i["nq"]), (B, i["nv"]), (B, i["nv"]), (B, i["nu"]), (B,), (B, i["com_width"]),
               (B, i["nv"]), (B,))]
    for index in (0, 1, 3, 6):
        values[index].requires_grad_(requires_grad)
    return values


class ContextChecks(unittest.TestCase):
    def test_forward_skips_contact_output_and_backward_routes_force_diagnostics(self):
        from unittest.mock import Mock

        with contexts() as owners, patch.object(F, "_ready", return_value=0), \
                patch.object(F, "_check"), \
                patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=17)):
            owner = owners[0]
            owner.info["maxcon"] = 256
            native_forward, native_adjoint = Mock(return_value=0), Mock(return_value=0)
            owner._state["fns"] = dict(forward=native_forward, work_bytes=lambda _: 1,
                                        adjoint={(0, 3): native_adjoint}, adj_work_bytes={(0, 3): lambda _: 0})
            q, v, w, u, fr, com, force, kp = inputs(owner)
            owner._diagnostics_on = False
            result = F._forward_impl(q, v, w, u, fr, com, force, kp, 10, 4, True, False, owner.context_id)
            self.assertIsNone(native_forward.call_args.args[14])  # optional contact presentation output
            self.assertEqual(len(result), 5)
            self.assertEqual(result[4].shape, (2, 4 * owner.info["rec_words"]))
            owner._diagnostics_on = True
            for applied in (None, force):
                with patch.object(F, "_diag_backward") as diag:
                    grads = F._backward_impl(result[4], fr, com, q, v, None, [], [], 4, 0, 3, 0.,
                                             applied, kp, owner.context_id)
                self.assertIs(diag.call_args.args[8], None if applied is None else grads[4])

    def test_backward_workspace_uses_variant_bytes_and_is_not_retained(self):
        import weakref
        from unittest.mock import Mock

        with contexts() as owners, patch.object(F, "_ready", return_value=0), patch.object(F, "_check"), \
                patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=17)):
            for owner, variant, size in ((owners[0], (0, 3), 0), (owners[1], (2, 0), 2 * 12 * 2816)):
                owner._diagnostics_on = False
                q, v, _w, u, fr, com, force, kp = inputs(owner)
                tape = q.new_empty((2, 4 * owner.info["rec_words"]))
                allocations, empty = [], torch.empty

                def allocate(*args, **kwargs):
                    result = empty(*args, **kwargs)
                    if kwargs.get("dtype") == torch.uint8:
                        allocations.append(weakref.ref(result))
                    return result

                def launch(*args):
                    self.assertEqual(len(args), 18)
                    if size:
                        work = allocations[-1]()
                        self.assertEqual(work.shape, (size,))
                        self.assertEqual(args[-1].value, work.data_ptr())
                    else:
                        self.assertIsNone(args[-1])
                    return 0

                native, work_bytes = Mock(side_effect=launch), Mock(return_value=size)
                owner._state["fns"] = dict(adjoint={variant: native}, adj_work_bytes={variant: work_bytes})
                ctrl = u if variant[0] == 2 else None
                with patch.object(torch, "empty", side_effect=allocate):
                    grads = F._backward_impl(tape, fr, com, q, v, ctrl, [], [], 4, *variant, 0.,
                                             force, kp, owner.context_id)
                work_bytes.assert_called_once_with(2)
                self.assertEqual(len(grads), 5)
                self.assertEqual(len(allocations), int(size > 0))
                self.assertTrue(all(value() is None for value in allocations))
                native.reset_mock()
                work_bytes.reset_mock()
                F._backward_impl(tape[:0], fr[:0], com[:0], q[:0], v[:0], None if ctrl is None else ctrl[:0],
                                 [], [], 4, *variant, 0., force[:0], kp[:0], owner.context_id)
                native.assert_not_called()
                work_bytes.assert_not_called()

    @unittest.skipUnless(shutil.which("c++"), "adjoint workspace check requires a C++ compiler")
    def test_adjoint_workspace_export_sizes_native_and_primitive_models(self):
        from nablaengine import jax_ffi
        from nablaengine.g1_engine import adjoint

        helper = "static inline size_t adj_work_bytes(int nw)" + adjoint.ADJ_KERNEL.split(
            "static inline size_t adj_work_bytes(int nw)", 1)[1].split("\n}", 1)[0] + "\n}\n"
        export = next(line for line in jax_ffi.VARIANT_UNIT.splitlines() if "_adj_work_bytes(int nw)" in line)
        source = ("#include <cassert>\n#include <cstddef>\n#include <initializer_list>\n#include <type_traits>\n"
                  "struct alignas(8) CMReplayScratch { unsigned char bytes[2816]; };\n" + helper
                  + export.replace("@PFX@", "workspace") + r'''
static_assert(std::is_same_v<decltype(&workspace_adj_work_bytes), long long (*)(int)>);
int main() {
  for (int worlds : {0, 2, 410}) {
    long long expected = HAS_NATIVE_COLLISION ? (long long)worlds * 12 * sizeof(CMReplayScratch) : 0;
    assert(workspace_adj_work_bytes(worlds) == expected);
  }
}
''')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"workspace.cc").write_text(source)
            for native in (0, 1):
                subprocess.run(("c++", "-std=c++17", f"-DHAS_NATIVE_COLLISION={native}", str(path/"workspace.cc"),
                                "-o", str(path/"workspace")), check=True)
                subprocess.run((str(path/"workspace"),), check=True)

    @unittest.skipUnless(shutil.which("c++"), "contact output check requires a C++ compiler")
    def test_contact_output_is_optional_in_both_forward_kernels(self):
        from nablaengine.g1_engine.host import KERNEL

        # Execute both actual presentation writers; internal collision/row work is outside these fragments.
        fused = KERNEL.split("__global__ void k_substep(", 1)[1].split("// The split forward:", 1)[0]
        fused = fused[fused.index("  if (ocon)"):].rsplit("\n}", 1)[0]
        split = KERNEL.split("__device__ __forceinline__ void pre_world(", 1)[1].split("template <class S>", 1)[0]
        split = split[split.index("  if (ocon &&"):].split("#ifdef DK_PROFILE", 1)[0]
        source = r'''#include <algorithm>
#include <cassert>
#include <cstddef>
using std::min;
constexpr int MAXCON = 2;
struct Contacts {
  int ncon = 1, con_pair[2] = {7, 8};
  float con_dist[2] = {-.5f, .75f}, con_pos[2][3] = {{1, 2, 3}, {4, 5, 6}};
  float con_frame[2][3] = {{7, 8, 9}, {10, 11, 12}};
};
void fused(const Contacts& s, int lane, int w, float* ocon) {
''' + fused + r'''
}
void split(const Contacts& s, int lane, int w, float* ocon, int k, int nsub) {
  int ncon = min(s.ncon, MAXCON);
''' + split + r'''
}
int main() {
  Contacts s;
  const float expected[16] = {7, -.5f, 1, 2, 3, 7, 8, 9, -1, 0, 0, 0, 0, 0, 0, 0};
  for (bool use_split : {false, true}) {
    float out[32]; std::fill(out, out + 32, -99.f);
    for (int lane = 0; lane < 32; ++lane) {
      if (use_split) {
        split(s, lane, 1, nullptr, 3, 4);
        split(s, lane, 1, out, 2, 4);
      } else fused(s, lane, 1, nullptr);
    }
    for (float value : out) assert(value == -99.f);
    for (int lane = 0; lane < 32; ++lane) {
      if (use_split) split(s, lane, 1, out, 3, 4);
      else fused(s, lane, 1, out);
    }
    assert(std::equal(out + 16, out + 32, expected));
    for (int i = 0; i < 16; ++i) assert(out[i] == -99.f);
  }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"contacts.cc").write_text(source)
            subprocess.run(("c++", "-std=c++17", str(path/"contacts.cc"), "-o", str(path/"contacts")), check=True)
            subprocess.run((str(path/"contacts"),), check=True)

    @unittest.skipUnless(shutil.which("c++"), "diagnostic scan check requires a C++ compiler")
    def test_nonfinite_force_gradient_counts_once_without_changing_legacy_bits(self):
        from nablaengine.g1_engine.adjoint import ADJ_REPLAY_MISMATCH

        finite = F.TORCH_UNIT.split("__device__ __forceinline__ bool all_finite", 1)[1].split("__global__", 1)[0]
        backward = F.TORCH_UNIT.split("__global__ void k_tdiag_bwd", 1)[1].split("unsigned diag_grid", 1)[0]
        defines = [(f"D_{name.upper()}", i) for i, name in enumerate(F.DIAG)]
        defines += [(name, getattr(F, name)) for name in
                    ("NONFINITE_QPOS", "NONFINITE_QVEL", "NONFINITE_CTRL", "NONFINITE_COT", "NONFINITE_SHIFT")]
        defines += [("ADJ_REPLAY_MISMATCH", ADJ_REPLAY_MISMATCH)]
        source = "\n".join(f"#define {name} {value}" for name, value in defines) + r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
using std::isfinite;
constexpr int NQ = 2, NV = 3, NU = 1;
struct { int x; } blockIdx{0}, threadIdx{0}, blockDim{1}, gridDim{1};
void diag_add(unsigned long long* d, int k, unsigned value) { d[k] += value; }
void atomicAdd(unsigned long long* d, unsigned long long value) { *d += value; }
bool all_finite''' + finite + "void k_tdiag_bwd" + backward + r'''
int main() {
  float gq[6]{}, gv[9]{}, gu[3]{}, cq[6]{}, cv[9]{}, gf[9]{}, finite_force[9]{};
  gf[2] = std::numeric_limits<float>::quiet_NaN();
  gf[5] = std::numeric_limits<float>::infinity();
  gf[8] = gu[2] = std::numeric_limits<float>::quiet_NaN();
  const float* forces[] = {gf, finite_force, nullptr};
  for (const float* force : forces) {
    unsigned long long d[18]{};
    int info[3] = {ADJ_REPLAY_MISMATCH, 0, 0};
    k_tdiag_bwd(d, 3, gq, gv, gu, cq, cv, force, info);
    assert(d[D_NONFINITE_ROWS] == (force == gf ? 3 : 1));
    assert(d[D_NONFINITE_CTRL_ROWS] == 1);
    assert(d[D_NONFINITE_QPOS_ROWS] == 0 && d[D_NONFINITE_QVEL_ROWS] == 0);
    assert(d[D_NONFINITE_COTANGENT_ROWS] == 0 && d[D_REPLAY_MISMATCH_WORLDS] == 1);
    assert(d[D_BWD_CALLS] == 1 && d[D_BWD_ROWS] == 3);
    assert(info[0] == ADJ_REPLAY_MISMATCH && info[1] == 0);
    assert(info[2] == (NONFINITE_CTRL << NONFINITE_SHIFT));
  }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"diagnostics.cc").write_text(source)
            subprocess.run(("c++", "-std=c++17", str(path/"diagnostics.cc"), "-o", str(path/"diagnostics")), check=True)
            subprocess.run((str(path/"diagnostics"),), check=True)

    def test_old_tape_artifact_is_refused_before_loading_its_library(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model.so").touch()
            (path / "model.npy").touch()
            artifact = path / "model.json"
            artifact.write_text(json.dumps(dict(artifact_version=3, arch="sm_90",
                library="model.so", consts="model.npy")))
            self.assertEqual(F._stale(artifact, "sm_90"),
                f"artifact version 3, this module reads {F.ARTIFACT_VERSION}")

    @unittest.skipUnless(shutil.which("c++"), "host loader source check requires a C++ compiler")
    def test_uint64_constant_loader_retains_high_body_bits(self):
        from nablaengine.g1_engine.host import loader

        statements = loader((("mask", "q", "nb", 1),), {"nb": 2})
        source = """#include <cassert>
#include <cstring>
unsigned long long mask[2];
#define cudaMemcpyToSymbol(symbol, data, bytes) std::memcpy(symbol, data, bytes)
int main() {
  const float words[] = {2, 0, 16384, 0, 65535, 65535, 65535, 32768};
  const float* p = words;
""" + statements + """
  assert(mask[0] == ((1ull << 46) | 2ull));
  assert(mask[1] == 0x8000ffffffffffffull);
  assert(p == words + 8);
}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"loader.cc").write_text(source)
            subprocess.run(("c++", str(path/"loader.cc"), "-o", str(path/"loader")), check=True)
            subprocess.run((str(path/"loader"),), check=True)

    def test_diagnostics_are_context_owned_and_default_switch_stays_lazy(self):
        with patch.object(F, "_DEFAULT_CONTEXT", None), patch.object(F, "_DEFAULT_DIAGNOSTICS_ON", True), \
                patch.object(F, "init", side_effect=AssertionError("must stay lazy")), contexts() as owners:
            self.assertTrue(F.set_diagnostics(False))
            self.assertFalse(F.set_diagnostics(True))
            self.assertTrue(owners[0].set_diagnostics(False))
            self.assertFalse(owners[0]._diagnostics_on)
            self.assertTrue(owners[1]._diagnostics_on)

    def test_fake_shapes_and_tape_bytes_belong_to_context(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        ops = F._custom_ops()
        with contexts() as owners, FakeTensorMode():
            for context in (*owners, owners[0]):
                q, v, w, u, fr, com, force, kp = inputs(context)
                out = ops["forward"](q, v, w, u, fr, com, force, kp, 10, 4, True, True, context.context_id)
                self.assertEqual(out[4].shape, (2, 4 * context.info["rec_words"]))
                self.assertEqual(out[6].shape, (2, context.info["nbody"], 3))
                self.assertEqual(context.tape_bytes(2), 4 * 2 * 4 * context.info["rec_words"])
                grads = ops["backward"](out[4], fr, com, q, v, None, [], [], 4, 0, 3, 0.,
                                        force, kp, context.context_id)
                self.assertEqual(grads[2].shape, u.shape)
                self.assertEqual(grads[4].shape, force.shape)

    def test_delayed_backward_retains_context_and_force_gradient(self):
        launches = []

        def forward(q, v, w, u, fr, com, force, kp, iterations, substeps, tape, kin, cid):
            gain = float(cid + 2)
            outq = q * gain
            outv = (v + u.sum(-1, keepdim=True) + force) * gain
            tp = q.new_zeros((q.shape[0], substeps * F._sizes(cid)[4]))
            return [outq, outv, w.clone(), q.new_zeros((q.shape[0], 4), dtype=torch.int32), tp]

        def backward(tp, fr, com, gq, gv, ctrl, ks, kc, substeps, kv, normal, soft, force, kp, cid):
            launches.append(cid)
            gain = float(cid + 2)
            gu = gv.sum(-1, keepdim=True).expand(-1, F._sizes(cid)[2]) * gain
            return [gq * gain, gv * gain, gu, tp.new_zeros((tp.shape[0],), dtype=torch.int32), gv * gain]

        F._custom_ops()  # real init also registers before any compilation
        with contexts() as owners, patch.dict(F._IMPL, forward=forward, backward=backward), \
                patch.object(F, "_forward_impl", forward), patch.object(F, "_backward_impl", backward):
            for compiled in (False, True):
                launches.clear()
                batches, results = [], []
                for owner in (owners[0], owners[1], owners[0]):
                    batch = inputs(owner, requires_grad=True)
                    batches.append((owner, batch))
                    program = torch.compile(owner.step, backend="aot_eager", fullgraph=True) if compiled else owner.step
                    results.append(program(*batch))
                for result in reversed(results):
                    (result[0].sum() + result[1].sum()).backward()
                self.assertEqual(launches, [owners[0].context_id, owners[1].context_id, owners[0].context_id])
                for owner, batch in batches:
                    gain = float(owner.context_id + 2)
                    self.assertTrue(torch.equal(batch[0].grad, torch.full_like(batch[0], gain)))
                    self.assertTrue(torch.equal(batch[6].grad, torch.full_like(batch[6], gain)))
                    self.assertTrue(torch.equal(batch[3].grad, torch.full_like(batch[3], gain * owner.info["nv"])))

    def test_optional_info_preserves_state_and_kinematic_vjps_in_eager_and_aot(self):
        def forward(q, v, w, u, fr, com, force, kp, iterations, substeps, tape, kin, cid):
            info = torch.tensor((3, 127, 576, F.OVERFLOW_ROWS), dtype=torch.int32).expand(q.shape[0], -1).clone()
            tp = q.new_zeros((q.shape[0], substeps * F._sizes(cid)[4]))
            result = [2*q, v+u.sum(-1, keepdim=True)+force, w.clone(), info, tp]
            return result + [2*q, *(q.new_zeros(shape) for shape in F._kin_shapes(q.shape[0], cid)[1:])] if kin else result

        def backward(tp, fr, com, gq, gv, ctrl, ks, kc, substeps, kv, normal, soft, force, kp, cid):
            qbar = 2*gq + (2*kc[0] if kc else 0)
            ubar = gv.sum(-1, keepdim=True).expand(-1, F._sizes(cid)[2]).clone()
            return [qbar, gv.clone(), ubar, tp.new_zeros((tp.shape[0],), dtype=torch.int32), gv.clone()]

        F._custom_ops()
        with contexts() as owners, patch.dict(F._IMPL, forward=forward, backward=backward), \
                patch.object(F, "_forward_impl", forward), patch.object(F, "_backward_impl", backward):
            owner = owners[1]  # two FREE roots, six COM operands
            for compiled in (False, True):
                for kin in (False, True):
                    for return_info in (False, True):
                        def program(*args):
                            return owner.step(*args, kinematics=kin, return_info=return_info)
                        run = torch.compile(program, backend="aot_eager", fullgraph=True) if compiled else program
                        batch = inputs(owner, requires_grad=True)
                        output = run(*batch)
                        self.assertEqual(len(output), 3+int(return_info)+5*int(kin))
                        if return_info:
                            self.assertEqual(output[3].dtype, torch.int32)
                            self.assertFalse(output[3].requires_grad)
                            self.assertTrue(torch.equal(output[3], torch.tensor((3, 127, 576, F.OVERFLOW_ROWS)).expand(2, -1)))
                        loss = output[0].sum()+output[1].sum()
                        if kin:
                            loss = loss+output[3+int(return_info)].sum()
                        grads = torch.autograd.grad(loss, [batch[i] for i in (0, 1, 3, 6)])
                        for observed, expected in zip(grads, (2+2*int(kin), 1, owner.info["nv"], 1), strict=True):
                            self.assertTrue(torch.equal(observed, torch.full_like(observed, expected)))
                        with torch.no_grad():
                            plain = run(*batch)
                        self.assertTrue(all(torch.equal(left, right) for left, right in zip(output, plain, strict=True)))


def cuda_check(artifacts):
    """Real isolation/late VJP and eager/compiled/CUDA-graph equality on one GPU."""
    if len(artifacts) < 2:
        raise ValueError("CUDA context isolation needs at least two actual model artifacts")
    owners = [F.create_context(artifact=path) for path in artifacts]
    batches = []
    for owner in owners:
        batch = inputs(owner, device="cuda", requires_grad=True)
        with torch.no_grad():
            batch[0].copy_(torch.tensor(owner.info["qpos0"], dtype=torch.float32, device="cuda"))
        batch[4].fill_(1.)
        batch[7].fill_(1.)
        batches.append(batch)
    references = []
    for owner, batch in zip(owners, batches):
        result = owner.step(*batch)
        grad = torch.autograd.grad(result[1].square().sum(), [batch[i] for i in (0, 1, 3, 6)])
        references.append((result, grad))
    # Keep every tape alive across every other model, then A again.
    delayed = [owner.step(*batch) for owner, batch in zip(owners, batches)]
    owners[0].step(*batches[0])
    for result, batch, (expected, grad) in zip(reversed(delayed), reversed(batches), reversed(references)):
        observed = torch.autograd.grad(result[1].square().sum(), [batch[i] for i in (0, 1, 3, 6)])
        assert all(torch.equal(x, y) for x, y in zip(expected, result))
        assert all(torch.equal(x, y) for x, y in zip(grad, observed))
    graphs = []
    for owner, batch in zip(owners, batches):
        def run(*args):
            return owner.step(*args)
        expected = run(*batch)
        compiled = torch.compile(run, fullgraph=True)(*batch)
        assert all(torch.equal(x, y) for x, y in zip(expected, compiled))
        eager_grad = torch.autograd.grad(expected[1].square().sum(), [batch[i] for i in (0, 1, 3, 6)])
        compiled_grad = torch.autograd.grad(compiled[1].square().sum(), [batch[i] for i in (0, 1, 3, 6)])
        assert all(torch.equal(x, y) for x, y in zip(eager_grad, compiled_grad))
        static = [value.detach().clone() for value in batch]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                owner.step(*static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = owner.step(*static)
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(x, y) for x, y in zip(expected, captured))
        graphs.append((graph, captured, tuple(value.detach().clone() for value in expected)))
    for graph, captured, expected in (*graphs, graphs[0]):
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(x, y) for x, y in zip(expected, captured))
    print(f"CUDA {len(owners)} contexts, delayed backward, compile and interleaved graphs: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", nargs="+", metavar="ARTIFACT")
    args, remaining = parser.parse_known_args()
    if args.cuda:
        cuda_check(args.cuda)
    else:
        unittest.main(argv=[__file__, *remaining])
