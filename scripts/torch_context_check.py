"""CPU checks of context routing; --cuda also checks real A/B/A kernels and graphs.

CPU checks deliberately replace launches: they establish binding ownership and
fake shapes, not CUDA numerical correctness. CUDA uses two real model artifacts.
"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

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
        context._state["info"] = dict(key=f"context-check-model-{index}", nq=nq, nv=nv, nu=nu,
            nbody=nb, nr=nr, com_width=3*nr, rec_words=rec)
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
                out = ops["forward"](q, v, w, u, fr, com, force, kp, 10, 4, True, True,
                    context.context_id, context.info["key"])
                self.assertEqual(out[4].shape, (2, 4 * context.info["rec_words"]))
                self.assertEqual(out[6].shape, (2, context.info["nbody"], 3))
                self.assertEqual(context.tape_bytes(2), 4 * 2 * 4 * context.info["rec_words"])
                grads = ops["backward"](out[4], fr, com, q, v, None, [], [], 4, 0, 3, 0.,
                                        force, kp, context.context_id, context.info["key"])
                self.assertEqual(grads[2].shape, u.shape)
                self.assertEqual(grads[4].shape, force.shape)

    def test_inductor_cache_distinguishes_models_reusing_a_context_id(self):
        # The same context integer and operand shapes can describe a different
        # model in the next process. Both processes must share the same cache.
        code = """
import sys
import torch
from nablaengine import torch_ffi as F

owner = F.ModelContext(0)
owner._state['info'] = dict(key=sys.argv[1], nq=46, nv=44, nu=32, nbody=35,
    nr=2, com_width=6, rec_words=int(sys.argv[2]))
F._CONTEXTS[0] = owner

def native(q, v, w, u, fr, com, force, kp, iterations, substeps, tape, kin, cid):
    rec = F._CONTEXTS[cid].info['rec_words']
    return [q.clone(), v.clone(), w.clone(), torch.zeros(q.shape[0], 4, dtype=torch.int32),
        torch.zeros(q.shape[0], substeps*rec)]

F._forward_impl = native
F._custom_ops()

def program(q, v, w, u, fr, com):
    return owner.forward(q, v, w, u, fr, com, tape=True)[4]

batch = [torch.zeros(shape) for shape in ((2,46), (2,44), (2,44), (2,32), (2,), (2,6))]
result = torch.compile(program, fullgraph=True)(*batch)
assert result.shape == (2, 4*owner.info['rec_words']), result.shape
print(tuple(result.shape))
"""
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TORCHINDUCTOR_CACHE_DIR=directory,
                TORCHINDUCTOR_FX_GRAPH_CACHE="1", TORCHINDUCTOR_AUTOGRAD_CACHE="1")
            env["PYTHONPATH"] = str(Path(F.__file__).resolve().parents[1])+os.pathsep+env.get("PYTHONPATH", "")
            for key, rec in (("mesh-model", 1352), ("capsule-model", 584)):
                result = subprocess.run((sys.executable, "-c", code, key, str(rec)),
                    env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertEqual(result.stdout.strip(), str((2, 4*rec)))

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
        with contexts() as owners, patch.object(F, "_forward_impl", forward), \
                patch.object(F, "_backward_impl", backward):
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
        with contexts() as owners, patch.object(F, "_forward_impl", forward), \
                patch.object(F, "_backward_impl", backward):
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
