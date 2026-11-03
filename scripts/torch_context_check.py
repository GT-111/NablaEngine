"""CPU checks of context routing; --cuda also checks real A/B/A kernels and graphs.

CPU checks deliberately replace launches: they establish binding ownership and
fake shapes, not CUDA numerical correctness. CUDA uses two real model artifacts.
"""

import argparse
from contextlib import contextmanager
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
        context._state["info"] = dict(nq=nq, nv=nv, nu=nu, nbody=nb, rec_words=rec)
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
              ((B, i["nq"]), (B, i["nv"]), (B, i["nv"]), (B, i["nu"]), (B,), (B, 3),
               (B, i["nv"]), (B,))]
    for index in (0, 1, 3, 6):
        values[index].requires_grad_(requires_grad)
    return values


class ContextChecks(unittest.TestCase):
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

        with contexts() as owners, patch.dict(F._IMPL, forward=forward, backward=backward):
            batches, results = [], []
            for owner in (owners[0], owners[1], owners[0]):
                batch = inputs(owner, requires_grad=True)
                batches.append((owner, batch))
                results.append(owner.step(*batch))
            for result in reversed(results):
                (result[0].sum() + result[1].sum()).backward()
            self.assertEqual(launches, [owners[0].context_id, owners[1].context_id, owners[0].context_id])
            for owner, batch in batches:
                gain = float(owner.context_id + 2)
                self.assertTrue(torch.equal(batch[0].grad, torch.full_like(batch[0], gain)))
                self.assertTrue(torch.equal(batch[6].grad, torch.full_like(batch[6], gain)))
                self.assertTrue(torch.equal(batch[3].grad, torch.full_like(batch[3], gain * owner.info["nv"])))


def cuda_check(artifacts):
    """Real isolation/late VJP and eager/compiled/CUDA-graph equality on one GPU."""
    if len(artifacts) < 2:
        raise ValueError("CUDA context isolation needs at least two actual model artifacts")
    owners = [F.create_context(artifact=path) for path in artifacts]
    batches = []
    for owner in owners:
        batch = inputs(owner, device="cuda", requires_grad=True)
        batch[0].data[:, 3] = 1  # each supported model has one FREE quaternion
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
