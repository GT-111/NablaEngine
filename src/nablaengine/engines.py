"""The engine generator for a model: nablaengine.g1_engine for the G1 class (body 1 a free root and every body in its
tree: legged robots). The interfaces (jax_ffi, torch_ffi) build from the modules modules(m) returns; the G1's text is
g1_engine's, unchanged (scripts/identity_gate.py)."""
from types import SimpleNamespace

import mujoco


def _outside(m):
    """True unless body 1 has a free joint and every other body descends from it (the G1 class)."""
    if m.nbody < 2 or m.body_jntnum[1] != 1 or m.jnt_type[m.body_jntadr[1]] != mujoco.mjtJoint.mjJNT_FREE:
        return True
    return any(int(m.body_rootid[b]) != 1 for b in range(1, m.nbody))


def modules(m):
    """SimpleNamespace(adjoint, host, kinematics, model_hooks, source, step, tape) of the model's engine; ValueError for
    a model outside the G1 class (several trees under the world, or no free root)."""
    if _outside(m):
        raise ValueError("the engine supports models whose body 1 has a free joint and is the root of every other body "
                         "(nablaengine.model.capability)")
    from nablaengine.g1_engine import adjoint, host, kinematics, model_hooks, source, step, tape
    return SimpleNamespace(adjoint=adjoint, host=host, kinematics=kinematics, model_hooks=model_hooks, source=source,
                           step=step, tape=tape)
