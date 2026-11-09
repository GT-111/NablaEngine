"""The native generator for one or two FREE trees, selected by the FFI interfaces.

The capability checker owns the supported actuator, joint and contact chart.
"""
from types import SimpleNamespace

import mujoco


def _outside(m):
    """One or two FREE trees; the capability checker validates their native charts."""
    roots = [body for body in range(1, m.nbody) if m.body_parentid[body] == 0]
    return not 1 <= len(roots) <= 2 or any(
        m.body_jntnum[body] != 1
        or m.jnt_type[m.body_jntadr[body]] != mujoco.mjtJoint.mjJNT_FREE for body in roots)


def modules(m):
    """SimpleNamespace(adjoint, host, kinematics, model_hooks, source, step, tape) of the model's engine; ValueError for
    a model outside the supported one- or two-FREE-tree chart."""
    if _outside(m):
        raise ValueError("the engine supports one or two native FREE trees "
                         "(nablaengine.model.capability)")
    from nablaengine.g1_engine import adjoint, host, kinematics, model_hooks, source, step, tape
    return SimpleNamespace(adjoint=adjoint, host=host, kinematics=kinematics, model_hooks=model_hooks, source=source,
                           step=step, tape=tape)
