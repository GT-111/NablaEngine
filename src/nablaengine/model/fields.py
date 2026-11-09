"""Every field of MuJoCo's MjModel (mujoco 3.17), classified by what the engine does with it: the completeness proof
behind the model validation (nablaengine.model.capability).

Kinds:
  consumed    read by the engine's code generation, constants or interface (nablaengine.g1_engine, jax_ffi); its
              value is used, under the checks of the code given (e.g. jnt_margin: H20 requires 0)
  checked     the engine does not model it; the capability check of the code given rejects the values it cannot
              represent (e.g. opt.integrator: H23, implicitfast only)
  default     the engine does not model it and has no check of its own: every row must equal MuJoCo's compile default
              (default_rows(), from a probe model compiled by this MuJoCo), else an error under the code given. This is
              where a physics field that a MuJoCo release adds lands first (actuator_damping, *_dampingpoly,
              actuator_delay of 3.17), so a model that sets it is rejected instead of silently simulated without it
  derived     computed by MuJoCo's compiler from fields classified elsewhere (addresses, sparsity, bounding volumes,
              qpos0 inertias): no independent information
  irrelevant  no effect on the engine's outputs (rendering, names, sensors, keyframes, user data, solver settings of
              MuJoCo's own paths the engine does not take); the note says why

classify(name) gives the Field of a name ("opt.<x>", "stat.<x>" for the option and statistics structs, "vis" for the
whole visual struct); fields of element types the engine rejects as a whole (flexes, tendons, equality constraints,
plugins: H25 checks their counts) and of purely visual / bookkeeping element types (cameras, lights, materials,
textures, meshes, skins, height fields, sensors, sites, keyframes, names, custom data) are classified by prefix
(PREFIXES). model_fields(m) lists the names of an MjModel; unclassified(m) those classify() does not know: a test
(scripts/capability_check.py, section fields) and the checker's code F0 fail on any, so a MuJoCo upgrade that adds a
field forces a decision.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

KINDS = ("consumed", "checked", "default", "derived", "irrelevant")


@dataclass(frozen=True)
class Field:
    kind: str
    code: str = None
    note: str = ""


def _f(kind, code=None, note=""):
    assert kind in KINDS, kind
    return Field(kind, code, note)


VISUAL = "rendering only"
USER = "user data, not read"

FIELDS = {
    # ---- generalized coordinates
    "qpos0": _f("consumed", "H8", "the hinges' must be 0 (the engine's angle is qpos)"),
    "qpos_spring": _f("irrelevant", "H22", "spring reference poses; every stiffness is default (0, H22)"),
    # ---- bodies
    "body_parentid": _f("consumed", "H9"), "body_rootid": _f("consumed", "H6"),
    "body_weldid": _f("derived", "H7"), "body_mocapid": _f("derived", "H25"),
    "body_jntnum": _f("consumed", "H7"), "body_jntadr": _f("consumed", "H7"),
    "body_dofnum": _f("consumed", "H7b"), "body_dofadr": _f("consumed", "H7b"),
    "body_treeid": _f("derived"), "body_geomnum": _f("derived"), "body_geomadr": _f("derived"),
    "body_simple": _f("derived"), "body_sameframe": _f("derived"),
    "body_pos": _f("consumed"), "body_quat": _f("consumed"), "body_ipos": _f("consumed"),
    "body_iquat": _f("consumed"), "body_mass": _f("consumed"), "body_inertia": _f("consumed"),
    "body_subtreemass": _f("derived"), "body_invweight0": _f("consumed"),
    "body_gravcomp": _f("default", "H22", "gravity compensation"),
    "body_margin": _f("derived"), "body_user": _f("irrelevant", note=USER),
    "body_plugin": _f("default", "H25", "body plugins"),
    "body_contype": _f("derived", "H15"), "body_conaffinity": _f("derived", "H15"),
    "body_bvhadr": _f("derived"), "body_bvhnum": _f("derived"),
    # ---- joints
    "jnt_type": _f("consumed", "H7"), "jnt_qposadr": _f("consumed"), "jnt_dofadr": _f("consumed", "H11a"),
    "jnt_bodyid": _f("derived"), "jnt_actuatorid": _f("derived", "H11d", "the actuator whose damping / armature acts "
                                                      "on the joint (both default-checked)"),
    "jnt_group": _f("irrelevant", note=VISUAL), "jnt_limited": _f("consumed", "H20"),
    "jnt_actfrclimited": _f("consumed", "H12"), "jnt_actfrcrange": _f("consumed", "H12"),
    "jnt_actgravcomp": _f("default", "H22", "gravity compensation through the actuators"),
    "jnt_solref": _f("consumed", "H19"), "jnt_solimp": _f("consumed", "H20"), "jnt_pos": _f("consumed", "H7"),
    "jnt_axis": _f("consumed"), "jnt_stiffness": _f("default", "H22", "joint springs"),
    "jnt_stiffnesspoly": _f("default", "H22", "joint springs (higher order)"), "jnt_range": _f("consumed"),
    "jnt_margin": _f("consumed", "H20"), "jnt_user": _f("irrelevant", note=USER),
    # ---- dofs
    "dof_bodyid": _f("consumed", "H7b"), "dof_jntid": _f("derived"), "dof_parentid": _f("derived"),
    "dof_treeid": _f("derived"), "dof_Madr": _f("derived"), "dof_simplenum": _f("derived"),
    "dof_solref": _f("consumed", "H19"), "dof_solimp": _f("consumed", "H20"),
    "dof_frictionloss": _f("consumed", "H21"), "dof_armature": _f("consumed"),
    "dof_damping": _f("consumed", "H22", "linear joint damping"),
    "dof_dampingpoly": _f("default", "H22", "joint damping (higher order)"),
    "dof_invweight0": _f("consumed"), "dof_M0": _f("derived"), "dof_length": _f("derived"),
    # ---- kinematic trees
    "tree_bodyadr": _f("derived", "H6"), "tree_bodynum": _f("derived", "H6"), "tree_dofadr": _f("derived"),
    "tree_dofnum": _f("derived"),
    "tree_sleep_policy": _f("irrelevant", "H24b", "island sleeping (mjENBL_SLEEP is rejected, H24b)"),
    # ---- geoms
    "geom_type": _f("consumed", "H14a"), "geom_contype": _f("consumed", "H15"),
    "geom_conaffinity": _f("consumed", "H15"), "geom_condim": _f("consumed", "H17"),
    "geom_bodyid": _f("consumed"), "geom_dataid": _f("derived", "H14a", "mesh / height field of a geom (the "
                                                     "collision geoms' types are checked)"),
    "geom_matid": _f("irrelevant", note=VISUAL), "geom_group": _f("irrelevant", note=VISUAL),
    "geom_priority": _f("consumed", "H15"), "geom_plugin": _f("default", "H25", "geom plugins (SDF)"),
    "geom_sameframe": _f("derived"), "geom_solmix": _f("consumed", "H15"),
    "geom_solref": _f("consumed", "H19"), "geom_solimp": _f("consumed", "H20"),
    "geom_size": _f("consumed"), "geom_aabb": _f("derived"), "geom_rbound": _f("consumed"),
    "geom_pos": _f("consumed"), "geom_quat": _f("consumed"), "geom_friction": _f("consumed", "H16"),
    "geom_margin": _f("consumed", "H16"), "geom_gap": _f("consumed", "H16"),
    "geom_surfacevel": _f("default", "H16", "contact surface velocity"),
    "geom_adhesion": _f("default", "H16", "contact adhesion"),
    "geom_fluid": _f("default", "H22", "the ellipsoid fluid model"), "geom_user": _f("irrelevant", note=USER),
    "geom_rgba": _f("irrelevant", note=VISUAL),
    # ---- explicit contact pairs
    "pair_dim": _f("consumed", "H17"), "pair_geom1": _f("consumed", "H14a"), "pair_geom2": _f("consumed", "H14a"),
    "pair_signature": _f("derived"), "pair_solref": _f("consumed", "H19"),
    "pair_solreffriction": _f("default", "H16", "a separate solref of the friction rows"),
    "pair_solimp": _f("consumed"), "pair_margin": _f("consumed", "H16"), "pair_gap": _f("consumed", "H16"),
    "pair_adhesion": _f("default", "H16", "contact adhesion"),
    "pair_friction": _f("consumed", "H16", "tangential friction; spin and rolling unused at condim 1 / 3 (H17)"),
    "exclude_signature": _f("consumed", "H15", "excluded body pairs of contype / conaffinity contacts"),
    # ---- actuators
    "actuator_trntype": _f("consumed", "H11a"), "actuator_dyntype": _f("checked", "H11b"),
    "actuator_gaintype": _f("checked", "H11b"), "actuator_biastype": _f("consumed", "H11a"),
    "actuator_ctrladr": _f("checked", "H11a"), "actuator_ctrlnum": _f("checked", "H11a"),
    "actuator_ctrlspec": _f("checked", "H11b", "the input signature: 0 (the gain type's default) or mjINPUT_POS "
                                              "(the position shortcut's) for a fixed gain"),
    "actuator_outadr": _f("checked", "H11a"), "actuator_outnum": _f("checked", "H11a"),
    "actuator_actadr": _f("derived", "H11b"), "actuator_actnum": _f("derived", "H11b"),
    "actuator_trnid": _f("consumed", "H11a"),
    "actuator_cranklength": _f("irrelevant", "H11a", "slider-crank transmissions only (joint transmissions, H11a)"),
    "actuator_dynprm": _f("irrelevant", "H11b", "activation dynamics only (dyntype none, H11b)"),
    "actuator_gainprm": _f("consumed", "H11a"), "actuator_biasprm": _f("consumed", "H11a"),
    "actuator_actlimited": _f("irrelevant", "H11b", "activations only (none, H11b)"),
    "actuator_actrange": _f("irrelevant", "H11b", "activations only (none, H11b)"),
    "actuator_actearly": _f("irrelevant", "H11b", "activations only (none, H11b)"),
    "actuator_history": _f("default", "H11e", "the actuator's history buffer (delays)"),
    "actuator_historyadr": _f("derived", "H11e"),
    "actuator_delay": _f("default", "H11e", "actuator delay"),
    "actuator_damping": _f("default", "H22", "actuator damping (added to its joint's)"),
    "actuator_dampingpoly": _f("default", "H22", "actuator damping (higher order)"),
    "actuator_armature": _f("default", "H11d", "actuator armature (added to its joint's)"),
    "actuator_group": _f("irrelevant", "H24b", "visualization groups; disabling groups (opt.disableactuator) is "
                                                "rejected by H24b"),
    "actuator_user": _f("irrelevant", note=USER), "actuator_plugin": _f("default", "H25", "actuator plugins"),
    "actuator_forcelimited": _f("consumed", "H11a"),
    "actuator_forcerange": _f("consumed", "H12", "symmetric force limit combined with the joint force limit"),
    "actuator_ctrllimited": _f("checked", "H11c"), "actuator_ctrlrange": _f("checked", "H11c"),
    "actuator_gear": _f("consumed", "H11a"), "actuator_acc0": _f("derived"), "actuator_length0": _f("derived"),
    "actuator_lengthrange": _f("derived", note="computed for muscles; no muscles (H11b)"),
    # ---- flags (derived from the arrays above)
    "flg_gravcomp": _f("derived", "H22"), "flg_surfacevel": _f("derived", "H16"), "flg_adhesion": _f("derived", "H16"),
    # ---- names, paths, signatures
    "names": _f("consumed", "H13", "the COM-offset body's name lookup"),
    "names_map": _f("derived"), "paths": _f("irrelevant", note="asset paths"),
    "signature": _f("irrelevant", note="the compiler's signature"),
    # ---- sparse structures (derived from the tree)
    "B_rownnz": _f("derived"), "B_rowadr": _f("derived"), "B_colind": _f("derived"),
    "M_rownnz": _f("derived"), "M_rowadr": _f("derived"), "M_colind": _f("derived"),
    "D_rownnz": _f("derived"), "D_rowadr": _f("derived"), "D_diag": _f("derived"), "D_colind": _f("derived"),
    "mapM2M": _f("derived"), "mapM2D": _f("derived"), "mapD2M": _f("derived"),
    "plugin": _f("checked", "H25", "plugin instances (nplugin = 0)"),
    # ---- options (mjOption)
    "opt.timestep": _f("consumed", "I1", "baked into the constants"),
    "opt.impratio": _f("consumed"), "opt.tolerance": _f("consumed", note="the Newton tolerance (scaled by "
                                                                          "stat.meaninertia, as MuJoCo)"),
    "opt.ls_tolerance": _f("irrelevant", note="the engine's line search is exact (a converged Newton solve gives "
                                               "MuJoCo's unique minimizer)"),
    "opt.noslip_tolerance": _f("irrelevant", "H24b", "noslip is off (H24b)"),
    "opt.ccd_tolerance": _f("checked", "H14d", "native CCD: box-box only (MuJoCo Warp's GJK / EPA, whose converged "
                                               "result the forward computes): at most the default"),
    "opt.sleep_tolerance": _f("irrelevant", "H24b", "island sleeping is off (H24b)"),
    "opt.gravity": _f("consumed"), "opt.wind": _f("checked", "H22"),
    "opt.magnetic": _f("irrelevant", note="magnetometer sensors only"),
    "opt.density": _f("checked", "H22"), "opt.viscosity": _f("checked", "H22"),
    "opt.o_margin": _f("irrelevant", "H24b", "contact override: mjENBL_OVERRIDE is rejected (H24b)"),
    "opt.o_solref": _f("irrelevant", "H24b", "contact override: mjENBL_OVERRIDE is rejected (H24b)"),
    "opt.o_solimp": _f("irrelevant", "H24b", "contact override: mjENBL_OVERRIDE is rejected (H24b)"),
    "opt.o_friction": _f("irrelevant", "H24b", "contact override: mjENBL_OVERRIDE is rejected (H24b)"),
    "opt.integrator": _f("checked", "H23"), "opt.cone": _f("checked", "H18"),
    "opt.jacobian": _f("irrelevant", note="MuJoCo's dense / sparse representation"),
    "opt.solver": _f("checked", "H24a"),
    "opt.iterations": _f("consumed", note="the default of EnginePhysics' iterations (a call option)"),
    "opt.ls_iterations": _f("irrelevant", note="the engine's line search is exact (at most 40 steps)"),
    "opt.noslip_iterations": _f("checked", "H24b"),
    "opt.ccd_iterations": _f("checked", "H14d", "native CCD: box-box only (MuJoCo Warp's GJK / EPA, whose converged "
                                                "result the forward computes): at least the default"),
    "opt.disableflags": _f("checked", "H24b"), "opt.enableflags": _f("checked", "H24b"),
    "opt.disableactuator": _f("checked", "H24b"),
    "opt.sdf_initpoints": _f("irrelevant", "H14a", "SDF collisions: none"),
    "opt.sdf_iterations": _f("irrelevant", "H14a", "SDF collisions: none"),
    # ---- statistics (mjStatistic) and visual options
    "stat.meaninertia": _f("consumed", note="scales the Newton tolerance"),
    "stat.meanmass": _f("irrelevant", note=VISUAL), "stat.meansize": _f("irrelevant", note=VISUAL),
    "stat.extent": _f("irrelevant", note=VISUAL), "stat.center": _f("irrelevant", note=VISUAL),
    "vis": _f("irrelevant", note=VISUAL),
}

SIZES = {  # mjModel's sizes: consumed / checked as their arrays, derived counts, or of irrelevant element types
    "nq": _f("consumed"), "nv": _f("consumed", "H3"), "nu": _f("consumed", "H11a"),
    "nactuator": _f("checked", "H11a"), "nout": _f("checked", "H11a"), "na": _f("checked", "H11b"),
    "nbody": _f("consumed", "H1"), "njnt": _f("consumed"), "ngeom": _f("consumed"), "npair": _f("consumed", "H26b"),
    "nmocap": _f("checked", "H25"), "neq": _f("checked", "H25"), "ntendon": _f("checked", "H25"),
    "nwrap": _f("checked", "H25"), "nJten": _f("derived", "H25"), "nflex": _f("checked", "H25"),
    "nplugin": _f("checked", "H25"), "npluginattr": _f("derived", "H25"), "npluginstate": _f("derived", "H25"),
    "nhistory": _f("derived", "H11e"), "ngravcomp": _f("derived", "H22"), "nexclude": _f("derived", "H15"),
    **{k: _f("derived", "H25") for k in (
        "nflexnode", "nflexvert", "nflexedge", "nflexelem", "nflexelemdata", "nflexstiffness", "nflexbending",
        "nefm0dof", "nefm0L", "nefmCvert", "nflexelemedge", "nflexshelldata", "nflextexcoord", "nJfe", "nJfv")},
    **{k: _f("derived") for k in ("nbvh", "nbvhstatic", "nbvhdynamic", "ntree", "nM", "nB", "nC", "nD", "nJmom",
                                  "nnames_map")},
    **{k: _f("irrelevant", note=VISUAL) for k in (
        "noct", "nsite", "ncam", "nlight", "nmesh", "nmeshvert", "nmeshnormal", "nmeshtexcoord", "nmeshface",
        "nmeshgraph", "nmeshpoly", "nmeshpolyvert", "nmeshpolymap", "nskin", "nskinvert", "nskintexvert", "nskinface",
        "nskinbone", "nskinbonevert", "nhfield", "nhfielddata", "ntex", "ntexdata", "nmat", "npolygonmax",
        "nmeshdegmax")},
    **{k: _f("irrelevant", note="sensors, custom data, keyframes, names: not read") for k in (
        "nsensor", "nsensordata", "nnumeric", "nnumericdata", "ntext", "ntextdata", "ntuple", "ntupledata", "nkey",
        "nuser_body", "nuser_jnt", "nuser_geom", "nuser_site", "nuser_cam", "nuser_tendon", "nuser_actuator",
        "nuser_sensor", "nnames", "npaths", "nuserdata")},
    **{k: _f("irrelevant", note="MuJoCo's own memory sizes (mjData)") for k in (
        "narena", "nbuffer", "nemax", "njmax", "nconmax")},
}

PREFIXES = (  # (prefix, Field) for element types classified as a whole; checked after FIELDS and SIZES
    ("flex_", _f("checked", "H25", "flexes (nflex = 0)")),
    ("flexedge_", _f("checked", "H25", "flexes (nflex = 0)")),
    ("flexvert_", _f("checked", "H25", "flexes (nflex = 0)")),
    ("efm0_", _f("checked", "H25", "flexes (nflex = 0)")), ("efmC_", _f("checked", "H25", "flexes (nflex = 0)")),
    ("eq_", _f("consumed", "H25b", "soft linear joint equality rows")),
    ("tendon_", _f("checked", "H25", "tendons (ntendon = 0)")), ("ten_J_", _f("checked", "H25", "tendons")),
    ("wrap_", _f("checked", "H25", "tendon wraps (nwrap = 0)")),
    ("plugin_", _f("checked", "H25", "plugin instances (nplugin = 0)")),
    ("bvh_", _f("derived", note="bounding volume hierarchy (MuJoCo's collision acceleration)")),
    ("oct_", _f("irrelevant", note="mesh octrees (SDF collisions: none)")),
    ("mesh_", _f("irrelevant", "H14a", "mesh assets (the collision geoms' types are checked)")),
    ("hfield_", _f("irrelevant", "H14a", "height field assets (the collision geoms' types are checked)")),
    ("skin_", _f("irrelevant", note=VISUAL)), ("tex_", _f("irrelevant", note=VISUAL)),
    ("mat_", _f("irrelevant", note=VISUAL)), ("cam_", _f("irrelevant", note=VISUAL)),
    ("light_", _f("irrelevant", note=VISUAL)),
    ("site_", _f("irrelevant", note="sites: used only by sensors, tendons and site transmissions (none)")),
    ("sensor_", _f("irrelevant", note="sensors: the engine computes none (they do not act on the dynamics)")),
    ("key_", _f("irrelevant", note="keyframes: the engine takes its states as inputs")),
    ("name_", _f("irrelevant", note="name addresses")),
    ("numeric_", _f("irrelevant", note=USER)), ("text_", _f("irrelevant", note=USER)),
    ("tuple_", _f("irrelevant", note=USER)),
)


def classify(name):
    """The Field of an MjModel field name (opt.<x>, stat.<x>, vis included), None if unclassified."""
    if name in FIELDS:
        return FIELDS[name]
    if name in SIZES:
        return SIZES[name]
    for p, f in PREFIXES:
        if name.startswith(p):
            return f
    return None


def model_fields(m):
    """The field names of an MjModel: its public data attributes (methods and the opt / stat / vis structs left
    out), then opt.<x> and stat.<x>, and vis."""
    out = []
    for k in dir(m):
        if k.startswith("_") or k in ("opt", "stat", "vis"):
            continue
        try:
            v = getattr(m, k)
        except Exception:  # noqa: BLE001  (an accessor MuJoCo does not implement for this model)
            continue
        if not callable(v):
            out.append(k)
    for sub in ("opt", "stat"):
        o = getattr(m, sub)
        out += [f"{sub}.{k}" for k in dir(o) if not k.startswith("_") and not callable(getattr(o, k))]
    return sorted(out) + ["vis"]


def unclassified(m):
    """The fields of m that classify() does not know."""
    return [k for k in model_fields(m) if classify(k) is None]


def default_fields():
    """The names of the fields of kind default."""
    return sorted(k for k, f in FIELDS.items() if f.kind == "default")


# The probe model for default_rows(): one element of each kind the default fields belong to, nothing set beyond what
# the element needs (row indices in PROBE_ROW)
PROBE_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="root">
      <freejoint/>
      <geom name="torso" type="sphere" size="0.1"/>
      <body name="link" pos="0 0 -0.2">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="leg" type="capsule" size="0.05 0.1"/>
      </body>
    </body>
  </worldbody>
  <contact>
    <pair geom1="floor" geom2="leg"/>
  </contact>
  <actuator>
    <position name="servo" joint="hinge" kp="10"/>
  </actuator>
</mujoco>
"""
PROBE_ROW = {"body": 2, "jnt": 1, "dof": 6, "geom": 2, "pair": 0, "actuator": 0}


@lru_cache(maxsize=1)
def _probe():
    import mujoco
    return mujoco.MjModel.from_xml_string(PROBE_XML)


def default_rows():
    """{field: its compile default (one row)} for the fields of kind default, from the probe model compiled by the
    installed MuJoCo (its row of the element kind's probe element, PROBE_ROW)."""
    m = _probe()
    out = {}
    for k in default_fields():
        kind = k.split("_", 1)[0]
        out[k] = np.array(getattr(m, k)[PROBE_ROW[kind]], copy=True)
    return out


def option_default(name):
    """MuJoCo's default of the option opt.<name> (the probe model's, compiled by the installed MuJoCo)."""
    v = getattr(_probe().opt, name)
    return v.item() if isinstance(v, np.generic) else v


def non_default(m, name, defaults=None):
    """Row indices of m's field `name` (kind default) that differ from the compile default (world body row 0 not
    counted: it has no inertial or joint attributes)."""
    d = (defaults or default_rows())[name]
    a = np.asarray(getattr(m, name))
    if a.size == 0:
        return []
    rows = a.reshape(len(a), -1) != d.reshape(1, -1)
    bad = np.flatnonzero(rows.any(axis=1))
    if name.startswith("body_"):
        bad = bad[bad > 0]
    return [int(i) for i in bad]
