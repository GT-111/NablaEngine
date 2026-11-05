"""Inspect the compiled hand scene without advancing physics or loading CUDA.

PYTHONPATH=src python scripts/diagnose_hand_contact_graph.py scene.xml [scene2.xml] --out runs/hand_graph.json

Pass the actual floating scenes, including their contact masks and exclusions.
An empty explicit <pair> table does not imply an empty collision graph. Pair
counts here are eligible pairs, not simultaneous active contacts or a capacity
qualification. CPU regression: scripts/hand_contact_graph_check.py.

Measure a bounded full-state cohort with mj_fwdPosition (no integration):
  ... scene.xml --states actual.npz --embodiment-id inspire_rh56f1 --asset-choice native --max-frames 256

JSON has qpos [frames,nq], qvel [frames,nv], and metadata. NPZ has the same
arrays and metadata as a scalar JSON string; no pickle. Metadata contains:
  embodiment_id, asset_choice, source_identity,
  joint_names, qpos_addresses, qvel_addresses (complete compiled joint order),
  frames: [{"frame_id": ..., "source_key": ..., ...}, ...].
Export actual physical Q/V, including every FREE and passive coordinate. The
reader rejects a different chart and never fills missing coordinates. The
explicit frame ceiling rejects oversized inputs instead of sampling them.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import mujoco
import numpy as np

from nablaengine.model import capability


def contact_graph(model):
    """Return the actual eligible graph and coordinate requirements of an MjModel."""
    contact_enabled = not bool(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT))
    explicit = [(int(a), int(b)) for a, b in zip(model.pair_geom1, model.pair_geom2)] if contact_enabled else []
    implicit = capability.filter_pairs(model)
    pairs = explicit + implicit
    collision_geoms = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    parameter_values = {name: np.unique(np.asarray(getattr(model, name))[collision_geoms], axis=0).tolist()
                        for name in ("geom_condim", "geom_margin", "geom_gap", "geom_friction", "geom_solref",
                                     "geom_solimp", "geom_priority", "geom_solmix")}
    supports = [set() for _ in range(model.nbody)]
    depths = [0] * model.nbody
    children = [0] * model.nbody
    widths = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3,
              mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}
    for body in range(1, model.nbody):
        parent = int(model.body_parentid[body])
        supports[body] = supports[parent].copy()
        depths[body] = depths[parent] + 1
        children[parent] += 1
        first = int(model.body_jntadr[body])
        for joint in range(first, first + int(model.body_jntnum[body])):
            start = int(model.jnt_dofadr[joint])
            supports[body].update(range(start, start + widths[int(model.jnt_type[joint])]))
    roots = []
    for joint in range(model.njnt):
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            body = int(model.jnt_bodyid[joint])
            roots.append(dict(joint_id=joint, joint_name=model.joint(joint).name, body_id=body,
                              body_name=model.body(body).name, qpos_address=int(model.jnt_qposadr[joint]),
                              qvel_address=int(model.jnt_dofadr[joint])))
    root_pairs, types, contact_bounds = Counter(), Counter(), Counter()
    max_support = 0
    for first, second in pairs:
        body1, body2 = int(model.geom_bodyid[first]), int(model.geom_bodyid[second])
        root1, root2 = sorted((int(model.body_rootid[body1]), int(model.body_rootid[body2])))
        root_pairs[(root1, root2)] += 1
        kinds = sorted(mujoco.mjtGeom(int(model.geom_type[g])).name.removeprefix("mjGEOM_").lower()
                       for g in (first, second))
        types["-".join(kinds)] += 1
        contact_bounds[int(mujoco.mj_maxContact(model, first, second, -1))] += 1
        max_support = max(max_support, len(supports[body1] ^ supports[body2]))
    return dict(mujoco_version=mujoco.__version__, contact_enabled=contact_enabled,
                cone=mujoco.mjtCone(int(model.opt.cone)).name, timestep=float(model.opt.timestep),
                nativeccd_enabled=not bool(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)),
                multiccd_enabled=not bool(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)),
                ccd_iterations=int(model.opt.ccd_iterations), ccd_tolerance=float(model.opt.ccd_tolerance),
                nbody=int(model.nbody), nq=int(model.nq), nv=int(model.nv), nu=int(model.nu),
                neq=int(model.neq), ngeom=int(model.ngeom),
                collision_geoms=len(collision_geoms),
                geom_collision_masks=[dict(geom_id=g, geom_name=model.geom(g).name,
                                           body_id=int(model.geom_bodyid[g]), contype=int(model.geom_contype[g]),
                                           conaffinity=int(model.geom_conaffinity[g])) for g in collision_geoms],
                collision_geom_parameter_values=parameter_values,
                explicit_pairs=int(model.npair), implicit_eligible_pairs=len(implicit), eligible_pairs=len(pairs),
                self_pairs=sum(count for (a, b), count in root_pairs.items() if a == b),
                cross_root_pairs=sum(count for (a, b), count in root_pairs.items() if a != b),
                pair_types=dict(sorted(types.items())),
                max_contacts_per_pair_histogram=dict(sorted(contact_bounds.items())),
                contact_count_upper_bound=sum(bound * count for bound, count in contact_bounds.items()),
                root_pair_counts=[dict(root_body_ids=list(roots), pairs=count)
                                  for roots, count in sorted(root_pairs.items())],
                pair_geom_ids=[list(pair) for pair in pairs],
                excluded_body_pairs=[dict(body_ids=[int(sig) >> 16, int(sig) & 65535],
                                          body_names=[model.body(int(sig) >> 16).name,
                                                      model.body(int(sig) & 65535).name])
                                     for sig in model.exclude_signature],
                free_roots=roots, max_body_depth=max(depths), max_children=max(children),
                limited_joint_count=int(sum(model.jnt_limited)),
                frictionloss_dof_count=int(sum(model.dof_frictionloss > 0)),
                npolygonmax=int(model.npolygonmax), nmeshdegmax=int(model.nmeshdegmax),
                max_relative_dof_support=max_support,
                pair_index_bits_required=max(1, (len(pairs) - 1).bit_length()),
                current_engine_errors=capability.check(model, com_body=tuple(r["body_id"] for r in roots)).codes("error"))


def load_states(path):
    """Read the same explicit metadata/full-state arrays from JSON or NPZ."""
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return archive["qpos"].copy(), archive["qvel"].copy(), json.loads(archive["metadata"].item())
    value = json.loads(path.read_text())
    return np.asarray(value["qpos"], dtype=np.float64), np.asarray(value["qvel"], dtype=np.float64), value["metadata"]


def _quantiles(values):
    return dict(zip(("min", "p50", "p90", "p95", "p99", "max"),
                    np.quantile(values, (0, .5, .9, .95, .99, 1)).tolist(), strict=True)) if values else None


def contact_occupancy(model, qpos, qvel, metadata, *, embodiment_id, asset_choice, max_frames):
    """Measure supplied coordinates as-is, separating failed and soft-limit states."""
    qpos, qvel = np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)
    if metadata["embodiment_id"] != embodiment_id or metadata["asset_choice"] != asset_choice:
        raise ValueError("state hand/asset binding differs from the selected model")
    expected = dict(joint_names=[model.joint(j).name for j in range(model.njnt)],
                    qpos_addresses=model.jnt_qposadr.tolist(), qvel_addresses=model.jnt_dofadr.tolist())
    if any(metadata[key] != value for key, value in expected.items()):
        raise ValueError("state joint names/addresses differ from the complete native chart")
    if (qpos.ndim != 2 or qvel.ndim != 2 or qpos.shape[1] != model.nq or qvel.shape[1] != model.nv
            or qpos.shape[0] != qvel.shape[0]):
        raise ValueError(f"full native state dimensions must be [frames,{model.nq}] and [frames,{model.nv}]")
    count = qpos.shape[0]
    if not 0 < count <= max_frames:
        raise ValueError(f"state cohort has {count} frames; explicit ceiling is {max_frames}")
    frames = metadata["frames"]
    if (not metadata["source_identity"] or len(frames) != count
            or any("frame_id" not in frame or "source_key" not in frame or frame["source_key"] is None for frame in frames)):
        raise ValueError("every actual frame requires its source key and the cohort source identity")
    graph = contact_graph(model)
    limits = [j for j in range(model.njnt) if model.jnt_limited[j]
              and int(model.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
    reports = []
    for index, frame in enumerate(frames):
        result = dict(input_index=index, frame=frame, evaluated=False)
        reports.append(result)
        if not np.isfinite(qpos[index]).all() or not np.isfinite(qvel[index]).all():
            result["failure"] = "nonfinite input coordinates"
            continue
        if any(not np.any(qpos[index, root["qpos_address"]+3:root["qpos_address"]+7]) for root in graph["free_roots"]):
            result["failure"] = "zero FREE-joint quaternion in the supplied physical state"
            continue
        result["joint_limit_violations"] = [
            dict(joint_name=model.joint(j).name, qpos_address=int(model.jnt_qposadr[j]),
                 value=float(qpos[index, model.jnt_qposadr[j]]), range=model.jnt_range[j].tolist())
            for j in limits if not model.jnt_range[j, 0] <= qpos[index, model.jnt_qposadr[j]] <= model.jnt_range[j, 1]]
        result["out_of_range"] = bool(result["joint_limit_violations"])
        data = mujoco.MjData(model)
        data.qpos[:] = qpos[index]
        data.qvel[:] = qvel[index]
        try:
            mujoco.mj_fwdPosition(model, data)
        except (mujoco.FatalError, mujoco.UnexpectedError) as error:
            result["failure"] = str(error)
            continue
        warnings = [dict(code=mujoco.mjtWarning(i).name, count=int(w.number), lastinfo=int(w.lastinfo))
                    for i, w in enumerate(data.warning) if w.number]
        if warnings or not np.isfinite(data.efc_pos[:data.nefc]).all():
            result["failure"] = "MuJoCo warning or nonfinite constraint position"
            result["warnings"] = warnings
            continue
        pairs, types, active_types = {}, Counter(), Counter()
        cross = active = 0
        for contact in data.contact[:data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            kind = "-".join(sorted(mujoco.mjtGeom(int(model.geom_type[g])).name.removeprefix("mjGEOM_").lower()
                                   for g in (first, second)))
            root1, root2 = (int(model.body_rootid[model.geom_bodyid[g]]) for g in (first, second))
            is_active = contact.efc_address >= 0
            cross += root1 != root2
            active += is_active
            types[kind] += 1
            active_types[kind] += is_active
            pair = pairs.setdefault(tuple(sorted((first, second))), dict(contacts=0, constraint_contacts=0, type=kind))
            pair["contacts"] += 1
            pair["constraint_contacts"] += int(is_active)
        row_types = Counter(mujoco.mjtConstraint(int(kind)).name for kind in data.efc_type[:data.nefc])
        contact_rows = sum(value for key, value in row_types.items() if key.startswith("mjCNSTR_CONTACT_"))
        result.update(evaluated=True, contacts=int(data.ncon), constraint_contacts=int(active), rows=int(data.nefc),
                      contact_rows=contact_rows, self_contacts=int(data.ncon-cross), cross_root_contacts=int(cross),
                      contact_pair_types=dict(types), constraint_contact_pair_types=dict(active_types),
                      constraint_rows_by_type=dict(row_types),
                      geom_pair_counts=[dict(geom_ids=list(pair), geom_names=[model.geom(g).name for g in pair], **value)
                                        for pair, value in sorted(pairs.items())],
                      packed_contact_jacobian_bytes=4*int(data.ncon)*3*graph["max_relative_dof_support"],
                      contact_dof_index_bytes=int(data.ncon)*graph["max_relative_dof_support"],
                      row_scalar_array_bytes=19*int(data.nefc),
                      tape_contact_table_bytes=4*((int(data.ncon)+1)//2),
                      tape_row_zone_bytes=4*((int(data.nefc)+31)//32))
    evaluated = [row for row in reports if row["evaluated"]]
    keys = ("contacts", "constraint_contacts", "rows", "contact_rows", "self_contacts", "cross_root_contacts",
            "packed_contact_jacobian_bytes", "contact_dof_index_bytes", "row_scalar_array_bytes",
            "tape_contact_table_bytes", "tape_row_zone_bytes")
    summaries = {name: dict(frame_count=len(selected), **{key: _quantiles([row[key] for row in selected]) for key in keys})
                 for name, selected in (("all_evaluated", evaluated),
                                        ("within_joint_ranges", [row for row in evaluated if not row["out_of_range"]]),
                                        ("outside_joint_ranges", [row for row in evaluated if row["out_of_range"]]))}
    total_types, total_rows = Counter(), Counter()
    for row in evaluated:
        total_types.update(row["contact_pair_types"])
        total_rows.update(row["constraint_rows_by_type"])
    return dict(operation="mj_fwdPosition", advancing_steps=0, nq=int(model.nq), nv=int(model.nv),
                metadata=metadata, supplied_frames=count, evaluated_frames=len(evaluated),
                failed_frames=count-len(evaluated), out_of_range_frames=sum(row.get("out_of_range", False) for row in reports),
                eligible_pair_count=graph["eligible_pairs"], static_contact_upper_bound=graph["contact_count_upper_bound"],
                contact_pair_types=dict(total_types), constraint_rows_by_type=dict(total_rows),
                summaries=summaries, frames=reports,
                state_and_two_factor_tape_bytes=4*(int(model.nq)+3*int(model.nv)+int(model.nv)*(int(model.nv)+1)),
                memory_scope="Array payloads only: excludes alignment, manifold residuals, solver scratch and allocation capacity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--states", type=Path)
    parser.add_argument("--embodiment-id")
    parser.add_argument("--asset-choice")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.states and (len(args.models) != 1 or not args.embodiment_id or not args.asset_choice or not args.max_frames):
        parser.error("--states requires one model, --embodiment-id, --asset-choice and --max-frames")
    reports = []
    for path in args.models:
        model = (mujoco.MjModel.from_binary_path(str(path)) if path.suffix == ".mjb"
                 else mujoco.MjModel.from_xml_path(str(path)))
        report = dict(model_file=str(path.resolve()), **contact_graph(model))
        if args.states:
            report["occupancy"] = contact_occupancy(model, *load_states(args.states), embodiment_id=args.embodiment_id,
                                                     asset_choice=args.asset_choice, max_frames=args.max_frames)
            report["occupancy"]["state_file"] = str(args.states.resolve())
        reports.append(report)
    result = json.dumps(reports, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
