"""CPU checks of the actual eligible graph; no physics steps or CUDA imports.

PYTHONPATH=src python scripts/hand_contact_graph_check.py
"""

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from diagnose_hand_contact_graph import contact_graph, contact_occupancy, load_states


def two_roots(*, contact="enable", exclude=False, explicit=False, hinge=False, extra_geom=False):
    finger = ('<body name="finger"><joint name="finger_joint" range="-1 1"/>'
              '<geom name="finger_geom" type="sphere" size=".02"/></body>' if hinge else "")
    extra = '<geom name="far_geom" type="sphere" pos="2 0 0" size=".03"/>' if extra_geom else ""
    contact_xml = ('<exclude body1="left" body2="right"/>' if exclude else "")
    contact_xml += ('<pair geom1="left_geom" geom2="right_geom"/>' if explicit else "")
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><option integrator="implicitfast"><flag contact="{contact}"/></option>'
        '<worldbody><body name="left"><freejoint name="left_free"/>'
        '<geom name="left_geom" type="sphere" size=".03"/>' + finger + extra + '</body>'
        '<body name="right" pos=".2 0 0"><freejoint name="right_free"/>'
        '<geom name="right_geom" type="sphere" size=".03"/></body></worldbody>'
        f'<contact>{contact_xml}</contact></mujoco>')


class ContactGraphCheck(unittest.TestCase):
    def test_zero_explicit_pairs_keeps_cross_root_collision_graph(self):
        report = contact_graph(two_roots())
        self.assertEqual(report["explicit_pairs"], 0)
        self.assertEqual(report["eligible_pairs"], 1)
        self.assertEqual(report["cross_root_pairs"], 1)
        self.assertEqual(report["pair_types"], {"sphere-sphere": 1})
        self.assertEqual(report["max_relative_dof_support"], 12)
        self.assertNotIn("H6", report["current_engine_errors"])
        self.assertNotIn("H15", report["current_engine_errors"])

    def test_authored_exclusion_removes_the_cross_root_pair(self):
        report = contact_graph(two_roots(exclude=True))
        self.assertEqual(report["eligible_pairs"], 0)
        self.assertEqual(report["excluded_body_pairs"][0]["body_names"], ["left", "right"])
        self.assertTrue(report["contact_enabled"])

    def test_explicit_pair_replaces_only_itself(self):
        report = contact_graph(two_roots(explicit=True, hinge=True))
        self.assertEqual(report["explicit_pairs"], 1)
        self.assertEqual(report["implicit_eligible_pairs"], 1)
        self.assertEqual(report["eligible_pairs"], 2)
        self.assertEqual(len({tuple(pair) for pair in report["pair_geom_ids"]}), 2)

    def test_disabled_contacts_are_distinct_from_empty_pair_table(self):
        report = contact_graph(two_roots(contact="disable", explicit=True))
        self.assertFalse(report["contact_enabled"])
        self.assertEqual(report["explicit_pairs"], 1)
        self.assertEqual(report["eligible_pairs"], 0)

    def test_real_second_root_addresses_and_passive_hinge_are_retained(self):
        report = contact_graph(two_roots(hinge=True))
        self.assertEqual(report["free_roots"][1]["qpos_address"], 8)
        self.assertEqual(report["free_roots"][1]["qvel_address"], 7)
        self.assertEqual(report["max_relative_dof_support"], 13)
        self.assertEqual(report["self_pairs"], 0)  # the authored parent filter excludes palm/finger
        self.assertEqual(report["cross_root_pairs"], 2)

    def test_native_mesh_multiccd_capacity_is_not_one_contact_per_pair(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><asset><mesh name="cube" vertex="-.1 -.1 -.1  .1 -.1 -.1  '
            '-.1 .1 -.1  .1 .1 -.1  -.1 -.1 .1  .1 -.1 .1  -.1 .1 .1  .1 .1 .1"/></asset>'
            '<worldbody><body><freejoint/><geom type="mesh" mesh="cube"/></body>'
            '<body pos="1 0 0"><freejoint/><geom type="mesh" mesh="cube"/></body></worldbody></mujoco>')
        report = contact_graph(model)
        self.assertTrue(report["nativeccd_enabled"])
        self.assertTrue(report["multiccd_enabled"])
        self.assertEqual(report["max_contacts_per_pair_histogram"], {4: 1})
        self.assertEqual(report["contact_count_upper_bound"], 4)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        self.assertEqual(contact_graph(model)["contact_count_upper_bound"], 1)


def cohort(model, count):
    metadata = dict(embodiment_id="fixture", asset_choice="native", source_identity={"fixture": "two_roots"},
                    joint_names=[model.joint(j).name for j in range(model.njnt)],
                    qpos_addresses=model.jnt_qposadr.tolist(), qvel_addresses=model.jnt_dofadr.tolist(),
                    frames=[dict(frame_id=i, source_key={"episode": "fixture", "tick": i}) for i in range(count)])
    return np.repeat(model.qpos0[None], count, axis=0), np.zeros((count, model.nv)), metadata


class OccupancyCheck(unittest.TestCase):
    def test_actual_cross_contact_is_smaller_than_static_upper_bound(self):
        model = two_roots(extra_geom=True)
        q, v, meta = cohort(model, 2)
        q[1, 7] = .04
        report = contact_occupancy(model, q, v, meta, embodiment_id="fixture", asset_choice="native", max_frames=2)
        self.assertEqual(report["static_contact_upper_bound"], 2)
        self.assertEqual([row["contacts"] for row in report["frames"]], [0, 1])
        self.assertEqual(report["frames"][1]["cross_root_contacts"], 1)
        self.assertEqual(report["frames"][1]["contact_rows"], 4)
        self.assertEqual(report["summaries"]["all_evaluated"]["contacts"]["max"], 1)
        self.assertEqual(report["advancing_steps"], 0)
        np.testing.assert_array_equal(q[1, 7:10], [.04, 0, 0])

    def test_complete_chart_and_explicit_ceiling_are_required(self):
        model = two_roots(hinge=True)
        q, v, meta = cohort(model, 2)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            contact_occupancy(model, q[:, :-1], v, meta, embodiment_id="fixture", asset_choice="native", max_frames=2)
        with self.assertRaisesRegex(ValueError, "ceiling"):
            contact_occupancy(model, q, v, meta, embodiment_id="fixture", asset_choice="native", max_frames=1)
        with self.assertRaisesRegex(ValueError, "hand/asset binding"):
            contact_occupancy(model, q, v, meta, embodiment_id="another_hand", asset_choice="native", max_frames=2)
        meta["joint_names"] = meta["joint_names"][::-1]
        with self.assertRaisesRegex(ValueError, "native chart"):
            contact_occupancy(model, q, v, meta, embodiment_id="fixture", asset_choice="native", max_frames=2)

    def test_invalid_and_soft_limit_states_are_reported_separately(self):
        model = two_roots(hinge=True)
        q, v, meta = cohort(model, 4)
        q[1, 7] = 1.1
        q[2, 0] = np.nan
        q[3, 3:7] = 0
        report = contact_occupancy(model, q, v, meta, embodiment_id="fixture", asset_choice="native", max_frames=4)
        self.assertEqual((report["evaluated_frames"], report["failed_frames"], report["out_of_range_frames"]), (2, 2, 1))
        self.assertTrue(report["frames"][1]["evaluated"])
        self.assertEqual(report["frames"][1]["joint_limit_violations"][0]["joint_name"], "finger_joint")
        self.assertIn("nonfinite", report["frames"][2]["failure"])
        self.assertIn("quaternion", report["frames"][3]["failure"])
        self.assertEqual(report["summaries"]["outside_joint_ranges"]["rows"]["max"], 1)

    def test_json_and_npz_preserve_the_same_actual_chart_and_sources(self):
        model = two_roots()
        q, v, meta = cohort(model, 1)
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"states.json").write_text(json.dumps(dict(qpos=q.tolist(), qvel=v.tolist(), metadata=meta)))
            np.savez(path/"states.npz", qpos=q, qvel=v, metadata=json.dumps(meta))
            for name in ("states.json", "states.npz"):
                qr, vr, mr = load_states(path/name)
                np.testing.assert_array_equal(qr, q)
                np.testing.assert_array_equal(vr, v)
                self.assertEqual(mr, meta)


if __name__ == "__main__":
    unittest.main()
