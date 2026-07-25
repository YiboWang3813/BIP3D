import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects"))

from sparse_grounding.protocol import (
    CameraGraphStats,
    SparseSceneProtocol,
    ViewSelection,
)


def make_protocol(**overrides):
    values = {
        "scene_id": "scene0000_00",
        "dataset": "scannet",
        "protocol_version": "local-v1",
        "trajectory_type": "local_connected",
        "seed": 20260724,
        "selections": (
            ViewSelection(3, ("12", "24", "31")),
            ViewSelection(5, ("12", "24", "31", "42", "57")),
            ViewSelection(
                8,
                ("12", "24", "31", "42", "57", "61", "63", "68"),
            ),
        ),
        "candidate_heldout_frame_ids": ("72", "81"),
        "camera_graph_stats": CameraGraphStats(
            node_count=100,
            edge_count=210,
            mean_translation_m=0.43,
            max_translation_m=1.12,
            mean_rotation_deg=18.2,
        ),
    }
    values.update(overrides)
    return SparseSceneProtocol(**values)


class ViewSelectionTest(unittest.TestCase):
    def test_budget_must_match_frame_count(self):
        with self.assertRaisesRegex(ValueError, "exactly 3 frames"):
            ViewSelection(3, ("1", "2"))

    def test_frame_ids_must_be_unique(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            ViewSelection(3, ("1", "2", "2"))

    def test_frame_ids_must_be_immutable(self):
        with self.assertRaisesRegex(ValueError, "immutable tuple"):
            ViewSelection(3, ["1", "2", "3"])


class SparseSceneProtocolTest(unittest.TestCase):
    def test_json_round_trip_is_stable(self):
        protocol = make_protocol()

        decoded = SparseSceneProtocol.from_json(protocol.to_json())

        self.assertEqual(decoded, protocol)
        self.assertEqual(decoded.to_json(), protocol.to_json())

    def test_file_round_trip(self):
        protocol = make_protocol()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scene0000_00.json"
            protocol.dump(path)

            self.assertEqual(SparseSceneProtocol.load(path), protocol)
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))

    def test_selections_must_be_nested(self):
        selections = (
            ViewSelection(3, ("1", "2", "3")),
            ViewSelection(5, ("1", "2", "4", "5", "6")),
        )

        with self.assertRaisesRegex(ValueError, "nested"):
            make_protocol(selections=selections)

    def test_heldout_frames_cannot_overlap_selection(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            make_protocol(candidate_heldout_frame_ids=("68", "72"))

    def test_unknown_fields_are_rejected(self):
        value = make_protocol().to_dict()
        value["query_id"] = "forbidden-query-specific-field"

        with self.assertRaisesRegex(ValueError, "unknown=.*query_id"):
            SparseSceneProtocol.from_dict(value)

    def test_unsupported_schema_version_is_rejected(self):
        value = make_protocol().to_dict()
        value["schema_version"] = "2.0"

        with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
            SparseSceneProtocol.from_dict(value)

    def test_selection_entries_must_be_objects(self):
        value = make_protocol().to_dict()
        value["selections"] = ["not-an-object"]

        with self.assertRaisesRegex(ValueError, "selection must be an object"):
            SparseSceneProtocol.from_dict(value)

    def test_invalid_trajectory_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "trajectory_type"):
            make_protocol(trajectory_type="target_oracle")


if __name__ == "__main__":
    unittest.main()
