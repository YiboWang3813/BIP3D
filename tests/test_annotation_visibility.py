import json
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.annotation_visibility import (
    evaluate_query_visibility,
    is_view_dependent,
    random_oracle_support_probability,
    run_annotation_visibility_audit,
)
from projects.sparse_grounding.protocol import (
    CameraGraphStats,
    SparseSceneProtocol,
    ViewSelection,
)


SCAN_ID = "3rscan/scene"
FRAMES = tuple(f"3rscan/scene/{index}.jpg" for index in range(12))


def make_scene():
    return {
        "sample_idx": SCAN_ID,
        "instances": [
            {"bbox_id": 10, "bbox_label_3d": 1},
            {"bbox_id": 20, "bbox_label_3d": 2},
            {"bbox_id": 20, "bbox_label_3d": 3},
        ],
        "images": [
            {
                "img_path": frame,
                "visible_instance_ids": (
                    [1] if index in {1, 8, 9} else []
                ),
            }
            for index, frame in enumerate(FRAMES)
        ],
    }


def make_protocol():
    return SparseSceneProtocol(
        scene_id="3rscan%2Fscene",
        dataset="synthetic",
        protocol_version="v1",
        trajectory_type="global_fps",
        seed=0,
        selections=(
            ViewSelection(3, FRAMES[:3]),
            ViewSelection(5, FRAMES[:5]),
            ViewSelection(8, FRAMES[:8]),
        ),
        candidate_heldout_frame_ids=FRAMES[8:],
        camera_graph_stats=CameraGraphStats(12, 11, 1, 1, 1),
    )


def make_query():
    return {
        "scan_id": SCAN_ID,
        "target_id": 20,
        "target": "chair",
        "text": "the chair on the left",
        "distractor_ids": [1, 2, 3, 4],
    }


class OracleProbabilityTest(unittest.TestCase):
    def test_probability_without_replacement(self):
        probability = random_oracle_support_probability(4, 2, 2)

        self.assertAlmostEqual(probability, 5 / 6)

    def test_zero_candidate_cases(self):
        self.assertEqual(random_oracle_support_probability(0, 0, 4), 0)
        self.assertEqual(random_oracle_support_probability(4, 0, 4), 0)


class QueryVisibilityTest(unittest.TestCase):
    def test_bbox_id_is_mapped_to_instance_index(self):
        record = evaluate_query_visibility(
            make_scene(),
            make_query(),
            make_protocol(),
            budgets=(3, 5, 8),
            oracle_view_budget=4,
            category_names={1: "table", 2: "chair", 3: "wall"},
        )

        self.assertEqual(record["instance_index"], 1)
        self.assertTrue(record["is_hard"])
        self.assertFalse(record["is_unique"])
        self.assertTrue(record["is_view_dependent"])
        self.assertEqual(record["reference_visible_frame_count"], 3)
        self.assertEqual(record["by_budget"]["3"]["sparse_visible_frame_count"], 1)
        self.assertTrue(record["by_budget"]["3"]["sparse_supported"])

    def test_view_dependence_matches_bip3d_word_rule(self):
        self.assertTrue(is_view_dependent("chair behind table"))
        self.assertFalse(is_view_dependent("bright chair near table"))


class AnnotationAuditTest(unittest.TestCase):
    def test_writes_query_and_unique_target_summaries(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            info_file = root / "info.pkl"
            with info_file.open("wb") as stream:
                pickle.dump(
                    {
                        "metainfo": {
                            "categories": {
                                "table": 1,
                                "chair": 2,
                                "wall": 3,
                            }
                        },
                        "data_list": [make_scene()],
                    },
                    stream,
                )
            vg_file = root / "vg.json"
            vg_file.write_text(json.dumps([make_query(), make_query()]))
            protocol_dir = root / "protocols"
            protocol_dir.mkdir()
            make_protocol().dump(protocol_dir / "3rscan%2Fscene.json")

            result = run_annotation_visibility_audit(
                info_file=info_file,
                vg_file=vg_file,
                protocol_dir=protocol_dir,
                budgets=(3, 5, 8),
                oracle_view_budget=4,
                source_datasets=("3rscan",),
            )

        self.assertEqual(result["summary"]["query_count"], 2)
        self.assertEqual(result["summary"]["unique_target_count"], 1)
        query_metrics = result["summary"]["by_budget"]["3"]["queries"]["overall"]
        target_metrics = result["summary"]["by_budget"]["3"]["unique_targets"]["overall"]
        self.assertEqual(query_metrics["sparse_supported_count"], 2)
        self.assertEqual(target_metrics["sparse_supported_count"], 1)
        self.assertEqual(result["summary"]["error_count"], 0)


if __name__ == "__main__":
    unittest.main()
