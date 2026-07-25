import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.error_analysis import build_error_analysis


def record(query_id, hit, max_iou):
    return {
        "query_id": query_id,
        "scan_id": "3rscan/scene",
        "target_id": 1,
        "target": "chair",
        "text": "the chair",
        "is_hard": False,
        "is_unique": True,
        "is_view_dependent": False,
        "max_iou": max_iou,
        "hits": {"0.25": hit, "0.5": hit},
        "selected_frame_ids": ["base.jpg"],
        "oracle_frame_ids": None,
        "top_bboxes_3d": [[0] * 9],
        "gt_bboxes_3d": [[0] * 9],
    }


def report(records):
    return {
        "schema_version": "1.0",
        "iou_thresholds": [0.25, 0.5],
        "records": records,
    }


class SparseErrorAnalysisTest(unittest.TestCase):
    def test_classifies_and_ranks_paired_query_outcomes(self):
        baseline = report(
            [
                record("recovered", False, 0.1),
                record("harmed", True, 0.8),
                record("failed", False, 0.2),
                record("success", True, 0.6),
            ]
        )
        candidate_records = [
            record("recovered", True, 0.7),
            record("harmed", False, 0.1),
            record("failed", False, 0.15),
            record("success", True, 0.9),
        ]
        for value in candidate_records:
            value["selected_frame_ids"] = ["base.jpg", "oracle.jpg"]
            value["oracle_frame_ids"] = ["oracle.jpg"]
        candidate = report(candidate_records)

        analysis = build_error_analysis(
            baseline,
            candidate,
            top_per_status=1,
        )

        self.assertEqual(
            analysis["status_counts"],
            {
                "recovered": 1,
                "harmed": 1,
                "persistent_failure": 1,
                "robust_success": 1,
            },
        )
        recovered = analysis["cases"]["recovered"][0]
        self.assertEqual(recovered["query_id"], "recovered")
        self.assertAlmostEqual(recovered["iou_gain"], 0.6)
        self.assertEqual(
            recovered["candidate_oracle_frame_ids"],
            ["oracle.jpg"],
        )

    def test_rejects_invalid_analysis_settings(self):
        value = report([record("query", False, 0.1)])

        with self.assertRaisesRegex(ValueError, "invalid"):
            build_error_analysis(
                value,
                value,
                iou_threshold=0,
            )


if __name__ == "__main__":
    unittest.main()
