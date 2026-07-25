import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bip3d.structures import EulerDepthInstance3DBoxes
from projects.sparse_grounding.query_metrics import (
    SparseGroundingMetric,
    build_query_record,
    summarize_query_records,
)


def boxes(values):
    return EulerDepthInstance3DBoxes(
        torch.tensor(values, dtype=torch.float32),
        origin=(0.5, 0.5, 0.5),
    )


def annotation(query_id="vg.json:7"):
    return {
        "sparse_query_id": query_id,
        "sparse_query_file": "vg.json",
        "sparse_query_index": 7,
        "scan_id": "3rscan/scene",
        "target_id": 4,
        "target": "chair",
        "text": "the chair on the left",
        "is_hard": True,
        "is_unique": False,
        "is_view_dep": True,
        "gt_bboxes_3d": boxes([[0, 0, 0, 2, 2, 2, 0, 0, 0]]),
    }


def prediction():
    return {
        "bboxes_3d": boxes(
            [
                [8, 0, 0, 2, 2, 2, 0, 0, 0],
                [0, 0, 0, 2, 2, 2, 0, 0, 0],
            ]
        ),
        "target_scores_3d": torch.tensor([0.9, 0.8]),
    }


class QueryRecordTest(unittest.TestCase):
    def test_uses_official_top_k_any_hit_semantics(self):
        record = build_query_record(annotation(), prediction())

        self.assertEqual(record["query_id"], "vg.json:7")
        self.assertAlmostEqual(record["max_iou"], 1.0)
        self.assertTrue(record["hits"]["0.25"])
        self.assertTrue(record["hits"]["0.5"])
        self.assertAlmostEqual(record["top_scores"][0], 0.9)
        self.assertAlmostEqual(record["top_scores"][1], 0.8)

    def test_missing_stable_query_id_is_rejected(self):
        value = annotation()
        del value["sparse_query_id"]

        with self.assertRaisesRegex(ValueError, "sparse_query_id"):
            build_query_record(value, prediction())

    def test_summarizes_official_strata(self):
        first = build_query_record(annotation(), prediction())
        second = dict(first)
        second.update(
            {
                "query_id": "vg.json:8",
                "is_hard": False,
                "is_unique": True,
                "is_view_dependent": False,
                "max_iou": 0.0,
                "hits": {"0.25": False, "0.5": False},
            }
        )

        summary = summarize_query_records([first, second])

        self.assertEqual(summary["overall"]["count"], 2)
        self.assertEqual(summary["overall"]["hit_rate@0.25"], 0.5)
        self.assertEqual(summary["hard"]["hit_rate@0.5"], 1.0)
        self.assertEqual(summary["easy"]["hit_rate@0.5"], 0.0)


class SparseGroundingMetricTest(unittest.TestCase):
    def test_exports_atomic_per_query_report(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "per-query.json"
            metric = SparseGroundingMetric(query_result_file=str(output))

            metrics = metric.compute_metrics([(annotation(), prediction())])
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertAlmostEqual(metrics["Overall@0.25"], 1.0)
        self.assertEqual(report["query_count"], 1)
        self.assertEqual(report["records"][0]["query_id"], "vg.json:7")
        self.assertEqual(report["summary"]["hard"]["hit_count@0.5"], 1)


if __name__ == "__main__":
    unittest.main()
