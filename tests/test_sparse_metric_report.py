import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.metric_report import (
    build_metric_report,
    main,
)


def record(query_id, hit25, hit50=False, **overrides):
    value = {
        "query_id": query_id,
        "scan_id": "3rscan/scene",
        "target_id": 1,
        "target": "chair",
        "is_hard": False,
        "is_unique": True,
        "is_view_dependent": False,
        "hits": {"0.25": hit25, "0.5": hit50},
    }
    value.update(overrides)
    return value


def report(records):
    return {
        "schema_version": "1.0",
        "iou_thresholds": [0.25, 0.5],
        "records": records,
    }


class SparseMetricReportTest(unittest.TestCase):
    def test_computes_paired_beneficial_and_harmful_rates(self):
        baseline = report(
            [
                record("q1", False),
                record("q2", True),
                record("q3", False, is_hard=True, is_unique=False),
            ]
        )
        candidate = report(
            [
                record("q1", True),
                record("q2", False),
                record("q3", True, is_hard=True, is_unique=False),
            ]
        )

        combined = build_metric_report(
            {"k3": baseline, "k5": candidate},
            baseline_name="k3",
        )

        overall = combined["comparisons"]["k5"]["by_stratum"]["overall"]["0.25"]
        self.assertEqual(overall["count"], 3)
        self.assertEqual(overall["beneficial_count"], 2)
        self.assertEqual(overall["harmful_count"], 1)
        self.assertAlmostEqual(overall["hit_rate_gain"], 1 / 3)
        hard = combined["comparisons"]["k5"]["by_stratum"]["hard"]["0.25"]
        self.assertEqual(hard["beneficial_rate"], 1.0)

    def test_rejects_misaligned_query_metadata(self):
        baseline = report([record("q1", False)])
        candidate = report([record("q1", True, target_id=2)])

        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            build_metric_report(
                {"baseline": baseline, "candidate": candidate},
                baseline_name="baseline",
            )

    def test_cli_writes_json_and_csv(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "k3.json"
            second = root / "k5.json"
            first.write_text(
                json.dumps(report([record("q1", False)])),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(report([record("q1", True)])),
                encoding="utf-8",
            )
            output = root / "summary.json"
            csv_output = root / "summary.csv"

            return_code = main(
                [
                    "--input",
                    f"k3={first}",
                    "--input",
                    f"k5={second}",
                    "--baseline",
                    "k3",
                    "--output",
                    str(output),
                    "--csv-output",
                    str(csv_output),
                ]
            )
            combined = json.loads(output.read_text(encoding="utf-8"))
            with csv_output.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(return_code, 0)
        self.assertEqual(combined["experiment_count"], 2)
        self.assertTrue(any(row["row_type"] == "comparison" for row in rows))


if __name__ == "__main__":
    unittest.main()
