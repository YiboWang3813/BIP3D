import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.reliability import (
    binary_auroc,
    build_reliability_report,
    cycle_reprojection_error,
    depth_hypothesis_variance,
    expected_calibration_error,
    hypothesis_variance,
    real_support_ratio,
    reliability_binned_gain,
    spearman_correlation,
)
from tools.evaluate_generation_reliability import main


class GenerationReliabilityTest(unittest.TestCase):
    def test_hypothesis_and_depth_variance(self):
        values = np.array([[0.0, 2.0], [2.0, 4.0]])
        feature = hypothesis_variance(values)
        depths = np.array(
            [
                [[1.0, 0.0], [3.0, np.nan]],
                [[3.0, 2.0], [5.0, 4.0]],
            ]
        )
        depth = depth_hypothesis_variance(depths)

        self.assertAlmostEqual(feature["mean_variance"], 1.0)
        self.assertEqual(feature["element_count"], 2)
        self.assertEqual(depth["valid_element_count"], 2)
        self.assertAlmostEqual(depth["valid_fraction"], 0.5)

    def test_cycle_error_and_real_support(self):
        error = cycle_reprojection_error(
            np.array([1.0, 3.0, np.nan]),
            np.array([2.0, 1.0, 5.0]),
        )
        support = real_support_ratio([2.0, 1.0], [1.0])

        self.assertAlmostEqual(error["mae"], 1.5)
        self.assertAlmostEqual(error["rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(support, 0.75)

    def test_rank_metrics_are_tie_aware(self):
        self.assertAlmostEqual(
            spearman_correlation([1, 2, 3], [10, 20, 30]),
            1.0,
        )
        self.assertAlmostEqual(binary_auroc([0, 1], [0.2, 0.8]), 1.0)
        self.assertAlmostEqual(binary_auroc([0, 1], [0.5, 0.5]), 0.5)

    def test_calibration_and_equal_count_gain_bins(self):
        calibration = expected_calibration_error(
            [0, 0, 1, 1],
            [0.1, 0.2, 0.8, 0.9],
            bin_count=2,
        )
        bins = reliability_binned_gain(
            [0.1, 0.2, 0.8, 0.9],
            [-1.0, -0.5, 0.5, 1.0],
            bin_count=2,
        )

        self.assertAlmostEqual(calibration["ece"], 0.15)
        self.assertEqual([item["count"] for item in bins], [2, 2])
        self.assertEqual(bins[0]["beneficial_rate"], 0.0)
        self.assertEqual(bins[1]["beneficial_rate"], 1.0)

    def test_report_supports_error_direction_and_cli(self):
        records = [
            {"cycle_error": 0.1, "gain": 1.0, "probability": 0.9},
            {"cycle_error": 0.2, "gain": 0.5, "probability": 0.8},
            {"cycle_error": 0.8, "gain": -0.5, "probability": 0.2},
            {"cycle_error": 0.9, "gain": -1.0, "probability": 0.1},
        ]
        report = build_reliability_report(
            records,
            score_field="cycle_error",
            gain_field="gain",
            higher_is_reliable=False,
            probability_field="probability",
            bin_count=2,
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "records.json"
            output_path = root / "report.json"
            input_path.write_text(json.dumps(records), encoding="utf-8")
            result = main(
                [
                    "--input",
                    str(input_path),
                    "--score-field",
                    "cycle_error",
                    "--gain-field",
                    "gain",
                    "--lower-is-reliable",
                    "--probability-field",
                    "probability",
                    "--bin-count",
                    "2",
                    "--output",
                    str(output_path),
                ]
            )

        self.assertAlmostEqual(report["beneficial_auroc"], 1.0)
        self.assertAlmostEqual(report["spearman_gain"], 1.0)
        self.assertEqual(result, 0)

    def test_degenerate_rank_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "constant"):
            spearman_correlation([1, 1], [0, 1])
        with self.assertRaisesRegex(ValueError, "both"):
            binary_auroc([1, 1], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
