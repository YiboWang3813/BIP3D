import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.oracle_manifest import (
    OracleQuerySelection,
    RealViewOracleManifest,
)


def make_manifest():
    return RealViewOracleManifest(
        policy="random_real",
        base_view_budget=3,
        oracle_view_budget=2,
        trajectory_type="global_fps",
        records=(
            OracleQuerySelection(
                query_id="vg.json:0",
                scan_id="3rscan/scene",
                frame_ids=("heldout-0.jpg", "heldout-1.jpg"),
            ),
        ),
    )


class OracleManifestTest(unittest.TestCase):
    def test_round_trip_is_stable(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "oracle.json"
            make_manifest().dump(path)

            loaded = RealViewOracleManifest.load(path)

        self.assertEqual(loaded, make_manifest())
        self.assertEqual(
            loaded.records_by_query_id["vg.json:0"].scan_id,
            "3rscan/scene",
        )

    def test_rejects_selection_over_budget(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            RealViewOracleManifest(
                policy="random_real",
                base_view_budget=3,
                oracle_view_budget=1,
                trajectory_type="global_fps",
                records=make_manifest().records,
            )

    def test_rejects_duplicate_query_ids(self):
        record = make_manifest().records[0]
        with self.assertRaisesRegex(ValueError, "unique, sorted"):
            RealViewOracleManifest(
                policy="random_real",
                base_view_budget=3,
                oracle_view_budget=2,
                trajectory_type="global_fps",
                records=(record, record),
            )


if __name__ == "__main__":
    unittest.main()
