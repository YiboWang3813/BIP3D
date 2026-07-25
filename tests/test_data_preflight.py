import json
import pickle
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.data_preflight import run_preflight
from tools.check_sparse_grounding_data import main


def create_layout(root: Path) -> None:
    for relative in (
        "scannet/posed_images/scene0000_00",
        "3rscan/00d42bed",
        "matterport3d/17DRP5sb8fy",
    ):
        (root / relative).mkdir(parents=True)
    annotation_root = root / "embodiedscan"
    annotation_root.mkdir()
    for split in ("train", "val"):
        info = {
            "data_list": [
                {
                    "sample_idx": f"scannet/scene-{split}",
                    "images": [
                        {
                            "img_path": f"scannet/{split}.jpg",
                            "depth_path": f"scannet/{split}.png",
                        }
                    ],
                }
            ]
        }
        with (annotation_root / f"embodiedscan_infos_{split}.pkl").open(
            "wb"
        ) as stream:
            pickle.dump(info, stream)
        vg = [{"scan_id": f"scannet/scene-{split}", "text": "a chair"}]
        (annotation_root / f"embodiedscan_{split}_vg_all.json").write_text(
            json.dumps(vg),
            encoding="utf-8",
        )


def issue_codes(report):
    return {issue["code"] for issue in report["issues"]}


class DataPreflightTest(unittest.TestCase):
    def test_missing_layout_is_reported(self):
        with TemporaryDirectory() as directory:
            report = run_preflight(Path(directory))

        self.assertFalse(report["ok"])
        self.assertIn("missing_directory", issue_codes(report))
        self.assertIn("missing_file", issue_codes(report))

    def test_basic_check_accepts_complete_layout(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            create_layout(root)

            report = run_preflight(root)

        self.assertTrue(report["ok"])
        self.assertFalse(report["pickle_inspected"])

    def test_deep_check_validates_references_and_scan_ids(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            create_layout(root)

            report = run_preflight(root, inspect_pickle=True)

        self.assertFalse(report["ok"])
        self.assertIn("missing_data_reference", issue_codes(report))
        self.assertNotIn("unknown_vg_scan_id", issue_codes(report))

    def test_deep_check_passes_when_references_exist(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            create_layout(root)
            for split in ("train", "val"):
                (root / "scannet" / f"{split}.jpg").touch()
                (root / "scannet" / f"{split}.png").touch()

            report = run_preflight(root, inspect_pickle=True)

        self.assertTrue(report["ok"])

    def test_unknown_vg_scan_id_is_reported(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            create_layout(root)
            annotation = root / "embodiedscan/embodiedscan_val_vg_all.json"
            annotation.write_text(
                json.dumps([{"scan_id": "scannet/not-in-info"}]),
                encoding="utf-8",
            )

            report = run_preflight(root, inspect_pickle=True)

        self.assertIn("unknown_vg_scan_id", issue_codes(report))

    def test_cli_writes_machine_readable_report(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reports/preflight.json"

            with redirect_stdout(StringIO()):
                return_code = main(
                    [
                        "--data-root",
                        str(root / "missing"),
                        "--json-output",
                        str(output),
                    ]
                )

            self.assertEqual(return_code, 1)
            self.assertFalse(json.loads(output.read_text())["ok"])


if __name__ == "__main__":
    unittest.main()
