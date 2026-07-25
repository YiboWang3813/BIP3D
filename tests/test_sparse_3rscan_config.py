import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class Sparse3RScanConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from mmengine.config import Config
        except ImportError as error:
            raise unittest.SkipTest(str(error)) from error
        cls.Config = Config

    def test_environment_selects_protocol_and_budget(self):
        environment = {
            "SPARSE_PROTOCOL_DIR": "/protocols/global",
            "SPARSE_VIEW_BUDGET": "5",
            "SPARSE_TRAJECTORY_TYPE": "global_fps",
        }
        with patch.dict(os.environ, environment, clear=False):
            config = self.Config.fromfile(
                REPO_ROOT / "configs/sparse_grounding_3rscan.py"
            )

        dataset = config.test_dataloader.dataset
        self.assertEqual(dataset.type, "SparseProtocolGroundingDataset")
        self.assertEqual(dataset.part, ["3rscan"])
        self.assertEqual(dataset.protocol_dir, "/protocols/global")
        self.assertEqual(dataset.view_budget, 5)
        self.assertEqual(
            dataset.ann_file,
            "embodiedscan/embodiedscan_infos_val.pkl",
        )
        self.assertEqual(config.test_evaluator.type, "GroundingMetric")
        self.assertFalse(config.test_evaluator.get("format_only", False))

    def test_custom_dataset_import_is_required(self):
        config = self.Config.fromfile(
            REPO_ROOT / "configs/sparse_grounding_3rscan.py"
        )

        self.assertEqual(
            config.custom_imports.imports,
            ["projects.sparse_grounding.protocol_dataset"],
        )


if __name__ == "__main__":
    unittest.main()
