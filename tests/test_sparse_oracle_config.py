import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class SparseOracleConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from mmengine.config import Config
        except ImportError as error:
            raise unittest.SkipTest(str(error)) from error
        cls.Config = Config

    def test_environment_selects_oracle_manifest_and_policy(self):
        environment = {
            "SPARSE_ORACLE_MANIFEST": "/oracles/random-k3-m4.json",
            "SPARSE_ORACLE_POLICY": "random_real",
            "SPARSE_MISSING_ORACLE": "skip",
        }
        with patch.dict(os.environ, environment, clear=False):
            config = self.Config.fromfile(
                REPO_ROOT / "configs/sparse_grounding_3rscan_oracle.py"
            )

        dataset = config.test_dataloader.dataset
        self.assertEqual(dataset.type, "OracleProtocolGroundingDataset")
        self.assertEqual(
            dataset.oracle_manifest,
            "/oracles/random-k3-m4.json",
        )
        self.assertEqual(dataset.expected_oracle_policy, "random_real")
        self.assertEqual(dataset.missing_oracle, "skip")


if __name__ == "__main__":
    unittest.main()
