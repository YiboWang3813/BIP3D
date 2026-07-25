import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.oracle_experiment_matrix import (
    build_oracle_experiment_matrix,
)
from projects.sparse_grounding.oracle_manifest import (
    OracleQuerySelection,
    RealViewOracleManifest,
)


class OracleExperimentMatrixTest(unittest.TestCase):
    def make_inputs(self, root: Path):
        checkpoint = root / "model.pth"
        checkpoint.write_bytes(b"checkpoint")
        config = root / "oracle.py"
        config.write_text("# config\n", encoding="utf-8")
        script = root / "eval.sbatch"
        script.write_text("#!/bin/bash\n", encoding="utf-8")
        protocols = root / "protocols"
        oracles = root / "oracles"
        oracles.mkdir()
        for trajectory, directory in (
            ("global_fps", "global"),
            ("local_connected", "local"),
        ):
            (protocols / "3rscan" / directory).mkdir(parents=True)
            for policy in ("random_real", "annotation_visible"):
                manifest = RealViewOracleManifest(
                    policy=policy,
                    base_view_budget=5,
                    oracle_view_budget=4,
                    trajectory_type=trajectory,
                    records=(
                        OracleQuerySelection(
                            query_id="vg.json:0",
                            scan_id="3rscan/scene",
                            frame_ids=("heldout.jpg",),
                        ),
                    ),
                )
                manifest.dump(
                    oracles
                    / f"3rscan_{directory}_{policy}_k5_m4.json"
                )
        return checkpoint, config, script, protocols, oracles

    def test_builds_four_validated_oracle_experiments(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, config, script, protocols, oracles = self.make_inputs(
                root
            )

            matrix = build_oracle_experiment_matrix(
                protocol_root=protocols,
                oracle_root=oracles,
                output_root=root / "outputs",
                checkpoint=checkpoint,
                config=config,
                slurm_script=script,
            )

        self.assertEqual(matrix["experiment_count"], 4)
        first = matrix["experiments"][0]
        self.assertEqual(first["method"], "BIP3D-K+M-Real")
        self.assertEqual(
            first["slurm"]["environment"]["SPARSE_ORACLE_POLICY"],
            "random_real",
        )

    def test_rejects_manifest_metadata_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, config, script, protocols, oracles = self.make_inputs(
                root
            )
            wrong = RealViewOracleManifest(
                policy="random_real",
                base_view_budget=3,
                oracle_view_budget=4,
                trajectory_type="global_fps",
                records=(
                    OracleQuerySelection(
                        query_id="vg.json:0",
                        scan_id="3rscan/scene",
                        frame_ids=("heldout.jpg",),
                    ),
                ),
            )
            wrong.dump(oracles / "3rscan_global_random_real_k5_m4.json")

            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                build_oracle_experiment_matrix(
                    protocol_root=protocols,
                    oracle_root=oracles,
                    output_root=root / "outputs",
                    checkpoint=checkpoint,
                    config=config,
                    slurm_script=script,
                )


if __name__ == "__main__":
    unittest.main()
