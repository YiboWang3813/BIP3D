import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.experiment_matrix import (
    build_experiment_matrix,
    main,
)


class SparseExperimentMatrixTest(unittest.TestCase):
    def make_inputs(self, root: Path):
        checkpoint = root / "model.pth"
        checkpoint.write_bytes(b"checkpoint")
        config = root / "config.py"
        config.write_text("# config\n", encoding="utf-8")
        slurm_script = root / "eval.sbatch"
        slurm_script.write_text("#!/bin/bash\n", encoding="utf-8")
        protocol_root = root / "protocols"
        for directory in ("global", "local"):
            (protocol_root / "3rscan" / directory).mkdir(parents=True)
        return checkpoint, config, slurm_script, protocol_root

    def test_builds_complete_nested_matrix(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, config, script, protocol_root = self.make_inputs(root)

            matrix = build_experiment_matrix(
                protocol_root=protocol_root,
                output_root=root / "outputs",
                checkpoint=checkpoint,
                config=config,
                slurm_script=script,
            )

        self.assertEqual(matrix["experiment_count"], 6)
        identifiers = [
            experiment["experiment_id"]
            for experiment in matrix["experiments"]
        ]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        first = matrix["experiments"][0]
        self.assertEqual(first["view_budget"], 3)
        self.assertEqual(first["trajectory_type"], "global_fps")
        self.assertEqual(
            first["slurm"]["environment"]["SPARSE_VIEW_BUDGET"],
            "3",
        )

    def test_missing_inputs_fail_before_launch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)

            with self.assertRaisesRegex(FileNotFoundError, "inputs are missing"):
                build_experiment_matrix(
                    protocol_root=root / "protocols",
                    output_root=root / "outputs",
                    checkpoint=root / "missing.pth",
                    config=root / "missing.py",
                )

    def test_cli_writes_atomic_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, config, script, protocol_root = self.make_inputs(root)
            output = root / "matrix.json"

            return_code = main(
                [
                    "--protocol-root",
                    str(protocol_root),
                    "--output-root",
                    str(root / "outputs"),
                    "--checkpoint",
                    str(checkpoint),
                    "--config",
                    str(config),
                    "--slurm-script",
                    str(script),
                    "--output",
                    str(output),
                ]
            )
            matrix = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(return_code, 0)
        self.assertEqual(matrix["experiment_count"], 6)


if __name__ == "__main__":
    unittest.main()
