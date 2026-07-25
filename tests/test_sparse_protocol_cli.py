import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.pose_manifest import PoseManifest
from projects.sparse_grounding.protocol import SparseSceneProtocol
from projects.sparse_grounding.protocol_cli import generate_protocols, main


def transform_at(x):
    matrix = np.eye(4)
    matrix[0, 3] = x
    return matrix.tolist()


def manifest_dict(scene_sizes=(10,), convention="camera_to_world"):
    scenes = []
    for scene_index, size in enumerate(scene_sizes):
        scenes.append(
            {
                "scene_id": f"scene-{scene_index}",
                "frames": [
                    {
                        "frame_id": f"{frame_index:02d}",
                        "transform": transform_at(frame_index * 0.5),
                    }
                    for frame_index in range(size)
                ],
            }
        )
    return {
        "dataset": "synthetic",
        "pose_convention": convention,
        "scenes": scenes,
    }


class PoseManifestTest(unittest.TestCase):
    def test_json_round_trip_is_stable(self):
        manifest = PoseManifest.from_dict(manifest_dict())

        self.assertEqual(
            PoseManifest.from_dict(manifest.to_dict()).to_json(),
            manifest.to_json(),
        )

    def test_world_to_camera_is_converted(self):
        value = manifest_dict(scene_sizes=(1,), convention="world_to_camera")
        value["scenes"][0]["frames"][0]["transform"] = transform_at(-2)

        manifest = PoseManifest.from_dict(value)

        self.assertAlmostEqual(
            manifest.scenes[0].frames[0].pose.center_world[0],
            2,
        )

    def test_query_specific_fields_are_rejected(self):
        value = manifest_dict(scene_sizes=(1,))
        value["scenes"][0]["query_id"] = "forbidden"

        with self.assertRaisesRegex(ValueError, "query_id"):
            PoseManifest.from_dict(value)

    def test_unsafe_scene_id_is_rejected(self):
        value = manifest_dict(scene_sizes=(1,))
        value["scenes"][0]["scene_id"] = "../escape"

        with self.assertRaisesRegex(ValueError, "safe"):
            PoseManifest.from_dict(value)


class ProtocolGenerationTest(unittest.TestCase):
    def test_generation_is_idempotent(self):
        manifest = PoseManifest.from_dict(manifest_dict())
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            arguments = {
                "trajectory_type": "local_connected",
                "budgets": (3, 5, 8),
                "global_seed": 7,
                "protocol_version": "test-v1",
                "max_translation_m": 0.6,
                "max_rotation_deg": 1,
                "overwrite": False,
            }

            first = generate_protocols(manifest, output_dir, **arguments)
            second = generate_protocols(manifest, output_dir, **arguments)

            self.assertEqual(first["results"][0]["status"], "generated")
            self.assertEqual(second["results"][0]["status"], "unchanged")
            protocol = SparseSceneProtocol.load(output_dir / "scene-0.json")
            self.assertEqual(protocol.dataset, "synthetic")
            self.assertEqual(
                [selection.budget for selection in protocol.selections],
                [3, 5, 8],
            )

    def test_scene_failures_are_recorded_without_hiding_success(self):
        manifest = PoseManifest.from_dict(manifest_dict(scene_sizes=(10, 2)))
        with TemporaryDirectory() as directory:
            summary = generate_protocols(
                manifest,
                Path(directory),
                trajectory_type="local_connected",
                budgets=(3, 5, 8),
                global_seed=7,
                protocol_version="test-v1",
                max_translation_m=0.6,
                max_rotation_deg=1,
                overwrite=False,
            )

            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(summary["failure_count"], 1)
            self.assertEqual(summary["results"][1]["status"], "failed")

    def test_cli_writes_summary_and_returns_failure_status(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "poses.json"
            output_dir = root / "protocols"
            input_path.write_text(
                json.dumps(manifest_dict(scene_sizes=(2,))),
                encoding="utf-8",
            )

            return_code = main(
                [
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(return_code, 1)
            summary = json.loads(
                (output_dir / "generation_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
