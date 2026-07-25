import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.embodiedscan_adapter import (
    encode_scene_id,
    load_embodiedscan_pose_manifest,
)
from projects.sparse_grounding.pose_manifest import PoseManifest
from tools.export_embodiedscan_poses import main


def transform(x=0.0):
    matrix = np.eye(4)
    matrix[0, 3] = x
    return matrix


def annotation():
    alignment = transform(10)
    noisy_pose = transform(2)
    noisy_pose[0, 0] += 2e-6
    return {
        "metainfo": {},
        "data_list": [
            {
                "sample_idx": "scannet/scene0000_00",
                "axis_align_matrix": alignment,
                "images": [
                    {
                        "img_path": "scannet/posed_images/scene0000_00/1.jpg",
                        "cam2global": noisy_pose,
                    },
                    {
                        "img_path": "scannet/posed_images/scene0000_00/0.jpg",
                        "cam2global": transform(1),
                    },
                ],
            },
            {
                "sample_idx": "3rscan/abc",
                "axis_align_matrix": np.eye(4),
                "images": [
                    {
                        "img_path": "3rscan/abc/frame-0.color.jpg",
                        "cam2global": transform(3),
                    }
                ],
            },
        ],
    }


def write_annotation(path: Path) -> None:
    with path.open("wb") as stream:
        pickle.dump(annotation(), stream)


class EmbodiedScanAdapterTest(unittest.TestCase):
    def test_scene_id_encoding_is_reversible(self):
        original = "matterport3d/a/region0"
        encoded = encode_scene_id(original)

        self.assertNotIn("/", encoded)
        self.assertEqual(unquote(encoded), original)

    def test_alignment_filtering_and_rotation_cleanup(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "infos.pkl"
            write_annotation(path)

            manifest = load_embodiedscan_pose_manifest(
                path,
                dataset_name="test",
                source_datasets=("scannet",),
            )

        self.assertEqual(len(manifest.scenes), 1)
        scene = manifest.scenes[0]
        self.assertEqual(
            [frame.frame_id for frame in scene.frames],
            sorted(frame.frame_id for frame in scene.frames),
        )
        self.assertAlmostEqual(scene.frames[0].pose.center_world[0], 11)
        self.assertAlmostEqual(scene.frames[1].pose.center_world[0], 12)

    def test_missing_requested_scene_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "infos.pkl"
            write_annotation(path)

            with self.assertRaisesRegex(ValueError, "not found"):
                load_embodiedscan_pose_manifest(
                    path,
                    dataset_name="test",
                    scene_ids=("scannet/missing",),
                )

    def test_excessively_nonrigid_pose_is_rejected(self):
        value = annotation()
        value["data_list"][0]["images"][0]["cam2global"][0, 0] = 1.1
        with TemporaryDirectory() as directory:
            path = Path(directory) / "infos.pkl"
            with path.open("wb") as stream:
                pickle.dump(value, stream)

            with self.assertRaisesRegex(ValueError, "excessive"):
                load_embodiedscan_pose_manifest(
                    path,
                    dataset_name="test",
                    max_scenes=1,
                )

    def test_cli_output_round_trip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            info_path = root / "embodiedscan_infos_val.pkl"
            output_path = root / "poses.json"
            write_annotation(info_path)

            return_code = main(
                [
                    "--info-file",
                    str(info_path),
                    "--output",
                    str(output_path),
                    "--max-scenes",
                    "1",
                ]
            )
            loaded = PoseManifest.load(output_path)

        self.assertEqual(return_code, 0)
        self.assertEqual(loaded.dataset, "embodiedscan-v1-val")
        self.assertEqual(len(loaded.scenes), 1)


if __name__ == "__main__":
    unittest.main()
