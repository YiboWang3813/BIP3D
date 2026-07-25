import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.protocol import (
    CameraGraphStats,
    SparseSceneProtocol,
    ViewSelection,
)


def make_protocol():
    return SparseSceneProtocol(
        scene_id="3rscan%2Fscene",
        dataset="embodiedscan-v1-val",
        protocol_version="v1",
        trajectory_type="global_fps",
        seed=7,
        selections=(
            ViewSelection(1, ("frame-2.jpg",)),
            ViewSelection(2, ("frame-2.jpg", "frame-0.jpg")),
        ),
        candidate_heldout_frame_ids=("frame-1.jpg",),
        camera_graph_stats=CameraGraphStats(3, 2, 0.5, 1.0, 10.0),
    )


class ProtocolSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from projects.sparse_grounding.protocol_dataset import (
                load_protocol_frame_ids,
                protocol_path,
                select_scene_frames,
            )
        except ImportError as error:
            raise unittest.SkipTest(str(error)) from error
        cls.load_protocol_frame_ids = staticmethod(load_protocol_frame_ids)
        cls.protocol_path = staticmethod(protocol_path)
        cls.select_scene_frames = staticmethod(select_scene_frames)

    def test_protocol_path_encodes_scan_id(self):
        path = self.protocol_path(Path("/protocols"), "3rscan/scene")

        self.assertEqual(path.name, "3rscan%2Fscene.json")

    def test_loads_exact_budget_and_validates_metadata(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            make_protocol().dump(path)

            frame_ids = self.load_protocol_frame_ids(
                path,
                budget=2,
                expected_dataset="embodiedscan-v1-val",
                expected_trajectory_type="global_fps",
                expected_protocol_version="v1",
            )

        self.assertEqual(frame_ids, ("frame-2.jpg", "frame-0.jpg"))

    def test_metadata_mismatch_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            make_protocol().dump(path)

            with self.assertRaisesRegex(ValueError, "trajectory_type"):
                self.load_protocol_frame_ids(
                    path,
                    budget=2,
                    expected_trajectory_type="local_connected",
                )

    def test_selects_all_aligned_multiview_fields_in_protocol_order(self):
        scene = {
            "images": [
                {"img_path": f"frame-{index}.jpg"} for index in range(3)
            ],
            "img_path": [f"rgb-{index}" for index in range(3)],
            "depth_img_path": [f"depth-{index}" for index in range(3)],
            "depth2img": {
                "extrinsic": [f"ext-{index}" for index in range(3)],
                "intrinsic": [f"int-{index}" for index in range(3)],
            },
            "ann_info": {
                "visible_instance_masks": [
                    np.array([index], dtype=bool) for index in range(3)
                ]
            },
        }

        selected = self.select_scene_frames(
            scene,
            ("frame-2.jpg", "frame-0.jpg"),
        )

        self.assertEqual(selected["img_path"], ["rgb-2", "rgb-0"])
        self.assertEqual(selected["depth_img_path"], ["depth-2", "depth-0"])
        self.assertEqual(
            selected["depth2img"]["extrinsic"],
            ["ext-2", "ext-0"],
        )
        self.assertEqual(
            selected["depth2img"]["intrinsic"],
            ["int-2", "int-0"],
        )
        self.assertEqual(
            [mask.tolist() for mask in selected["ann_info"]["visible_instance_masks"]],
            [[True], [False]],
        )
        self.assertEqual(len(scene["images"]), 3)

    def test_missing_frame_is_rejected(self):
        scene = {
            "images": [{"img_path": "frame-0.jpg"}],
            "img_path": ["rgb-0"],
            "depth_img_path": ["depth-0"],
            "depth2img": {"extrinsic": ["ext-0"], "intrinsic": np.eye(3)},
        }

        with self.assertRaisesRegex(ValueError, "absent"):
            self.select_scene_frames(scene, ("missing.jpg",))


if __name__ == "__main__":
    unittest.main()
