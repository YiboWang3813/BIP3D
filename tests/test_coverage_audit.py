import json
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.coverage_audit import (
    compute_scene_target_coverages,
    run_coverage_audit,
)
from projects.sparse_grounding.protocol import (
    CameraGraphStats,
    SparseSceneProtocol,
    ViewSelection,
)


SCAN_ID = "3rscan/scene"
FRAME_1 = "3rscan/scene/sequence/1.color.jpg"
FRAME_2 = "3rscan/scene/sequence/2.color.jpg"


def make_scene():
    return {
        "sample_idx": SCAN_ID,
        "axis_align_matrix": np.eye(4),
        "depth_cam2img": np.eye(4),
        "instances": [
            {
                "bbox_id": 7,
                "bbox_3d": [0, 0, 1.5, 1, 1, 3, 0, 0, 0],
            }
        ],
        "images": [
            {
                "img_path": FRAME_1,
                "depth_path": "depth-1.pgm",
                "cam2global": np.eye(4),
                "visible_instance_ids": [0],
            },
            {
                "img_path": FRAME_2,
                "depth_path": "depth-2.pgm",
                "cam2global": np.eye(4),
                "visible_instance_ids": [0],
            },
        ],
    }


def depth_loader(path: Path):
    return np.array([[1000 if path.name == "depth-1.pgm" else 2000]])


class SceneCoverageTest(unittest.TestCase):
    def test_computes_unique_target_coverage(self):
        results = compute_scene_target_coverages(
            make_scene(),
            target_ids=[7],
            selected_frame_ids=[FRAME_1],
            data_root=Path("/unused"),
            voxel_size_m=0.5,
            depth_loader=depth_loader,
        )

        self.assertEqual(results[7]["coverage"], 0.5)
        self.assertEqual(results[7]["coverage_band"], "moderate")
        self.assertEqual(results[7]["sparse_annotated_visible_frames"], 1)
        self.assertEqual(results[7]["reference_annotated_visible_frames"], 2)
        self.assertEqual(results[7]["sparse_occupied_frames"], 1)
        self.assertEqual(results[7]["reference_occupied_frames"], 2)

    def test_missing_protocol_frame_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "absent from scene"):
            compute_scene_target_coverages(
                make_scene(),
                target_ids=[7],
                selected_frame_ids=["missing"],
                data_root=Path("/unused"),
                depth_loader=depth_loader,
            )


class CoverageAuditTest(unittest.TestCase):
    def test_deduplicates_targets_and_expands_query_records(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            info_file = root / "info.pkl"
            with info_file.open("wb") as stream:
                pickle.dump({"data_list": [make_scene()]}, stream)
            vg_file = root / "vg.json"
            vg_file.write_text(
                json.dumps(
                    [
                        {
                            "scan_id": SCAN_ID,
                            "target_id": 7,
                            "target": "object",
                            "text": "first query",
                            "distractor_ids": [],
                        },
                        {
                            "scan_id": SCAN_ID,
                            "target_id": 7,
                            "target": "object",
                            "text": "second query",
                            "distractor_ids": [8],
                        },
                    ]
                )
            )
            protocol_dir = root / "protocols"
            protocol_dir.mkdir()
            SparseSceneProtocol(
                scene_id="3rscan%2Fscene",
                dataset="synthetic",
                protocol_version="v1",
                trajectory_type="global_fps",
                seed=0,
                selections=(ViewSelection(1, (FRAME_1,)),),
                candidate_heldout_frame_ids=(FRAME_2,),
                camera_graph_stats=CameraGraphStats(2, 1, 1, 1, 1),
            ).dump(protocol_dir / "3rscan%2Fscene.json")

            result = run_coverage_audit(
                data_root=root,
                info_file=info_file,
                vg_file=vg_file,
                protocol_dir=protocol_dir,
                budget=1,
                source_datasets=["3rscan"],
                voxel_size_m=0.5,
                depth_loader=depth_loader,
            )

        self.assertEqual(result["summary"]["processed_scene_count"], 1)
        self.assertEqual(result["summary"]["unique_target_count"], 1)
        self.assertEqual(result["summary"]["query_count"], 2)
        self.assertEqual(result["summary"]["coverage_band_counts"], {"moderate": 2})
        self.assertEqual(
            result["summary"]["unique_target_coverage_band_counts"],
            {"moderate": 1},
        )
        self.assertEqual([record["query_index"] for record in result["records"]], [0, 1])


if __name__ == "__main__":
    unittest.main()
