import json
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.oracle_builder import build_real_view_oracle
from projects.sparse_grounding.protocol import (
    CameraGraphStats,
    SparseSceneProtocol,
    ViewSelection,
)


SCAN_ID = "3rscan/scene"
QUERY_FILE_ID = "embodiedscan/embodiedscan_val_vg_all.json"
BASE_FRAMES = tuple(f"3rscan/scene/base-{index}.jpg" for index in range(3))
HELDOUT_FRAMES = tuple(
    f"3rscan/scene/heldout-{index}.jpg" for index in range(4)
)


def scene():
    frames = (*BASE_FRAMES, *HELDOUT_FRAMES)
    return {
        "sample_idx": SCAN_ID,
        "instances": [{"bbox_id": 4, "bbox_label_3d": 1}],
        "images": [
            {
                "img_path": frame,
                "visible_instance_ids": (
                    [0] if frame in {HELDOUT_FRAMES[1], HELDOUT_FRAMES[3]} else []
                ),
            }
            for frame in frames
        ],
    }


def query():
    return {
        "scan_id": SCAN_ID,
        "target_id": 4,
        "target": "chair",
        "text": "the chair",
        "distractor_ids": [],
    }


def protocol():
    return SparseSceneProtocol(
        scene_id="3rscan%2Fscene",
        dataset="embodiedscan-v1-val",
        protocol_version="v1",
        trajectory_type="global_fps",
        seed=1,
        selections=(ViewSelection(3, BASE_FRAMES),),
        candidate_heldout_frame_ids=HELDOUT_FRAMES,
        camera_graph_stats=CameraGraphStats(7, 6, 0.5, 1.0, 10),
    )


class OracleBuilderTest(unittest.TestCase):
    def make_inputs(self, root: Path):
        info = root / "info.pkl"
        with info.open("wb") as stream:
            pickle.dump(
                {
                    "metainfo": {"categories": {"chair": 1}},
                    "data_list": [scene()],
                },
                stream,
            )
        vg = root / "vg.json"
        vg.write_text(json.dumps([query(), query()]), encoding="utf-8")
        protocols = root / "protocols"
        protocols.mkdir()
        protocol().dump(protocols / "3rscan%2Fscene.json")
        return info, vg, protocols

    def build(self, root: Path, policy: str):
        info, vg, protocols = self.make_inputs(root)
        return build_real_view_oracle(
            info_file=info,
            vg_file=vg,
            query_file_id=QUERY_FILE_ID,
            protocol_dir=protocols,
            source_dataset="3rscan",
            trajectory_type="global_fps",
            policy=policy,
            base_view_budget=3,
            oracle_view_budget=2,
            global_seed=7,
        )

    def test_random_real_is_query_independent_within_scene(self):
        with TemporaryDirectory() as directory:
            result = self.build(Path(directory), "random_real")

        self.assertEqual(result.query_count, 2)
        first, second = result.manifest.records
        self.assertEqual(first.frame_ids, second.frame_ids)
        self.assertEqual(len(first.frame_ids), 2)
        self.assertTrue(set(first.frame_ids).issubset(HELDOUT_FRAMES))

    def test_annotation_visible_prioritizes_target_visible_frames(self):
        with TemporaryDirectory() as directory:
            result = self.build(Path(directory), "annotation_visible")

        selected = result.manifest.records[0].frame_ids
        self.assertEqual(selected, (HELDOUT_FRAMES[1], HELDOUT_FRAMES[3]))
        self.assertEqual(result.summary()["unresolved_query_count"], 0)

    def test_max_queries_preserves_original_query_index(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            info, vg, protocols = self.make_inputs(root)
            result = build_real_view_oracle(
                info_file=info,
                vg_file=vg,
                query_file_id=QUERY_FILE_ID,
                protocol_dir=protocols,
                source_dataset="3rscan",
                trajectory_type="global_fps",
                policy="random_real",
                base_view_budget=3,
                oracle_view_budget=2,
                max_queries=1,
            )

        self.assertEqual(
            result.manifest.records[0].query_id,
            f"{QUERY_FILE_ID}:0",
        )


if __name__ == "__main__":
    unittest.main()
