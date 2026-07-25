import json
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.protocol import (
    CameraGraphStats,
    SparseSceneProtocol,
    ViewSelection,
)
from tools.build_sparse_smoke_fixture import build_fixture


def make_protocol(path: Path) -> None:
    SparseSceneProtocol(
        scene_id="3rscan%2Fscene",
        dataset="embodiedscan-v1-val",
        protocol_version="v1",
        trajectory_type="global_fps",
        seed=1,
        selections=(
            ViewSelection(1, ("3rscan/scene/sequence/1.color.jpg",)),
            ViewSelection(
                2,
                (
                    "3rscan/scene/sequence/1.color.jpg",
                    "3rscan/scene/sequence/0.color.jpg",
                ),
            ),
        ),
        candidate_heldout_frame_ids=(),
        camera_graph_stats=CameraGraphStats(2, 1, 0.5, 0.5, 5),
    ).dump(path)


class SmokeFixtureTest(unittest.TestCase):
    def test_builds_minimal_atomic_fixture(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "source"
            scene_root = data_root / "3rscan/scene/sequence"
            scene_root.mkdir(parents=True)
            images = []
            for index in range(3):
                image_path = f"3rscan/scene/sequence/{index}.color.jpg"
                depth_path = f"3rscan/scene/sequence/{index}.depth.pgm"
                (data_root / image_path).write_bytes(f"rgb-{index}".encode())
                (data_root / depth_path).write_bytes(f"depth-{index}".encode())
                images.append(
                    {"img_path": image_path, "depth_path": depth_path}
                )

            info_file = root / "embodiedscan_infos_val.pkl"
            with info_file.open("wb") as stream:
                pickle.dump(
                    {
                        "metainfo": {"categories": {}},
                        "data_list": [
                            {
                                "sample_idx": "3rscan/scene",
                                "images": images,
                            }
                        ],
                    },
                    stream,
                )
            vg_file = root / "embodiedscan_val_vg_all.json"
            vg_file.write_text(
                json.dumps(
                    [
                        {"scan_id": "other", "text": "skip"},
                        {"scan_id": "3rscan/scene", "text": "first"},
                        {"scan_id": "3rscan/scene", "text": "second"},
                    ]
                )
            )
            protocol_file = root / "3rscan%2Fscene.json"
            make_protocol(protocol_file)
            output = root / "fixture"

            manifest = build_fixture(
                data_root=data_root,
                info_file=info_file,
                vg_file=vg_file,
                protocol_file=protocol_file,
                output_dir=output,
                scan_id="3rscan/scene",
                budget=2,
                max_queries=1,
            )

            with (output / "embodiedscan" / info_file.name).open("rb") as stream:
                fixture_info = pickle.load(stream)
            fixture_vg = json.loads(
                (output / "embodiedscan" / vg_file.name).read_text()
            )

        self.assertEqual(manifest["query_count"], 1)
        self.assertEqual(len(fixture_info["data_list"][0]["images"]), 2)
        self.assertEqual(fixture_vg[0]["text"], "first")
        self.assertEqual(len(manifest["files"]), 7)

    def test_existing_output_is_rejected(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(FileExistsError):
                build_fixture(
                    data_root=output,
                    info_file=output / "missing.pkl",
                    vg_file=output / "missing.json",
                    protocol_file=output / "missing-protocol.json",
                    output_dir=output,
                    scan_id="3rscan/scene",
                    budget=1,
                    max_queries=1,
                )


if __name__ == "__main__":
    unittest.main()
