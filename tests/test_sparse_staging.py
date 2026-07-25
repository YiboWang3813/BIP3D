import json
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.oracle_manifest import (
    OracleQuerySelection,
    RealViewOracleManifest,
)
from projects.sparse_grounding.protocol import (
    CameraGraphStats,
    SparseSceneProtocol,
    ViewSelection,
)
from projects.sparse_grounding.staging import (
    StagingManifest,
    build_staging_manifest,
    execute_staging,
    resolve_data_path,
)
from tools.build_sparse_staging_manifest import main as build_main


def write_protocol(path: Path) -> None:
    protocol = SparseSceneProtocol(
        scene_id="3rscan%2Fscene",
        dataset="3rscan",
        protocol_version="test",
        trajectory_type="local_connected",
        seed=1,
        selections=(
            ViewSelection(1, ("3rscan/scene/frame-0.color.jpg",)),
            ViewSelection(
                2,
                (
                    "3rscan/scene/frame-0.color.jpg",
                    "3rscan/scene/frame-1.color.jpg",
                ),
            ),
        ),
        candidate_heldout_frame_ids=(
            "3rscan/scene/frame-2.color.jpg",
        ),
        camera_graph_stats=CameraGraphStats(3, 2, 0.1, 0.2, 5.0),
    )
    protocol.dump(path)


def write_info(path: Path) -> None:
    images = [
        {
            "img_path": f"3rscan/scene/frame-{index}.color.jpg",
            "depth_path": f"3rscan/scene/frame-{index}.depth.pgm",
        }
        for index in range(3)
    ]
    with path.open("wb") as stream:
        pickle.dump(
            {
                "data_list": [
                    {"sample_idx": "3rscan/scene", "images": images}
                ]
            },
            stream,
        )


class SparseStagingTest(unittest.TestCase):
    def build_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        info = root / "info.pkl"
        protocol_dir = root / "protocols"
        protocol_dir.mkdir()
        write_info(info)
        write_protocol(protocol_dir / "scene.json")
        (protocol_dir / "generation_summary.json").write_text(
            json.dumps({"success_count": 1}),
            encoding="utf-8",
        )
        oracle = root / "oracle.json"
        RealViewOracleManifest(
            policy="random_real",
            base_view_budget=2,
            oracle_view_budget=1,
            trajectory_type="local_connected",
            records=(
                OracleQuerySelection(
                    "query-0",
                    "3rscan/scene",
                    ("3rscan/scene/frame-2.color.jpg",),
                ),
            ),
        ).dump(oracle)
        return info, protocol_dir, oracle

    def test_build_round_trip_uses_protocol_and_oracle_union(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            info, protocol_dir, oracle = self.build_fixture(root)
            manifest = build_staging_manifest(
                dataset="3rscan",
                info_files=(info,),
                protocol_dirs=(protocol_dir,),
                oracle_files=(oracle,),
            )
            output = root / "manifest.json"
            manifest.dump(output)
            loaded = StagingManifest.load(output)

        self.assertEqual(len(loaded.files), 6)
        self.assertEqual(loaded.modality_counts, {"depth": 3, "rgb": 3})

    def test_copy_is_resumable_and_accepts_dataset_source_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            info, protocol_dir, oracle = self.build_fixture(root)
            manifest = build_staging_manifest(
                dataset="3rscan",
                info_files=(info,),
                protocol_dirs=(protocol_dir,),
                oracle_files=(oracle,),
            )
            source = root / "source" / "3rscan"
            destination = root / "destination"
            for entry in manifest.files:
                path = resolve_data_path(
                    source, entry.relative_path, "3rscan"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(entry.relative_path.encode())

            dry_run = execute_staging(
                manifest,
                source_root=source,
                destination_root=destination,
            )
            copied = execute_staging(
                manifest,
                source_root=source,
                destination_root=destination,
                execute=True,
            )
            repeated = execute_staging(
                manifest,
                source_root=source,
                destination_root=destination,
                execute=True,
            )

        self.assertEqual(dry_run["needs_copy"], 6)
        self.assertEqual(dry_run["copied"], 0)
        self.assertEqual(copied["copied"], 6)
        self.assertEqual(repeated["already_complete"], 6)
        self.assertEqual(repeated["copied"], 0)

    def test_missing_annotation_frame_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            info, protocol_dir, _ = self.build_fixture(root)
            value = pickle.loads(info.read_bytes())
            value["data_list"][0]["images"].pop()
            info.write_bytes(pickle.dumps(value))
            oracle = root / "bad-oracle.json"
            RealViewOracleManifest(
                policy="random_real",
                base_view_budget=2,
                oracle_view_budget=1,
                trajectory_type="local_connected",
                records=(
                    OracleQuerySelection(
                        "query-0",
                        "3rscan/scene",
                        ("3rscan/scene/frame-2.color.jpg",),
                    ),
                ),
            ).dump(oracle)

            with self.assertRaisesRegex(ValueError, "absent"):
                build_staging_manifest(
                    dataset="3rscan",
                    info_files=(info,),
                    protocol_dirs=(protocol_dir,),
                    oracle_files=(oracle,),
                )

    def test_manifest_rejects_tampered_counts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            info, protocol_dir, oracle = self.build_fixture(root)
            manifest = build_staging_manifest(
                dataset="3rscan",
                info_files=(info,),
                protocol_dirs=(protocol_dir,),
                oracle_files=(oracle,),
            )
            value = manifest.to_dict()
            value["file_count"] = 99
            path = root / "bad.json"
            path.write_text(json.dumps(value), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "file_count"):
                StagingManifest.load(path)

    def test_build_cli_writes_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            info, protocol_dir, oracle = self.build_fixture(root)
            output = root / "staging.json"

            result = build_main(
                [
                    "--dataset",
                    "3rscan",
                    "--info-file",
                    str(info),
                    "--protocol-dir",
                    str(protocol_dir),
                    "--oracle-file",
                    str(oracle),
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(result, 0)
            self.assertEqual(len(StagingManifest.load(output).files), 6)


if __name__ == "__main__":
    unittest.main()
