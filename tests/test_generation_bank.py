import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.generation_bank import (
    GenerationBank,
    GenerationHypothesis,
    GeneratorIdentity,
    build_generation_bank,
    validate_generation_cache,
)
from projects.sparse_grounding.geometry import CameraPose
from projects.sparse_grounding.pose_manifest import (
    FramePose,
    PoseManifest,
    ScenePoses,
)
from projects.sparse_grounding.protocol import (
    CameraGraphStats,
    SparseSceneProtocol,
    ViewSelection,
)
from tools.build_generation_bank import main as build_main


FRAMES = tuple(f"3rscan/scene/frame-{index}.jpg" for index in range(6))


def write_fixture(root: Path) -> tuple[Path, Path]:
    pose_path = root / "poses.json"
    frames = []
    for index, frame_id in enumerate(FRAMES):
        transform = np.eye(4)
        transform[0, 3] = index
        frames.append(FramePose(frame_id, CameraPose(transform)))
    PoseManifest(
        dataset="embodiedscan-v1-val",
        pose_convention="camera_to_world",
        scenes=(ScenePoses("3rscan%2Fscene", tuple(frames)),),
    ).dump(pose_path)

    protocol_dir = root / "protocols"
    protocol_dir.mkdir()
    SparseSceneProtocol(
        scene_id="3rscan%2Fscene",
        dataset="embodiedscan-v1-val",
        protocol_version="test",
        trajectory_type="local_connected",
        seed=1,
        selections=(
            ViewSelection(3, FRAMES[:3]),
            ViewSelection(5, FRAMES[:5]),
        ),
        candidate_heldout_frame_ids=(FRAMES[5],),
        camera_graph_stats=CameraGraphStats(6, 5, 1.0, 1.0, 0.0),
    ).dump(protocol_dir / "scene.json")
    (protocol_dir / "generation_summary.json").write_text(
        json.dumps({"success_count": 1}),
        encoding="utf-8",
    )
    return pose_path, protocol_dir


class GenerationBankTest(unittest.TestCase):
    def test_bank_is_deterministic_and_round_trips(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pose_path, protocol_dir = write_fixture(root)
            kwargs = {
                "pose_manifest": pose_path,
                "protocol_dir": protocol_dir,
                "dataset": "3rscan",
                "trajectory_type": "local_connected",
                "base_view_budget": 5,
                "candidate_budget": 12,
                "hypothesis_count": 3,
                "generator": GeneratorIdentity("seva", "model", "abc123"),
            }
            first = build_generation_bank(**kwargs)
            second = build_generation_bank(**kwargs)
            output = root / "bank.json"
            first.dump(output)
            loaded = GenerationBank.load(output)

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(loaded.to_dict(), first.to_dict())
        candidate = loaded.scenes[0].candidates[0]
        self.assertEqual(candidate.conditioning_frame_ids, FRAMES[:5])
        self.assertEqual(len(candidate.hypotheses), 3)

    def test_pending_bank_validates_but_is_not_complete(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pose_path, protocol_dir = write_fixture(root)
            bank = build_generation_bank(
                pose_manifest=pose_path,
                protocol_dir=protocol_dir,
                dataset="3rscan",
                trajectory_type="local_connected",
                base_view_budget=5,
                candidate_budget=12,
                hypothesis_count=2,
                generator=GeneratorIdentity("seva", "model", "abc123"),
            )
            pending = validate_generation_cache(bank, cache_root=root)
            complete = validate_generation_cache(
                bank, cache_root=root, require_complete=True
            )

        self.assertTrue(pending["ok"])
        self.assertEqual(pending["hypothesis_counts"]["pending"], 2)
        self.assertFalse(complete["ok"])
        self.assertEqual(complete["issue_count"], 2)

    def test_completed_cache_requires_nonempty_output(self):
        hypothesis = GenerationHypothesis(
            sample_id="sample_00",
            seed=1,
            status="completed",
            rgb_path="scene/camera/sample_00.png",
            depth_path=None,
            error=None,
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pose_path, protocol_dir = write_fixture(root)
            bank = build_generation_bank(
                pose_manifest=pose_path,
                protocol_dir=protocol_dir,
                dataset="3rscan",
                trajectory_type="local_connected",
                base_view_budget=5,
                candidate_budget=12,
                hypothesis_count=1,
                generator=GeneratorIdentity("seva", "model", "abc123"),
            )
            candidate = bank.scenes[0].candidates[0]
            value = bank.to_dict()
            value["scenes"][0]["candidates"][0]["hypotheses"] = [
                hypothesis.to_dict()
            ]
            completed = GenerationBank.from_dict(value)
            missing = validate_generation_cache(completed, cache_root=root)
            output = root / hypothesis.rgb_path
            output.parent.mkdir(parents=True)
            output.write_bytes(b"image")
            present = validate_generation_cache(completed, cache_root=root)

        self.assertEqual(candidate.hypotheses[0].status, "pending")
        self.assertFalse(missing["ok"])
        self.assertTrue(present["ok"])

    def test_invalid_status_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pending"):
            GenerationHypothesis(
                sample_id="sample_00",
                seed=1,
                status="pending",
                rgb_path="output.png",
                depth_path=None,
                error=None,
            )

    def test_cli_builds_plan(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pose_path, protocol_dir = write_fixture(root)
            output = root / "bank.json"
            result = build_main(
                [
                    "--pose-manifest",
                    str(pose_path),
                    "--protocol-dir",
                    str(protocol_dir),
                    "--dataset",
                    "3rscan",
                    "--trajectory-type",
                    "local_connected",
                    "--generator-name",
                    "seva",
                    "--generator-checkpoint",
                    "model",
                    "--generator-revision",
                    "abc123",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(result, 0)
            self.assertEqual(len(GenerationBank.load(output).scenes), 1)


if __name__ == "__main__":
    unittest.main()
