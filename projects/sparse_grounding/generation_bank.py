"""Strict, query-independent generated-view bank planning and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import unquote

from .geometry import CameraPose
from .pose_manifest import PoseManifest
from .protocol import SparseSceneProtocol
from .sampling import derive_scene_seed


SCHEMA_VERSION = "1.0"
HYPOTHESIS_STATUSES = frozenset({"pending", "completed", "failed"})


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} fields mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _optional_cache_path(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be null or a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{field} must be a safe relative path")
    return path.as_posix()


@dataclass(frozen=True)
class GeneratorIdentity:
    name: str
    checkpoint: str
    revision: str

    def __post_init__(self) -> None:
        for field in ("name", "checkpoint", "revision"):
            if not isinstance(getattr(self, field), str) or not getattr(
                self, field
            ):
                raise ValueError(f"generator.{field} must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratorIdentity":
        _exact_keys(value, {"name", "checkpoint", "revision"}, "generator")
        return cls(**value)


@dataclass(frozen=True)
class GenerationHypothesis:
    sample_id: str
    seed: int
    status: str
    rgb_path: str | None
    depth_path: str | None
    error: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if self.status not in HYPOTHESIS_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(HYPOTHESIS_STATUSES)}"
            )
        object.__setattr__(
            self,
            "rgb_path",
            _optional_cache_path(self.rgb_path, "rgb_path"),
        )
        object.__setattr__(
            self,
            "depth_path",
            _optional_cache_path(self.depth_path, "depth_path"),
        )
        if self.error is not None and (
            not isinstance(self.error, str) or not self.error
        ):
            raise ValueError("error must be null or a non-empty string")
        if self.status == "pending" and any(
            value is not None
            for value in (self.rgb_path, self.depth_path, self.error)
        ):
            raise ValueError("pending hypotheses cannot have outputs or errors")
        if self.status == "completed" and (
            self.rgb_path is None or self.error is not None
        ):
            raise ValueError("completed hypotheses require RGB and no error")
        if self.status == "failed" and (
            self.error is None
            or self.rgb_path is not None
            or self.depth_path is not None
        ):
            raise ValueError("failed hypotheses require only an error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "seed": self.seed,
            "status": self.status,
            "rgb_path": self.rgb_path,
            "depth_path": self.depth_path,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationHypothesis":
        _exact_keys(
            value,
            {
                "sample_id",
                "seed",
                "status",
                "rgb_path",
                "depth_path",
                "error",
            },
            "hypothesis",
        )
        return cls(**value)


@dataclass(frozen=True)
class CandidateCamera:
    camera_id: str
    target_frame_id: str
    target_pose: CameraPose
    conditioning_frame_ids: tuple[str, ...]
    hypotheses: tuple[GenerationHypothesis, ...]

    def __post_init__(self) -> None:
        for field in ("camera_id", "target_frame_id"):
            if not isinstance(getattr(self, field), str) or not getattr(
                self, field
            ):
                raise ValueError(f"{field} must be a non-empty string")
        if not isinstance(self.target_pose, CameraPose):
            raise ValueError("target_pose must be a CameraPose")
        if (
            not self.conditioning_frame_ids
            or len(set(self.conditioning_frame_ids))
            != len(self.conditioning_frame_ids)
        ):
            raise ValueError("conditioning frames must be non-empty and unique")
        sample_ids = [item.sample_id for item in self.hypotheses]
        if not sample_ids or sample_ids != sorted(set(sample_ids)):
            raise ValueError("hypotheses must have unique, sorted sample IDs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "target_frame_id": self.target_frame_id,
            "target_camera_to_world": (
                self.target_pose.camera_to_world.tolist()
            ),
            "conditioning_frame_ids": list(self.conditioning_frame_ids),
            "hypotheses": [item.to_dict() for item in self.hypotheses],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateCamera":
        _exact_keys(
            value,
            {
                "camera_id",
                "target_frame_id",
                "target_camera_to_world",
                "conditioning_frame_ids",
                "hypotheses",
            },
            "candidate",
        )
        return cls(
            camera_id=value["camera_id"],
            target_frame_id=value["target_frame_id"],
            target_pose=CameraPose(value["target_camera_to_world"]),
            conditioning_frame_ids=tuple(value["conditioning_frame_ids"]),
            hypotheses=tuple(
                GenerationHypothesis.from_dict(item)
                for item in value["hypotheses"]
            ),
        )


@dataclass(frozen=True)
class GenerationScene:
    scene_id: str
    candidates: tuple[CandidateCamera, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("scene_id must be a non-empty string")
        camera_ids = [item.camera_id for item in self.candidates]
        if not camera_ids or camera_ids != sorted(set(camera_ids)):
            raise ValueError("candidates must have unique, sorted camera IDs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "candidates": [item.to_dict() for item in self.candidates],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationScene":
        _exact_keys(value, {"scene_id", "candidates"}, "scene")
        return cls(
            scene_id=value["scene_id"],
            candidates=tuple(
                CandidateCamera.from_dict(item)
                for item in value["candidates"]
            ),
        )


@dataclass(frozen=True)
class GenerationBank:
    dataset: str
    trajectory_type: str
    base_view_budget: int
    candidate_budget: int
    hypothesis_count: int
    generator: GeneratorIdentity
    scenes: tuple[GenerationScene, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported generation bank schema_version")
        for field in ("dataset", "trajectory_type"):
            if not isinstance(getattr(self, field), str) or not getattr(
                self, field
            ):
                raise ValueError(f"{field} must be a non-empty string")
        for field in (
            "base_view_budget",
            "candidate_budget",
            "hypothesis_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        scene_ids = [scene.scene_id for scene in self.scenes]
        if not scene_ids or scene_ids != sorted(set(scene_ids)):
            raise ValueError("scenes must have unique, sorted scene IDs")
        for scene in self.scenes:
            if len(scene.candidates) > self.candidate_budget:
                raise ValueError("scene exceeds candidate_budget")
            for candidate in scene.candidates:
                if (
                    len(candidate.conditioning_frame_ids)
                    != self.base_view_budget
                ):
                    raise ValueError("candidate conditioning budget mismatch")
                if len(candidate.hypotheses) != self.hypothesis_count:
                    raise ValueError("candidate hypothesis count mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "trajectory_type": self.trajectory_type,
            "base_view_budget": self.base_view_budget,
            "candidate_budget": self.candidate_budget,
            "hypothesis_count": self.hypothesis_count,
            "generator": self.generator.to_dict(),
            "scenes": [scene.to_dict() for scene in self.scenes],
        }

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            f"{json.dumps(self.to_dict(), indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationBank":
        _exact_keys(
            value,
            {
                "schema_version",
                "dataset",
                "trajectory_type",
                "base_view_budget",
                "candidate_budget",
                "hypothesis_count",
                "generator",
                "scenes",
            },
            "bank",
        )
        return cls(
            schema_version=value["schema_version"],
            dataset=value["dataset"],
            trajectory_type=value["trajectory_type"],
            base_view_budget=value["base_view_budget"],
            candidate_budget=value["candidate_budget"],
            hypothesis_count=value["hypothesis_count"],
            generator=GeneratorIdentity.from_dict(value["generator"]),
            scenes=tuple(
                GenerationScene.from_dict(scene) for scene in value["scenes"]
            ),
        )

    @classmethod
    def load(cls, path: Path) -> "GenerationBank":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("generation bank root must be an object")
        return cls.from_dict(value)


def _evenly_spaced(values: tuple[str, ...], limit: int) -> tuple[str, ...]:
    if len(values) <= limit:
        return values
    if limit == 1:
        return values[:1]
    indices = [
        round(index * (len(values) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return tuple(values[index] for index in indices)


def _camera_id(frame_id: str) -> str:
    return hashlib.sha256(frame_id.encode("utf-8")).hexdigest()[:16]


def build_generation_bank(
    *,
    pose_manifest: Path,
    protocol_dir: Path,
    dataset: str,
    trajectory_type: str,
    base_view_budget: int,
    candidate_budget: int,
    hypothesis_count: int,
    generator: GeneratorIdentity,
    global_seed: int = 20260725,
    max_scenes: int | None = None,
) -> GenerationBank:
    """Plan a deterministic scene-level bank from held-out real cameras."""
    poses = PoseManifest.load(pose_manifest)
    if max_scenes is not None and (
        isinstance(max_scenes, bool)
        or not isinstance(max_scenes, int)
        or max_scenes <= 0
    ):
        raise ValueError("max_scenes must be a positive integer")
    pose_scenes = {scene.scene_id: scene for scene in poses.scenes}
    paths = sorted(
        path
        for path in protocol_dir.glob("*.json")
        if path.name != "generation_summary.json"
    )
    if not paths:
        raise ValueError(f"no protocol files found in {protocol_dir}")

    scenes = []
    for path in paths:
        protocol = SparseSceneProtocol.load(path)
        source_dataset = unquote(protocol.scene_id).split("/", 1)[0]
        if source_dataset != dataset:
            continue
        if protocol.dataset != poses.dataset:
            raise ValueError(f"{path}: pose/protocol dataset mismatch")
        if protocol.trajectory_type != trajectory_type:
            raise ValueError(f"{path}: trajectory_type mismatch")
        selections = {
            selection.budget: selection for selection in protocol.selections
        }
        if base_view_budget not in selections:
            raise ValueError(f"{path}: missing budget {base_view_budget}")
        pose_scene = pose_scenes.get(protocol.scene_id)
        if pose_scene is None:
            raise ValueError(f"{path}: scene missing from pose manifest")
        selected_targets = _evenly_spaced(
            protocol.candidate_heldout_frame_ids,
            candidate_budget,
        )
        candidates = []
        for target_frame_id in selected_targets:
            target_pose = pose_scene.poses.get(target_frame_id)
            if target_pose is None:
                raise ValueError(f"{path}: missing pose for {target_frame_id}")
            camera_id = _camera_id(target_frame_id)
            hypotheses = []
            for sample_index in range(hypothesis_count):
                sample_id = f"sample_{sample_index:02d}"
                seed = derive_scene_seed(
                    global_seed,
                    dataset,
                    f"{protocol.scene_id}\0{camera_id}\0{sample_id}",
                )
                hypotheses.append(
                    GenerationHypothesis(
                        sample_id=sample_id,
                        seed=seed,
                        status="pending",
                        rgb_path=None,
                        depth_path=None,
                        error=None,
                    )
                )
            candidates.append(
                CandidateCamera(
                    camera_id=camera_id,
                    target_frame_id=target_frame_id,
                    target_pose=target_pose,
                    conditioning_frame_ids=selections[
                        base_view_budget
                    ].frame_ids,
                    hypotheses=tuple(hypotheses),
                )
            )
        if candidates:
            scenes.append(
                GenerationScene(
                    scene_id=protocol.scene_id,
                    candidates=tuple(
                        sorted(candidates, key=lambda item: item.camera_id)
                    ),
                )
            )
        if max_scenes is not None and len(scenes) >= max_scenes:
            break
    if not scenes:
        raise ValueError(f"no protocol scenes matched source dataset {dataset}")
    return GenerationBank(
        dataset=dataset,
        trajectory_type=trajectory_type,
        base_view_budget=base_view_budget,
        candidate_budget=candidate_budget,
        hypothesis_count=hypothesis_count,
        generator=generator,
        scenes=tuple(sorted(scenes, key=lambda scene: scene.scene_id)),
    )


def validate_generation_cache(
    bank: GenerationBank,
    *,
    cache_root: Path,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Validate status/output consistency against an on-disk cache."""
    counts = {
        status: 0 for status in sorted(HYPOTHESIS_STATUSES)
    }
    issues = []
    for scene in bank.scenes:
        for candidate in scene.candidates:
            for hypothesis in candidate.hypotheses:
                counts[hypothesis.status] += 1
                if require_complete and hypothesis.status != "completed":
                    issues.append(
                        {
                            "scene_id": scene.scene_id,
                            "camera_id": candidate.camera_id,
                            "sample_id": hypothesis.sample_id,
                            "issue": f"status_{hypothesis.status}",
                        }
                    )
                if hypothesis.status != "completed":
                    continue
                for modality, relative in (
                    ("rgb", hypothesis.rgb_path),
                    ("depth", hypothesis.depth_path),
                ):
                    if relative is None:
                        continue
                    path = cache_root / relative
                    if not path.is_file() or path.stat().st_size == 0:
                        issues.append(
                            {
                                "scene_id": scene.scene_id,
                                "camera_id": candidate.camera_id,
                                "sample_id": hypothesis.sample_id,
                                "issue": f"missing_{modality}",
                                "path": str(path),
                            }
                        )
    return {
        "ok": not issues,
        "scene_count": len(bank.scenes),
        "candidate_count": sum(
            len(scene.candidates) for scene in bank.scenes
        ),
        "hypothesis_counts": counts,
        "issue_count": len(issues),
        "issues": issues,
    }
