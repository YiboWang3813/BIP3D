"""Strict pose-manifest schema used before dataset-specific adapters exist."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .geometry import CameraPose


POSE_CONVENTIONS = frozenset({"camera_to_world", "world_to_camera"})


def _require_keys(
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


@dataclass(frozen=True)
class FramePose:
    frame_id: str
    pose: CameraPose

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id must be a non-empty string")
        if not isinstance(self.pose, CameraPose):
            raise ValueError("pose must be a CameraPose")


@dataclass(frozen=True)
class ScenePoses:
    scene_id: str
    frames: tuple[FramePose, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.scene_id, str)
            or not self.scene_id
            or "/" in self.scene_id
            or "\\" in self.scene_id
            or self.scene_id in {".", ".."}
        ):
            raise ValueError("scene_id must be a safe, non-empty filename")
        if not isinstance(self.frames, tuple) or not self.frames:
            raise ValueError("frames must be a non-empty tuple")
        frame_ids = [frame.frame_id for frame in self.frames]
        if frame_ids != sorted(set(frame_ids)):
            raise ValueError("frames must have unique, sorted frame IDs")

    @property
    def poses(self) -> Mapping[str, CameraPose]:
        return MappingProxyType(
            {frame.frame_id: frame.pose for frame in self.frames}
        )


@dataclass(frozen=True)
class PoseManifest:
    dataset: str
    pose_convention: str
    scenes: tuple[ScenePoses, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("dataset must be a non-empty string")
        if self.pose_convention not in POSE_CONVENTIONS:
            raise ValueError(
                f"pose_convention must be one of {sorted(POSE_CONVENTIONS)}"
            )
        if not isinstance(self.scenes, tuple) or not self.scenes:
            raise ValueError("scenes must be a non-empty tuple")
        scene_ids = [scene.scene_id for scene in self.scenes]
        if scene_ids != sorted(set(scene_ids)):
            raise ValueError("scenes must have unique, sorted scene IDs")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PoseManifest":
        _require_keys(
            value,
            {"dataset", "pose_convention", "scenes"},
            "manifest",
        )
        convention = value["pose_convention"]
        if convention not in POSE_CONVENTIONS:
            raise ValueError(
                f"pose_convention must be one of {sorted(POSE_CONVENTIONS)}"
            )
        raw_scenes = value["scenes"]
        if not isinstance(raw_scenes, list):
            raise ValueError("scenes must be a list")

        scenes = []
        for raw_scene in raw_scenes:
            if not isinstance(raw_scene, Mapping):
                raise ValueError("each scene must be an object")
            _require_keys(raw_scene, {"scene_id", "frames"}, "scene")
            raw_frames = raw_scene["frames"]
            if not isinstance(raw_frames, list):
                raise ValueError("scene.frames must be a list")
            frames = []
            for raw_frame in raw_frames:
                if not isinstance(raw_frame, Mapping):
                    raise ValueError("each frame must be an object")
                _require_keys(
                    raw_frame,
                    {"frame_id", "transform"},
                    "frame",
                )
                if convention == "camera_to_world":
                    pose = CameraPose(raw_frame["transform"])
                else:
                    pose = CameraPose.from_world_to_camera(
                        raw_frame["transform"]
                    )
                frames.append(FramePose(raw_frame["frame_id"], pose))
            scenes.append(
                ScenePoses(
                    scene_id=raw_scene["scene_id"],
                    frames=tuple(sorted(frames, key=lambda frame: frame.frame_id)),
                )
            )
        return cls(
            dataset=value["dataset"],
            pose_convention=convention,
            scenes=tuple(sorted(scenes, key=lambda scene: scene.scene_id)),
        )

    @classmethod
    def load(cls, path: Path) -> "PoseManifest":
        decoded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(decoded, Mapping):
            raise ValueError("pose manifest root must be an object")
        return cls.from_dict(decoded)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "pose_convention": self.pose_convention,
            "scenes": [
                {
                    "scene_id": scene.scene_id,
                    "frames": [
                        {
                            "frame_id": frame.frame_id,
                            "transform": frame.pose.camera_to_world.tolist(),
                        }
                        for frame in scene.frames
                    ],
                }
                for scene in self.scenes
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    def dump(self, path: Path) -> None:
        path.write_text(f"{self.to_json()}\n", encoding="utf-8")
