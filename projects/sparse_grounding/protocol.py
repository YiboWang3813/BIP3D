"""Versioned, query-independent sparse-view protocol schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "1.0"
TRAJECTORY_TYPES = frozenset({"local_connected", "global_fps"})


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{context} fields mismatch: missing={missing}, unknown={unknown}"
        )


def _require_nonnegative_number(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    if value < 0:
        raise ValueError(f"{field} must be nonnegative")


@dataclass(frozen=True)
class CameraGraphStats:
    node_count: int
    edge_count: int
    mean_translation_m: float
    max_translation_m: float
    mean_rotation_deg: float

    def __post_init__(self) -> None:
        for field in ("node_count", "edge_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        for field in (
            "mean_translation_m",
            "max_translation_m",
            "mean_rotation_deg",
        ):
            _require_nonnegative_number(getattr(self, field), field)
        if self.mean_translation_m > self.max_translation_m:
            raise ValueError(
                "mean_translation_m cannot exceed max_translation_m"
            )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "mean_translation_m": self.mean_translation_m,
            "max_translation_m": self.max_translation_m,
            "mean_rotation_deg": self.mean_rotation_deg,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CameraGraphStats":
        expected = {
            "node_count",
            "edge_count",
            "mean_translation_m",
            "max_translation_m",
            "mean_rotation_deg",
        }
        _require_exact_keys(value, expected, "camera_graph_stats")
        return cls(**value)


@dataclass(frozen=True)
class ViewSelection:
    budget: int
    frame_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.budget, bool)
            or not isinstance(self.budget, int)
            or self.budget <= 0
        ):
            raise ValueError("budget must be a positive integer")
        if not isinstance(self.frame_ids, tuple):
            raise ValueError("frame_ids must be an immutable tuple")
        if len(self.frame_ids) != self.budget:
            raise ValueError(
                f"budget {self.budget} requires exactly {self.budget} frames"
            )
        if any(
            not isinstance(frame_id, str) or not frame_id
            for frame_id in self.frame_ids
        ):
            raise ValueError("frame_ids must contain non-empty strings")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("frame_ids must be unique within each selection")

    def to_dict(self) -> dict[str, object]:
        return {"budget": self.budget, "frame_ids": list(self.frame_ids)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ViewSelection":
        _require_exact_keys(value, {"budget", "frame_ids"}, "selection")
        frame_ids = value["frame_ids"]
        if not isinstance(frame_ids, list):
            raise ValueError("selection.frame_ids must be a list")
        return cls(budget=value["budget"], frame_ids=tuple(frame_ids))


@dataclass(frozen=True)
class SparseSceneProtocol:
    scene_id: str
    dataset: str
    protocol_version: str
    trajectory_type: str
    seed: int
    selections: tuple[ViewSelection, ...]
    candidate_heldout_frame_ids: tuple[str, ...]
    camera_graph_stats: CameraGraphStats
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in ("scene_id", "dataset", "protocol_version"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty string")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        if self.trajectory_type not in TRAJECTORY_TYPES:
            raise ValueError(
                f"trajectory_type must be one of {sorted(TRAJECTORY_TYPES)}"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.selections, tuple) or not all(
            isinstance(selection, ViewSelection)
            for selection in self.selections
        ):
            raise ValueError(
                "selections must be an immutable tuple of ViewSelection"
            )
        if not self.selections:
            raise ValueError("selections cannot be empty")

        budgets = [selection.budget for selection in self.selections]
        if budgets != sorted(set(budgets)):
            raise ValueError("selections must have unique, increasing budgets")

        previous_frames: set[str] = set()
        for selection in self.selections:
            current_frames = set(selection.frame_ids)
            if not previous_frames.issubset(current_frames):
                raise ValueError("view selections must be nested by budget")
            previous_frames = current_frames

        heldout = self.candidate_heldout_frame_ids
        if not isinstance(heldout, tuple):
            raise ValueError(
                "candidate_heldout_frame_ids must be an immutable tuple"
            )
        if any(not isinstance(frame_id, str) or not frame_id for frame_id in heldout):
            raise ValueError(
                "candidate_heldout_frame_ids must contain non-empty strings"
            )
        if len(set(heldout)) != len(heldout):
            raise ValueError("candidate_heldout_frame_ids must be unique")
        if previous_frames.intersection(heldout):
            raise ValueError("selected and held-out frame IDs must be disjoint")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "dataset": self.dataset,
            "protocol_version": self.protocol_version,
            "trajectory_type": self.trajectory_type,
            "seed": self.seed,
            "selections": [selection.to_dict() for selection in self.selections],
            "candidate_heldout_frame_ids": list(
                self.candidate_heldout_frame_ids
            ),
            "camera_graph_stats": self.camera_graph_stats.to_dict(),
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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SparseSceneProtocol":
        expected = {
            "schema_version",
            "scene_id",
            "dataset",
            "protocol_version",
            "trajectory_type",
            "seed",
            "selections",
            "candidate_heldout_frame_ids",
            "camera_graph_stats",
        }
        _require_exact_keys(value, expected, "protocol")

        selections = value["selections"]
        heldout = value["candidate_heldout_frame_ids"]
        graph_stats = value["camera_graph_stats"]
        if not isinstance(selections, list):
            raise ValueError("selections must be a list")
        if not all(isinstance(selection, Mapping) for selection in selections):
            raise ValueError("each selection must be an object")
        if not isinstance(heldout, list):
            raise ValueError("candidate_heldout_frame_ids must be a list")
        if not isinstance(graph_stats, Mapping):
            raise ValueError("camera_graph_stats must be an object")

        return cls(
            schema_version=value["schema_version"],
            scene_id=value["scene_id"],
            dataset=value["dataset"],
            protocol_version=value["protocol_version"],
            trajectory_type=value["trajectory_type"],
            seed=value["seed"],
            selections=tuple(
                ViewSelection.from_dict(selection) for selection in selections
            ),
            candidate_heldout_frame_ids=tuple(heldout),
            camera_graph_stats=CameraGraphStats.from_dict(graph_stats),
        )

    @classmethod
    def from_json(cls, value: str) -> "SparseSceneProtocol":
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise ValueError("protocol JSON root must be an object")
        return cls.from_dict(decoded)

    @classmethod
    def load(cls, path: Path) -> "SparseSceneProtocol":
        return cls.from_json(path.read_text(encoding="utf-8"))
