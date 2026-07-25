"""Build and execute minimal RGB-D staging plans for sparse experiments."""

from __future__ import annotations

import json
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .oracle_manifest import RealViewOracleManifest
from .protocol import SparseSceneProtocol


SCHEMA_VERSION = "1.0"
MODALITIES = frozenset({"rgb", "depth"})


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


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ValueError(f"unsafe relative data path: {value!r}")
    return path.as_posix()


@dataclass(frozen=True)
class StagingFile:
    relative_path: str
    modality: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _safe_relative_path(self.relative_path),
        )
        if self.modality not in MODALITIES:
            raise ValueError(f"unsupported modality: {self.modality!r}")

    def to_dict(self) -> dict[str, str]:
        return {
            "relative_path": self.relative_path,
            "modality": self.modality,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StagingFile":
        _exact_keys(value, {"relative_path", "modality"}, "file")
        return cls(
            relative_path=value["relative_path"],
            modality=value["modality"],
        )


@dataclass(frozen=True)
class StagingManifest:
    dataset: str
    files: tuple[StagingFile, ...]
    protocol_sources: tuple[str, ...]
    oracle_sources: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported staging manifest schema_version")
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("dataset must be a non-empty string")
        if not self.files:
            raise ValueError("files cannot be empty")
        file_keys = [
            (entry.relative_path, entry.modality) for entry in self.files
        ]
        if file_keys != sorted(set(file_keys)):
            raise ValueError("files must be unique and sorted")
        for field in ("protocol_sources", "oracle_sources"):
            values = getattr(self, field)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field} must be unique and sorted")

    @property
    def modality_counts(self) -> dict[str, int]:
        return {
            modality: sum(
                entry.modality == modality for entry in self.files
            )
            for modality in sorted(MODALITIES)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "file_count": len(self.files),
            "modality_counts": self.modality_counts,
            "protocol_sources": list(self.protocol_sources),
            "oracle_sources": list(self.oracle_sources),
            "files": [entry.to_dict() for entry in self.files],
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
    def from_dict(cls, value: Mapping[str, Any]) -> "StagingManifest":
        _exact_keys(
            value,
            {
                "schema_version",
                "dataset",
                "file_count",
                "modality_counts",
                "protocol_sources",
                "oracle_sources",
                "files",
            },
            "manifest",
        )
        files = value["files"]
        if not isinstance(files, list):
            raise ValueError("manifest.files must be a list")
        manifest = cls(
            schema_version=value["schema_version"],
            dataset=value["dataset"],
            protocol_sources=tuple(value["protocol_sources"]),
            oracle_sources=tuple(value["oracle_sources"]),
            files=tuple(StagingFile.from_dict(entry) for entry in files),
        )
        if value["file_count"] != len(manifest.files):
            raise ValueError("manifest.file_count is inconsistent")
        if value["modality_counts"] != manifest.modality_counts:
            raise ValueError("manifest.modality_counts is inconsistent")
        return manifest

    @classmethod
    def load(cls, path: Path) -> "StagingManifest":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("staging manifest root must be an object")
        return cls.from_dict(value)


def _load_frame_pairs(
    info_files: Iterable[Path],
    *,
    dataset: str,
) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for info_file in info_files:
        with info_file.open("rb") as stream:
            root = pickle.load(stream)
        if not isinstance(root, Mapping) or not isinstance(
            root.get("data_list"), list
        ):
            raise ValueError(f"{info_file}: invalid trusted info annotation")
        for scene in root["data_list"]:
            if not isinstance(scene, Mapping):
                raise ValueError(f"{info_file}: scene must be an object")
            sample_idx = scene.get("sample_idx")
            if (
                not isinstance(sample_idx, str)
                or sample_idx.split("/", 1)[0] != dataset
            ):
                continue
            images = scene.get("images")
            if not isinstance(images, list):
                raise ValueError(f"{info_file}: {sample_idx}.images invalid")
            for image in images:
                if not isinstance(image, Mapping):
                    raise ValueError(f"{info_file}: image must be an object")
                rgb = image.get("img_path")
                depth = image.get("depth_path")
                if not isinstance(rgb, str) or not isinstance(depth, str):
                    raise ValueError(
                        f"{info_file}: {sample_idx} has missing RGB/depth path"
                    )
                previous = pairs.setdefault(rgb, depth)
                if previous != depth:
                    raise ValueError(f"conflicting depth paths for {rgb}")
    if not pairs:
        raise ValueError(f"no {dataset} RGB-D frame pairs found")
    return pairs


def _selected_protocol_frames(protocol_dir: Path) -> set[str]:
    paths = sorted(
        path
        for path in protocol_dir.glob("*.json")
        if path.name != "generation_summary.json"
    )
    if not paths:
        raise ValueError(f"no protocol JSON files found in {protocol_dir}")
    frames = set()
    for path in paths:
        protocol = SparseSceneProtocol.load(path)
        frames.update(protocol.selections[-1].frame_ids)
    return frames


def build_staging_manifest(
    *,
    dataset: str,
    info_files: Iterable[Path],
    protocol_dirs: Iterable[Path],
    oracle_files: Iterable[Path] = (),
) -> StagingManifest:
    """Return the exact RGB/depth union referenced by experiment inputs."""
    info_files = tuple(info_files)
    protocol_dirs = tuple(protocol_dirs)
    oracle_files = tuple(oracle_files)
    if not info_files or not protocol_dirs:
        raise ValueError("at least one info file and protocol directory required")

    selected_frames = set()
    for protocol_dir in protocol_dirs:
        selected_frames.update(_selected_protocol_frames(protocol_dir))
    for oracle_file in oracle_files:
        manifest = RealViewOracleManifest.load(oracle_file)
        for record in manifest.records:
            selected_frames.update(record.frame_ids)

    pairs = _load_frame_pairs(info_files, dataset=dataset)
    missing = sorted(selected_frames - pairs.keys())
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"{len(missing)} selected RGB frames are absent from info files: "
            f"{preview}"
        )

    entries = {
        StagingFile(relative_path=rgb, modality="rgb")
        for rgb in selected_frames
    }
    entries.update(
        StagingFile(relative_path=pairs[rgb], modality="depth")
        for rgb in selected_frames
    )
    return StagingManifest(
        dataset=dataset,
        files=tuple(
            sorted(entries, key=lambda entry: (entry.relative_path, entry.modality))
        ),
        protocol_sources=tuple(
            sorted(str(path.resolve()) for path in protocol_dirs)
        ),
        oracle_sources=tuple(
            sorted(str(path.resolve()) for path in oracle_files)
        ),
    )


def resolve_data_path(root: Path, relative_path: str, dataset: str) -> Path:
    """Resolve either a parent data root or the dataset directory itself."""
    relative = Path(_safe_relative_path(relative_path))
    if relative.parts[0] != dataset:
        raise ValueError(
            f"path {relative_path!r} does not start with dataset {dataset!r}"
        )
    if root.name == dataset:
        relative = Path(*relative.parts[1:])
    return root / relative


def execute_staging(
    manifest: StagingManifest,
    *,
    source_root: Path,
    destination_root: Path,
    execute: bool = False,
) -> dict[str, int | bool]:
    """Copy missing files atomically, or verify the plan in dry-run mode."""
    counts = {
        "file_count": len(manifest.files),
        "source_missing": 0,
        "already_complete": 0,
        "needs_copy": 0,
        "copied": 0,
        "bytes_total": 0,
        "bytes_to_copy": 0,
    }
    for entry in manifest.files:
        source = resolve_data_path(
            source_root, entry.relative_path, manifest.dataset
        )
        destination = resolve_data_path(
            destination_root, entry.relative_path, manifest.dataset
        )
        if not source.is_file():
            counts["source_missing"] += 1
            continue
        source_size = source.stat().st_size
        counts["bytes_total"] += source_size
        if destination.is_file() and destination.stat().st_size == source_size:
            counts["already_complete"] += 1
            continue
        counts["needs_copy"] += 1
        counts["bytes_to_copy"] += source_size
        if not execute:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.staging"
        )
        try:
            shutil.copy2(source, temporary)
            if temporary.stat().st_size != source_size:
                raise OSError(f"size mismatch after copying {source}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        counts["copied"] += 1
    return {"executed": execute, **counts}
