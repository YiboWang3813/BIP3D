"""Convert trusted EmbodiedScan info annotations to pose manifests."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote

import numpy as np

from .geometry import CameraPose
from .pose_manifest import FramePose, PoseManifest, ScenePoses


SOURCE_DATASETS = frozenset({"scannet", "3rscan", "matterport3d"})
MAX_RIGID_CORRECTION = 1e-4


def encode_scene_id(sample_idx: str) -> str:
    if not isinstance(sample_idx, str) or not sample_idx:
        raise ValueError("sample_idx must be a non-empty string")
    return quote(sample_idx, safe="")


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _nearest_rigid_transform(value: Any, context: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{context} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=MAX_RIGID_CORRECTION):
        raise ValueError(f"{context} has an invalid homogeneous bottom row")

    rotation = matrix[:3, :3]
    if np.linalg.det(rotation) <= 0:
        raise ValueError(f"{context} rotation must preserve orientation")
    left, _, right = np.linalg.svd(rotation)
    nearest = left @ right
    correction = float(np.max(np.abs(rotation - nearest)))
    if correction > MAX_RIGID_CORRECTION:
        raise ValueError(
            f"{context} rotation needs excessive correction: {correction}"
        )

    rigid = matrix.copy()
    rigid[:3, :3] = nearest
    rigid[3] = (0, 0, 0, 1)
    return rigid


def _convert_scene(raw_scene: Any) -> tuple[str, ScenePoses]:
    scene = _as_mapping(raw_scene, "scene")
    sample_idx = scene.get("sample_idx")
    if not isinstance(sample_idx, str) or "/" not in sample_idx:
        raise ValueError("scene.sample_idx must contain a dataset prefix")
    source_dataset = sample_idx.split("/", 1)[0]
    if source_dataset not in SOURCE_DATASETS:
        raise ValueError(f"unsupported source dataset: {source_dataset}")

    images = scene.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"{sample_idx}.images must be a non-empty list")
    axis_align = _nearest_rigid_transform(
        scene.get("axis_align_matrix", np.eye(4)),
        f"{sample_idx}.axis_align_matrix",
    )

    frames = []
    for index, raw_image in enumerate(images):
        image = _as_mapping(raw_image, f"{sample_idx}.images[{index}]")
        frame_id = image.get("img_path")
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError(
                f"{sample_idx}.images[{index}].img_path must be a string"
            )
        cam2global = _nearest_rigid_transform(
            image.get("cam2global"),
            f"{sample_idx}.images[{index}].cam2global",
        )
        camera_to_aligned_world = _nearest_rigid_transform(
            axis_align @ cam2global,
            f"{sample_idx}.images[{index}].aligned_cam2global",
        )
        frames.append(
            FramePose(frame_id, CameraPose(camera_to_aligned_world))
        )

    return source_dataset, ScenePoses(
        scene_id=encode_scene_id(sample_idx),
        frames=tuple(sorted(frames, key=lambda frame: frame.frame_id)),
    )


def load_embodiedscan_pose_manifest(
    info_path: Path,
    *,
    dataset_name: str,
    source_datasets: Iterable[str] = SOURCE_DATASETS,
    scene_ids: Iterable[str] = (),
    max_scenes: int | None = None,
) -> PoseManifest:
    """Load an official trusted info pickle and extract aligned camera poses."""
    selected_datasets = frozenset(source_datasets)
    unknown_datasets = selected_datasets - SOURCE_DATASETS
    if unknown_datasets:
        raise ValueError(
            f"unsupported source datasets: {sorted(unknown_datasets)}"
        )
    selected_scene_ids = frozenset(scene_ids)
    if max_scenes is not None and (
        isinstance(max_scenes, bool)
        or not isinstance(max_scenes, int)
        or max_scenes <= 0
    ):
        raise ValueError("max_scenes must be a positive integer")

    with info_path.open("rb") as stream:
        value = pickle.load(stream)
    root = _as_mapping(value, "annotation")
    data_list = root.get("data_list")
    if not isinstance(data_list, list):
        raise ValueError("annotation.data_list must be a list")

    scenes = []
    found_scene_ids = set()
    for raw_scene in data_list:
        scene = _as_mapping(raw_scene, "scene")
        sample_idx = scene.get("sample_idx")
        if not isinstance(sample_idx, str) or "/" not in sample_idx:
            raise ValueError("scene.sample_idx must contain a dataset prefix")
        source_dataset = sample_idx.split("/", 1)[0]
        if source_dataset not in selected_datasets:
            continue
        if selected_scene_ids and sample_idx not in selected_scene_ids:
            continue
        _, converted = _convert_scene(scene)
        scenes.append(converted)
        found_scene_ids.add(sample_idx)
        if max_scenes is not None and len(scenes) >= max_scenes:
            break

    missing_scene_ids = selected_scene_ids - found_scene_ids
    if missing_scene_ids:
        raise ValueError(
            f"requested scenes were not found: {sorted(missing_scene_ids)}"
        )
    if not scenes:
        raise ValueError("no scenes matched the requested filters")
    return PoseManifest(
        dataset=dataset_name,
        pose_convention="camera_to_world",
        scenes=tuple(sorted(scenes, key=lambda scene: scene.scene_id)),
    )
