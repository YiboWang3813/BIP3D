"""Protocol-aware EmbodiedScan grounding dataset."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from mmengine.fileio import load

from bip3d.datasets import EmbodiedScanDetGroundingDataset
from bip3d.registry import DATASETS

from .protocol import SparseSceneProtocol


def protocol_path(protocol_dir: Path, scan_id: str) -> Path:
    """Return the reversible, filesystem-safe protocol path for a scan."""
    return protocol_dir / f"{quote(scan_id, safe='')}.json"


def load_protocol_frame_ids(
    path: Path,
    *,
    budget: int,
    expected_dataset: str | None = None,
    expected_trajectory_type: str | None = None,
    expected_protocol_version: str | None = None,
) -> tuple[str, ...]:
    protocol = SparseSceneProtocol.load(path)
    for expected, actual, field in (
        (expected_dataset, protocol.dataset, "dataset"),
        (
            expected_trajectory_type,
            protocol.trajectory_type,
            "trajectory_type",
        ),
        (
            expected_protocol_version,
            protocol.protocol_version,
            "protocol_version",
        ),
    ):
        if expected is not None and actual != expected:
            raise ValueError(
                f"{path}: expected {field}={expected!r}, got {actual!r}"
            )
    selection = next(
        (
            selection
            for selection in protocol.selections
            if selection.budget == budget
        ),
        None,
    )
    if selection is None:
        raise ValueError(f"{path}: protocol has no budget {budget}")
    return selection.frame_ids


def select_scene_frames(
    scene: dict[str, Any],
    frame_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Return a deep-copied scene restricted to protocol frame order."""
    selected = copy.deepcopy(scene)
    images = selected.get("images")
    if not isinstance(images, list):
        raise ValueError("scene.images must be a list")
    image_indices = {}
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise ValueError("scene.images entries must be objects")
        frame_id = image.get("img_path")
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("scene image has no valid img_path")
        if frame_id in image_indices:
            raise ValueError(f"duplicate scene frame ID: {frame_id}")
        image_indices[frame_id] = index

    missing = [frame_id for frame_id in frame_ids if frame_id not in image_indices]
    if missing:
        raise ValueError(f"protocol frames are absent from scene: {missing[:3]}")
    indices = [image_indices[frame_id] for frame_id in frame_ids]

    for key in ("images", "img_path", "depth_img_path"):
        if key in selected:
            value = selected[key]
            if not isinstance(value, list) or len(value) != len(images):
                raise ValueError(f"scene.{key} must align with scene.images")
            selected[key] = [value[index] for index in indices]

    depth2img = selected.get("depth2img")
    if isinstance(depth2img, dict):
        extrinsic = depth2img.get("extrinsic")
        if not isinstance(extrinsic, list) or len(extrinsic) != len(images):
            raise ValueError("scene.depth2img.extrinsic must align with images")
        depth2img["extrinsic"] = [extrinsic[index] for index in indices]
        intrinsic = depth2img.get("intrinsic")
        if isinstance(intrinsic, list):
            if len(intrinsic) != len(images):
                raise ValueError("scene.depth2img.intrinsic must align with images")
            depth2img["intrinsic"] = [intrinsic[index] for index in indices]

    for key in ("cam2img", "depth_cam2img", "visible_instance_masks"):
        value = selected.get(key)
        if isinstance(value, list):
            if len(value) != len(images):
                raise ValueError(f"scene.{key} must align with scene.images")
            selected[key] = [value[index] for index in indices]

    annotations = []
    for key in ("ann_info", "eval_ann_info"):
        value = selected.get(key)
        if isinstance(value, dict) and all(value is not item for item in annotations):
            annotations.append(value)
            masks = value.get("visible_instance_masks")
            if isinstance(masks, list):
                if len(masks) != len(images):
                    raise ValueError(
                        f"scene.{key}.visible_instance_masks must align with images"
                    )
                value["visible_instance_masks"] = [
                    masks[index] for index in indices
                ]

    selected["sparse_view_frame_ids"] = list(frame_ids)
    selected["sparse_view_budget"] = len(frame_ids)
    return selected


@DATASETS.register_module()
class SparseProtocolGroundingDataset(EmbodiedScanDetGroundingDataset):
    """Grounding dataset restricted by versioned scene-level protocols."""

    def __init__(
        self,
        *args,
        protocol_dir: str,
        view_budget: int,
        missing_protocol: str = "error",
        expected_protocol_dataset: str | None = None,
        expected_trajectory_type: str | None = None,
        expected_protocol_version: str | None = None,
        **kwargs,
    ):
        if missing_protocol not in {"error", "skip"}:
            raise ValueError("missing_protocol must be 'error' or 'skip'")
        if (
            isinstance(view_budget, bool)
            or not isinstance(view_budget, int)
            or view_budget <= 0
        ):
            raise ValueError("view_budget must be a positive integer")
        self.protocol_dir = Path(protocol_dir)
        self.view_budget = view_budget
        self.missing_protocol = missing_protocol
        self.expected_protocol_dataset = expected_protocol_dataset
        self.expected_trajectory_type = expected_trajectory_type
        self.expected_protocol_version = expected_protocol_version
        self.protocol_frame_ids: dict[str, tuple[str, ...]] = {}
        super().__init__(*args, **kwargs)

    def load_language_data(self):
        self.convert_info_to_scan()
        vg_files = [self.vg_file] if isinstance(self.vg_file, str) else self.vg_file
        language_annotations = []
        for filename in vg_files:
            loaded = load(os.path.join(self.data_root, filename))
            if not isinstance(loaded, list):
                raise ValueError(f"grounding annotation must be a list: {filename}")
            language_annotations.extend(loaded)

        known_scan_ids = set(self.scan_ids)
        language_annotations = [
            item
            for item in language_annotations
            if isinstance(item, dict) and item.get("scan_id") in known_scan_ids
        ]
        if self.dataset_length is not None:
            if self.dataset_length <= 0:
                raise ValueError("dataset_length must be positive")
            interval = len(language_annotations) / self.dataset_length
            language_annotations = [
                language_annotations[int(interval * index)]
                for index in range(self.dataset_length)
            ]

        retained_scan_ids = []
        for scan_id in self.scan_ids:
            path = protocol_path(self.protocol_dir, scan_id)
            if not path.is_file():
                if self.missing_protocol == "error":
                    raise FileNotFoundError(f"missing sparse protocol: {path}")
                continue
            self.protocol_frame_ids[scan_id] = load_protocol_frame_ids(
                path,
                budget=self.view_budget,
                expected_dataset=self.expected_protocol_dataset,
                expected_trajectory_type=self.expected_trajectory_type,
                expected_protocol_version=self.expected_protocol_version,
            )
            retained_scan_ids.append(scan_id)

        retained = set(retained_scan_ids)
        self.scans = {
            scan_id: self.scans[scan_id] for scan_id in retained_scan_ids
        }
        self.scan_ids = retained_scan_ids
        self.data_list = [
            item
            for item in language_annotations
            if item["scan_id"] in retained
        ]
        self.scan_id_to_data_idx = {
            scan_id: [] for scan_id in retained_scan_ids
        }
        for index, item in enumerate(self.data_list):
            self.scan_id_to_data_idx[item["scan_id"]].append(index)

    def get_data_info_grounding(self, data_info):
        scene = super().get_data_info_grounding(data_info)
        scan_id = scene["scan_id"]
        return select_scene_frames(
            scene,
            self.protocol_frame_ids[scan_id],
        )
