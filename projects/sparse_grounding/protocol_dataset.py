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

from .oracle_manifest import (
    OracleQuerySelection,
    RealViewOracleManifest,
)
from .protocol import SparseSceneProtocol


def protocol_path(protocol_dir: Path, scan_id: str) -> Path:
    """Return the reversible, filesystem-safe protocol path for a scan."""
    return protocol_dir / f"{quote(scan_id, safe='')}.json"


def identify_sparse_query(
    query: dict[str, Any],
    *,
    filename: str,
    query_index: int,
) -> dict[str, Any]:
    """Copy a query and attach a stable source-file identity."""
    if not isinstance(filename, str) or not filename:
        raise ValueError("filename must be a non-empty string")
    if (
        isinstance(query_index, bool)
        or not isinstance(query_index, int)
        or query_index < 0
    ):
        raise ValueError("query_index must be a nonnegative integer")
    identified = copy.deepcopy(query)
    identified["sparse_query_file"] = filename
    identified["sparse_query_index"] = query_index
    identified["sparse_query_id"] = f"{filename}:{query_index}"
    return identified


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


def oracle_augmented_frame_ids(
    protocol: SparseSceneProtocol,
    oracle_selection: OracleQuerySelection,
    *,
    base_view_budget: int,
) -> tuple[str, ...]:
    """Validate and append query-level held-out frames to sparse inputs."""
    base_frame_ids = next(
        (
            selection.frame_ids
            for selection in protocol.selections
            if selection.budget == base_view_budget
        ),
        None,
    )
    if base_frame_ids is None:
        raise ValueError(
            f"protocol has no base budget {base_view_budget}"
        )
    if quote(oracle_selection.scan_id, safe="") != protocol.scene_id:
        raise ValueError(
            "oracle selection scan_id does not match encoded protocol scene"
        )
    heldout = set(protocol.candidate_heldout_frame_ids)
    invalid = [
        frame_id
        for frame_id in oracle_selection.frame_ids
        if frame_id not in heldout
    ]
    if invalid:
        raise ValueError(
            f"oracle frames are not in the held-out pool: {invalid[:3]}"
        )
    return (*base_frame_ids, *oracle_selection.frame_ids)


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
        self.protocols: dict[str, SparseSceneProtocol] = {}
        super().__init__(*args, **kwargs)

    def load_language_data(self):
        self.convert_info_to_scan()
        vg_files = [self.vg_file] if isinstance(self.vg_file, str) else self.vg_file
        language_annotations = []
        for filename in vg_files:
            loaded = load(os.path.join(self.data_root, filename))
            if not isinstance(loaded, list):
                raise ValueError(f"grounding annotation must be a list: {filename}")
            for query_index, item in enumerate(loaded):
                if not isinstance(item, dict):
                    continue
                language_annotations.append(
                    identify_sparse_query(
                        item,
                        filename=filename,
                        query_index=query_index,
                    )
                )

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
            protocol = SparseSceneProtocol.load(path)
            self.protocols[scan_id] = protocol
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

    def selected_frame_ids(self, data_info: dict[str, Any]) -> tuple[str, ...]:
        return self.protocol_frame_ids[data_info["scan_id"]]

    def evaluation_view_metadata(
        self,
        data_info: dict[str, Any],
        frame_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        protocol = self.protocols[data_info["scan_id"]]
        return {
            "selected_frame_ids": list(frame_ids),
            "base_view_budget": self.view_budget,
            "trajectory_type": protocol.trajectory_type,
        }

    def get_data_info_grounding(self, data_info):
        query_metadata = {
            key: copy.deepcopy(data_info.get(key))
            for key in (
                "sparse_query_file",
                "sparse_query_index",
                "sparse_query_id",
                "target_id",
                "target",
                "text",
            )
        }
        scene = super().get_data_info_grounding(data_info)
        eval_ann_info = scene.get("eval_ann_info")
        if isinstance(eval_ann_info, dict):
            eval_ann_info.update(query_metadata)
        scan_id = scene["scan_id"]
        frame_ids = self.selected_frame_ids(data_info)
        selected = select_scene_frames(
            scene,
            frame_ids,
        )
        eval_ann_info = selected.get("eval_ann_info")
        if isinstance(eval_ann_info, dict):
            eval_ann_info.update(
                self.evaluation_view_metadata(data_info, frame_ids)
            )
        return selected


@DATASETS.register_module()
class OracleProtocolGroundingDataset(SparseProtocolGroundingDataset):
    """Sparse grounding dataset augmented by validated held-out real views."""

    def __init__(
        self,
        *args,
        oracle_manifest: str,
        expected_oracle_policy: str | None = None,
        missing_oracle: str = "error",
        **kwargs,
    ):
        if missing_oracle not in {"error", "skip"}:
            raise ValueError("missing_oracle must be 'error' or 'skip'")
        self.oracle_manifest = RealViewOracleManifest.load(
            Path(oracle_manifest)
        )
        if (
            expected_oracle_policy is not None
            and self.oracle_manifest.policy != expected_oracle_policy
        ):
            raise ValueError(
                "oracle manifest policy mismatch: "
                f"expected {expected_oracle_policy!r}, "
                f"got {self.oracle_manifest.policy!r}"
            )
        self.missing_oracle = missing_oracle
        super().__init__(*args, **kwargs)

    def load_language_data(self):
        super().load_language_data()
        if self.oracle_manifest.base_view_budget != self.view_budget:
            raise ValueError(
                "oracle manifest base_view_budget does not match dataset"
            )
        expected_trajectory = self.expected_trajectory_type
        if (
            expected_trajectory is not None
            and self.oracle_manifest.trajectory_type != expected_trajectory
        ):
            raise ValueError(
                "oracle manifest trajectory_type does not match dataset"
            )

        available = self.oracle_manifest.records_by_query_id
        missing = [
            item["sparse_query_id"]
            for item in self.data_list
            if item["sparse_query_id"] not in available
        ]
        if missing and self.missing_oracle == "error":
            raise ValueError(
                f"oracle manifest is missing {len(missing)} queries: "
                f"{missing[:3]}"
            )
        if self.missing_oracle == "skip":
            self.data_list = [
                item
                for item in self.data_list
                if item["sparse_query_id"] in available
            ]

    def selected_frame_ids(self, data_info: dict[str, Any]) -> tuple[str, ...]:
        query_id = data_info["sparse_query_id"]
        oracle_selection = self.oracle_manifest.records_by_query_id[query_id]
        scan_id = data_info["scan_id"]
        return oracle_augmented_frame_ids(
            self.protocols[scan_id],
            oracle_selection,
            base_view_budget=self.view_budget,
        )

    def evaluation_view_metadata(
        self,
        data_info: dict[str, Any],
        frame_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        metadata = super().evaluation_view_metadata(data_info, frame_ids)
        metadata.update(
            {
                "oracle_policy": self.oracle_manifest.policy,
                "oracle_view_budget": self.oracle_manifest.oracle_view_budget,
                "oracle_frame_ids": list(frame_ids[self.view_budget :]),
            }
        )
        return metadata
