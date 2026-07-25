"""Batch visibility coverage audit for EmbodiedScan grounding queries."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote

import numpy as np

from .coverage import OrientedBox9D, coverage_band, voxel_indices
from .embodiedscan_adapter import (
    SOURCE_DATASETS,
    _nearest_rigid_transform,
)
from .geometry import CameraIntrinsics, CameraPose, backproject_depth
from .protocol import SparseSceneProtocol


DepthLoader = Callable[[Path], np.ndarray]


def load_depth_image(path: Path) -> np.ndarray:
    import cv2

    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"failed to decode depth image: {path}")
    if depth.ndim != 2:
        raise ValueError(f"depth image must be single-channel: {path}")
    return depth


def _selection(
    protocol: SparseSceneProtocol,
    budget: int,
) -> tuple[str, ...]:
    for selection in protocol.selections:
        if selection.budget == budget:
            return selection.frame_ids
    raise ValueError(f"protocol has no budget {budget}")


def _intrinsics_for_frame(
    scene: Mapping[str, Any],
    frame_index: int,
    width: int,
    height: int,
) -> CameraIntrinsics:
    value = scene.get("depth_cam2img", scene.get("cam2img"))
    if isinstance(value, list):
        value = value[frame_index]
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape not in {(3, 3), (4, 4)}:
        raise ValueError(
            f"depth intrinsics must have shape (3, 3) or (4, 4), got "
            f"{matrix.shape}"
        )
    return CameraIntrinsics(
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        width=width,
        height=height,
    )


def compute_scene_target_coverages(
    scene: Mapping[str, Any],
    *,
    target_ids: Iterable[int],
    selected_frame_ids: Iterable[str],
    data_root: Path,
    voxel_size_m: float = 0.02,
    depth_loader: DepthLoader = load_depth_image,
) -> dict[int, dict[str, Any]]:
    """Compute one coverage record per unique target instance in a scene."""
    scan_id = scene.get("sample_idx")
    if not isinstance(scan_id, str) or "/" not in scan_id:
        raise ValueError("scene.sample_idx must contain a dataset prefix")
    source_dataset = scan_id.split("/", 1)[0]
    if source_dataset not in SOURCE_DATASETS:
        raise ValueError(f"unsupported source dataset: {source_dataset}")
    depth_scale_m = 1 / (4000 if source_dataset == "matterport3d" else 1000)

    instances = scene.get("instances")
    images = scene.get("images")
    if not isinstance(instances, list) or not isinstance(images, list):
        raise ValueError("scene instances and images must be lists")
    boxes = {}
    instance_indices = {}
    for instance_index, instance in enumerate(instances):
        if not isinstance(instance, Mapping):
            raise ValueError("scene instance entries must be objects")
        bbox_id = instance.get("bbox_id")
        if not isinstance(bbox_id, int) or isinstance(bbox_id, bool):
            raise ValueError("instance.bbox_id must be an integer")
        if bbox_id in boxes:
            raise ValueError(f"duplicate instance bbox_id: {bbox_id}")
        boxes[bbox_id] = OrientedBox9D.from_array(instance.get("bbox_3d"))
        instance_indices[bbox_id] = instance_index

    requested_ids = frozenset(target_ids)
    missing_ids = requested_ids - boxes.keys()
    if missing_ids:
        raise ValueError(f"query target IDs are absent from scene: {missing_ids}")
    selected = frozenset(selected_frame_ids)
    image_frame_ids = {
        image.get("img_path")
        for image in images
        if isinstance(image, Mapping)
    }
    missing_frames = selected - image_frame_ids
    if missing_frames:
        raise ValueError(f"protocol frames are absent from scene: {missing_frames}")

    reference_voxels: dict[int, set[tuple[int, int, int]]] = defaultdict(set)
    sparse_voxels: dict[int, set[tuple[int, int, int]]] = defaultdict(set)
    reference_annotated_frames: Counter[int] = Counter()
    sparse_annotated_frames: Counter[int] = Counter()
    reference_occupied_frames: Counter[int] = Counter()
    sparse_occupied_frames: Counter[int] = Counter()
    axis_align = _nearest_rigid_transform(
        scene.get("axis_align_matrix", np.eye(4)),
        f"{scan_id}.axis_align_matrix",
    )

    for frame_index, image in enumerate(images):
        if not isinstance(image, Mapping):
            raise ValueError("scene image entries must be objects")
        frame_id = image.get("img_path")
        depth_path = image.get("depth_path")
        if not isinstance(frame_id, str) or not isinstance(depth_path, str):
            raise ValueError("image paths must be strings")
        visible_ids = image.get("visible_instance_ids")
        if visible_ids is None:
            visible_targets = requested_ids
        else:
            visible_indices = frozenset(visible_ids)
            visible_targets = {
                target_id
                for target_id in requested_ids
                if instance_indices[target_id] in visible_indices
            }
        if not visible_targets:
            continue

        is_sparse = frame_id in selected
        for target_id in visible_targets:
            reference_annotated_frames[target_id] += 1
            if is_sparse:
                sparse_annotated_frames[target_id] += 1
        depth = np.asarray(depth_loader(data_root / depth_path))
        if depth.ndim != 2:
            raise ValueError(f"depth image must be two-dimensional: {depth_path}")
        intrinsics = _intrinsics_for_frame(
            scene,
            frame_index,
            width=depth.shape[1],
            height=depth.shape[0],
        )
        cam2global = _nearest_rigid_transform(
            image.get("cam2global"),
            f"{scan_id}.{frame_id}.cam2global",
        )
        aligned_cam2global = _nearest_rigid_transform(
            axis_align @ cam2global,
            f"{scan_id}.{frame_id}.aligned_cam2global",
        )
        points, _ = backproject_depth(
            depth,
            intrinsics,
            CameraPose(aligned_cam2global),
            depth_unit_scale_m=depth_scale_m,
        )
        for target_id in visible_targets:
            target_points = points[boxes[target_id].contains(points)]
            voxels = {
                tuple(row)
                for row in voxel_indices(target_points, voxel_size_m).tolist()
            }
            if voxels:
                reference_voxels[target_id].update(voxels)
                reference_occupied_frames[target_id] += 1
                if is_sparse:
                    sparse_voxels[target_id].update(voxels)
                    sparse_occupied_frames[target_id] += 1

    results = {}
    for target_id in sorted(requested_ids):
        reference = reference_voxels[target_id]
        sparse = sparse_voxels[target_id]
        if not reference:
            results[target_id] = {
                "status": "unavailable",
                "error": "reference target has no occupied voxels",
                "reference_annotated_visible_frames": (
                    reference_annotated_frames[target_id]
                ),
                "sparse_annotated_visible_frames": (
                    sparse_annotated_frames[target_id]
                ),
                "reference_occupied_frames": 0,
                "sparse_occupied_frames": 0,
            }
            continue
        overlap = sparse & reference
        novel = sparse - reference
        coverage = len(overlap) / len(reference)
        results[target_id] = {
            "status": "ok",
            "coverage": coverage,
            "coverage_band": coverage_band(coverage),
            "sparse_voxel_count": len(sparse),
            "reference_voxel_count": len(reference),
            "overlap_voxel_count": len(overlap),
            "novel_sparse_voxel_count": len(novel),
            "reference_annotated_visible_frames": (
                reference_annotated_frames[target_id]
            ),
            "sparse_annotated_visible_frames": sparse_annotated_frames[target_id],
            "reference_occupied_frames": reference_occupied_frames[target_id],
            "sparse_occupied_frames": sparse_occupied_frames[target_id],
        }
    return results


def run_coverage_audit(
    *,
    data_root: Path,
    info_file: Path,
    vg_file: Path,
    protocol_dir: Path,
    budget: int,
    source_datasets: Iterable[str] = SOURCE_DATASETS,
    voxel_size_m: float = 0.02,
    max_scenes: int | None = None,
    depth_loader: DepthLoader = load_depth_image,
) -> dict[str, Any]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    selected_sources = frozenset(source_datasets)
    unknown_sources = selected_sources - SOURCE_DATASETS
    if unknown_sources:
        raise ValueError(f"unsupported source datasets: {unknown_sources}")
    if max_scenes is not None and max_scenes <= 0:
        raise ValueError("max_scenes must be positive")

    with info_file.open("rb") as stream:
        annotation = pickle.load(stream)
    scenes = annotation.get("data_list") if isinstance(annotation, dict) else None
    if not isinstance(scenes, list):
        raise ValueError("annotation.data_list must be a list")
    queries = json.loads(vg_file.read_text(encoding="utf-8"))
    if not isinstance(queries, list):
        raise ValueError("grounding annotation must be a list")
    queries_by_scene: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for query_index, query in enumerate(queries):
        if not isinstance(query, Mapping):
            raise ValueError("grounding query entries must be objects")
        scan_id = query.get("scan_id")
        if isinstance(scan_id, str) and scan_id.split("/", 1)[0] in selected_sources:
            queries_by_scene[scan_id].append((query_index, query))

    records = []
    scene_results = []
    processed_scenes = 0
    unique_target_count = 0
    target_status_counts: Counter[str] = Counter()
    target_band_counts: Counter[str] = Counter()
    for scene in scenes:
        if not isinstance(scene, Mapping):
            continue
        scan_id = scene.get("sample_idx")
        scene_queries = queries_by_scene.get(scan_id, [])
        if not scene_queries:
            continue
        path = protocol_dir / f"{quote(scan_id, safe='')}.json"
        if not path.is_file():
            scene_results.append(
                {"scan_id": scan_id, "status": "skipped", "error": "missing protocol"}
            )
            continue
        protocol = SparseSceneProtocol.load(path)
        expected_protocol_scene = quote(scan_id, safe="")
        if protocol.scene_id != expected_protocol_scene:
            scene_results.append(
                {
                    "scan_id": scan_id,
                    "status": "failed",
                    "error": (
                        f"protocol scene {protocol.scene_id!r} does not match "
                        f"{expected_protocol_scene!r}"
                    ),
                }
            )
            continue
        selected_frames = _selection(protocol, budget)
        target_ids = {
            query.get("target_id")
            for _, query in scene_queries
            if isinstance(query.get("target_id"), int)
            and not isinstance(query.get("target_id"), bool)
        }
        try:
            target_results = compute_scene_target_coverages(
                scene,
                target_ids=target_ids,
                selected_frame_ids=selected_frames,
                data_root=data_root,
                voxel_size_m=voxel_size_m,
                depth_loader=depth_loader,
            )
        except (OSError, ValueError) as error:
            scene_results.append(
                {"scan_id": scan_id, "status": "failed", "error": str(error)}
            )
            continue

        processed_scenes += 1
        unique_target_count += len(target_results)
        target_status_counts.update(
            result["status"] for result in target_results.values()
        )
        target_band_counts.update(
            result["coverage_band"]
            for result in target_results.values()
            if result["status"] == "ok"
        )
        scene_results.append(
            {
                "scan_id": scan_id,
                "status": "ok",
                "query_count": len(scene_queries),
                "unique_target_count": len(target_results),
            }
        )
        for query_index, query in scene_queries:
            target_id = query.get("target_id")
            result = target_results.get(target_id)
            if result is None:
                result = {
                    "status": "unavailable",
                    "error": "query has no valid integer target_id",
                }
            records.append(
                {
                    "query_index": query_index,
                    "scan_id": scan_id,
                    "target_id": target_id,
                    "target": query.get("target"),
                    "text": query.get("text"),
                    "distractor_count": len(query.get("distractor_ids", [])),
                    **result,
                }
            )
        if max_scenes is not None and processed_scenes >= max_scenes:
            break

    status_counts = Counter(record["status"] for record in records)
    band_counts = Counter(
        record["coverage_band"]
        for record in records
        if record["status"] == "ok"
    )
    return {
        "schema_version": "v1",
        "budget": budget,
        "voxel_size_m": voxel_size_m,
        "source_datasets": sorted(selected_sources),
        "summary": {
            "processed_scene_count": processed_scenes,
            "unique_target_count": unique_target_count,
            "query_count": len(records),
            "query_status_counts": dict(sorted(status_counts.items())),
            "coverage_band_counts": dict(sorted(band_counts.items())),
            "unique_target_status_counts": dict(
                sorted(target_status_counts.items())
            ),
            "unique_target_coverage_band_counts": dict(
                sorted(target_band_counts.items())
            ),
            "failed_scene_count": sum(
                result["status"] == "failed" for result in scene_results
            ),
            "skipped_scene_count": sum(
                result["status"] == "skipped" for result in scene_results
            ),
        },
        "scene_results": scene_results,
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--info-file", type=Path, required=True)
    parser.add_argument("--vg-file", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--voxel-size-m", type=float, default=0.02)
    parser.add_argument(
        "--source-dataset",
        action="append",
        choices=sorted(SOURCE_DATASETS),
        dest="source_datasets",
    )
    parser.add_argument("--max-scenes", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output
    del args.output
    options = vars(args)
    options["source_datasets"] = options["source_datasets"] or SOURCE_DATASETS
    result = run_coverage_audit(**options)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(result, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 1 if result["summary"]["failed_scene_count"] else 0
