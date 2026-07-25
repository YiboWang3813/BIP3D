"""Per-query sparse grounding metrics and reproducible result export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from bip3d.eval.metrics.grounding_metric import GroundingMetric
from bip3d.registry import METRICS
from bip3d.structures import EulerDepthInstance3DBoxes


STRATA = {
    "overall": lambda record: True,
    "easy": lambda record: not record["is_hard"],
    "hard": lambda record: record["is_hard"],
    "view_dependent": lambda record: record["is_view_dependent"],
    "view_independent": lambda record: not record["is_view_dependent"],
    "unique": lambda record: record["is_unique"],
    "multi": lambda record: not record["is_unique"],
}


def _atomic_json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(value, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_query_record(
    annotation: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    iou_thresholds: Iterable[float] = (0.25, 0.5),
    top_k: int = 10,
) -> dict[str, Any]:
    """Build the official top-k hit metrics for one grounding query."""
    thresholds = tuple(float(value) for value in iou_thresholds)
    if not thresholds or any(value <= 0 or value > 1 for value in thresholds):
        raise ValueError("iou_thresholds must be in (0, 1]")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    scores = prediction["target_scores_3d"]
    predicted_boxes = prediction["bboxes_3d"]
    ground_truth_boxes = annotation["gt_bboxes_3d"]
    indices = scores.argsort(dim=-1, descending=True)[:top_k]
    top_scores = scores[indices]
    top_boxes = EulerDepthInstance3DBoxes(
        predicted_boxes[indices].tensor,
        origin=(0.5, 0.5, 0.5),
    )
    gt_boxes = EulerDepthInstance3DBoxes(
        ground_truth_boxes.tensor,
        origin=(0.5, 0.5, 0.5),
    )
    overlaps = top_boxes.overlaps(top_boxes, gt_boxes)
    max_iou = float(overlaps.max().item()) if overlaps.numel() else 0.0

    query_id = annotation.get("sparse_query_id")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("annotation is missing sparse_query_id")
    record = {
        "query_id": query_id,
        "query_file": annotation.get("sparse_query_file"),
        "query_index": annotation.get("sparse_query_index"),
        "scan_id": annotation.get("scan_id"),
        "target_id": annotation.get("target_id"),
        "target": annotation.get("target"),
        "text": annotation.get("text"),
        "trajectory_type": annotation.get("trajectory_type"),
        "base_view_budget": annotation.get("base_view_budget"),
        "selected_frame_ids": annotation.get("selected_frame_ids"),
        "oracle_policy": annotation.get("oracle_policy"),
        "oracle_view_budget": annotation.get("oracle_view_budget"),
        "oracle_frame_ids": annotation.get("oracle_frame_ids"),
        "is_hard": bool(annotation.get("is_hard", False)),
        "is_unique": bool(annotation.get("is_unique", False)),
        "is_view_dependent": bool(annotation.get("is_view_dep", False)),
        "top_k": top_k,
        "max_iou": max_iou,
        "hits": {
            str(threshold): max_iou > threshold for threshold in thresholds
        },
        "top_scores": top_scores.tolist(),
        "top_bboxes_3d": top_boxes.tensor.tolist(),
        "gt_bboxes_3d": ground_truth_boxes.tensor.tolist(),
    }
    return record


def summarize_query_records(
    records: Iterable[Mapping[str, Any]],
    *,
    iou_thresholds: Iterable[float] = (0.25, 0.5),
) -> dict[str, Any]:
    """Summarize per-query hit rates with BIP3D's standard strata."""
    rows = list(records)
    thresholds = tuple(float(value) for value in iou_thresholds)
    summary = {}
    for name, predicate in STRATA.items():
        selected = [record for record in rows if predicate(record)]
        metrics = {"count": len(selected)}
        for threshold in thresholds:
            key = str(threshold)
            hit_count = sum(bool(record["hits"][key]) for record in selected)
            metrics[f"hit_count@{key}"] = hit_count
            metrics[f"hit_rate@{key}"] = (
                hit_count / len(selected) if selected else None
            )
        summary[name] = metrics
    return summary


@METRICS.register_module()
class SparseGroundingMetric(GroundingMetric):
    """Official BIP3D metric with an optional atomic per-query JSON export."""

    def __init__(self, *args, query_result_file: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_result_file = (
            Path(query_result_file) if query_result_file is not None else None
        )

    def compute_metrics(self, results: list) -> dict[str, float]:
        metrics = super().compute_metrics(results)
        if self.format_only or self.query_result_file is None:
            return metrics

        records = [
            build_query_record(
                annotation,
                prediction,
                iou_thresholds=self.iou_thr,
            )
            for annotation, prediction in results
        ]
        report = {
            "schema_version": "1.0",
            "metric": "BIP3D top-10 grounding hit rate",
            "iou_thresholds": list(self.iou_thr),
            "query_count": len(records),
            "summary": summarize_query_records(
                records,
                iou_thresholds=self.iou_thr,
            ),
            "records": records,
        }
        _atomic_json_dump(self.query_result_file, report)
        return metrics
