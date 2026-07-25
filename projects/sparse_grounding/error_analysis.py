"""Export ranked paired sparse-grounding error analysis cases."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .metric_report import build_metric_report


CASE_STATUSES = (
    "recovered",
    "harmed",
    "persistent_failure",
    "robust_success",
)


def _index_records(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("report.records must be a list")
    indexed = {}
    for record in records:
        query_id = record.get("query_id") if isinstance(record, Mapping) else None
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("every record must have a query_id")
        if query_id in indexed:
            raise ValueError(f"duplicate query_id: {query_id}")
        indexed[query_id] = record
    return indexed


def _status(baseline_hit: bool, candidate_hit: bool) -> str:
    if not baseline_hit and candidate_hit:
        return "recovered"
    if baseline_hit and not candidate_hit:
        return "harmed"
    if not baseline_hit:
        return "persistent_failure"
    return "robust_success"


def build_error_analysis(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    iou_threshold: float = 0.25,
    top_per_status: int = 50,
) -> dict[str, Any]:
    """Pair query records, classify outcomes, and retain ranked cases."""
    if (
        iou_threshold <= 0
        or iou_threshold > 1
        or isinstance(top_per_status, bool)
        or not isinstance(top_per_status, int)
        or top_per_status <= 0
    ):
        raise ValueError("invalid iou_threshold or top_per_status")
    build_metric_report(
        {"baseline": baseline, "candidate": candidate},
        baseline_name="baseline",
    )
    baseline_records = _index_records(baseline)
    candidate_records = _index_records(candidate)
    common_ids = sorted(set(baseline_records) & set(candidate_records))
    threshold_key = str(float(iou_threshold))
    cases = []
    for query_id in common_ids:
        first = baseline_records[query_id]
        second = candidate_records[query_id]
        baseline_hit = bool(first["hits"][threshold_key])
        candidate_hit = bool(second["hits"][threshold_key])
        baseline_iou = float(first["max_iou"])
        candidate_iou = float(second["max_iou"])
        cases.append(
            {
                "query_id": query_id,
                "status": _status(baseline_hit, candidate_hit),
                "scan_id": first.get("scan_id"),
                "target_id": first.get("target_id"),
                "target": first.get("target"),
                "text": first.get("text"),
                "is_hard": first.get("is_hard"),
                "is_unique": first.get("is_unique"),
                "is_view_dependent": first.get("is_view_dependent"),
                "baseline_max_iou": baseline_iou,
                "candidate_max_iou": candidate_iou,
                "iou_gain": candidate_iou - baseline_iou,
                "baseline_selected_frame_ids": first.get(
                    "selected_frame_ids"
                ),
                "candidate_selected_frame_ids": second.get(
                    "selected_frame_ids"
                ),
                "candidate_oracle_frame_ids": second.get(
                    "oracle_frame_ids"
                ),
                "baseline_top_bboxes_3d": first.get("top_bboxes_3d"),
                "candidate_top_bboxes_3d": second.get("top_bboxes_3d"),
                "gt_bboxes_3d": first.get("gt_bboxes_3d"),
            }
        )

    status_counts = Counter(case["status"] for case in cases)
    ranked = {}
    for status in CASE_STATUSES:
        selected = [case for case in cases if case["status"] == status]
        reverse = status in {"recovered", "robust_success"}
        selected.sort(
            key=lambda case: (
                case["iou_gain"],
                case["candidate_max_iou"],
                case["query_id"],
            ),
            reverse=reverse,
        )
        ranked[status] = selected[:top_per_status]
    return {
        "schema_version": "1.0",
        "iou_threshold": iou_threshold,
        "paired_query_count": len(common_ids),
        "baseline_only_count": len(baseline_records) - len(common_ids),
        "candidate_only_count": len(candidate_records) - len(common_ids),
        "status_counts": {
            status: status_counts.get(status, 0)
            for status in CASE_STATUSES
        },
        "top_per_status": top_per_status,
        "cases": ranked,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(value, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, analysis: Mapping[str, Any]) -> None:
    rows = []
    for status in CASE_STATUSES:
        for case in analysis["cases"][status]:
            rows.append(
                {
                    key: value
                    for key, value in case.items()
                    if not key.endswith("bboxes_3d")
                    and not key.endswith("frame_ids")
                }
            )
    fieldnames = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--top-per-status", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    analysis = build_error_analysis(
        baseline,
        candidate,
        iou_threshold=args.iou_threshold,
        top_per_status=args.top_per_status,
    )
    _atomic_json(args.output, analysis)
    if args.csv_output is not None:
        _atomic_csv(args.csv_output, analysis)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paired_query_count": analysis["paired_query_count"],
                "status_counts": analysis["status_counts"],
            },
            indent=2,
        )
    )
    return 0
