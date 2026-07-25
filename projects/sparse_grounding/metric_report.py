"""Combine per-query grounding outputs into paired experiment reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .query_metrics import STRATA, summarize_query_records


PAIR_METADATA_FIELDS = (
    "scan_id",
    "target_id",
    "target",
    "is_hard",
    "is_unique",
    "is_view_dependent",
)


def _records_by_id(
    name: str,
    report: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if report.get("schema_version") != "1.0":
        raise ValueError(f"{name}: unsupported report schema")
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{name}: records must be a list")
    indexed = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"{name}: every record must be an object")
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"{name}: record has no query_id")
        if query_id in indexed:
            raise ValueError(f"{name}: duplicate query_id {query_id!r}")
        indexed[query_id] = record
    return indexed


def _validate_pair(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    query_id: str,
) -> None:
    mismatches = [
        field
        for field in PAIR_METADATA_FIELDS
        if baseline.get(field) != candidate.get(field)
    ]
    if mismatches:
        raise ValueError(
            f"paired query metadata mismatch for {query_id}: {mismatches}"
        )


def _paired_stratum_metrics(
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    threshold: float,
) -> dict[str, Any]:
    key = str(threshold)
    baseline_hits = [bool(first["hits"][key]) for first, _ in pairs]
    candidate_hits = [bool(second["hits"][key]) for _, second in pairs]
    count = len(pairs)
    baseline_count = sum(baseline_hits)
    candidate_count = sum(candidate_hits)
    beneficial = sum(
        not baseline_hit and candidate_hit
        for baseline_hit, candidate_hit in zip(
            baseline_hits,
            candidate_hits,
        )
    )
    harmful = sum(
        baseline_hit and not candidate_hit
        for baseline_hit, candidate_hit in zip(
            baseline_hits,
            candidate_hits,
        )
    )
    return {
        "count": count,
        "baseline_hit_count": baseline_count,
        "candidate_hit_count": candidate_count,
        "baseline_hit_rate": baseline_count / count if count else None,
        "candidate_hit_rate": candidate_count / count if count else None,
        "hit_rate_gain": (
            (candidate_count - baseline_count) / count if count else None
        ),
        "beneficial_count": beneficial,
        "beneficial_rate": beneficial / count if count else None,
        "harmful_count": harmful,
        "harmful_rate": harmful / count if count else None,
    }


def build_metric_report(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    baseline_name: str,
) -> dict[str, Any]:
    """Build summaries and paired deltas against a named baseline."""
    if not reports:
        raise ValueError("at least one report is required")
    if baseline_name not in reports:
        raise ValueError(f"unknown baseline {baseline_name!r}")

    thresholds_by_name = {
        name: tuple(float(value) for value in report.get("iou_thresholds", ()))
        for name, report in reports.items()
    }
    thresholds = thresholds_by_name[baseline_name]
    if not thresholds:
        raise ValueError("baseline has no iou_thresholds")
    inconsistent = {
        name: values
        for name, values in thresholds_by_name.items()
        if values != thresholds
    }
    if inconsistent:
        raise ValueError(f"inconsistent iou_thresholds: {inconsistent}")

    indexed = {
        name: _records_by_id(name, report)
        for name, report in reports.items()
    }
    experiment_summaries = {
        name: {
            "query_count": len(indexed[name]),
            "summary": summarize_query_records(
                indexed[name].values(),
                iou_thresholds=thresholds,
            ),
        }
        for name in reports
    }

    comparisons = {}
    baseline_records = indexed[baseline_name]
    for name, candidate_records in indexed.items():
        if name == baseline_name:
            continue
        common_ids = sorted(set(baseline_records) & set(candidate_records))
        pairs = []
        for query_id in common_ids:
            first = baseline_records[query_id]
            second = candidate_records[query_id]
            _validate_pair(first, second, query_id=query_id)
            pairs.append((first, second))
        by_stratum = {}
        for stratum, predicate in STRATA.items():
            selected = [
                pair for pair in pairs if predicate(pair[0])
            ]
            by_stratum[stratum] = {
                str(threshold): _paired_stratum_metrics(
                    selected,
                    threshold=threshold,
                )
                for threshold in thresholds
            }
        comparisons[name] = {
            "baseline": baseline_name,
            "candidate": name,
            "paired_query_count": len(common_ids),
            "baseline_only_count": len(baseline_records) - len(common_ids),
            "candidate_only_count": len(candidate_records) - len(common_ids),
            "by_stratum": by_stratum,
        }

    return {
        "schema_version": "1.0",
        "baseline": baseline_name,
        "iou_thresholds": list(thresholds),
        "experiment_count": len(reports),
        "experiments": experiment_summaries,
        "comparisons": comparisons,
    }


def metric_report_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten experiment rates and paired deltas for CSV export."""
    rows = []
    for name, experiment in report["experiments"].items():
        for stratum, metrics in experiment["summary"].items():
            for threshold in report["iou_thresholds"]:
                key = str(threshold)
                rows.append(
                    {
                        "row_type": "experiment",
                        "name": name,
                        "baseline": "",
                        "stratum": stratum,
                        "iou_threshold": threshold,
                        "count": metrics["count"],
                        "hit_rate": metrics[f"hit_rate@{key}"],
                        "baseline_hit_rate": "",
                        "hit_rate_gain": "",
                        "beneficial_count": "",
                        "harmful_count": "",
                    }
                )
    for name, comparison in report["comparisons"].items():
        for stratum, thresholds in comparison["by_stratum"].items():
            for threshold, metrics in thresholds.items():
                rows.append(
                    {
                        "row_type": "comparison",
                        "name": name,
                        "baseline": comparison["baseline"],
                        "stratum": stratum,
                        "iou_threshold": threshold,
                        "count": metrics["count"],
                        "hit_rate": metrics["candidate_hit_rate"],
                        "baseline_hit_rate": metrics["baseline_hit_rate"],
                        "hit_rate_gain": metrics["hit_rate_gain"],
                        "beneficial_count": metrics["beneficial_count"],
                        "harmful_count": metrics["harmful_count"],
                    }
                )
    return rows


def _parse_inputs(values: Iterable[str]) -> dict[str, Path]:
    inputs = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError("--input must use NAME=PATH")
        if name in inputs:
            raise ValueError(f"duplicate input name: {name}")
        inputs[name] = Path(path)
    return inputs


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(value, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = _parse_inputs(args.input)
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    result = build_metric_report(reports, baseline_name=args.baseline)
    _atomic_json(args.output, result)
    if args.csv_output is not None:
        _atomic_csv(args.csv_output, metric_report_rows(result))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "csv_output": (
                    str(args.csv_output)
                    if args.csv_output is not None
                    else None
                ),
                "experiment_count": result["experiment_count"],
                "comparison_count": len(result["comparisons"]),
            },
            indent=2,
        )
    )
    return 0
