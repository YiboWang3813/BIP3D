"""Annotation-only sparse visibility and held-out oracle audit."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote

from .embodiedscan_adapter import SOURCE_DATASETS
from .protocol import SparseSceneProtocol


VIEW_DEPENDENT_WORDS = frozenset(
    {
        "front",
        "behind",
        "back",
        "left",
        "right",
        "facing",
        "leftmost",
        "rightmost",
        "looking",
        "across",
    }
)


def is_view_dependent(text: str) -> bool:
    """Match BIP3D's word-based view-dependence flag."""
    return bool(set(text.split()) & VIEW_DEPENDENT_WORDS)


def random_oracle_support_probability(
    candidate_count: int,
    visible_candidate_count: int,
    view_budget: int,
) -> float:
    """Probability that uniform sampling without replacement sees the target."""
    if (
        candidate_count < 0
        or visible_candidate_count < 0
        or visible_candidate_count > candidate_count
        or view_budget < 0
    ):
        raise ValueError("invalid oracle candidate counts or view budget")
    sample_count = min(view_budget, candidate_count)
    if sample_count == 0 or visible_candidate_count == 0:
        return 0.0
    invisible_count = candidate_count - visible_candidate_count
    if sample_count > invisible_count:
        return 1.0
    return 1.0 - (
        math.comb(invisible_count, sample_count)
        / math.comb(candidate_count, sample_count)
    )


def _selection(
    protocol: SparseSceneProtocol,
    budget: int,
) -> tuple[str, ...]:
    for selection in protocol.selections:
        if selection.budget == budget:
            return selection.frame_ids
    raise ValueError(f"protocol has no budget {budget}")


def _target_visibility(
    scene: Mapping[str, Any],
    target_id: int,
    target_name: str,
    category_names: Mapping[int, str],
) -> tuple[set[str], int]:
    instances = scene.get("instances")
    images = scene.get("images")
    if not isinstance(instances, list) or not isinstance(images, list):
        raise ValueError("scene instances and images must be lists")
    matches = [
        index
        for index, instance in enumerate(instances)
        if (
            isinstance(instance, Mapping)
            and instance.get("bbox_id") == target_id
            and category_names.get(instance.get("bbox_label_3d"), "others")
            == target_name
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one instance with bbox_id={target_id} and "
            f"target={target_name!r}, found {len(matches)}"
        )
    instance_index = matches[0]
    visible_frames = set()
    for image in images:
        if not isinstance(image, Mapping):
            raise ValueError("scene image entries must be objects")
        frame_id = image.get("img_path")
        visible_ids = image.get("visible_instance_ids")
        if not isinstance(frame_id, str):
            raise ValueError("image.img_path must be a string")
        if visible_ids is None or instance_index in visible_ids:
            visible_frames.add(frame_id)
    return visible_frames, instance_index


def evaluate_query_visibility(
    scene: Mapping[str, Any],
    query: Mapping[str, Any],
    protocol: SparseSceneProtocol,
    *,
    budgets: Iterable[int],
    oracle_view_budget: int,
    category_names: Mapping[int, str],
) -> dict[str, Any]:
    target_id = query.get("target_id")
    target_name = query.get("target")
    text = query.get("text")
    distractors = query.get("distractor_ids", [])
    if not isinstance(target_id, int) or isinstance(target_id, bool):
        raise ValueError("query.target_id must be an integer")
    if (
        not isinstance(target_name, str)
        or not isinstance(text, str)
        or not isinstance(distractors, list)
    ):
        raise ValueError("query text and distractor_ids are invalid")

    visible_frames, instance_index = _target_visibility(
        scene,
        target_id,
        target_name,
        category_names,
    )
    heldout = set(protocol.candidate_heldout_frame_ids)
    heldout_visible_count = len(visible_frames & heldout)
    random_probability = random_oracle_support_probability(
        len(heldout),
        heldout_visible_count,
        oracle_view_budget,
    )
    by_budget = {}
    for budget in budgets:
        selected = set(_selection(protocol, budget))
        sparse_visible_count = len(visible_frames & selected)
        sparse_supported = sparse_visible_count > 0
        oracle_added = min(oracle_view_budget, heldout_visible_count)
        by_budget[str(budget)] = {
            "sparse_visible_frame_count": sparse_visible_count,
            "sparse_supported": sparse_supported,
            "oracle_added_visible_frame_count": oracle_added,
            "oracle_supported": sparse_supported or oracle_added > 0,
            "random_oracle_support_probability": (
                1.0 if sparse_supported else random_probability
            ),
        }
    return {
        "target_id": target_id,
        "instance_index": instance_index,
        "target": target_name,
        "text": text,
        "is_unique": len(distractors) == 0,
        "is_hard": len(distractors) > 3,
        "is_view_dependent": is_view_dependent(text),
        "reference_visible_frame_count": len(visible_frames),
        "heldout_candidate_count": len(heldout),
        "heldout_visible_frame_count": heldout_visible_count,
        "by_budget": by_budget,
    }


def _metric_summary(rows: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {
            "count": 0,
            "sparse_supported_count": 0,
            "sparse_supported_rate": None,
            "unsupported_count": 0,
            "oracle_supported_count": 0,
            "oracle_supported_rate": None,
            "oracle_recovered_count": 0,
            "oracle_gap_recovery_rate": None,
            "random_oracle_expected_supported_count": 0.0,
            "random_oracle_expected_supported_rate": None,
        }
    values = [row["by_budget"][str(budget)] for row in rows]
    sparse_count = sum(value["sparse_supported"] for value in values)
    oracle_count = sum(value["oracle_supported"] for value in values)
    unsupported_count = count - sparse_count
    recovered_count = oracle_count - sparse_count
    random_expected = sum(
        value["random_oracle_support_probability"] for value in values
    )
    return {
        "count": count,
        "sparse_supported_count": sparse_count,
        "sparse_supported_rate": sparse_count / count,
        "unsupported_count": unsupported_count,
        "oracle_supported_count": oracle_count,
        "oracle_supported_rate": oracle_count / count,
        "oracle_recovered_count": recovered_count,
        "oracle_gap_recovery_rate": (
            recovered_count / unsupported_count if unsupported_count else None
        ),
        "random_oracle_expected_supported_count": random_expected,
        "random_oracle_expected_supported_rate": random_expected / count,
    }


def _strata(rows: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    predicates = {
        "overall": lambda row: True,
        "easy": lambda row: not row["is_hard"],
        "hard": lambda row: row["is_hard"],
        "view_dependent": lambda row: row["is_view_dependent"],
        "view_independent": lambda row: not row["is_view_dependent"],
        "unique": lambda row: row["is_unique"],
        "multi": lambda row: not row["is_unique"],
    }
    return {
        name: _metric_summary(
            [row for row in rows if predicate(row)],
            budget,
        )
        for name, predicate in predicates.items()
    }


def run_annotation_visibility_audit(
    *,
    info_file: Path,
    vg_file: Path,
    protocol_dir: Path,
    budgets: Iterable[int] = (3, 5, 8),
    oracle_view_budget: int = 4,
    source_datasets: Iterable[str] = SOURCE_DATASETS,
) -> dict[str, Any]:
    budgets = tuple(budgets)
    if (
        not budgets
        or any(isinstance(value, bool) or value <= 0 for value in budgets)
        or tuple(sorted(set(budgets))) != budgets
    ):
        raise ValueError("budgets must be unique, increasing positive integers")
    if oracle_view_budget < 0:
        raise ValueError("oracle_view_budget must be non-negative")
    sources = frozenset(source_datasets)
    unknown = sources - SOURCE_DATASETS
    if unknown:
        raise ValueError(f"unsupported source datasets: {sorted(unknown)}")

    with info_file.open("rb") as stream:
        annotation = pickle.load(stream)
    scenes = annotation.get("data_list") if isinstance(annotation, dict) else None
    if not isinstance(scenes, list):
        raise ValueError("annotation.data_list must be a list")
    metainfo = annotation.get("metainfo")
    categories = metainfo.get("categories") if isinstance(metainfo, Mapping) else None
    if not isinstance(categories, Mapping):
        raise ValueError("annotation.metainfo.categories must be an object")
    category_names = {category_id: name for name, category_id in categories.items()}
    scenes_by_id = {
        scene.get("sample_idx"): scene
        for scene in scenes
        if isinstance(scene, Mapping)
        and isinstance(scene.get("sample_idx"), str)
        and scene["sample_idx"].split("/", 1)[0] in sources
    }
    queries = json.loads(vg_file.read_text(encoding="utf-8"))
    if not isinstance(queries, list):
        raise ValueError("grounding annotation must be a list")

    protocols = {}
    for scan_id in scenes_by_id:
        path = protocol_dir / f"{quote(scan_id, safe='')}.json"
        if path.is_file():
            protocol = SparseSceneProtocol.load(path)
            if protocol.scene_id != quote(scan_id, safe=""):
                raise ValueError(f"protocol scene mismatch: {path}")
            for budget in budgets:
                _selection(protocol, budget)
            protocols[scan_id] = protocol

    records = []
    unresolved_queries = []
    for query_index, query in enumerate(queries):
        if not isinstance(query, Mapping):
            unresolved_queries.append(
                {"query_index": query_index, "error": "query is not an object"}
            )
            continue
        scan_id = query.get("scan_id")
        if scan_id not in scenes_by_id or scan_id not in protocols:
            continue
        try:
            record = evaluate_query_visibility(
                scenes_by_id[scan_id],
                query,
                protocols[scan_id],
                budgets=budgets,
                oracle_view_budget=oracle_view_budget,
                category_names=category_names,
            )
        except ValueError as error:
            unresolved_queries.append(
                {"query_index": query_index, "error": str(error)}
            )
            continue
        record.update(
            {
                "query_index": query_index,
                "scan_id": scan_id,
                "source_dataset": scan_id.split("/", 1)[0],
            }
        )
        records.append(record)

    unique_rows_by_target = {}
    for record in records:
        unique_rows_by_target.setdefault(
            (record["scan_id"], record["target_id"], record["target"]),
            record,
        )
    unique_rows = list(unique_rows_by_target.values())
    summary = {
        "protocol_scene_count": len(protocols),
        "query_count": len(records),
        "unique_target_count": len(unique_rows),
        "unresolved_query_count": len(unresolved_queries),
        "by_budget": {
            str(budget): {
                "queries": _strata(records, budget),
                "unique_targets": _strata(unique_rows, budget),
            }
            for budget in budgets
        },
    }
    return {
        "schema_version": "1.0",
        "budgets": list(budgets),
        "oracle_view_budget": oracle_view_budget,
        "source_datasets": sorted(sources),
        "summary": summary,
        "unresolved_queries": unresolved_queries,
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info-file", type=Path, required=True)
    parser.add_argument("--vg-file", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--oracle-view-budget", type=int, default=4)
    parser.add_argument(
        "--source-dataset",
        action="append",
        choices=sorted(SOURCE_DATASETS),
        dest="source_datasets",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output
    del args.output
    options = vars(args)
    options["source_datasets"] = options["source_datasets"] or SOURCE_DATASETS
    result = run_annotation_visibility_audit(**options)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(result, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0
