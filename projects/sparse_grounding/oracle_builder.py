"""Build deterministic query-level held-out real-view oracle manifests."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from .annotation_visibility import target_visibility
from .oracle_manifest import (
    OracleQuerySelection,
    RealViewOracleManifest,
)
from .protocol import SparseSceneProtocol
from .sampling import derive_scene_seed


BUILD_POLICIES = frozenset({"random_real", "annotation_visible"})


@dataclass(frozen=True)
class OracleBuildResult:
    manifest: RealViewOracleManifest
    scene_count: int
    query_count: int
    unresolved_queries: tuple[dict[str, Any], ...]

    def summary(self) -> dict[str, Any]:
        return {
            "policy": self.manifest.policy,
            "base_view_budget": self.manifest.base_view_budget,
            "oracle_view_budget": self.manifest.oracle_view_budget,
            "trajectory_type": self.manifest.trajectory_type,
            "scene_count": self.scene_count,
            "query_count": self.query_count,
            "unresolved_query_count": len(self.unresolved_queries),
            "unresolved_queries": list(self.unresolved_queries),
        }


def _sample_frames(
    candidates: tuple[str, ...],
    *,
    budget: int,
    seed: int,
) -> tuple[str, ...]:
    sample_count = min(budget, len(candidates))
    if sample_count == 0:
        return ()
    return tuple(random.Random(seed).sample(candidates, sample_count))


def build_real_view_oracle(
    *,
    info_file: Path,
    vg_file: Path,
    query_file_id: str,
    protocol_dir: Path,
    source_dataset: str,
    trajectory_type: str,
    policy: str,
    base_view_budget: int,
    oracle_view_budget: int,
    global_seed: int = 20260724,
    max_queries: int | None = None,
) -> OracleBuildResult:
    """Build random or annotation-visible held-out real-view selections."""
    if policy not in BUILD_POLICIES:
        raise ValueError(f"policy must be one of {sorted(BUILD_POLICIES)}")
    if not isinstance(query_file_id, str) or not query_file_id:
        raise ValueError("query_file_id must be a non-empty string")
    if max_queries is not None and (
        isinstance(max_queries, bool)
        or not isinstance(max_queries, int)
        or max_queries <= 0
    ):
        raise ValueError("max_queries must be a positive integer")

    with info_file.open("rb") as stream:
        annotation = pickle.load(stream)
    if not isinstance(annotation, Mapping):
        raise ValueError("annotation root must be an object")
    scenes = annotation.get("data_list")
    metainfo = annotation.get("metainfo")
    categories = metainfo.get("categories") if isinstance(metainfo, Mapping) else None
    if not isinstance(scenes, list) or not isinstance(categories, Mapping):
        raise ValueError("annotation scenes or categories are invalid")
    category_names = {
        category_id: name for name, category_id in categories.items()
    }
    scenes_by_id = {
        scene["sample_idx"]: scene
        for scene in scenes
        if isinstance(scene, Mapping)
        and isinstance(scene.get("sample_idx"), str)
        and scene["sample_idx"].split("/", 1)[0] == source_dataset
    }

    protocols = {}
    for scan_id in scenes_by_id:
        path = protocol_dir / f"{quote(scan_id, safe='')}.json"
        if not path.is_file():
            continue
        protocol = SparseSceneProtocol.load(path)
        if protocol.trajectory_type != trajectory_type:
            raise ValueError(
                f"{path}: expected trajectory_type={trajectory_type!r}"
            )
        if not any(
            selection.budget == base_view_budget
            for selection in protocol.selections
        ):
            raise ValueError(
                f"{path}: protocol has no budget {base_view_budget}"
            )
        protocols[scan_id] = protocol

    queries = json.loads(vg_file.read_text(encoding="utf-8"))
    if not isinstance(queries, list):
        raise ValueError("grounding annotation must be a list")

    scene_random_frames = {}
    records = []
    unresolved = []
    for query_index, query in enumerate(queries):
        if not isinstance(query, Mapping):
            continue
        scan_id = query.get("scan_id")
        if scan_id not in scenes_by_id or scan_id not in protocols:
            continue
        protocol = protocols[scan_id]
        heldout = tuple(protocol.candidate_heldout_frame_ids)
        scene_seed = derive_scene_seed(
            global_seed,
            source_dataset,
            f"{scan_id}\0{policy}",
        )
        if policy == "random_real":
            if scan_id not in scene_random_frames:
                scene_random_frames[scan_id] = _sample_frames(
                    heldout,
                    budget=oracle_view_budget,
                    seed=scene_seed,
                )
            selected = scene_random_frames[scan_id]
        else:
            try:
                visible_frames, _ = target_visibility(
                    scenes_by_id[scan_id],
                    query.get("target_id"),
                    query.get("target"),
                    category_names,
                )
            except ValueError as error:
                unresolved.append(
                    {
                        "query_index": query_index,
                        "scan_id": scan_id,
                        "error": str(error),
                    }
                )
                continue
            visible = tuple(
                frame_id for frame_id in heldout if frame_id in visible_frames
            )
            selected_visible = visible[:oracle_view_budget]
            remaining_budget = oracle_view_budget - len(selected_visible)
            remaining = tuple(
                frame_id
                for frame_id in heldout
                if frame_id not in set(selected_visible)
            )
            filler = _sample_frames(
                remaining,
                budget=remaining_budget,
                seed=scene_seed,
            )
            selected = (*selected_visible, *filler)

        records.append(
            OracleQuerySelection(
                query_id=f"{query_file_id}:{query_index}",
                scan_id=scan_id,
                frame_ids=tuple(selected),
            )
        )
        if max_queries is not None and len(records) >= max_queries:
            break

    manifest = RealViewOracleManifest(
        policy=policy,
        base_view_budget=base_view_budget,
        oracle_view_budget=oracle_view_budget,
        trajectory_type=trajectory_type,
        records=tuple(sorted(records, key=lambda record: record.query_id)),
    )
    return OracleBuildResult(
        manifest=manifest,
        scene_count=len({record.scan_id for record in records}),
        query_count=len(records),
        unresolved_queries=tuple(unresolved),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info-file", type=Path, required=True)
    parser.add_argument("--vg-file", type=Path, required=True)
    parser.add_argument("--query-file-id", required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--trajectory-type", required=True)
    parser.add_argument("--policy", choices=sorted(BUILD_POLICIES), required=True)
    parser.add_argument("--base-view-budget", type=int, default=5)
    parser.add_argument("--oracle-view-budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    return parser


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output
    summary_output = args.summary_output or output.with_suffix(".summary.json")
    options = vars(args)
    del options["output"], options["summary_output"]
    options["global_seed"] = options.pop("seed")
    result = build_real_view_oracle(**options)
    _atomic_write(output, f"{result.manifest.to_json()}\n")
    summary = result.summary()
    _atomic_write(
        summary_output,
        f"{json.dumps(summary, indent=2, sort_keys=True)}\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
