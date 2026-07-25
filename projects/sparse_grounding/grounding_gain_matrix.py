"""Build small brute-force held-out view grounding-gain matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from .oracle_manifest import (
    OracleQuerySelection,
    RealViewOracleManifest,
)
from .protocol import SparseSceneProtocol


def _diverse_query_records(
    records: tuple[OracleQuerySelection, ...],
    limit: int,
) -> tuple[OracleQuerySelection, ...]:
    selected = []
    seen_scenes = set()
    for record in records:
        if record.scan_id in seen_scenes:
            continue
        selected.append(record)
        seen_scenes.add(record.scan_id)
        if len(selected) >= limit:
            return tuple(selected)
    for record in records:
        if record in selected:
            continue
        selected.append(record)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _evenly_spaced(values: tuple[str, ...], limit: int | None) -> tuple[str, ...]:
    if limit is None or len(values) <= limit:
        return values
    if limit == 1:
        return values[:1]
    indices = [
        round(index * (len(values) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return tuple(values[index] for index in indices)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(value, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_grounding_gain_matrix(
    *,
    query_manifest: Path,
    protocol_dir: Path,
    output_root: Path,
    checkpoint: Path,
    config: Path = Path("configs/sparse_grounding_3rscan_oracle.py"),
    slurm_script: Path = Path("scripts/slurm/sparse_grounding_eval.sbatch"),
    max_queries: int = 5,
    max_candidates_per_query: int | None = 8,
    require_inputs: bool = True,
) -> dict:
    """Create one single-held-out-view evaluation per query candidate."""
    if (
        isinstance(max_queries, bool)
        or not isinstance(max_queries, int)
        or max_queries <= 0
    ):
        raise ValueError("max_queries must be a positive integer")
    if max_candidates_per_query is not None and (
        isinstance(max_candidates_per_query, bool)
        or not isinstance(max_candidates_per_query, int)
        or max_candidates_per_query <= 0
    ):
        raise ValueError("max_candidates_per_query must be positive")
    inputs = [query_manifest, protocol_dir, checkpoint, config, slurm_script]
    if require_inputs:
        missing = [str(path) for path in inputs if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"grounding-gain matrix inputs are missing: {missing}"
            )

    source = RealViewOracleManifest.load(query_manifest)
    selected_queries = _diverse_query_records(source.records, max_queries)
    manifest_dir = output_root / "grounding_gain_manifests"
    evaluation_root = output_root / "grounding_gain_eval"
    experiments = []
    candidate_counts = {}
    for query in selected_queries:
        protocol_path = protocol_dir / f"{quote(query.scan_id, safe='')}.json"
        protocol = SparseSceneProtocol.load(protocol_path)
        if protocol.trajectory_type != source.trajectory_type:
            raise ValueError(f"{protocol_path}: trajectory_type mismatch")
        candidates = _evenly_spaced(
            protocol.candidate_heldout_frame_ids,
            max_candidates_per_query,
        )
        candidate_counts[query.query_id] = len(candidates)
        query_digest = hashlib.sha256(
            query.query_id.encode("utf-8")
        ).hexdigest()[:12]
        for candidate_index, frame_id in enumerate(candidates):
            experiment_id = (
                f"grounding-gain-{query_digest}-c{candidate_index:03d}"
            )
            candidate_manifest = RealViewOracleManifest(
                policy="grounding_gain",
                base_view_budget=source.base_view_budget,
                oracle_view_budget=1,
                trajectory_type=source.trajectory_type,
                records=(
                    OracleQuerySelection(
                        query_id=query.query_id,
                        scan_id=query.scan_id,
                        frame_ids=(frame_id,),
                    ),
                ),
            )
            manifest_path = manifest_dir / f"{experiment_id}.json"
            _atomic_json(manifest_path, candidate_manifest.to_dict())
            work_dir = evaluation_root / experiment_id
            query_result_file = work_dir / "per_query_metrics.json"
            experiments.append(
                {
                    "experiment_id": experiment_id,
                    "method": "BIP3D-K+1-Real-Candidate",
                    "query_id": query.query_id,
                    "scan_id": query.scan_id,
                    "candidate_frame_id": frame_id,
                    "candidate_index": candidate_index,
                    "oracle_manifest": str(manifest_path.resolve()),
                    "work_dir": str(work_dir.resolve()),
                    "query_result_file": str(query_result_file.resolve()),
                    "slurm": {
                        "script": str(slurm_script.resolve()),
                        "environment": {
                            "BIP3D_CHECKPOINT": str(checkpoint.resolve()),
                            "SPARSE_CONFIG": str(config.resolve()),
                            "SPARSE_PROTOCOL_DIR": str(protocol_dir.resolve()),
                            "SPARSE_VIEW_BUDGET": str(
                                source.base_view_budget
                            ),
                            "SPARSE_TRAJECTORY_TYPE": source.trajectory_type,
                            "SPARSE_ORACLE_MANIFEST": str(
                                manifest_path.resolve()
                            ),
                            "SPARSE_ORACLE_POLICY": "grounding_gain",
                            "SPARSE_MISSING_ORACLE": "skip",
                            "SPARSE_WORK_DIR": str(work_dir.resolve()),
                            "SPARSE_QUERY_RESULT_FILE": str(
                                query_result_file.resolve()
                            ),
                        },
                    },
                }
            )
    return {
        "schema_version": "1.0",
        "method": "BIP3D-K+1-Real-Candidate",
        "base_view_budget": source.base_view_budget,
        "trajectory_type": source.trajectory_type,
        "query_count": len(selected_queries),
        "experiment_count": len(experiments),
        "candidate_counts": candidate_counts,
        "experiments": experiments,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sparse_grounding_3rscan_oracle.py"),
    )
    parser.add_argument(
        "--slurm-script",
        type=Path,
        default=Path("scripts/slurm/sparse_grounding_eval.sbatch"),
    )
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--max-candidates-per-query", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output
    del args.output
    matrix = build_grounding_gain_matrix(**vars(args))
    _atomic_json(output, matrix)
    print(
        json.dumps(
            {
                "output": str(output),
                "query_count": matrix["query_count"],
                "experiment_count": matrix["experiment_count"],
            },
            indent=2,
        )
    )
    return 0
