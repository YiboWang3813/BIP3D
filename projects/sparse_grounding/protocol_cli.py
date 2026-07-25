"""Batch generation for versioned sparse-view scene protocols."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .camera_graph import build_camera_graph
from .pose_manifest import PoseManifest
from .sampling import SamplingError, sample_scene_protocol


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_protocol(
    path: Path,
    content: str,
    *,
    overwrite: bool,
) -> str:
    serialized = f"{content}\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return "unchanged"
        if not overwrite:
            raise ValueError(
                f"output already exists with different content: {path}"
            )
    _atomic_write(path, serialized)
    return "generated"


def generate_protocols(
    manifest: PoseManifest,
    output_dir: Path,
    *,
    trajectory_type: str,
    budgets: tuple[int, ...],
    global_seed: int,
    protocol_version: str,
    max_translation_m: float,
    max_rotation_deg: float,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for scene in manifest.scenes:
        output_path = output_dir / f"{scene.scene_id}.json"
        graph = None
        try:
            graph = build_camera_graph(
                scene.poses,
                max_translation_m=max_translation_m,
                max_rotation_deg=max_rotation_deg,
            )
            protocol = sample_scene_protocol(
                graph,
                scene_id=scene.scene_id,
                dataset=manifest.dataset,
                global_seed=global_seed,
                protocol_version=protocol_version,
                trajectory_type=trajectory_type,
                budgets=budgets,
            )
            status = _write_protocol(
                output_path,
                protocol.to_json(),
                overwrite=overwrite,
            )
        except (SamplingError, ValueError) as error:
            failure = {
                "scene_id": scene.scene_id,
                "status": "failed",
                "error": str(error),
            }
            if graph is not None:
                component_sizes = sorted(
                    (len(component) for component in graph.connected_components()),
                    reverse=True,
                )
                failure["graph_node_count"] = len(graph.nodes)
                failure["graph_edge_count"] = len(graph.edges)
                failure["largest_connected_component_size"] = (
                    component_sizes[0] if component_sizes else 0
                )
            results.append(failure)
        else:
            results.append(
                {
                    "scene_id": scene.scene_id,
                    "status": status,
                    "output": str(output_path),
                }
            )

    failure_count = sum(
        result["status"] == "failed" for result in results
    )
    failure_reasons = Counter(
        result["error"]
        for result in results
        if result["status"] == "failed"
    )
    return {
        "dataset": manifest.dataset,
        "trajectory_type": trajectory_type,
        "protocol_version": protocol_version,
        "global_seed": global_seed,
        "budgets": list(budgets),
        "camera_graph": {
            "max_translation_m": max_translation_m,
            "max_rotation_deg": max_rotation_deg,
        },
        "scene_count": len(results),
        "success_count": len(results) - failure_count,
        "failure_count": failure_count,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trajectory-type",
        choices=("local_connected", "global_fps"),
        default="local_connected",
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--protocol-version", default="v1")
    parser.add_argument("--max-translation-m", type=float, default=0.65)
    parser.add_argument("--max-rotation-deg", type=float, default=35.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-scene-failures",
        action="store_true",
        help="return success after recording scenes that cannot meet the budgets",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = PoseManifest.load(args.input)
    summary = generate_protocols(
        manifest,
        args.output_dir,
        trajectory_type=args.trajectory_type,
        budgets=tuple(args.budgets),
        global_seed=args.seed,
        protocol_version=args.protocol_version,
        max_translation_m=args.max_translation_m,
        max_rotation_deg=args.max_rotation_deg,
        overwrite=args.overwrite,
    )
    serialized = json.dumps(
        summary,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    )
    _atomic_write(
        args.output_dir / "generation_summary.json",
        f"{serialized}\n",
    )
    print(serialized)
    return (
        1
        if summary["failure_count"] and not args.allow_scene_failures
        else 0
    )
