"""Build reproducible sparse-grounding evaluation matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


TRAJECTORY_DIRS = {
    "global_fps": "global",
    "local_connected": "local",
}


def build_experiment_matrix(
    *,
    protocol_root: Path,
    output_root: Path,
    checkpoint: Path,
    config: Path,
    slurm_script: Path = Path("scripts/slurm/sparse_grounding_eval.sbatch"),
    source_dataset: str = "3rscan",
    trajectories: Iterable[str] = ("global_fps", "local_connected"),
    budgets: Iterable[int] = (3, 5, 8),
    require_inputs: bool = True,
) -> dict[str, Any]:
    """Describe the BIP3D-K evaluation grid without launching jobs."""
    trajectory_values = tuple(trajectories)
    budget_values = tuple(budgets)
    unknown = set(trajectory_values) - set(TRAJECTORY_DIRS)
    if unknown:
        raise ValueError(f"unsupported trajectories: {sorted(unknown)}")
    if (
        not budget_values
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in budget_values
        )
        or tuple(sorted(set(budget_values))) != budget_values
    ):
        raise ValueError("budgets must be unique, increasing positive integers")
    if not source_dataset:
        raise ValueError("source_dataset must be non-empty")

    paths_to_check = [checkpoint, config, slurm_script]
    experiments = []
    for trajectory in trajectory_values:
        protocol_dir = (
            protocol_root / source_dataset / TRAJECTORY_DIRS[trajectory]
        )
        paths_to_check.append(protocol_dir)
        for budget in budget_values:
            experiment_id = f"bip3d-k-{source_dataset}-{trajectory}-k{budget}"
            work_dir = output_root / experiment_id
            experiments.append(
                {
                    "experiment_id": experiment_id,
                    "method": "BIP3D-K",
                    "source_dataset": source_dataset,
                    "trajectory_type": trajectory,
                    "view_budget": budget,
                    "config": str(config.resolve()),
                    "checkpoint": str(checkpoint.resolve()),
                    "protocol_dir": str(protocol_dir.resolve()),
                    "work_dir": str(work_dir.resolve()),
                    "query_result_file": str(
                        (work_dir / "per_query_metrics.json").resolve()
                    ),
                    "slurm": {
                        "script": str(slurm_script.resolve()),
                        "environment": {
                            "BIP3D_CHECKPOINT": str(checkpoint.resolve()),
                            "SPARSE_CONFIG": str(config.resolve()),
                            "SPARSE_PROTOCOL_DIR": str(protocol_dir.resolve()),
                            "SPARSE_VIEW_BUDGET": str(budget),
                            "SPARSE_TRAJECTORY_TYPE": trajectory,
                            "SPARSE_WORK_DIR": str(work_dir.resolve()),
                            "SPARSE_QUERY_RESULT_FILE": str(
                                (
                                    work_dir / "per_query_metrics.json"
                                ).resolve()
                            ),
                        },
                    },
                }
            )

    if require_inputs:
        missing = [str(path) for path in paths_to_check if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"experiment matrix inputs are missing: {missing}"
            )
    return {
        "schema_version": "1.0",
        "method": "BIP3D-K",
        "source_dataset": source_dataset,
        "experiment_count": len(experiments),
        "experiments": experiments,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sparse_grounding_3rscan.py"),
    )
    parser.add_argument(
        "--slurm-script",
        type=Path,
        default=Path("scripts/slurm/sparse_grounding_eval.sbatch"),
    )
    parser.add_argument("--source-dataset", default="3rscan")
    parser.add_argument(
        "--trajectory",
        action="append",
        choices=tuple(TRAJECTORY_DIRS),
        dest="trajectories",
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing-inputs", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output
    trajectories = args.trajectories or tuple(TRAJECTORY_DIRS)
    del args.output, args.trajectories
    options = vars(args)
    options["trajectories"] = trajectories
    options["require_inputs"] = not options.pop("allow_missing_inputs")
    matrix = build_experiment_matrix(**options)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(matrix, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), **matrix}, indent=2))
    return 0
