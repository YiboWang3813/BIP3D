"""Build validated BIP3D K+M held-out real-view evaluation matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .experiment_matrix import TRAJECTORY_DIRS
from .oracle_manifest import ORACLE_POLICIES, RealViewOracleManifest


def build_oracle_experiment_matrix(
    *,
    protocol_root: Path,
    oracle_root: Path,
    output_root: Path,
    checkpoint: Path,
    config: Path = Path("configs/sparse_grounding_3rscan_oracle.py"),
    slurm_script: Path = Path("scripts/slurm/sparse_grounding_eval.sbatch"),
    source_dataset: str = "3rscan",
    trajectories: Iterable[str] = ("global_fps", "local_connected"),
    policies: Iterable[str] = ("random_real", "annotation_visible"),
    base_view_budget: int = 5,
    oracle_view_budget: int = 4,
    require_inputs: bool = True,
) -> dict[str, Any]:
    """Describe oracle evaluations and validate every selection manifest."""
    trajectory_values = tuple(trajectories)
    policy_values = tuple(policies)
    unknown = set(trajectory_values) - set(TRAJECTORY_DIRS)
    if unknown:
        raise ValueError(f"unsupported trajectories: {sorted(unknown)}")
    if not policy_values:
        raise ValueError("at least one oracle policy is required")
    unknown_policies = set(policy_values) - ORACLE_POLICIES
    if unknown_policies:
        raise ValueError(
            f"unsupported oracle policies: {sorted(unknown_policies)}"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        for value in (base_view_budget, oracle_view_budget)
    ):
        raise ValueError("view budgets must be positive integers")

    paths_to_check = [checkpoint, config, slurm_script]
    experiments = []
    for trajectory in trajectory_values:
        directory = TRAJECTORY_DIRS[trajectory]
        protocol_dir = protocol_root / source_dataset / directory
        paths_to_check.append(protocol_dir)
        for policy in policy_values:
            manifest_path = oracle_root / (
                f"{source_dataset}_{directory}_{policy}_"
                f"k{base_view_budget}_m{oracle_view_budget}.json"
            )
            paths_to_check.append(manifest_path)
            manifest = (
                RealViewOracleManifest.load(manifest_path)
                if manifest_path.is_file()
                else None
            )
            if manifest is not None:
                expected = {
                    "policy": (manifest.policy, policy),
                    "trajectory_type": (
                        manifest.trajectory_type,
                        trajectory,
                    ),
                    "base_view_budget": (
                        manifest.base_view_budget,
                        base_view_budget,
                    ),
                    "oracle_view_budget": (
                        manifest.oracle_view_budget,
                        oracle_view_budget,
                    ),
                }
                mismatches = {
                    field: {"actual": actual, "expected": target}
                    for field, (actual, target) in expected.items()
                    if actual != target
                }
                if mismatches:
                    raise ValueError(
                        f"{manifest_path}: manifest mismatch {mismatches}"
                    )

            experiment_id = (
                f"bip3d-km-real-{source_dataset}-{trajectory}-"
                f"{policy}-k{base_view_budget}-m{oracle_view_budget}"
            )
            work_dir = output_root / experiment_id
            query_result_file = work_dir / "per_query_metrics.json"
            environment = {
                "BIP3D_CHECKPOINT": str(checkpoint.resolve()),
                "SPARSE_CONFIG": str(config.resolve()),
                "SPARSE_PROTOCOL_DIR": str(protocol_dir.resolve()),
                "SPARSE_VIEW_BUDGET": str(base_view_budget),
                "SPARSE_TRAJECTORY_TYPE": trajectory,
                "SPARSE_WORK_DIR": str(work_dir.resolve()),
                "SPARSE_QUERY_RESULT_FILE": str(
                    query_result_file.resolve()
                ),
                "SPARSE_ORACLE_MANIFEST": str(manifest_path.resolve()),
                "SPARSE_ORACLE_POLICY": policy,
            }
            experiments.append(
                {
                    "experiment_id": experiment_id,
                    "method": "BIP3D-K+M-Real",
                    "source_dataset": source_dataset,
                    "trajectory_type": trajectory,
                    "oracle_policy": policy,
                    "base_view_budget": base_view_budget,
                    "oracle_view_budget": oracle_view_budget,
                    "oracle_manifest": str(manifest_path.resolve()),
                    "protocol_dir": str(protocol_dir.resolve()),
                    "work_dir": str(work_dir.resolve()),
                    "query_result_file": str(query_result_file.resolve()),
                    "slurm": {
                        "script": str(slurm_script.resolve()),
                        "environment": environment,
                    },
                }
            )

    if require_inputs:
        missing = [str(path) for path in paths_to_check if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"oracle experiment matrix inputs are missing: {missing}"
            )
    return {
        "schema_version": "1.0",
        "method": "BIP3D-K+M-Real",
        "source_dataset": source_dataset,
        "experiment_count": len(experiments),
        "experiments": experiments,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
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
    parser.add_argument("--source-dataset", default="3rscan")
    parser.add_argument("--base-view-budget", type=int, default=5)
    parser.add_argument("--oracle-view-budget", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing-inputs", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output
    del args.output
    options = vars(args)
    options["require_inputs"] = not options.pop("allow_missing_inputs")
    matrix = build_oracle_experiment_matrix(**options)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(matrix, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "experiment_count": matrix["experiment_count"],
            },
            indent=2,
        )
    )
    return 0
