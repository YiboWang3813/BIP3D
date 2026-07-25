#!/usr/bin/env python3
"""Validate BIP3D data before running sparse-grounding experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.data_preflight import run_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--vg-profile",
        choices=("all", "mini"),
        default="all",
        help="grounding annotation pair required by the selected config",
    )
    parser.add_argument(
        "--inspect-pickle",
        action="store_true",
        help="load trusted official pickle files and validate their references",
    )
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_preflight(
        args.data_root,
        vg_profile=args.vg_profile,
        inspect_pickle=args.inspect_pickle,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(f"{serialized}\n", encoding="utf-8")
    print(serialized)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
