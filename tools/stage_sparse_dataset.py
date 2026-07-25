#!/usr/bin/env python3
"""Dry-run, copy, or verify files in a sparse staging manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.staging import (
    StagingManifest,
    execute_staging,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="copy missing/mismatched files; default is a read-only dry run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = StagingManifest.load(args.manifest)
    report = execute_staging(
        manifest,
        source_root=args.source_root,
        destination_root=args.destination_root,
        execute=args.execute,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["source_missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
