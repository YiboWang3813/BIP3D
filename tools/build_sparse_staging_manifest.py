#!/usr/bin/env python3
"""Build a minimal RGB-D staging manifest from sparse experiment inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.staging import build_staging_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--info-file",
        action="append",
        type=Path,
        required=True,
        dest="info_files",
    )
    parser.add_argument(
        "--protocol-dir",
        action="append",
        type=Path,
        required=True,
        dest="protocol_dirs",
    )
    parser.add_argument(
        "--oracle-file",
        action="append",
        type=Path,
        default=[],
        dest="oracle_files",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output
    del args.output
    manifest = build_staging_manifest(**vars(args))
    manifest.dump(output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "file_count": len(manifest.files),
                "modality_counts": manifest.modality_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
