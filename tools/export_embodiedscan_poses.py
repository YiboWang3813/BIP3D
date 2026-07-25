#!/usr/bin/env python3
"""Export aligned camera poses from a trusted EmbodiedScan info pickle."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.embodiedscan_adapter import (
    SOURCE_DATASETS,
    load_embodiedscan_pose_manifest,
)


def _infer_dataset_name(path: Path) -> str:
    match = re.search(r"embodiedscan_infos_(train|val|test)\.pkl$", path.name)
    if match is None:
        raise ValueError(
            "--dataset-name is required when the split cannot be inferred "
            "from the info filename"
        )
    return f"embodiedscan-v1-{match.group(1)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument(
        "--source-dataset",
        action="append",
        choices=sorted(SOURCE_DATASETS),
        dest="source_datasets",
    )
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument("--max-scenes", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_embodiedscan_pose_manifest(
        args.info_file,
        dataset_name=args.dataset_name or _infer_dataset_name(args.info_file),
        source_datasets=args.source_datasets or SOURCE_DATASETS,
        scene_ids=args.scene_id,
        max_scenes=args.max_scenes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    manifest.dump(temporary)
    temporary.replace(args.output)
    frame_count = sum(len(scene.frames) for scene in manifest.scenes)
    print(
        f"wrote {len(manifest.scenes)} scenes and {frame_count} frames "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
