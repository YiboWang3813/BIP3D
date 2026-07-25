#!/usr/bin/env python3
"""Build a small shared-data fixture from trusted EmbodiedScan annotations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.protocol import SparseSceneProtocol


def _safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe dataset path: {value!r}")
    return Path(*pure.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection(protocol: SparseSceneProtocol, budget: int) -> tuple[str, ...]:
    for selection in protocol.selections:
        if selection.budget == budget:
            return selection.frame_ids
    raise ValueError(f"protocol has no budget {budget}")


def build_fixture(
    *,
    data_root: Path,
    info_file: Path,
    vg_file: Path,
    protocol_file: Path,
    output_dir: Path,
    scan_id: str,
    budget: int,
    max_queries: int,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"fixture output already exists: {output_dir}")
    if max_queries <= 0:
        raise ValueError("max_queries must be positive")

    protocol = SparseSceneProtocol.load(protocol_file)
    expected_scene_id = quote(scan_id, safe="")
    if protocol.scene_id != expected_scene_id:
        raise ValueError(
            f"protocol scene {protocol.scene_id!r} does not match {scan_id!r}"
        )
    frame_ids = _selection(protocol, budget)

    with info_file.open("rb") as stream:
        annotations = pickle.load(stream)
    if not isinstance(annotations, dict):
        raise ValueError("info annotation root must be an object")
    scenes = annotations.get("data_list")
    if not isinstance(scenes, list):
        raise ValueError("info annotation data_list must be a list")
    matches = [
        scene
        for scene in scenes
        if isinstance(scene, dict) and scene.get("sample_idx") == scan_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one scene {scan_id!r}, found {len(matches)}")
    scene = copy.deepcopy(matches[0])
    images = scene.get("images")
    if not isinstance(images, list):
        raise ValueError("scene.images must be a list")
    images_by_path = {
        image.get("img_path"): image
        for image in images
        if isinstance(image, dict)
    }
    if len(images_by_path) != len(images):
        raise ValueError("scene image paths must be unique strings")
    missing_frames = [
        frame_id for frame_id in frame_ids if frame_id not in images_by_path
    ]
    if missing_frames:
        raise ValueError(f"protocol frames are missing: {missing_frames[:3]}")
    scene["images"] = [images_by_path[frame_id] for frame_id in frame_ids]

    language = json.loads(vg_file.read_text(encoding="utf-8"))
    if not isinstance(language, list):
        raise ValueError("VG annotation root must be a list")
    queries = [
        query
        for query in language
        if isinstance(query, dict) and query.get("scan_id") == scan_id
    ][:max_queries]
    if not queries:
        raise ValueError(f"no grounding queries found for {scan_id}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.",
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        embodiedscan_dir = temporary / "embodiedscan"
        embodiedscan_dir.mkdir()
        output_info = embodiedscan_dir / info_file.name
        with output_info.open("wb") as stream:
            pickle.dump(
                {
                    "metainfo": annotations.get("metainfo", {}),
                    "data_list": [scene],
                },
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        output_vg = embodiedscan_dir / vg_file.name
        output_vg.write_text(
            f"{json.dumps(queries, ensure_ascii=True)}\n",
            encoding="utf-8",
        )

        copied_paths = []
        for image in scene["images"]:
            for key in ("img_path", "depth_path"):
                relative = _safe_relative_path(image[key])
                source = data_root / relative
                if not source.is_file():
                    raise FileNotFoundError(f"missing source data: {source}")
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied_paths.append(destination.relative_to(temporary))

        protocol_dir = temporary / "protocols"
        protocol_dir.mkdir()
        output_protocol = protocol_dir / protocol_file.name
        shutil.copy2(protocol_file, output_protocol)

        manifest_files = [
            output_info.relative_to(temporary),
            output_vg.relative_to(temporary),
            output_protocol.relative_to(temporary),
            *copied_paths,
        ]
        manifest = {
            "scan_id": scan_id,
            "budget": budget,
            "query_count": len(queries),
            "frame_ids": list(frame_ids),
            "files": [
                {
                    "path": path.as_posix(),
                    "bytes": (temporary / path).stat().st_size,
                    "sha256": _sha256(temporary / path),
                }
                for path in sorted(manifest_files)
            ],
        }
        (temporary / "fixture_manifest.json").write_text(
            f"{json.dumps(manifest, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--info-file", type=Path, required=True)
    parser.add_argument("--vg-file", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--max-queries", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_fixture(**vars(args))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
