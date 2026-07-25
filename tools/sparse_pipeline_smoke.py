#!/usr/bin/env python3
"""Validate a sparse grounding fixture through preprocessing and CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_fixture_manifest(
    fixture_root: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Load a fixture manifest and verify every declared file."""
    fixture_root = fixture_root.resolve()
    manifest_path = manifest_path or fixture_root / "fixture_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("fixture manifest must contain a non-empty files list")

    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("fixture manifest file entries must be objects")
        relative_path = entry.get("path")
        expected_bytes = entry.get("bytes")
        expected_sha256 = entry.get("sha256")
        if (
            not isinstance(relative_path, str)
            or not isinstance(expected_bytes, int)
            or not isinstance(expected_sha256, str)
        ):
            raise ValueError(f"invalid fixture manifest entry: {entry!r}")
        path = (fixture_root / relative_path).resolve()
        if not path.is_relative_to(fixture_root):
            raise ValueError(f"fixture file escapes fixture root: {relative_path}")
        if not path.is_file():
            raise FileNotFoundError(f"missing fixture file: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"fixture size mismatch for {relative_path}: "
                f"expected {expected_bytes}, got {actual_bytes}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"fixture SHA256 mismatch for {relative_path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
    return manifest


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    fixture_root = args.fixture_root.resolve()
    protocol_dir = (
        args.protocol_dir.resolve()
        if args.protocol_dir
        else fixture_root / "protocols"
    )
    manifest = verify_fixture_manifest(fixture_root, args.fixture_manifest)

    os.environ["SPARSE_PROTOCOL_DIR"] = str(protocol_dir)
    os.environ["SPARSE_VIEW_BUDGET"] = str(args.budget)
    os.environ["SPARSE_TRAJECTORY_TYPE"] = args.trajectory_type

    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.utils import import_modules_from_strings

    from bip3d.registry import DATASETS

    cfg = Config.fromfile(str(args.config))
    import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get("default_scope", "bip3d"))
    dataset_cfg = cfg.test_dataloader.dataset
    dataset_cfg.data_root = str(fixture_root)
    dataset_cfg.protocol_dir = str(protocol_dir)
    dataset = DATASETS.build(dataset_cfg)

    expected_queries = manifest.get("query_count")
    if len(dataset) != expected_queries:
        raise ValueError(
            f"expected {expected_queries} fixture queries, got {len(dataset)}"
        )
    if len(dataset) != 1:
        raise ValueError("pipeline smoke requires exactly one fixture query")

    sample = dataset[0]
    image = sample["inputs"]["img"]
    depth = sample["inputs"]["depth_img"]
    expected_image_shape = (args.budget, 3, 512, 512)
    expected_depth_shape = (args.budget, 1, 512, 512)
    if tuple(image.shape) != expected_image_shape:
        raise ValueError(
            f"expected image shape {expected_image_shape}, got {tuple(image.shape)}"
        )
    if tuple(depth.shape) != expected_depth_shape:
        raise ValueError(
            f"expected depth shape {expected_depth_shape}, got {tuple(depth.shape)}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    image_device = image.to(device=device, dtype=torch.float32)
    depth_device = depth.to(device=device, dtype=torch.float32)
    finite = bool(
        torch.isfinite(image_device).all().item()
        and torch.isfinite(depth_device).all().item()
    )
    if not finite:
        raise ValueError("preprocessed fixture tensors contain non-finite values")
    image_sum = float(image_device.sum().item())
    depth_sum = float(depth_device.sum().item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    metainfo = sample["data_samples"].metainfo
    scan_id = metainfo.get("scan_id")
    if scan_id != manifest.get("scan_id"):
        raise ValueError(
            f"expected scan_id {manifest.get('scan_id')!r}, got {scan_id!r}"
        )
    selected_frame_paths = metainfo.get("img_path")
    if not isinstance(selected_frame_paths, list):
        raise ValueError("pipeline metainfo has no img_path list")
    selected_frames = [
        str(Path(path).resolve().relative_to(fixture_root))
        for path in selected_frame_paths
    ]
    if selected_frames != manifest.get("frame_ids"):
        raise ValueError("pipeline frame order differs from fixture manifest")

    eval_ann_info = sample["data_samples"].eval_ann_info
    query_id = eval_ann_info.get("sparse_query_id")
    query_index = eval_ann_info.get("sparse_query_index")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("pipeline evaluation annotation has no stable query ID")
    if query_index != 0 or not query_id.endswith(":0"):
        raise ValueError(
            f"unexpected fixture query identity: {query_id!r}, {query_index!r}"
        )

    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else str(device)
    )
    return {
        "ok": True,
        "fixture_root": str(fixture_root),
        "verified_files": len(manifest["files"]),
        "scan_id": scan_id,
        "query_count": len(dataset),
        "query_id": query_id,
        "frame_ids": selected_frames,
        "image": {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "sum": image_sum,
        },
        "depth": {
            "shape": list(depth.shape),
            "dtype": str(depth.dtype),
            "sum": depth_sum,
        },
        "device": {
            "requested": str(device),
            "name": device_name,
            "torch_cuda_version": torch.version.cuda,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sparse_grounding_3rscan.py"),
    )
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--trajectory-type", default="global_fps")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    report = run_smoke(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
