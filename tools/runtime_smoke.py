#!/usr/bin/env python3
"""Run a small CUDA computation and verify BIP3D extension imports."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from typing import Any


REQUIRED_PACKAGES = ("torch", "torchvision", "mmcv", "mmdet", "mmengine")
CUDA_EXTENSIONS = (
    "deformable_aggregation_ext",
    "deformable_aggregation_with_depth_ext",
)


def validate_report(
    report: dict[str, Any],
    *,
    require_extensions: bool,
) -> list[str]:
    issues = []
    if not report["cuda"]["available"]:
        issues.append("CUDA is not available to PyTorch")
    if report["cuda"]["device_count"] < 1:
        issues.append("PyTorch reports no CUDA devices")
    if report["cuda"]["matmul_sum"] is None:
        issues.append("CUDA matrix multiplication did not complete")
    if require_extensions:
        missing = [
            name
            for name, status in report["extensions"].items()
            if not status["ok"]
        ]
        if missing:
            issues.append(
                "BIP3D CUDA extensions are unavailable: " + ", ".join(missing)
            )
    return issues


def collect_report() -> dict[str, Any]:
    torch = importlib.import_module("torch")
    packages = {}
    for package in REQUIRED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    devices = []
    matmul_sum = None
    if cuda_available:
        devices = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(device_count)
        ]
        left = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
        right = torch.eye(4, device="cuda")
        matmul_sum = float((left @ right).sum().item())

    extensions = {}
    for module_name in CUDA_EXTENSIONS:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            extensions[module_name] = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        else:
            extensions[module_name] = {
                "ok": True,
                "file": getattr(module, "__file__", None),
            }

    return {
        "packages": packages,
        "cuda": {
            "available": cuda_available,
            "torch_cuda_version": torch.version.cuda,
            "device_count": device_count,
            "devices": devices,
            "matmul_sum": matmul_sum,
        },
        "extensions": extensions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-extensions",
        action="store_true",
        help="fail unless both custom CUDA extensions can be imported",
    )
    args = parser.parse_args()

    report = collect_report()
    issues = validate_report(
        report,
        require_extensions=args.require_extensions,
    )
    report["issues"] = issues
    report["ok"] = not issues
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
