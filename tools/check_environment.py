#!/usr/bin/env python3
"""Check the runtime prerequisites required by this BIP3D checkout."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable


SUPPORTED_PYTHON_MIN = (3, 8)
SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 12)
PINNED_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[A-Za-z0-9_.+-]+)$"
)


def parse_pinned_requirements(path: Path) -> dict[str, str]:
    """Return exact package pins from a pip requirements file."""
    requirements: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = PINNED_REQUIREMENT.fullmatch(line)
        if match is None:
            raise ValueError(f"Unsupported requirement entry: {raw_line!r}")
        requirements[match.group("name")] = match.group("version")
    return requirements


def check_python(version_info: tuple[int, ...] | None = None) -> dict[str, object]:
    version = tuple(version_info or sys.version_info[:3])
    supported = (
        SUPPORTED_PYTHON_MIN
        <= version[:2]
        < SUPPORTED_PYTHON_MAX_EXCLUSIVE
    )
    return {
        "actual": ".".join(str(part) for part in version[:3]),
        "expected": ">=3.8,<3.12",
        "ok": supported,
    }


def check_packages(
    requirements: dict[str, str],
    version_getter=importlib.metadata.version,
) -> list[dict[str, object]]:
    results = []
    for package, expected in sorted(requirements.items()):
        try:
            actual = version_getter(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        results.append(
            {
                "package": package,
                "expected": expected,
                "actual": actual,
                "ok": actual == expected,
            }
        )
    return results


def check_commands(
    commands: Iterable[str],
    command_finder=shutil.which,
) -> list[dict[str, object]]:
    results = []
    for command in commands:
        path = command_finder(command)
        results.append({"command": command, "path": path, "ok": bool(path)})
    return results


def build_report(requirements_path: Path, require_cuda: bool) -> dict[str, object]:
    packages = check_packages(parse_pinned_requirements(requirements_path))
    commands = check_commands(("nvidia-smi", "nvcc")) if require_cuda else []
    python = check_python()
    return {
        "python": python,
        "packages": packages,
        "commands": commands,
        "ok": (
            bool(python["ok"])
            and all(bool(item["ok"]) for item in packages)
            and all(bool(item["ok"]) for item in commands)
        ),
    }


def print_text_report(report: dict[str, object]) -> None:
    python = report["python"]
    assert isinstance(python, dict)
    print(
        f"[{'OK' if python['ok'] else 'FAIL'}] Python "
        f"{python['actual']} (expected {python['expected']})"
    )
    for package in report["packages"]:
        assert isinstance(package, dict)
        actual = package["actual"] or "not installed"
        print(
            f"[{'OK' if package['ok'] else 'FAIL'}] {package['package']} "
            f"{actual} (expected {package['expected']})"
        )
    for command in report["commands"]:
        assert isinstance(command, dict)
        location = command["path"] or "not found"
        print(
            f"[{'OK' if command['ok'] else 'FAIL'}] "
            f"{command['command']}: {location}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "requirements.txt",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="also require nvidia-smi and nvcc to be available",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    report = build_report(args.requirements, args.require_cuda)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
