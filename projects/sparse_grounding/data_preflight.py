"""Preflight checks for the BIP3D sparse-grounding datasets."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LAYOUT_PATHS = {
    "embodiedscan": Path("embodiedscan"),
    "scannet": Path("scannet/posed_images"),
    "3rscan": Path("3rscan"),
    "matterport3d": Path("matterport3d"),
}
INFO_FILES = {
    "train": "embodiedscan_infos_train.pkl",
    "val": "embodiedscan_infos_val.pkl",
}
VG_FILES = {
    "mini": {
        "train": "embodiedscan_train_mini_vg.json",
        "val": "embodiedscan_val_mini_vg.json",
    },
    "all": {
        "train": "embodiedscan_train_vg_all.json",
        "val": "embodiedscan_val_vg_all.json",
    },
}


@dataclass(frozen=True)
class PreflightIssue:
    severity: str
    code: str
    path: str
    message: str


def _issue(
    severity: str,
    code: str,
    path: Path,
    message: str,
) -> PreflightIssue:
    return PreflightIssue(severity, code, str(path), message)


def _check_directory(
    path: Path,
    dataset: str,
    *,
    require_scenes: bool = True,
) -> tuple[dict[str, Any], list[PreflightIssue]]:
    issues = []
    if path.is_symlink() and not path.exists():
        issues.append(
            _issue("error", "broken_symlink", path, "symlink target is missing")
        )
    elif not path.exists():
        issues.append(
            _issue("error", "missing_directory", path, "directory is missing")
        )
    elif not path.is_dir():
        issues.append(
            _issue("error", "not_a_directory", path, "expected a directory")
        )

    scene_count = 0
    if path.is_dir() and require_scenes:
        scene_count = sum(child.is_dir() for child in path.iterdir())
        if scene_count == 0:
            issues.append(
                _issue(
                    "error",
                    "empty_dataset",
                    path,
                    "no scene directories were found",
                )
            )
    return {
        "dataset": dataset,
        "path": str(path),
        "scene_count": scene_count,
        "ok": not issues,
    }, issues


def _check_file(path: Path) -> list[PreflightIssue]:
    if path.is_symlink() and not path.exists():
        return [
            _issue("error", "broken_symlink", path, "symlink target is missing")
        ]
    if not path.exists():
        return [_issue("error", "missing_file", path, "file is missing")]
    if not path.is_file():
        return [_issue("error", "not_a_file", path, "expected a regular file")]
    if path.stat().st_size == 0:
        return [_issue("error", "empty_file", path, "file is empty")]
    return []


def _load_vg(path: Path) -> tuple[set[str], int, list[PreflightIssue]]:
    issues = _check_file(path)
    if issues:
        return set(), 0, issues
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return set(), 0, [
            _issue("error", "invalid_json", path, f"cannot load JSON: {error}")
        ]
    if not isinstance(value, list):
        return set(), 0, [
            _issue("error", "invalid_vg_schema", path, "top level must be a list")
        ]

    scan_ids = set()
    malformed = 0
    for record in value:
        if not isinstance(record, dict):
            malformed += 1
            continue
        scan_id = record.get("scan_id")
        if not isinstance(scan_id, str) or not scan_id:
            malformed += 1
            continue
        scan_ids.add(scan_id)
    if malformed:
        issues.append(
            _issue(
                "error",
                "invalid_vg_record",
                path,
                f"{malformed} records have no non-empty string scan_id",
            )
        )
    if not value:
        issues.append(
            _issue("error", "empty_vg", path, "grounding annotation list is empty")
        )
    return scan_ids, len(value), issues


def _load_info(
    path: Path,
    data_root: Path,
) -> tuple[set[str], int, int, list[PreflightIssue]]:
    """Load a trusted official pickle and validate its BIP3D path fields."""
    issues = _check_file(path)
    if issues:
        return set(), 0, 0, issues
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream)
    except Exception as error:
        return set(), 0, 0, [
            _issue("error", "invalid_pickle", path, f"cannot load pickle: {error}")
        ]

    data_list = value.get("data_list") if isinstance(value, dict) else None
    if not isinstance(data_list, list):
        return set(), 0, 0, [
            _issue(
                "error",
                "invalid_info_schema",
                path,
                "top-level data_list must be a list",
            )
        ]

    scan_ids = set()
    references = []
    malformed = 0
    for scene in data_list:
        if not isinstance(scene, dict):
            malformed += 1
            continue
        scan_id = scene.get("sample_idx")
        images = scene.get("images")
        if not isinstance(scan_id, str) or not scan_id:
            malformed += 1
        else:
            scan_ids.add(scan_id)
        if not isinstance(images, list):
            malformed += 1
            continue
        for image in images:
            if not isinstance(image, dict):
                malformed += 1
                continue
            for key in ("img_path", "depth_path"):
                reference = image.get(key)
                if not isinstance(reference, str) or not reference:
                    malformed += 1
                else:
                    references.append(reference)

    if malformed:
        issues.append(
            _issue(
                "error",
                "invalid_info_record",
                path,
                f"{malformed} required scene or image fields are malformed",
            )
        )

    missing = [
        reference
        for reference in references
        if not (data_root / reference).is_file()
    ]
    if missing:
        examples = ", ".join(missing[:3])
        issues.append(
            _issue(
                "error",
                "missing_data_reference",
                path,
                f"{len(missing)} referenced files are missing; examples: {examples}",
            )
        )
    return scan_ids, len(data_list), len(references), issues


def run_preflight(
    data_root: Path,
    *,
    vg_profile: str = "all",
    inspect_pickle: bool = False,
) -> dict[str, Any]:
    """Check dataset layout, annotations, and optionally trusted pickle content."""
    if vg_profile not in VG_FILES:
        raise ValueError(f"unsupported VG profile: {vg_profile}")
    data_root = data_root.resolve()
    issues: list[PreflightIssue] = []
    datasets = []
    for dataset, relative_path in LAYOUT_PATHS.items():
        result, directory_issues = _check_directory(
            data_root / relative_path,
            dataset,
            require_scenes=dataset != "embodiedscan",
        )
        datasets.append(result)
        issues.extend(directory_issues)

    annotation_root = data_root / "embodiedscan"
    annotations = []
    info_scan_ids: dict[str, set[str]] = {}
    for split, filename in INFO_FILES.items():
        path = annotation_root / filename
        if inspect_pickle:
            scan_ids, records, references, file_issues = _load_info(
                path,
                data_root,
            )
            info_scan_ids[split] = scan_ids
        else:
            file_issues = _check_file(path)
            records = None
            references = None
        issues.extend(file_issues)
        annotations.append(
            {
                "path": str(path),
                "kind": "info",
                "records": records,
                "references": references,
                "ok": not file_issues,
            }
        )

    for split, filename in VG_FILES[vg_profile].items():
        path = annotation_root / filename
        scan_ids, records, file_issues = _load_vg(path)
        issues.extend(file_issues)
        annotations.append(
            {
                "path": str(path),
                "kind": "grounding",
                "records": records,
                "references": None,
                "ok": not file_issues,
            }
        )

        unknown = sorted(scan_ids - info_scan_ids.get(split, set()))
        if inspect_pickle and unknown:
            examples = ", ".join(unknown[:3])
            issues.append(
                _issue(
                    "error",
                    "unknown_vg_scan_id",
                    annotation_root,
                    f"{len(unknown)} {split} VG scan IDs are absent from the "
                    f"matching info file; "
                    f"examples: {examples}",
                )
            )

    serialized_issues = [asdict(item) for item in issues]
    error_count = sum(item.severity == "error" for item in issues)
    return {
        "data_root": str(data_root),
        "vg_profile": vg_profile,
        "pickle_inspected": inspect_pickle,
        "datasets": datasets,
        "annotations": annotations,
        "issues": serialized_issues,
        "error_count": error_count,
        "warning_count": sum(item.severity == "warning" for item in issues),
        "ok": error_count == 0,
    }
