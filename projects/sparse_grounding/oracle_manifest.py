"""Strict query-level held-out real-view oracle manifest schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


ORACLE_POLICIES = frozenset(
    {
        "random_real",
        "annotation_visible",
        "geometry_coverage",
        "grounding_gain",
    }
)


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} fields mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


@dataclass(frozen=True)
class OracleQuerySelection:
    query_id: str
    scan_id: str
    frame_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.query_id, self.scan_id)
        ):
            raise ValueError("query_id and scan_id must be non-empty strings")
        if not isinstance(self.frame_ids, tuple):
            raise ValueError("frame_ids must be an immutable tuple")
        if any(
            not isinstance(frame_id, str) or not frame_id
            for frame_id in self.frame_ids
        ):
            raise ValueError("frame_ids must contain non-empty strings")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("frame_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "scan_id": self.scan_id,
            "frame_ids": list(self.frame_ids),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "OracleQuerySelection":
        _exact_keys(value, {"query_id", "scan_id", "frame_ids"}, "record")
        frame_ids = value["frame_ids"]
        if not isinstance(frame_ids, list):
            raise ValueError("record.frame_ids must be a list")
        return cls(
            query_id=value["query_id"],
            scan_id=value["scan_id"],
            frame_ids=tuple(frame_ids),
        )


@dataclass(frozen=True)
class RealViewOracleManifest:
    policy: str
    base_view_budget: int
    oracle_view_budget: int
    trajectory_type: str
    records: tuple[OracleQuerySelection, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported oracle manifest schema_version")
        if self.policy not in ORACLE_POLICIES:
            raise ValueError(
                f"policy must be one of {sorted(ORACLE_POLICIES)}"
            )
        for field in ("base_view_budget", "oracle_view_budget"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{field} must be a positive integer")
        if not isinstance(self.trajectory_type, str) or not self.trajectory_type:
            raise ValueError("trajectory_type must be a non-empty string")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("records must be a non-empty tuple")
        query_ids = [record.query_id for record in self.records]
        if query_ids != sorted(set(query_ids)):
            raise ValueError("records must have unique, sorted query IDs")
        if any(
            len(record.frame_ids) > self.oracle_view_budget
            for record in self.records
        ):
            raise ValueError("record exceeds oracle_view_budget")

    @property
    def records_by_query_id(self) -> Mapping[str, OracleQuerySelection]:
        return MappingProxyType(
            {record.query_id: record for record in self.records}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "base_view_budget": self.base_view_budget,
            "oracle_view_budget": self.oracle_view_budget,
            "trajectory_type": self.trajectory_type,
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )

    def dump(self, path: Path) -> None:
        path.write_text(f"{self.to_json()}\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RealViewOracleManifest":
        _exact_keys(
            value,
            {
                "schema_version",
                "policy",
                "base_view_budget",
                "oracle_view_budget",
                "trajectory_type",
                "records",
            },
            "manifest",
        )
        records = value["records"]
        if not isinstance(records, list):
            raise ValueError("manifest.records must be a list")
        return cls(
            schema_version=value["schema_version"],
            policy=value["policy"],
            base_view_budget=value["base_view_budget"],
            oracle_view_budget=value["oracle_view_budget"],
            trajectory_type=value["trajectory_type"],
            records=tuple(
                sorted(
                    (
                        OracleQuerySelection.from_dict(record)
                        for record in records
                    ),
                    key=lambda record: record.query_id,
                )
            ),
        )

    @classmethod
    def load(cls, path: Path) -> "RealViewOracleManifest":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("oracle manifest root must be an object")
        return cls.from_dict(value)
