"""Deterministic camera graph construction from rigid poses."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .geometry import CameraPose
from .protocol import CameraGraphStats


@dataclass(frozen=True)
class CameraNode:
    frame_id: str
    pose: CameraPose

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id must be a non-empty string")
        if not isinstance(self.pose, CameraPose):
            raise ValueError("pose must be a CameraPose")


@dataclass(frozen=True)
class CameraEdge:
    source: str
    target: str
    translation_m: float
    rotation_deg: float

    def __post_init__(self) -> None:
        if any(
            not isinstance(frame_id, str) or not frame_id
            for frame_id in (self.source, self.target)
        ):
            raise ValueError("edge endpoints must be non-empty strings")
        if self.source >= self.target:
            raise ValueError("edge endpoints must be ordered and distinct")
        for field_name in ("translation_m", "rotation_deg"):
            value = getattr(self, field_name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and nonnegative")


@dataclass(frozen=True)
class CameraGraph:
    nodes: tuple[CameraNode, ...]
    edges: tuple[CameraEdge, ...]
    _node_by_id: Mapping[str, CameraNode] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _neighbors: Mapping[str, tuple[str, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, tuple) or not isinstance(self.edges, tuple):
            raise ValueError("camera graph nodes and edges must be tuples")
        if not self.nodes:
            raise ValueError("camera graph must contain at least one node")
        frame_ids = [node.frame_id for node in self.nodes]
        if frame_ids != sorted(set(frame_ids)):
            raise ValueError("camera graph nodes must have unique, sorted IDs")

        node_by_id = {node.frame_id: node for node in self.nodes}
        neighbors: dict[str, set[str]] = {
            frame_id: set() for frame_id in frame_ids
        }
        edge_keys = []
        for edge in self.edges:
            key = (edge.source, edge.target)
            edge_keys.append(key)
            if edge.source not in node_by_id or edge.target not in node_by_id:
                raise ValueError(f"edge references unknown node: {key}")
            neighbors[edge.source].add(edge.target)
            neighbors[edge.target].add(edge.source)
        if edge_keys != sorted(set(edge_keys)):
            raise ValueError("camera graph edges must be unique and sorted")

        immutable_neighbors = {
            frame_id: tuple(sorted(values))
            for frame_id, values in neighbors.items()
        }
        object.__setattr__(
            self,
            "_node_by_id",
            MappingProxyType(node_by_id),
        )
        object.__setattr__(
            self,
            "_neighbors",
            MappingProxyType(immutable_neighbors),
        )

    @property
    def frame_ids(self) -> tuple[str, ...]:
        return tuple(node.frame_id for node in self.nodes)

    def node(self, frame_id: str) -> CameraNode:
        try:
            return self._node_by_id[frame_id]
        except KeyError as error:
            raise KeyError(f"unknown camera frame: {frame_id}") from error

    def neighbors(self, frame_id: str) -> tuple[str, ...]:
        try:
            return self._neighbors[frame_id]
        except KeyError as error:
            raise KeyError(f"unknown camera frame: {frame_id}") from error

    def connected_components(self) -> tuple[tuple[str, ...], ...]:
        remaining = set(self.frame_ids)
        components = []
        while remaining:
            start = min(remaining)
            frontier = [start]
            component = set()
            while frontier:
                frame_id = frontier.pop()
                if frame_id in component:
                    continue
                component.add(frame_id)
                frontier.extend(self.neighbors(frame_id))
            remaining.difference_update(component)
            components.append(tuple(sorted(component)))
        return tuple(components)

    def stats(self) -> CameraGraphStats:
        if not self.edges:
            return CameraGraphStats(
                node_count=len(self.nodes),
                edge_count=0,
                mean_translation_m=0.0,
                max_translation_m=0.0,
                mean_rotation_deg=0.0,
            )
        translations = np.array(
            [edge.translation_m for edge in self.edges],
            dtype=np.float64,
        )
        rotations = np.array(
            [edge.rotation_deg for edge in self.edges],
            dtype=np.float64,
        )
        return CameraGraphStats(
            node_count=len(self.nodes),
            edge_count=len(self.edges),
            mean_translation_m=float(translations.mean()),
            max_translation_m=float(translations.max()),
            mean_rotation_deg=float(rotations.mean()),
        )


def build_camera_graph(
    poses: Mapping[str, CameraPose],
    *,
    max_translation_m: float = 0.65,
    max_rotation_deg: float = 35.0,
) -> CameraGraph:
    """Connect camera pairs satisfying both pose-distance thresholds."""
    for field_name, value in (
        ("max_translation_m", max_translation_m),
        ("max_rotation_deg", max_rotation_deg),
    ):
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{field_name} must be finite and nonnegative")
    if not poses:
        raise ValueError("poses cannot be empty")

    nodes = tuple(
        CameraNode(frame_id, poses[frame_id])
        for frame_id in sorted(poses)
    )
    edges = []
    for first, second in combinations(nodes, 2):
        translation = first.pose.translation_distance(second.pose)
        if translation > max_translation_m:
            continue
        rotation = first.pose.rotation_distance_deg(second.pose)
        if rotation > max_rotation_deg:
            continue
        edges.append(
            CameraEdge(
                source=first.frame_id,
                target=second.frame_id,
                translation_m=translation,
                rotation_deg=rotation,
            )
        )
    return CameraGraph(nodes=nodes, edges=tuple(edges))
