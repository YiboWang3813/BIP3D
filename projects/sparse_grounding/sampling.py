"""Query-independent sparse camera sampling protocols."""

from __future__ import annotations

import hashlib
import random
from typing import Iterable

import numpy as np

from .camera_graph import CameraGraph
from .protocol import SparseSceneProtocol, ViewSelection


DEFAULT_BUDGETS = (3, 5, 8)


class SamplingError(RuntimeError):
    """Raised when a scene cannot satisfy a sampling protocol."""


def derive_scene_seed(global_seed: int, dataset: str, scene_id: str) -> int:
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise ValueError("global_seed must be an integer")
    if global_seed < 0:
        raise ValueError("global_seed must be nonnegative")
    if any(
        not isinstance(value, str) or not value
        for value in (dataset, scene_id)
    ):
        raise ValueError("dataset and scene_id must be non-empty strings")
    value = f"{global_seed}\0{dataset}\0{scene_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _validate_budgets(budgets: tuple[int, ...]) -> None:
    if not isinstance(budgets, tuple) or not budgets:
        raise ValueError("budgets must be a non-empty tuple")
    if any(
        isinstance(budget, bool)
        or not isinstance(budget, int)
        or budget <= 0
        for budget in budgets
    ):
        raise ValueError("budgets must contain positive integers")
    if list(budgets) != sorted(set(budgets)):
        raise ValueError("budgets must be unique and increasing")


def _validate_seed(seed: int) -> None:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer")


def _validate_duplicate_thresholds(
    translation_m: float,
    rotation_deg: float,
) -> None:
    for field, value in (
        ("min_translation_m", translation_m),
        ("min_rotation_deg", rotation_deg),
    ):
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and nonnegative")


def _is_near_duplicate(
    graph: CameraGraph,
    candidate: str,
    references: Iterable[str],
    *,
    min_translation_m: float,
    min_rotation_deg: float,
) -> bool:
    candidate_pose = graph.node(candidate).pose
    return any(
        candidate_pose.translation_distance(graph.node(reference).pose)
        <= min_translation_m
        and candidate_pose.rotation_distance_deg(graph.node(reference).pose)
        <= min_rotation_deg
        for reference in references
    )


def _make_selections(
    ordered_frame_ids: list[str],
    budgets: tuple[int, ...],
) -> tuple[ViewSelection, ...]:
    return tuple(
        ViewSelection(budget, tuple(ordered_frame_ids[:budget]))
        for budget in budgets
    )


def sample_local_connected(
    graph: CameraGraph,
    *,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    seed: int,
    min_translation_m: float = 0.05,
    min_rotation_deg: float = 2.0,
) -> tuple[ViewSelection, ...]:
    """Sample nested views by expanding the selected set's graph boundary."""
    _validate_budgets(budgets)
    _validate_seed(seed)
    _validate_duplicate_thresholds(min_translation_m, min_rotation_deg)
    max_budget = budgets[-1]
    eligible_nodes = tuple(
        frame_id
        for component in graph.connected_components()
        if len(component) >= max_budget
        for frame_id in component
    )
    if not eligible_nodes:
        raise SamplingError(
            f"no connected component can provide {max_budget} views"
        )

    rng = random.Random(seed)
    start = eligible_nodes[rng.randrange(len(eligible_nodes))]
    selected = [start]
    selected_set = {start}
    boundary = set(graph.neighbors(start))

    while len(selected) < max_budget:
        candidates = [
            frame_id
            for frame_id in sorted(boundary)
            if not _is_near_duplicate(
                graph,
                frame_id,
                selected,
                min_translation_m=min_translation_m,
                min_rotation_deg=min_rotation_deg,
            )
        ]
        if not candidates:
            raise SamplingError(
                "connected sampling exhausted non-duplicate boundary frames "
                f"after selecting {len(selected)} of {max_budget}"
            )
        chosen = candidates[rng.randrange(len(candidates))]
        selected.append(chosen)
        selected_set.add(chosen)
        boundary.discard(chosen)
        boundary.update(
            neighbor
            for neighbor in graph.neighbors(chosen)
            if neighbor not in selected_set
        )

    return _make_selections(selected, budgets)


def sample_global_fps(
    graph: CameraGraph,
    *,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    seed: int,
) -> tuple[ViewSelection, ...]:
    """Sample nested views using farthest-point camera-center coverage."""
    _validate_budgets(budgets)
    _validate_seed(seed)
    max_budget = budgets[-1]
    if len(graph.nodes) < max_budget:
        raise SamplingError(
            f"scene has {len(graph.nodes)} frames, needs {max_budget}"
        )

    frame_ids = graph.frame_ids
    centers = np.stack(
        [graph.node(frame_id).pose.center_world for frame_id in frame_ids]
    )
    rng = random.Random(seed)
    first_index = rng.randrange(len(frame_ids))
    selected_indices = [first_index]
    selected_mask = np.zeros(len(frame_ids), dtype=bool)
    selected_mask[first_index] = True
    min_distances = np.linalg.norm(centers - centers[first_index], axis=1)

    while len(selected_indices) < max_budget:
        scores = min_distances.copy()
        scores[selected_mask] = -np.inf
        next_index = int(np.argmax(scores))
        selected_indices.append(next_index)
        selected_mask[next_index] = True
        distances = np.linalg.norm(centers - centers[next_index], axis=1)
        min_distances = np.minimum(min_distances, distances)

    selected = [frame_ids[index] for index in selected_indices]
    return _make_selections(selected, budgets)


def build_heldout_pool(
    graph: CameraGraph,
    selected_frame_ids: Iterable[str],
    *,
    min_translation_m: float = 0.05,
    min_rotation_deg: float = 2.0,
) -> tuple[str, ...]:
    """Return unselected frames that are not near-duplicates of inputs."""
    _validate_duplicate_thresholds(min_translation_m, min_rotation_deg)
    selected = tuple(selected_frame_ids)
    unknown = sorted(set(selected) - set(graph.frame_ids))
    if unknown:
        raise ValueError(f"selected frames are not in camera graph: {unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_frame_ids must be unique")

    return tuple(
        frame_id
        for frame_id in graph.frame_ids
        if frame_id not in selected
        and not _is_near_duplicate(
            graph,
            frame_id,
            selected,
            min_translation_m=min_translation_m,
            min_rotation_deg=min_rotation_deg,
        )
    )


def sample_scene_protocol(
    graph: CameraGraph,
    *,
    scene_id: str,
    dataset: str,
    global_seed: int,
    protocol_version: str,
    trajectory_type: str = "local_connected",
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
) -> SparseSceneProtocol:
    """Sample one complete, versioned scene protocol."""
    scene_seed = derive_scene_seed(global_seed, dataset, scene_id)
    if trajectory_type == "local_connected":
        selections = sample_local_connected(
            graph,
            budgets=budgets,
            seed=scene_seed,
        )
    elif trajectory_type == "global_fps":
        selections = sample_global_fps(
            graph,
            budgets=budgets,
            seed=scene_seed,
        )
    else:
        raise ValueError(f"unsupported trajectory_type: {trajectory_type}")

    heldout = build_heldout_pool(graph, selections[-1].frame_ids)
    return SparseSceneProtocol(
        scene_id=scene_id,
        dataset=dataset,
        protocol_version=protocol_version,
        trajectory_type=trajectory_type,
        seed=scene_seed,
        selections=selections,
        candidate_heldout_frame_ids=heldout,
        camera_graph_stats=graph.stats(),
    )
