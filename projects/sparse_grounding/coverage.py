"""Visibility coverage from RGB-D point clouds and oriented 3D boxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _rotation_matrix_zxy(angles: FloatArray) -> FloatArray:
    """Match PyTorch3D ``euler_angles_to_matrix(angles, "ZXY")``."""
    z, x, y = angles
    cos_z, sin_z = np.cos(z), np.sin(z)
    cos_x, sin_x = np.cos(x), np.sin(x)
    cos_y, sin_y = np.cos(y), np.sin(y)
    rotation_z = np.array(
        ((cos_z, -sin_z, 0), (sin_z, cos_z, 0), (0, 0, 1)),
        dtype=np.float64,
    )
    rotation_x = np.array(
        ((1, 0, 0), (0, cos_x, -sin_x), (0, sin_x, cos_x)),
        dtype=np.float64,
    )
    rotation_y = np.array(
        ((cos_y, 0, sin_y), (0, 1, 0), (-sin_y, 0, cos_y)),
        dtype=np.float64,
    )
    return rotation_z @ rotation_x @ rotation_y


def _points_array(points: ArrayLike, field: str) -> FloatArray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{field} must have shape (N, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must contain only finite values")
    return array


@dataclass(frozen=True, eq=False)
class OrientedBox9D:
    """Center, dimensions, and ZXY Euler angles in aligned world coordinates."""

    center: FloatArray
    dimensions: FloatArray
    euler_zxy: FloatArray

    def __post_init__(self) -> None:
        for field in ("center", "dimensions", "euler_zxy"):
            array = np.asarray(getattr(self, field), dtype=np.float64)
            if array.shape != (3,):
                raise ValueError(f"{field} must have shape (3,), got {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{field} must contain only finite values")
            if field == "dimensions" and np.any(array <= 0):
                raise ValueError("dimensions must be positive")
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, field, array)

    @classmethod
    def from_array(cls, box: ArrayLike) -> "OrientedBox9D":
        array = np.asarray(box, dtype=np.float64)
        if array.shape != (9,):
            raise ValueError(f"box must have shape (9,), got {array.shape}")
        return cls(array[:3], array[3:6], array[6:9])

    @property
    def rotation(self) -> FloatArray:
        rotation = _rotation_matrix_zxy(self.euler_zxy)
        rotation.setflags(write=False)
        return rotation

    def contains(
        self,
        points: ArrayLike,
        *,
        tolerance_m: float = 1e-6,
    ) -> NDArray[np.bool_]:
        if tolerance_m < 0 or not np.isfinite(tolerance_m):
            raise ValueError("tolerance_m must be finite and non-negative")
        point_array = _points_array(points, "points")
        local_points = (point_array - self.center) @ self.rotation
        half_dimensions = self.dimensions / 2.0 + tolerance_m
        return np.all(np.abs(local_points) <= half_dimensions, axis=1)


def voxel_indices(points: ArrayLike, voxel_size_m: float) -> IntArray:
    """Return sorted unique global voxel indices for finite 3D points."""
    if voxel_size_m <= 0 or not np.isfinite(voxel_size_m):
        raise ValueError("voxel_size_m must be finite and positive")
    point_array = _points_array(points, "points")
    if not len(point_array):
        return np.empty((0, 3), dtype=np.int64)
    indices = np.floor(point_array / voxel_size_m).astype(np.int64)
    return np.unique(indices, axis=0)


def _voxel_rows(voxels: IntArray) -> set[tuple[int, int, int]]:
    return {tuple(row) for row in voxels.tolist()}


@dataclass(frozen=True)
class CoverageResult:
    coverage: float
    sparse_point_count: int
    reference_point_count: int
    sparse_voxel_count: int
    reference_voxel_count: int
    overlap_voxel_count: int
    novel_sparse_voxel_count: int


def compute_visibility_coverage(
    sparse_points_world: ArrayLike,
    reference_points_world: ArrayLike,
    target_box: OrientedBox9D | ArrayLike,
    *,
    voxel_size_m: float = 0.02,
    box_tolerance_m: float = 1e-6,
) -> CoverageResult:
    """Compute target surface coverage using occupied global voxels."""
    sparse_points = _points_array(sparse_points_world, "sparse_points_world")
    reference_points = _points_array(
        reference_points_world,
        "reference_points_world",
    )
    box = (
        target_box
        if isinstance(target_box, OrientedBox9D)
        else OrientedBox9D.from_array(target_box)
    )
    sparse_target = sparse_points[
        box.contains(sparse_points, tolerance_m=box_tolerance_m)
    ]
    reference_target = reference_points[
        box.contains(reference_points, tolerance_m=box_tolerance_m)
    ]
    sparse_voxels = voxel_indices(sparse_target, voxel_size_m)
    reference_voxels = voxel_indices(reference_target, voxel_size_m)
    if not len(reference_voxels):
        raise ValueError("reference target has no occupied voxels")

    sparse_set = _voxel_rows(sparse_voxels)
    reference_set = _voxel_rows(reference_voxels)
    overlap_count = len(sparse_set & reference_set)
    novel_count = len(sparse_set - reference_set)
    return CoverageResult(
        coverage=overlap_count / len(reference_set),
        sparse_point_count=len(sparse_target),
        reference_point_count=len(reference_target),
        sparse_voxel_count=len(sparse_set),
        reference_voxel_count=len(reference_set),
        overlap_voxel_count=overlap_count,
        novel_sparse_voxel_count=novel_count,
    )


def coverage_band(coverage: float) -> str:
    """Classify coverage using the fixed Go/No-Go protocol thresholds."""
    if not np.isfinite(coverage) or coverage < 0 or coverage > 1:
        raise ValueError("coverage must be finite and within [0, 1]")
    if coverage >= 0.60:
        return "well_observed"
    if coverage >= 0.30:
        return "moderate"
    if coverage >= 0.10:
        return "severe"
    return "unsupported"
