"""Camera geometry with explicit OpenCV pinhole conventions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_float_array(value: ArrayLike, shape: tuple[int, ...], field: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{field} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must contain only finite values")
    return array


def _validate_rigid_transform(matrix: FloatArray, field: str) -> None:
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
        raise ValueError(f"{field} must have homogeneous bottom row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{field} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{field} rotation determinant must be +1")


def _invert_rigid_transform(matrix: FloatArray) -> FloatArray:
    inverse = np.eye(4, dtype=np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        for field in ("fx", "fy", "cx", "cy"):
            value = getattr(self, field)
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"{field} must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx and fy must be positive")
        for field in ("width", "height"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")

    @property
    def matrix(self) -> FloatArray:
        return np.array(
            (
                (self.fx, 0.0, self.cx),
                (0.0, self.fy, self.cy),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )

    def scaled(self, width: int, height: int) -> "CameraIntrinsics":
        if width <= 0 or height <= 0:
            raise ValueError("scaled dimensions must be positive")
        scale_x = width / self.width
        scale_y = height / self.height
        return CameraIntrinsics(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            width=width,
            height=height,
        )


@dataclass(frozen=True, eq=False)
class CameraPose:
    """Rigid camera-to-world transform."""

    camera_to_world: FloatArray

    def __post_init__(self) -> None:
        matrix = _as_float_array(
            self.camera_to_world,
            (4, 4),
            "camera_to_world",
        ).copy()
        _validate_rigid_transform(matrix, "camera_to_world")
        matrix.setflags(write=False)
        object.__setattr__(self, "camera_to_world", matrix)

    @classmethod
    def from_world_to_camera(cls, world_to_camera: ArrayLike) -> "CameraPose":
        matrix = _as_float_array(
            world_to_camera,
            (4, 4),
            "world_to_camera",
        )
        _validate_rigid_transform(matrix, "world_to_camera")
        return cls(_invert_rigid_transform(matrix))

    @property
    def world_to_camera(self) -> FloatArray:
        matrix = _invert_rigid_transform(self.camera_to_world)
        matrix.setflags(write=False)
        return matrix

    @property
    def center_world(self) -> FloatArray:
        center = self.camera_to_world[:3, 3].copy()
        center.setflags(write=False)
        return center

    def translation_distance(self, other: "CameraPose") -> float:
        return float(np.linalg.norm(self.center_world - other.center_world))

    def rotation_distance_deg(self, other: "CameraPose") -> float:
        rotation = (
            self.camera_to_world[:3, :3].T
            @ other.camera_to_world[:3, :3]
        )
        cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))


def backproject_depth(
    depth: ArrayLike,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    *,
    depth_unit_scale_m: float = 1.0,
    valid_mask: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Backproject valid depth pixels into world coordinates."""
    depth_array = np.asarray(depth, dtype=np.float64)
    expected_shape = (intrinsics.height, intrinsics.width)
    if depth_array.shape != expected_shape:
        raise ValueError(
            f"depth must have shape {expected_shape}, got {depth_array.shape}"
        )
    if depth_unit_scale_m <= 0 or not np.isfinite(depth_unit_scale_m):
        raise ValueError("depth_unit_scale_m must be finite and positive")

    valid = np.isfinite(depth_array) & (depth_array > 0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != expected_shape:
            raise ValueError(
                f"valid_mask must have shape {expected_shape}, got {mask.shape}"
            )
        valid &= mask

    rows, columns = np.nonzero(valid)
    depth_m = depth_array[rows, columns] * depth_unit_scale_m
    x = (columns - intrinsics.cx) * depth_m / intrinsics.fx
    y = (rows - intrinsics.cy) * depth_m / intrinsics.fy
    points_camera = np.column_stack((x, y, depth_m))

    rotation = pose.camera_to_world[:3, :3]
    translation = pose.camera_to_world[:3, 3]
    points_world = points_camera @ rotation.T + translation
    pixels = np.column_stack((columns, rows)).astype(np.float64)
    return points_world, pixels


def project_world(
    points_world: ArrayLike,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Project world points and report positive-depth, in-frame validity."""
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points_world must contain only finite values")

    world_to_camera = pose.world_to_camera
    points_camera = (
        points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    )
    depth = points_camera[:, 2]
    positive_depth = depth > 0
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[positive_depth, 0] = (
        points_camera[positive_depth, 0]
        * intrinsics.fx
        / depth[positive_depth]
        + intrinsics.cx
    )
    pixels[positive_depth, 1] = (
        points_camera[positive_depth, 1]
        * intrinsics.fy
        / depth[positive_depth]
        + intrinsics.cy
    )
    in_frame = (
        positive_depth
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < intrinsics.width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < intrinsics.height)
    )
    return pixels, depth, in_frame
