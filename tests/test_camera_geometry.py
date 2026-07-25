import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects"))

from sparse_grounding.geometry import (
    CameraIntrinsics,
    CameraPose,
    backproject_depth,
    project_world,
)


def identity_pose():
    return CameraPose(np.eye(4))


class CameraIntrinsicsTest(unittest.TestCase):
    def test_matrix_and_scaling(self):
        intrinsics = CameraIntrinsics(100, 120, 50, 40, 100, 80)

        scaled = intrinsics.scaled(50, 40)

        np.testing.assert_allclose(
            scaled.matrix,
            ((50, 0, 25), (0, 60, 20), (0, 0, 1)),
        )

    def test_invalid_focal_length_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            CameraIntrinsics(0, 100, 50, 40, 100, 80)


class CameraPoseTest(unittest.TestCase):
    def test_world_to_camera_round_trip(self):
        camera_to_world = np.eye(4)
        camera_to_world[:3, 3] = (1, 2, 3)
        pose = CameraPose(camera_to_world)

        recovered = CameraPose.from_world_to_camera(pose.world_to_camera)

        np.testing.assert_allclose(
            recovered.camera_to_world,
            camera_to_world,
        )
        with self.assertRaises(ValueError):
            pose.camera_to_world[0, 0] = 2

    def test_translation_and_rotation_distance(self):
        first = identity_pose()
        second_matrix = np.array(
            (
                (0, -1, 0, 3),
                (1, 0, 0, 4),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            ),
            dtype=float,
        )
        second = CameraPose(second_matrix)

        self.assertAlmostEqual(first.translation_distance(second), 5.0)
        self.assertAlmostEqual(first.rotation_distance_deg(second), 90.0)

    def test_non_rigid_transform_is_rejected(self):
        invalid = np.eye(4)
        invalid[0, 0] = 2

        with self.assertRaisesRegex(ValueError, "orthonormal"):
            CameraPose(invalid)


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.intrinsics = CameraIntrinsics(1, 1, 0, 0, 2, 2)

    def test_backprojection_uses_depth_scale_and_pose(self):
        matrix = np.eye(4)
        matrix[:3, 3] = (10, 0, 0)
        depth_mm = np.array(((1000, 2000), (0, np.nan)))

        points, pixels = backproject_depth(
            depth_mm,
            self.intrinsics,
            CameraPose(matrix),
            depth_unit_scale_m=0.001,
        )

        np.testing.assert_allclose(points, ((10, 0, 1), (12, 0, 2)))
        np.testing.assert_allclose(pixels, ((0, 0), (1, 0)))

    def test_projection_round_trip(self):
        depth = np.array(((1, 2), (3, 4)), dtype=float)
        points, source_pixels = backproject_depth(
            depth,
            self.intrinsics,
            identity_pose(),
        )

        pixels, projected_depth, valid = project_world(
            points,
            self.intrinsics,
            identity_pose(),
        )

        np.testing.assert_allclose(pixels, source_pixels)
        np.testing.assert_allclose(projected_depth, (1, 2, 3, 4))
        np.testing.assert_array_equal(valid, (True, True, True, True))

    def test_projection_rejects_points_behind_camera(self):
        pixels, depth, valid = project_world(
            ((0, 0, -1), (0, 0, 1)),
            self.intrinsics,
            identity_pose(),
        )

        self.assertTrue(np.isnan(pixels[0]).all())
        np.testing.assert_allclose(depth, (-1, 1))
        np.testing.assert_array_equal(valid, (False, True))

    def test_valid_mask_filters_backprojection(self):
        points, pixels = backproject_depth(
            np.ones((2, 2)),
            self.intrinsics,
            identity_pose(),
            valid_mask=((False, True), (False, False)),
        )

        np.testing.assert_allclose(points, ((1, 0, 1),))
        np.testing.assert_allclose(pixels, ((1, 0),))


if __name__ == "__main__":
    unittest.main()
