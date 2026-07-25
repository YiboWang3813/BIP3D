import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects"))

from sparse_grounding.coverage import (
    OrientedBox9D,
    compute_visibility_coverage,
    coverage_band,
    voxel_indices,
)


class OrientedBoxTest(unittest.TestCase):
    def test_axis_aligned_point_membership(self):
        box = OrientedBox9D.from_array((0, 0, 0, 2, 4, 6, 0, 0, 0))

        mask = box.contains(((1, 2, 3), (1.01, 0, 0), (0, 0, 0)))

        np.testing.assert_array_equal(mask, (True, False, True))

    def test_z_rotation_matches_zxy_convention(self):
        box = OrientedBox9D.from_array(
            (0, 0, 0, 2, 1, 1, np.pi / 2, 0, 0)
        )

        mask = box.contains(((0, 0.9, 0), (0.9, 0, 0)))

        np.testing.assert_array_equal(mask, (True, False))

    def test_invalid_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            OrientedBox9D.from_array((0, 0, 0, 1, 0, 1, 0, 0, 0))


class VoxelCoverageTest(unittest.TestCase):
    def setUp(self):
        self.box = OrientedBox9D.from_array((2, 0, 0, 10, 2, 2, 0, 0, 0))
        self.reference = np.array(
            ((0.1, 0, 0), (1.1, 0, 0), (2.1, 0, 0), (3.1, 0, 0))
        )

    def test_unique_voxels_use_global_floor_coordinates(self):
        voxels = voxel_indices(((-0.1, 0, 0), (0.1, 0, 0), (0.2, 0, 0)), 1)

        np.testing.assert_array_equal(voxels, ((-1, 0, 0), (0, 0, 0)))

    def test_sparse_subset_coverage(self):
        result = compute_visibility_coverage(
            self.reference[:2],
            self.reference,
            self.box,
            voxel_size_m=1,
        )

        self.assertEqual(result.coverage, 0.5)
        self.assertEqual(result.sparse_voxel_count, 2)
        self.assertEqual(result.reference_voxel_count, 4)
        self.assertEqual(result.overlap_voxel_count, 2)
        self.assertEqual(result.novel_sparse_voxel_count, 0)

    def test_points_outside_target_box_are_ignored(self):
        result = compute_visibility_coverage(
            np.vstack((self.reference[:1], (20, 0, 0))),
            np.vstack((self.reference, (20, 0, 0))),
            self.box,
            voxel_size_m=1,
        )

        self.assertEqual(result.sparse_point_count, 1)
        self.assertEqual(result.reference_point_count, 4)
        self.assertEqual(result.coverage, 0.25)

    def test_non_reference_sparse_voxels_are_reported(self):
        result = compute_visibility_coverage(
            ((0.1, 0, 0), (4.1, 0, 0)),
            self.reference,
            self.box,
            voxel_size_m=1,
        )

        self.assertEqual(result.coverage, 0.25)
        self.assertEqual(result.novel_sparse_voxel_count, 1)

    def test_empty_reference_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no occupied voxels"):
            compute_visibility_coverage(
                np.empty((0, 3)),
                ((20, 0, 0),),
                self.box,
            )


class CoverageBandTest(unittest.TestCase):
    def test_fixed_protocol_boundaries(self):
        expected = {
            0.0: "unsupported",
            0.099: "unsupported",
            0.10: "severe",
            0.299: "severe",
            0.30: "moderate",
            0.599: "moderate",
            0.60: "well_observed",
            1.0: "well_observed",
        }

        for coverage, band in expected.items():
            with self.subTest(coverage=coverage):
                self.assertEqual(coverage_band(coverage), band)

    def test_out_of_range_coverage_is_rejected(self):
        for coverage in (-0.1, 1.1, np.nan):
            with self.subTest(coverage=coverage):
                with self.assertRaisesRegex(ValueError, "within"):
                    coverage_band(coverage)


if __name__ == "__main__":
    unittest.main()
