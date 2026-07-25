import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects"))

from sparse_grounding.camera_graph import build_camera_graph
from sparse_grounding.geometry import CameraPose
from sparse_grounding.sampling import (
    SamplingError,
    build_heldout_pool,
    derive_scene_seed,
    sample_global_fps,
    sample_local_connected,
    sample_scene_protocol,
)


def pose_at(x, rotation_deg=0):
    angle = np.radians(rotation_deg)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    matrix = np.array(
        (
            (cosine, -sine, 0, x),
            (sine, cosine, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )
    return CameraPose(matrix)


def line_graph(count=10, spacing=0.5):
    return build_camera_graph(
        {f"{index:02d}": pose_at(index * spacing) for index in range(count)},
        max_translation_m=spacing + 1e-6,
        max_rotation_deg=1,
    )


class SeedTest(unittest.TestCase):
    def test_scene_seed_is_stable_and_namespaced(self):
        first = derive_scene_seed(20260724, "scannet", "scene0000_00")

        self.assertEqual(
            first,
            derive_scene_seed(20260724, "scannet", "scene0000_00"),
        )
        self.assertNotEqual(
            first,
            derive_scene_seed(20260724, "3rscan", "scene0000_00"),
        )

    def test_invalid_seed_and_namespace_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            derive_scene_seed(-1, "scannet", "scene0000_00")
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            derive_scene_seed(1, "scannet", 123)


class LocalConnectedSamplingTest(unittest.TestCase):
    def test_is_reproducible_nested_and_connected(self):
        graph = line_graph()

        first = sample_local_connected(
            graph,
            budgets=(3, 5, 8),
            seed=42,
        )
        second = sample_local_connected(
            graph,
            budgets=(3, 5, 8),
            seed=42,
        )

        self.assertEqual(first, second)
        self.assertEqual(first[0].frame_ids, first[2].frame_ids[:3])
        self.assertEqual(first[1].frame_ids, first[2].frame_ids[:5])
        ordered = first[-1].frame_ids
        for index, frame_id in enumerate(ordered[1:], start=1):
            self.assertTrue(
                set(graph.neighbors(frame_id)).intersection(ordered[:index])
            )

    def test_requires_one_large_enough_component(self):
        graph = build_camera_graph(
            {
                "a": pose_at(0),
                "b": pose_at(0.5),
                "c": pose_at(5),
                "d": pose_at(5.5),
            },
            max_translation_m=0.6,
            max_rotation_deg=1,
        )

        with self.assertRaisesRegex(SamplingError, "connected component"):
            sample_local_connected(graph, budgets=(3,), seed=1)

    def test_near_duplicate_filter_can_reject_scene(self):
        graph = build_camera_graph(
            {
                "a": pose_at(0),
                "b": pose_at(0.01),
                "c": pose_at(0.02),
            },
            max_translation_m=1,
            max_rotation_deg=1,
        )

        with self.assertRaisesRegex(SamplingError, "non-duplicate"):
            sample_local_connected(
                graph,
                budgets=(2,),
                seed=1,
                min_translation_m=0.05,
                min_rotation_deg=2,
            )

    def test_direct_sampling_rejects_negative_seed(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            sample_local_connected(
                line_graph(),
                budgets=(3,),
                seed=-1,
            )


class GlobalFpsSamplingTest(unittest.TestCase):
    def test_is_reproducible_and_uses_unique_frames(self):
        graph = line_graph()

        selections = sample_global_fps(
            graph,
            budgets=(3, 5, 8),
            seed=10,
        )

        self.assertEqual(
            selections,
            sample_global_fps(
                graph,
                budgets=(3, 5, 8),
                seed=10,
            ),
        )
        self.assertEqual(len(set(selections[-1].frame_ids)), 8)
        self.assertEqual(
            selections[0].frame_ids,
            selections[-1].frame_ids[:3],
        )

    def test_rejects_insufficient_scene(self):
        with self.assertRaisesRegex(SamplingError, "needs 8"):
            sample_global_fps(
                line_graph(count=5),
                budgets=(3, 5, 8),
                seed=1,
            )


class HeldoutAndProtocolTest(unittest.TestCase):
    def test_heldout_pool_excludes_near_duplicates(self):
        graph = build_camera_graph(
            {
                "selected": pose_at(0),
                "duplicate": pose_at(0.01),
                "rotated": pose_at(0.01, rotation_deg=10),
                "far": pose_at(1),
            },
            max_translation_m=2,
            max_rotation_deg=20,
        )

        heldout = build_heldout_pool(graph, ("selected",))

        self.assertEqual(heldout, ("far", "rotated"))

    def test_complete_protocol_has_no_query_specific_input(self):
        protocol = sample_scene_protocol(
            line_graph(count=12),
            scene_id="scene0000_00",
            dataset="scannet",
            global_seed=20260724,
            protocol_version="local-v1",
            trajectory_type="local_connected",
        )

        selected = set(protocol.selections[-1].frame_ids)
        self.assertTrue(
            selected.isdisjoint(protocol.candidate_heldout_frame_ids)
        )
        self.assertNotIn("query_id", protocol.to_dict())
        self.assertEqual(protocol.camera_graph_stats.node_count, 12)


if __name__ == "__main__":
    unittest.main()
