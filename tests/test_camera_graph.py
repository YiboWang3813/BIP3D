import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects"))

from sparse_grounding.camera_graph import (
    CameraEdge,
    CameraGraph,
    CameraNode,
    build_camera_graph,
)
from sparse_grounding.geometry import CameraPose


def translated_pose(x=0.0, rotation_deg=0.0):
    angle = np.radians(rotation_deg)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    matrix = np.array(
        (
            (cosine, -sine, 0, x),
            (sine, cosine, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ),
        dtype=float,
    )
    return CameraPose(matrix)


class CameraGraphConstructionTest(unittest.TestCase):
    def test_builds_thresholded_undirected_graph(self):
        graph = build_camera_graph(
            {
                "frame-3": translated_pose(2.0),
                "frame-1": translated_pose(0.5),
                "frame-0": translated_pose(0.0),
                "frame-2": translated_pose(1.0),
            },
            max_translation_m=0.6,
            max_rotation_deg=10,
        )

        self.assertEqual(
            graph.frame_ids,
            ("frame-0", "frame-1", "frame-2", "frame-3"),
        )
        self.assertEqual(graph.neighbors("frame-0"), ("frame-1",))
        self.assertEqual(
            graph.neighbors("frame-1"),
            ("frame-0", "frame-2"),
        )
        self.assertEqual(
            graph.connected_components(),
            (("frame-0", "frame-1", "frame-2"), ("frame-3",)),
        )

    def test_rotation_threshold_is_enforced(self):
        graph = build_camera_graph(
            {
                "a": translated_pose(rotation_deg=0),
                "b": translated_pose(rotation_deg=30),
            },
            max_translation_m=0,
            max_rotation_deg=20,
        )

        self.assertEqual(graph.edges, ())

    def test_threshold_boundary_is_inclusive(self):
        graph = build_camera_graph(
            {"a": translated_pose(0), "b": translated_pose(0.5)},
            max_translation_m=0.5,
            max_rotation_deg=0,
        )

        self.assertEqual(len(graph.edges), 1)
        self.assertAlmostEqual(graph.edges[0].translation_m, 0.5)

    def test_graph_stats_summarize_edges(self):
        graph = build_camera_graph(
            {
                "a": translated_pose(0),
                "b": translated_pose(0.5),
                "c": translated_pose(1.0),
            },
            max_translation_m=0.6,
            max_rotation_deg=0,
        )

        stats = graph.stats()

        self.assertEqual(stats.node_count, 3)
        self.assertEqual(stats.edge_count, 2)
        self.assertAlmostEqual(stats.mean_translation_m, 0.5)
        self.assertAlmostEqual(stats.max_translation_m, 0.5)

    def test_isolated_graph_has_zero_edge_stats(self):
        graph = build_camera_graph({"a": translated_pose()})

        stats = graph.stats()

        self.assertEqual(stats.edge_count, 0)
        self.assertEqual(stats.mean_translation_m, 0)


class CameraGraphValidationTest(unittest.TestCase):
    def test_graph_storage_must_be_immutable(self):
        with self.assertRaisesRegex(ValueError, "must be tuples"):
            CameraGraph(nodes=[CameraNode("a", translated_pose())], edges=[])

    def test_duplicate_or_unsorted_nodes_are_rejected(self):
        nodes = (
            CameraNode("b", translated_pose()),
            CameraNode("a", translated_pose()),
        )

        with self.assertRaisesRegex(ValueError, "unique, sorted"):
            CameraGraph(nodes=nodes, edges=())

    def test_unknown_edge_endpoint_is_rejected(self):
        nodes = (CameraNode("a", translated_pose()),)
        edge = CameraEdge("a", "b", 0.5, 0)

        with self.assertRaisesRegex(ValueError, "unknown node"):
            CameraGraph(nodes=nodes, edges=(edge,))

    def test_unknown_neighbor_query_is_explicit(self):
        graph = build_camera_graph({"a": translated_pose()})

        with self.assertRaisesRegex(KeyError, "unknown camera frame"):
            graph.neighbors("missing")

    def test_boolean_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            build_camera_graph(
                {"a": translated_pose()},
                max_translation_m=True,
            )


if __name__ == "__main__":
    unittest.main()
