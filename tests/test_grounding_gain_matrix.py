import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.grounding_gain_matrix import (
    build_grounding_gain_matrix,
)
from projects.sparse_grounding.oracle_manifest import (
    OracleQuerySelection,
    RealViewOracleManifest,
)
from projects.sparse_grounding.protocol import (
    CameraGraphStats,
    SparseSceneProtocol,
    ViewSelection,
)


class GroundingGainMatrixTest(unittest.TestCase):
    def make_inputs(self, root: Path):
        checkpoint = root / "model.pth"
        checkpoint.write_bytes(b"checkpoint")
        config = root / "oracle.py"
        config.write_text("# config\n", encoding="utf-8")
        script = root / "eval.sbatch"
        script.write_text("#!/bin/bash\n", encoding="utf-8")
        protocol_dir = root / "protocols"
        protocol_dir.mkdir()
        records = []
        for scene_index in range(2):
            scan_id = f"3rscan/scene-{scene_index}"
            records.append(
                OracleQuerySelection(
                    query_id=f"vg.json:{scene_index}",
                    scan_id=scan_id,
                    frame_ids=("unused.jpg",),
                )
            )
            SparseSceneProtocol(
                scene_id=quote(scan_id, safe=""),
                dataset="embodiedscan-v1-val",
                protocol_version="v1",
                trajectory_type="global_fps",
                seed=1,
                selections=(
                    ViewSelection(
                        3,
                        tuple(f"{scan_id}/base-{i}.jpg" for i in range(3)),
                    ),
                ),
                candidate_heldout_frame_ids=tuple(
                    f"{scan_id}/heldout-{i}.jpg" for i in range(4)
                ),
                camera_graph_stats=CameraGraphStats(7, 6, 0.5, 1, 10),
            ).dump(protocol_dir / f"{quote(scan_id, safe='')}.json")
        source = RealViewOracleManifest(
            policy="annotation_visible",
            base_view_budget=3,
            oracle_view_budget=4,
            trajectory_type="global_fps",
            records=tuple(records),
        )
        query_manifest = root / "queries.json"
        source.dump(query_manifest)
        return query_manifest, protocol_dir, checkpoint, config, script

    def test_builds_diverse_bounded_candidate_jobs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            query_manifest, protocols, checkpoint, config, script = (
                self.make_inputs(root)
            )

            matrix = build_grounding_gain_matrix(
                query_manifest=query_manifest,
                protocol_dir=protocols,
                output_root=root / "outputs",
                checkpoint=checkpoint,
                config=config,
                slurm_script=script,
                max_queries=2,
                max_candidates_per_query=2,
            )

            manifests = list(
                (root / "outputs" / "grounding_gain_manifests").glob("*.json")
            )
            candidate = RealViewOracleManifest.load(manifests[0])

        self.assertEqual(matrix["query_count"], 2)
        self.assertEqual(matrix["experiment_count"], 4)
        self.assertEqual(len(manifests), 4)
        self.assertEqual(candidate.policy, "grounding_gain")
        self.assertEqual(candidate.oracle_view_budget, 1)
        self.assertEqual(len(candidate.records), 1)
        self.assertEqual(
            matrix["experiments"][0]["slurm"]["environment"][
                "SPARSE_MISSING_ORACLE"
            ],
            "skip",
        )


if __name__ == "__main__":
    unittest.main()
