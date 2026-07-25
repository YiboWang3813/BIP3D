"""BIP3D sparse grounding augmented by query-level held-out real views."""

_base_ = ["./sparse_grounding_3rscan.py"]

import os


oracle_manifest = os.environ["SPARSE_ORACLE_MANIFEST"]
oracle_policy = os.environ.get("SPARSE_ORACLE_POLICY")
missing_oracle = os.environ.get("SPARSE_MISSING_ORACLE", "error")

test_dataloader = dict(
    dataset=dict(
        type="OracleProtocolGroundingDataset",
        oracle_manifest=oracle_manifest,
        expected_oracle_policy=oracle_policy,
        missing_oracle=missing_oracle,
    )
)

del os
