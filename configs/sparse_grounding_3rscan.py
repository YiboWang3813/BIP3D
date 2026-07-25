"""Sparse-view grounding evaluation on the EmbodiedScan v1 3RScan subset."""

_base_ = ["./bip3d_grounding.py"]

import os


custom_imports = dict(
    imports=["projects.sparse_grounding.protocol_dataset"],
    allow_failed_imports=False,
)

protocol_dir = os.environ.get(
    "SPARSE_PROTOCOL_DIR",
    "work_dirs/sparse_grounding/protocols",
)
view_budget = int(os.environ.get("SPARSE_VIEW_BUDGET", "3"))
trajectory_type = os.environ.get(
    "SPARSE_TRAJECTORY_TYPE",
    "global_fps",
)

sparse_dataset = dict(
    type="SparseProtocolGroundingDataset",
    part=["3rscan"],
    protocol_dir=protocol_dir,
    view_budget=view_budget,
    missing_protocol="skip",
    expected_protocol_dataset="embodiedscan-v1-val",
    expected_trajectory_type=trajectory_type,
    expected_protocol_version="v1",
)

val_dataloader = dict(dataset=sparse_dataset)
test_dataloader = dict(
    dataset=dict(
        **sparse_dataset,
        ann_file="embodiedscan/embodiedscan_infos_val.pkl",
        vg_file="embodiedscan/embodiedscan_val_vg_all.json",
    )
)
test_evaluator = dict(
    _delete_=True,
    type="GroundingMetric",
    collect_dir=None,
)

work_dir = os.path.join(
    "work_dirs",
    "sparse_grounding_3rscan",
    trajectory_type,
    f"budget_{view_budget}",
)

del os
