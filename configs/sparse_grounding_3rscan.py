"""Sparse-view grounding evaluation on the EmbodiedScan v1 3RScan subset."""

_base_ = ["./bip3d_grounding.py"]

import os


custom_imports = dict(
    imports=[
        "projects.sparse_grounding.protocol_dataset",
        "projects.sparse_grounding.query_metrics",
    ],
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
query_result_file = os.environ.get("SPARSE_QUERY_RESULT_FILE")
bert_path = os.environ.get(
    "BIP3D_BERT_PATH",
    "./ckpt/bert-base-uncased",
)
anchor_path = os.environ.get(
    "BIP3D_ANCHOR_PATH",
    "anchor_files/embodiedscan_kmeans.npy",
)
data_root = os.environ.get("BIP3D_DATA_ROOT", "data")

model = dict(
    text_encoder=dict(name=bert_path),
    decoder=dict(instance_bank=dict(anchor=anchor_path)),
)

sparse_dataset = dict(
    type="SparseProtocolGroundingDataset",
    data_root=data_root,
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
    type="SparseGroundingMetric",
    collect_dir=None,
    query_result_file=query_result_file,
)

work_dir = os.path.join(
    "work_dirs",
    "sparse_grounding_3rscan",
    trajectory_type,
    f"budget_{view_budget}",
)

del os
