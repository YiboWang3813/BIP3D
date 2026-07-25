# Quick Start

### Set up python environment
On the target Slurm cluster, the pinned environment can be created and the
custom CUDA operation compiled with:
```bash
bash scripts/setup_bip3d_env.sh bip3d
```

For a manual installation:
```bash
virtualenv mm_bip3d --python=python3.8
source mm_bip3d/bin/activate
pip3 install --upgrade pip

bip3d_path="path/to/bip3d"
cd ${bip3d_path}
# MMCV recommends installing via a wheel package, url: https://download.openmmlab.com/mmcv/dist/cu{$cuda_version}/torch{$torch_version}/index.html
pip3 install -r requirements.txt

# Verify Python and pinned package versions. Add --require-cuda on the GPU
# build node before compiling the custom CUDA operation.
python3 tools/check_environment.py
```

### Compile the deformable_aggregation CUDA op
```bash
cd bip3d/ops
python3 setup.py develop
cd ../../
```

### Validate on a Slurm GPU node
The login node does not need to expose a GPU. After creating the `bip3d`
Conda environment and compiling the custom operation, validate the complete
runtime on one GPU:
```bash
sbatch scripts/slurm/bip3d_smoke.sbatch
```
Set `BIP3D_CONDA_ENV` before submission when the environment has a different
name, or set `BIP3D_PYTHON` to the environment's Python executable.

### Prepare the data
Download the [EmbodiedScan dataset](https://github.com/OpenRobotLab/EmbodiedScan) and create symbolic links.
```bash
cd ${bip3d_path}
mkdir data
ln -s path/to/embodiedscan ./data/embodiedscan
```

Download datasets [ScanNet](https://github.com/ScanNet/ScanNet), [3RScan](https://github.com/WaldJohannaU/3RScan), [Matterport3D](https://github.com/niessner/Matterport), and optionally download [ARKitScenes](https://github.com/apple/ARKitScenes). Adjust the data directory structure as follows:
```bash
${bip3d_path}
└──data
    ├──embodiedscan
    │   ├──embodiedscan_infos_train.pkl
    │   ├──embodiedscan_infos_val.pkl
    │   ...
    ├──3rscan
    │   ├──00d42bed-778d-2ac6-86a7-0e0e5f5f5660
    │   ...
    ├──scannet
    │   └──posed_images
    ├──matterport3d
    │   ├──17DRP5sb8fy
    │   ...
    └──arkitscenes
        ├──Training
        └──Validation
```

Check the directory layout and grounding annotations before starting a job:
```bash
python tools/check_sparse_grounding_data.py \
    --data-root data \
    --vg-profile all \
    --inspect-pickle \
    --json-output work_dirs/data_preflight.json
```
Only use `--inspect-pickle` with the official annotation files you trust. Use
`--vg-profile mini` when the config selects `data_version="v1-mini"`.

Export a small pose manifest and generate sparse-view protocols:
```bash
python tools/export_embodiedscan_poses.py \
    --info-file data/embodiedscan/embodiedscan_infos_val.pkl \
    --output work_dirs/sparse_grounding/val_poses.json \
    --source-dataset scannet \
    --max-scenes 3

python tools/sample_sparse_protocol.py \
    --input work_dirs/sparse_grounding/val_poses.json \
    --output-dir work_dirs/sparse_grounding/protocols \
    --trajectory-type local_connected
```
The info pickle must come from the official trusted annotation archive. Remove
`--max-scenes` and repeat the export per source dataset for a full run.
Generate full protocols on `cpu1`, not on the login node:
```bash
SPARSE_SOURCE_DATASET=matterport3d \
SPARSE_TRAJECTORY=local \
sbatch scripts/slurm/sparse_protocol_generation.sbatch
```
The Slurm script uses the standard `0.65 m / 35 degree` camera-graph limits.
For Matterport3D local protocols it uses `0.8 m / 90 degree`, because its
frames are discrete panorama directions rather than adjacent video frames.
The resolved limits and aggregated failure reasons are recorded in
`generation_summary.json`.

### Prepare pre-trained weights
Download the required Grounding-DINO pre-trained weights: [Swin-Tiny](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth) and [Swin-Base](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth).
```bash
mkdir ckpt

# Swin-Tiny
wget https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth -O ckpt/groundingdino_swint_ogc_mmdet-822d7e9d.pth
python tools/ckpt_rename.py ckpt/groundingdino_swint_ogc_mmdet-822d7e9d.pth

# Swin-Base
wget https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth -O ckpt/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth
python tools/ckpt_rename.py ckpt/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth
```
Download bert config and pretrain weights from [huggingface](https://huggingface.co/google-bert/bert-base-uncased/tree/main).
```bash
${bip3d_path}
└──ckpt
    ├──groundingdino_swint_ogc_mmdet-822d7e9d.pth
    ├──groundingdino_swinb_cogcoor_mmdet-55949c9c.pth
    └──bert-base-uncased
        ├──config.json
        ├──tokenizer_config.json
        ├──tokenizer.json
        ├──pytorch_model.bin
        ...
```

### Generate anchors by K-means
```bash
mkdir anchor_files
python3 tools/anchor_bbox3d_kmeans.py \
    --ann_file data/embodiedscan/embodiedscan_infos_train.pkl \
    --output_file anchor_files/embodiedscan_kmeans.npy
```
You can also download the anchor file we provide.


### Modify config
According to personal needs, modify some config items, such as [`DEBUG`](../configs/bip3d_det.py#L8), [`collect_dir`](../configs/bip3d_det.py#L425), [`save_dir`](../configs/bip3d_det.py#L475), and [`load_from`](../configs/bip3d_det.py#L485).

If performing multi-machine training, ensure that [`collect_dir`](../configs/bip3d_det.py#L425) is a shared folder accessible by all machines.

### Run local training and testing
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# train
bash engine.sh  configs/xxx.py

# test
bash test.sh  configs/xxx.py  path/to/checkpoint
```
