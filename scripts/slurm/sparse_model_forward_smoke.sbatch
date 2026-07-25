#!/usr/bin/env bash
#SBATCH --job-name=bip3d-sparse-forward
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out

set -euxo pipefail
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

repo_root="${BIP3D_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
artifact_root="${BIP3D_ARTIFACT_ROOT:-/home/boywang25/paper2/artifacts/bip3d}"
fixture_root="${BIP3D_FIXTURE_ROOT:-/home/boywang25/paper2/fixtures/sparse_3rscan_val_b3}"
checkpoint="${BIP3D_CHECKPOINT:-${artifact_root}/grounding_rgbd_dat/model_checkpoint.pth}"
bert_root="${BIP3D_BERT_ROOT:-${artifact_root}/bert-base-uncased}"
anchor_file="${BIP3D_ANCHOR_FILE:-${repo_root}/anchor_files/embodiedscan_kmeans.npy}"
cuda_home="${BIP3D_CUDA_HOME:-/opt/ohpc/pub/cuda-11.8.0}"
python_bin="${BIP3D_PYTHON:-${HOME}/miniconda3/envs/bip3d/bin/python}"

test -x "${python_bin}"
test -f "${checkpoint}"
test -f "${bert_root}/pytorch_model.bin"
test -f "${anchor_file}"
test -f "${fixture_root}/fixture_manifest.json"
echo "70961f0825b571072b3c6a58bffb830b3316f0d70a13a8ebca724330c0530b2f  ${checkpoint}" \
    | sha256sum --check --strict

export CUDA_HOME="${cuda_home}"
export PATH="${cuda_home}/bin:${PATH}"
export LD_LIBRARY_PATH="${cuda_home}/lib64:${LD_LIBRARY_PATH:-}"
export SPARSE_PROTOCOL_DIR="${fixture_root}/protocols"
export SPARSE_VIEW_BUDGET="${SPARSE_VIEW_BUDGET:-3}"
export SPARSE_TRAJECTORY_TYPE="${SPARSE_TRAJECTORY_TYPE:-global_fps}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"

cd "${repo_root}"
"${python_bin}" tools/test.py \
    configs/sparse_grounding_3rscan.py \
    "${checkpoint}" \
    --work-dir work_dirs/sparse_model_forward_smoke \
    --cfg-options \
    "model.text_encoder.name=${bert_root}" \
    "model.decoder.instance_bank.anchor=${anchor_file}" \
    "test_dataloader.dataset.data_root=${fixture_root}" \
    "test_dataloader.dataset.protocol_dir=${fixture_root}/protocols" \
    "test_dataloader.num_workers=0" \
    "test_dataloader.persistent_workers=False"
