#!/usr/bin/env bash
#SBATCH --job-name=bip3d-sparse-eval
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

repo_root="${BIP3D_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
python_bin="${BIP3D_PYTHON:-${HOME}/miniconda3/envs/bip3d/bin/python}"
config="${SPARSE_CONFIG:-${repo_root}/configs/sparse_grounding_3rscan.py}"
checkpoint="${BIP3D_CHECKPOINT:?set BIP3D_CHECKPOINT}"
protocol_dir="${SPARSE_PROTOCOL_DIR:?set SPARSE_PROTOCOL_DIR}"
view_budget="${SPARSE_VIEW_BUDGET:?set SPARSE_VIEW_BUDGET}"
trajectory_type="${SPARSE_TRAJECTORY_TYPE:?set SPARSE_TRAJECTORY_TYPE}"
work_dir="${SPARSE_WORK_DIR:?set SPARSE_WORK_DIR}"

export SPARSE_QUERY_RESULT_FILE="${SPARSE_QUERY_RESULT_FILE:-${work_dir}/per_query_metrics.json}"
export BIP3D_BERT_PATH="${BIP3D_BERT_PATH:-/home/boywang25/paper2/artifacts/bip3d/bert-base-uncased}"
export BIP3D_ANCHOR_PATH="${BIP3D_ANCHOR_PATH:-${repo_root}/anchor_files/embodiedscan_kmeans.npy}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"

test -x "${python_bin}"
test -f "${config}"
test -f "${checkpoint}"
test -d "${protocol_dir}"
test -d "${BIP3D_BERT_PATH}"
test -f "${BIP3D_ANCHOR_PATH}"
mkdir -p "${work_dir}"

cd "${repo_root}"
"${python_bin}" tools/test.py \
    "${config}" \
    "${checkpoint}" \
    --work-dir "${work_dir}"
