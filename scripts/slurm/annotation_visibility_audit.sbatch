#!/usr/bin/env bash
#SBATCH --job-name=bip3d-ann-visibility
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out

set -euxo pipefail
export PYTHONUNBUFFERED=1

repo_root="${BIP3D_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
experiment_root="${SPARSE_EXPERIMENT_ROOT:-/home/boywang25/paper2/experiments/sparse_grounding/v1}"
source_dataset="${SPARSE_SOURCE_DATASET:?set SPARSE_SOURCE_DATASET}"
trajectory="${SPARSE_TRAJECTORY:?set SPARSE_TRAJECTORY}"
python_bin="${BIP3D_PYTHON:-${HOME}/miniconda3/envs/bip3d/bin/python}"

case "${source_dataset}" in
    scannet|3rscan|matterport3d) ;;
    *) echo "unsupported source dataset: ${source_dataset}" >&2; exit 2 ;;
esac
case "${trajectory}" in
    local|global) ;;
    *) echo "unsupported trajectory: ${trajectory}" >&2; exit 2 ;;
esac

protocol_dir="${experiment_root}/val/${source_dataset}/${trajectory}"
output="${experiment_root}/annotation_visibility/${source_dataset}_${trajectory}.json"
test -x "${python_bin}"
test -d "${protocol_dir}"

cd "${repo_root}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
"${python_bin}" tools/audit_annotation_visibility.py \
    --info-file data/embodiedscan/embodiedscan_infos_val.pkl \
    --vg-file data/embodiedscan/embodiedscan_val_vg_all.json \
    --protocol-dir "${protocol_dir}" \
    --output "${output}" \
    --budgets 3 5 8 \
    --oracle-view-budget 4 \
    --source-dataset "${source_dataset}"
