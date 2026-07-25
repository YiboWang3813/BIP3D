#!/usr/bin/env bash
#SBATCH --job-name=bip3d-real-oracle
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

repo_root="${BIP3D_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
python_bin="${BIP3D_PYTHON:-${HOME}/miniconda3/envs/bip3d/bin/python}"
experiment_root="${SPARSE_EXPERIMENT_ROOT:-/home/boywang25/paper2/experiments/sparse_grounding/v1}"
source_dataset="${SPARSE_SOURCE_DATASET:-3rscan}"
trajectory="${SPARSE_TRAJECTORY:?set SPARSE_TRAJECTORY}"
policy="${SPARSE_ORACLE_POLICY:?set SPARSE_ORACLE_POLICY}"
base_budget="${SPARSE_VIEW_BUDGET:-5}"
oracle_budget="${SPARSE_ORACLE_VIEW_BUDGET:-4}"

case "${trajectory}" in
    local) trajectory_type="local_connected" ;;
    global) trajectory_type="global_fps" ;;
    *) echo "unsupported trajectory: ${trajectory}" >&2; exit 2 ;;
esac
case "${policy}" in
    random_real|annotation_visible) ;;
    *) echo "unsupported oracle policy: ${policy}" >&2; exit 2 ;;
esac

protocol_dir="${experiment_root}/val/${source_dataset}/${trajectory}"
output_dir="${experiment_root}/real_view_oracle"
output="${output_dir}/${source_dataset}_${trajectory}_${policy}_k${base_budget}_m${oracle_budget}.json"

test -x "${python_bin}"
test -d "${protocol_dir}"
mkdir -p "${output_dir}"

cd "${repo_root}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
"${python_bin}" tools/build_real_view_oracle.py \
    --info-file data/embodiedscan/embodiedscan_infos_val.pkl \
    --vg-file data/embodiedscan/embodiedscan_val_vg_all.json \
    --query-file-id embodiedscan/embodiedscan_val_vg_all.json \
    --protocol-dir "${protocol_dir}" \
    --source-dataset "${source_dataset}" \
    --trajectory-type "${trajectory_type}" \
    --policy "${policy}" \
    --base-view-budget "${base_budget}" \
    --oracle-view-budget "${oracle_budget}" \
    --seed "${SPARSE_SEED:-20260724}" \
    --output "${output}"
