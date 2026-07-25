#!/usr/bin/env bash
#SBATCH --job-name=bip3d-sparse-protocol
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

repo_root="${BIP3D_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
experiment_root="${SPARSE_EXPERIMENT_ROOT:-/home/boywang25/paper2/experiments/sparse_grounding/v1}"
source_dataset="${SPARSE_SOURCE_DATASET:?set SPARSE_SOURCE_DATASET}"
trajectory="${SPARSE_TRAJECTORY:?set SPARSE_TRAJECTORY}"
split="${SPARSE_SPLIT:-val}"
python_bin="${BIP3D_PYTHON:-${HOME}/miniconda3/envs/bip3d/bin/python}"

case "${source_dataset}" in
    scannet|3rscan|matterport3d) ;;
    *) echo "unsupported source dataset: ${source_dataset}" >&2; exit 2 ;;
esac
case "${trajectory}" in
    local)
        trajectory_type="local_connected"
        ;;
    global)
        trajectory_type="global_fps"
        ;;
    *)
        echo "unsupported trajectory: ${trajectory}" >&2
        exit 2
        ;;
esac

max_translation_m="${SPARSE_MAX_TRANSLATION_M:-0.65}"
max_rotation_deg="${SPARSE_MAX_ROTATION_DEG:-35}"
if [[ "${source_dataset}" == "matterport3d" && "${trajectory}" == "local" ]]; then
    # Matterport views are discretized panorama directions, unlike video frames.
    max_translation_m="${SPARSE_MAX_TRANSLATION_M:-0.8}"
    max_rotation_deg="${SPARSE_MAX_ROTATION_DEG:-90}"
fi

manifest="${SPARSE_POSE_MANIFEST:-${experiment_root}/${split}/${source_dataset}/poses.json}"
output_dir="${experiment_root}/${split}/${source_dataset}/${trajectory}"
overwrite=()
if [[ "${SPARSE_OVERWRITE:-0}" == "1" ]]; then
    overwrite=(--overwrite)
fi

test -x "${python_bin}"
test -f "${manifest}"
mkdir -p "${output_dir}"

cd "${repo_root}"
"${python_bin}" tools/sample_sparse_protocol.py \
    --input "${manifest}" \
    --output-dir "${output_dir}" \
    --trajectory-type "${trajectory_type}" \
    --budgets 3 5 8 \
    --seed "${SPARSE_SEED:-20260724}" \
    --protocol-version "${SPARSE_PROTOCOL_VERSION:-v1}" \
    --max-translation-m "${max_translation_m}" \
    --max-rotation-deg "${max_rotation_deg}" \
    --allow-scene-failures \
    "${overwrite[@]}"
