#!/usr/bin/env bash
#SBATCH --job-name=bip3d-sparse-pipeline
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out

set -euxo pipefail
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

repo_root="${BIP3D_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
fixture_root="${BIP3D_FIXTURE_ROOT:-/home/boywang25/paper2/fixtures/sparse_3rscan_val_b3}"
cuda_home="${BIP3D_CUDA_HOME:-/opt/ohpc/pub/cuda-11.8.0}"
python_bin="${BIP3D_PYTHON:-${HOME}/miniconda3/envs/bip3d/bin/python}"

test -x "${python_bin}"
test -d "${fixture_root}"
export CUDA_HOME="${cuda_home}"
export PATH="${cuda_home}/bin:${PATH}"
export LD_LIBRARY_PATH="${cuda_home}/lib64:${LD_LIBRARY_PATH:-}"

cd "${repo_root}"
"${python_bin}" tools/sparse_pipeline_smoke.py \
    --fixture-root "${fixture_root}" \
    --budget "${SPARSE_VIEW_BUDGET:-3}" \
    --trajectory-type "${SPARSE_TRAJECTORY_TYPE:-global_fps}" \
    --device cuda
