#!/usr/bin/env bash
#SBATCH --job-name=bip3d-smoke
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out

set -euxo pipefail
export PYTHONUNBUFFERED=1

repo_root="${BIP3D_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
conda_env="${BIP3D_CONDA_ENV:-bip3d}"
cuda_home="${BIP3D_CUDA_HOME:-/opt/ohpc/pub/cuda-11.8.0}"
python_bin="${BIP3D_PYTHON:-${HOME}/miniconda3/envs/${conda_env}/bin/python}"

echo "Using CUDA toolkit: ${cuda_home}"
test -x "${cuda_home}/bin/nvcc"
export CUDA_HOME="${cuda_home}"
export PATH="${cuda_home}/bin:${PATH}"
export LD_LIBRARY_PATH="${cuda_home}/lib64:${LD_LIBRARY_PATH:-}"
echo "Using Python environment: ${python_bin}"
test -x "${python_bin}"

echo "Validating repository: ${repo_root}"
cd "${repo_root}"
test -f requirements.txt
"${python_bin}" tools/check_environment.py --require-cuda
echo "Running CUDA and extension smoke test"
"${python_bin}" tools/runtime_smoke.py --require-extensions
