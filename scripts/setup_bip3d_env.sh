#!/usr/bin/env bash
# Create the pinned BIP3D environment and compile its CUDA extensions.

set -euo pipefail

env_name="${1:-bip3d}"
cuda_home="${BIP3D_CUDA_HOME:-/opt/ohpc/pub/cuda-11.8.0}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$(conda info --base)/envs/${env_name}/bin/python"
filtered_requirements="$(mktemp)"
trap 'rm -f "${filtered_requirements}"' EXIT

conda create -n "${env_name}" python=3.10 pip -y

"${python_bin}" -m pip install \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

grep -Ev \
    '^(torch|torchvision|torchaudio|mmcv|pytorch3d)==' \
    "${repo_root}/requirements.txt" > "${filtered_requirements}"
"${python_bin}" -m pip install -r "${filtered_requirements}"

"${python_bin}" -m pip install \
    mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
"${python_bin}" -m pip install \
    pytorch3d==0.7.5 \
    -f \
    https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html

test -x "${cuda_home}/bin/nvcc"
export CUDA_HOME="${cuda_home}"
export PATH="${cuda_home}/bin:${PATH}"
export FORCE_CUDA=1
(
    cd "${repo_root}/bip3d/ops"
    "${python_bin}" setup.py develop
)

"${python_bin}" "${repo_root}/tools/check_environment.py"
echo "Environment ${env_name} is ready."
