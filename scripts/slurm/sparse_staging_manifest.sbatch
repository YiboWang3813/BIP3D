#!/bin/bash
#SBATCH --job-name=sparse-stage-plan
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm

: "${SPARSE_STAGING_OUTPUT:?set SPARSE_STAGING_OUTPUT}"
: "${SPARSE_PROTOCOL_LOCAL:?set SPARSE_PROTOCOL_LOCAL}"
: "${SPARSE_PROTOCOL_GLOBAL:?set SPARSE_PROTOCOL_GLOBAL}"

INFO_FILE="${EMBODIEDSCAN_INFO_FILE:-data/embodiedscan/embodiedscan_infos_val.pkl}"
ORACLE_ROOT="${SPARSE_ORACLE_ROOT:-/home/boywang25/paper2/experiments/sparse_grounding/v1/real_view_oracle}"
GAIN_ROOT="${SPARSE_GAIN_ROOT:-/home/boywang25/paper2/experiments/sparse_grounding/v1/model_eval/grounding_gain_local_k5/grounding_gain_manifests}"

oracle_args=()
for path in \
  "${ORACLE_ROOT}"/3rscan_{local,global}_{random_real,annotation_visible}_k5_m4.json \
  "${GAIN_ROOT}"/*.json; do
  oracle_args+=(--oracle-file "${path}")
done

python tools/build_sparse_staging_manifest.py \
  --dataset 3rscan \
  --info-file "${INFO_FILE}" \
  --protocol-dir "${SPARSE_PROTOCOL_LOCAL}" \
  --protocol-dir "${SPARSE_PROTOCOL_GLOBAL}" \
  "${oracle_args[@]}" \
  --output "${SPARSE_STAGING_OUTPUT}"
