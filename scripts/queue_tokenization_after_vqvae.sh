#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
if [[ -z "${WAIT_PID}" ]]; then
  echo "usage: $0 <vqvae_pid>" >&2
  exit 1
fi

QUEUE_LOG="/workspace/logs/tokenize_queue.log"
PHYSICS_LOG="/workspace/logs/physics_tokenize.log"
SHUFFLED_LOG="/workspace/logs/shuffled_tokenize.log"

timestamp() {
  date -u +"%Y-%m-%d %H:%M:%S UTC"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

mkdir -p /workspace/logs /workspace/data/physics_tokenized /workspace/data/shuffled_tokenized
exec > >(tee -a "${QUEUE_LOG}") 2>&1

log "queue started; waiting for VQ-VAE pid ${WAIT_PID}"
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  log "VQ-VAE still running"
  sleep 60
done
log "detected VQ-VAE pid ${WAIT_PID} exit"

required_files=(
  /workspace/models/vqvae/kinematic_vqvae.pt
  /workspace/models/vqvae/interaction_vqvae.pt
  /workspace/models/vqvae/kinematic_stats.npz
  /workspace/models/vqvae/interaction_stats.npz
)

for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    log "missing required artifact: ${path}"
    exit 1
  fi
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate motion

log "starting physics tokenization"
python /workspace/scripts/tokenize_physics.py \
  --input_dir /workspace/data/synthetic_physics \
  --output_dir /workspace/data/physics_tokenized \
  --vqvae_dir /workspace/models/vqvae \
  --device cuda \
  --batch_size 16384 2>&1 | tee "${PHYSICS_LOG}"

log "starting shuffled tokenization"
python /workspace/scripts/tokenize_physics.py \
  --input_dir /workspace/data/shuffled_temporal \
  --output_dir /workspace/data/shuffled_tokenized \
  --vqvae_dir /workspace/models/vqvae \
  --device cuda \
  --batch_size 16384 2>&1 | tee "${SHUFFLED_LOG}"

log "tokenization queue completed successfully"
