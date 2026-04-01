#!/usr/bin/env bash
set -euo pipefail

SHUFFLED_SESSION="${1:-shuffled_ppt}"
NCA_DATA_DIR="${2:-/workspace/data/nca_tokenized}"
QUEUE_LOG="${3:-/workspace/logs/nca_ppt_queue.log}"

timestamp() {
  date -u +"%Y-%m-%d %H:%M:%S UTC"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

mkdir -p /workspace/logs /workspace/models/nca_ppt
exec > >(tee -a "${QUEUE_LOG}") 2>&1

log "queue started; waiting for session=${SHUFFLED_SESSION} and nca tokens in ${NCA_DATA_DIR}"

while tmux has-session -t "${SHUFFLED_SESSION}" 2>/dev/null; do
  log "waiting: ${SHUFFLED_SESSION} still running"
  sleep 60
done

log "detected ${SHUFFLED_SESSION} completion"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate motion

while true; do
  if [[ -f "${NCA_DATA_DIR}/metadata.json" && -f "${NCA_DATA_DIR}/tokens.bin" ]]; then
    TOKENS_WRITTEN="$(NCA_META_PATH="${NCA_DATA_DIR}/metadata.json" python - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ["NCA_META_PATH"])
if p.exists():
    data = json.loads(p.read_text())
    print(int(data.get("tokens_written", 0)))
else:
    print(0)
PY
)"
    if [[ "${TOKENS_WRITTEN}" -ge 164000000 ]]; then
      log "nca token prerequisite satisfied: tokens_written=${TOKENS_WRITTEN}"
      break
    fi
    log "waiting: nca metadata present but tokens_written=${TOKENS_WRITTEN} (<164000000)"
  else
    log "waiting: nca metadata/tokens not ready yet"
  fi
  sleep 60
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log "starting nca_ppt pretraining"
python /workspace/scripts/pretrain_on_tokens.py \
  --data_path /workspace/data/nca_tokenized \
  --output_dir /workspace/models/nca_ppt \
  --total_tokens 164000000 \
  --vocab_size 10002 \
  --run_name nca_ppt 2>&1 | tee /workspace/logs/nca_ppt.log

log "nca_ppt completed"
