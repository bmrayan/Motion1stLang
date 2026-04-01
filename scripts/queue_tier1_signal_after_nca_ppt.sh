#!/usr/bin/env bash
set -euo pipefail

WAIT_SESSION="${1:-nca_ppt}"
QUEUE_LOG="${2:-/workspace/logs/tier1_queue.log}"

timestamp() {
  date -u +"%Y-%m-%d %H:%M:%S UTC"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

mkdir -p /workspace/logs /workspace/checkpoints /workspace/results
exec > >(tee -a "${QUEUE_LOG}") 2>&1

log "tier1 queue started; waiting for session=${WAIT_SESSION}"
while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
  log "waiting: ${WAIT_SESSION} still running"
  sleep 60
done

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate motion
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_with_retry() {
  local name="$1"
  shift
  if "$@"; then
    return 0
  fi
  log "warning: ${name} failed on first attempt, retrying once in 30s"
  sleep 30
  "$@"
}

ensure_ppt_complete() {
  local name="$1"
  local data_path="$2"
  local output_dir="$3"
  local vocab_size="$4"
  local run_name="$5"
  local summary_path="${output_dir}/training_summary.json"
  local log_path="/workspace/logs/${name}.log"

  if [[ -f "${summary_path}" ]]; then
    log "pre-pretrain already complete: ${name} (${summary_path})"
    return 0
  fi

  log "pre-pretrain missing/incomplete, running now: ${name}"
  run_with_retry "${name}" bash -lc "
    python /workspace/scripts/pretrain_on_tokens.py \
      --data_path '${data_path}' \
      --output_dir '${output_dir}' \
      --total_tokens 164000000 \
      --vocab_size '${vocab_size}' \
      --run_name '${run_name}' 2>&1 | tee -a '${log_path}'
  "

  if [[ ! -f "${summary_path}" ]]; then
    log "ERROR: expected summary not found after ${name}: ${summary_path}"
    exit 1
  fi
}

run_lang_job() {
  local name="$1"
  local init_from="$2"
  local output_dir="$3"
  local run_name="$4"
  local prepare_data="$5"
  local log_path="/workspace/logs/lang_${name}.log"

  if [[ -f "${output_dir}/training_summary.json" ]]; then
    log "language run already complete: ${name} (${output_dir}/training_summary.json)"
    return 0
  fi

  log "starting language run: ${name}"
  run_with_retry "${name}" bash -lc "
    python /workspace/scripts/language_pretrain.py \
      --init_from '${init_from}' \
      --data_path /workspace/data/openwebtext \
      --output_dir '${output_dir}' \
      --run_name '${run_name}' \
      --total_tokens 2000000000 \
      --save_every 1000 \
      --prepare_data '${prepare_data}' 2>&1 | tee -a '${log_path}'
  "
}

ensure_ppt_complete physics_ppt /workspace/data/physics_tokenized /workspace/models/physics_ppt 5122 physics_ppt
ensure_ppt_complete shuffled_ppt /workspace/data/shuffled_tokenized /workspace/models/shuffled_ppt 5122 shuffled_ppt
ensure_ppt_complete nca_ppt /workspace/data/nca_tokenized /workspace/models/nca_ppt 10002 nca_ppt

run_lang_job scratch scratch /workspace/checkpoints/scratch scratch_baseline 1
run_lang_job physics /workspace/models/physics_ppt /workspace/checkpoints/physics physics_lang 0
run_lang_job shuffled /workspace/models/shuffled_ppt /workspace/checkpoints/shuffled shuffled_lang 0
run_lang_job nca /workspace/models/nca_ppt /workspace/checkpoints/nca nca_lang 0

log "running Tier 1 result aggregation"
run_with_retry corpus_gzip python /workspace/scripts/evaluate.py corpus-gzip \
  --inputs \
  physics=/workspace/data/physics_tokenized \
  shuffled=/workspace/data/shuffled_tokenized \
  nca=/workspace/data/nca_tokenized \
  --output /workspace/results/gzip_complexity.json

run_with_retry speedups python /workspace/scripts/evaluate.py speedups \
  --logs \
  scratch=/workspace/checkpoints/scratch/logs/train_metrics.csv \
  physics=/workspace/checkpoints/physics/logs/train_metrics.csv \
  shuffled=/workspace/checkpoints/shuffled/logs/train_metrics.csv \
  nca=/workspace/checkpoints/nca/logs/train_metrics.csv \
  --output /workspace/results/convergence_speedups.json

run_with_retry plot_results python /workspace/scripts/plot_results.py \
  --logs \
  scratch=/workspace/checkpoints/scratch/logs/train_metrics.csv \
  physics=/workspace/checkpoints/physics/logs/train_metrics.csv \
  shuffled=/workspace/checkpoints/shuffled/logs/train_metrics.csv \
  nca=/workspace/checkpoints/nca/logs/train_metrics.csv \
  --output_dir /workspace/results

python - <<'PY'
import json
from pathlib import Path

paths = {
    "scratch": Path("/workspace/checkpoints/scratch/training_summary.json"),
    "physics": Path("/workspace/checkpoints/physics/training_summary.json"),
    "shuffled": Path("/workspace/checkpoints/shuffled/training_summary.json"),
    "nca": Path("/workspace/checkpoints/nca/training_summary.json"),
}

out = {}
for name, path in paths.items():
    data = json.loads(path.read_text())
    out[name] = {
        "run_name": data.get("run_name"),
        "final_val_loss": data.get("final_val_loss"),
        "final_val_perplexity": data.get("final_val_perplexity"),
        "tokens_seen": data.get("tokens_seen"),
        "steps": data.get("steps"),
    }

Path("/workspace/results/final_perplexities.json").write_text(json.dumps(out, indent=2, sort_keys=True))
PY

log "tier1 queue finished successfully"
