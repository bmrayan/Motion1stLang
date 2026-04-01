# Motion1stLang Repro Bundle

This repository contains the exact scripts, logs, metadata, and Tier 1 outputs used for the Motion vs NCA signal-check experiment (`GPT-2 Small`, `OpenWebText`, `164M` pre-pre-train tokens, `2B` language tokens).

## What Is Included

- `scripts/`: end-to-end generation, tokenization, pre-pre-training, language training, plotting, and evaluation code.
- `logs/`: run logs from completed jobs.
- `metadata/`: compact run metadata (tokenization summaries, hparams, training summaries, metric CSVs).
- `results/`: canonical outputs, including `all_results.json`, figures, and the LaTeX table.

## What Is Not Included

- Multi-hundred-GB raw datasets and model weight binaries are not committed.
- Reproduce them with the commands below.

## Environment Setup (A100, Ubuntu)

```bash
apt-get update && apt-get install -y git git-lfs tmux htop nvtop wget curl unzip

conda create -n motion python=3.10 -y
conda activate motion
pip install -r requirements.txt
```

## External Repositories (Official Sources)

```bash
mkdir -p /workspace/repos
cd /workspace/repos
git clone https://github.com/danihyunlee/nca-pre-pretraining.git
git clone https://github.com/stanford-crfm/mistral.git
git clone https://github.com/jasperjjian/abstraction.git
```

## Required Data Sources

- OpenWebText: `Skylion007/openwebtext`
- GPT-2 tokenizer only: `openai-community/gpt2`
- Stanford scratch reference checkpoints: `stanford-crfm/alias-gpt2-small-x21`

## Reproduction Order (Tier 1)

Run from `/workspace/Motion1stLang`:

```bash
conda activate motion

# 1) Physics generation + shuffled control
python scripts/generate_synthetic_physics.py --output_dir /workspace/data/synthetic_physics
python scripts/create_shuffled_temporal.py --input_dir /workspace/data/synthetic_physics --output_dir /workspace/data/shuffled_temporal

# 2) VQ-VAE + tokenization
python scripts/train_vqvae.py --data_dir /workspace/data/synthetic_physics --output_dir /workspace/models/vqvae
python scripts/tokenize_physics.py --input_dir /workspace/data/synthetic_physics --output_dir /workspace/data/physics_tokenized --vqvae_dir /workspace/models/vqvae
python scripts/tokenize_physics.py --input_dir /workspace/data/shuffled_temporal --output_dir /workspace/data/shuffled_tokenized --vqvae_dir /workspace/models/vqvae

# 3) NCA tokens (official NCA code path)
python scripts/generate_nca_tokens.py --repo_dir /workspace/repos/nca-pre-pretraining --output_dir /workspace/data/nca_tokenized --target_tokens 164000000

# 4) Pre-pre-training
python scripts/pretrain_on_tokens.py --data_path /workspace/data/physics_tokenized --output_dir /workspace/models/physics_ppt --total_tokens 164000000 --vocab_size 5122 --run_name physics_ppt
python scripts/pretrain_on_tokens.py --data_path /workspace/data/shuffled_tokenized --output_dir /workspace/models/shuffled_ppt --total_tokens 164000000 --vocab_size 5122 --run_name shuffled_ppt
python scripts/pretrain_on_tokens.py --data_path /workspace/data/nca_tokenized --output_dir /workspace/models/nca_ppt --total_tokens 164000000 --vocab_size 10002 --run_name nca_ppt

# 5) Language pre-training
python scripts/language_pretrain.py --init_from scratch --data_path /workspace/data/openwebtext --output_dir /workspace/checkpoints/scratch --run_name scratch_baseline --total_tokens 2000000000 --save_every 1000
python scripts/language_pretrain.py --init_from /workspace/models/physics_ppt --data_path /workspace/data/openwebtext --output_dir /workspace/checkpoints/physics --run_name physics_lang --total_tokens 2000000000 --save_every 1000 --prepare_data 0
python scripts/language_pretrain.py --init_from /workspace/models/shuffled_ppt --data_path /workspace/data/openwebtext --output_dir /workspace/checkpoints/shuffled --run_name shuffled_lang --total_tokens 2000000000 --save_every 1000 --prepare_data 0
python scripts/language_pretrain.py --init_from /workspace/models/nca_ppt --data_path /workspace/data/openwebtext --output_dir /workspace/checkpoints/nca --run_name nca_lang --total_tokens 2000000000 --save_every 1000 --prepare_data 0

# 6) Aggregation + plots
python scripts/evaluate.py corpus-gzip --inputs physics=/workspace/data/physics_tokenized shuffled=/workspace/data/shuffled_tokenized nca=/workspace/data/nca_tokenized --output /workspace/results/gzip_complexity.json
python scripts/evaluate.py speedups --logs scratch=/workspace/checkpoints/scratch/logs/train_metrics.csv physics=/workspace/checkpoints/physics/logs/train_metrics.csv shuffled=/workspace/checkpoints/shuffled/logs/train_metrics.csv nca=/workspace/checkpoints/nca/logs/train_metrics.csv --output /workspace/results/convergence_speedups.json
python scripts/plot_results.py --logs scratch=/workspace/checkpoints/scratch/logs/train_metrics.csv physics=/workspace/checkpoints/physics/logs/train_metrics.csv shuffled=/workspace/checkpoints/shuffled/logs/train_metrics.csv nca=/workspace/checkpoints/nca/logs/train_metrics.csv --output_dir /workspace/results
```

## Canonical Saved Output

- `results/all_results.json` is the single canonical aggregate file to archive.
