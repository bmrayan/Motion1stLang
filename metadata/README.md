# Metadata Contents

This directory stores compact reproducibility metadata only (no large model/data binaries):

- `data/`: generation/tokenization summaries and corpus metadata.
- `models/`: pre-pre-training and VQ-VAE hyperparameters/summaries.
- `checkpoints/`: language-training summaries, hparams, and metric CSV logs.

These files are sufficient to audit run settings and compare outputs against regenerated runs.
