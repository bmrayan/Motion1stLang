#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from _motion_utils import ensure_dir, save_json, sequence_length_summary, set_seed, token_frequency_summary
from _vqvae import MLPVQVAE, VQVAEConfig


SEP_TOKEN = 4096
SCENE_TOKEN = 4097
INTERACTION_OFFSET = 4098


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize physics feature shards using trained VQ-VAE models.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vqvae_dir", type=str, default="/workspace/models/vqvae")
    parser.add_argument("--kinematic_model", type=str, default="kinematic_vqvae.pt")
    parser.add_argument("--kinematic_stats", type=str, default="kinematic_stats.npz")
    parser.add_argument("--interaction_model", type=str, default="interaction_vqvae.pt")
    parser.add_argument("--interaction_stats", type=str, default="interaction_stats.npz")
    parser.add_argument("--feature_key", type=str, default="features")
    parser.add_argument("--include_interactions", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> MLPVQVAE:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = VQVAEConfig(**payload["config"])
    model = MLPVQVAE(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def encode_rows(
    model: MLPVQVAE,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if len(rows) == 0:
        return np.zeros((0,), dtype=np.uint16)
    outputs = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        normalized = (batch - mean) / std
        tensor = torch.from_numpy(normalized.astype(np.float32)).to(device)
        with torch.no_grad():
            indices = model.encode_indices(tensor).cpu().numpy().astype(np.uint16)
        outputs.append(indices.reshape(-1))
    return np.concatenate(outputs, axis=0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(ensure_dir(args.output_dir))
    vqvae_dir = Path(args.vqvae_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    kin_model = load_model(vqvae_dir / args.kinematic_model, device)
    kin_stats = np.load(vqvae_dir / args.kinematic_stats)
    kin_mean, kin_std = kin_stats["mean"].astype(np.float32), kin_stats["std"].astype(np.float32)

    interaction_model = None
    interaction_mean = interaction_std = None
    if args.include_interactions:
        interaction_model = load_model(vqvae_dir / args.interaction_model, device)
        interaction_stats = np.load(vqvae_dir / args.interaction_stats)
        interaction_mean = interaction_stats["mean"].astype(np.float32)
        interaction_std = interaction_stats["std"].astype(np.float32)

    token_path = output_dir / "tokens.bin"
    metadata_path = output_dir / "metadata.json"

    sequence_offsets = [0]
    sequence_lengths = []
    total_tokens = 0

    with open(token_path, "wb") as handle:
        for shard in sorted(input_dir.glob("*_shard_*.npz")):
            with np.load(shard) as data:
                features = data[args.feature_key].astype(np.float32)
                feature_offsets = data["feature_offsets"]
                object_offsets = data["object_offsets"]
                scene_object_offsets = data["scene_object_offsets"]
                interaction_features = data["interaction_features"].astype(np.float32)
                interaction_offsets = data["interaction_offsets"]

                encoded_features = encode_rows(kin_model, features, kin_mean, kin_std, args.batch_size, device)
                encoded_interactions = (
                    encode_rows(interaction_model, interaction_features, interaction_mean, interaction_std, args.batch_size, device)
                    if args.include_interactions and interaction_model is not None and len(interaction_features)
                    else np.zeros((0,), dtype=np.uint16)
                )

                for scene_idx in range(len(feature_offsets) - 1):
                    pieces = []
                    object_start = int(scene_object_offsets[scene_idx])
                    object_end = int(scene_object_offsets[scene_idx + 1])
                    for object_idx in range(object_start, object_end):
                        start = int(object_offsets[object_idx])
                        end = int(object_offsets[object_idx + 1])
                        pieces.append(encoded_features[start:end])
                        pieces.append(np.asarray([SEP_TOKEN], dtype=np.uint16))

                    i_start = int(interaction_offsets[scene_idx])
                    i_end = int(interaction_offsets[scene_idx + 1])
                    if args.include_interactions and i_end > i_start:
                        pieces.append((encoded_interactions[i_start:i_end] + INTERACTION_OFFSET).astype(np.uint16))
                    pieces.append(np.asarray([SCENE_TOKEN], dtype=np.uint16))
                    seq = np.concatenate(pieces)
                    seq.tofile(handle)
                    total_tokens += int(len(seq))
                    sequence_lengths.append(int(len(seq)))
                    sequence_offsets.append(total_tokens)

    sample = np.memmap(token_path, dtype=np.uint16, mode="r")
    summary = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "feature_key": args.feature_key,
        "include_interactions": bool(args.include_interactions),
        "vocab_size": 5122,
        "special_tokens": {
            "sep": SEP_TOKEN,
            "scene": SCENE_TOKEN,
            "interaction_offset": INTERACTION_OFFSET,
        },
        "total_tokens": total_tokens,
        "sequence_count": len(sequence_lengths),
        "sequence_length_summary": sequence_length_summary(sequence_lengths),
        "token_frequency_summary": token_frequency_summary(sample[: min(len(sample), 2_000_000)]),
    }
    np.save(output_dir / "sequence_offsets.npy", np.asarray(sequence_offsets, dtype=np.int64))
    save_json(str(metadata_path), summary)

    print(f"Total token count: {total_tokens}")
    print(f"Sequence count: {len(sequence_lengths)}")
    print(
        "Sequence length mean/std: "
        f"{summary['sequence_length_summary']['mean']:.2f} / {summary['sequence_length_summary']['std']:.2f}"
    )
    print(f"Top token summary: {summary['token_frequency_summary']['top_tokens'][:5]}")


if __name__ == "__main__":
    main()
