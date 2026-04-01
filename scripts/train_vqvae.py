#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from _motion_utils import CSVLogger, ensure_dir, save_json, set_seed
from _vqvae import MLPVQVAE, VQVAEConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VQ-VAE tokenizers for physics features.")
    parser.add_argument("--data_dir", type=str, default="/workspace/data/synthetic_physics")
    parser.add_argument("--output_dir", type=str, default="/workspace/models/vqvae")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout_scenes", type=int, default=50_000)
    parser.add_argument("--max_vectors_per_scene", type=int, default=64)
    parser.add_argument("--max_interactions_per_scene", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--commitment_beta", type=float, default=0.5)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_interaction", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--kmeans_init_sample_size", type=int, default=65536)
    parser.add_argument("--kmeans_iters", type=int, default=20)
    parser.add_argument("--reinit_util_threshold", type=float, default=50.0)
    parser.add_argument("--reinit_patience_epochs", type=int, default=5)
    parser.add_argument("--feature_key", type=str, default="features")
    parser.add_argument("--feature_input_dim", type=int, default=28)
    parser.add_argument("--feature_hidden_dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--feature_embedding_dim", type=int, default=64)
    parser.add_argument("--feature_codebook_size", type=int, default=4096)
    parser.add_argument("--interaction_hidden_dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--interaction_embedding_dim", type=int, default=32)
    parser.add_argument("--interaction_codebook_size", type=int, default=1024)
    return parser.parse_args()


def list_shards(data_dir: Path) -> List[Path]:
    return sorted(data_dir.glob("physics_shard_*.npz"))


def choose_scene_indices(shards: List[Path], heldout_scenes: int, seed: int) -> set:
    scene_counts = []
    total = 0
    for shard in shards:
        with np.load(shard) as data:
            count = len(data["feature_offsets"]) - 1
        scene_counts.append(count)
        total += count
    heldout_scenes = min(heldout_scenes, total)
    rng = np.random.default_rng(seed)
    selected = rng.choice(total, size=heldout_scenes, replace=False)
    return set(int(idx) for idx in selected.tolist())


def sample_rows(rows: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if len(rows) <= max_rows:
        return rows
    indices = rng.choice(len(rows), size=max_rows, replace=False)
    return rows[np.sort(indices)]


def collect_training_arrays(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    shards = list_shards(Path(args.data_dir))
    selected_indices = choose_scene_indices(shards, args.heldout_scenes, args.seed)
    rng = np.random.default_rng(args.seed)

    feature_batches = []
    interaction_batches = []
    global_scene_idx = 0

    for shard in shards:
        with np.load(shard) as data:
            features = data[args.feature_key]
            feature_offsets = data["feature_offsets"]
            interaction_features = data["interaction_features"]
            interaction_offsets = data["interaction_offsets"]
            num_scenes = len(feature_offsets) - 1

            for local_scene_idx in range(num_scenes):
                if global_scene_idx not in selected_indices:
                    global_scene_idx += 1
                    continue

                f_start = int(feature_offsets[local_scene_idx])
                f_end = int(feature_offsets[local_scene_idx + 1])
                scene_rows = sample_rows(features[f_start:f_end], args.max_vectors_per_scene, rng)
                feature_batches.append(scene_rows.astype(np.float32))

                if args.train_interaction:
                    i_start = int(interaction_offsets[local_scene_idx])
                    i_end = int(interaction_offsets[local_scene_idx + 1])
                    if i_end > i_start:
                        interaction_rows = sample_rows(
                            interaction_features[i_start:i_end], args.max_interactions_per_scene, rng
                        )
                        interaction_batches.append(interaction_rows.astype(np.float32))
                global_scene_idx += 1

        print(
            f"loaded_shard={shard.name} selected_scenes_so_far={len(feature_batches)}",
            flush=True,
        )

    features = np.concatenate(feature_batches, axis=0)
    interactions = (
        np.concatenate(interaction_batches, axis=0).astype(np.float32)
        if interaction_batches
        else np.zeros((0, 13), dtype=np.float32)
    )
    return features, interactions


def split_train_val(rows: np.ndarray, seed: int, val_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    split = int(len(rows) * (1.0 - val_fraction))
    return rows[indices[:split]], rows[indices[split:]]


@torch.no_grad()
def run_kmeans(
    vectors: torch.Tensor,
    codebook_size: int,
    iters: int,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    if vectors.shape[0] < codebook_size:
        repeat = (codebook_size + vectors.shape[0] - 1) // vectors.shape[0]
        vectors = vectors.repeat(repeat, 1)
    perm = torch.randperm(vectors.shape[0])[:codebook_size]
    centers = vectors[perm].to(device).clone()

    for _ in range(iters):
        counts = torch.zeros(codebook_size, device=device)
        sums = torch.zeros_like(centers)
        for start in range(0, vectors.shape[0], chunk_size):
            batch = vectors[start : start + chunk_size].to(device)
            distances = (
                batch.pow(2).sum(dim=1, keepdim=True)
                - 2.0 * batch @ centers.t()
                + centers.pow(2).sum(dim=1, keepdim=True).t()
            )
            assign = distances.argmin(dim=1)
            counts += torch.bincount(assign, minlength=codebook_size).to(counts.dtype)
            sums.index_add_(0, assign, batch)
        used = counts > 0
        centers[used] = sums[used] / counts[used].unsqueeze(1)
        if (~used).any():
            refill_idx = torch.randint(0, vectors.shape[0], ((~used).sum().item(),))
            centers[~used] = vectors[refill_idx].to(device)
    return centers.detach()


@torch.no_grad()
def initialize_codebook_from_latents(
    model: MLPVQVAE,
    train_norm: np.ndarray,
    sample_size: int,
    kmeans_iters: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, len(train_norm))
    sample_idx = rng.choice(len(train_norm), size=sample_size, replace=False)
    sample = torch.from_numpy(train_norm[sample_idx].astype(np.float32))
    latent_batches = []
    for start in range(0, len(sample), batch_size):
        batch = sample[start : start + batch_size].to(device, non_blocking=True)
        latent_batches.append(model.encode_latents(batch).cpu())
    latents = torch.cat(latent_batches, dim=0)
    centers = run_kmeans(
        vectors=latents,
        codebook_size=model.config.codebook_size,
        iters=kmeans_iters,
        device=device,
        chunk_size=batch_size,
    )
    model.initialize_codebook(centers.to(device))


def warmup_autoencoder(
    model: MLPVQVAE,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    output_prefix: str,
) -> None:
    if epochs <= 0:
        return
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=3e-4,
    )
    for epoch in range(1, epochs + 1):
        running = 0.0
        examples = 0
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon = model.decoder(model.encoder(batch))
            loss = F.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(batch)
            examples += len(batch)
        print(
            f"{output_prefix} warmup_epoch={epoch}/{epochs} recon={running / max(examples, 1):.6f}",
            flush=True,
        )


def fit_model(
    rows: np.ndarray,
    output_prefix: str,
    config: VQVAEConfig,
    stats_path: Path,
    model_path: Path,
    csv_path: Path,
    epochs: int,
    warmup_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    val_fraction: float,
    num_workers: int,
    kmeans_init_sample_size: int,
    kmeans_iters: int,
    reinit_util_threshold: float,
    reinit_patience_epochs: int,
) -> dict:
    train_rows, val_rows = split_train_val(rows, seed=seed, val_fraction=val_fraction)
    mean = train_rows.mean(axis=0).astype(np.float32)
    std = train_rows.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    np.savez(stats_path, mean=mean, std=std)

    train_norm = ((train_rows - mean) / std).astype(np.float32)
    val_norm = ((val_rows - mean) / std).astype(np.float32)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_norm)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_tensor = torch.from_numpy(val_norm)

    model = MLPVQVAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger = CSVLogger(str(csv_path), ["epoch", "train_loss", "val_recon_error", "codebook_utilization"])

    warmup_autoencoder(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=warmup_epochs,
        output_prefix=output_prefix,
    )
    initialize_codebook_from_latents(
        model=model,
        train_norm=train_norm,
        sample_size=kmeans_init_sample_size,
        kmeans_iters=kmeans_iters,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    print(
        f"{output_prefix} initialized_codebook sample_size={min(kmeans_init_sample_size, len(train_norm))} "
        f"kmeans_iters={kmeans_iters}",
        flush=True,
    )

    best_summary = {}
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        examples = 0
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = model(batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch)
            examples += len(batch)

        model.eval()
        with torch.no_grad():
            val_indices_chunks = []
            val_recon_total = 0.0
            val_examples = 0
            for start in range(0, len(val_tensor), batch_size):
                batch = val_tensor[start : start + batch_size].to(device, non_blocking=True)
                _, val_recon, val_indices, _ = model(batch)
                val_indices_chunks.append(val_indices.reshape(-1).cpu())
                val_recon_total += float(val_recon.item()) * len(batch)
                val_examples += len(batch)
            all_indices = torch.cat(val_indices_chunks, dim=0)
            unique_codes = torch.unique(all_indices).numel()
            utilization = 100.0 * unique_codes / config.codebook_size
            summary = {
                "epoch": epoch,
                "train_loss": running_loss / max(examples, 1),
                "val_recon_error": val_recon_total / max(val_examples, 1),
                "codebook_utilization": utilization,
            }
            logger.log(summary)
            best_summary = summary
            print(
                f"{output_prefix} epoch={epoch}/{epochs} "
                f"train_loss={summary['train_loss']:.6f} "
                f"val_recon={summary['val_recon_error']:.6f} "
                f"util={summary['codebook_utilization']:.2f}%",
                flush=True,
            )

            if epoch <= reinit_patience_epochs and utilization < reinit_util_threshold:
                print(
                    f"{output_prefix} utilization {utilization:.2f}% below threshold "
                    f"{reinit_util_threshold:.2f}% at epoch {epoch}; reinitializing codebook.",
                    flush=True,
                )
                initialize_codebook_from_latents(
                    model=model,
                    train_norm=train_norm,
                    sample_size=kmeans_init_sample_size,
                    kmeans_iters=kmeans_iters,
                    batch_size=batch_size,
                    device=device,
                    seed=seed + epoch,
                )

    torch.save(
        {
            "config": config.__dict__,
            "state_dict": model.state_dict(),
            "stats_path": str(stats_path),
        },
        model_path,
    )
    best_summary["model_path"] = str(model_path)
    best_summary["stats_path"] = str(stats_path)
    save_json(str(model_path.with_suffix(".summary.json")), best_summary)
    print(
        f"{output_prefix}: val recon error={best_summary['val_recon_error']:.6f}, "
        f"codebook utilization={best_summary['codebook_utilization']:.2f}%"
    )
    return best_summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(ensure_dir(args.output_dir))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    features, interactions = collect_training_arrays(args)
    print(
        f"collected_features={features.shape} collected_interactions={interactions.shape}",
        flush=True,
    )
    if features.shape[1] != args.feature_input_dim:
        raise ValueError(
            f"Feature key {args.feature_key} has dim {features.shape[1]}, expected {args.feature_input_dim}."
        )

    feature_summary = fit_model(
        rows=features,
        output_prefix="kinematic_vqvae",
        config=VQVAEConfig(
            input_dim=args.feature_input_dim,
            hidden_dims=args.feature_hidden_dims,
            embedding_dim=args.feature_embedding_dim,
            codebook_size=args.feature_codebook_size,
            commitment_beta=args.commitment_beta,
            decay=args.ema_decay,
            dead_code_threshold=2.0,
        ),
        stats_path=output_dir / "kinematic_stats.npz",
        model_path=output_dir / "kinematic_vqvae.pt",
        csv_path=output_dir / "kinematic_vqvae.csv",
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        kmeans_init_sample_size=args.kmeans_init_sample_size,
        kmeans_iters=args.kmeans_iters,
        reinit_util_threshold=args.reinit_util_threshold,
        reinit_patience_epochs=args.reinit_patience_epochs,
    )

    interaction_summary = {}
    if args.train_interaction and len(interactions):
        interaction_summary = fit_model(
            rows=interactions,
            output_prefix="interaction_vqvae",
            config=VQVAEConfig(
                input_dim=13,
                hidden_dims=args.interaction_hidden_dims,
                embedding_dim=args.interaction_embedding_dim,
                codebook_size=args.interaction_codebook_size,
                commitment_beta=args.commitment_beta,
                decay=args.ema_decay,
                dead_code_threshold=2.0,
            ),
            stats_path=output_dir / "interaction_stats.npz",
            model_path=output_dir / "interaction_vqvae.pt",
            csv_path=output_dir / "interaction_vqvae.csv",
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed,
            val_fraction=args.val_fraction,
            num_workers=args.num_workers,
            kmeans_init_sample_size=min(args.kmeans_init_sample_size, 16384),
            kmeans_iters=args.kmeans_iters,
            reinit_util_threshold=args.reinit_util_threshold,
            reinit_patience_epochs=args.reinit_patience_epochs,
        )

    summary = {
        "feature_key": args.feature_key,
        "feature_rows_used": int(len(features)),
        "interaction_rows_used": int(len(interactions)),
        "kinematic_summary": feature_summary,
        "interaction_summary": interaction_summary,
    }
    save_json(str(output_dir / "training_summary.json"), summary)


if __name__ == "__main__":
    main()
