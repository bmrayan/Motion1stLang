#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np

from _motion_utils import ensure_dir, load_json, save_json, set_seed, token_frequency_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create temporally shuffled physics control data.")
    parser.add_argument("--input_dir", type=str, default="/workspace/data/synthetic_physics")
    parser.add_argument("--output_dir", type=str, default="/workspace/data/shuffled_temporal")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def shuffle_shard(npz_path: Path, json_path: Path, output_dir: Path, seed: int):
    rng = np.random.default_rng(seed + hash(npz_path.name) % 100_000)
    data = np.load(npz_path)

    features = data["features"].copy()
    posonly = data["position_only_features"].copy()
    posvel = data["position_velocity_features"].copy()
    object_offsets = data["object_offsets"]
    scene_object_offsets = data["scene_object_offsets"]
    interaction_features = data["interaction_features"].copy()
    interaction_offsets = data["interaction_offsets"]
    interaction_pairs = data["interaction_pairs"].copy()
    interaction_frames = data["interaction_frames"].copy()

    for scene_idx in range(len(scene_object_offsets) - 1):
        object_start = int(scene_object_offsets[scene_idx])
        object_end = int(scene_object_offsets[scene_idx + 1])
        for object_idx in range(object_start, object_end):
            start = int(object_offsets[object_idx])
            end = int(object_offsets[object_idx + 1])
            if end - start <= 1:
                continue
            permutation = rng.permutation(end - start)
            features[start:end] = features[start:end][permutation]
            posonly[start:end] = posonly[start:end][permutation]
            posvel[start:end] = posvel[start:end][permutation]

        i_start = int(interaction_offsets[scene_idx])
        i_end = int(interaction_offsets[scene_idx + 1])
        if i_end - i_start > 1:
            permutation = rng.permutation(i_end - i_start)
            interaction_features[i_start:i_end] = interaction_features[i_start:i_end][permutation]
            interaction_pairs[i_start:i_end] = interaction_pairs[i_start:i_end][permutation]
            interaction_frames[i_start:i_end] = interaction_frames[i_start:i_end][permutation]

    out_npz = output_dir / npz_path.name.replace("physics_", "shuffled_")
    np.savez_compressed(
        out_npz,
        features=features.astype(np.float32),
        position_only_features=posonly.astype(np.float32),
        position_velocity_features=posvel.astype(np.float32),
        feature_offsets=data["feature_offsets"],
        object_offsets=object_offsets,
        scene_object_offsets=scene_object_offsets,
        interaction_features=interaction_features.astype(np.float32),
        interaction_offsets=interaction_offsets,
        interaction_pairs=interaction_pairs.astype(np.int16),
        interaction_frames=interaction_frames.astype(np.int32),
    )

    meta = load_json(str(json_path))
    meta["source_shard"] = npz_path.name
    meta["control"] = "independent_temporal_shuffle_per_object"
    out_json = output_dir / json_path.name.replace("physics_", "shuffled_")
    save_json(str(out_json), meta)

    stats = {
        "feature_rows": int(features.shape[0]),
        "interaction_rows": int(interaction_features.shape[0]),
        "feature_l2_mean": float(np.linalg.norm(features, axis=1).mean()) if len(features) else 0.0,
        "interaction_l2_mean": float(np.linalg.norm(interaction_features, axis=1).mean()) if len(interaction_features) else 0.0,
    }
    return stats


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(ensure_dir(args.output_dir))
    shard_stats = []

    for npz_path in sorted(input_dir.glob("physics_shard_*.npz")):
        json_path = npz_path.with_suffix(".json")
        shard_stats.append(shuffle_shard(npz_path, json_path, output_dir, args.seed))

    summary = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "num_shards": len(shard_stats),
        "feature_rows_total": int(sum(stat["feature_rows"] for stat in shard_stats)),
        "interaction_rows_total": int(sum(stat["interaction_rows"] for stat in shard_stats)),
    }
    save_json(str(output_dir / "shuffle_summary.json"), summary)
    print(f"Shuffled {summary['num_shards']} shards")
    print(f"Feature rows preserved: {summary['feature_rows_total']}")
    print(f"Interaction rows preserved: {summary['interaction_rows_total']}")


if __name__ == "__main__":
    main()
