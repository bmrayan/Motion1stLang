#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np

from _motion_utils import ensure_dir, save_json


def summarize_array(values):
    values = np.asarray(values)
    return {
        "mean": float(values.mean()) if values.size else 0.0,
        "std": float(values.std()) if values.size else 0.0,
        "min": float(values.min()) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
        "median": float(np.median(values)) if values.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generated synthetic physics shards.")
    parser.add_argument("--input_dir", type=str, default="/workspace/data/synthetic_physics")
    parser.add_argument("--output_path", type=str, default="/workspace/results/physics_data_quality.json")
    parser.add_argument("--feature_sample_per_shard", type=int, default=5000)
    parser.add_argument("--interaction_sample_per_shard", type=int, default=3000)
    parser.add_argument("--trajectory_objects_per_shard", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.input_dir)
    shards = sorted(root.glob("physics_shard_*.npz"))
    rng = np.random.default_rng(args.seed)

    num_scenes = 0
    num_objects = 0
    num_feature_rows = 0
    num_interaction_rows = 0
    finite_failures = 0
    scene_consistency_failures = 0

    object_counts = []
    frame_counts = []
    estimated_tokens = []
    interaction_rows_per_scene = []
    contact_rows_per_scene = []
    feature_samples = []
    interaction_samples = []
    trajectory_dv = []

    for shard in shards:
        with np.load(shard) as data:
            features = data["features"]
            interactions = data["interaction_features"]
            feature_offsets = data["feature_offsets"]
            object_offsets = data["object_offsets"]
            scene_object_offsets = data["scene_object_offsets"]
            interaction_offsets = data["interaction_offsets"]

            if not np.isfinite(features).all() or not np.isfinite(interactions).all():
                finite_failures += 1

            shard_scenes = len(feature_offsets) - 1
            shard_objects = len(object_offsets) - 1
            num_scenes += shard_scenes
            num_objects += shard_objects
            num_feature_rows += int(feature_offsets[-1])
            num_interaction_rows += int(interaction_offsets[-1])

            object_lengths = np.diff(object_offsets)
            scene_object_counts = np.diff(scene_object_offsets)
            object_counts.extend(scene_object_counts.tolist())

            object_cursor = 0
            for scene_idx, scene_object_count in enumerate(scene_object_counts):
                lengths = object_lengths[object_cursor : object_cursor + scene_object_count]
                object_cursor += scene_object_count
                if len(np.unique(lengths)) != 1:
                    scene_consistency_failures += 1
                frames = int(lengths[0])
                frame_counts.append(frames)

                interaction_count = int(interaction_offsets[scene_idx + 1] - interaction_offsets[scene_idx])
                interaction_rows_per_scene.append(interaction_count)
                interaction_slice = interactions[interaction_offsets[scene_idx] : interaction_offsets[scene_idx + 1]]
                contact_rows_per_scene.append(int(interaction_slice[:, 8].sum()) if len(interaction_slice) else 0)
                estimated_tokens.append(int(scene_object_count * frames + scene_object_count + interaction_count + 1))

            sample_n = min(args.feature_sample_per_shard, len(features))
            feature_samples.append(features[rng.choice(len(features), size=sample_n, replace=False)])

            if len(interactions):
                sample_n_i = min(args.interaction_sample_per_shard, len(interactions))
                interaction_samples.append(interactions[rng.choice(len(interactions), size=sample_n_i, replace=False)])

            for start, end in zip(object_offsets[: args.trajectory_objects_per_shard], object_offsets[1 : args.trajectory_objects_per_shard + 1]):
                traj = features[start:end]
                if len(traj) > 1:
                    vel = traj[:, 3:6]
                    trajectory_dv.append(float(np.linalg.norm(np.diff(vel, axis=0), axis=1).mean()))

    feature_sample = np.concatenate(feature_samples, axis=0)
    interaction_sample = np.concatenate(interaction_samples, axis=0) if interaction_samples else np.zeros((0, 13), dtype=np.float32)
    feature_std = feature_sample.std(axis=0)

    summary = {
        "num_shards": len(shards),
        "num_scenes": int(num_scenes),
        "num_objects": int(num_objects),
        "num_feature_rows": int(num_feature_rows),
        "num_interaction_rows": int(num_interaction_rows),
        "finite_failures": int(finite_failures),
        "scene_consistency_failures": int(scene_consistency_failures),
        "objects_per_scene": summarize_array(object_counts),
        "frames_per_scene": summarize_array(frame_counts),
        "tokens_per_scene_estimated": {
            **summarize_array(estimated_tokens),
            "total_estimated": int(np.sum(estimated_tokens)),
        },
        "interaction_rows_per_scene": summarize_array(interaction_rows_per_scene),
        "contact_rows_per_scene": {
            **summarize_array(contact_rows_per_scene),
            "zero_fraction": float(np.mean(np.asarray(contact_rows_per_scene) == 0)),
        },
        "trajectory_smoothness_dv_per_frame": summarize_array(trajectory_dv),
        "feature_dim_mean": feature_sample.mean(axis=0).tolist(),
        "feature_dim_std": feature_std.tolist(),
        "feature_constant_like_dims": [int(idx) for idx, std in enumerate(feature_std) if std < 1e-6],
        "feature_named_stats": {
            "speed": summarize_array(feature_sample[:, 15]),
            "acceleration_mag": summarize_array(feature_sample[:, 16]),
            "angular_speed": summarize_array(feature_sample[:, 17]),
            "height": summarize_array(feature_sample[:, 21]),
            "curvature": summarize_array(feature_sample[:, 23]),
            "jerk_mag": summarize_array(feature_sample[:, 25]),
            "radial_dist": summarize_array(feature_sample[:, 26]),
        },
    }

    if len(interaction_sample):
        summary["interaction_named_stats"] = {
            "relative_velocity_norm": summarize_array(np.linalg.norm(interaction_sample[:, 0:3], axis=1)),
            "closing_speed": summarize_array(interaction_sample[:, 3]),
            "relative_angular_velocity_norm": summarize_array(np.linalg.norm(interaction_sample[:, 4:7], axis=1)),
            "distance": summarize_array(interaction_sample[:, 7]),
            "contact_flag": summarize_array(interaction_sample[:, 8]),
            "impulse_magnitude": summarize_array(interaction_sample[:, 9]),
        }

    ensure_dir(str(Path(args.output_path).parent))
    save_json(args.output_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
