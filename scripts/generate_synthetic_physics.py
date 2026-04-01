#!/usr/bin/env python
import argparse
import itertools
import math
import multiprocessing as mp
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from _motion_utils import ensure_dir, save_json, sequence_length_summary, set_seed


FPS = 30.0
DT = 1.0 / FPS
BOX_BOUNDS = 3.0
GROUND_Z = 0.0
CEILING_Z = BOX_BOUNDS
POSITION_ONLY_DIM = 4
POSITION_VELOCITY_DIM = 8


@dataclass
class ShapeSpec:
    shape: str
    params: Dict[str, float]
    mass: float
    volume: float
    collision_radius: float
    inertia_scalar: float


def random_unit_quaternion(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4).astype(np.float32)
    q /= np.linalg.norm(q) + 1e-8
    if q[0] < 0:
        q *= -1.0
    return q


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def integrate_quaternion(q: np.ndarray, angular_velocity: np.ndarray, dt: float) -> np.ndarray:
    omega_norm = np.linalg.norm(angular_velocity)
    if omega_norm < 1e-8:
        return q
    axis = angular_velocity / omega_norm
    half_theta = 0.5 * omega_norm * dt
    dq = np.array(
        [math.cos(half_theta), *(math.sin(half_theta) * axis)],
        dtype=np.float32,
    )
    out = quaternion_multiply(q, dq)
    out /= np.linalg.norm(out) + 1e-8
    if out[0] < 0:
        out *= -1.0
    return out.astype(np.float32)


def build_shape(rng: np.random.Generator) -> ShapeSpec:
    shape = rng.choice(["sphere", "box", "cylinder"])
    mass = float(rng.uniform(0.5, 5.0))
    if shape == "sphere":
        radius = float(rng.uniform(0.12, 0.38))
        volume = 4.0 / 3.0 * math.pi * radius**3
        collision_radius = radius
        inertia = 0.4 * mass * radius**2
        params = {"radius": radius}
    elif shape == "box":
        dims = rng.uniform(0.18, 0.5, size=3).astype(np.float32)
        volume = float(np.prod(dims))
        collision_radius = float(0.5 * np.linalg.norm(dims))
        inertia = float((mass / 12.0) * float(np.sum(dims**2)))
        params = {"width": float(dims[0]), "depth": float(dims[1]), "height": float(dims[2])}
    else:
        radius = float(rng.uniform(0.12, 0.35))
        height = float(rng.uniform(0.2, 0.7))
        volume = math.pi * radius**2 * height
        collision_radius = float(math.sqrt(radius**2 + (0.5 * height) ** 2))
        inertia = float(0.5 * mass * radius**2 + (mass * height**2) / 12.0)
        params = {"radius": radius, "height": height}
    return ShapeSpec(
        shape=shape,
        params=params,
        mass=mass,
        volume=float(volume),
        collision_radius=float(collision_radius),
        inertia_scalar=max(float(inertia), 1e-4),
    )


def sample_scene_parameters(rng: np.random.Generator) -> Dict[str, float]:
    restitution = 0.8
    friction = 0.2
    gravity = -9.81
    if rng.random() < 0.30:
        restitution = float(rng.uniform(0.2, 0.95))
    if rng.random() < 0.30:
        friction = float(rng.uniform(0.05, 0.8))
    if rng.random() < 0.10:
        gravity = float(rng.uniform(-15.0, -2.0))
    return {"restitution": restitution, "friction": friction, "gravity": gravity}


def initialize_objects(rng: np.random.Generator, num_objects: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[ShapeSpec]]:
    specs = [build_shape(rng) for _ in range(num_objects)]
    positions = np.zeros((num_objects, 3), dtype=np.float32)
    velocities = np.zeros((num_objects, 3), dtype=np.float32)
    angular_velocities = np.zeros((num_objects, 3), dtype=np.float32)
    quaternions = np.zeros((num_objects, 4), dtype=np.float32)

    for idx, spec in enumerate(specs):
        placed = False
        z_min = GROUND_Z + spec.collision_radius + 0.05
        z_max = max(z_min + 0.1, CEILING_Z - spec.collision_radius - 0.05)
        for _ in range(256):
            candidate = np.array(
                [
                    rng.uniform(-BOX_BOUNDS + spec.collision_radius, BOX_BOUNDS - spec.collision_radius),
                    rng.uniform(-BOX_BOUNDS + spec.collision_radius, BOX_BOUNDS - spec.collision_radius),
                    rng.uniform(z_min, z_max),
                ],
                dtype=np.float32,
            )
            valid = True
            for prev in range(idx):
                min_dist = specs[prev].collision_radius + spec.collision_radius + 0.1
                if np.linalg.norm(candidate - positions[prev]) < min_dist:
                    valid = False
                    break
            if valid:
                positions[idx] = candidate
                placed = True
                break
        if not placed:
            positions[idx] = np.array([0.0, 0.0, z_min], dtype=np.float32)
        velocities[idx] = rng.uniform(-1.75, 1.75, size=3).astype(np.float32)
        velocities[idx, 2] += float(rng.uniform(-0.5, 2.5))
        angular_velocities[idx] = rng.uniform(-2.0, 2.0, size=3).astype(np.float32)
        quaternions[idx] = random_unit_quaternion(rng)

    return positions, velocities, angular_velocities, quaternions, specs


def resolve_boundary_collisions(
    position: np.ndarray,
    velocity: np.ndarray,
    angular_velocity: np.ndarray,
    spec: ShapeSpec,
    restitution: float,
    friction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    impulse_total = 0.0
    bounds = [
        (0, -BOX_BOUNDS + spec.collision_radius, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        (0, BOX_BOUNDS - spec.collision_radius, np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
        (1, -BOX_BOUNDS + spec.collision_radius, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        (1, BOX_BOUNDS - spec.collision_radius, np.array([0.0, -1.0, 0.0], dtype=np.float32)),
        (2, GROUND_Z + spec.collision_radius, np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        (2, CEILING_Z - spec.collision_radius, np.array([0.0, 0.0, -1.0], dtype=np.float32)),
    ]

    for axis, limit, normal in bounds:
        if axis in (0, 1):
            lower = -BOX_BOUNDS + spec.collision_radius
            upper = BOX_BOUNDS - spec.collision_radius
            if position[axis] < lower:
                position[axis] = lower
                normal = np.abs(normal)
            elif position[axis] > upper:
                position[axis] = upper
                normal = -np.abs(normal)
            else:
                continue
        else:
            lower = GROUND_Z + spec.collision_radius
            upper = CEILING_Z - spec.collision_radius
            if position[axis] < lower:
                position[axis] = lower
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            elif position[axis] > upper:
                position[axis] = upper
                normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            else:
                continue

        vn = float(np.dot(velocity, normal))
        if vn < 0:
            impulse = -(1.0 + restitution) * vn * spec.mass
            velocity += (impulse / spec.mass) * normal
            tangential = velocity - np.dot(velocity, normal) * normal
            velocity -= min(friction, 0.95) * tangential
            angular_velocity *= 1.0 - min(0.5 * friction, 0.8)
            impulse_total += abs(impulse)

    return position, velocity, angular_velocity, impulse_total


def resolve_object_collisions(
    positions: np.ndarray,
    velocities: np.ndarray,
    angular_velocities: np.ndarray,
    specs: List[ShapeSpec],
    restitution: float,
    friction: float,
) -> Dict[Tuple[int, int], float]:
    impulses: Dict[Tuple[int, int], float] = {}
    num_objects = positions.shape[0]

    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            delta = positions[j] - positions[i]
            distance = float(np.linalg.norm(delta))
            min_distance = specs[i].collision_radius + specs[j].collision_radius
            if distance < 1e-8:
                normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                distance = 1e-8
            else:
                normal = delta / distance

            if distance >= min_distance:
                continue

            overlap = min_distance - distance
            inv_mass_i = 1.0 / specs[i].mass
            inv_mass_j = 1.0 / specs[j].mass
            total_inv_mass = inv_mass_i + inv_mass_j
            positions[i] -= normal * overlap * (inv_mass_i / total_inv_mass)
            positions[j] += normal * overlap * (inv_mass_j / total_inv_mass)

            relative_velocity = velocities[j] - velocities[i]
            normal_speed = float(np.dot(relative_velocity, normal))
            if normal_speed > 0:
                impulses[(i, j)] = impulses.get((i, j), 0.0)
                continue

            normal_impulse_mag = -(1.0 + restitution) * normal_speed / total_inv_mass
            normal_impulse = normal_impulse_mag * normal
            velocities[i] -= normal_impulse * inv_mass_i
            velocities[j] += normal_impulse * inv_mass_j

            tangential_velocity = relative_velocity - normal_speed * normal
            tangential_norm = float(np.linalg.norm(tangential_velocity))
            tangential_impulse = np.zeros(3, dtype=np.float32)
            if tangential_norm > 1e-8:
                tangent = tangential_velocity / tangential_norm
                jt = min(friction * normal_impulse_mag, tangential_norm / total_inv_mass)
                tangential_impulse = -jt * tangent
                velocities[i] -= tangential_impulse * inv_mass_i
                velocities[j] += tangential_impulse * inv_mass_j

            lever_i = normal * specs[i].collision_radius
            lever_j = -normal * specs[j].collision_radius
            angular_velocities[i] += np.cross(lever_i, normal_impulse + tangential_impulse) / specs[i].inertia_scalar
            angular_velocities[j] += np.cross(lever_j, -(normal_impulse + tangential_impulse)) / specs[j].inertia_scalar

            total_impulse = float(np.linalg.norm(normal_impulse + tangential_impulse))
            impulses[(i, j)] = impulses.get((i, j), 0.0) + total_impulse

    return impulses


def finite_difference(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    if len(values) <= 1:
        return out
    out[1:] = (values[1:] - values[:-1]) * FPS
    out[0] = out[1]
    return out


def scene_to_feature_matrix(
    positions: np.ndarray,
    velocities: np.ndarray,
    angular_velocities: np.ndarray,
    quaternions: np.ndarray,
    specs: List[ShapeSpec],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames, num_objects, _ = positions.shape
    accelerations = finite_difference(velocities)
    angular_acc = finite_difference(angular_velocities)
    jerk = finite_difference(accelerations)
    features = []
    posonly = []
    posvel = []
    for obj_idx, spec in enumerate(specs):
        pos = positions[:, obj_idx]
        vel = velocities[:, obj_idx]
        acc = accelerations[:, obj_idx]
        ang = angular_velocities[:, obj_idx]
        ang_accel = angular_acc[:, obj_idx]
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        acc_mag = np.linalg.norm(acc, axis=1, keepdims=True)
        ang_speed = np.linalg.norm(ang, axis=1, keepdims=True)
        ang_acc_mag = np.linalg.norm(ang_accel, axis=1, keepdims=True)
        jerk_mag = np.linalg.norm(jerk[:, obj_idx], axis=1, keepdims=True)
        volume = np.full((num_frames, 1), spec.volume, dtype=np.float32)
        rigid_share = np.ones((num_frames, 1), dtype=np.float32)
        height = pos[:, 2:3]
        kinetic = 0.5 * speed**2
        cross = np.cross(vel, acc)
        curvature = np.linalg.norm(cross, axis=1, keepdims=True) / np.maximum(speed**3, 1e-6)
        quat_w = quaternions[:, obj_idx, 0:1]
        radial_dist = np.linalg.norm(pos[:, :2], axis=1, keepdims=True)
        vertical_vel = vel[:, 2:3]
        obj_features = np.concatenate(
            [
                pos,
                vel,
                acc,
                ang,
                ang_accel,
                speed,
                acc_mag,
                ang_speed,
                volume,
                rigid_share,
                ang_acc_mag,
                height,
                kinetic,
                curvature,
                quat_w,
                jerk_mag,
                radial_dist,
                vertical_vel,
            ],
            axis=1,
        ).astype(np.float32)
        assert obj_features.shape[1] == 28
        features.append(obj_features)
        posonly.append(np.concatenate([pos, volume], axis=1).astype(np.float32))
        posvel.append(np.concatenate([pos, vel, speed, volume], axis=1).astype(np.float32))
    return np.concatenate(features, axis=0), np.concatenate(posonly, axis=0), np.concatenate(posvel, axis=0)


def simulate_scene(scene_index: int, seed: int) -> Optional[Dict[str, object]]:
    rng = np.random.default_rng(seed + scene_index)
    num_objects = int(rng.integers(2, 7))
    num_frames = int(rng.integers(30, 151))
    scene_params = sample_scene_parameters(rng)
    positions, velocities, angular_velocities, quaternions, specs = initialize_objects(rng, num_objects)

    positions_t = np.zeros((num_frames, num_objects, 3), dtype=np.float32)
    velocities_t = np.zeros((num_frames, num_objects, 3), dtype=np.float32)
    angular_t = np.zeros((num_frames, num_objects, 3), dtype=np.float32)
    quaternions_t = np.zeros((num_frames, num_objects, 4), dtype=np.float32)
    interaction_rows: List[np.ndarray] = []
    interaction_pairs: List[np.ndarray] = []
    interaction_frames: List[int] = []
    contacts_per_frame = []
    boundary_impulse_total = 0.0

    for frame_idx in range(num_frames):
        positions_t[frame_idx] = positions
        velocities_t[frame_idx] = velocities
        angular_t[frame_idx] = angular_velocities
        quaternions_t[frame_idx] = quaternions

        pairwise_impulses: Dict[Tuple[int, int], float] = {}
        for i in range(num_objects):
            positions[i], velocities[i], angular_velocities[i], boundary_impulse = resolve_boundary_collisions(
                positions[i],
                velocities[i],
                angular_velocities[i],
                specs[i],
                scene_params["restitution"],
                scene_params["friction"],
            )
            boundary_impulse_total += boundary_impulse

        for i in range(num_objects):
            velocities[i, 2] += scene_params["gravity"] * DT
            positions[i] += velocities[i] * DT
            quaternions[i] = integrate_quaternion(quaternions[i], angular_velocities[i], DT)
            angular_velocities[i] *= 0.995

        pairwise_impulses.update(
            resolve_object_collisions(
                positions,
                velocities,
                angular_velocities,
                specs,
                scene_params["restitution"],
                scene_params["friction"],
            )
        )

        contact_count = 0
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                delta = positions[j] - positions[i]
                distance = float(np.linalg.norm(delta))
                threshold = specs[i].collision_radius + specs[j].collision_radius + 0.03
                if distance > threshold and (i, j) not in pairwise_impulses:
                    continue
                rel_v = velocities[j] - velocities[i]
                rel_ang = angular_velocities[j] - angular_velocities[i]
                normal = delta / max(distance, 1e-8)
                closing_speed = max(0.0, -float(np.dot(rel_v, normal)))
                contact_flag = 1.0 if distance <= specs[i].collision_radius + specs[j].collision_radius + 1e-5 else 0.0
                impulse_mag = float(pairwise_impulses.get((i, j), 0.0))
                if contact_flag > 0 or impulse_mag > 0:
                    contact_count += 1
                interaction_rows.append(
                    np.concatenate(
                        [
                            rel_v,
                            np.array([closing_speed], dtype=np.float32),
                            rel_ang,
                            np.array([distance], dtype=np.float32),
                            np.array([contact_flag], dtype=np.float32),
                            np.array([impulse_mag], dtype=np.float32),
                            delta.astype(np.float32),
                        ]
                    ).astype(np.float32)
                )
                interaction_pairs.append(np.array([i, j], dtype=np.int16))
                interaction_frames.append(frame_idx)
        contacts_per_frame.append(contact_count)

    if not np.isfinite(positions_t).all() or not np.isfinite(velocities_t).all():
        return None

    mean_velocity = float(np.linalg.norm(velocities_t, axis=2).mean())
    if mean_velocity < 0.01:
        return None

    features, posonly, posvel = scene_to_feature_matrix(
        positions_t, velocities_t, angular_t, quaternions_t, specs
    )
    estimated_tokens = features.shape[0] + num_objects + len(interaction_rows) + 1
    if estimated_tokens < 20:
        return None

    metadata = {
        "scene_index": scene_index,
        "num_objects": num_objects,
        "num_frames": num_frames,
        "scene_params": scene_params,
        "object_specs": [asdict(spec) for spec in specs],
        "mean_velocity_magnitude": mean_velocity,
        "contacts_per_frame_mean": float(np.mean(contacts_per_frame) if contacts_per_frame else 0.0),
        "contacts_per_frame_std": float(np.std(contacts_per_frame) if contacts_per_frame else 0.0),
        "boundary_impulse_total": boundary_impulse_total,
        "estimated_tokens": estimated_tokens,
    }

    return {
        "features": features,
        "position_only_features": posonly,
        "position_velocity_features": posvel,
        "interaction_features": np.asarray(interaction_rows, dtype=np.float32).reshape(-1, 13),
        "interaction_pairs": np.asarray(interaction_pairs, dtype=np.int16).reshape(-1, 2),
        "interaction_frames": np.asarray(interaction_frames, dtype=np.int32),
        "scene_feature_lengths": np.full(num_objects, num_frames, dtype=np.int32),
        "metadata": metadata,
        "contacts_per_scene": int(sum(contacts_per_frame)),
        "tokens_per_scene": int(estimated_tokens),
        "trajectory_smoothness": float(np.linalg.norm(np.diff(velocities_t, axis=0), axis=2).mean() if num_frames > 1 else 0.0),
    }


def write_shard(
    shard_id: int,
    output_dir: str,
    shard_scenes: List[Dict[str, object]],
) -> Dict[str, float]:
    shard_dir = ensure_dir(output_dir)
    npz_path = Path(shard_dir) / f"physics_shard_{shard_id:05d}.npz"
    json_path = Path(shard_dir) / f"physics_shard_{shard_id:05d}.json"

    features = []
    posonly = []
    posvel = []
    feature_offsets = [0]
    object_offsets = [0]
    scene_object_offsets = [0]

    interaction_features = []
    interaction_offsets = [0]
    interaction_pairs = []
    interaction_frames = []
    metadata_rows = []

    total_objects = 0
    for scene in shard_scenes:
        scene_features = scene["features"]
        scene_posonly = scene["position_only_features"]
        scene_posvel = scene["position_velocity_features"]
        scene_interactions = scene["interaction_features"]
        scene_pairs = scene["interaction_pairs"]
        scene_frames = scene["interaction_frames"]
        scene_feature_lengths = scene["scene_feature_lengths"]

        features.append(scene_features)
        posonly.append(scene_posonly)
        posvel.append(scene_posvel)
        feature_offsets.append(feature_offsets[-1] + len(scene_features))
        for length in scene_feature_lengths:
            object_offsets.append(object_offsets[-1] + int(length))
        total_objects += len(scene_feature_lengths)
        scene_object_offsets.append(total_objects)

        interaction_features.append(scene_interactions)
        interaction_offsets.append(interaction_offsets[-1] + len(scene_interactions))
        if len(scene_interactions):
            interaction_pairs.append(scene_pairs)
            interaction_frames.append(scene_frames)
        metadata_rows.append(scene["metadata"])

    save_json(str(json_path), {"shard_id": shard_id, "num_scenes": len(shard_scenes), "scenes": metadata_rows})
    np.savez_compressed(
        npz_path,
        features=np.concatenate(features, axis=0).astype(np.float32),
        position_only_features=np.concatenate(posonly, axis=0).astype(np.float32),
        position_velocity_features=np.concatenate(posvel, axis=0).astype(np.float32),
        feature_offsets=np.asarray(feature_offsets, dtype=np.int64),
        object_offsets=np.asarray(object_offsets, dtype=np.int64),
        scene_object_offsets=np.asarray(scene_object_offsets, dtype=np.int64),
        interaction_features=np.concatenate(interaction_features, axis=0).astype(np.float32)
        if interaction_features
        else np.zeros((0, 13), dtype=np.float32),
        interaction_offsets=np.asarray(interaction_offsets, dtype=np.int64),
        interaction_pairs=np.concatenate(interaction_pairs, axis=0).astype(np.int16)
        if interaction_pairs
        else np.zeros((0, 2), dtype=np.int16),
        interaction_frames=np.concatenate(interaction_frames, axis=0).astype(np.int32)
        if interaction_frames
        else np.zeros((0,), dtype=np.int32),
    )

    feature_count = int(feature_offsets[-1])
    contact_counts = [row["contacts_per_frame_mean"] * row["num_frames"] for row in metadata_rows]
    token_counts = [row["estimated_tokens"] for row in metadata_rows]
    return {
        "feature_count": feature_count,
        "scene_count": len(shard_scenes),
        "contact_mean": float(np.mean(contact_counts) if contact_counts else 0.0),
        "contact_std": float(np.std(contact_counts) if contact_counts else 0.0),
        "token_mean": float(np.mean(token_counts) if token_counts else 0.0),
        "token_std": float(np.std(token_counts) if token_counts else 0.0),
        "smoothness_mean": float(np.mean([scene["trajectory_smoothness"] for scene in shard_scenes]) if shard_scenes else 0.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic rigid-body physics shards.")
    parser.add_argument("--output_dir", type=str, default="/workspace/data/synthetic_physics")
    parser.add_argument("--target_feature_vectors", type=int, default=170_000_000)
    parser.add_argument("--shard_size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=max(1, mp.cpu_count() // 2))
    parser.add_argument("--max_scenes", type=int, default=None, help="Optional cap for debugging.")
    return parser.parse_args()


def _simulate_scene_wrapper(task):
    return simulate_scene(*task)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    shard_scenes: List[Dict[str, object]] = []
    shard_stats = []
    total_features = 0
    total_scenes = 0
    shard_id = 0
    contacts = []
    tokens = []
    smoothness = []

    max_scenes = args.max_scenes if args.max_scenes is not None else None
    tasks = (
        ((scene_idx, args.seed) for scene_idx in range(max_scenes))
        if max_scenes is not None
        else ((scene_idx, args.seed) for scene_idx in itertools.count())
    )
    target_reached = False
    pool = mp.Pool(processes=args.num_workers)
    try:
        for scene in pool.imap_unordered(_simulate_scene_wrapper, tasks, chunksize=32):
            if scene is None:
                continue
            shard_scenes.append(scene)
            total_features += len(scene["features"])
            total_scenes += 1
            contacts.append(scene["contacts_per_scene"])
            tokens.append(scene["tokens_per_scene"])
            smoothness.append(scene["trajectory_smoothness"])

            if len(shard_scenes) >= args.shard_size:
                shard_stats.append(write_shard(shard_id, args.output_dir, shard_scenes))
                print(
                    f"Wrote shard {shard_id:05d} "
                    f"(scenes={len(shard_scenes)} total_scenes={total_scenes} total_features={total_features})",
                    flush=True,
                )
                shard_scenes = []
                shard_id += 1

            if total_features >= args.target_feature_vectors:
                target_reached = True
                pool.terminate()
                break
        if not target_reached:
            pool.close()
    finally:
        pool.join()

    if shard_scenes:
        shard_stats.append(write_shard(shard_id, args.output_dir, shard_scenes))
        print(
            f"Wrote shard {shard_id:05d} "
            f"(scenes={len(shard_scenes)} total_scenes={total_scenes} total_features={total_features})",
            flush=True,
        )

    validation = {
        "total_scenes_generated": total_scenes,
        "total_feature_vectors": total_features,
        "contacts_per_scene": sequence_length_summary(contacts),
        "tokens_per_scene": sequence_length_summary(tokens),
        "trajectory_smoothness_check": sequence_length_summary(smoothness),
        "shards_written": len(shard_stats),
    }
    save_json(str(Path(args.output_dir) / "generation_summary.json"), validation)
    print(f"Total scenes generated: {total_scenes}", flush=True)
    print(f"Total feature vectors: {total_features}", flush=True)
    print(
        "Mean/std contacts per scene: "
        f"{validation['contacts_per_scene']['mean']:.3f} / {validation['contacts_per_scene']['std']:.3f}",
        flush=True,
    )
    print(
        "Mean/std tokens per scene: "
        f"{validation['tokens_per_scene']['mean']:.3f} / {validation['tokens_per_scene']['std']:.3f}",
        flush=True,
    )
    print(
        "Sample trajectory smoothness check: "
        f"{validation['trajectory_smoothness_check']['mean']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
