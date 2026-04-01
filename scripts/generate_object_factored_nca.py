#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np

from _motion_utils import ensure_dir, save_json, set_seed, token_frequency_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate object-factored NCA tokens using the official Han et al. NCA implementation."
    )
    parser.add_argument("--repo_dir", type=str, default="/workspace/repos/nca-pre-pretraining")
    parser.add_argument("--output_dir", type=str, default="/workspace/data/of_nca_tokenized")
    parser.add_argument("--target_tokens", type=int, default=164_000_000)
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--patch", type=int, default=2)
    parser.add_argument("--num_colors", type=int, default=10)
    parser.add_argument("--min_objects", type=int, default=2)
    parser.add_argument("--max_objects", type=int, default=6)
    parser.add_argument("--timesteps_min", type=int, default=20)
    parser.add_argument("--timesteps_max", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1e-4)
    parser.add_argument("--identity_bias", type=float, default=0.0)
    parser.add_argument("--dT", type=int, default=1)
    parser.add_argument("--init_rollout_steps", type=int, default=10)
    parser.add_argument("--filter_n_steps", type=int, default=10)
    parser.add_argument("--filter_rules_threshold", type=float, default=0.5)
    parser.add_argument("--filter_rules_upper_bound", type=float, default=1.0)
    parser.add_argument("--filter_rules_mode", type=str, default="gzip", choices=["gzip", "diff"])
    parser.add_argument("--max_scenes", type=int, default=None)
    return parser.parse_args()


def load_official_nca(repo_dir: Path):
    repo_dir = repo_dir.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    import jax
    from utils.nca import generate_nca_dataset, generate_rules_batch, gzip_complexity
    from utils.tokenizers import NCA_Tokenizer

    return jax, generate_nca_dataset, generate_rules_batch, gzip_complexity, NCA_Tokenizer


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    repo_dir = Path(args.repo_dir)
    output_dir = Path(ensure_dir(args.output_dir))
    token_path = output_dir / "tokens.bin"
    metadata_path = output_dir / "metadata.json"

    jax, generate_nca_dataset, generate_rules_batch, gzip_complexity, NCA_Tokenizer = load_official_nca(repo_dir)
    tokenizer = NCA_Tokenizer(args.patch, num_colors=args.num_colors)
    sep_token = int(tokenizer.end_tk) + 1
    scene_token = int(tokenizer.end_tk) + 2

    total_tokens = 0
    scene_lengths = []
    nca_complexities = []
    num_scenes = 0
    max_scenes = args.max_scenes if args.max_scenes is not None else 10**12
    rng = np.random.default_rng(args.seed)
    jax_rng = jax.random.PRNGKey(args.seed)

    with open(token_path, "wb") as handle:
        while total_tokens < args.target_tokens and num_scenes < max_scenes:
            num_objects = int(rng.integers(args.min_objects, args.max_objects + 1))
            pieces = []
            complexities = []
            valid = True

            for _ in range(num_objects):
                timesteps = int(rng.integers(args.timesteps_min, args.timesteps_max + 1))
                jax_rng, rule_rng, sim_rng = jax.random.split(jax_rng, 3)
                rule_seed = generate_rules_batch(
                    seed=rule_rng,
                    num_rules=1,
                    tokenizer=tokenizer,
                    threshold=args.filter_rules_threshold,
                    upper_bound=args.filter_rules_upper_bound,
                    dT=args.dT,
                    n_steps=args.filter_n_steps,
                    mode=args.filter_rules_mode,
                    start_step=args.init_rollout_steps,
                    grid=args.grid_size,
                    d_state=args.num_colors,
                    identity_bias=args.identity_bias,
                    temperature=args.temperature,
                )
                sims = generate_nca_dataset(
                    sim_rng,
                    num_sims=1,
                    grid=args.grid_size,
                    d_state=args.num_colors,
                    n_groups=1,
                    identity_bias=args.identity_bias,
                    temperature=args.temperature,
                    num_examples=timesteps,
                    num_rules=1,
                    dT=args.dT,
                    start_step=args.init_rollout_steps,
                    rule_seeds=rule_seed,
                )
                seq, _ = tokenizer.encode_task(sims)
                object_tokens = seq.numpy().astype(np.uint16, copy=False).reshape(-1)
                complexity = float(gzip_complexity(object_tokens.tobytes()) * 100.0)
                if complexity < args.filter_rules_threshold * 100.0:
                    valid = False
                    break
                complexities.append(complexity)
                pieces.append(object_tokens)
                pieces.append(np.asarray([sep_token], dtype=np.uint16))

            if not valid:
                continue

            pieces.append(np.asarray([scene_token], dtype=np.uint16))
            scene_tokens = np.concatenate(pieces)
            scene_tokens.tofile(handle)
            total_tokens += int(len(scene_tokens))
            scene_lengths.append(int(len(scene_tokens)))
            nca_complexities.extend(complexities)
            num_scenes += 1

            if num_scenes % 1000 == 0:
                print(
                    f"scenes={num_scenes} tokens={total_tokens} gzip_mean={np.mean(nca_complexities):.2f}",
                    flush=True,
                )

    summary = {
        "source": "official_han_repo",
        "repo_dir": str(repo_dir),
        "token_path": str(token_path),
        "target_tokens": args.target_tokens,
        "tokens_written": total_tokens,
        "num_scenes": num_scenes,
        "grid_size": args.grid_size,
        "patch": args.patch,
        "num_colors": args.num_colors,
        "num_objects_range": [args.min_objects, args.max_objects],
        "timesteps_range": [args.timesteps_min, args.timesteps_max],
        "temperature": args.temperature,
        "identity_bias": args.identity_bias,
        "dT": args.dT,
        "init_rollout_steps": args.init_rollout_steps,
        "filter_rules_threshold_raw": args.filter_rules_threshold,
        "filter_rules_upper_bound_raw": args.filter_rules_upper_bound,
        "filter_rules_mode": args.filter_rules_mode,
        "vocab_size": int(scene_token) + 1,
        "special_tokens": {
            "start": int(tokenizer.start_tk),
            "end": int(tokenizer.end_tk),
            "sep": int(sep_token),
            "scene": int(scene_token),
        },
        "scene_length_mean": float(np.mean(scene_lengths) if scene_lengths else 0.0),
        "scene_length_std": float(np.std(scene_lengths) if scene_lengths else 0.0),
        "gzip_complexity_mean": float(np.mean(nca_complexities) if nca_complexities else 0.0),
        "gzip_complexity_std": float(np.std(nca_complexities) if nca_complexities else 0.0),
    }
    save_json(str(metadata_path), summary)

    sample = np.memmap(token_path, dtype=np.uint16, mode="r")
    save_json(
        str(output_dir / "token_frequency_summary.json"),
        token_frequency_summary(sample[: min(len(sample), 2_000_000)]),
    )
    print(f"Wrote {total_tokens} object-factored NCA tokens across {num_scenes} scenes")


if __name__ == "__main__":
    main()
