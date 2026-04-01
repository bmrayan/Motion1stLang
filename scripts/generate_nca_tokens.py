#!/usr/bin/env python
import argparse
import math
import sys
from pathlib import Path

import numpy as np

from _motion_utils import ensure_dir, save_json, set_seed, token_frequency_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the standard NCA token corpus using the official Han et al. repo implementation."
    )
    parser.add_argument("--repo_dir", type=str, default="/workspace/repos/nca-pre-pretraining")
    parser.add_argument("--output_dir", type=str, default="/workspace/data/nca_tokenized")
    parser.add_argument("--target_tokens", type=int, default=164_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid", type=int, default=12)
    parser.add_argument("--patch", type=int, default=2)
    parser.add_argument("--num_colors", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1e-4)
    parser.add_argument("--identity_bias", type=float, default=0.0)
    parser.add_argument("--dT", type=int, default=1)
    parser.add_argument("--init_rollout_steps", type=int, default=10)
    parser.add_argument("--filter_n_steps", type=int, default=10)
    parser.add_argument("--filter_rules_threshold", type=float, default=0.5)
    parser.add_argument("--filter_rules_upper_bound", type=float, default=1.0)
    parser.add_argument("--filter_rules_mode", type=str, default="gzip", choices=["gzip", "diff"])
    parser.add_argument("--rules_per_batch", type=int, default=1024)
    parser.add_argument("--sequences_per_batch", type=int, default=4096)
    parser.add_argument("--max_batches", type=int, default=None)
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
    grid_len = (args.grid // args.patch) ** 2 + 2
    num_examples = int(math.ceil(args.seq_len / grid_len))
    vocab_size = args.num_colors ** (args.patch ** 2) + 2

    total_tokens = 0
    sequences_kept = 0
    batch_idx = 0
    sequence_lengths = []
    sequence_complexities = []
    rng = jax.random.PRNGKey(args.seed)

    with open(token_path, "wb") as handle:
        while total_tokens < args.target_tokens and (args.max_batches is None or batch_idx < args.max_batches):
            rng, rule_rng, sim_rng = jax.random.split(rng, 3)
            rule_seeds = generate_rules_batch(
                seed=rule_rng,
                num_rules=args.rules_per_batch,
                tokenizer=tokenizer,
                threshold=args.filter_rules_threshold,
                upper_bound=args.filter_rules_upper_bound,
                dT=args.dT,
                n_steps=args.filter_n_steps,
                mode=args.filter_rules_mode,
                start_step=args.init_rollout_steps,
                grid=args.grid,
                d_state=args.num_colors,
                identity_bias=args.identity_bias,
                temperature=args.temperature,
            )

            sims = generate_nca_dataset(
                sim_rng,
                num_sims=args.sequences_per_batch,
                grid=args.grid,
                d_state=args.num_colors,
                n_groups=1,
                identity_bias=args.identity_bias,
                temperature=args.temperature,
                num_examples=num_examples,
                num_rules=int(rule_seeds.shape[0]),
                dT=args.dT,
                start_step=args.init_rollout_steps,
                rule_seeds=rule_seeds,
            )

            seq, _ = tokenizer.encode_task(sims)
            seq_np = seq.numpy().astype(np.uint16, copy=False)

            for row in seq_np:
                if total_tokens >= args.target_tokens:
                    break
                row.tofile(handle)
                total_tokens += int(len(row))
                sequences_kept += 1
                sequence_lengths.append(int(len(row)))
                sequence_complexities.append(float(gzip_complexity(row.tobytes()) * 100.0))

            batch_idx += 1
            print(
                f"batch={batch_idx} sequences={sequences_kept} tokens={total_tokens} "
                f"gzip_mean={np.mean(sequence_complexities):.2f}",
                flush=True,
            )

    summary = {
        "source": "official_han_repo",
        "repo_dir": str(repo_dir),
        "token_path": str(token_path),
        "target_tokens": args.target_tokens,
        "tokens_written": total_tokens,
        "sequences_kept": sequences_kept,
        "grid": args.grid,
        "patch": args.patch,
        "num_colors": args.num_colors,
        "seq_len_target": args.seq_len,
        "num_examples_per_sequence": num_examples,
        "temperature": args.temperature,
        "identity_bias": args.identity_bias,
        "dT": args.dT,
        "init_rollout_steps": args.init_rollout_steps,
        "filter_rules_threshold_raw": args.filter_rules_threshold,
        "filter_rules_upper_bound_raw": args.filter_rules_upper_bound,
        "filter_rules_mode": args.filter_rules_mode,
        "vocab_size": vocab_size,
        "special_tokens": {
            "start": int(tokenizer.start_tk),
            "end": int(tokenizer.end_tk),
        },
        "sequence_lengths": {
            "mean": float(np.mean(sequence_lengths) if sequence_lengths else 0.0),
            "std": float(np.std(sequence_lengths) if sequence_lengths else 0.0),
            "min": int(np.min(sequence_lengths) if sequence_lengths else 0),
            "max": int(np.max(sequence_lengths) if sequence_lengths else 0),
        },
        "gzip_complexity_percent": {
            "mean": float(np.mean(sequence_complexities) if sequence_complexities else 0.0),
            "std": float(np.std(sequence_complexities) if sequence_complexities else 0.0),
            "min": float(np.min(sequence_complexities) if sequence_complexities else 0.0),
            "max": float(np.max(sequence_complexities) if sequence_complexities else 0.0),
        },
    }
    save_json(str(metadata_path), summary)

    sample = np.memmap(token_path, dtype=np.uint16, mode="r")
    freq = token_frequency_summary(sample[: min(len(sample), 2_000_000)])
    save_json(str(output_dir / "token_frequency_summary.json"), freq)

    print(f"Wrote {total_tokens} official NCA tokens across {sequences_kept} sequences")
    print(
        "Gzip complexity mean/std: "
        f"{summary['gzip_complexity_percent']['mean']:.2f} / {summary['gzip_complexity_percent']['std']:.2f}"
    )


if __name__ == "__main__":
    main()
