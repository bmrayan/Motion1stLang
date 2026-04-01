#!/usr/bin/env python
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

from _gpt_utils import evaluate_loss
from _motion_utils import FlatTokenDataset, compute_gzip_complexity, save_json


def parse_kv_list(items: List[str]) -> Dict[str, str]:
    parsed = {}
    for item in items:
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def cmd_corpus_gzip(args: argparse.Namespace) -> None:
    corpora = parse_kv_list(args.inputs)
    results = {}
    for name, path in corpora.items():
        token_path = Path(path) / "tokens.bin"
        data = np.memmap(token_path, dtype=np.uint16, mode="r")
        sample = np.asarray(data[: min(len(data), args.sample_tokens)], dtype=np.uint16)
        results[name] = {
            "path": str(token_path),
            "sample_tokens": int(len(sample)),
            "gzip_complexity_percent": float(compute_gzip_complexity(sample)),
        }
    save_json(args.output, results)
    print(f"Saved gzip complexity summary to {args.output}")


def cmd_perplexity(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint_dir).to(device)
    dataset = FlatTokenDataset(str(Path(args.data_path) / "val.bin"), seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loss = evaluate_loss(model, loader, device=device, mixed_precision=args.mixed_precision, max_batches=args.max_batches)
    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "data_path": args.data_path,
        "val_loss": float(val_loss),
        "val_perplexity": float(math.exp(min(val_loss, 20.0))),
    }
    save_json(args.output, result)
    print(result)


def read_metrics_csv(path: str) -> List[Dict[str, float]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value in ("", None):
                    parsed[key] = None
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def first_tokens_to_threshold(rows: List[Dict[str, float]], threshold: float) -> float:
    for row in rows:
        val_ppl = row.get("val_perplexity")
        if val_ppl is None:
            continue
        if val_ppl <= threshold:
            return row["tokens_seen"]
    return float("inf")


def cmd_speedups(args: argparse.Namespace) -> None:
    logs = parse_kv_list(args.logs)
    rows_by_name = {name: read_metrics_csv(path) for name, path in logs.items()}
    scratch_rows = rows_by_name["scratch"]
    scratch_final = None
    for row in reversed(scratch_rows):
        if row.get("val_perplexity") is not None:
            scratch_final = row["val_perplexity"]
            break
    if scratch_final is None:
        raise ValueError("Scratch log has no validation perplexity.")

    results = {"scratch_final_perplexity": scratch_final}
    for name, rows in rows_by_name.items():
        if name == "scratch":
            continue
        tokens_needed = first_tokens_to_threshold(rows, scratch_final)
        scratch_tokens = scratch_rows[-1]["tokens_seen"]
        results[name] = {
            "tokens_to_threshold": tokens_needed,
            "speedup_vs_scratch": float(scratch_tokens / tokens_needed) if math.isfinite(tokens_needed) and tokens_needed > 0 else None,
        }
    save_json(args.output, results)
    print(f"Saved convergence speedups to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tier 1 evaluation helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gzip_parser = subparsers.add_parser("corpus-gzip")
    gzip_parser.add_argument("--inputs", nargs="+", required=True, help="name=/path/to/tokenized_dir")
    gzip_parser.add_argument("--sample_tokens", type=int, default=2_000_000)
    gzip_parser.add_argument("--output", type=str, required=True)
    gzip_parser.set_defaults(func=cmd_corpus_gzip)

    ppl_parser = subparsers.add_parser("perplexity")
    ppl_parser.add_argument("--checkpoint_dir", type=str, required=True)
    ppl_parser.add_argument("--data_path", type=str, required=True)
    ppl_parser.add_argument("--output", type=str, required=True)
    ppl_parser.add_argument("--seq_len", type=int, default=1024)
    ppl_parser.add_argument("--batch_size", type=int, default=8)
    ppl_parser.add_argument("--num_workers", type=int, default=4)
    ppl_parser.add_argument("--max_batches", type=int, default=None)
    ppl_parser.add_argument("--mixed_precision", type=str, default="bf16")
    ppl_parser.add_argument("--device", type=str, default="cuda")
    ppl_parser.set_defaults(func=cmd_perplexity)

    speed_parser = subparsers.add_parser("speedups")
    speed_parser.add_argument("--logs", nargs="+", required=True, help="scratch=... physics=... nca=...")
    speed_parser.add_argument("--output", type=str, required=True)
    speed_parser.set_defaults(func=cmd_speedups)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
