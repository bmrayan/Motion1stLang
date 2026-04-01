#!/usr/bin/env python
import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


STYLE_MAP = {
    "scratch": {"color": "#808080", "linestyle": "-", "linewidth": 2.0, "label": "Scratch"},
    "nca": {"color": "#1f77b4", "linestyle": "--", "linewidth": 2.0, "label": "NCA"},
    "shuffled": {"color": "#ff7f0e", "linestyle": ":", "linewidth": 2.0, "label": "Shuffled-Temporal"},
    "physics": {"color": "#d62728", "linestyle": "-", "linewidth": 3.0, "label": "Physics"},
    "of_nca": {"color": "#1f77b4", "linestyle": ":", "linewidth": 2.0, "label": "Object-Factored NCA"},
    "posonly": {"color": "#e377c2", "linestyle": "--", "linewidth": 2.0, "label": "Position-Only"},
    "posvel": {"color": "#e377c2", "linestyle": "-", "linewidth": 2.0, "label": "Position+Velocity"},
}


def parse_kv_list(items: List[str]) -> Dict[str, str]:
    out = {}
    for item in items:
        key, value = item.split("=", 1)
        out[key] = value
    return out


def read_metrics(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["val_perplexity"] in ("", None):
                continue
            rows.append(
                {
                    "tokens_seen": float(row["tokens_seen"]),
                    "val_perplexity": float(row["val_perplexity"]),
                }
            )
    return rows


def align_improvement(reference_rows, target_rows):
    ref = {row["tokens_seen"]: row["val_perplexity"] for row in reference_rows}
    improvements_x = []
    improvements_y = []
    for row in target_rows:
        tokens = row["tokens_seen"]
        if tokens not in ref:
            continue
        scratch_ppl = ref[tokens]
        improvement = (scratch_ppl - row["val_perplexity"]) / scratch_ppl * 100.0
        improvements_x.append(tokens)
        improvements_y.append(improvement)
    return improvements_x, improvements_y


def plot_curves(logs: Dict[str, str], output_dir: Path) -> None:
    rows = {name: read_metrics(path) for name, path in logs.items()}
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, series in rows.items():
        xs = [row["tokens_seen"] / 1e6 for row in series]
        ys = [row["val_perplexity"] for row in series]
        style = STYLE_MAP.get(name, {"label": name, "linewidth": 2.0})
        ax.plot(xs, ys, **style)
    ax.set_xlabel("Training Tokens (Millions)")
    ax.set_ylabel("Validation Perplexity")
    ax.set_title("Main Convergence Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_curves.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    scratch_rows = rows["scratch"]
    for name, series in rows.items():
        if name == "scratch":
            continue
        xs, ys = align_improvement(scratch_rows, series)
        style = STYLE_MAP.get(name, {"label": name, "linewidth": 2.0})
        ax.plot(np.asarray(xs) / 1e6, ys, **style)
    ax.set_xlabel("Training Tokens (Millions)")
    ax.set_ylabel("Relative Improvement Over Scratch (%)")
    ax.set_title("Relative Improvement Over Scratch")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "relative_improvement.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, series in rows.items():
        style = STYLE_MAP.get(name, {"label": name, "linewidth": 2.0})
        cropped = [row for row in series if row["tokens_seen"] <= 500_000_000]
        xs = [row["tokens_seen"] / 1e6 for row in cropped]
        ys = [row["val_perplexity"] for row in cropped]
        ax.plot(xs, ys, **style)
    ax.set_xlabel("Training Tokens (Millions)")
    ax.set_ylabel("Validation Perplexity")
    ax.set_title("Early Training Dynamics (First 500M Tokens)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_curves_early.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Tier 1 convergence figures.")
    parser.add_argument("--logs", nargs="+", required=True, help="scratch=... physics=... nca=...")
    parser.add_argument("--output_dir", type=str, default="/workspace/results")
    args = parser.parse_args()
    plot_curves(parse_kv_list(args.logs), Path(args.output_dir))


if __name__ == "__main__":
    main()
