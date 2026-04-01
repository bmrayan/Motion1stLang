#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
from pathlib import Path


BENCHMARK_TO_SCRIPT = {
    "gsm8k": "src/eval/gsm8k.py",
    "humaneval": "src/eval/humaneval.py",
    "bigbench": "src/eval/bigbench.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run official NCA evaluation scripts from the Han et al. repo. "
            "This wrapper intentionally delegates to the repo's own evaluation code."
        )
    )
    parser.add_argument(
        "benchmark",
        choices=sorted(BENCHMARK_TO_SCRIPT.keys()),
        help="Official benchmark entrypoint to run.",
    )
    parser.add_argument(
        "--repo_dir",
        type=str,
        default="/workspace/repos/nca-pre-pretraining",
        help="Path to the official Han et al. repository clone.",
    )
    parser.add_argument(
        "--python_executable",
        type=str,
        default=sys.executable,
        help="Python executable to use for the official script.",
    )
    parser.add_argument(
        "official_args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments passed through verbatim to the official script. "
            "Prefix with '--' before the official arguments."
        ),
    )
    return parser.parse_args()


def extract_flag_value(args: list[str], flag: str) -> str | None:
    for idx, item in enumerate(args):
        if item == flag and idx + 1 < len(args):
            return args[idx + 1]
    return None


def looks_like_hf_gpt2_checkpoint(model_path: str | None) -> bool:
    if not model_path:
        return False
    path = Path(model_path)
    if not path.exists() or not path.is_dir():
        return False
    hf_markers = {"config.json", "generation_config.json", "model.safetensors", "pytorch_model.bin"}
    return any((path / marker).exists() for marker in hf_markers)


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir)
    script_rel = BENCHMARK_TO_SCRIPT[args.benchmark]
    script_path = repo_dir / script_rel

    if not script_path.exists():
        raise FileNotFoundError(f"Official script not found: {script_path}")

    official_args = list(args.official_args)
    if official_args and official_args[0] == "--":
        official_args = official_args[1:]

    model_path = extract_flag_value(official_args, "--model_path")
    if looks_like_hf_gpt2_checkpoint(model_path):
        raise SystemExit(
            "Refusing to run the official Han eval script on a HuggingFace-style GPT-2 checkpoint directory. "
            "Those official benchmark scripts build the repo's own model classes and are not directly compatible "
            "with our HF GPT-2 checkpoints."
        )

    cmd = [args.python_executable, str(script_path), *official_args]
    print("Running official NCA eval command:")
    print(" ".join(cmd))
    print(f"Working directory: {repo_dir}")
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo_dir))
    subprocess.run(cmd, cwd=repo_dir, env=env, check=True)


if __name__ == "__main__":
    main()
