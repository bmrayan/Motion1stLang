import csv
import gzip
import json
import math
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def serializable_config(config: Any) -> Dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return {k: getattr(config, k) for k in dir(config) if not k.startswith("_") and not callable(getattr(config, k))}


def save_json(path: str, payload: Any) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class CSVLogger:
    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        ensure_dir(str(Path(path).parent))
        self._initialized = os.path.exists(path) and os.path.getsize(path) > 0

    def log(self, row: Dict[str, Any]) -> None:
        write_header = not self._initialized
        with open(self.path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
                self._initialized = True
            writer.writerow({key: row.get(key, "") for key in self.fieldnames})


def init_wandb(project: str, name: str, config: Dict[str, Any]):
    try:
        import wandb

        mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"
        run = wandb.init(project=project, name=name, config=config, mode=mode)
        return run
    except Exception:
        return None


def log_wandb(run, payload: Dict[str, Any]) -> None:
    if run is None:
        return
    try:
        run.log(payload)
    except Exception:
        return


def finish_wandb(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        return


def cosine_lr(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.0) -> float:
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return max(1e-8, float(step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


class FlatTokenDataset(Dataset):
    """
    Sequential block dataset over a flat uint16 token stream.
    """

    def __init__(self, bin_path: str, seq_len: int, stride: Optional[int] = None):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        max_start = len(self.data) - (self.seq_len + 1)
        if max_start < 0:
            raise ValueError(f"Token file {bin_path} is too short for seq_len={seq_len}")
        self.starts = np.arange(0, max_start + 1, self.stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = int(self.starts[idx])
        window = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(window[:-1].copy())
        y = torch.from_numpy(window[1:].copy())
        return x, y


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_dir(str(Path(path).parent))
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra_state": extra_state or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return payload.get("extra_state", {})


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def compute_gzip_complexity(token_ids: np.ndarray) -> float:
    token_ids = np.asarray(token_ids)
    if token_ids.dtype != np.uint16:
        token_ids = token_ids.astype(np.uint16)
    raw = token_ids.tobytes()
    if not raw:
        return 0.0
    compressed = gzip.compress(raw)
    return 100.0 * len(compressed) / len(raw)


def token_frequency_summary(token_ids: np.ndarray, top_k: int = 20) -> Dict[str, Any]:
    token_ids = np.asarray(token_ids, dtype=np.int64)
    unique, counts = np.unique(token_ids, return_counts=True)
    order = np.argsort(counts)[::-1]
    order = order[: min(top_k, len(order))]
    return {
        "num_unique_tokens": int(len(unique)),
        "top_tokens": [
            {"token_id": int(unique[i]), "count": int(counts[i]), "frequency": float(counts[i] / len(token_ids))}
            for i in order
        ],
    }


def sequence_length_summary(lengths: Iterable[int]) -> Dict[str, float]:
    lengths = np.asarray(list(lengths), dtype=np.int64)
    if lengths.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(lengths.size),
        "mean": float(lengths.mean()),
        "std": float(lengths.std()),
        "min": float(lengths.min()),
        "max": float(lengths.max()),
    }


def timestamp() -> float:
    return time.time()
