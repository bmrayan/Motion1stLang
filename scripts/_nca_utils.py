import gzip
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class NCARule:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray


def sample_nca_rule(rng: np.random.Generator, num_states: int = 10) -> NCARule:
    return NCARule(
        w1=rng.normal(0.0, 0.55, size=(4, num_states, 3, 3)).astype(np.float32),
        b1=rng.normal(0.0, 0.1, size=(4,)).astype(np.float32),
        w2=rng.normal(0.0, 0.45, size=(16, 4, 1, 1)).astype(np.float32),
        b2=rng.normal(0.0, 0.1, size=(16,)).astype(np.float32),
        w3=rng.normal(0.0, 0.45, size=(num_states, 16, 1, 1)).astype(np.float32),
        b3=rng.normal(0.0, 0.1, size=(num_states,)).astype(np.float32),
    )


def conv2d_wrap(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    batch, channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="wrap")
    out = np.zeros((batch, out_channels, height, width), dtype=np.float32)
    for oy in range(height):
        for ox in range(width):
            patch = x_pad[:, :, oy : oy + kernel_h, ox : ox + kernel_w]
            out[:, :, oy, ox] = np.tensordot(patch, weight, axes=([1, 2, 3], [1, 2, 3]))
    out += bias[None, :, None, None]
    return out


def step_nca(grid: np.ndarray, rule: NCARule, rng: np.random.Generator, temperature: float = 1.0) -> np.ndarray:
    x = np.eye(rule.w3.shape[0], dtype=np.float32)[grid].transpose(2, 0, 1)[None, ...]
    h1 = conv2d_wrap(x, rule.w1, rule.b1)
    h2 = conv2d_wrap(h1, rule.w2, rule.b2)
    h2 = np.maximum(h2, 0.0)
    logits = conv2d_wrap(h2, rule.w3, rule.b3)[0].transpose(1, 2, 0)
    logits = logits / max(temperature, 1e-4)
    logits -= logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True)
    flat = probs.reshape(-1, probs.shape[-1])
    sampled = np.array([rng.choice(probs.shape[-1], p=row) for row in flat], dtype=np.int16)
    return sampled.reshape(grid.shape)


def rollout_nca(
    rng: np.random.Generator,
    grid_size: int,
    timesteps: int,
    num_states: int = 10,
    rule: Optional[NCARule] = None,
    temperature: float = 1.0,
) -> np.ndarray:
    if rule is None:
        rule = sample_nca_rule(rng, num_states=num_states)
    grid = rng.integers(0, num_states, size=(grid_size, grid_size), dtype=np.int16)
    states = [grid.copy()]
    for _ in range(timesteps - 1):
        grid = step_nca(grid, rule, rng, temperature=temperature)
        states.append(grid.copy())
    return np.stack(states, axis=0)


def patch_tokenize(frames: np.ndarray, patch: int = 2, num_states: int = 10) -> np.ndarray:
    timesteps, height, width = frames.shape
    n_h = height // patch
    n_w = width // patch
    patches = frames.reshape(timesteps, n_h, patch, n_w, patch).transpose(0, 1, 3, 2, 4)
    patches = patches.reshape(timesteps, n_h * n_w, patch * patch)
    powers = (num_states ** np.arange(patch * patch)).astype(np.int32)
    return np.einsum("tnp,p->tn", patches.astype(np.int32), powers).astype(np.int32)


def gzip_complexity_percent(token_ids: np.ndarray) -> float:
    token_ids = np.asarray(token_ids, dtype=np.uint16)
    raw = token_ids.tobytes()
    if not raw:
        return 0.0
    return 100.0 * len(gzip.compress(raw)) / len(raw)


def flatten_with_delimiters(frames_tokens: np.ndarray, frame_delim: int, scene_delim: int) -> np.ndarray:
    pieces: List[np.ndarray] = []
    for frame in frames_tokens:
        pieces.append(frame.astype(np.uint16))
        pieces.append(np.asarray([frame_delim], dtype=np.uint16))
    pieces.append(np.asarray([scene_delim], dtype=np.uint16))
    return np.concatenate(pieces)
