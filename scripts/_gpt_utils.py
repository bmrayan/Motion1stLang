import math
from pathlib import Path
from typing import Iterable, Optional

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from _motion_utils import cosine_lr


def create_gpt2_small(vocab_size: int, n_positions: int = 1024) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    return GPT2LMHeadModel(config)


def get_autocast_dtype(mixed_precision: str):
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return None


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr(step=step, total_steps=total_steps, warmup_steps=warmup_steps),
    )


def cycle_dataloader(dataloader: Iterable):
    while True:
        for batch in dataloader:
            yield batch


@torch.no_grad()
def evaluate_loss(
    model: GPT2LMHeadModel,
    dataloader: Iterable,
    device: torch.device,
    mixed_precision: str = "bf16",
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    autocast_dtype = get_autocast_dtype(mixed_precision)
    losses = []
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        input_ids, labels = batch
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            out = model(input_ids=input_ids, labels=labels)
        losses.append(float(out.loss.item()))
    model.train()
    return float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def sample_sequences(
    model: GPT2LMHeadModel,
    vocab_size: int,
    device: torch.device,
    num_sequences: int = 10,
    prompt_len: int = 8,
    max_new_tokens: int = 64,
) -> list:
    model.eval()
    outputs = []
    for _ in range(num_sequences):
        prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)
        generated = model.generate(
            input_ids=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_k=min(vocab_size, 64),
            pad_token_id=0,
        )
        outputs.append(generated[0].tolist())
    model.train()
    return outputs
