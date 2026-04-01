#!/usr/bin/env python
import argparse
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from _gpt_utils import build_scheduler, create_gpt2_small, cycle_dataloader, evaluate_loss, get_autocast_dtype
from _motion_utils import (
    CSVLogger,
    FlatTokenDataset,
    count_parameters,
    ensure_dir,
    finish_wandb,
    init_wandb,
    log_wandb,
    save_checkpoint,
    save_json,
    set_seed,
)


EARLY_SAVE_STEPS = {10, 20, 50, 100, 200, 500}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Language pre-train GPT-2 Small on OpenWebText.")
    parser.add_argument("--init_from", type=str, required=True, help="scratch or pre-pretrain checkpoint dir")
    parser.add_argument("--data_path", type=str, required=True, help="Directory with raw/ and train.bin/val.bin")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--total_tokens", type=int, required=True)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--effective_batch_size", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prepare_data", type=int, default=1)
    parser.add_argument("--transfer_wpe", type=int, default=1)
    parser.add_argument("--reinit_modules", type=str, nargs="*", default=[])
    return parser.parse_args()


def prepare_openwebtext(data_dir: Path, seq_len: int, seed: int) -> Dict[str, int]:
    raw_dir = data_dir / "raw"
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    if train_bin.exists() and val_bin.exists():
        train_tokens = int(np.memmap(train_bin, dtype=np.uint16, mode="r").shape[0])
        val_tokens = int(np.memmap(val_bin, dtype=np.uint16, mode="r").shape[0])
        return {"train_tokens": train_tokens, "val_tokens": val_tokens}

    dataset = load_from_disk(str(raw_dir))
    split = dataset.train_test_split(test_size=0.05, seed=seed, shuffle=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    eos = tokenizer.eos_token_id

    def write_split(ds, path: Path) -> int:
        total_tokens = 0
        with open(path, "wb") as handle:
            for row in ds:
                text = row["text"]
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                token_ids.append(eos)
                array = np.asarray(token_ids, dtype=np.uint16)
                array.tofile(handle)
                total_tokens += int(len(array))
        return total_tokens

    train_tokens = write_split(split["train"], train_bin)
    val_tokens = write_split(split["test"], val_bin)
    save_json(
        str(data_dir / "tokenization_summary.json"),
        {
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "split_ratio": {"train": 0.95, "val": 0.05},
            "seq_len": seq_len,
        },
    )
    return {"train_tokens": train_tokens, "val_tokens": val_tokens}


def init_text_model(args: argparse.Namespace, device: torch.device) -> GPT2LMHeadModel:
    model = create_gpt2_small(vocab_size=50_257, n_positions=args.seq_len)
    if args.init_from == "scratch":
        return model.to(device)

    ppt_model = GPT2LMHeadModel.from_pretrained(args.init_from)
    ppt_state = ppt_model.state_dict()
    text_state = model.state_dict()
    for key, value in ppt_state.items():
        if "transformer.wte" in key or "lm_head" in key:
            continue
        if not args.transfer_wpe and "transformer.wpe" in key:
            continue
        if key in text_state and text_state[key].shape == value.shape:
            text_state[key] = value
    model.load_state_dict(text_state)
    if args.reinit_modules:
        reinitialize_modules(model, args.reinit_modules)
    return model.to(device)


def reinitialize_modules(model: GPT2LMHeadModel, modules: List[str]) -> None:
    module_set = set(modules)
    for name, param in model.named_parameters():
        if "attn" in module_set and (".attn.c_attn." in name or ".attn.c_proj." in name):
            if name.endswith(".weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                torch.nn.init.zeros_(param)
        if "mlp" in module_set and (".mlp.c_fc." in name or ".mlp.c_proj." in name):
            if name.endswith(".weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                torch.nn.init.zeros_(param)
        if "ln" in module_set and (".ln_1." in name or ".ln_2." in name or "transformer.ln_f" in name):
            if name.endswith(".weight"):
                torch.nn.init.ones_(param)
            else:
                torch.nn.init.zeros_(param)


def save_hf_checkpoint(
    model: GPT2LMHeadModel,
    output_dir: Path,
    step: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    tokens_seen: int,
) -> None:
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ensure_dir(str(ckpt_dir))
    model.save_pretrained(ckpt_dir)
    save_checkpoint(
        str(ckpt_dir / "trainer_state.pt"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        extra_state={"step": step, "tokens_seen": tokens_seen},
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.effective_batch_size % args.micro_batch_size != 0:
        raise ValueError("effective_batch_size must be divisible by micro_batch_size")

    data_dir = Path(args.data_path)
    output_dir = Path(ensure_dir(args.output_dir))
    log_dir = output_dir / "logs"
    ensure_dir(str(log_dir))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.prepare_data:
        tokenization_summary = prepare_openwebtext(data_dir, args.seq_len, args.seed)
    else:
        tokenization_summary = {
            "train_tokens": int(np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r").shape[0]),
            "val_tokens": int(np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r").shape[0]),
        }

    grad_accum_steps = args.effective_batch_size // args.micro_batch_size
    tokens_per_step = args.effective_batch_size * args.seq_len
    total_steps = math.ceil(args.total_tokens / tokens_per_step)
    warmup_steps = int(total_steps * args.warmup_ratio)

    train_dataset = FlatTokenDataset(str(data_dir / "train.bin"), seq_len=args.seq_len)
    val_dataset = FlatTokenDataset(str(data_dir / "val.bin"), seq_len=args.seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    train_iter = cycle_dataloader(train_loader)

    model = init_text_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == "fp16")
    autocast_dtype = get_autocast_dtype(args.mixed_precision)

    config = vars(args).copy()
    config.update(
        {
            "parameter_count": count_parameters(model),
            "grad_accum_steps": grad_accum_steps,
            "tokens_per_step": tokens_per_step,
            "total_steps": total_steps,
            "tokenization_summary": tokenization_summary,
        }
    )
    save_json(str(output_dir / "hparams.json"), config)
    csv_logger = CSVLogger(
        str(log_dir / "train_metrics.csv"),
        ["step", "tokens_seen", "train_loss", "val_loss", "val_perplexity", "learning_rate", "wall_time_sec"],
    )
    wandb_run = init_wandb("motion-first-language", args.run_name, config)

    model.train()
    start_time = time.time()
    train_losses = []
    final_val_loss = None

    for step in range(1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum_steps):
            input_ids, labels = next(train_iter)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / grad_accum_steps
            step_loss += float(loss.item()) * grad_accum_steps
            if args.mixed_precision == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if args.grad_clip > 0:
            if args.mixed_precision == "fp16":
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if args.mixed_precision == "fp16":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        tokens_seen = min(args.total_tokens, step * tokens_per_step)
        train_losses.append(step_loss)
        lr = float(optimizer.param_groups[0]["lr"])

        val_loss = None
        val_perplexity = None
        if step % args.eval_every == 0 or step in EARLY_SAVE_STEPS or step == total_steps:
            val_loss = evaluate_loss(
                model,
                val_loader,
                device=device,
                mixed_precision=args.mixed_precision,
                max_batches=args.eval_batches,
            )
            final_val_loss = val_loss
            val_perplexity = float(math.exp(min(val_loss, 20.0)))

        row = {
            "step": step,
            "tokens_seen": tokens_seen,
            "train_loss": step_loss,
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
            "learning_rate": lr,
            "wall_time_sec": time.time() - start_time,
        }
        csv_logger.log(row)
        log_wandb(wandb_run, {k: v for k, v in row.items() if v is not None})

        if step % 10 == 0 or step == 1:
            suffix = f" val_ppl={val_perplexity:.4f}" if val_perplexity is not None else ""
            print(f"step={step}/{total_steps} tokens={tokens_seen} loss={step_loss:.6f}{suffix}")

        if step in EARLY_SAVE_STEPS or step % args.save_every == 0 or step == total_steps:
            save_hf_checkpoint(model, output_dir, step, optimizer, scheduler, scaler, tokens_seen)

    model.save_pretrained(output_dir)
    save_checkpoint(
        str(output_dir / "training_state.pt"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        extra_state={"step": total_steps, "tokens_seen": min(args.total_tokens, total_steps * tokens_per_step)},
    )
    summary = {
        "run_name": args.run_name,
        "init_from": args.init_from,
        "reinit_modules": args.reinit_modules,
        "initial_train_loss": float(train_losses[0]) if train_losses else None,
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        "final_val_loss": float(final_val_loss) if final_val_loss is not None else None,
        "final_val_perplexity": float(math.exp(min(final_val_loss, 20.0))) if final_val_loss is not None else None,
        "steps": total_steps,
        "tokens_seen": min(args.total_tokens, total_steps * tokens_per_step),
    }
    save_json(str(output_dir / "training_summary.json"), summary)
    finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
