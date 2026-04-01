#!/usr/bin/env python
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from _gpt_utils import build_scheduler, create_gpt2_small, cycle_dataloader, get_autocast_dtype, sample_sequences
from _motion_utils import (
    CSVLogger,
    FlatTokenDataset,
    compute_gzip_complexity,
    count_parameters,
    ensure_dir,
    finish_wandb,
    init_wandb,
    log_wandb,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-pre-train GPT-2 Small on tokenized physics or NCA data.")
    parser.add_argument("--data_path", type=str, required=True, help="Directory containing tokens.bin")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--total_tokens", type=int, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--effective_batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.effective_batch_size % args.micro_batch_size != 0:
        raise ValueError("effective_batch_size must be divisible by micro_batch_size")

    output_dir = Path(ensure_dir(args.output_dir))
    log_dir = output_dir / "logs"
    ensure_dir(str(log_dir))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    grad_accum_steps = args.effective_batch_size // args.micro_batch_size
    tokens_per_step = args.effective_batch_size * args.seq_len
    total_steps = math.ceil(args.total_tokens / tokens_per_step)
    warmup_steps = int(total_steps * args.warmup_ratio)

    dataset = FlatTokenDataset(str(Path(args.data_path) / "tokens.bin"), seq_len=args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    batch_iter = cycle_dataloader(dataloader)

    model = create_gpt2_small(vocab_size=args.vocab_size, n_positions=args.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == "fp16")
    autocast_dtype = get_autocast_dtype(args.mixed_precision)

    config = vars(args).copy()
    config.update(
        {
            "grad_accum_steps": grad_accum_steps,
            "tokens_per_step": tokens_per_step,
            "total_steps": total_steps,
            "parameter_count": count_parameters(model),
        }
    )
    save_json(str(output_dir / "hparams.json"), config)
    csv_logger = CSVLogger(
        str(log_dir / "train_metrics.csv"),
        ["step", "tokens_seen", "train_loss", "learning_rate", "wall_time_sec"],
    )
    wandb_run = init_wandb("motion-first-language", args.run_name, config)

    model.train()
    global_step = 0
    tokens_seen = 0
    train_losses = []
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()

    while global_step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        total_micro_loss = 0.0
        for _ in range(grad_accum_steps):
            input_ids, labels = next(batch_iter)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / grad_accum_steps
            total_micro_loss += float(loss.item()) * grad_accum_steps
            if args.mixed_precision == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if args.mixed_precision == "fp16":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        global_step += 1
        tokens_seen = min(args.total_tokens, global_step * tokens_per_step)
        train_losses.append(total_micro_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        end_time.record()
        torch.cuda.synchronize(device)
        wall_time_sec = float(start_time.elapsed_time(end_time) / 1000.0)

        row = {
            "step": global_step,
            "tokens_seen": tokens_seen,
            "train_loss": total_micro_loss,
            "learning_rate": lr,
            "wall_time_sec": wall_time_sec,
        }
        csv_logger.log(row)
        log_wandb(wandb_run, row)

        if global_step % 50 == 0 or global_step == 1:
            print(f"step={global_step}/{total_steps} tokens={tokens_seen} loss={total_micro_loss:.6f} lr={lr:.6e}")

        if global_step % args.save_every == 0 and global_step < total_steps:
            save_checkpoint(
                str(output_dir / f"checkpoint_step_{global_step}.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                extra_state={"step": global_step, "tokens_seen": tokens_seen},
            )

    model.save_pretrained(output_dir)
    samples = sample_sequences(model, args.vocab_size, device=device, num_sequences=10)
    sample_complexities = [compute_gzip_complexity(np.asarray(seq, dtype=np.uint16)) for seq in samples]
    summary = {
        "run_name": args.run_name,
        "initial_loss": float(train_losses[0]) if train_losses else None,
        "final_loss": float(train_losses[-1]) if train_losses else None,
        "tokens_seen": tokens_seen,
        "steps": global_step,
        "sample_sequences": samples,
        "sample_gzip_complexity_mean": float(np.mean(sample_complexities) if sample_complexities else 0.0),
        "sample_gzip_complexity_std": float(np.std(sample_complexities) if sample_complexities else 0.0),
    }
    save_json(str(output_dir / "training_summary.json"), summary)
    save_json(str(output_dir / "sample_sequences.json"), {"samples": samples})
    save_checkpoint(
        str(output_dir / "training_state.pt"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        extra_state={"step": global_step, "tokens_seen": tokens_seen},
    )
    finish_wandb(wandb_run)
    print(f"Finished pre-pre-training {args.run_name} with final loss {summary['final_loss']:.6f}")


if __name__ == "__main__":
    main()
