"""Run FoMEMO training on synthetic multi-objective data."""

import warnings
from linear_operator.utils.warnings import NumericalWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NumericalWarning)

import argparse
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR

from fomemo.checkpoint import save_model
from fomemo.config import CALIBRATION_CONFIG, DATA_CONFIG, DEMO_CONFIG, TRAINING_CONFIG
from fomemo.prior import sample_training_batch
from fomemo.training_utils import attach_ood_criteria, build_model, generate_ys, isolated_random_state, make_criteria, validate_config


def train(model, args, device, rank):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    def schedule(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch) / float(max(1, args.warmup_epochs))
        progress = float(epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, schedule)
    inner_model = model.module if isinstance(model, DistributedDataParallel) else model
    losses_by_epoch, training_times = [], []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start, total_loss = time.time(), 0.0
        for _ in range(args.steps_per_epoch):
            before_batch = time.time()
            batch = sample_training_batch(args.batch_size, args.seq_len, device, epoch)
            data_time = time.time() - before_batch
            before_forward = time.time()
            output = model((batch.style, batch.x, batch.y), single_eval_pos=batch.single_eval_pos, only_return_standard_out=False)["standard"]
            forward_time = time.time() - before_forward
            targets = batch.target_y[batch.single_eval_pos:]
            losses = inner_model.criterion[batch.y.size(-1)](output, targets).view(-1, output.shape[1])
            sequence_losses = losses.mean(0)
            valid = ~torch.isnan(sequence_losses)
            loss = torch.where(valid, sequence_losses, torch.zeros_like(sequence_losses)).sum(0) / valid.sum(0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step_time = time.time() - before_forward
            if not torch.isnan(loss):
                total_loss += loss.detach().cpu().item()
        losses_by_epoch.append(total_loss / args.steps_per_epoch)
        training_times.append([data_time, step_time, forward_time])
        if rank == 0:
            print(f"  Epoch {epoch}/{args.epochs} | loss={losses_by_epoch[-1]:.6f} | lr={scheduler.get_last_lr()[0]:.2e} | time={time.time() - epoch_start:.2f}s", flush=True)
        scheduler.step()
        if rank == 0 and (epoch % TRAINING_CONFIG["checkpoint_interval"] == 0 or epoch == args.epochs):
            save_model(inner_model, args.output_dir / "checkpoint" / f"fomemo_epoch_{epoch}.pt")
    if rank == 0:
        torch.save(losses_by_epoch, args.output_dir / "loss_record.pt")
        np.save(args.output_dir / "training_times.npy", np.array(training_times))
    return inner_model


def main(args):
    if args.demo:
        for name in ("epochs", "steps_per_epoch", "batch_size", "seq_len", "warmup_epochs"):
            setattr(args, name, DEMO_CONFIG[name])
    if min(args.epochs, args.steps_per_epoch, args.batch_size) < 1 or args.seq_len < 2 or args.lr <= 0 or args.warmup_epochs < 0:
        raise ValueError("Training budgets and learning rate must be positive; sequence length must be at least 2 and warmup nonnegative.")
    validate_config(args)
    objectives = range(DATA_CONFIG["min_objectives"], DATA_CONFIG["max_objectives"] + 1)
    rank, local_rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        device = torch.device("cuda", local_rank if world_size > 1 else (device.index or 0))
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("Training supports CPU or CUDA devices.")
    if world_size > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=24))
    try:
        args.output_dir = Path(args.output_dir)
        criteria = None
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            print("-------------------FoMEMO training-------------------", flush=True)
            print(f"device: {device}, processes: {world_size}", flush=True)
            print(f"epochs: {args.epochs}, steps per epoch: {args.steps_per_epoch}", flush=True)
            print(f"batch size per process: {args.batch_size}, sequence length: {args.seq_len}", flush=True)
            print(f"learning rate: {args.lr:.6g}, warmup epochs: {args.warmup_epochs}", flush=True)
            print(f"output directory: {args.output_dir}, demo: {args.demo}", flush=True)
            print("\n[1/3] Training calibration", flush=True)
            with isolated_random_state(device):
                generate_ys(args.output_dir / "ys_train", objectives, device, args.demo)
            criteria = make_criteria(args.output_dir / "ys_train", objectives)
        if world_size > 1:
            shared = [criteria]
            dist.broadcast_object_list(shared, src=0, device=device)
            criteria = shared[0]
        # The reference initialization consumes a full-dimension test batch first.
        sample_training_batch(args.batch_size, args.seq_len, device, epoch=0)
        model = build_model(criteria).to(device)
        if world_size > 1:
            model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None, broadcast_buffers=False, find_unused_parameters=False)
        if rank == 0:
            print("\n[2/3] Model training", flush=True)
        trained_model = train(model, args, device, rank)
        if world_size > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if rank == 0:
        print("\n[3/3] OOD calibration", flush=True)
        with isolated_random_state(device):
            generate_ys(args.output_dir / "ys_ood", CALIBRATION_CONFIG["ood_objectives"], device, args.demo)
        attach_ood_criteria(trained_model, args.output_dir / "ys_ood")
        destination = args.output_dir / "fomemo_model.pt"
        save_model(trained_model, destination)
        print(f"\nFinal model saved: {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda device")
    parser.add_argument("--output-dir", type=Path, default=Path("training_results"), help="Training output directory")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["epochs"])
    parser.add_argument("--steps-per-epoch", type=int, default=TRAINING_CONFIG["steps_per_epoch"])
    parser.add_argument("--batch-size", type=int, default=TRAINING_CONFIG["batch_size"], help="Batch size per process")
    parser.add_argument("--seq-len", type=int, default=DATA_CONFIG["max_sample_length"])
    parser.add_argument("--lr", type=float, default=TRAINING_CONFIG["learning_rate"])
    parser.add_argument("--warmup-epochs", type=int, default=TRAINING_CONFIG["warmup_epochs"])
    parser.add_argument("--demo", action="store_true", help="Use a minimal training and calibration budget")
    main(parser.parse_args())

# python train.py --demo

# python train.py --demo --device cpu

# CUDA_VISIBLE_DEVICES=0 python train.py

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py
