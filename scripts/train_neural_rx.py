"""
scripts/train_neural_rx.py
--------------------------
End-to-end training of the DeepRx-style neural receiver on TDL-C.

Each minibatch draws:
  - Fresh TDL-C channel realisations (new multi-path environment each step)
  - Random SNR sampled uniformly from [snr_min, snr_max] dB

Loss: binary cross-entropy on transmitted bits (BCE maximises MI on bit channel).
Optimizer: Adam with cosine LR schedule.

Outputs:
  models/neural_rx_best.pt     — best checkpoint by validation BLER
  models/neural_rx_final.pt    — final checkpoint after training
  reports/training_log.json    — loss + val BLER curve

Usage:
  python scripts/train_neural_rx.py [--steps 100000] [--batch-size 64]
                                    [--lr 1e-3] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from neural_rx.neural_rx import NeuralReceiver
from neural_rx.sionna_link import LinkConfig

MODELS = Path(__file__).parent.parent / "models"
REPORTS = Path(__file__).parent.parent / "reports"

# Fixed validation SNR for checkpoint selection
_VAL_SNR_DB: float = 5.0
_VAL_BATCHES: int = 20
_VAL_BATCH_SIZE: int = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DeepRx neural receiver")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--snr-min", type=float, default=-5.0)
    p.add_argument("--snr-max", type=float, default=20.0)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=5)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sample_snr_batch(snr_min: float, snr_max: float, device: torch.device) -> torch.Tensor:
    """Sample a random SNR in [snr_min, snr_max] dB → linear noise power."""
    snr_db = torch.empty(1).uniform_(snr_min, snr_max).item()
    snr_linear = 10.0 ** (snr_db / 10.0)
    return torch.tensor(1.0 / snr_linear, dtype=torch.float32, device=device)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("DeepRx-style Neural Receiver — Training")
    print("=" * 60)
    print(f"Device : {device}")
    print(f"Steps  : {args.steps:,}  |  batch_size: {args.batch_size}")
    print(f"LR     : {args.lr}  |  SNR range: [{args.snr_min}, {args.snr_max}] dB")

    cfg = LinkConfig(seed=args.seed)
    model = NeuralReceiver(cfg, base_channels=args.base_channels, num_blocks=args.num_blocks)
    model = model.to(device)
    print(f"Model  : {model.count_parameters():,} trainable parameters")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    loss_fn = nn.BCEWithLogitsLoss()

    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    log: dict = {
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "snr_min_db": args.snr_min,
            "snr_max_db": args.snr_max,
            "base_channels": args.base_channels,
            "num_blocks": args.num_blocks,
            "parameters": model.count_parameters(),
            "device": str(device),
        },
        "train_loss": [],
        "val_bler": [],
        "val_ber": [],
        "val_steps": [],
    }

    # For uncoded QPSK on TDL-C, BLER stays at 1.0 for large blocks even at moderate SNR.
    # Save checkpoints by val BER instead of BLER.
    best_val_bler = float("inf")
    best_val_ber = float("inf")
    running_loss = 0.0
    t0 = time.perf_counter()
    step = 0

    print(f"{'Step':>8}  {'Loss':>10}  {'LR':>10}  {'Elapsed':>8}")
    print("-" * 50)

    while step < args.steps:
        model.train()
        no = sample_snr_batch(args.snr_min, args.snr_max, device)
        bits = cfg.sample_bits(args.batch_size).to(device)
        y, _ = cfg.transmit(bits, no)

        llrs = model(y)                                    # [B, 1, 1, num_bits]
        loss = loss_fn(llrs, bits)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        step += 1

        # ---- logging ----
        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.perf_counter() - t0
            print(f"{step:>8,}  {avg_loss:>10.5f}  {lr_now:>10.2e}  {elapsed:>7.0f}s")
            log["train_loss"].append({"step": step, "loss": avg_loss})
            running_loss = 0.0

        # ---- validation ----
        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            val_bler, val_ber = model.bler_at_snr(
                _VAL_SNR_DB, num_batches=_VAL_BATCHES, batch_size=_VAL_BATCH_SIZE
            )
            tag = " ← best" if val_bler < best_val_bler else ""
            print(
                f"  [val @ {_VAL_SNR_DB}dB] BLER={val_bler:.4f}  BER={val_ber:.6f}{tag}"
            )
            log["val_bler"].append(val_bler)
            log["val_ber"].append(val_ber)
            log["val_steps"].append(step)

            if val_ber < best_val_ber:
                best_val_ber = val_ber
                best_val_bler = val_bler
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_bler": val_bler,
                        "val_ber": val_ber,
                        "config": log["config"],
                    },
                    MODELS / "neural_rx_best.pt",
                )

    # ---- save final checkpoint ----
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_bler": log["val_bler"][-1] if log["val_bler"] else None,
            "config": log["config"],
        },
        MODELS / "neural_rx_final.pt",
    )

    log["best_val_bler"] = best_val_bler
    log_path = REPORTS / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    total_time = time.perf_counter() - t0
    print()
    print(f"Training complete in {total_time/60:.1f} min")
    print(f"Best val BLER @ {_VAL_SNR_DB} dB: {best_val_bler:.4f}")
    print(f"Checkpoints: {MODELS}/neural_rx_best.pt  |  neural_rx_final.pt")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
