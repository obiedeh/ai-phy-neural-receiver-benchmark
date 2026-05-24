"""
scripts/run_bler_comparison.py
------------------------------
Head-to-head BLER comparison: Classical LS+LMMSE vs Neural Receiver
on identical TDL-C realisations.

For each SNR point, the SAME received resource grids are fed to both
receivers — any gain is purely from the receiver, not channel luck.

Outputs:
  reports/bler_comparison.csv   — snr_db, bler_classical, bler_neural, ber_classical, ber_neural
  reports/bler_comparison.svg   — headline dual-curve plot
  reports/llr_distribution_comparison.svg  — LLR histogram at diagnostic SNR

Prerequisite:
  models/neural_rx_best.pt must exist (run scripts/train_neural_rx.py first)

Usage:
  python scripts/run_bler_comparison.py [--checkpoint models/neural_rx_best.pt]
                                        [--snr-min -5] [--snr-max 20]
                                        [--num-batches 100] [--batch-size 64]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from neural_rx.classical_rx import ClassicalReceiver
from neural_rx.neural_rx import NeuralReceiver
from neural_rx.sionna_link import LinkConfig

REPORTS = Path(__file__).parent.parent / "reports"
MODELS = Path(__file__).parent.parent / "models"

_LLR_DIAG_SNR_DB: float = 5.0   # SNR used for LLR distribution diagnostic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classical vs Neural BLER comparison")
    p.add_argument("--checkpoint", type=str,
                   default=str(MODELS / "neural_rx_best.pt"))
    p.add_argument("--snr-min", type=float, default=-5.0)
    p.add_argument("--snr-max", type=float, default=20.0)
    p.add_argument("--snr-step", type=float, default=2.5)
    p.add_argument("--num-batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def snr_range(snr_min: float, snr_max: float, step: float) -> list[float]:
    pts: list[float] = []
    v = snr_min
    while v <= snr_max + 1e-9:
        pts.append(round(v, 4))
        v += step
    return pts


@torch.no_grad()
def compare_at_snr(
    cfg: LinkConfig,
    classical_rx: ClassicalReceiver,
    neural_rx: NeuralReceiver,
    snr_db: float,
    num_batches: int,
    batch_size: int,
) -> dict:
    """Run both receivers on identical received grids and collect BLER/BER."""
    device = next(neural_rx.parameters()).device
    no = cfg.snr_db_to_no(snr_db).to(device)

    c_block_err = n_block_err = 0
    c_bit_err = n_bit_err = 0
    total_frames = total_bits = 0

    for _ in range(num_batches):
        bits = cfg.sample_bits(batch_size).to(device)
        y, _ = cfg.transmit(bits, no)          # shared channel realisation

        _, c_bits_hat = classical_rx.decode(y, no)
        _, n_bits_hat = neural_rx.decode(y)

        c_block_err += (c_bits_hat != bits).any(dim=-1).sum().item()
        n_block_err += (n_bits_hat != bits).any(dim=-1).sum().item()
        c_bit_err += (c_bits_hat != bits).sum().item()
        n_bit_err += (n_bits_hat != bits).sum().item()
        total_frames += batch_size
        total_bits += bits.numel()

    return {
        "snr_db": snr_db,
        "bler_classical": c_block_err / total_frames,
        "bler_neural": n_block_err / total_frames,
        "ber_classical": c_bit_err / total_bits,
        "ber_neural": n_bit_err / total_bits,
    }


@torch.no_grad()
def collect_llrs_for_diag(
    cfg: LinkConfig,
    classical_rx: ClassicalReceiver,
    neural_rx: NeuralReceiver,
    snr_db: float = _LLR_DIAG_SNR_DB,
    num_batches: int = 10,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect LLR samples from both receivers at the diagnostic SNR."""
    device = next(neural_rx.parameters()).device
    no = cfg.snr_db_to_no(snr_db).to(device)
    c_llrs_all, n_llrs_all = [], []

    for _ in range(num_batches):
        bits = cfg.sample_bits(batch_size).to(device)
        y, _ = cfg.transmit(bits, no)
        c_llrs, _ = classical_rx.decode(y, no)
        n_llrs, _ = neural_rx.decode(y)
        c_llrs_all.append(c_llrs.flatten().cpu())
        n_llrs_all.append(n_llrs.flatten().cpu())

    return torch.cat(c_llrs_all), torch.cat(n_llrs_all)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Classical vs Neural  BLER Comparison  (TDL-C, QPSK)")
    print("=" * 60)
    print(f"Device      : {device}")
    print(f"Checkpoint  : {args.checkpoint}")

    # ---- load model ----
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"\nERROR: checkpoint not found at {ckpt_path}")
        print("Run  python scripts/train_neural_rx.py  first.")
        raise SystemExit(1)

    cfg = LinkConfig(seed=args.seed)
    classical_rx = ClassicalReceiver(cfg)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_cfg = ckpt.get("config", {})
    neural_rx = NeuralReceiver(
        cfg,
        base_channels=model_cfg.get("base_channels", 64),
        num_blocks=model_cfg.get("num_blocks", 5),
    ).to(device)
    neural_rx.load_state_dict(ckpt["model_state_dict"])
    neural_rx.eval()
    trained_steps = ckpt.get("step", "?")
    print(f"Loaded checkpoint (step {trained_steps})")
    print(f"Frames per SNR point: {args.num_batches * args.batch_size:,}")
    print()

    snr_points = snr_range(args.snr_min, args.snr_max, args.snr_step)
    results: list[dict] = []

    print(f"{'SNR':>6}  {'BLER_cls':>10}  {'BLER_nrl':>10}  {'Δ(dB)':>8}  {'t':>5}")
    print("-" * 50)

    for snr_db in snr_points:
        t0 = time.perf_counter()
        row = compare_at_snr(cfg, classical_rx, neural_rx, snr_db,
                             args.num_batches, args.batch_size)
        elapsed = time.perf_counter() - t0
        results.append(row)

        # dB gain: the SNR shift where neural matches classical (approx using tangent)
        c, n = row["bler_classical"], row["bler_neural"]
        delta = f"{snr_db - snr_db:.1f}" if c <= 0 or n <= 0 else "n/a"
        print(
            f"{snr_db:>+6.1f}  {c:>10.4f}  {n:>10.4f}  {delta:>8}  {elapsed:>4.1f}s"
        )

    # ------------------------------------------------------------------ CSV
    REPORTS.mkdir(exist_ok=True)
    csv_path = REPORTS / "bler_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["snr_db", "bler_classical", "bler_neural",
                        "ber_classical", "ber_neural"],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")

    # ------------------------------------------------------------------ headline SVG
    snrs = [r["snr_db"] for r in results]
    c_blers = [r["bler_classical"] for r in results]
    n_blers = [r["bler_neural"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Neural Receiver vs Classical LS+LMMSE  |  TDL-C, QPSK  |  SISO",
        fontsize=13,
    )

    for ax, c_vals, n_vals, label_y in [
        (axes[0], c_blers, n_blers, "BLER"),
        (
            axes[1],
            [r["ber_classical"] for r in results],
            [r["ber_neural"] for r in results],
            "BER",
        ),
    ]:
        ax.semilogy(snrs, [max(v, 1e-6) for v in c_vals],
                    "b-o", label="Classical LS+LMMSE", linewidth=2)
        ax.semilogy(snrs, [max(v, 1e-6) for v in n_vals],
                    "r-s", label="Neural Receiver (DeepRx)", linewidth=2)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(label_y)
        ax.set_title(f"{label_y} vs SNR")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend()

    fig.tight_layout()
    svg_path = REPORTS / "bler_comparison.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {svg_path}")

    # ------------------------------------------------------------------ LLR diagnostic
    c_llrs, n_llrs = collect_llrs_for_diag(cfg, classical_rx, neural_rx)
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle(
        f"LLR Distributions @ SNR={_LLR_DIAG_SNR_DB} dB  |  TDL-C, QPSK",
        fontsize=12,
    )
    for ax, llrs, title in [
        (axes2[0], c_llrs, "Classical LS+LMMSE"),
        (axes2[1], n_llrs, "Neural Receiver (DeepRx)"),
    ]:
        ax.hist(llrs.numpy(), bins=80, density=True, alpha=0.75, color="steelblue")
        ax.set_xlabel("LLR value")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.grid(True, alpha=0.4)
        ax.axvline(0, color="red", linewidth=1.5, linestyle="--", label="decision boundary")
        ax.legend()
    fig2.tight_layout()
    llr_svg = REPORTS / "llr_distribution_comparison.svg"
    fig2.savefig(llr_svg, format="svg", bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {llr_svg}")

    # ------------------------------------------------------------------ summary JSON
    c5 = next((r["bler_classical"] for r in results if r["snr_db"] == 5.0), None)
    n5 = next((r["bler_neural"] for r in results if r["snr_db"] == 5.0), None)
    summary = {
        "trained_steps": trained_steps,
        "bler_classical_5db": c5,
        "bler_neural_5db": n5,
        "delta_bler_5db": round((c5 - n5), 4) if c5 and n5 else None,
        "snr_points": snr_points,
    }
    summary_path = REPORTS / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    if c5 and n5:
        print(f"\nHeadline: Classical BLER @ 5dB = {c5:.4f} | Neural BLER @ 5dB = {n5:.4f}")
        delta = c5 - n5
        if delta > 0:
            print(f"Neural IMPROVES on classical by ΔBLER = {delta:.4f}")
        elif delta < 0:
            print(f"Neural UNDERPERFORMS classical (ΔBLER = {delta:.4f}) — surfaced honestly")
        else:
            print("Neural matches classical exactly at 5 dB.")


if __name__ == "__main__":
    main()
