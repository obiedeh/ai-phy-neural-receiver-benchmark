"""
scripts/run_bler_classical.py
-----------------------------
Sweep SNR for the classical LS + LMMSE + soft-demap receiver on TDL-C.

Outputs:
  reports/bler_classical.csv   — SNR, BLER, BER columns
  reports/bler_classical.svg   — publication-quality BLER vs SNR plot

Usage:
  python scripts/run_bler_classical.py [--snr-min -5] [--snr-max 20]
                                       [--snr-step 2.5]
                                       [--num-batches 100] [--batch-size 64]
                                       [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from neural_rx.classical_rx import ClassicalReceiver
from neural_rx.sionna_link import LinkConfig

REPORTS = Path(__file__).parent.parent / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classical LS+LMMSE BLER sweep")
    p.add_argument("--snr-min", type=float, default=-5.0)
    p.add_argument("--snr-max", type=float, default=20.0)
    p.add_argument("--snr-step", type=float, default=2.5)
    p.add_argument("--num-batches", type=int, default=100,
                   help="Batches per SNR point (higher → tighter confidence)")
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Classical LS + LMMSE  BLER sweep  (TDL-C, QPSK)")
    print("=" * 60)

    cfg = LinkConfig(seed=args.seed)
    rx = ClassicalReceiver(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Frames per SNR point: {args.num_batches * args.batch_size:,}")
    print(f"SNR range: {args.snr_min} → {args.snr_max} dB  (step {args.snr_step} dB)")
    print()

    snr_points = snr_range(args.snr_min, args.snr_max, args.snr_step)
    results: list[dict] = []

    for snr_db in snr_points:
        t0 = time.perf_counter()
        bler, ber = rx.bler_at_snr(
            snr_db,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
        )
        elapsed = time.perf_counter() - t0
        results.append({"snr_db": snr_db, "bler": bler, "ber": ber})
        print(f"  SNR={snr_db:+6.1f} dB  BLER={bler:.4f}  BER={ber:.6f}  ({elapsed:.1f}s)")

    # ------------------------------------------------------------------ CSV
    REPORTS.mkdir(exist_ok=True)
    csv_path = REPORTS / "bler_classical.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["snr_db", "bler", "ber"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")

    # ------------------------------------------------------------------ SVG
    snrs = [r["snr_db"] for r in results]
    blers = [r["bler"] for r in results]
    bers = [r["ber"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Classical LS + LMMSE  |  TDL-C, QPSK  |  SISO", fontsize=13)

    for ax, vals, ylabel, label in [
        (axes[0], blers, "BLER", "LS+LMMSE BLER"),
        (axes[1], bers,  "BER",  "LS+LMMSE BER"),
    ]:
        ax.semilogy(snrs, [max(v, 1e-6) for v in vals], "b-o", label=label, linewidth=2)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs SNR")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend()

    fig.tight_layout()
    svg_path = REPORTS / "bler_classical.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {svg_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
