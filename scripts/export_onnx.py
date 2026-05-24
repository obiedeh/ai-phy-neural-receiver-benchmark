"""
scripts/export_onnx.py
----------------------
Export the trained neural receiver to ONNX and verify output parity.

Exports the pure-PyTorch NeuralReceiver nn.Module (no Sionna wrapper),
validates floating-point parity between PyTorch and ONNX Runtime, and
writes an evidence JSON.

Outputs:
  models/neural_rx.onnx              — ONNX model (opset 17)
  reports/onnx_parity_test.json      — parity test results

Prerequisite:
  models/neural_rx_best.pt must exist.

Usage:
  python scripts/export_onnx.py [--checkpoint models/neural_rx_best.pt]
                                [--opset 17]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neural_rx.neural_rx import NeuralReceiver
from neural_rx.sionna_link import LinkConfig

MODELS = Path(__file__).parent.parent / "models"
REPORTS = Path(__file__).parent.parent / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export NeuralReceiver to ONNX")
    p.add_argument("--checkpoint", default=str(MODELS / "neural_rx_best.pt"))
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        raise SystemExit(1)

    print("=" * 60)
    print("NeuralReceiver → ONNX Export")
    print("=" * 60)

    # ---- load model ----
    device = torch.device("cpu")   # export from CPU for portability
    cfg = LinkConfig(seed=args.seed)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_cfg = ckpt.get("config", {})
    model = NeuralReceiver(
        cfg,
        base_channels=model_cfg.get("base_channels", 64),
        num_blocks=model_cfg.get("num_blocks", 5),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # ---- build dummy input ----
    # y shape: [batch, 1, 1, num_ofdm_symbols, fft_size] complex64
    B = 4
    T = cfg.num_ofdm_symbols
    K = cfg.fft_size
    dummy_y_real = torch.randn(B, 1, 1, T, K, dtype=torch.float32)
    dummy_y_imag = torch.randn(B, 1, 1, T, K, dtype=torch.float32)
    dummy_y = torch.complex(dummy_y_real, dummy_y_imag)

    # ---- run PyTorch inference for reference ----
    with torch.no_grad():
        pt_llrs = model(dummy_y).numpy()

    # ---- ONNX export ----
    # The model expects complex input.  ONNX doesn't natively support complex64,
    # so we split into two real float32 channels [B, 1, 1, T, K, 2] for export.
    class _ExportWrapper(torch.nn.Module):
        """Wraps NeuralReceiver, accepting (real, imag) float tensors."""
        def __init__(self, inner: NeuralReceiver) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, y_real: torch.Tensor, y_imag: torch.Tensor) -> torch.Tensor:
            y_c = torch.complex(y_real, y_imag)
            return self.inner(y_c)

    wrapper = _ExportWrapper(model)
    wrapper.eval()

    MODELS.mkdir(exist_ok=True)
    onnx_path = MODELS / "neural_rx.onnx"

    torch.onnx.export(
        wrapper,
        (dummy_y_real, dummy_y_imag),
        str(onnx_path),
        opset_version=args.opset,
        input_names=["y_real", "y_imag"],
        output_names=["llrs"],
        dynamic_axes={
            "y_real": {0: "batch"},
            "y_imag": {0: "batch"},
            "llrs":   {0: "batch"},
        },
        verbose=False,
    )
    print(f"Exported: {onnx_path}")

    # ---- parity test with ONNX Runtime ----
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path))
        ort_inputs = {
            "y_real": dummy_y_real.numpy(),
            "y_imag": dummy_y_imag.numpy(),
        }
        ort_llrs = sess.run(["llrs"], ort_inputs)[0]

        max_abs_diff = float(np.abs(pt_llrs - ort_llrs).max())
        mean_abs_diff = float(np.abs(pt_llrs - ort_llrs).mean())
        parity_pass = max_abs_diff < 1e-4

        result = {
            "onnx_path": str(onnx_path),
            "opset": args.opset,
            "input_shape": list(dummy_y_real.shape),
            "output_shape": list(pt_llrs.shape),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "parity_pass": parity_pass,
            "tolerance": 1e-4,
        }

        tag = "✅ PASS" if parity_pass else "❌ FAIL"
        print(f"Parity test {tag}  max_diff={max_abs_diff:.2e}  mean_diff={mean_abs_diff:.2e}")

    except ImportError:
        result = {
            "onnx_path": str(onnx_path),
            "opset": args.opset,
            "parity_pass": None,
            "note": "onnxruntime not installed — parity not verified",
        }
        print("Note: onnxruntime not installed — skipping parity check")
        print("Install with: pip install onnxruntime")

    REPORTS.mkdir(exist_ok=True)
    parity_path = REPORTS / "onnx_parity_test.json"
    with open(parity_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {parity_path}")


if __name__ == "__main__":
    main()
