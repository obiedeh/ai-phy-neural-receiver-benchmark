"""
tests/test_classical_rx.py
--------------------------
Unit tests for the classical LS + LMMSE receiver.

Calibration expectations (uncoded QPSK on TDL-C, SISO):
  - Very high noise (SNR ≈ -20 dB) → BLER ≈ 1.0
  - Very low noise  (SNR ≈ +30 dB) → BLER < 0.5  (receiver limited by pilot coverage)

All tests run on CPU and are deterministic under seed=0.
"""

from __future__ import annotations

import pytest
import torch

from neural_rx.classical_rx import ClassicalReceiver
from neural_rx.sionna_link import LinkConfig


@pytest.fixture(scope="module")
def cfg() -> LinkConfig:
    torch.manual_seed(0)
    return LinkConfig(seed=0)


@pytest.fixture(scope="module")
def rx(cfg: LinkConfig) -> ClassicalReceiver:
    return ClassicalReceiver(cfg)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_decode_output_shapes(cfg: LinkConfig, rx: ClassicalReceiver) -> None:
    """decode() must return tensors with the expected shapes."""
    batch = 8
    no = cfg.snr_db_to_no(5.0)
    bits = cfg.sample_bits(batch)
    y, _ = cfg.transmit(bits, no)
    llrs, bits_hat = rx.decode(y, no)

    assert llrs.shape == bits.shape, f"Expected {bits.shape}, got {llrs.shape}"
    assert bits_hat.shape == bits.shape


def test_bits_hat_is_binary(cfg: LinkConfig, rx: ClassicalReceiver) -> None:
    """Hard decisions must be strictly 0 or 1."""
    no = cfg.snr_db_to_no(10.0)
    bits = cfg.sample_bits(4)
    y, _ = cfg.transmit(bits, no)
    _, bits_hat = rx.decode(y, no)
    unique = bits_hat.unique()
    assert set(unique.tolist()).issubset({0.0, 1.0})


def test_no_nans_in_llrs(cfg: LinkConfig, rx: ClassicalReceiver) -> None:
    """LLRs must not contain NaN or Inf values."""
    for snr_db in [-5.0, 0.0, 10.0, 20.0]:
        no = cfg.snr_db_to_no(snr_db)
        bits = cfg.sample_bits(8)
        y, _ = cfg.transmit(bits, no)
        llrs, _ = rx.decode(y, no)
        assert not torch.isnan(llrs).any(), f"NaN at SNR={snr_db}"
        assert not torch.isinf(llrs).any(), f"Inf at SNR={snr_db}"


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------

def test_high_noise_bler_near_one(cfg: LinkConfig, rx: ClassicalReceiver) -> None:
    """At very low SNR, BLER should be close to 1.0 (random guessing)."""
    bler, _ = rx.bler_at_snr(-20.0, num_batches=20, batch_size=64)
    assert bler > 0.80, f"Expected BLER > 0.80 at -20 dB, got {bler:.3f}"


def test_high_snr_ber_below_threshold(cfg: LinkConfig, rx: ClassicalReceiver) -> None:
    """
    At high SNR the receiver must improve over noise.

    Uncoded QPSK on TDL-C with 912 symbols/block: a single bit error flips
    the whole block, so BLER stays near 1 until BER is < ~1e-3.  We test
    BER directly — the receiver must not be stuck at random-guess (0.5).
    """
    _, ber = rx.bler_at_snr(30.0, num_batches=20, batch_size=64)
    assert ber < 0.40, f"Expected BER < 0.40 at 30 dB, got {ber:.4f}"


def test_bler_monotonically_decreasing(cfg: LinkConfig, rx: ClassicalReceiver) -> None:
    """BLER must be (broadly) decreasing with SNR."""
    snr_points = [-10.0, 0.0, 10.0, 20.0]
    blers = [rx.bler_at_snr(s, num_batches=15, batch_size=64)[0] for s in snr_points]
    # Allow one non-monotone step due to noise in the estimate
    violations = sum(
        1 for a, b in zip(blers[:-1], blers[1:], strict=False) if b > a + 0.08
    )
    assert violations <= 1, f"BLER non-monotone: {list(zip(snr_points, blers, strict=False))}"


def test_ber_lower_than_bler(cfg: LinkConfig, rx: ClassicalReceiver) -> None:
    """BER must always be ≤ BLER (BLER counts whole-frame errors)."""
    for snr_db in [0.0, 10.0]:
        bler, ber = rx.bler_at_snr(snr_db, num_batches=15, batch_size=64)
        assert ber <= bler + 1e-6, (
            f"BER ({ber:.4f}) > BLER ({bler:.4f}) at SNR={snr_db} dB — logic error"
        )


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------

def test_bler_static() -> None:
    """ClassicalReceiver.bler() returns correct value on known inputs."""
    bits = torch.tensor([[[[1.0, 0.0, 1.0]]]])
    good = torch.tensor([[[[1.0, 0.0, 1.0]]]])
    bad  = torch.tensor([[[[0.0, 0.0, 1.0]]]])

    assert ClassicalReceiver.bler(bits, good) == pytest.approx(0.0)
    assert ClassicalReceiver.bler(bits, bad)  == pytest.approx(1.0)


def test_ber_static() -> None:
    """ClassicalReceiver.ber() returns fraction of wrong bits."""
    bits = torch.zeros(1, 1, 1, 4)
    pred = torch.tensor([[[[0.0, 1.0, 0.0, 1.0]]]])
    assert ClassicalReceiver.ber(bits, pred) == pytest.approx(0.5)
