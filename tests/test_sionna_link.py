"""
tests/test_sionna_link.py
-------------------------
Smoke and contract tests for LinkConfig.

Validates that the shared link object builds cleanly and its helper
methods return correctly-shaped tensors.
"""

from __future__ import annotations

import pytest
import torch

from neural_rx.sionna_link import LinkConfig


@pytest.fixture(scope="module")
def cfg() -> LinkConfig:
    torch.manual_seed(99)
    return LinkConfig(seed=99)


def test_build_does_not_raise() -> None:
    """LinkConfig must build without errors."""
    LinkConfig()


def test_sample_bits_shape(cfg: LinkConfig) -> None:
    """sample_bits() must return [batch, 1, 1, num_bits_per_frame]."""
    bits = cfg.sample_bits(16)
    assert bits.shape == (16, 1, 1, cfg.num_bits_per_frame)


def test_sample_bits_is_binary(cfg: LinkConfig) -> None:
    """Bits must be 0 or 1."""
    bits = cfg.sample_bits(32)
    assert set(bits.unique().tolist()).issubset({0.0, 1.0})


def test_transmit_output_shapes(cfg: LinkConfig) -> None:
    """transmit() must return y of shape [batch, 1, 1, T, K] (complex)."""
    bits = cfg.sample_bits(4)
    no = cfg.snr_db_to_no(5.0)
    y, h = cfg.transmit(bits, no)
    T, K = cfg.num_ofdm_symbols, cfg.fft_size
    assert y.shape == (4, 1, 1, T, K), f"y shape {y.shape}"
    assert y.is_complex()


def test_snr_db_to_no_positive(cfg: LinkConfig) -> None:
    """Noise power must be positive for any finite SNR."""
    for snr in [-20.0, 0.0, 10.0, 30.0]:
        no = cfg.snr_db_to_no(snr)
        assert no.item() > 0


def test_snr_db_to_no_decreasing(cfg: LinkConfig) -> None:
    """Higher SNR → lower noise power."""
    no_low  = cfg.snr_db_to_no(0.0).item()
    no_high = cfg.snr_db_to_no(20.0).item()
    assert no_high < no_low


def test_num_data_symbols_positive(cfg: LinkConfig) -> None:
    assert cfg.num_data_symbols > 0


def test_num_bits_per_frame(cfg: LinkConfig) -> None:
    assert cfg.num_bits_per_frame == cfg.num_data_symbols * cfg.bits_per_symbol


def test_pilot_mask_shape(cfg: LinkConfig) -> None:
    mask = cfg.pilot_mask()
    assert mask.shape == (cfg.num_ofdm_symbols, cfg.fft_size)
    assert mask.dtype == torch.bool


def test_pilot_mask_has_some_pilots(cfg: LinkConfig) -> None:
    """At least some resource elements must be pilots."""
    mask = cfg.pilot_mask()
    assert mask.sum() > 0, "Pilot mask is all-False — no pilots!"
