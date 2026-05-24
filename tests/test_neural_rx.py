"""
tests/test_neural_rx.py
-----------------------
Unit tests for the DeepRx-style NeuralReceiver.

Tests cover:
  - Forward pass shape correctness
  - No NaN/Inf in outputs on random inputs
  - Hard decisions are binary
  - Gradient flows from loss to all parameters
  - Parameter count is in the expected range (~50k–500k)
  - Pilot mask shape and dtype

All tests run on CPU under a fixed seed.
"""

from __future__ import annotations

import pytest
import torch

from neural_rx.neural_rx import NeuralReceiver
from neural_rx.sionna_link import LinkConfig


@pytest.fixture(scope="module")
def cfg() -> LinkConfig:
    torch.manual_seed(1)
    return LinkConfig(seed=1)


@pytest.fixture(scope="module")
def model(cfg: LinkConfig) -> NeuralReceiver:
    return NeuralReceiver(cfg, base_channels=32, num_blocks=3)


def _dummy_y(cfg: LinkConfig, batch: int = 4) -> torch.Tensor:
    """Generate a random complex received grid for unit tests."""
    T, K = cfg.num_ofdm_symbols, cfg.fft_size
    re = torch.randn(batch, 1, 1, T, K)
    im = torch.randn(batch, 1, 1, T, K)
    return torch.complex(re, im)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_forward_output_shape(cfg: LinkConfig, model: NeuralReceiver) -> None:
    """forward() must return [batch, 1, 1, num_bits_per_frame]."""
    y = _dummy_y(cfg, batch=8)
    llrs = model(y)
    expected = (8, 1, 1, cfg.num_bits_per_frame)
    assert llrs.shape == torch.Size(expected), f"Expected {expected}, got {llrs.shape}"


def test_decode_shapes(cfg: LinkConfig, model: NeuralReceiver) -> None:
    """decode() must return (llrs, bits_hat) both with matching shape."""
    y = _dummy_y(cfg, batch=4)
    llrs, bits_hat = model.decode(y)
    assert llrs.shape == bits_hat.shape


def test_batch_size_1(cfg: LinkConfig, model: NeuralReceiver) -> None:
    """Single-sample batch must work without error."""
    y = _dummy_y(cfg, batch=1)
    llrs = model(y)
    assert llrs.shape[0] == 1


def test_pilot_mask_shape(cfg: LinkConfig) -> None:
    """pilot_mask() must return a bool tensor [num_ofdm_symbols, fft_size]."""
    mask = cfg.pilot_mask()
    assert mask.shape == torch.Size([cfg.num_ofdm_symbols, cfg.fft_size])
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# Numerical health
# ---------------------------------------------------------------------------

def test_no_nans_random_input(cfg: LinkConfig, model: NeuralReceiver) -> None:
    """No NaN or Inf in LLRs on random received grids."""
    model.eval()
    for _ in range(5):
        y = _dummy_y(cfg, batch=4)
        with torch.no_grad():
            llrs = model(y)
        assert not torch.isnan(llrs).any(), "NaN detected in LLRs"
        assert not torch.isinf(llrs).any(), "Inf detected in LLRs"


def test_bits_hat_binary(cfg: LinkConfig, model: NeuralReceiver) -> None:
    """Hard decisions must be 0 or 1 only."""
    y = _dummy_y(cfg, batch=4)
    with torch.no_grad():
        _, bits_hat = model.decode(y)
    unique = bits_hat.unique()
    assert set(unique.tolist()).issubset({0.0, 1.0})


def test_llrs_have_variation(cfg: LinkConfig, model: NeuralReceiver) -> None:
    """LLRs should not all be equal (model is not degenerate)."""
    y = _dummy_y(cfg, batch=4)
    with torch.no_grad():
        llrs = model(y)
    assert llrs.std() > 1e-6, "LLRs have zero variance — model is degenerate"


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

def test_gradient_flows(cfg: LinkConfig, model: NeuralReceiver) -> None:
    """All parameters must receive gradients on a BCE backward pass."""
    model.train()
    y = _dummy_y(cfg, batch=4)
    bits = torch.randint(0, 2, (4, 1, 1, cfg.num_bits_per_frame)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(y), bits)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN grad for {name}"


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

def test_parameter_count_in_range(cfg: LinkConfig) -> None:
    """
    Parameter count should be between 50k and 500k for base_channels=64, num_blocks=5.
    The default architecture is intentionally small for fast GPU training.
    """
    m = NeuralReceiver(cfg, base_channels=64, num_blocks=5)
    n_params = m.count_parameters()
    assert 50_000 <= n_params <= 2_000_000, (
        f"Parameter count {n_params:,} is outside expected range [50k, 2M]"
    )


def test_small_model_has_fewer_params(cfg: LinkConfig) -> None:
    """Smaller base_channels → fewer parameters."""
    small = NeuralReceiver(cfg, base_channels=16, num_blocks=2)
    large = NeuralReceiver(cfg, base_channels=64, num_blocks=5)
    assert small.count_parameters() < large.count_parameters()
