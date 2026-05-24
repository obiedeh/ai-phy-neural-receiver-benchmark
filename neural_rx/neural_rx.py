"""
neural_rx/neural_rx.py
----------------------
DeepRx-style CNN neural receiver.

Architecture follows the DeepRx pattern (Honkala et al., 2021):
    - Input: received resource grid (real + imag) + pilot mask
      → shape [batch, 3, num_ofdm_symbols, fft_size]
    - 5 residual conv blocks (32→64→64→32 channels, 3×3 kernels)
    - Output projection → LLRs for all data symbols
      → shape [batch, 1, 1, num_bits_per_frame]

Reference: Honkala, Korpi, Huttunen (2021), "DeepRx: Fully Convolutional
Deep Learning Receiver", IEEE TWC. https://arxiv.org/abs/2005.01494
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_rx.sionna_link import LinkConfig

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Residual conv block: BN → ReLU → Conv → BN → ReLU → Conv + skip."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _DownProj(nn.Module):
    """1×1 conv to change channel depth without spatial mixing."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


# ---------------------------------------------------------------------------
# Neural receiver
# ---------------------------------------------------------------------------

class NeuralReceiver(nn.Module):
    """
    DeepRx-style fully-convolutional neural receiver.

    Parameters
    ----------
    cfg:
        LinkConfig holding resource-grid geometry and num_bits_per_frame.
    base_channels:
        Width of the convolutional trunk (default: 64).  Halved / doubled
        at the stem / head transitions.  Total params ≈ 50k–500k depending
        on this value and num_blocks.
    num_blocks:
        Number of residual blocks in the trunk (default: 5).
    """

    def __init__(
        self,
        cfg: LinkConfig,
        base_channels: int = 64,
        num_blocks: int = 5,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        bps = cfg.bits_per_symbol
        self._data_out_size = cfg.num_bits_per_frame  # flat LLR output

        # ---- stem: 3 input channels → base_channels // 2 ----
        stem_ch = base_channels // 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
        )

        # ---- up-proj: stem_ch → base_channels ----
        self.up_proj = _DownProj(stem_ch, base_channels)

        # ---- residual trunk ----
        self.trunk = nn.Sequential(*[_ResBlock(base_channels) for _ in range(num_blocks)])

        # ---- per-element LLR head: base_channels → bits_per_symbol (1×1 conv) ----
        # Much cheaper than a flat linear head: base_ch * bps * 1 * 1 params only.
        # Output shape [B, bps, T, K]; we then mask out data positions.
        self.llr_head = nn.Conv2d(base_channels, bps, kernel_size=1, bias=True)

        # Precompute the pilot mask as a CPU buffer.
        # register_buffer ensures it moves with the model when .to(device) is called.
        # shape: [1, 1, T, K]  — broadcast across batch / channel
        pilot_mask = cfg.pilot_mask().float().cpu().unsqueeze(0).unsqueeze(0)
        self.register_buffer("_pilot_mask", pilot_mask)

        # Data-symbol flat indices within the resource grid [num_data_symbols]
        # data = everything that is NOT a pilot (no guard carriers in our setup)
        data_mask = ~cfg.pilot_mask().cpu()            # [T, K] bool
        data_idx = data_mask.flatten().nonzero(as_tuple=False).squeeze(1)
        self.register_buffer("_data_idx", data_idx)    # [num_data_symbols]

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: Received resource grid [batch, 1, 1, num_ofdm_symbols, fft_size]
               complex tensor from OFDMChannel.

        Returns:
            llrs: [batch, 1, 1, num_bits_per_frame]  — raw (pre-sigmoid) LLRs.
        """
        # Move input to model device (buffers and parameters share the same device
        # after .to(device) is called on the model)
        dev = self._pilot_mask.device
        y = y.to(dev)

        # Squeeze SISO dimensions → [batch, num_ofdm_symbols, fft_size] complex
        y_sq = y.squeeze(1).squeeze(1)               # [B, T, K] complex

        # Build 3-channel real tensor: [B, 3, T, K]
        y_real = y_sq.real.unsqueeze(1)              # [B, 1, T, K]
        y_imag = y_sq.imag.unsqueeze(1)              # [B, 1, T, K]
        pilot = self._pilot_mask.expand(y_sq.shape[0], -1, -1, -1)  # [B, 1, T, K]
        x = torch.cat([y_real, y_imag, pilot], dim=1)  # [B, 3, T, K]

        # CNN forward
        x = self.stem(x)
        x = self.up_proj(x)
        x = self.trunk(x)

        # Per-element LLR prediction: [B, bps, T, K]
        llr_grid = self.llr_head(x)

        # Extract data symbol positions:
        # Flatten spatial: [B, bps, T*K], index data positions → [B, bps, num_data_sym]
        B, bps, T, K = llr_grid.shape
        llr_flat = llr_grid.reshape(B, bps, T * K)
        idx = self._data_idx.to(llr_flat.device)
        llr_data = llr_flat[:, :, idx]               # [B, bps, num_data_sym]

        # Interleave bits: [B, num_data_sym * bps] → [B, 1, 1, num_bits_per_frame]
        # permute to [B, num_data_sym, bps] then flatten last two dims
        llrs_flat = llr_data.permute(0, 2, 1).reshape(B, -1)
        return llrs_flat.unsqueeze(1).unsqueeze(1)

    def decode(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward + hard decision.

        Returns:
            llrs:     [batch, 1, 1, num_bits_per_frame]
            bits_hat: {0,1} float tensor, same shape
        """
        llrs = self.forward(y)
        # Convention: neural RX trained with BCE-with-logits → positive logit = bit 1
        bits_hat = (llrs > 0).float()
        return llrs, bits_hat

    @torch.no_grad()
    def bler_at_snr(
        self,
        snr_db: float,
        num_batches: int = 50,
        batch_size: int = 64,
    ) -> tuple[float, float]:
        """Estimate BLER and BER at a single SNR point (no_grad)."""
        cfg = self.cfg
        no = cfg.snr_db_to_no(snr_db)
        dev = next(self.parameters()).device

        total_block_errors = 0
        total_bit_errors = 0
        total_frames = 0
        total_bits = 0

        for _ in range(num_batches):
            bits = cfg.sample_bits(batch_size).to(dev)
            y, _ = cfg.transmit(bits, no.to(dev))
            _, bits_hat = self.decode(y)

            total_block_errors += (bits_hat != bits).any(dim=-1).sum().item()
            total_bit_errors += (bits_hat != bits).sum().item()
            total_frames += batch_size
            total_bits += bits.numel()

        return total_block_errors / total_frames, total_bit_errors / total_bits

    # ------------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
