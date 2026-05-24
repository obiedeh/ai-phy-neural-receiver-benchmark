"""
neural_rx/classical_rx.py
-------------------------
Classical receiver chain: LS channel estimation → LMMSE equalization
→ max-log soft demapping → hard-decision BLER.

This is the curve to beat.  All computation is in PyTorch on whatever
device the LinkConfig was built for.
"""

from __future__ import annotations

import torch

from neural_rx.sionna_link import LinkConfig


class ClassicalReceiver:
    """
    LS + LMMSE + max-log soft demap receiver.

    Wraps the Sionna blocks held in ``LinkConfig``.  All calls are
    differentiable and GPU-accelerated when the config uses CUDA.

    Usage::

        cfg = LinkConfig()
        rx  = ClassicalReceiver(cfg)
        llrs, bits_hat = rx.decode(y, no)
        bler = rx.bler(bits, bits_hat)
    """

    def __init__(self, cfg: LinkConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode(
        self, y: torch.Tensor, no: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the full classical receive chain.

        Args:
            y:   Received resource grid [batch, 1, 1, num_ofdm_symbols, fft_size]
            no:  Noise power — scalar or broadcastable tensor

        Returns:
            llrs:     Soft LLRs [batch, 1, 1, num_bits_per_frame]
            bits_hat: Hard decisions {0,1} [batch, 1, 1, num_bits_per_frame]
        """
        cfg = self.cfg
        h_hat, err_var = cfg.ls_estimator(y, no)
        x_hat, no_eff = cfg.lmmse_equalizer(y, h_hat, err_var, no)
        llrs = cfg.demapper(x_hat, no_eff)
        # Sionna Demapper convention: positive LLR → bit=1
        bits_hat = (llrs > 0).float()
        return llrs, bits_hat

    @staticmethod
    def bler(bits_tx: torch.Tensor, bits_hat: torch.Tensor) -> float:
        """
        Block error rate: fraction of frames where ANY bit is wrong.

        Args:
            bits_tx:  Transmitted bits [batch, 1, 1, num_bits_per_frame]
            bits_hat: Decoded bits     [batch, 1, 1, num_bits_per_frame]

        Returns:
            BLER in [0, 1]
        """
        block_errors = (bits_hat != bits_tx).any(dim=-1).float()
        return block_errors.mean().item()

    @staticmethod
    def ber(bits_tx: torch.Tensor, bits_hat: torch.Tensor) -> float:
        """Bit error rate."""
        return (bits_hat != bits_tx).float().mean().item()

    # ------------------------------------------------------------------
    # BLER sweep helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def bler_at_snr(
        self,
        snr_db: float,
        num_batches: int = 50,
        batch_size: int = 64,
    ) -> tuple[float, float]:
        """
        Estimate BLER and BER at a single SNR point.

        Args:
            snr_db:     SNR in dB
            num_batches: Number of batches to average over
            batch_size:  Frames per batch

        Returns:
            (bler, ber) — averaged over num_batches × batch_size frames
        """
        cfg = self.cfg
        no = cfg.snr_db_to_no(snr_db)

        total_block_errors = 0
        total_bit_errors = 0
        total_frames = 0
        total_bits = 0

        for _ in range(num_batches):
            bits = cfg.sample_bits(batch_size)
            y, _ = cfg.transmit(bits, no)
            _, bits_hat = self.decode(y, no)

            total_block_errors += (bits_hat != bits).any(dim=-1).sum().item()
            total_bit_errors += (bits_hat != bits).sum().item()
            total_frames += batch_size
            total_bits += bits.numel()

        bler = total_block_errors / total_frames
        ber = total_bit_errors / total_bits
        return bler, ber
