"""
neural_rx/sionna_link.py
------------------------
Shared link configuration for the 5G NR SISO TDL-C OFDM link.

Creates and owns all Sionna objects so both the classical and neural
receivers share identical channel, resource grid, and mapping setup.
Any change to link parameters flows through a single LinkConfig instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sionna.phy as _sionna_phy
import torch
from sionna.phy.channel import OFDMChannel
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.mapping import BinarySource, Demapper, Mapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    LMMSEEqualizer,
    LSChannelEstimator,
    ResourceGrid,
    ResourceGridMapper,
)

# ---------------------------------------------------------------------------
# Default link parameters (locked per dated handoff - do not re-debate)
# ---------------------------------------------------------------------------
_CARRIER_FREQ_HZ: float = 3.5e9        # 3.5 GHz sub-6 NR band
_DELAY_SPREAD_S: float = 100e-9        # TDL-C nominal delay spread
_SUBCARRIER_SPACING_HZ: float = 30e3   # 30 kHz SCS (NR numerology μ=1)
_NUM_OFDM_SYMBOLS: int = 14            # one slot (normal CP, μ=1)
_FFT_SIZE: int = 76                    # ~2.3 MHz useful BW (demo scale)
_CP_LENGTH: int = 6                    # cyclic prefix length in samples
_PILOT_SYMBOL_INDICES: tuple = (2, 11) # DMRS positions in slot
_BITS_PER_SYMBOL: int = 2              # QPSK


@dataclass
class LinkConfig:
    """
    Container for all Sionna link-layer objects needed by both receivers.

    Instantiate once and pass to ClassicalReceiver / NeuralReceiver.
    All fields are read-only after construction — use a new LinkConfig
    to change parameters.
    """

    carrier_frequency: float = _CARRIER_FREQ_HZ
    delay_spread: float = _DELAY_SPREAD_S
    subcarrier_spacing: float = _SUBCARRIER_SPACING_HZ
    num_ofdm_symbols: int = _NUM_OFDM_SYMBOLS
    fft_size: int = _FFT_SIZE
    cp_length: int = _CP_LENGTH
    pilot_ofdm_symbol_indices: tuple = _PILOT_SYMBOL_INDICES
    bits_per_symbol: int = _BITS_PER_SYMBOL
    normalize_channel: bool = True
    seed: int = 42
    device: str | None = None

    # Populated by __post_init__ — do not pass in
    resource_grid: ResourceGrid = field(init=False, repr=False)
    stream_management: StreamManagement = field(init=False, repr=False)
    channel_model: TDL = field(init=False, repr=False)
    ofdm_channel: OFDMChannel = field(init=False, repr=False)
    mapper: Mapper = field(init=False, repr=False)
    rg_mapper: ResourceGridMapper = field(init=False, repr=False)
    ls_estimator: LSChannelEstimator = field(init=False, repr=False)
    lmmse_equalizer: LMMSEEqualizer = field(init=False, repr=False)
    demapper: Demapper = field(init=False, repr=False)
    binary_source: BinarySource = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Sionna 2.0 requires "cuda:0" (not bare "cuda")
        dev = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        # SISO: 1 TX, 1 RX, 1 stream
        rx_tx_association = np.array([[1]], dtype=np.int32)
        self.stream_management = StreamManagement(rx_tx_association, num_streams_per_tx=1)

        # KroneckerPilotPattern generates random pilot VALUES via Sionna's internal
        # torch.Generator (sionna.phy.config._torch_rngs).  Two LinkConfig instances
        # created with the same seed but at different points in the program can get
        # different pilot sequences — breaking inference on a model trained with a
        # different LinkConfig instance.
        #
        # Fix: temporarily set sionna.phy.config.seed to a deterministic value so
        # that KroneckerPilotPattern always produces the SAME pilots for the SAME
        # LinkConfig.seed, regardless of when it is constructed.
        # We restore config.seed to None immediately after so that subsequent Sionna
        # calls (BinarySource, TDL channel, …) remain stochastic.
        _prev_sionna_seed = _sionna_phy.config.seed
        _sionna_phy.config.seed = self.seed   # deterministic pilots

        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=self.num_ofdm_symbols,
            fft_size=self.fft_size,
            subcarrier_spacing=self.subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=1,
            cyclic_prefix_length=self.cp_length,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=list(self.pilot_ofdm_symbol_indices),
            device=dev,
        )

        # Restore Sionna's random seed so bits/channel stay stochastic.
        _sionna_phy.config.seed = _prev_sionna_seed

        self.channel_model = TDL(
            model="C",
            delay_spread=self.delay_spread,
            carrier_frequency=self.carrier_frequency,
            num_rx_ant=1,
            num_tx_ant=1,
            device=dev,
        )

        self.ofdm_channel = OFDMChannel(
            self.channel_model,
            self.resource_grid,
            normalize_channel=self.normalize_channel,
            return_channel=True,
            device=dev,
        )

        self.mapper = Mapper("qam", self.bits_per_symbol, device=dev)
        self.rg_mapper = ResourceGridMapper(self.resource_grid, device=dev)
        self.ls_estimator = LSChannelEstimator(
            self.resource_grid, interpolation_type="nn", device=dev
        )
        self.lmmse_equalizer = LMMSEEqualizer(
            self.resource_grid, self.stream_management, device=dev
        )
        self.demapper = Demapper("maxlog", "qam", self.bits_per_symbol, device=dev)
        self.binary_source = BinarySource(device=dev)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def num_data_symbols(self) -> int:
        return self.resource_grid.num_data_symbols  # type: ignore[return-value]

    @property
    def num_bits_per_frame(self) -> int:
        return self.num_data_symbols * self.bits_per_symbol

    def sample_bits(self, batch_size: int) -> torch.Tensor:
        """Return random bits shaped [batch, 1, 1, num_bits_per_frame]."""
        return self.binary_source([batch_size, 1, 1, self.num_bits_per_frame])

    def transmit(self, bits: torch.Tensor, no: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Bits → transmitted OFDM grid → TDL-C channel → received grid.

        Args:
            bits: [batch, 1, 1, num_bits_per_frame]
            no:   scalar or broadcastable noise power tensor

        Returns:
            y:      [batch, 1, 1, num_ofdm_symbols, fft_size] received grid
            h_freq: [batch, 1, 1, 1, 1, num_ofdm_symbols, fft_size] true channel
        """
        x = self.mapper(bits)           # [B, 1, 1, num_data_symbols]
        x_grid = self.rg_mapper(x)      # [B, 1, 1, num_ofdm_symbols, fft_size]
        y, h_freq = self.ofdm_channel(x_grid, no)
        return y, h_freq

    def pilot_mask(self) -> torch.Tensor:
        """
        Boolean tensor [num_ofdm_symbols, fft_size]: True at pilot positions.

        Used by the neural receiver as an extra input channel.
        """
        rg = self.resource_grid
        # pilot_pattern.mask shape: [1, 1, num_ofdm_symbols, fft_size]
        mask = rg.pilot_pattern.mask.squeeze()  # [num_ofdm_symbols, fft_size]
        return mask.bool()

    def snr_db_to_no(self, snr_db: float) -> torch.Tensor:
        """Convert SNR in dB to linear noise power (per complex dim)."""
        snr_linear = 10.0 ** (snr_db / 10.0)
        return torch.tensor(1.0 / snr_linear, dtype=torch.float32)
