# Technical Brief: 5G NR Neural Receiver AI-PHY Evidence Pack

## Open The Evidence Pack

> [▶ Open the live dashboard](https://obiedeh.github.io/neural-receiver-5g-nr/reports/dashboard.html) · [Landing page](https://obiedeh.github.io/neural-receiver-5g-nr/reports/index.html) · [Business case](BUSINESS_CASE.md) · [Technical brief](TECH_BRIEF.md) · [Validation matrix](VALIDATION_MATRIX.md)

GitHub shows committed HTML files as source code. Use the live GitHub Pages links above to open the rendered dashboard and landing page.

## System Purpose

Compare a learned DeepRx-style receiver against a classical LS+LMMSE baseline on the same Sionna-modeled 5G NR link.

The purpose is controlled AI-PHY evidence, not deployment theater. The project shows where the learned receiver helps, where the classical baseline remains strong, and which validation gates are still missing before stronger AI-RAN, SDR, or hardware-facing claims are justified.

## Link Configuration

- Sionna 2.0
- 3GPP TR 38.901 TDL-C
- QPSK
- CP-OFDM
- 14 symbols/slot
- 76 subcarriers
- 30 kHz SCS
- CP=6
- SISO
- random SNR training range: [-5, +20] dB

## Receiver Paths

Classical path:

```text
received grid
-> LS pilot channel estimation
-> LMMSE equalization
-> max-log soft demapping
-> LLRs
```

Neural path:

```text
received grid + pilot mask
-> DeepRx-style residual CNN
-> learned joint channel estimation / equalization / demapping
-> LLRs
```

The neural receiver uses received-grid real/imaginary tensors plus the pilot mask. It outputs LLRs and is trained with end-to-end BCE loss through the Sionna channel.

## Evaluation Workflow

1. Configure the shared Sionna TDL-C/QPSK/SISO link.
2. Run the classical LS+LMMSE baseline sweep.
3. Train the neural receiver with random SNR minibatches.
4. Run the head-to-head BER/BLER comparison.
5. Export the neural receiver to ONNX.
6. Run the ONNXRuntime parity check.
7. Generate the dashboard and landing page.
8. Document the validation gates that remain open.

## Evidence Artifacts

- `reports/index.html`
- `reports/dashboard.html`
- `reports/bler_comparison.csv`
- `reports/bler_comparison.svg`
- `reports/llr_distribution_comparison.svg`
- `reports/onnx_parity_test.json`
- `reports/training_log.json`
- `reports/comparison_summary.json`
- `VALIDATION_MATRIX.md`

## Key Results

- BER @ 5 dB improves from 0.122 classical to 0.062 neural.
- BER @ 10 dB improves from 0.026 classical to 0.014 neural.
- BLER @ 12.5 dB improves from 0.938 classical to 0.585 neural.
- The effective moderate-SNR BER advantage is about 2-3 dB.
- At high SNR, the classical baseline remains competitive.
- ONNX parity passes with max_diff=1.38e-05.

## Engineering Interpretation

| Region | Neural receiver behavior | Classical receiver behavior | Interpretation |
|---|---|---|---|
| Low SNR | Improves BER, but frame errors remain high | Also frame-error limited | Useful signal, not an operating-point victory |
| Moderate SNR | Shows the clearest measured advantage | Falls behind under the same TDL-C/QPSK/SISO setup | Strongest current evidence for learned receiver value |
| High SNR | No longer dominates | Remains competitive | The result is not "neural always wins" |

## Engineering Notes

The deterministic pilot RNG bug was found and fixed during development. Sionna's `KroneckerPilotPattern` can generate different pilot values across identical-looking link instances if its internal random generator is not controlled. The fix sets `sionna.phy.config.seed` before `ResourceGrid` construction and restores it afterward.

This matters because a trained neural receiver can silently fail if inference-time pilots differ from training-time pilots. Surfacing and fixing this issue is one of the strongest engineering signals in the repo.

## Current Evidence Boundary

| Area | Current evidence | Status |
|---|---|---:|
| Classical receiver baseline | LS + LMMSE + max-log soft demapping | done |
| Neural receiver | DeepRx-style residual CNN | done |
| Channel model | Sionna 5G NR TDL-C | done |
| Waveform scope | CP-OFDM / QPSK / SISO | done |
| Error-rate evidence | BER/BLER sweep across SNR | done |
| Export check | ONNXRuntime parity | done |
| Runtime suitability | Latency / throughput benchmark | not measured |
| Higher-order modulation | 16-QAM / 64-QAM | not measured |
| Coded transport-block behavior | LDPC-coded BLER | not measured |
| Hardware validation | SDR / captured IQ / OAI | not measured |
| RAN integration | gNB / O-RAN / Aerial | not claimed |

## Next Validation Gates

The next upgrade should make the benchmark harder, not louder.

1. Add 16-QAM and 64-QAM sweeps.
2. Add LDPC-coded transport-block BLER.
3. Add Doppler or mobility sweeps.
4. Add channel mismatch/generalization tests.
5. Add SIMO or MIMO scope.
6. Add ONNXRuntime and optional TensorRT latency measurements.
7. Add SDR/OAI hardware-loop validation only after the simulation benchmark is stronger.

See `VALIDATION_MATRIX.md` for the full validation plan.

## Reproducibility

```bash
pip install -e ".[dev]"
python scripts/run_bler_classical.py
python scripts/train_neural_rx.py --steps 100000
python scripts/run_bler_comparison.py
python scripts/export_onnx.py
python build_dashboard.py
make verify
```

`make verify` expects a trained checkpoint in `models/neural_rx_best.pt`.

## Known Limits

- no SDR
- no real RAN
- no MIMO
- no higher-order QAM
- no LDPC-coded system claim unless implemented
- no production deployment
- no gNB/O-RAN integration
- no NVIDIA Aerial integration

These limits should stay visible. Hiding them would make the project weaker, not stronger.
