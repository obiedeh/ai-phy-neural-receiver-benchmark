# Validation Matrix: 5G NR Neural Receiver Evidence Pack

This file defines what the current project proves, what it does not prove, and which validation gates would be required before making stronger AI-RAN, SDR, or hardware-facing claims.

The current repo is intentionally scoped as a simulated-link AI-PHY evidence pack. It is not a production neural receiver, not an SDR deployment, and not an O-RAN/gNB integration.

## Current Evidence Boundary

| Area | Current evidence | Status | Interpretation |
|---|---|---:|---|
| Classical baseline | LS pilot estimation, LMMSE equalization, max-log soft demapping | Done | The neural receiver is compared against a real baseline, not a straw man. |
| Learned receiver | DeepRx-style residual CNN receiver | Done | Tests joint learned receiver behavior across channel estimation, equalization, and demapping. |
| Channel model | Sionna 5G NR TDL-C | Done | Useful for controlled link evidence, but still simulated. |
| Waveform scope | CP-OFDM, QPSK, SISO | Done | Good first scope, but too narrow for broad 5G NR receiver claims. |
| BER/BLER sweep | Measured comparison across SNR | Done | Shows strongest neural advantage in the moderate-SNR region. |
| Export parity | ONNXRuntime parity pass, max_diff=1.38e-05 | Done | Proves export correctness, not production runtime readiness. |
| Reproducibility | Committed CSV, SVG, JSON, dashboard, tests | Done | Makes the result inspectable by reviewers. |
| Deployment claim | Live RAN, SDR, O-RAN, gNB, Aerial | Not claimed | Correct boundary. Do not oversell this. |

## Next Validation Gates

| Priority | Gate | Current state | Why it matters | Acceptance evidence |
|---:|---|---|---|---|
| 1 | 16-QAM / 64-QAM modulation sweep | Not implemented | QPSK-only is a useful starting point, but too easy to overread. | BER/BLER curves for QPSK, 16-QAM, and 64-QAM under the same baseline and neural paths. |
| 2 | LDPC-coded transport-block BLER | Not implemented | Real 5G link decisions care about coded block success, not only uncoded bit behavior. | Transport-block BLER with LDPC encode/decode path and fixed MCS assumptions. |
| 3 | Doppler / mobility sweep | Not implemented | AI-PHY value depends on channel dynamics, not only static channel-family behavior. | BLER versus Doppler or user speed, with neural and classical curves shown together. |
| 4 | Channel mismatch / generalization test | Not implemented | A neural receiver can overfit one channel family. | Train on one channel configuration, test on another, and report degradation. |
| 5 | SIMO or MIMO receiver scope | Not implemented | SISO caps credibility for modern wireless receiver claims. | 1x2 SIMO or 2x2 MIMO benchmark with explicit antenna/channel assumptions. |
| 6 | Runtime latency benchmark | ONNX parity only | Export correctness is not runtime suitability. | ONNXRuntime and, if available, TensorRT inference latency with p50/p95 timing. |
| 7 | Hardware-loop / SDR experiment | Not implemented | Required before hardware-facing AI-RAN claims. | USRP/OAI or equivalent captured-IQ workflow with measured BER/BLER or decoder success. |
| 8 | AI-RAN integration path | Not implemented | Link-level evidence does not equal RAN integration. | Clear architecture note mapping receiver artifact to an SDR/OAI/Aerial-style future integration boundary. |

## Recommended Upgrade Sequence

### Phase 1: Make the benchmark harder

Add higher-order modulation and coded BLER before touching hardware. This is the highest signal-to-effort upgrade because it strengthens the current evidence without pretending the project is deployment-ready.

Minimum outputs:

- `reports/modulation_sweep.csv`
- `reports/modulation_sweep.svg`
- `reports/coded_bler_comparison.csv`
- dashboard section: "QPSK vs 16-QAM vs 64-QAM"
- README update with measured values only

### Phase 2: Stress channel generalization

Test whether the neural receiver still holds when the channel changes. This is where weak neural receiver demos usually break.

Minimum outputs:

- train/evaluate split by channel family or Doppler condition
- mismatch penalty table
- dashboard section: "Where the neural receiver fails"

### Phase 3: Add runtime evidence

Run ONNXRuntime timing first. Add TensorRT only if the environment is stable and the pipeline remains simple.

Minimum outputs:

- `reports/runtime_latency.json`
- p50/p95 latency table
- CPU/GPU/device details
- batch size and tensor shape disclosure

### Phase 4: Hardware-loop path

Only after the simulation benchmark is stronger, add SDR/OAI or captured-IQ validation. Jumping to hardware too early creates noise and hides whether the receiver itself is strong.

Minimum outputs:

- captured-IQ input description
- reproducible replay script
- measured decoder or error-rate outcome
- explicit hardware limitations

## Strong Public Framing

Use this language:

> This is a reproducible AI-PHY evidence pack for comparing a learned receiver against a classical LS+LMMSE chain under controlled Sionna-modeled 5G NR conditions. It shows where the neural receiver helps, where the classical baseline holds, and what must be validated next before any AI-RAN deployment claim.

Avoid this language:

- "production neural receiver"
- "AI-RAN deployment"
- "O-RAN integrated"
- "SDR validated"
- "novel neural receiver architecture"
- "neural receiver always beats classical"

Those claims are not supported by the current evidence.

## Bottom Line

The current project is strong because it is honest and reproducible. Its uniqueness is not research novelty. Its uniqueness is packaging the receiver comparison as an engineering evidence pack with visible boundaries.

The next upgrade should make the benchmark harder, not louder.
