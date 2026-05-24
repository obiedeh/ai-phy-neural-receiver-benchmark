# Business Case: AI-PHY Neural Receiver Evidence Pack

## Decision Question

Can a learned receiver improve link reliability over a classical LS+LMMSE receiver under the same controlled 5G NR TDL-C channel model, and what validation gates are still required before stronger AI-RAN claims are credible?

## Executive Summary

This project is not a production neural receiver claim. It is a controlled AI-PHY evidence pack.

The current evidence shows that a DeepRx-style neural receiver improves BER around the moderate-SNR region on a Sionna-modeled 5G NR TDL-C/QPSK/SISO link, while the classical LS+LMMSE baseline remains competitive at high SNR.

That is the right result to show publicly because it is measured, bounded, and honest. The value is not "neural replaces classical everywhere." The value is knowing where the learned receiver helps, where it does not, and what must be validated next.

## Operational / Engineering Relevance

Receiver design affects link reliability, scheduler margin, and radio-system robustness. A learned receiver is only useful if it improves the error curve under controlled conditions and the boundary conditions are clear.

For AI-native radio work, the first credible step is not a dashboard full of claims. It is a fair baseline comparison with reproducible artifacts.

## Problem

Neural receiver claims can be easy to overstate. Without a fair classical baseline, deterministic pilots, and reproducible BER/BLER curves, the result is not useful engineering evidence.

A weak project would say: "the neural receiver wins."

A stronger project says: "the neural receiver wins here, classical holds there, and these are the validation gates still missing."

## What I Built

I built a DeepRx-style neural receiver on a Sionna-modeled 5G NR link and compared it against a classical LS+LMMSE+soft-demap baseline using the same TDL-C channel setup.

The evidence pack includes:

- same simulated 5G NR link for both receiver paths
- LS+LMMSE classical baseline
- DeepRx-style residual CNN receiver
- BER/BLER comparison across SNR
- ONNX export parity check
- deterministic pilot fix
- committed CSV/SVG/JSON artifacts
- rendered dashboard and landing page
- validation matrix for next gates

## Finding

The neural receiver improved BER by roughly 2x around 5-10 dB and improved BLER at 12.5 dB from 0.938 to 0.585. At higher SNR, the classical receiver remained competitive.

The finding is not "neural always wins." The finding is more useful than that: the neural receiver has a measured moderate-SNR advantage under the current channel assumptions, but the benchmark is not yet broad enough for production or AI-RAN deployment claims.

## Engineering Value

The engineering value is a reproducible AI-PHY comparison pattern:

- fair baseline
- same channel setup
- visible BER/BLER curves
- deterministic pilot handling
- ONNX parity
- dashboard evidence
- explicit boundaries

This gives reviewers something stronger than screenshots or claims. It gives them an inspectable artifact trail.

## Recommendation

Use this project as a controlled AI-PHY evidence pattern. Do not position it as novel neural receiver research or production AI-RAN deployment.

The next best upgrades are:

1. 16-QAM / 64-QAM sweep
2. LDPC-coded transport-block BLER
3. Doppler / mobility sweep
4. channel mismatch and generalization test
5. SIMO or MIMO receiver scope
6. ONNXRuntime or TensorRT latency benchmark
7. SDR/OAI hardware-loop path only after the simulation benchmark is stronger

## Boundaries

This is simulated link evidence only. It does not claim live 5G deployment, SDR validation, gNB integration, O-RAN integration, NVIDIA Aerial integration, MIMO, higher-order QAM, LDPC-coded performance, or production AI-RAN readiness.

That boundary is not a weakness. It is what keeps the project credible.
