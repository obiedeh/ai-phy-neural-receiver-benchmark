# AGENTS.md — Coding Agent Guide

**Neural Receiver — 5G NR (DeepRx pattern)**
Operating instructions for AI coding agents working in this repo.
Read [HANDOFF.md](HANDOFF.md) first.

---

## Project Purpose

A GPU-accelerated reference for AI-PHY: a DeepRx-style neural receiver trained end-to-end on a Sionna-modeled 5G NR link, compared head-to-head against the classical LS + LMMSE + soft-demap baseline on the same TDL-C channel realisations.

**This is not a production receiver, not a standards-compliant modem, not an AI-RAN base station.** It is a measurable research testbed where the engineering pattern — Sionna PHY + differentiable training + honest BLER comparison + ONNX deployment + Jetson benchmark — is the deliverable.

The companion repo [`wireless-link-intelligence-system`](https://github.com/obiedeh/wireless-link-intelligence-system) is the production-discipline pattern this repo applies. Treat its commit history, structure, and conventions as canonical.

---

## Non-Negotiable Rules

1. **No oracle leakage into the neural receiver.** The neural RX must see only the received resource grid + pilot positions. It must NEVER see the transmitted bits, the true channel impulse response, or the noise variance as an input. If you find yourself wanting to pass any of those in, you're solving a different problem.
2. **Same realisations for classical and neural.** The BLER comparison must use identical channel realisations and noise for both receivers — apples-to-apples is the entire point.
3. **Honest BLER reporting.** If the neural RX matches LMMSE at high SNR and pulls away at low SNR, report it as the textbook DeepRx result. If it matches LMMSE everywhere, report that. If it loses to LMMSE, **report that too** and treat it as a calibration finding (more training, deeper network, different hyperparameters — but never hidden).
4. **Tests stay green.** `pytest -q` and `ruff check .` must pass before every commit.
5. **Deterministic seeding.** All RNGs (`torch.manual_seed`, `numpy.random.default_rng`, Sionna seeds) must be configured from a single top-level seed, threaded through every stochastic step.
6. **No "AI-RAN" / "6G" / "world model" framing in any user-facing text.** The work either supports the claim or it doesn't; the language follows.

---

## Architecture

```
                ┌─────────────────────────────┐
                │   Sionna 5G NR link model   │
                │   (PyTorch backend)         │
                └────────────┬────────────────┘
                             │
                ┌────────────┼────────────┐
                │                         │
                ▼                         ▼
        TX: random bits           Channel: TDL-C
        → QPSK → CP-OFDM          (3GPP TR 38.901)
        + pilots                  + AWGN
                                          │
                                          ▼
                                 Received resource grid
                                          │
                          ┌───────────────┼───────────────┐
                          ▼                               ▼
                Classical RX (baseline)         Neural RX (DeepRx pattern)
                  LS channel estimation            CNN over resource grid
                  LMMSE equalization               → bit LLRs
                  Soft demap → LLRs                (trained end-to-end via BCE)
                          │                               │
                          └───────────────┬───────────────┘
                                          ▼
                              BLER vs SNR comparison
                              (same realisations)
                                          │
                                          ▼
                                 ONNX export + Jetson benchmark
```

---

## Module layout (to be created)

```
neural_rx/
  sionna_link.py     # Sionna CP-OFDM resource grid + TDL-C + pilots
  classical_rx.py    # LS + LMMSE + soft demap
  neural_rx.py       # DeepRx-style CNN
  training.py        # End-to-end training loop helpers
scripts/
  run_bler_classical.py
  train_neural_rx.py
  run_bler_comparison.py
  export_onnx.py
edge/
  jetson_benchmark.py
tests/
  test_classical_rx.py
  test_neural_rx_shapes.py
  test_sionna_link.py
reports/                # Generated artifacts
notebooks/legacy/       # Exploration notebooks (stripped of outputs)
build_dashboard.py      # HTML dashboard generator
```

---

## Dependencies

Runtime: `numpy`, `scipy`, `matplotlib`, `torch`, `sionna`.
Dev: `pytest`, `ruff`.
Edge: `onnx`, `onnxruntime`, `onnxruntime-gpu` (optional).

PyTorch with CUDA and Sionna are installed via the manual recipe in [HANDOFF.md §3](HANDOFF.md#3-environment-setup-phase-0). They are not in `[project.dependencies]` because the right CUDA channel and Sionna backend depend on the box.

---

## Adding a feature

1. Open a focused commit per phase (see HANDOFF.md §4).
2. Add unit tests in `tests/`.
3. Update Makefile and CI workflow as needed.
4. Commit with a clear subject + body explaining why and what's measured.
5. Confirm `make verify` regenerates the artifacts deterministically.

---

## Running tests

```bash
PYTHONWARNINGS="ignore::DeprecationWarning" pytest -q
ruff check .
```

(Sionna and PyTorch both emit deprecation warnings — silence them in test output, not in the codebase.)
