.PHONY: install install-dev test lint bler-classical train compare export dashboard verify clean

PYTHON ?= .venv/bin/python
PIP    ?= $(PYTHON) -m pip

.venv:
	python3 -m venv .venv

install: .venv
	$(PIP) install -e .

install-dev: .venv
	$(PIP) install -e ".[dev,edge]"

test: .venv
	$(PYTHON) -m pytest -q

lint: .venv
	$(PYTHON) -m ruff check .

# ---- Phase 1: classical baseline BLER sweep ----
bler-classical:
	$(PYTHON) scripts/run_bler_classical.py

# ---- Phase 3: train neural receiver ----
train:
	$(PYTHON) scripts/train_neural_rx.py

# ---- Phase 4: head-to-head BLER comparison ----
compare:
	$(PYTHON) scripts/run_bler_comparison.py

# ---- Phase 5: ONNX export + parity test ----
export:
	$(PYTHON) scripts/export_onnx.py

# ---- Phase 6: build dashboard ----
dashboard:
	$(PYTHON) build_dashboard.py

# ---- Full reproduction chain ----
# Requires a trained checkpoint in models/neural_rx_best.pt
verify: lint test bler-classical compare export dashboard
	@echo "All verification steps complete."

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ */__pycache__ */*/__pycache__
	rm -rf dist build *.egg-info
