"""Smoke test that proves the scaffold imports and the test harness runs.

This is a placeholder so the CI workflow exits zero on the empty scaffold.
Replace / delete this file as soon as Phase 1 real tests land.
"""

from __future__ import annotations


def test_package_import():
    import neural_rx

    assert hasattr(neural_rx, "__version__")


def test_python_arithmetic_is_sane():
    # If this fails, your environment is broken — not the code.
    assert 1 + 1 == 2
