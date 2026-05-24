from __future__ import annotations

from pathlib import Path

import build_dashboard

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _html_without_boundary_negations(text: str) -> str:
    allowed_boundary_phrases = [
        "No live 5G network",
        "No SDR/hardware-loop validation",
        "No O-RAN/gNB integration",
        "No NVIDIA Aerial integration",
        "No MIMO/higher-order QAM claim",
        "No LDPC-coded system claim",
        "No production AI-RAN deployment claim",
        "does not claim live 5G deployment",
        "SDR validation",
        "gNB integration",
        "O-RAN integration",
        "NVIDIA Aerial integration",
        "MIMO",
        "higher-order QAM",
        "LDPC-coded performance",
        "production AI-RAN readiness",
        "Live 5G deployment",
        "SDR validated receiver",
        "O-RAN/gNB integration",
        "NVIDIA Aerial integration",
        "MIMO receiver",
        "LDPC-coded system",
    ]
    cleaned = text
    for phrase in allowed_boundary_phrases:
        cleaned = cleaned.replace(phrase, "")
    return cleaned


def test_build_dashboard_generates_dashboard_and_index():
    build_dashboard.build()

    assert (REPORTS / "dashboard.html").exists()
    assert (REPORTS / "index.html").exists()


def test_index_is_visual_landing_page():
    build_dashboard.build()
    html = _read(REPORTS / "index.html")

    assert "5G NR Neural Receiver Evidence Pack" in html
    assert "Why This Exists" in html
    assert "What I Built" in html
    assert "What This Is" in html
    assert "What This Is Not" in html
    assert "bler_comparison.svg" in html
    assert "llr_distribution_comparison.svg" in html
    assert "bler_classical.svg" in html
    assert "dashboard.html" in html
    assert "../BUSINESS_CASE.md" in html
    assert "../TECH_BRIEF.md" in html
    assert "https://github.com/obiedeh/ai-phy-neural-receiver-benchmark" in html
    assert "Neural BER @ 5 dB" in html
    assert "Neural BER @ 10 dB" in html
    assert "Neural BLER @ 12.5 dB" in html


def test_dashboard_contains_operator_console_sections():
    build_dashboard.build()
    html = _read(REPORTS / "dashboard.html")

    expected = [
        "Technical Decision Summary",
        "Problem",
        "What I Built",
        "What I Found",
        "What I Would Validate Next",
        "Where Neural Wins",
        "Where Classical Holds",
        "Receiver Architecture Comparison",
        "Engineering Practices That Matter",
        "What This Is",
        "What This Is Not",
        "Evidence vs Boundary",
        "ONNX parity",
        "deterministic pilot",
        "TDL-C",
        "LS+LMMSE",
        "DeepRx",
        "No live 5G network",
        "No SDR",
        "No production AI-RAN deployment",
    ]
    for text in expected:
        assert text in html


def test_business_case_and_technical_brief_exist_with_required_sections():
    business = _read(ROOT / "BUSINESS_CASE.md")
    brief = _read(ROOT / "TECH_BRIEF.md")

    for text in [
        "Decision Question",
        "Problem",
        "What I Built",
        "Finding",
        "Engineering Value",
        "Recommendation",
        "Boundaries",
    ]:
        assert text in business

    for text in [
        "System Purpose",
        "Link Configuration",
        "Receiver Paths",
        "Evaluation Workflow",
        "Evidence Artifacts",
        "Reproducibility",
        "Engineering Notes",
        "Known Limits",
    ]:
        assert text in brief


def test_readme_preserves_links_and_neural_does_not_always_win_boundary():
    readme = _read(ROOT / "README.md")

    assert "Open the live dashboard" in readme
    assert "https://obiedeh.github.io/ai-phy-neural-receiver-benchmark/reports/index.html" in readme
    assert "https://obiedeh.github.io/ai-phy-neural-receiver-benchmark/reports/dashboard.html" in readme
    assert "GitHub shows committed HTML files as source code" in readme
    assert "BUSINESS_CASE.md" in readme
    assert "TECH_BRIEF.md" in readme
    assert "Why This Exists" in readme
    assert "What This Is" in readme
    assert "What This Is Not" in readme
    assert "classical baseline remains competitive at high SNR" in readme
    assert 'The result is not "neural always wins"' in readme


def test_generated_html_has_no_positive_overclaim_language():
    build_dashboard.build()
    dashboard = _html_without_boundary_negations(_read(REPORTS / "dashboard.html"))
    index = _html_without_boundary_negations(_read(REPORTS / "index.html"))
    combined = f"{dashboard}\n{index}"

    forbidden = [
        "live 5G deployment",
        "production AI-RAN deployment",
        "NVIDIA Aerial integration",
        "gNB integration",
        "O-RAN integration",
        "SDR validated",
        "hardware validated",
        "MIMO support",
        "LDPC-coded performance",
        "production ready",
    ]
    for phrase in forbidden:
        assert phrase not in combined
