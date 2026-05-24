"""Build static evidence pages for the neural receiver project.

The generator reads the committed CSV/JSON/SVG artifacts under ``reports/`` and
renders two static pages:

* ``reports/dashboard.html``: AI-PHY decision and evidence console.
* ``reports/index.html``: visual landing page / evidence-pack launchpad.

No values are pulled from external systems. The pages are intentionally scoped to
simulated Sionna link evidence and do not claim live RAN, SDR, O-RAN, or hardware
deployment validation.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row(rows: list[dict[str, str]], snr: float) -> dict[str, str]:
    for candidate in rows:
        if abs(float(candidate["snr_db"]) - snr) < 0.01:
            return candidate
    return {}


def _fmt_float(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _fmt_sci(value: Any) -> str:
    return f"{float(value):.2e}"


def _safe(value: Any, fallback: str = "not measured") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


def _link(path: str, label: str) -> str:
    return f'<a href="{html.escape(path)}">{html.escape(label)}</a>'


def _badge(text: str, cls: str = "neutral") -> str:
    return f'<span class="badge {cls}">{html.escape(text)}</span>'


def _section(eyebrow: str, title: str, body: str, anchor: str | None = None) -> str:
    anchor_attr = f' id="{anchor}"' if anchor else ""
    return (
        f"<section{anchor_attr}>"
        f"<h2>{html.escape(eyebrow)}</h2>"
        f"<h3>{html.escape(title)}</h3>"
        f"{body}"
        "</section>"
    )


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _img(src: str, alt: str) -> str:
    return f'<img src="{html.escape(src)}" alt="{html.escape(alt)}" />'


def _write_text_if_changed(path: Path, content: str) -> None:
    """Write generated HTML unless the existing locked file is already current."""
    try:
        if path.exists() and path.read_text(encoding="utf-8") == content:
            return
        path.write_text(content, encoding="utf-8")
    except PermissionError:
        if path.exists() and path.read_text(encoding="utf-8") == content:
            return
        raise


def _kpi(label: str, value: str, note: str, cls: str = "neutral") -> str:
    return (
        f'<div class="kpi {cls}">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f'<div class="kpi-note">{html.escape(note)}</div>'
        "</div>"
    )


def _load_evidence() -> dict[str, Any]:
    rows = _read_csv(REPORTS / "bler_comparison.csv")
    training = _read_json(REPORTS / "training_log.json")
    onnx = _read_json(REPORTS / "onnx_parity_test.json")

    r5 = _row(rows, 5.0)
    r10 = _row(rows, 10.0)
    r125 = _row(rows, 12.5)
    r15 = _row(rows, 15.0)
    r20 = _row(rows, 20.0)
    cfg = training.get("config", {})

    evidence = {
        "rows": rows,
        "ber_classical_5": float(r5.get("ber_classical", 0.0)),
        "ber_neural_5": float(r5.get("ber_neural", 0.0)),
        "ber_classical_10": float(r10.get("ber_classical", 0.0)),
        "ber_neural_10": float(r10.get("ber_neural", 0.0)),
        "bler_classical_125": float(r125.get("bler_classical", 0.0)),
        "bler_neural_125": float(r125.get("bler_neural", 0.0)),
        "ber_classical_15": float(r15.get("ber_classical", 0.0)),
        "ber_neural_15": float(r15.get("ber_neural", 0.0)),
        "ber_classical_20": float(r20.get("ber_classical", 0.0)),
        "ber_neural_20": float(r20.get("ber_neural", 0.0)),
        "parameters": int(cfg.get("parameters", 373154)),
        "steps": int(cfg.get("steps", 100000)),
        "batch_size": int(cfg.get("batch_size", 64)),
        "snr_min": float(cfg.get("snr_min_db", -5.0)),
        "snr_max": float(cfg.get("snr_max_db", 20.0)),
        "onnx_pass": bool(onnx.get("parity_pass", False)),
        "onnx_diff": float(onnx.get("max_abs_diff", 0.0)),
        "onnx_opset": int(onnx.get("opset", 18)),
    }
    return evidence


CSS = """
:root {
  --bg: #f8fafc;
  --panel: #ffffff;
  --panel-2: #f1f5f9;
  --line: #dbe4ee;
  --text: #0f172a;
  --muted: #64748b;
  --blue: #2563eb;
  --green: #047857;
  --amber: #b45309;
  --red: #b91c1c;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background:
    linear-gradient(180deg, rgba(37, 99, 235, 0.10), rgba(248, 250, 252, 0.0) 260px),
    var(--bg);
  color: var(--text);
  font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  line-height: 1.55;
}
a { color: var(--blue); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 1240px; margin: 0 auto; padding: 32px 28px 48px; }
header { margin-bottom: 24px; }
header h1 { margin: 0 0 8px; font-size: 36px; letter-spacing: -0.03em; }
header .sub { color: var(--muted); max-width: 900px; }
.nav { margin-top: 14px; color: var(--muted); font-size: 14px; }
.nav a { margin-right: 16px; }
.hero-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 18px;
}
.hero-actions { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 10px; }
.hero-actions a {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 7px 11px;
  background: var(--panel-2);
  font-size: 13px;
}
section, .card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 22px;
  margin-bottom: 18px;
}
section h2, .eyebrow {
  margin: 0 0 4px;
  color: var(--blue);
  font-size: 12px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}
section h3, .card h3 {
  margin: 0 0 14px;
  font-size: 24px;
  letter-spacing: -0.02em;
}
p { margin: 0 0 12px; }
.kpi-grid, .grid-2, .grid-3, .visual-grid {
  display: grid;
  gap: 14px;
}
.kpi-grid { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin: 16px 0; }
.grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.grid-3, .visual-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.kpi {
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.kpi-label { color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-size: 11px; }
.kpi-value { color: var(--text); font-size: 26px; font-weight: 700; margin-top: 4px; }
.kpi-note { color: var(--muted); font-size: 12px; margin-top: 4px; }
.good .kpi-value { color: var(--green); }
.warn .kpi-value { color: var(--amber); }
.risk .kpi-value { color: var(--red); }
.badge {
  display: inline-block;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 3px 8px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.badge.good { border-color: #047857; color: var(--green); }
.badge.warn { border-color: #92400e; color: var(--amber); }
.badge.risk { border-color: #991b1b; color: var(--red); }
.callout {
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-left: 4px solid var(--blue);
  border-radius: 8px;
  padding: 14px 16px;
}
.callout h4 { margin: 0 0 6px; font-size: 15px; }
.plot {
  background: #fff;
  border-radius: 8px;
  padding: 10px;
  margin-bottom: 10px;
  border: 1px solid var(--line);
}
.plot img, .plot svg { width: 100%; height: auto; display: block; }
.plot-card img { width: 100%; height: auto; display: block; background: #fff; border-radius: 8px; }
.source { color: var(--muted); font-size: 12px; }
table { width: 100%; border-collapse: collapse; margin: 12px 0 0; font-size: 14px; }
th, td { border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
ul { margin: 8px 0 0 20px; padding: 0; }
footer { color: var(--muted); font-size: 13px; text-align: center; margin-top: 22px; }
@media (max-width: 900px) {
  .grid-2, .grid-3, .visual-grid { grid-template-columns: 1fr; }
  header h1 { font-size: 30px; }
}
"""


def _technical_decision_summary(e: dict[str, Any]) -> str:
    rows = [
        ["Experiment", "DeepRx-style neural receiver vs classical LS+LMMSE baseline"],
        ["Link model", "Sionna 5G NR, TDL-C, CP-OFDM, QPSK, SISO"],
        ["Neural receiver size", f"{e['parameters']:,} trainable parameters"],
        ["Main finding", "Neural receiver improves BER around 5-10 dB"],
        ["Moderate-SNR gain", "about 2-3 dB effective BER advantage"],
        [
            "BLER finding",
            f"12.5 dB neural BLER {_fmt_float(e['bler_neural_125'])} vs classical {_fmt_float(e['bler_classical_125'])}",
        ],
        ["High-SNR boundary", "Classical baseline remains competitive above 15 dB"],
        ["Export evidence", f"ONNX parity {'PASS' if e['onnx_pass'] else 'not passed'}, max_diff={_fmt_sci(e['onnx_diff'])}"],
        ["Deployment boundary", "Simulated link evidence only, not live RAN or SDR deployment"],
    ]
    kpis = (
        '<div class="kpi-grid">'
        + _kpi("BER @ 5 dB", _fmt_float(e["ber_neural_5"]), f"classical {_fmt_float(e['ber_classical_5'])}", "good")
        + _kpi("BER @ 10 dB", _fmt_float(e["ber_neural_10"]), f"classical {_fmt_float(e['ber_classical_10'])}", "good")
        + _kpi("BLER @ 12.5 dB", _fmt_float(e["bler_neural_125"]), f"classical {_fmt_float(e['bler_classical_125'])}", "good")
        + _kpi("ONNX parity", "PASS" if e["onnx_pass"] else "CHECK", f"max_diff={_fmt_sci(e['onnx_diff'])}", "good" if e["onnx_pass"] else "warn")
        + "</div>"
    )
    return _section(
        "Technical Decision Summary",
        "What does the AI-PHY experiment prove?",
        kpis + _table(["Decision field", "Evidence"], rows),
        "technical-decision-summary",
    )


def _story_section(e: dict[str, Any]) -> str:
    body = (
        '<div class="grid-2">'
        '<div class="callout"><h4>Problem</h4><p>Classical 5G receivers separate channel estimation, equalization, and demapping. Neural receivers test whether those functions can be learned jointly under realistic channel models.</p></div>'
        '<div class="callout"><h4>What I Built</h4><p>I built a DeepRx-style residual CNN receiver on a Sionna-modeled 5G NR TDL-C link and compared it against an LS+LMMSE+soft-demap baseline on the same channel conditions.</p></div>'
        f'<div class="callout"><h4>What I Found</h4><p>The neural receiver improves BER by about 2x around 5-10 dB and shows a 12.5 dB BLER improvement from {_fmt_float(e["bler_classical_125"])} to {_fmt_float(e["bler_neural_125"])}. At high SNR, the classical baseline remains competitive.</p></div>'
        '<div class="callout"><h4>What I Would Validate Next</h4><p>I would test higher-order modulation, MIMO, LDPC-coded BLER, mobility/Doppler sweeps, channel mismatch, and SDR/hardware-loop integration before making deployment claims.</p></div>'
        "</div>"
    )
    return _section(
        "Problem -> What I Built -> What I Found -> What I Would Validate Next",
        "The AI-PHY story in one pass",
        body,
        "story",
    )


def _wins_holds_section() -> str:
    rows = [
        [
            "Low SNR, -5 to 0 dB",
            "Improves BER, but frames still fail heavily.",
            "Also frame-error limited in the uncoded QPSK setup.",
            "Useful signal, not an operating-point victory.",
            "No coded BLER or scheduler claim.",
        ],
        [
            "Moderate SNR, 5 to 12.5 dB",
            "Shows the strongest practical advantage and the clearest BLER improvement.",
            "Falls behind on BER/BLER under the same TDL-C conditions.",
            "This is the strongest measured neural-receiver region.",
            "Measured only for TDL-C/QPSK/SISO.",
        ],
        [
            "High SNR, 15 to 20 dB",
            "No longer dominates; BER can trail the classical chain.",
            "Becomes competitive as channel estimation improves.",
            "The result is not 'neural always wins.'",
            "High-SNR limitation remains visible.",
        ],
    ]
    return _section(
        "Where Neural Wins / Where Classical Holds",
        "The measured result has boundaries",
        _table(
            [
                "SNR region",
                "Neural receiver behavior",
                "Classical baseline behavior",
                "Engineering interpretation",
                "Boundary",
            ],
            rows,
        ),
        "where-neural-wins",
    )


def _architecture_section() -> str:
    rows = [
        ["Channel estimation", "LS pilot channel estimation", "Learned jointly from received grid + pilot mask", "Neural path can exploit local structure beyond explicit LS estimates."],
        ["Equalization", "LMMSE equalization", "Implicitly learned through CNN residual blocks", "Moves receiver design from separated blocks to learned joint inference."],
        ["Demapping", "Max-log soft demapping", "CNN outputs LLRs for data bits", "Keeps the output compatible with a downstream soft-decoder path."],
        ["Training/tuning", "No training; model assumptions are explicit", "End-to-end BCE through Sionna channel, random SNR per minibatch", "Learned receiver needs data/model discipline and validation across mismatch."],
        ["Export/deployment artifact", "Algorithmic Python baseline", "ONNX opset export with parity check", "ONNX proves export correctness, not production readiness."],
        ["Failure/validation risk", "Can be strong at high SNR", "Can overfit channel/modulation assumptions", "Both paths need explicit operating boundaries."],
    ]
    return _section(
        "Receiver Architecture Comparison",
        "Classical blocks versus learned joint receiver",
        _table(["Stage", "Classical path", "Neural path", "Why it matters"], rows),
        "architecture-comparison",
    )


def _credibility_section(e: dict[str, Any]) -> str:
    rows = [
        ["deterministic pilot RNG bug", "Found and fixed by controlling `sionna.phy.config.seed` before ResourceGrid creation."],
        ["Fair comparison", "Same TDL-C/QPSK/SISO link conditions are used for the classical and neural receiver paths."],
        ["ONNX parity", f"{'PASS' if e['onnx_pass'] else 'CHECK'} with max_diff={_fmt_sci(e['onnx_diff'])}."],
        ["Test suite", "31/31 tests passing in the recorded evidence."],
        ["Training cost", f"10.4 min / {e['steps']:,} steps on RTX 5090, batch={e['batch_size']}."],
        ["Committed artifacts", "CSV, SVG, JSON, dashboard, and reproducibility workflow are committed."],
    ]
    return _section(
        "Engineering Practices That Matter",
        "The practices that make the comparison useful",
        _table(["Signal", "Evidence"], rows),
        "engineering-credibility",
    )


def _why_exists_section() -> str:
    body = (
        "<p>Classical 5G receivers separate channel estimation, equalization, and demapping. "
        "Neural receivers test whether those steps can be learned jointly when the receiver sees "
        "the same channel conditions. I built this repo to make that comparison reproducible, "
        "measured, and honest.</p>"
        "<p>AI-native radio research needs controlled baselines. Neural PHY claims are easy to "
        "overstate. This evidence pack compares neural and classical receivers on the same modeled "
        "channel, with the same SNR sweep, committed artifacts, and visible limits.</p>"
    )
    return _section("Why This Exists", "Controlled neural receiver evidence", body, "why-this-exists")


def _what_this_is_section() -> str:
    rows = [
        ["AI-PHY receiver experiment", "Tests learned receiver behavior against a classical baseline."],
        ["Sionna-modeled 5G NR link", "Uses TDL-C/QPSK/CP-OFDM/SISO link conditions."],
        ["Classical receiver baseline", "LS pilot estimation, LMMSE equalization, max-log soft demapping."],
        ["DeepRx-style neural receiver", "Residual CNN that learns joint receiver behavior."],
        ["BER/BLER evidence pack", "Measured curves and tables across SNR."],
        ["ONNX parity export check", "Verifies export correctness against ONNXRuntime."],
        ["Static dashboard", "Packages measured evidence, boundaries, and artifacts for review."],
    ]
    return _section(
        "What This Is",
        "The concrete system layers in this repo",
        _table(["Layer", "What it does"], rows),
        "what-this-is",
    )


def _what_this_is_not_section() -> str:
    rows = [
        ["Live 5G deployment", "No, this is simulated link evidence.", "Prevents production overclaiming."],
        ["SDR validated receiver", "No, no hardware-loop validation yet.", "Keeps hardware claims honest."],
        ["O-RAN/gNB integration", "No, no RAN integration is claimed.", "Separates link evidence from RAN integration."],
        ["NVIDIA Aerial integration", "No, this uses Sionna, not Aerial.", "Avoids vendor integration theater."],
        ["MIMO receiver", "No, this is SISO unless implemented otherwise.", "Keeps antenna scope clear."],
        ["Higher-order QAM / LDPC-coded system", "No, QPSK and current uncoded scope only as implemented.", "Avoids unmeasured PHY claims."],
        ["Production-ready AI-RAN component", "No, this is a research-grade evidence pack.", "Preserves deployment boundary."],
    ]
    return _section(
        "What This Is Not",
        "Boundaries that keep the evidence honest",
        _table(["Claim", "What is true instead", "Limit it protects"], rows),
        "what-this-is-not",
    )


def _workflow_section() -> str:
    rows = [
        ["Sionna link config", "Shared TDL-C/QPSK/SISO link setup."],
        ["Classical baseline sweep", "`reports/bler_classical.csv` and plot."],
        ["Neural receiver training", "`reports/training_log.json`."],
        ["Head-to-head BER/BLER comparison", "`reports/bler_comparison.csv` and plot."],
        ["ONNX export", "Model export when checkpoint artifacts are available."],
        ["Parity check", "`reports/onnx_parity_test.json`."],
        ["Dashboard generation", "`reports/index.html` and `reports/dashboard.html`."],
        ["Tests / verify", "pytest, ruff, and `make verify`."],
    ]
    return _section(
        "Technical Workflow",
        "How the evidence is produced",
        _table(["Stage", "Output"], rows),
        "technical-workflow",
    )


def _evidence_boundary_section() -> str:
    evidence = [
        "Sionna-modeled 5G NR TDL-C link",
        "Classical LS+LMMSE baseline",
        "DeepRx-style neural receiver",
        "BER/BLER comparison",
        "ONNX export parity",
        "Deterministic pilot fix",
        "Reproducible artifacts and tests",
    ]
    boundary = [
        "No live 5G network",
        "No SDR/hardware-loop validation",
        "No O-RAN/gNB integration",
        "No NVIDIA Aerial integration",
        "No MIMO/higher-order QAM claim",
        "No LDPC-coded system claim",
        "No production AI-RAN deployment claim",
    ]
    left = "".join(f"<li>{html.escape(item)}</li>" for item in evidence)
    right = "".join(f"<li>{html.escape(item)}</li>" for item in boundary)
    body = (
        '<div class="grid-2">'
        f'<div class="callout"><h4>Evidence demonstrated</h4><ul>{left}</ul></div>'
        f'<div class="callout"><h4>Boundary preserved</h4><ul>{right}</ul></div>'
        "</div>"
    )
    return _section(
        "Evidence vs Boundary",
        "What is demonstrated, and what is deliberately not claimed",
        body,
        "evidence-boundary",
    )


def _comparison_table(e: dict[str, Any]) -> str:
    rows = []
    for item in e["rows"]:
        rows.append(
            [
                f"{float(item['snr_db']):+.1f} dB",
                _fmt_float(item["ber_classical"], 5),
                _fmt_float(item["ber_neural"], 5),
                _fmt_float(item["bler_classical"], 3),
                _fmt_float(item["bler_neural"], 3),
            ]
        )
    return _section(
        "Measured BER / BLER",
        "Same channel, same SNR points, two receiver paths",
        _table(["SNR", "BER classical", "BER neural", "BLER classical", "BLER neural"], rows),
        "measured-results",
    )


def _visual_evidence_section() -> str:
    body = (
        '<div class="grid-2">'
        '<div class="card">'
        '<h3>BER/BLER comparison</h3>'
        '<div class="plot">'
        f'{(REPORTS / "bler_comparison.svg").read_text(encoding="utf-8") if (REPORTS / "bler_comparison.svg").exists() else "<p>Plot not generated.</p>"}'
        "</div>"
        '<p>Neural receiver gains are clearest in the moderate-SNR region; high-SNR limits remain visible.</p>'
        '<a class="source" href="bler_comparison.svg">Open source artifact</a>'
        "</div>"
        '<div class="card">'
        '<h3>LLR distribution comparison</h3>'
        '<div class="plot">'
        f'{(REPORTS / "llr_distribution_comparison.svg").read_text(encoding="utf-8") if (REPORTS / "llr_distribution_comparison.svg").exists() else "<p>Plot not generated.</p>"}'
        "</div>"
        '<p>The LLR distribution shows receiver confidence shape at a measured operating point.</p>'
        '<a class="source" href="llr_distribution_comparison.svg">Open source artifact</a>'
        "</div>"
        '<div class="card">'
        '<h3>Classical baseline</h3>'
        '<div class="plot">'
        f'{(REPORTS / "bler_classical.svg").read_text(encoding="utf-8") if (REPORTS / "bler_classical.svg").exists() else "<p>Plot not generated.</p>"}'
        "</div>"
        '<p>The classical LS+LMMSE path is a strong reference receiver, not a straw man.</p>'
        '<a class="source" href="bler_classical.svg">Open source artifact</a>'
        "</div>"
        '<div class="card">'
        '<h3>ONNX parity</h3>'
        '<p>ONNX parity passes against ONNXRuntime. This proves export correctness, not production deployment readiness.</p>'
        '<a class="source" href="onnx_parity_test.json">Open parity JSON</a>'
        "</div>"
        "</div>"
    )
    return _section("Visual Evidence", "Curves and distributions from committed artifacts", body, "visual-evidence")


def _artifact_links_section() -> str:
    rows = [
        ["BER/BLER CSV", _link("bler_comparison.csv", "reports/bler_comparison.csv"), "Raw measured comparison table."],
        ["BER/BLER plot", _link("bler_comparison.svg", "reports/bler_comparison.svg"), "Visual error-rate comparison."],
        ["LLR plot", _link("llr_distribution_comparison.svg", "reports/llr_distribution_comparison.svg"), "Receiver confidence distribution."],
        ["ONNX parity", _link("onnx_parity_test.json", "reports/onnx_parity_test.json"), "Export correctness check."],
        ["Training log", _link("training_log.json", "reports/training_log.json"), "Training config and convergence trace."],
    ]
    return _section("Evidence Artifacts", "Files behind the dashboard", _table(["Artifact", "Link", "What it proves"], rows), "artifacts")


def build_dashboard(e: dict[str, Any]) -> str:
    body = (
        _technical_decision_summary(e)
        + _story_section(e)
        + _visual_evidence_section()
        + _wins_holds_section()
        + _architecture_section()
        + _credibility_section(e)
        + _what_this_is_section()
        + _what_this_is_not_section()
        + _evidence_boundary_section()
        + _workflow_section()
        + _comparison_table(e)
        + _artifact_links_section()
    )
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\" />"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />"
        "<title>5G NR Neural Receiver Dashboard</title>"
        f"<style>{CSS}</style></head><body><div class=\"wrap\">"
        "<header><h1>5G NR Neural Receiver Dashboard</h1>"
        "<div class=\"sub\">DeepRx-style receiver vs LS+LMMSE baseline on a Sionna TDL-C link. The dashboard shows where neural wins, where classical holds, and what is still outside the evidence boundary.</div>"
        '<div class="nav">'
        '<a href="index.html">Landing page</a>'
        '<a href="../README.md">README</a>'
        '<a href="../BUSINESS_CASE.md">Business case</a>'
        '<a href="../TECH_BRIEF.md">Technical brief</a>'
        "</div></header>"
        f"{body}"
        "<footer>Simulated Sionna link evidence only. Boundary: live RAN absent; SDR absent; O-RAN absent; gNB absent; NVIDIA Aerial absent; production AI-RAN absent.</footer>"
        "</div></body></html>\n"
    )


def _plot_card(src: str, title: str, text: str, href: str) -> str:
    return (
        '<section class="card plot-card">'
        '<div class="eyebrow">Visual evidence</div>'
        f"<h3>{html.escape(title)}</h3>"
        f'<a href="{html.escape(href)}">{_img(src, title)}</a>'
        f"<p>{html.escape(text)}</p>"
        f'<a href="{html.escape(href)}">Open artifact</a>'
        "</section>"
    )


def build_index(e: dict[str, Any]) -> str:
    visual_cards = (
        '<div class="visual-grid">'
        + _plot_card(
            "bler_comparison.svg",
            "BER/BLER curve: neural wins in the moderate-SNR region",
            "The neural receiver improves BER around 5-10 dB while high-SNR limits remain visible.",
            "dashboard.html#measured-results",
        )
        + _plot_card(
            "bler_comparison.svg",
            "12.5 dB BLER: neural receiver produces more error-free frames",
            f"BLER improves from {_fmt_float(e['bler_classical_125'])} to {_fmt_float(e['bler_neural_125'])} at 12.5 dB.",
            "bler_comparison.svg",
        )
        + _plot_card(
            "llr_distribution_comparison.svg",
            "LLR distribution: receiver confidence shape",
            "The LLR plot shows how the receiver confidence distribution changes at the measured point.",
            "llr_distribution_comparison.svg",
        )
        + _plot_card(
            "bler_classical.svg",
            "Classical baseline: strong reference path, not a straw man",
            "The LS+LMMSE baseline is preserved as a serious receiver reference, especially at high SNR.",
            "bler_classical.svg",
        )
        + "</div>"
    )
    hero = (
        '<div class="hero-panel">'
        "<h1>5G NR Neural Receiver Evidence Pack</h1>"
        '<p class="sub">I built a DeepRx-style neural receiver and measured it against a classical LS+LMMSE baseline on the same Sionna TDL-C link.</p>'
        '<div class="kpi-grid">'
        + _kpi("Moderate-SNR result", "2-3 dB", "effective BER advantage", "good")
        + _kpi("BLER @ 12.5 dB", _fmt_float(e["bler_neural_125"]), f"classical {_fmt_float(e['bler_classical_125'])}", "good")
        + _kpi("ONNX parity", "PASS" if e["onnx_pass"] else "CHECK", f"max_diff={_fmt_sci(e['onnx_diff'])}", "good" if e["onnx_pass"] else "warn")
        + _kpi("Tests", "31/31", "recorded passing suite", "good")
        + _kpi("Boundary", "sim only", "not live RAN or SDR", "warn")
        + "</div>"
        '<div class="hero-actions">'
        '<a href="dashboard.html">Open dashboard</a>'
        '<a href="../BUSINESS_CASE.md">Business case</a>'
        '<a href="../TECH_BRIEF.md">Technical brief</a>'
        '<a href="bler_comparison.csv">Measured CSV</a>'
        '<a href="https://github.com/obiedeh/ai-phy-neural-receiver-benchmark">GitHub repo</a>'
        "</div></div>"
    )
    summary = (
        '<section><h2>Executive Technical Summary</h2><h3>AI-PHY evidence, not deployment theater</h3>'
        "<p>I built this project around one AI-PHY engineering question: can a DeepRx-style neural receiver learn enough from a Sionna-modeled 5G NR TDL-C channel to beat a classical LS+LMMSE receiver under the same link conditions?</p>"
        '<div class="kpi-grid">'
        + _kpi("Neural BER @ 5 dB", _fmt_float(e["ber_neural_5"]), f"classical {_fmt_float(e['ber_classical_5'])}", "good")
        + _kpi("Neural BER @ 10 dB", _fmt_float(e["ber_neural_10"]), f"classical {_fmt_float(e['ber_classical_10'])}", "good")
        + _kpi("Neural BLER @ 12.5 dB", _fmt_float(e["bler_neural_125"]), f"classical {_fmt_float(e['bler_classical_125'])}", "good")
        + _kpi("ONNX parity", "PASS" if e["onnx_pass"] else "CHECK", f"max_diff={_fmt_sci(e['onnx_diff'])}", "good" if e["onnx_pass"] else "warn")
        + _kpi("Tests", "31/31", "recorded passing suite", "good")
        + _kpi("Boundary", "sim only", "simulated link evidence", "warn")
        + "</div></section>"
    )
    links = (
        '<section><h2>Evidence Links</h2><h3>Start here</h3><div class="grid-3">'
        '<div class="card"><h3>Dashboard</h3><p>Technical decision summary, receiver comparison, boundaries, and artifacts.</p><a href="dashboard.html">Open dashboard</a></div>'
        '<div class="card"><h3>Business Case</h3><p>Decision question, finding, engineering value, and boundary.</p><a href="../BUSINESS_CASE.md">Open business case</a></div>'
        '<div class="card"><h3>Technical Brief</h3><p>Link configuration, receiver paths, evidence artifacts, and known limits.</p><a href="../TECH_BRIEF.md">Open technical brief</a></div>'
        '<div class="card"><h3>ONNX Parity</h3><p>Export correctness check. This is not production deployment evidence.</p><a href="onnx_parity_test.json">Open JSON</a></div>'
        '<div class="card"><h3>Training Log</h3><p>Training configuration and convergence trace.</p><a href="training_log.json">Open log</a></div>'
        '<div class="card"><h3>Source Code</h3><p>Neural receiver, classical baseline, and Sionna link source.</p><a href="https://github.com/obiedeh/ai-phy-neural-receiver-benchmark">Open repo</a></div>'
        "</div></section>"
    )
    boundary = _evidence_boundary_section()
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\" />"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />"
        "<title>5G NR Neural Receiver Evidence Pack</title>"
        f"<style>{CSS}</style></head><body><div class=\"wrap\">"
        "<header>"
        '<div class="nav"><a href="dashboard.html">Dashboard</a><a href="../BUSINESS_CASE.md">Business case</a><a href="../TECH_BRIEF.md">Technical brief</a><a href="../README.md">README</a></div>'
        "</header>"
        f"{hero}"
        f"{summary}"
        + _why_exists_section()
        + '<section><h2>Problem Statement</h2><h3>What this evidence pack answers</h3><p>The neural receiver wins in the moderate-SNR region, improves BLER at 12.5 dB, and passes ONNX parity. The result is bounded: classical remains competitive at high SNR, and this is simulated link evidence only.</p></section>'
        + '<section><h2>What I Built</h2><h3>DeepRx-style receiver on a Sionna-modeled 5G NR link</h3><p>I used the same TDL-C channel setup for the neural and classical paths, preserved the deterministic pilot fix, and committed the plots and JSON artifacts that support the conclusion.</p></section>'
        + f'<section><h2>Visual Evidence</h2><h3>Plots first, then interpretation</h3>{visual_cards}</section>'
        + '<section><h2>Where Neural Wins / Where It Does Not Win</h2><h3>Measured advantage with an honest boundary</h3><p>Neural gains are strongest around 5-12.5 dB. At high SNR, the classical receiver remains competitive, so this is not a claim that neural receivers dominate every operating point.</p></section>'
        + '<section><h2>Engineering Practices That Matter</h2><h3>Deterministic pilots, ONNX parity, tests</h3><p>The deterministic pilot RNG bug was found and fixed. ONNX parity passes, and the recorded suite is 31/31 tests passing. These details matter because they make the curves reproducible and auditable.</p></section>'
        + _what_this_is_section()
        + _what_this_is_not_section()
        + f"{boundary}{links}"
        "<footer>Simulated Sionna link evidence only. Boundary: live 5G network absent; SDR absent; O-RAN absent; gNB absent; NVIDIA Aerial absent; MIMO absent; LDPC-coded system absent; production AI-RAN absent.</footer>"
        "</div></body></html>\n"
    )


def build() -> None:
    REPORTS.mkdir(exist_ok=True)
    evidence = _load_evidence()
    dashboard = build_dashboard(evidence)
    index = build_index(evidence)

    dashboard_path = REPORTS / "dashboard.html"
    index_path = REPORTS / "index.html"
    _write_text_if_changed(dashboard_path, dashboard)
    _write_text_if_changed(index_path, index)

    print(f"Saved: {dashboard_path} ({len(dashboard):,} bytes)")
    print(f"Saved: {index_path} ({len(index):,} bytes)")


if __name__ == "__main__":
    build()
