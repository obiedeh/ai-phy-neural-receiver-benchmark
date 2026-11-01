"""Enforce the canonical repository prose policy without rewriting measured records.

The policy is linked from AGENTS.md. Tracked-prose selection and measured-record
exclusion follow the existing repository hygiene guards in the product suites.
Local private terms are optional in CI and are never included in diagnostics.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROSE_SUFFIXES = {".md", ".html", ".rst", ".txt", ".log"}
HOME_PATHS = re.compile(r"/(?:home|Users)/[^/\s]+(?:/|\b)|[A-Za-z]:\\Users\\[^\\\s]+", re.I)
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
LOCAL_HOST = re.compile(r"\b[a-z0-9][a-z0-9-]*\.local\b", re.I)
ATTRIBUTION = re.compile(
    r"\b(?:" + "|".join(("clau" + "de", "co" + "dex", "chat" + "gpt",
                         "anthro" + "pic", "github " + "copilot",
                         "co-authored" + "-by")) + r")\b|(?m:^generated[- ]by:)",
    re.I,
)
HANDOFF_NAMES = re.compile(r"(?:^|/)(?:handoff[^/]*\.md|[^/]*handoff\.md)$", re.I)


def _text_violations(text: str, private_terms: tuple[str, ...] = ()) -> set[str]:
    """Return categories only, preserving privacy even when a test fails."""
    found: set[str] = set()
    if HOME_PATHS.search(text):
        found.add("home path")
    for value in IPV4.findall(text):
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            continue
        octets = tuple(address.packed)
        if (octets[0] == 10 or octets[:2] == (192, 168)
                or (octets[0] == 172 and 16 <= octets[1] <= 31)):
            found.add("LAN address")
    if LOCAL_HOST.search(text):
        found.add("local host")
    if ATTRIBUTION.search(text):
        found.add("attribution")
    if "\u2014" in text:
        found.add("em dash")
    folded = text.casefold()
    if any(term.casefold() in folded for term in private_terms):
        found.add("private term")
    return found


def _tracked(root: Path = ROOT) -> list[Path]:
    """Read the index, failing rather than silently skipping a broken checkout."""
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    )
    return [Path(value) for value in result.stdout.decode().split("\0") if value]


def _private_terms() -> tuple[str, ...]:
    """Use a local policy list when present; portable CI needs no private file."""
    path = Path.home() / ".config" / "git" / "private-terms.txt"
    if not path.exists():
        return ()
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _prose_findings(root: Path, paths: list[Path], terms: tuple[str, ...]) -> list[str]:
    """Report path hashes and violation categories, never source lines or terms."""
    findings: list[str] = []
    for path in paths:
        if path.suffix.lower() not in PROSE_SUFFIXES:
            continue
        categories = _text_violations((root / path).read_text(encoding="utf-8"), terms)
        if categories:
            key = hashlib.sha256(path.as_posix().encode()).hexdigest()
            findings.append(f"{key}: {', '.join(sorted(categories))}")
    return findings


def test_tracked_prose_follows_repository_policy() -> None:
    """Guard all tracked prose while leaving measured JSON and CSV untouched."""
    assert _prose_findings(ROOT, _tracked(), _private_terms()) == []


def test_handoff_notes_are_outside_the_repository() -> None:
    """Reject handoff file names without exposing identifying file names."""
    offending = [path for path in _tracked() if HANDOFF_NAMES.search(path.as_posix())]
    assert len(offending) == 0


def test_current_commit_message_follows_repository_policy() -> None:
    """Check the current message without requiring full history in shallow CI."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%B"], cwd=ROOT, check=True, capture_output=True
    )
    assert _text_violations(result.stdout.decode(), _private_terms()) == set()


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("/ho" + "me/example/data", "home path"),
        ("/Us" + "ers/example/data", "home path"),
        ("C:" + "\\Us" + "ers\\example\\data", "home path"),
        ("peer " + ".".join(("10", "1", "2", "3")), "LAN address"),
        ("peer " + ".".join(("172", "31", "2", "3")), "LAN address"),
        ("peer " + ".".join(("192", "168", "2", "3")), "LAN address"),
        ("example" + ".local", "local host"),
        ("co" + "dex", "attribution"),
        ("co-authored" + "-by: contributor", "attribution"),
        ("Generated" + "-by: service", "attribution"),
        ("left" + chr(0x2014) + "right", "em dash"),
    ],
)
def test_rejects_prohibited_prose(text: str, category: str) -> None:
    """Exercise each privacy boundary with constructed, non-identifying input."""
    assert category in _text_violations(text)


@pytest.mark.parametrize(
    "text",
    [
        "jetson-agx-thor; jetson-orin-nx; rtx-5090-workstation",
        "Run from ~/project and store records under /mnt/backup.",
        "Public endpoint 8.8.8.8, loopback 127.0.0.1, version 1.2.3.4.",
        "Non-private endpoint 172.32.0.1 and invalid address 999.1.1.1.",
        "Synthetic records generated by the deterministic fixture generator.",
    ],
)
def test_accepts_portable_prose(text: str) -> None:
    """Keep board labels, portable commands and public values usable."""
    assert _text_violations(text) == set()


def test_optional_private_terms_are_case_insensitive() -> None:
    """Prove local list enforcement using a synthetic token, never real terms."""
    assert _text_violations("SYNTHETIC_TOKEN", ("synthetic_token",)) == {"private term"}
    assert _text_violations("SYNTHETIC_TOKEN") == set()


def test_prose_scan_preserves_measured_records(tmp_path: Path) -> None:
    """Exercise real file selection without editing measured record contents."""
    content = ("/ho" + "me/example/data").encode()
    record = tmp_path / "measurement.json"
    record.write_bytes(content)
    prose = tmp_path / "README.md"
    prose.write_text("Use board labels and ~/project.\n", encoding="utf-8")
    assert _prose_findings(tmp_path, [Path(record.name), Path(prose.name)], ()) == []
    assert record.read_bytes() == content
    prose.write_bytes(content)
    findings = _prose_findings(tmp_path, [Path(prose.name)], ())
    assert len(findings) == 1
    assert "home path" in findings[0]
    assert content.decode() not in findings[0]


@pytest.mark.parametrize("path", ["HANDOFF.md", "docs/handoff-next.md"])
def test_rejects_handoff_paths(path: str) -> None:
    """Ensure filename enforcement covers root and nested handoff notes."""
    assert HANDOFF_NAMES.search(path)
