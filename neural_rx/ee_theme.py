# ruff: noqa: E501
"""EmbodiedEdge Labs evidence theme.

Applies the lab's look (embodiededge.ai) to a self-contained HTML evidence
page: a shared header band and footer, the lab palette and type mapped onto
the page's own CSS variables, dark by default with a light print version.
Styling only: no text or number on the page is changed, and nothing is
loaded from the network (web-font links are removed; the lab uses system
fonts).

Usage:
    from ee_theme import apply_theme
    html = apply_theme(html, repo_url="https://github.com/obiedeh/<repo>",
                       dark={"--bg": "#202224", ...}, light={...})
"""
from __future__ import annotations

import re

LAB_URL = "https://embodiededge.ai"
PROFILE_URL = "https://github.com/obiedeh"

SANS = '"Helvetica Neue",Helvetica,Arial,sans-serif'
MONO = 'ui-monospace,"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace'

# The lab palette (embodiededge.ai). Field = real hardware, sim = simulation.
LAB = {
    "bg": "#202224", "card": "#181b1d", "raised": "#1c1f21", "line": "#393d3f", "line_strong": "#4a4f52",
    "text": "#eef1e8", "text_2": "#c9cfc2", "muted": "#a3aa9c", "dim": "#7c8378",
    "field": "#b7f34a", "sim": "#68b7ff", "signal": "#ff9c59", "alert": "#ff7a6b",
}
LAB_LIGHT = {
    "bg": "#f6f7f3", "card": "#ffffff", "raised": "#fbfbf8", "line": "#d8dbd2", "line_strong": "#b9bdb3",
    "text": "#1a1c1e", "text_2": "#3d4239", "muted": "#5d6459", "dim": "#7c8378",
    "field": "#4d7c0f", "sim": "#1f6fd1", "signal": "#c2560c", "alert": "#c62828",
}

# Headings in the lab's register: medium weight, tight tracking.
TYPE_CSS = f"h1,h2,h3{{font-family:{SANS};font-weight:500;letter-spacing:-.02em}}h1{{letter-spacing:-.035em}}"

FIELD, SIM = "#b7f34a", "#68b7ff"  # lab accents: green marks the real result, blue the method or test


def type_css(eyebrow: str = ".ee-eyebrow", kicker: str = "", title2: str = "h2", title3: str = "h3") -> str:
    """One type scale for every EmbodiedEdge Labs page (the site case studies use the same sizes).

    eyebrow: label above the page title (green, mono, caps). kicker: label above a section
    title (blue, mono, caps). title2/title3: the elements that act as section and card titles.
    """
    label = f"font-family:{MONO}!important;font-size:12px!important;font-weight:600!important;letter-spacing:.08em!important;text-transform:uppercase!important;line-height:1.4!important"
    css = (f"body{{font-family:{SANS}}}"
           f"h1{{font-family:{SANS}!important;font-size:clamp(44px,4.6vw,66px)!important;font-weight:500!important;"
           f"line-height:1!important;letter-spacing:-.06em!important;text-transform:none!important;font-stretch:normal!important}}"
           f"h1 em,h1 .ee-g{{font-style:normal;color:{FIELD}!important;-webkit-text-fill-color:{FIELD}}}"
           f"h1 em.b,h1 .ee-b{{font-style:normal;color:{SIM}!important;-webkit-text-fill-color:{SIM}}}"
           f"{title2}{{font-family:{SANS}!important;font-size:clamp(28px,3.2vw,42px)!important;font-weight:500!important;"
           f"line-height:1.08!important;letter-spacing:-.04em!important;text-transform:none!important;font-stretch:normal!important;color:inherit}}"
           f"{title3}{{font-family:{SANS}!important;font-size:19px!important;font-weight:600!important;line-height:1.25!important;"
           f"letter-spacing:-.02em!important;text-transform:none!important;font-stretch:normal!important}}"
           f"{eyebrow}{{{label};color:{FIELD}!important}}")
    if kicker:
        css += f"{kicker}{{{label};color:{SIM}!important}}"
    return css


def accent_h1(html: str, accents: list[tuple[str, str]]) -> str:
    """Color phrases inside the first <h1> without changing its text: ("phrase", "g"|"b")."""
    m = re.search(r"<h1\b[^>]*>(.*?)</h1>", html, re.DOTALL)
    if not m or "ee-g" in m.group(1) or "ee-b" in m.group(1):
        return html
    inner = m.group(1)
    for phrase, tone in accents:
        if phrase in inner:
            inner = inner.replace(phrase, f'<span class="ee-{tone}">{phrase}</span>', 1)
    return html[:m.start(1)] + inner + html[m.end(1):]

BAND_CSS = f"""
body{{margin:0}}
.ee-band{{background:#181b1d;border-bottom:1px solid #393d3f;color:#eef1e8;font:14px/1.4 {SANS};position:relative;z-index:50}}
.ee-band .ee-in{{max-width:1120px;margin:0 auto;padding:10px 20px;display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;justify-content:space-between}}
.ee-band a{{color:#a3aa9c;text-decoration:none}}.ee-band a:hover{{color:#eef1e8}}
.ee-band .ee-mark{{display:inline-flex;align-items:center;gap:8px;color:#eef1e8;font-weight:600;letter-spacing:-.01em}}
.ee-band .ee-mark i{{display:inline-block;width:14px;height:14px;border-radius:2px;background:linear-gradient(135deg,#b7f34a 0 50%,#68b7ff 50% 100%)}}
.ee-band .ee-mark span{{color:#a3aa9c;font-weight:400}}.ee-band nav{{display:flex;gap:16px}}
.ee-foot{{border-top:1px solid #393d3f;background:#181b1d;color:#7c8378;font:13px/1.5 {SANS}}}
.ee-foot .ee-in{{max-width:1120px;margin:0 auto;padding:14px 20px}}.ee-foot a{{color:#a3aa9c}}
@media print{{.ee-band,.ee-foot{{background:#fff;color:#5d6459;border-color:#d8dbd2}}.ee-band .ee-mark{{color:#1a1c1e}}}}
"""


def _vars(tokens: dict[str, str]) -> str:
    return "".join(f"{k}:{v};" for k, v in tokens.items())


def band_html(repo_url: str, label: str = "Labs · Evidence") -> str:
    return (f'<div class="ee-band"><div class="ee-in"><a class="ee-mark" href="{LAB_URL}"><i></i>EmbodiedEdge '
            f'<span>{label}</span></a><nav><a href="{repo_url}">Repository</a><a href="{LAB_URL}">Lab</a>'
            f'<a href="{PROFILE_URL}">Obinna Edeh</a></nav></div></div>')


def foot_html() -> str:
    return (f'<div class="ee-foot"><div class="ee-in">An <a href="{LAB_URL}">EmbodiedEdge Labs</a> project. '
            'Measured on real hardware, published as found.</div></div>')


def apply_theme(html: str, *, repo_url: str, dark: dict[str, str], light: dict[str, str] | None = None,
                extra_css: str = "", root_selectors: str = ":root,:root:not([data-theme=\"light\"]),:root[data-theme=\"dark\"]",
                force_dark: bool = True, scheme: str = "dark", typeset: dict | None = None) -> str:
    """Return html with the lab theme applied. Idempotent: an already themed
    page is returned with its theme block refreshed, not duplicated.

    ``dark`` holds the default tokens; ``scheme`` names their color scheme
    (pass "light" with LAB_LIGHT tokens for a light page). ``typeset`` turns on the
    shared type scale: {"eyebrow", "kicker", "title2", "title3"} selectors, "accents"
    for the page title and "eyebrow_text" for pages without a label above the title."""
    html = re.sub(r'\s*<style id="ee-theme">.*?</style>', "", html, flags=re.DOTALL)
    html = re.sub(r'<div class="ee-band">.*?</div></div>', "", html, count=1, flags=re.DOTALL)
    html = re.sub(r'<div class="ee-foot">.*?</div></div>', "", html, count=1, flags=re.DOTALL)
    # web fonts: the lab uses system fonts, and the page must not fetch anything
    html = re.sub(r'\s*<link[^>]+(?:fonts\.googleapis\.com|fonts\.gstatic\.com)[^>]*>', "", html)
    css = f"{root_selectors}{{{_vars(dark)}color-scheme:{scheme}}}" + TYPE_CSS
    if light:
        css += f':root[data-theme="light"]{{{_vars(light)}color-scheme:light}}'
        css += f"@media print{{{root_selectors}{{{_vars(light)}color-scheme:light}}}}"
    if typeset is not None:
        opts = {k: v for k, v in typeset.items() if k in ("eyebrow", "kicker", "title2", "title3")}
        css += type_css(**opts)
        if typeset.get("accents"):
            html = accent_h1(html, typeset["accents"])
        if typeset.get("eyebrow_text") and 'class="ee-eyebrow"' not in html:
            html = re.sub(r"(<h1\b)", f'<div class="ee-eyebrow">{typeset["eyebrow_text"]}</div>\\1', html, count=1)
    css += BAND_CSS + extra_css
    style = f'<style id="ee-theme">{css}</style>'
    # declare UTF-8 first thing in the document, so GitHub Pages and every browser agree on it
    html = html.replace('<meta charset="utf-8">', "", 1) if html.count('<meta charset="utf-8">') == 1 else html
    if not re.search(r"<meta[^>]+charset", html, re.IGNORECASE):
        anchor = re.search(r"<head(?:\s[^>]*)?>", html) or re.match(r"\s*<!doctype[^>]*>", html, re.IGNORECASE)
        pos = anchor.end() if anchor else 0
        html = html[:pos] + '<meta charset="utf-8">' + html[pos:]
    m = re.search(r"<body[^>]*>", html)
    if m:  # a page with head and body: theme last in the head, band first in the body
        html = html.replace("</head>", style + "</head>", 1) if "</head>" in html else html
        body = re.search(r"<body[^>]*>", html)
        if body:
            html = html[:body.end()] + band_html(repo_url) + html[body.end():]
    else:  # a page without head/body tags: theme and band just before the first content block
        c = re.search(r"<(?:div|main|header|section|nav|article)\b", html)
        pos = c.start() if c else len(html)
        html = html[:pos] + style + band_html(repo_url) + html[pos:]
    if force_dark:
        html = re.sub(r"<html(?![^>]*data-theme)([^>]*)>", r'<html\1 data-theme="dark">', html, count=1)
    end = html.rfind("</body>")
    html = (html[:end] + foot_html() + html[end:]) if end != -1 else html + foot_html()
    return html
