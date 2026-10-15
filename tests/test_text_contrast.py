"""Explanatory text stays readable in the Fizgig 8 web UI — headless, no browser.

Users once reported the desktop Preferences tab as "grey on grey": the muted grey scored 2.54:1 on a card, failing
WCAG AA (4.5:1) and even the relaxed large-text bar (3:1). The fix moved prose onto its own brighter colour,
text_explain, and kept text_muted for genuinely de-emphasised bits (disabled controls, one-word captions beside a
widget). This file guarded that boundary in the desktop's Tk source.

Fizgig 8 is a web app, and the same palette lives as CSS custom properties in src/fizgig/web/static/style.css.
So the checks now read that stylesheet, resolving the cascade the way the browser does: a later :root block
overrides an earlier one, and the "web look" block (12 Oct) redefines the backgrounds and the secondary and muted
greys. What is pinned:
  1. text_explain clears 7:1 on both backgrounds, and still reads as secondary to headings;
  2. the prose classes (card descriptions, tab banners) are drawn in text_explain;
  3. the help under each form field (.field-hint, drawn in text_secondary) clears WCAG AA on a card. It measured
     6.35:1, below the desktop's 7:1 for prose but nowhere near the old 2.54:1. Upstream chose that tone in its
     web restyle, so the bar here is the accessibility standard, not a stricter house rule;
  4. text_muted (which fails AA for body text) is never the colour of any of that prose.

Retired with the desktop app: the Tk source sweeps (every tk.Label call's colour and point size), which have no
counterpart in a stylesheet.

Run: venv/Scripts/python.exe tests/test_text_contrast.py
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSS_PATH = os.path.join(REPO, "src", "fizgig", "web", "static", "style.css")
CSS = open(CSS_PATH, encoding="utf-8").read()

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


def tokens(css):
    """The colour custom properties as the browser resolves them: :root blocks in order, later ones win."""
    out = {}
    for block in re.findall(r":root\s*\{([^}]*)\}", css):
        for k, v in re.findall(r"--([a-z0-9-]+)\s*:\s*(#[0-9A-Fa-f]{6})\b", block):
            out[k] = v.upper()
    return out


def _lum(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def rule_color(selector):
    """The colour token a selector's LAST rule sets (later rules win), e.g. 'text-explain', or None."""
    found = None
    # Leaf blocks only ("sel { decls }"): a selector list cannot hold a brace, so a rule nested in @media matches
    # from after the @media's "{". Comments go first, since they may hold braces of their own.
    css = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        sels = [s.strip() for s in m.group(1).split(",")]
        if selector in sels:
            c = re.search(r"(?<![-\w])color\s*:\s*var\(--([a-z0-9-]+)\)", m.group(2))
            if c:
                found = c.group(1)
    return found


T = tokens(CSS)
ck("the palette tokens are all defined",
   all(k in T for k in ("bg-surface", "bg-deep", "text-primary", "text-secondary", "text-explain", "text-muted")),
   sorted(T))

# --- 1. text_explain ---------------------------------------------------------------------------
for bg in ("bg-surface", "bg-deep"):
    r = contrast(T["text-explain"], T[bg])
    ck(f"text-explain on {bg} clears 7:1", r >= 7.0, f"{r:.2f}")
r = contrast(T["text-primary"], T["text-explain"])
ck("text-explain still reads as secondary to headings", 1.2 <= r <= 2.0, f"{r:.2f}x")

# --- 2. the prose classes use it ---------------------------------------------------------------
for sel in (".card-desc", ".tab-banner p"):
    ck(f"{sel} is drawn in text-explain", rule_color(sel) == "text-explain", rule_color(sel))

# --- 3. the per-field help ---------------------------------------------------------------------
_hint = rule_color(".field-hint")
ck(".field-hint has a palette colour", _hint in T, _hint)
if _hint in T:
    r = contrast(T[_hint], T["bg-surface"])
    ck(f"  .field-hint ({_hint}) clears WCAG AA (4.5:1) on a card", r >= 4.5, f"{r:.2f}")

# --- 4. text_muted stays off prose -------------------------------------------------------------
r = contrast(T["text-muted"], T["bg-surface"])
ck("text-muted is still the muted tier (below AA, so it must never carry prose)", r < 4.5, f"{r:.2f}")
_prose = (".card-desc", ".tab-banner p", ".field-hint")
ck("  no prose class is drawn in text-muted", all(rule_color(s) != "text-muted" for s in _prose),
   {s: rule_color(s) for s in _prose})

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
