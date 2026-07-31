"""Explanatory text stays readable — headless, no GPU.

Users reported the Preferences tab as "grey on grey". Measured, the muted grey scored 2.54:1 on a
card, failing WCAG AA (4.5) and even the relaxed large-text bar (3.0). This locks in the fix and,
more usefully, guards the boundary: prose belongs on text_explain, while genuinely de-emphasised
UI (disabled controls, one-word captions beside widgets) stays on text_muted. A future "tidy-up"
that collapses the two back together should fail here.

Run: venv/Scripts/python.exe tests/test_text_contrast.py
"""
import io
import os
import re
import sys

REPO = r"W:/Peter/Documents/Development/Fizgig"
sys.path.insert(0, REPO)

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


def _lum(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(fg, bg):
    a, b = sorted([_lum(fg), _lum(bg)], reverse=True)
    return (a + 0.05) / (b + 0.05)


os.environ["FIZGIG_NO_PERSIST"] = "1"
import lora_trainer_gui as G  # noqa: E402

C = G.COLORS

# --- 1. contrast ------------------------------------------------------------------------------
for bg_name in ("bg_surface", "bg_deep"):
    r = contrast(C["text_explain"], C[bg_name])
    ck(f"text_explain on {bg_name} clears 7:1", r >= 7.0, f"{r:.2f}")

ck("text_explain still reads as secondary to headings",
   1.2 <= contrast(C["text_primary"], C["text_explain"]) <= 2.0,
   f"{contrast(C['text_primary'], C['text_explain']):.2f}x")

# The old value is deliberately still in the palette for disabled/caption use — it is only
# unreadable when used for prose, which is what this whole change is about.
ck("text_muted unchanged (still the muted tier)", C["text_muted"] == "#5A6B7E", C["text_muted"])

# --- 2. the sweep was complete, and no wider than intended ------------------------------------
SRC = io.open(os.path.join(REPO, "lora_trainer_gui.py"), encoding="utf-8").read()


def label_calls(src, needle="tk.Label("):
    out, i = [], 0
    while True:
        j = src.find(needle, i)
        if j < 0:
            break
        d, k = 0, j + len(needle) - 1
        while k < len(src):
            if src[k] == "(":
                d += 1
            elif src[k] == ")":
                d -= 1
                if d == 0:
                    break
            k += 1
        out.append(src[j:k + 1])
        i = k + 1
    return out


calls = label_calls(SRC)
muted = [c for c in calls if "text_muted" in c]
prose_left = [c for c in muted if "wraplength" in c]

# wraplength is the tell: you only set it on multi-line explanatory copy. Widget captions
# ("seed", "W", "H", "Ref") never do.
# The detail has to survive text= being a VARIABLE, not a literal — the one real failure this
# ever caught was `text=line`, and the regex returning None turned a legible FAIL into a
# traceback that hid which label was at fault.
def _snippet(c):
    m = re.search(r'text="([^"]{0,40})', c)
    return m.group(1) if m else " ".join(c.split())[:60]


ck("no prose label left on text_muted", len(prose_left) == 0,
   [_snippet(c) for c in prose_left[:3]] if prose_left else "")
# Lower bound, not equality: the point is that the sweep did not drag captions onto the bright
# tier, and new one-word captions get added over time. A DROP here means something was swept up.
ck("widget captions were NOT swept up", len([c for c in muted if "wraplength" not in c]) >= 93,
   len([c for c in muted if "wraplength" not in c]))
ck("explanatory labels moved across", len([c for c in calls if "text_explain" in c]) >= 34,
   len([c for c in calls if "text_explain" in c]))

# --- 3. the helpers that render most of it ----------------------------------------------------
for helper, marker in (("_start_section_card", "wraplength=760"),
                       ("_add_pref_row", "hint_text"),
                       ("_add_fetch_models_row", "blurb")):
    i = SRC.find(f"def {helper}")
    body = SRC[i:i + 3000] if i >= 0 else ""
    ck(f"{helper} renders prose on text_explain",
       "text_explain" in body and "text_muted" not in body.split("def ")[1][:2000])

# --- 4. sizes went up a point, none left at the old smallest --------------------------------
explain_fonts = re.findall(r'font=\(FONT_FAMILY, (\d+)(?:, "([a-z]+)")?\)[^)]*?text_explain'
                           r'|text_explain[^)]*?font=\(FONT_FAMILY, (\d+)(?:, "([a-z]+)")?\)',
                           SRC, re.S)
sizes = {int(a or c) for a, _b, c, _d in explain_fonts if (a or c)}
ck("no explanatory text left at 8pt", 8 not in sizes, sorted(sizes))

# --- 5. it all still builds -------------------------------------------------------------------
import tkinter as tk  # noqa: E402

G.LAST_USED_FILE = os.path.join(os.environ["TEMP"], "nope", ".last_used.json")
root = tk.Tk()
root.withdraw()
try:
    g = G.LoRATrainerGUI(root)
    ck("GUI builds with every tab", len(g.notebook.tabs()) >= 11, len(g.notebook.tabs()))
except Exception as e:
    ck("GUI builds with every tab", False, repr(e))
root.destroy()

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
