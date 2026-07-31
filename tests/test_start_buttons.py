"""The Start tab's button row: right labels, right colours, working links.

A Tk button with a broken command builds perfectly and only fails when someone clicks it, so
this walks the widget tree and actually INVOKES each one with webbrowser.open stubbed. That is
the whole point: the failure mode being guarded against here is a dead button, not a missing one.

Run: venv/Scripts/python.exe tests/test_start_buttons.py
"""
import os
import sys
import tkinter as tk
import webbrowser

os.environ["FIZGIG_NO_PERSIST"] = "1"          # traced vars auto-save; never touch real prefs
REPO = r"W:/Peter/Documents/Development/Fizgig"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
os.chdir(REPO)
from lora_trainer_gui import LoRATrainerGUI      # noqa: E402

opened = []
webbrowser.open = lambda u, *a, **k: opened.append(u)

root = tk.Tk()
root.withdraw()
gui = LoRATrainerGUI(root)

buttons = []


def walk(w):
    for c in w.winfo_children():
        if isinstance(c, tk.Button):
            try:
                buttons.append((c.cget("text"), c.cget("bg"), c))
            except Exception:
                pass
        walk(c)


walk(root)

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


texts = [t for t, _, _ in buttons]

# --- the tutorial button on every tab ---------------------------------------------------------
ck("no 'Get help on YouTube' left anywhere", not any("Get help on YouTube" in t for t in texts))
tut = [b for b in buttons if b[0].endswith("Tutorial")]
ck("a Tutorial button on every tab", len(tut) >= 10, f"{len(tut)} found")
ck("  still YouTube red", all(bg == "#CC0000" for _, bg, _ in tut), {bg for _, bg, _ in tut})
ck("  still carries the play glyph", all(t.startswith("\u25b6") for t, _, _ in tut))

# --- the Start row ------------------------------------------------------------------------------
start_rp = [b for b in buttons if b[0].strip() == "\u26a1  Deploy on RunPod"]
ck("one Deploy on RunPod button on the Start row", len(start_rp) == 1, f"{len(start_rp)} found")
ck("  RunPod's violet, not a Fizgig accent", start_rp[0][1] == "#7C3AED", start_rp[0][1])

# The Preferences card carries its own deploy button (the desktop-side advert). Two in the app
# is correct; this pins that number so a third does not appear unnoticed.
all_rp = [b for b in buttons if "Deploy on RunPod" in b[0]]
ck("two app-wide: the Start row and the Preferences advert", len(all_rp) == 2, len(all_rp))

sibs = [c.cget("text") for c in start_rp[0][2].master.winfo_children() if isinstance(c, tk.Button)]
ck("row order is Tutorial / coffee / RunPod / About",
   [s.strip().split("  ")[-1] for s in sibs]
   == ["Tutorial", "Buy me a coffee", "Deploy on RunPod", "About"], sibs)

# --- every link actually fires -------------------------------------------------------------------
for want, frag in (("Tutorial", "youtube"),
                   ("Buy me a coffee", "buymeacoffee"),
                   ("\u26a1  Deploy on RunPod", "console.runpod.io")):
    b = [x for x in buttons if want in x[0]][0]
    opened.clear()
    b[2].invoke()
    ck(f"{want.strip()} opens {frag}", len(opened) == 1 and frag in opened[0], opened)

opened.clear()
start_rp[0][2].invoke()
ck("the RunPod link carries the referral tag",
   f"ref={gui.RUNPOD_REFERRAL}" in opened[0], opened[0])

root.destroy()
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
