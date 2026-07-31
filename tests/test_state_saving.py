"""Save-state options: listing, pruning, and the GUI wiring — headless, no GPU.

Covers the three Training-tab controls (save state at each checkpoint / at end of training /
keep last N) across BOTH families, plus the safety rails on pruning: a state dir is LoRA +
optimizer moments (~474 MB at rank 32), so pruning is load-bearing, and a prune that deleted
everything would take the state just written with it.

Run: venv/Scripts/python.exe tests/test_state_saving.py
"""
import os
import shutil
import sys
import tempfile

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = r"W:/Peter/Documents/Development/Fizgig"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import tkinter as tk
import lora_trainer_gui as G
from fizgig.training.train_utils import list_state_dirs, prune_state_dirs

G.LAST_USED_FILE = os.path.join(os.environ["TEMP"], "nope", ".last_used.json")

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


def _mk(base, name, contents=("lora.safetensors",)):
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    for f in contents:
        open(os.path.join(d, f), "w").close()
    return d


# --- 1. list_state_dirs ----------------------------------------------------------------------
BASE = tempfile.mkdtemp()
for e in (1, 2, 10, 3):
    _mk(BASE, f"mylora-{e:06d}-state")
_mk(BASE, "mylora-state")                 # the old un-numbered end-of-run name: must be ignored
_mk(BASE, "otherlora-000005-state")       # a DIFFERENT LoRA sharing the output dir
open(os.path.join(BASE, "mylora-000004.safetensors"), "w").close()   # a checkpoint, not a state

found = list_state_dirs(BASE, "mylora")
ck("list_state_dirs returns newest-first", [n for n, _ in found] == [10, 3, 2, 1], [n for n, _ in found])
ck("  ignores the un-numbered <name>-state", all("mylora-state" not in p for _, p in found))
ck("  ignores another LoRA's states in the same dir",
   all("otherlora" not in p for _, p in found))
ck("  ignores .safetensors checkpoints", len(found) == 4, len(found))
ck("  missing dir returns empty, does not raise", list_state_dirs(os.path.join(BASE, "nope"), "x") == [])

# --- 2. prune_state_dirs ---------------------------------------------------------------------
prune_state_dirs(BASE, "mylora", 2)
after = [n for n, _ in list_state_dirs(BASE, "mylora")]
ck("prune keeps the N highest-numbered", after == [10, 3], after)
ck("  never touches the other LoRA's state",
   os.path.isdir(os.path.join(BASE, "otherlora-000005-state")))
ck("  never touches checkpoints", os.path.exists(os.path.join(BASE, "mylora-000004.safetensors")))

# keep_n <= 0 must NOT empty the folder — that would delete the state just written, and the
# adaptive-LR sidecar would then recreate the dir holding nothing but a JSON, which resume picks
# up and chokes on.
for bad in (0, -5, "", None, "abc"):
    _mk(BASE, "clamped-000001-state")
    _mk(BASE, "clamped-000002-state")
    prune_state_dirs(BASE, "clamped", bad)
    left = [n for n, _ in list_state_dirs(BASE, "clamped")]
    ck(f"  keep_n={bad!r} clamps to 1 rather than deleting everything", left == [2], left)
    shutil.rmtree(os.path.join(BASE, "clamped-000002-state"), ignore_errors=True)

# An undeletable dir must be logged and skipped, not raise — this runs at an epoch boundary of a
# multi-hour run, and Windows AV holding a fresh .safetensors is a real occurrence.
_real_rmtree = shutil.rmtree
shutil.rmtree = lambda *a, **k: (_ for _ in ()).throw(PermissionError("held by AV"))
try:
    for e in (1, 2, 3):
        _mk(BASE, f"locked-{e:06d}-state")
    prune_state_dirs(BASE, "locked", 1)
    ck("prune survives an undeletable dir", True)
except Exception as e:
    ck("prune survives an undeletable dir", False, repr(e))
finally:
    shutil.rmtree = _real_rmtree

# --- 3. GUI: settings snapshot + both command builders ---------------------------------------
root = tk.Tk()
root.withdraw()
g = G.LoRATrainerGUI(root)

ck("checkbox defaults are on", g.save_state_var.get() and g.save_state_on_train_end_var.get())
ck("keep-last entry defaults to 2", g.entries["KEEP_LAST_N_STATES"].get() == "2",
   g.entries["KEEP_LAST_N_STATES"].get())


def flags_for(is_krea2, save_state, on_end, keep_n):
    g.settings["SAVE_STATE"] = save_state
    g.settings["SAVE_STATE_ON_TRAIN_END"] = on_end
    g.settings["KEEP_LAST_N_STATES"] = keep_n
    return g._state_flags()


both = flags_for(False, True, True, 2)
ck("both on -> both flags + keep-n",
   both == ["--save_state", "--save_state_on_train_end", "--keep_last_n_states", "2"], both)
ck("checkpoint-only omits the train-end flag",
   flags_for(False, True, False, 3) == ["--save_state", "--keep_last_n_states", "3"])
ck("train-end-only omits the checkpoint flag",
   flags_for(False, False, True, 3) == ["--save_state_on_train_end", "--keep_last_n_states", "3"])
ck("both off -> NO flags at all (pause still saves, trainer-side)",
   flags_for(False, False, False, 2) == [])
for bad in ("0", "-2", "", "  ", "abc"):
    got = flags_for(False, True, True, bad)
    ck(f"  keep_n={bad!r} never reaches the trainer as < 1", int(got[-1]) >= 1, got[-1])

# The flags must actually appear in each family's real command line. Use the REAL architecture
# configs — the Klein builder reads train_script off them, so a stubbed dict proves nothing.
KLEIN_CFG = next(c for n, c in G.ARCHITECTURES.items() if not c.get("is_krea2"))
KREA_CFG = next(c for n, c in G.ARCHITECTURES.items() if c.get("is_krea2"))

g.settings["SAVE_STATE"] = True
g.settings["SAVE_STATE_ON_TRAIN_END"] = True
g.settings["KEEP_LAST_N_STATES"] = 2
try:
    klein_cmd = g.build_training_command(KLEIN_CFG)
    ck("Klein command carries --save_state", "--save_state" in klein_cmd)
    ck("  and --save_state_on_train_end", "--save_state_on_train_end" in klein_cmd)
    ck("  and --keep_last_n_states 2",
       klein_cmd[klein_cmd.index("--keep_last_n_states") + 1] == "2")
    ck("  --save_state appears exactly once (was hardcoded before)",
       klein_cmd.count("--save_state") == 1, klein_cmd.count("--save_state"))
except Exception as e:
    ck("Klein command builds", False, repr(e))

try:
    krea_cmd = g.build_training_command(KREA_CFG)
    ck("Krea 2 command carries --save_state", "--save_state" in krea_cmd)
    ck("  and --save_state_on_train_end", "--save_state_on_train_end" in krea_cmd)
    ck("  and --keep_last_n_states 2",
       krea_cmd[krea_cmd.index("--keep_last_n_states") + 1] == "2")
except Exception as e:
    ck("Krea 2 command builds", False, repr(e))

# Both off: neither builder emits anything.
g.settings["SAVE_STATE"] = False
g.settings["SAVE_STATE_ON_TRAIN_END"] = False
for label, cfg in (("Klein", KLEIN_CFG), ("Krea 2", KREA_CFG)):
    try:
        cmd = g.build_training_command(cfg)
        ck(f"{label}: both off emits no state flags",
           not any(f in cmd for f in ("--save_state", "--save_state_on_train_end",
                                      "--keep_last_n_states")))
    except Exception as e:
        ck(f"{label}: both off builds", False, repr(e))

# --- 4. preset round-trip --------------------------------------------------------------------
g.save_state_var.set(False)
g.save_state_on_train_end_var.set(True)
g.entries["KEEP_LAST_N_STATES"].delete(0, tk.END)
g.entries["KEEP_LAST_N_STATES"].insert(0, "5")
preset = g._collect_preset_values()
ck("preset captures SAVE_STATE (BooleanVar, not in self.entries)",
   preset.get("SAVE_STATE") is False, preset.get("SAVE_STATE"))
ck("preset captures SAVE_STATE_ON_TRAIN_END", preset.get("SAVE_STATE_ON_TRAIN_END") is True)
ck("preset captures KEEP_LAST_N_STATES (Entry, generic sweep)",
   str(preset.get("KEEP_LAST_N_STATES")) == "5", preset.get("KEEP_LAST_N_STATES"))

g.save_state_var.set(True)
g.save_state_on_train_end_var.set(False)
g.entries["KEEP_LAST_N_STATES"].delete(0, tk.END)
g.entries["KEEP_LAST_N_STATES"].insert(0, "1")
g._apply_preset_values(preset)
ck("preset restores SAVE_STATE", g.save_state_var.get() is False)
ck("preset restores SAVE_STATE_ON_TRAIN_END", g.save_state_on_train_end_var.get() is True)
ck("preset restores KEEP_LAST_N_STATES", g.entries["KEEP_LAST_N_STATES"].get() == "5",
   g.entries["KEEP_LAST_N_STATES"].get())

# A preset saved before this feature must not clobber the current setting.
g.save_state_var.set(True)
g._apply_preset_values({"MAX_TRAIN_EPOCHS": 10})
ck("an older preset without the keys leaves SAVE_STATE alone", g.save_state_var.get() is True)

# --- 5. the resume-a-finished-run warning ----------------------------------------------------
_asked = []
G.messagebox.askyesno = lambda t, m: (_asked.append(m), True)[1]
g.entries["MAX_TRAIN_EPOCHS"].delete(0, tk.END)
g.entries["MAX_TRAIN_EPOCHS"].insert(0, "30")


def _set_resume(p):
    g.entries["RESUME_TRAINING"].delete(0, tk.END)
    g.entries["RESUME_TRAINING"].insert(0, p)


_set_resume("")
_asked.clear()
ck("no resume path -> no warning", g._confirm_resume_has_epochs_left() and not _asked)

_set_resume(r"C:\out\mylora-000010-state")
_asked.clear()
ck("state below max epochs -> no warning", g._confirm_resume_has_epochs_left() and not _asked)

_set_resume(r"C:\out\mylora-000030-state")
_asked.clear()
g._confirm_resume_has_epochs_left()
ck("state AT max epochs -> warns", len(_asked) == 1, _asked)

_set_resume(r"C:\out\mylora-000045-state")
_asked.clear()
g._confirm_resume_has_epochs_left()
ck("state PAST max epochs -> warns", len(_asked) == 1)

_set_resume(r"C:\out\some-hand-made-folder")
_asked.clear()
ck("unparseable resume path -> proceeds without warning",
   g._confirm_resume_has_epochs_left() and not _asked)

root.destroy()
shutil.rmtree(BASE, ignore_errors=True)
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
