"""LoKR on the Training tab — headless GUI verification (no GPU, no model load).

Covers the Phase 3 wiring: the Network Type control exists only under Krea 2, LoKR swaps the
rank/alpha rows for the Factor dial, the command builder emits the flags, and both keys ride
the preset/persistence sweep.
"""
import os
import sys

os.environ["FIZGIG_NO_PERSIST"] = "1"
# Repo root, derived from this file's location -- was hardcoded to one machine's
# absolute path, which made the whole suite unrunnable anywhere else.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import tkinter as tk  # noqa: E402
import lora_trainer_gui as G  # noqa: E402

G.LAST_USED_FILE = os.path.join(os.environ["TEMP"], "nope", ".last_used.json")

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


def _visible(w):
    try:
        return bool(w.winfo_manager())
    except tk.TclError:
        return False


root = tk.Tk()
root.withdraw()
g = G.LoRATrainerGUI(root)

# Every family trains through the driver system now, and every one offers LoKR: the description's
# network_types is ("lora", "lokr") across Klein, Krea 2, MiniMax H3, Qwen 2.1, SDXL and Anima. So the
# old premise -- "Klein trains standard only, Network Type hidden there" -- is no longer behaviour to
# protect, and the "is_krea2" flag, _build_krea2_train_command, KREA2_BUILT_IN_PRESETS and the Krea 2
# fine-tune toggles it drove are gone. What follows pins the same wiring where it lives now: the shared
# widgets, the launch plan's command, and the family descriptions' presets. Families are picked by
# description key rather than display name, which upstream renames.
from fizgig.families import launch as L  # noqa: E402
from fizgig.families.registry import training_families  # noqa: E402


def arch_for(key):
    return next(a for a in G.ARCHITECTURES if getattr(g._family_desc(a), "key", None) == key)


# --- 1. widgets + visibility --------------------------------------------------------------
ck("NETWORK_TYPE and LOKR_FACTOR widgets exist",
   "NETWORK_TYPE" in g.entries and "LOKR_FACTOR" in g.entries)
ck("  default is standard LoRA (LoKR one pick away)",
   g.entries["NETWORK_TYPE"].get() == "LoRA (standard)")
ck("  default factor is 8", g.entries["LOKR_FACTOR"].get() == "8",
   g.entries["LOKR_FACTOR"].get())

# The combo/entry are packed inside row frames (widget + hint side by side), so the frames are what
# get shown/hidden -- check those. Every training family offers LoKR, so every one shows the row.
for d in training_families():
    g.architecture_var.set(arch_for(d.key))
    g.update_ui_for_architecture()
    g.entries["NETWORK_TYPE"].set("LoRA (standard)")
    g._on_network_type_changed()
    root.update()
    ck(f"{d.key}: Network Type shown, standard -> rank/alpha shown, factor hidden",
       _visible(g._network_type_rowf) and _visible(g.entries["NETWORK_DIM"])
       and not _visible(g._lokr_factor_rowf))
    g.entries["NETWORK_TYPE"].set("LoKR (Kronecker)")
    g._on_network_type_changed()
    root.update()
    ck(f"  {d.key}: LoKR -> factor row shown, rank/alpha hidden",
       _visible(g._lokr_factor_rowf) and not _visible(g.entries["NETWORK_DIM"])
       and not _visible(g.entries["NETWORK_ALPHA"]))

# --- 2. the launch plan's command ---------------------------------------------------------
K2 = g._family_desc(arch_for("krea2"))
g.architecture_var.set(arch_for("krea2"))
g.update_ui_for_architecture()
root.update()


def k2_cmd(**st):
    g.settings.update(st)
    return L.train_command(K2, g._family_launch_inputs(K2), L.LaunchPlan())


cmd = k2_cmd(NETWORK_TYPE="LoKR (Kronecker)", LOKR_FACTOR=16, FAMILY_SLIDER=False)
ck("LoKR -> command carries --network_type lokr --lokr_factor 16",
   "--network_type" in cmd and cmd[cmd.index("--network_type") + 1] == "lokr"
   and cmd[cmd.index("--lokr_factor") + 1] == "16")
cmd = k2_cmd(NETWORK_TYPE="LoRA (standard)")
ck("standard LoRA -> no network_type flag at all", "--network_type" not in cmd)
# The launch plan's own exclusion: a slider is always a plain LoRA (its strength is the dial), so
# LoKR must not reach a slider's command even when it is the selected Network Type.
_inp = dict(g._family_launch_inputs(K2), NETWORK_TYPE="LoKR (Kronecker)", LOKR_FACTOR=16,
            FAMILY_SLIDER=True)
ck("slider on -> LoKR never reaches the command (a slider is a plain LoRA)",
   L.slider_on(K2, _inp) and "--network_type" not in L.train_command(K2, _inp, L.LaunchPlan()))
# Retired: "fine-tune on -> --network_type not emitted". Fine-tune moved into the driver's trainer,
# which never builds a trainable adapter on a fine-tune (families/train.py adds it only outside the
# FT branch), so the flag is inert there by construction rather than suppressed in the command.

# --- 3. persistence sweep + built-in presets ----------------------------------------------
vals = g._collect_preset_values()
ck("preset sweep captures NETWORK_TYPE + LOKR_FACTOR",
   "NETWORK_TYPE" in vals and "LOKR_FACTOR" in vals,
   {k: vals.get(k) for k in ("NETWORK_TYPE", "LOKR_FACTOR")})
ck("NETWORK_TYPE is strict-combo protected (junk can't be .set() onto it)",
   "NETWORK_TYPE" in G.LoRATrainerGUI._STRICT_COMBO_KEYS)
# The built-in presets moved into each family's description. The invariant was Krea 2-only; it now
# holds for every family, so check them all.
_presets = [(d.key, n, pr) for d in training_families() for n, pr in d.presets]
ck("every family ships built-in presets", len(_presets) > 0 and all(d.presets for d in training_families()))
_bad = [f"{k}: {n}" for k, n, pr in _presets if pr.get("NETWORK_TYPE") != "LoRA (standard)"]
ck(f"  all {len(_presets)} built-in presets pin standard LoRA", not _bad, _bad[:3])

# Applying a built-in preset resets a LoKR selection back to standard.
g.entries["NETWORK_TYPE"].set("LoKR (Kronecker)")
g._apply_preset_values(dict(K2.presets[0][1]))
root.update()
ck("loading a built-in preset resets Network Type to standard",
   g.entries["NETWORK_TYPE"].get() == "LoRA (standard)", g.entries["NETWORK_TYPE"].get())

root.destroy()
print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
