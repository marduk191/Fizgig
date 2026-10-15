"""Save-state options: listing, pruning, and the form wiring — headless, no GPU.

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
# Repo root, derived from this file's location -- was hardcoded to one machine's
# absolute path, which made the whole suite unrunnable anywhere else.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# src only. The repo root holds Fizgig 8's fizgig.py, which would shadow the fizgig package.
sys.path.insert(0, os.path.join(REPO, "src"))

from fizgig.training.train_utils import list_state_dirs, prune_state_dirs
from fizgig.families import form as FORM
from fizgig.families import launch as L
from fizgig.families.registry import get as get_family

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

# --- 3. the form: defaults + both families' commands ----------------------------------------------
# Fizgig 8 replaced the desktop Training tab with families/form.py: the same controls as data (the web UI renders
# them), and form.inputs feeding the same launch plan. So these drive form -> inputs -> launch.train_command, the
# path the web app's Start takes (server._plan), for each family.


def rows(desc):
    return {r["key"]: r for sec in FORM.form(desc)["sections"] for r in sec["rows"] if r.get("key")}


def defaults(desc):
    return {k: r.get("default") for k, r in rows(desc).items()}


def family_cmd(key, **values):
    desc = get_family(key)
    t = tempfile.mkdtemp(dir=BASE)
    vals = dict(defaults(desc), LORA_NAME="t", LORA_OUTPUT_DIR=t, DATASET_CONFIG=os.path.join(t, "d.toml"), **values)
    inp = FORM.inputs(desc, vals, models={k: "m" for k in FORM.model_keys(desc)}, python=sys.executable,
                      repo_dir=REPO, image_folder="", cache_root=t, samples_dir=t)
    return L.train_command(desc, inp, L.LaunchPlan())


K2 = get_family("krea2")
r = rows(K2)
ck("checkbox defaults are on", r["SAVE_STATE"].get("default") is True
   and r["SAVE_STATE_ON_TRAIN_END"].get("default") is True)
ck("keep-last entry defaults to 2", str(r["KEEP_LAST_N_STATES"].get("default")) == "2",
   r["KEEP_LAST_N_STATES"].get("default"))
ck("  the three are real checkboxes / an entry, not free text",
   r["SAVE_STATE"]["kind"] == "check" and r["SAVE_STATE_ON_TRAIN_END"]["kind"] == "check"
   and r["KEEP_LAST_N_STATES"]["kind"] == "entry")

for label, key in (("Klein", "klein"), ("Krea 2", "krea2")):
    try:
        cmd = family_cmd(key, SAVE_STATE=True, SAVE_STATE_ON_TRAIN_END=True, KEEP_LAST_N_STATES="2")
        ck(f"{label} command carries --save_state", "--save_state" in cmd)
        ck("  and --save_state_on_train_end", "--save_state_on_train_end" in cmd)
        ck("  and --keep_last_n_states 2", cmd[cmd.index("--keep_last_n_states") + 1] == "2")
        ck("  --save_state appears exactly once (was hardcoded before)",
           cmd.count("--save_state") == 1, cmd.count("--save_state"))
        cmd = family_cmd(key, SAVE_STATE=False, SAVE_STATE_ON_TRAIN_END=False)
        ck(f"{label}: both off emits no state flags",
           not any(f in cmd for f in ("--save_state", "--save_state_on_train_end", "--keep_last_n_states")))
    except Exception as e:
        ck(f"{label} command builds", False, repr(e))

# --- 4. preset round-trip --------------------------------------------------------------------
vals = dict(defaults(K2), SAVE_STATE=False, SAVE_STATE_ON_TRAIN_END=True, KEEP_LAST_N_STATES="5")
preset = FORM.to_preset(K2, vals)
ck("preset captures SAVE_STATE", preset.get("SAVE_STATE") is False, preset.get("SAVE_STATE"))
ck("preset captures SAVE_STATE_ON_TRAIN_END", preset.get("SAVE_STATE_ON_TRAIN_END") is True)
ck("preset captures KEEP_LAST_N_STATES", str(preset.get("KEEP_LAST_N_STATES")) == "5",
   preset.get("KEEP_LAST_N_STATES"))

back = FORM.apply_preset(K2, dict(defaults(K2), SAVE_STATE=True, SAVE_STATE_ON_TRAIN_END=False,
                                  KEEP_LAST_N_STATES="1"), preset)
ck("preset restores SAVE_STATE", back.get("SAVE_STATE") is False, back.get("SAVE_STATE"))
ck("preset restores SAVE_STATE_ON_TRAIN_END", back.get("SAVE_STATE_ON_TRAIN_END") is True)
ck("preset restores KEEP_LAST_N_STATES", str(back.get("KEEP_LAST_N_STATES")) == "5", back.get("KEEP_LAST_N_STATES"))

# A preset saved before this feature must not clobber the current setting.
kept = FORM.apply_preset(K2, dict(defaults(K2), SAVE_STATE=True), {"MAX_TRAIN_EPOCHS": 10})
ck("an older preset without the keys leaves SAVE_STATE alone", kept.get("SAVE_STATE") is True,
   kept.get("SAVE_STATE"))

# --- 5. retired: the resume-a-finished-run warning ---------------------------------------------
# The desktop app asked before resuming a state whose epoch was already at or past Max Train Epochs. Fizgig 8 has no
# such check (its Resume field's help says to raise Max Train Epochs to train a finished LoRA further), so there is
# nothing left to pin here. Kept as a note rather than a passing check that tests nothing.

shutil.rmtree(BASE, ignore_errors=True)
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
