"""RunPod Preferences card, auto-stop, and the disk warning — headless, no GPU.

The auto-stop predicate is the dangerous one and gets the most coverage here. It shuts down a
machine somebody is paying for, so every way a training subprocess can end is asserted
individually. The two that look alike are the trap: a PAUSE exits 0 just like a completed run, and
a user STOP arrives as a non-zero code with no flag distinguishing it from a crash.

Run: venv/Scripts/python.exe tests/test_runpod_card.py
"""
import os
import sys

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = r"W:/Peter/Documents/Development/Fizgig"
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


def gui(pod: bool):
    if pod:
        os.environ["FIZGIG_POD"] = "1"
    else:
        os.environ.pop("FIZGIG_POD", None)
    root = tk.Tk()
    root.withdraw()
    return root, G.LoRATrainerGUI(root)


def all_text(widget):
    out = []
    for c in widget.winfo_children():
        try:
            t = c.cget("text")
            if t:
                out.append(str(t))
        except Exception:
            pass
        out.extend(all_text(c))
    return out


# --- 1. detection -----------------------------------------------------------------------------
os.environ.pop("FIZGIG_POD", None)
ck("no marker -> not a pod", not G._running_on_pod())
for val in ("1", "true", "yes"):
    os.environ["FIZGIG_POD"] = val
    ck(f"  FIZGIG_POD={val!r} -> pod", G._running_on_pod())
os.environ["FIZGIG_POD"] = "0"
ck("  FIZGIG_POD='0' -> not a pod", not G._running_on_pod())
os.environ.pop("FIZGIG_POD", None)

# --- 2. the card, both audiences --------------------------------------------------------------
root, g = gui(pod=False)
# Scope to the card, not the whole window: other Preferences cards legitimately mention VRAM
# figures ("fits 16GB cards", the 8 GB Krea 2 note), and those are correct where they are.
# _start_section_card returns the CONTENT frame; the title and description live on its parent.
txt = " ".join(all_text(g._runpod_card) + all_text(g._runpod_card.master))
ck("desktop shows the advert", "Run Fizgig on a rented GPU" in txt)
# Until the template is public the card must NOT offer a button that goes nowhere.
if G.LoRATrainerGUI.RUNPOD_TEMPLATE_LIVE:
    ck("  with a deploy button", "Deploy on RunPod" in txt)
    ck("  and discloses the referral", "supports Fizgig's development" in txt)
    ck("  LIVE means the URL is real, not the placeholder",
       "template=fizgig" not in G.LoRATrainerGUI.RUNPOD_DEPLOY_URL,
       G.LoRATrainerGUI.RUNPOD_DEPLOY_URL)
else:
    ck("  not live yet -> says Coming soon", "Coming soon" in txt)
    ck("  and offers NO dead deploy button", "Deploy on RunPod" not in txt)
# The guide is the only way to read about this while it says Coming soon, so it must be there in
# BOTH states, not just once a Deploy button exists.
ck("  links to the guide either way", "Read the guide" in txt)
ck("  and does NOT show pod controls", "Stop this pod when a training run finishes" not in txt)
ck("  no minimum-spec pitch (renting is about MORE card, not scraping by)",
   "8 GB" not in txt and "8GB" not in txt)
root.destroy()

root, g = gui(pod=True)
txt = " ".join(all_text(g._runpod_card) + all_text(g._runpod_card.master))
ck("pod shows the controls", "Stop this pod when a training run finishes" in txt)
ck("  storage line present", any("Storage:" in t for t in all_text(g._runpod_card)))
ck("  says closing the tab does not stop training",
   "Closing this browser tab does not stop training" in txt)
ck("  points at the file manager", "port 8080" in txt)
ck("  and does NOT show the advert", "Run Fizgig on a rented GPU" not in txt)

# --- 3. THE AUTO-STOP PREDICATE ---------------------------------------------------------------
# Every way a training subprocess can end. Only one of them may stop the machine.
g.prefs_vars["runpod_stop_when_done"].set("1")
calls = []
g._maybe_stop_pod_after_training = lambda: calls.append("stop")

CASES = [
    # (label,                              return_code, state_before, should_stop)
    ("a completed run",                              0, "running",   True),
    ("a PAUSE (also exits 0!)",                      0, "pausing",   False),
    ("a user Stop (taskkill on Windows)",            1, "running",   False),
    ("a user Stop (SIGTERM on POSIX)",             -15, "running",   False),
    ("a crash",                                      1, "running",   False),
    ("an OOM kill",                               -9,   "running",   False),
    ("exiting while already idle",                   0, "idle",      False),
]
for label, rc, state, expect in CASES:
    calls.clear()
    g.training_state = state
    try:
        g._on_training_subprocess_exited(rc)
    except Exception as e:
        ck(f"auto-stop: {label}", False, f"raised {type(e).__name__}: {e}")
        continue
    ck(f"auto-stop: {label} -> {'stops' if expect else 'does NOT stop'}",
       bool(calls) == expect, f"fired={bool(calls)}")

# The toggle has to actually gate it.
del g._maybe_stop_pod_after_training
g.prefs_vars["runpod_stop_when_done"].set("0")
shown = []
G.tk.Toplevel = (lambda *a, **k: (_ for _ in ()).throw(AssertionError("dialog shown!")))
try:
    g.training_state = "running"
    g._maybe_stop_pod_after_training()
    ck("toggle off -> no countdown even on a clean finish", True)
except AssertionError as e:
    ck("toggle off -> no countdown even on a clean finish", False, e)

# --- 3a. the version line ---------------------------------------------------------------------
# A pinned image plus a self-updating app means the two versions diverge by design, so a bug
# report needs both. Rebuild the card with the env a real pod provides.
os.environ["FIZGIG_IMAGE_VERSION"] = "2.13.0"
os.environ["RUNPOD_POD_ID"] = "dvjecz8zn02rn1"
os.environ["RUNPOD_GPU_NAME"] = "NVIDIA+GeForce+RTX+4090"
root.destroy()
root, g = gui(pod=True)
vtxt = " ".join(all_text(g._runpod_card))
ck("version line shows the image", "image 2.13.0" in vtxt, vtxt[-90:])
ck("  and the pod id", "dvjecz8zn02rn1" in vtxt)
ck("  and un-mangles the GPU name", "GeForce RTX 4090" in vtxt)
for k in ("FIZGIG_IMAGE_VERSION", "RUNPOD_POD_ID", "RUNPOD_GPU_NAME"):
    os.environ.pop(k, None)

# --- 3b. the API key field --------------------------------------------------------------------
# A public template hands its env vars to every container deployed from it, so a key cannot ship
# in one — it has to be per-user. That makes the prefs field the primary route, not a convenience.
ck("key field is masked", str(g._pod_key_entry.cget("show")) not in ("", "None"),
   repr(g._pod_key_entry.cget("show")))
g.prefs_vars["runpod_api_key"].set("")
os.environ.pop("RUNPOD_STOP_API_KEY", None)
ck("no key anywhere -> not ready", g._pod_stop_key() == "")
ck("  and the card says so", "needed" in g._pod_key_status.cget("text"))

g.prefs_vars["runpod_api_key"].set("rpa_pretend_account_key")
ck("key in prefs -> ready", g._pod_stop_key() == "rpa_pretend_account_key")
ck("  and the card confirms it", "ready" in g._pod_key_status.cget("text"))

g.prefs_vars["runpod_api_key"].set("")
os.environ["RUNPOD_STOP_API_KEY"] = "rpa_from_template"
ck("env var still works as a fallback", g._pod_stop_key() == "rpa_from_template")
g.prefs_vars["runpod_api_key"].set("rpa_prefs_wins")
ck("  prefs takes precedence over the env var", g._pod_stop_key() == "rpa_prefs_wins")
os.environ.pop("RUNPOD_STOP_API_KEY", None)
g.prefs_vars["runpod_api_key"].set("")

# The injected pod-scoped key must never be picked up: it 403s on every pod call, so using it
# would produce a failure at the one moment nobody is watching.
os.environ["RUNPOD_API_KEY"] = "rpa_pod_scoped_cannot_stop_pods"
ck("RunPod's own injected key is NOT used", g._pod_stop_key() == "")
os.environ.pop("RUNPOD_API_KEY", None)

# --- 4. prefs round-trip ----------------------------------------------------------------------
ck("runpod_stop_when_done is in DEFAULT_PREFS and defaults off",
   G.DEFAULT_PREFS.get("runpod_stop_when_done") == "0")
ck("input_dataset_dir is in DEFAULT_PREFS", "input_dataset_dir" in G.DEFAULT_PREFS)
ck("  both are StringVars, not BooleanVars (the prefs loop requires it)",
   isinstance(g.prefs_vars["runpod_stop_when_done"], tk.StringVar))

# --- 5. disk warning --------------------------------------------------------------------------
import shutil as _sh  # noqa: E402
_real_usage = _sh.disk_usage
asked = []
G.messagebox.askyesno = lambda t, m: (asked.append(m), True)[1]


class _Usage:
    def __init__(self, free_gb):
        self.total = 500 * 1024 ** 3
        self.used = self.total - int(free_gb * 1024 ** 3)
        self.free = int(free_gb * 1024 ** 3)


g.settings["LORA_OUTPUT_DIR"] = REPO
for free_gb, should_warn in ((3.2, True), (14.9, True), (15.1, False), (400, False)):
    asked.clear()
    _sh.disk_usage = lambda p, _f=free_gb: _Usage(_f)
    g._confirm_disk_headroom()
    ck(f"disk warning at {free_gb} GB free -> {'warns' if should_warn else 'silent'}",
       bool(asked) == should_warn)
    if should_warn and asked:
        ck(f"  shows the real figure ({free_gb} GB)", f"{free_gb:.1f} GB" in asked[0])

# --- 5b. the storage line copes with what real hosts report ----------------------------------
# A network volume reports the host's whole backing pool, not the volume quota: a real 100 GB
# volume read as "431035 GB free of 1430281 GB". Printing that verbatim looks broken.
for total_gb, free_gb, expect in ((1430281, 431035, "network volume"),
                                  (25, 23, "container disk"),
                                  (1000, 840, "GB free of")):
    class _U:
        total = int(total_gb * 1024 ** 3)
        free = int(free_gb * 1024 ** 3)
        used = total - free
    _sh.disk_usage = lambda p, _u=_U: _u
    g._pod_storage_lbl = tk.Label(root)
    g._refresh_pod_storage()
    shown = g._pod_storage_lbl.cget("text")
    ck(f"storage: {total_gb} GB total -> says '{expect}'", expect in shown, shown[:70])
    if total_gb > 10000:
        ck("  and does NOT print the absurd figure", "1430281" not in shown)

# A failed probe must never block a run.
_sh.disk_usage = lambda p: (_ for _ in ()).throw(OSError("no such device"))
ck("a failed disk probe does not block training", g._confirm_disk_headroom() is True)
_sh.disk_usage = _real_usage

root.destroy()
os.environ.pop("FIZGIG_POD", None)
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
