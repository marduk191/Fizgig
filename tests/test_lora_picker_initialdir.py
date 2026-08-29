"""LoRA Browse dialogs start in a folder that actually contains LoRAs — headless, no GPU.

On RunPod the input_lora_dir pref is never seeded, so the Repair Studio / Explorer / Extract /
Profiler / Context LoRA pickers opened in the process cwd — the git clone, which holds no
.safetensors and made the picker look broken. _lora_initialdir now falls back to the LoRA
output directory (the folder training just wrote into). These tests pin the fallback order:
input_lora_dir pref when valid, else the live Output Directory field, else settings, else "".

Run: venv/Scripts/python.exe tests/test_lora_picker_initialdir.py
"""
import os
import sys
import tempfile

os.environ["FIZGIG_NO_PERSIST"] = "1"
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


root = tk.Tk()
root.withdraw()
app = G.LoRATrainerGUI(root)

lora_dir = tempfile.mkdtemp(prefix="loras_")
out_dir = tempfile.mkdtemp(prefix="outputs_")


def set_output_field(value):
    app.entries["LORA_OUTPUT_DIR"].delete(0, tk.END)
    app.entries["LORA_OUTPUT_DIR"].insert(0, value)


# Pod-shaped state: no input_lora_dir pref, output dir NOT under the cwd.
app.prefs_vars["input_lora_dir"].set("")
set_output_field(out_dir)
ck("no pref -> falls back to the Output Directory field", app._lora_initialdir() == out_dir)

# The pref wins when set to a real folder.
app.prefs_vars["input_lora_dir"].set(lora_dir)
ck("input_lora_dir pref wins when valid", app._lora_initialdir() == lora_dir)

# A stale/deleted pref folder must not strand the dialog — fall through to outputs.
app.prefs_vars["input_lora_dir"].set(os.path.join(lora_dir, "deleted", "gone"))
ck("invalid pref falls through to outputs", app._lora_initialdir() == out_dir)

# Empty field -> settings (what start_training snapshots / entrypoint-era default).
app.prefs_vars["input_lora_dir"].set("")
set_output_field("")
app.settings["LORA_OUTPUT_DIR"] = out_dir
ck("empty field falls back to settings", app._lora_initialdir() == out_dir)

# Nothing valid anywhere -> "" (Tk keeps its own last-used folder; never crash).
app.settings["LORA_OUTPUT_DIR"] = os.path.join(out_dir, "nope")
ck("nothing valid -> empty string, not a crash", app._lora_initialdir() == "")

# Every LoRA picker actually uses the helper (guards against a future picker
# reverting to a bare askopenfilename with no initialdir).
import inspect  # noqa: E402
for name in ("_browse_repair_lora", "_browse_context_lora",
             "_browse_extract_source", "_browse_profiler_lora"):
    src = inspect.getsource(getattr(G.LoRATrainerGUI, name))
    ck(f"{name} seeds from _lora_initialdir", "_lora_initialdir()" in src)

root.destroy()
print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
