"""Dataset folders containing [square brackets] — headless, no GPU.

Brackets are glob character-class syntax, so an unescaped glob pattern silently matches NOTHING.
The Captions tab uses os.listdir and was perfectly happy reading and writing such a folder, while
the training pre-flight used glob and reported "No caption files found", blocking the run.

Run: venv/Scripts/python.exe tests/test_bracket_paths.py
"""
import glob
import os
import shutil
import sys
import tempfile

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = r"W:/Peter/Documents/Development/Fizgig"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import tkinter as tk
from PIL import Image
import lora_trainer_gui as G

G.LAST_USED_FILE = os.path.join(os.environ["TEMP"], "nope", ".last_used.json")

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


BASE = tempfile.mkdtemp()
# Names that are all valid on disk but are glob metacharacters.
FOLDERS = ["[subject] photos", "shoot [2026]", "set [a-z] final", "normal folder"]
for _f in FOLDERS:
    d = os.path.join(BASE, _f)
    os.makedirs(d)
    for i in range(3):
        Image.new("RGB", (64, 64)).save(os.path.join(d, f"img{i}.png"))
        with open(os.path.join(d, f"img{i}.txt"), "w", encoding="utf-8") as fh:
            fh.write("a caption")

# --- 1. the raw failure mode, so the test documents WHY escape is needed ------------------
_d = os.path.join(BASE, FOLDERS[0])
ck("unescaped glob finds nothing in a bracketed folder (the bug)",
   len(glob.glob(os.path.join(_d, "*.txt"))) == 0)
ck("  escaped glob finds all 3",
   len(glob.glob(os.path.join(glob.escape(_d), "*.txt"))) == 3)

# --- 2. the dataset loader ----------------------------------------------------------------
from fizgig.dataset.image_dataset import glob_images

for _f in FOLDERS:
    d = os.path.join(BASE, _f)
    ck(f"glob_images finds 3 captioned images in {_f!r}",
       len(glob_images(d, caption_extension=".txt")) == 3,
       len(glob_images(d, caption_extension=".txt")))

# --- 3. the training pre-flight that actually blocked the run -----------------------------
root = tk.Tk()
root.withdraw()
g = G.LoRATrainerGUI(root)

_errors = []
G.messagebox.showerror = lambda title, msg: _errors.append(msg)

for _f in FOLDERS:
    d = os.path.join(BASE, _f)
    g.image_folder_var.set(d)
    g.dataset_caption_ext_var.set(".txt")
    _errors.clear()
    try:
        g.validate_inputs()
    except Exception:
        pass  # other validation (model paths) may fail — we only care about the caption error
    blob = " ".join(_errors)
    ck(f"no bogus 'missing captions' error for {_f!r}",
       "No caption files" not in blob,
       [ln for ln in blob.splitlines() if "caption" in ln.lower()][:1])

# --- 4. the GUI's own caption discovery agrees with the pre-flight ------------------------
g.image_folder_var.set(os.path.join(BASE, FOLDERS[0]))
ck("Captions tab lists the images in a bracketed folder",
   len(g.get_caption_image_files()) == 3, len(g.get_caption_image_files()))
_analysis = g._analyze_dataset(os.path.join(BASE, FOLDERS[0]))
ck("  dataset analysis sees the captions too",
   _analysis and _analysis["has_captions"] and _analysis["caption_count"] == 3, _analysis)

root.destroy()
shutil.rmtree(BASE, ignore_errors=True)
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S)"))
sys.exit(1 if fails else 0)
