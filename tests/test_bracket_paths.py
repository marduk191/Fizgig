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
# Repo root, derived from this file's location -- was hardcoded to one machine's
# absolute path, which made the whole suite unrunnable anywhere else.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# src only. The repo root holds Fizgig 8's fizgig.py, which would shadow the fizgig package.
sys.path.insert(0, os.path.join(REPO, "src"))

from PIL import Image

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
# Fizgig 8: Start's checks are plain functions (fizgig.families.checks), the same ones the launch plan refuses on.
from fizgig.families import checks  # noqa: E402

for _f in FOLDERS:
    d = os.path.join(BASE, _f)
    problems = checks.training_folder(d, ".txt", media_exts={".png"})
    ck(f"no bogus 'missing captions' error for {_f!r}",
       not any("caption" in p.lower() for p in problems), problems[:1])

# And the check is still ALIVE: a bracketed folder that really has no captions must still be refused, or the
# assertion above would pass for a check that had simply stopped looking.
_bare = os.path.join(BASE, "[uncaptioned] set")
os.makedirs(_bare)
Image.new("RGB", (64, 64)).save(os.path.join(_bare, "img0.png"))
ck("  ...while a bracketed folder with NO captions is still refused",
   any("No caption files" in p for p in checks.training_folder(_bare, ".txt", media_exts={".png"})))

# --- 4. the Captions tab's own discovery agrees with the pre-flight ------------------------
from fizgig.web import captions as CAP  # noqa: E402

_bf = os.path.join(BASE, FOLDERS[0])
ck("Captions tab lists the images in a bracketed folder",
   len(CAP.media_files(_bf, clips=False)) == 3, len(CAP.media_files(_bf, clips=False)))
_fr = CAP.find_replace(_bf, "caption", "label", apply=False)
ck("  and Find & Replace finds its captions (glob-escaped too)", len(_fr["changes"]) == 3, len(_fr["changes"]))

shutil.rmtree(BASE, ignore_errors=True)
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S)"))
sys.exit(1 if fails else 0)
