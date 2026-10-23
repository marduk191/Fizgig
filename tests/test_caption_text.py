"""The web Captions tab reads caption files in any encoding, and its decoder matches training's — headless, no GPU.

The fork's caption decoder (fizgig.dataset.image_dataset.decode_caption, adopted upstream) fixed training: one
Windows-1252 caption used to abort a whole caching run. Fizgig 8's web server reads captions too, in its Captions
tab listing and in Find & Replace, and both did it as strict UTF-8, so the same file broke them: a
UnicodeDecodeError that took the whole request down, the valid captions with it. The server never imports torch,
so it can't use image_dataset's copy; fizgig.dataset.caption_text is a stdlib-only twin.

Pinned here:
  1. the two decoders agree on every byte pattern, so neither can drift from the other unnoticed;
  2. the Captions tab listing and Find & Replace survive a cp1252 caption and show its real text;
  3. the server still imports without torch (the reason caption_text exists at all).

Run: venv/Scripts/python.exe tests/test_caption_text.py
"""
import logging
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# src only. The repo root holds Fizgig 8's fizgig.py, which would shadow the fizgig package.
sys.path.insert(0, os.path.join(REPO, "src"))
logging.disable(logging.WARNING)            # the decoders warn on purpose for non-UTF-8 input

from PIL import Image  # noqa: E402

from fizgig.dataset.caption_text import decode_caption as light, read_caption  # noqa: E402

sys.stdout.reconfigure(errors="replace")   # a failing check prints U+FFFD, which a cp1252 console lacks
fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


# --- 1. parity with training's decoder ------------------------------------------------------
from fizgig.dataset.image_dataset import decode_caption as training  # noqa: E402  (imports torch: fine here)

CASES = {
    "plain utf-8": "a woman smiling".encode("utf-8"),
    "utf-8 curly quote": "the girl’s hat".encode("utf-8"),
    "utf-8 with BOM": b"\xef\xbb\xbf" + "with bom".encode("utf-8"),
    "cp1252 0x92 (the tw1 dataset)": b"the girl\x92s worm\x92s-eye view",
    "cp1252-undefined 0x81 (latin-1 fallback)": b"bad byte here: \x81 end",
    "utf-16 LE BOM (Notepad Unicode)": "utf16 caption".encode("utf-16"),
    "empty": b"",
}
for name, raw in CASES.items():
    a, b = light(raw, "x.txt"), training(raw, "x.txt")
    ck(f"same text as training's decoder: {name}", a == b, (a, b) if a != b else "")
ck("  cp1252 0x92 decodes to the real apostrophe (U+2019)",
   light(CASES["cp1252 0x92 (the tw1 dataset)"], "x") == "the girl’s worm’s-eye view")

# --- 2. the two web readers survive a cp1252 caption -----------------------------------------
from fizgig.web import captions as CAP  # noqa: E402
from fizgig.web import server as S  # noqa: E402

d = tempfile.mkdtemp(prefix="fizgig_capenc_")
for n in ("a", "b"):
    Image.new("RGB", (8, 8)).save(os.path.join(d, n + ".png"))
with open(os.path.join(d, "a.txt"), "wb") as f:
    f.write("the girl’s worm’s-eye view".encode("cp1252"))
with open(os.path.join(d, "b.txt"), "w", encoding="utf-8") as f:
    f.write("a plain utf-8 view")

ck("read_caption: a missing file is None, not an error", read_caption(os.path.join(d, "nope.txt")) is None)
class _Handler:
    """Stands in for the HTTP handler: _captions ends in self.json(payload); this hands the payload back."""

    def json(self, payload):
        return payload


def _listing():
    items = S.Handler._captions(_Handler(), d, ".txt")["items"]
    return {(it.get("name") if isinstance(it, dict) else it[0]): (it.get("caption") if isinstance(it, dict)
            else it[2]["caption"]) for it in items}


try:
    listing = _listing()
    ck("Captions tab lists a folder holding a cp1252 caption", set(listing) == {"a.png", "b.png"}, sorted(listing))
    ck("  and shows that caption's real text", listing.get("a.png") == "the girl’s worm’s-eye view",
       listing.get("a.png"))
    ck("  while the UTF-8 one is unchanged", listing.get("b.png") == "a plain utf-8 view", listing.get("b.png"))
except Exception as e:
    ck("Captions tab lists a folder holding a cp1252 caption", False, f"{type(e).__name__}: {e}")

try:
    r = CAP.find_replace(d, "view", "angle", apply=False)
    ck("Find & Replace survives a cp1252 caption and finds both matches",
       sorted(c["name"] for c in r["changes"]) == ["a.txt", "b.txt"], [c["name"] for c in r["changes"]])
    r = CAP.find_replace(d, "view", "angle", apply=True)
    with open(os.path.join(d, "a.txt"), encoding="utf-8") as f:
        after = f.read()
    ck("  and an applied edit keeps the apostrophe, written back as UTF-8",
       after == "the girl’s worm’s-eye angle", after)
except Exception as e:
    ck("Find & Replace survives a cp1252 caption and finds both matches", False, f"{type(e).__name__}: {e}")

# --- 2b. LoRA Isolation's Subjects card reads the same captions ------------------------------
# multi_person reads a subject folder's captions for the trigger word, the class word and the card's preview caption.
from fizgig.families import multi_person as MP  # noqa: E402

m = tempfile.mkdtemp(prefix="fizgig_capenc_mp_")
with open(os.path.join(m, "1.txt"), "wb") as f:
    f.write("zwxem, a woman’s portrait".encode("cp1252"))
with open(os.path.join(m, "2.txt"), "w", encoding="utf-8") as f:
    f.write("zwxem, a woman smiling")
ck("Subjects card: the preview caption shows a cp1252 caption's real text",
   MP.sample_caption(m, "zwxem") == "zwxem, a woman’s portrait", MP.sample_caption(m, "zwxem"))
ck("  the trigger word is still found", MP.caption_trigger(m) == "zwxem", MP.caption_trigger(m))
ck("  and every caption is read, decoded", sorted(MP._captions(m, ".txt")) ==
   ["zwxem, a woman smiling", "zwxem, a woman’s portrait"], MP._captions(m, ".txt"))

# --- 2c. upstream's own reader (#225) and the edit run's instruction ------------------------------
# Upstream adopted the any-encoding read for the Captions tab and Find & Replace (captions.read_text); the checks
# above run on it. Its text must match training's.
_rt = CAP.read_text(os.path.join(d, "a.txt"))
ck("captions.read_text reads a cp1252 caption as training does", _rt == "the girl’s worm’s-eye angle", _rt)
from fizgig.families import launch as LA  # noqa: E402
e_dir = tempfile.mkdtemp(prefix="fizgig_capenc_edit_")
Image.new("RGB", (8, 8)).save(os.path.join(e_dir, "a.png"))
with open(os.path.join(e_dir, "a.txt"), "wb") as f:
    f.write("make it the girl’s style\nsecond line".encode("cp1252"))
try:
    got = LA.edit_instruction({"image_folder": e_dir, "caption_ext": ".txt"})
    ck("an edit run's preview instruction reads a cp1252 caption (Start used to fail)",
       got == "make it the girl’s style", got)
except Exception as ex:
    ck("an edit run's preview instruction reads a cp1252 caption (Start used to fail)", False,
       f"{type(ex).__name__}: {ex}")

# --- 3. the server stays torch-free ---------------------------------------------------------
# Run in a fresh interpreter, from a folder with no fizgig.py in it (the repo root's would shadow the package).
r = subprocess.run([sys.executable, "-c", "import sys; import fizgig.web.server, fizgig.web.captions; "
                    "print('torch' in sys.modules)"], capture_output=True, text=True, cwd=tempfile.gettempdir(),
                   env=dict(os.environ, PYTHONPATH=os.path.join(REPO, "src")), timeout=180)
ck("importing the web server and Captions module does not load torch", r.stdout.strip() == "False",
   (r.stdout.strip(), r.stderr.strip()[-160:]))

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
