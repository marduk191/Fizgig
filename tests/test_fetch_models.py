"""Model fetcher: what it downloads, and — more importantly — what it refuses to touch.

The dangerous failure here isn't a failed download, it's a SUCCESSFUL one that overwrites a
path the user chose on purpose. Several legitimate choices are smaller than the file the
manifest names (the fp8_scaled RAW DiT instead of bf16, Klein's fp8mixed text encoder, a
community fine-tune), so any size-based validation applied to an existing pref would quietly
replace them.

No network: everything here is dry-run or operates on fake files.

Run: venv/Scripts/python.exe tests/test_fetch_models.py
"""
import json
import os
import shutil
import struct
import sys
import tempfile

# Repo root, derived from this file's location -- was hardcoded to one machine's
# absolute path, which made the whole suite unrunnable anywhere else.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)  # for lora_trainer_gui, to cross-check manifest keys against the prefs

from fizgig.scripts import fetch_models as F
import lora_trainer_gui as G

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


def fake_safetensors(path, size=4096):
    """A structurally valid but tiny .safetensors — stands in for a real or truncated file."""
    hdr = json.dumps({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        f.write(b"\0" * max(0, size - 8 - len(hdr)))


def weight(family, pref_key):
    return [w for w in F.FAMILIES[family] if w.pref_key == pref_key][0]


BASE = tempfile.mkdtemp()
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS)
RAW = weight("krea2", "krea2_raw_dit")          # manifest says 26 GB -> 20.8 GB floor
VAE = weight("krea2", "krea2_vae")
TE = weight("krea2", "krea2_text_encoder")

# --- the manifest itself -----------------------------------------------------------------
ck("every weight has a pref key, repo and path",
   all(w.pref_key and w.repo and w.path_in_repo
       for fam in F.FAMILIES.values() for w in fam))
ck("pref keys match what the GUI reads",
   {w.pref_key for w in F.FAMILIES["klein"]} >= {"base_dit", "distilled_dit", "vae", "text_encoder"}
   and {w.pref_key for w in F.FAMILIES["krea2"]} >= {"krea2_raw_dit", "krea2_turbo_dit",
                                                     "krea2_vae", "krea2_text_encoder",
                                                     "krea2_turbo_lora"})
# Klein's set is a superset, not an exact match: the Captions tab captions ANY dataset with the
# Krea 2 Qwen3-VL encoder, so every family's button now fetches it too. That shared entry is the
# invariant worth asserting -- it is the thing that silently goes missing if a family list is
# rebuilt from its own models alone, and a Klein-only user then has no captioner.
ck("every family fetches the shared captioning TE",
   all("krea2_text_encoder" in {w.pref_key for w in fam} for fam in F.FAMILIES.values()),
   {k: sorted(w.pref_key for w in v) for k, v in F.FAMILIES.items()})
# Every pref key the manifest can write must be one the GUI actually reads back, or a download
# lands somewhere Preferences never looks.
ck("no manifest key is unknown to the GUI",
   {w.pref_key for fam in F.FAMILIES.values() for w in fam} <= set(G.DEFAULT_PREFS),
   sorted({w.pref_key for fam in F.FAMILIES.values() for w in fam} - set(G.DEFAULT_PREFS)))
ck("Klein is flagged gated, Krea 2 is not",
   any(w.gated for w in F.FAMILIES["klein"]) and not any(w.gated for w in F.FAMILIES["krea2"]))
# The rule here was "nothing in a family is optional": a download button that silently omits a model
# leaves you finding out weeks later that Repair Studio will not open. Upstream has since made Krea 2's
# Turbo DiT and Turbo LoRA optional, and the reason the rule existed no longer holds for them: the
# driver workbench falls back to the Turbo LoRA on the RAW model, then to plain steps, so nothing fails
# to open without them. So pin what the rule was protecting, for every family, instead:
#   1. nothing TRAINING needs (the description's dit / vae / text_encoder roles) is optional UNLESS a
#      required file already carries it. SDXL is the case that defines the exception: its VAE and text
#      encoders sit inside the checkpoint (ModelFile.inside="sdxl_checkpoint"), and the separate files
#      are overrides, so leaving them out loses nothing;
#   2. optional files are only preview accelerators or such carried overrides.
from fizgig.families.registry import training_families as _tf  # noqa: E402

_CRITICAL = {"dit", "vae", "text_encoder"}
_mf = {(d.key, mf.pref_key): mf for d in _tf() for mf in d.model_files}
_required = {(d.key, mf.pref_key) for d in _tf() for mf in d.model_files if mf.required}


def _carried(fam, mf):
    return bool(getattr(mf, "inside", "")) and (fam, mf.inside) in _required


_crit_optional = [f"{fam}:{w.pref_key}" for fam, ws in F.FAMILIES.items() for w in ws
                  if w.optional and (fam, w.pref_key) in _mf
                  and _mf[(fam, w.pref_key)].role in _CRITICAL
                  and not _carried(fam, _mf[(fam, w.pref_key)])]
ck("no family marks a training-critical model optional (unless a required file carries it)",
   not _crit_optional, _crit_optional)
ck("  SDXL's optional VAE really is carried by its required checkpoint",
   _carried("sdxl", _mf[("sdxl", "sdxl_vae")]))
_k2_opt = {w.pref_key for w in F.FAMILIES["krea2"] if w.optional}
ck("  Krea 2's optional files are exactly its two preview accelerators",
   _k2_opt == {"krea2_turbo_dit", "krea2_turbo_lora"}
   and {_mf[("krea2", k)].role for k in _k2_opt} == {"preview_dit", "speed_lora"}, sorted(_k2_opt))
ck("  and Krea 2's base, VAE and text encoder are always fetched",
   not any(weight("krea2", k).optional for k in ("krea2_raw_dit", "krea2_vae", "krea2_text_encoder")))

# --- a user's own choice must survive ------------------------------------------------------
smaller = os.path.join(BASE, "krea2_raw_fp8_scaled.safetensors")
fake_safetensors(smaller)
prefs = {"krea2_raw_dit": smaller}
F.fetch_weight(RAW, MODELS, prefs, log=lambda *_: None, dry_run=True)
ck("a SMALLER legitimate variant is kept, not re-downloaded",
   prefs["krea2_raw_dit"] == smaller, prefs["krea2_raw_dit"])

custom = os.path.join(BASE, "Huihui-Qwen3-VL-4B-abliterated-fp8.safetensors")
fake_safetensors(custom)
prefs = {"krea2_text_encoder": custom}
F.fetch_weight(TE, MODELS, prefs, log=lambda *_: None, dry_run=True)
ck("a community fine-tune under another name is kept", prefs["krea2_text_encoder"] == custom)

# --- but a bad file we control must not be trusted ------------------------------------------
truncated = os.path.join(MODELS, RAW.filename)
fake_safetensors(truncated)          # 4 KB claiming to be the 26 GB RAW DiT
prefs = {}
F.fetch_weight(RAW, MODELS, prefs, log=lambda *_: None, dry_run=True)
ck("a truncated file at our own destination is NOT adopted",
   prefs.get("krea2_raw_dit") != truncated)

prefs = {"krea2_vae": os.path.join(BASE, "deleted.safetensors")}
_log = []
F.fetch_weight(VAE, MODELS, prefs, log=_log.append, dry_run=True)
ck("a pref pointing at a deleted file re-fetches", any("[get]" in l for l in _log))

# --- prefs.json handling --------------------------------------------------------------------
prefs_file = os.path.join(BASE, "prefs.json")
with open(prefs_file, "w", encoding="utf-8") as f:
    json.dump({"lora_output_dir": "keep/me", "krea2_vae": smaller}, f)
before = open(prefs_file, encoding="utf-8").read()
F.fetch(["tools"], models_dir=MODELS, repo_dir=BASE, log=lambda *_: None, dry_run=True)
ck("a tools-only run never rewrites prefs.json",
   open(prefs_file, encoding="utf-8").read() == before)

reloaded = F._load_prefs(prefs_file)
reloaded["krea2_vae"] = "new/path"
F._save_prefs(prefs_file, reloaded)
after = json.load(open(prefs_file, encoding="utf-8"))
ck("saving prefs preserves unrelated keys",
   after.get("lora_output_dir") == "keep/me" and after["krea2_vae"] == "new/path", after)

# --- header validation ------------------------------------------------------------------------
good = os.path.join(BASE, "good.safetensors")
fake_safetensors(good, 4096)
ck("valid header + big enough -> accepted", F._valid_safetensors(good, 1024))
ck("valid header + too small -> rejected", not F._valid_safetensors(good, 10 * 1024 * 1024))
html = os.path.join(BASE, "html.safetensors")
with open(html, "wb") as f:
    f.write(b"<!DOCTYPE html><html>error page</html>" * 200)
ck("an HTML error page is rejected", not F._valid_safetensors(html, 1024))
ck("a missing file is rejected", not F._valid_safetensors(os.path.join(BASE, "nope"), 1))

shutil.rmtree(BASE, ignore_errors=True)
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
