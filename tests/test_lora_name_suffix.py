"""Per-family LoRA name suffix + the shared sample resolutions (Fizgig 8 form) — headless, no GPU.

What this guarded in the desktop app: switching model family retagged the LoRA Name's trailing family suffix
(myface_k9b -> myface_krea2), touched only an EXACT trailing suffix, and refused to rename while a run was live or
paused (a rename would orphan the paused run's state folder).

Fizgig 8 dropped that retag. The web app keeps each family's values separately, so switching family brings back
that family's own name instead of rewriting the one shown; with no rename there is nothing to block during a run.
Sections 1-4 of the old file are therefore retired, not faked. What survives, and is pinned here:
  1. every family's default name carries its own suffix (my_lora_<suffix>), so a fresh run names its file by family;
  2. suffixes are unique, or two families' fresh LoRAs would collide in one output folder;
  3. the Samples width / height offer the one shared size list, which reaches 1536.

Run: venv/Scripts/python.exe tests/test_lora_name_suffix.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# src only. The repo root holds Fizgig 8's fizgig.py, which would shadow the fizgig package.
sys.path.insert(0, os.path.join(REPO, "src"))

from fizgig.families import form as FORM  # noqa: E402
from fizgig.families.registry import training_families  # noqa: E402

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


def rows(desc):
    return {r["key"]: r for sec in FORM.form(desc)["sections"] for r in sec["rows"] if r.get("key")}


FAMS = training_families()

# --- 1. each family's default name carries its suffix ----------------------------------------
for d in FAMS:
    want = f"my_lora_{d.lora_name_suffix}" if d.lora_name_suffix else "my_lora"
    got = rows(d)["LORA_NAME"].get("default")
    ck(f"{d.key}: default name is {want!r}", got == want, got)
ck("the two original families keep their suffixes (k9b, krea2)",
   {d.key: d.lora_name_suffix for d in FAMS}.get("klein") == "k9b"
   and {d.key: d.lora_name_suffix for d in FAMS}.get("krea2") == "krea2")

# --- 2. suffixes never collide -------------------------------------------------------------
_sfx = [d.lora_name_suffix for d in FAMS if d.lora_name_suffix]
ck("every family suffix is unique", len(_sfx) == len(set(_sfx)),
   sorted(s for s in set(_sfx) if _sfx.count(s) > 1))

# --- 3. the shared sample resolutions -------------------------------------------------------
ck("the shared size list reaches 1536", "1536" in FORM.SIZES and "1280" in FORM.SIZES, FORM.SIZES)
for d in FAMS:
    r = rows(d)
    if "SAMPLE_WIDTH" not in r:
        ck(f"{d.key}: renders no pictures -> no sample size rows", "SAMPLE_HEIGHT" not in r)
        continue
    ck(f"{d.key}: Samples width / height offer the shared list",
       tuple(r["SAMPLE_WIDTH"].get("choices") or ()) == tuple(FORM.SIZES)
       and tuple(r["SAMPLE_HEIGHT"].get("choices") or ()) == tuple(FORM.SIZES))
    ck(f"  {d.key}: its default size is one of them",
       str(r["SAMPLE_WIDTH"].get("default")) in FORM.SIZES and str(r["SAMPLE_HEIGHT"].get("default")) in FORM.SIZES,
       (r["SAMPLE_WIDTH"].get("default"), r["SAMPLE_HEIGHT"].get("default")))

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
