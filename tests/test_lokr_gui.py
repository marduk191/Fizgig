"""Network Type / LoKR across the Training form, its commands and the presets — headless, no GPU.

Fizgig 8 replaced the desktop Training tab with families/form.py: the same controls, as data the web UI
renders. form.form(desc) is what is shown, form.state(desc, values) is what is hidden for the values chosen,
and form.inputs feeds the same launch plan the Start button runs. So this drives those instead of Tk widgets.

Every family offers LoKR (each description's network_types is ("lora", "lokr")), so the desktop-era check
"Klein: Network Type hidden" is no longer behaviour to protect. What a family hides now depends on the KIND
of training (FAMILY_KIND): a slider is always a plain LoRA, and a fine-tune trains the model itself.

Run: venv/Scripts/python.exe tests/test_lokr_gui.py
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# src only. The repo root holds Fizgig 8's fizgig.py, which would shadow the fizgig package.
sys.path.insert(0, os.path.join(REPO, "src"))

from fizgig.families import form as FORM  # noqa: E402
from fizgig.families import launch as L  # noqa: E402
from fizgig.families.registry import get as get_family, training_families  # noqa: E402

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


def rows(desc):
    return {r["key"]: r for sec in FORM.form(desc)["sections"] for r in sec["rows"] if r.get("key")}


def defaults(desc):
    return {k: r.get("default") for k, r in rows(desc).items()}


def hidden(desc, **values):
    return set(FORM.state(desc, dict(defaults(desc), **values))["hidden"])


TMP = tempfile.mkdtemp(prefix="fizgig_lokr_")


def cmd(desc, **values):
    vals = dict(defaults(desc), LORA_NAME="t", LORA_OUTPUT_DIR=TMP, DATASET_CONFIG=os.path.join(TMP, "d.toml"),
                **values)
    inp = FORM.inputs(desc, vals, models={k: "m" for k in FORM.model_keys(desc)}, python=sys.executable,
                      repo_dir=REPO, image_folder="", cache_root=TMP, samples_dir=TMP)
    return L.train_command(desc, inp, L.LaunchPlan())


# --- 1. the controls ------------------------------------------------------------------------
K2 = get_family("krea2")
r = rows(K2)
ck("NETWORK_TYPE and LOKR_FACTOR are on the form", "NETWORK_TYPE" in r and "LOKR_FACTOR" in r)
ck("  default is standard LoRA (LoKR one pick away)", r["NETWORK_TYPE"].get("default") == "LoRA (standard)",
   r["NETWORK_TYPE"].get("default"))
ck("  default factor is 8", str(r["LOKR_FACTOR"].get("default")) == "8", r["LOKR_FACTOR"].get("default"))
# A strict choice, not an editable combo: junk typed in can't reach the command line.
ck("  Network Type is a fixed choice, not free text", r["NETWORK_TYPE"]["kind"] == "choice", r["NETWORK_TYPE"]["kind"])
ck("  offering exactly the two types",
   list(r["NETWORK_TYPE"].get("choices") or []) == ["LoRA (standard)", "LoKR (Kronecker)"],
   r["NETWORK_TYPE"].get("choices"))

# --- 2. what shows, per family and per Network Type -----------------------------------------
for d in training_families():
    if "lokr" not in d.network_types:
        ck(f"{d.key}: offers no LoKR -> nothing to check", True)
        continue
    h = hidden(d, NETWORK_TYPE="LoRA (standard)")
    ck(f"{d.key}: standard -> rank/alpha shown, factor hidden",
       "LOKR_FACTOR" in h and not {"NETWORK_DIM", "NETWORK_ALPHA"} & h and "NETWORK_TYPE" not in h, sorted(h & {
           "NETWORK_TYPE", "LOKR_FACTOR", "NETWORK_DIM", "NETWORK_ALPHA"}))
    h = hidden(d, NETWORK_TYPE="LoKR (Kronecker)")
    ck(f"  {d.key}: LoKR -> factor shown, rank/alpha hidden",
       "LOKR_FACTOR" not in h and {"NETWORK_DIM", "NETWORK_ALPHA"} <= h, sorted(h & {
           "NETWORK_TYPE", "LOKR_FACTOR", "NETWORK_DIM", "NETWORK_ALPHA"}))

# --- 3. the kind of training ----------------------------------------------------------------
# v8.0.7: a slider hides Network Type (always a plain LoRA); a fine-tune hides rank, alpha, Network Type and
# Resume Training (it trains the model itself and continues from its own checkpoint).
_kinds = dict(FORM.kinds(K2))
if "slider" in _kinds:
    h = hidden(K2, FAMILY_KIND="slider", NETWORK_TYPE="LoKR (Kronecker)")
    ck("slider -> Network Type and the factor hidden, even with LoKR picked",
       {"NETWORK_TYPE", "LOKR_FACTOR"} <= h, sorted(h & {"NETWORK_TYPE", "LOKR_FACTOR"}))
if "finetune" in _kinds:
    h = hidden(K2, FAMILY_KIND="finetune", NETWORK_TYPE="LoKR (Kronecker)")
    ck("fine-tune -> rank, alpha, Network Type, factor and Resume all hidden",
       {"NETWORK_TYPE", "LOKR_FACTOR", "NETWORK_DIM", "NETWORK_ALPHA", "RESUME_TRAINING"} <= h,
       sorted(h & {"NETWORK_TYPE", "LOKR_FACTOR", "NETWORK_DIM", "NETWORK_ALPHA", "RESUME_TRAINING"}))

# --- 4. the command -------------------------------------------------------------------------
c = cmd(K2, NETWORK_TYPE="LoKR (Kronecker)", LOKR_FACTOR="16")
ck("LoKR -> command carries --network_type lokr --lokr_factor 16",
   "--network_type" in c and c[c.index("--network_type") + 1] == "lokr"
   and c[c.index("--lokr_factor") + 1] == "16")
c = cmd(K2, NETWORK_TYPE="LoRA (standard)")
ck("standard LoRA -> no network_type flag at all", "--network_type" not in c)
# A slider is always a plain LoRA (its strength is the dial), so LoKR must never reach a slider's command, even
# when it is the Network Type still selected from before.
if "slider" in _kinds:
    c = cmd(K2, FAMILY_KIND="slider", NETWORK_TYPE="LoKR (Kronecker)", LOKR_FACTOR="16")
    ck("slider -> LoKR never reaches the command", "--network_type" not in c)

# --- 5. built-in presets --------------------------------------------------------------------
_presets = [(d.key, p["name"], p["values"]) for d in training_families() for p in FORM.form(d)["presets"]]
ck("every family ships built-in presets", all(FORM.form(d)["presets"] for d in training_families()))
_bad = [f"{k}: {n}" for k, n, v in _presets if v.get("NETWORK_TYPE") not in (None, "LoRA (standard)")]
ck(f"  no built-in preset (of {len(_presets)}) picks LoKR", not _bad, _bad[:3])
_first = FORM.form(K2)["presets"][0]
back = FORM.apply_preset(K2, dict(defaults(K2), NETWORK_TYPE="LoKR (Kronecker)"), _first["values"], whole=True)
ck("loading a built-in preset resets Network Type to standard", back.get("NETWORK_TYPE") == "LoRA (standard)",
   back.get("NETWORK_TYPE"))
saved = FORM.to_preset(K2, dict(defaults(K2), NETWORK_TYPE="LoKR (Kronecker)", LOKR_FACTOR="12"))
ck("a saved preset captures NETWORK_TYPE + LOKR_FACTOR",
   saved.get("NETWORK_TYPE") == "LoKR (Kronecker)" and str(saved.get("LOKR_FACTOR")) == "12",
   {k: saved.get(k) for k in ("NETWORK_TYPE", "LOKR_FACTOR")})

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
