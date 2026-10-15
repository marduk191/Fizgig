"""Qwen3-VL captioning (Fizgig 8 web Captions tab) — headless, no GPU, no model load.

The desktop Captions tab drove this through Tk widgets. Fizgig 8 serves the same choices from
fizgig.web.captions (options / instruction_override / the worker job) and the run's auto-recaption from the launch
plan. So the checks below drive those; section 3c (the caption presets' own text) is unchanged, because it pins the
words themselves.

Run: venv/Scripts/python.exe tests/test_qwen_captioner.py
"""
import io
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# src only. The repo root holds Fizgig 8's fizgig.py, which would shadow the fizgig package.
sys.path.insert(0, os.path.join(REPO, "src"))

from fizgig.krea2.caption_text import (CAPTION_TASKS, DEFAULT_CAPTION_TASK,  # noqa: E402
                                       ENCODE_SYSTEM_DESCRIPTOR, SUBJECT_RULE)
from fizgig.web import captions as CAP  # noqa: E402

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


TMP = tempfile.mkdtemp(prefix="fizgig_qwencap_")
FAKE_TE = os.path.join(TMP, "qwen3vl_4b_bf16.safetensors")
open(FAKE_TE, "wb").write(b"0")


def tasks(prefs=None, captioner=FAKE_TE):
    return {t["key"]: t for t in CAP.options(prefs or {}, captioner)["qwen_tasks"]}


# --- 1. gating on the file, not on model family -----------------------------------------
o = CAP.options({}, "")
ck("no text encoder set -> Florence models only, Qwen not offered",
   CAP.QWEN_MODEL not in o["models"] and o["qwen_available"] is False, o["models"])
o = CAP.options({}, FAKE_TE)
ck("text encoder set -> Qwen3-VL is offered", CAP.QWEN_MODEL in o["models"] and o["qwen_available"] is True)
ck("  ...and is the default model then", o["default_model"] == CAP.QWEN_MODEL, o["default_model"])
# options() takes no family at all: the captioner is offered whatever model family is being trained.

# --- 2. the task lists -------------------------------------------------------------------
t = tasks()
ck("Qwen -> every shipped preset + Custom", set(t) == set(CAPTION_TASKS) | {"custom"}, sorted(t))
ck("  Custom is last", CAP.options({}, FAKE_TE)["qwen_tasks"][-1]["key"] == "custom")
ck("  the style preset is offered", "style" in t)
ck("  default task is the doctrine one",
   CAP.options({}, FAKE_TE)["default_qwen_task"] == CAPTION_TASKS[DEFAULT_CAPTION_TASK][0])
ck("  each preset carries its suggested length",
   all(t[k]["tokens"] == CAPTION_TASKS[k][2] for k in CAPTION_TASKS))
ck("Florence keeps its own task tokens", list(CAP.options({}, FAKE_TE)["florence_tasks"]) == list(CAP.FLORENCE_TASKS))

# --- 3. instruction resolution -----------------------------------------------------------
for key, (label, instr, tok) in CAPTION_TASKS.items():
    ck(f"  task '{key}' resolves to its own instruction", t[key]["instruction"] == instr)
ck("Custom with an override resolves to it",
   tasks({"caption_qwen_instructions": {"custom": "MY CUSTOM INSTRUCTION"}})["custom"]["instruction"]
   == "MY CUSTOM INSTRUCTION")
ck("Custom with nothing saved falls back to the default task's text",
   t["custom"]["instruction"] == CAPTION_TASKS[DEFAULT_CAPTION_TASK][1])

# --- 3b. each preset is independently editable -------------------------------------------
p2 = {"caption_qwen_instructions": {"training": "MY TRAINING", "exhaustive": "MY EXHAUSTIVE"}}
t2 = tasks(p2)
ck("two presets hold different edits at once",
   t2["training"]["instruction"] == "MY TRAINING" and t2["exhaustive"]["instruction"] == "MY EXHAUSTIVE")
ck("  an unedited preset keeps its shipped text", t2["short"]["instruction"] == CAPTION_TASKS["short"][1])
ck("  edited flag is per preset", t2["training"]["edited"] and not t2["short"]["edited"])
ck("  builtin always returns the shipped text", t2["training"]["builtin"] == CAPTION_TASKS["training"][1])
ck("saving a preset back to its shipped text drops the override",
   "training" not in CAP.instruction_override(p2, "training", CAPTION_TASKS["training"][1]))
ck("  while another preset's edit is kept",
   CAP.instruction_override(p2, "training", CAPTION_TASKS["training"][1]).get("exhaustive") == "MY EXHAUSTIVE")
legacy = {"caption_qwen_instruction": "OLD SINGLE SLOT"}
ck("legacy single slot migrates into Custom", tasks(legacy)["custom"]["instruction"] == "OLD SINGLE SLOT")
ck("  and does not leak into the presets", tasks(legacy)["training"]["instruction"] == CAPTION_TASKS["training"][1])

# --- 3c. the style preset says the opposite of the identity ones ---------------------------
# These guard against one accident: writing the style preset by copy-pasting an identity one.
# SUBJECT_RULE can only produce 'a woman'/'a man'/'a girl'/'a boy', which is wrong for a dataset
# of landscapes and objects; and lighting must NOT be captioned for a style, or the look only
# fires under the lighting it was trained on. The identity presets ask for lighting correctly —
# there it varies and you want it steerable — so the two rules genuinely coexist.
_STYLE = CAPTION_TASKS["style"][1]
ck("  'style' does not use the person-only subject rule", SUBJECT_RULE not in _STYLE)
ck("  'style' never asks for lighting", "lighting" not in _STYLE.lower())
ck("  'style' excludes the style itself", "zero references to the image style" in _STYLE)
ck("  'style' asks for the contents", "factual contents of what is depicted" in _STYLE)
# The brevity is the finding, not an oversight. A four-fragment rule stack scored better on leak
# counting (1 caption in 9 vs 7) and trained worse on real runs across both Krea 2 and Klein: the
# short form yields ~2.4x richer captions, and a style word appearing in 7 of 9 behaves like a tag
# rather than the noise a 4-in-9 split creates. Anyone re-stacking rules here trips this.
ck("  'style' stays short", len(_STYLE.split()) <= 30, len(_STYLE.split()))
# Those richer captions run ~70 words; at the ~90 tokens the old presets used, every caption on
# the test set was cut off mid-word.
ck("  'style' has budget to finish its sentences", CAPTION_TASKS["style"][2] >= 160, CAPTION_TASKS["style"][2])
from fizgig.krea2.embedder import _strip_caption_preamble  # noqa: E402
ck("  a captioned style line survives preamble stripping",
   _strip_caption_preamble("a red car parked on a street") == "a red car parked on a street")
ck("  ...whereas a LEADING 'the image is a' would have been stripped",
   not _strip_caption_preamble("the image is a watercolour of a red car").startswith("the image"))
for _k in ("training", "detailed", "exhaustive"):
    ck(f"  identity preset '{_k}' still asks for lighting", "lighting" in CAPTION_TASKS[_k][1])

# --- 3d. auto-recaption: attempt 1 -> Training caption, attempt 2 -> Exhaustive detail ------
# Never "whatever the tab is set to". The launch plan maps the two edits to the trainer's two flags.
from fizgig.families import form as FORM  # noqa: E402
from fizgig.families import launch as L  # noqa: E402
from fizgig.families.registry import get as get_family  # noqa: E402

K2 = get_family("krea2")


def recaption_cmd(overrides):
    rows = {r["key"]: r for s in FORM.form(K2)["sections"] for r in s["rows"] if r.get("key")}
    vals = {k: r.get("default") for k, r in rows.items()}
    vals.update(LORA_NAME="t", LORA_OUTPUT_DIR=TMP, DATASET_CONFIG=os.path.join(TMP, "d.toml"), BATCH_SIZE="1",
                LOSS_WATCH_DETECT=True, AUTO_RECAPTION=True)
    inp = FORM.inputs(K2, vals, models={k: "m" for k in FORM.model_keys(K2)}, python=sys.executable,
                      repo_dir=REPO, image_folder="", cache_root=TMP, samples_dir=TMP, captioner=FAKE_TE,
                      caption_overrides=overrides)
    inp["batch_size"] = 1
    inp.setdefault("loss_watch", {})["recaption"] = True
    return L.train_command(K2, inp, L.LaunchPlan())


c = recaption_cmd({"training": "MY TRAINING", "exhaustive": "MY EXHAUSTIVE", "short": "ONLY SHORT"})
if "--captioner" in c:
    ck("attempt 1 gets the Training override",
       "--recaption_instruction" in c and c[c.index("--recaption_instruction") + 1] == "MY TRAINING")
    ck("attempt 2 gets the Exhaustive override",
       "--recaption_instruction_detailed" in c
       and c[c.index("--recaption_instruction_detailed") + 1] == "MY EXHAUSTIVE")
    ck("  an edit to any other preset never reaches the trainer", "ONLY SHORT" not in c)
    c = recaption_cmd({"short": "ONLY SHORT"})
    ck("  editing only a non-recaption preset sends no instruction at all",
       "--recaption_instruction" not in c and "--recaption_instruction_detailed" not in c)
else:
    ck("auto-recaption reaches the command (Batch 1, captioner set)", False, "no --captioner in command")

# --- 4. the model choice reaches the captioning worker ------------------------------------
cq = CAP._worker_config(CAP.QWEN_MODEL, FAKE_TE, clips=False)
ck("worker config: Qwen -> backend qwen + its text encoder",
   cq.get("backend") == "qwen" and cq.get("text_encoder") == FAKE_TE, cq)
cf = CAP._worker_config(CAP.FLORENCE_MODELS[0], FAKE_TE, clips=True)
ck("worker config: Florence -> backend florence, names the model, clips on",
   cf.get("backend") == "florence" and cf.get("florence_model") == CAP.FLORENCE_MODELS[0]
   and cf.get("include_video") is True, cf)
if CAP.FLORENCE_REVISIONS.get(CAP.FLORENCE_MODELS[0]):
    ck("  ...pinned to its revision (trust_remote_code runs what it downloads)",
       cf.get("florence_revision") == CAP.FLORENCE_REVISIONS[CAP.FLORENCE_MODELS[0]])

# --- 5. the trigger word, across the subprocess boundary ----------------------------------
# GUI half: ai_captions writes the trigger into the worker's job. JOB_DIR is redirected away from the real
# user_data, and the worker spawn is stubbed (nothing loads).
from PIL import Image  # noqa: E402

CAP.JOB_DIR = os.path.join(TMP, "jobs")
imgdir = os.path.join(TMP, "imgs")
os.makedirs(imgdir)
Image.new("RGB", (8, 8)).save(os.path.join(imgdir, "a.png"))


class _FakeWorker:
    def __init__(self):
        self.stdin, self.stdout = io.StringIO(), iter(())

    def poll(self):
        return None


CAP._popen = lambda cmd, prefs, stdin=False: _FakeWorker()
try:
    CAP.ai_captions({}, FAKE_TE, imgdir, CAP.QWEN_MODEL, "", "an instruction", 120, "zwxbsp", True, False)
    job = json.load(open(os.path.join(CAP.JOB_DIR, "job.json"), encoding="utf-8"))
    ck("GUI half: the trigger reaches the worker job", job.get("trigger") == "zwxbsp", job.get("trigger"))
except Exception as e:
    ck("GUI half: the trigger reaches the worker job", False, f"{type(e).__name__}: {e}")

from fizgig.scripts.batch_caption import _write_caption as _wc  # noqa: E402

img = os.path.join(imgdir, "a.png")
_wc(img, "a woman viewed from behind", "zwxbsp")
txt = open(os.path.splitext(img)[0] + ".txt", encoding="utf-8").read()
ck("worker half: trigger prepended", txt == "zwxbsp, a woman viewed from behind", txt)
_wc(img, "a woman viewed from behind", "")
txt = open(os.path.splitext(img)[0] + ".txt", encoding="utf-8").read()
ck("  no trigger -> no stray leading comma", txt == "a woman viewed from behind", txt)

# --- 6. captioning beside a training run --------------------------------------------------
from fizgig.web import server as S  # noqa: E402

S.list_gpus = lambda: [{"index": 0, "name": "card", "total_gb": 32.0, "used_gb": 26.0, "uuid": "GPU-x"}]
with S._lock:
    S._run["proc"] = None
try:
    S._caption_gpu_check(8.0)
    ck("no run training -> a caption job may start", True)
except ValueError as e:
    ck("no run training -> a caption job may start", False, e)


class _LiveRun:
    def poll(self):
        return None


with S._lock:
    S._run["proc"] = _LiveRun()
try:
    S._caption_gpu_check(8.0)          # 6 GB free, needs 8 + 3
    ck("a run training on a card without room -> refused", False, "not refused")
except ValueError as e:
    ck("a run training on a card without room -> refused, and says why", "A run is training" in str(e), e)
try:
    S._caption_gpu_check(1.0)          # 6 GB free, needs 1 + 3
    ck("  ...but a small job that fits beside it may start", True)
except ValueError as e:
    ck("  ...but a small job that fits beside it may start", False, e)
with S._lock:
    S._run["proc"] = None

# --- 7. the encode system prompt is read from the module, never duplicated -----------------
_web = "".join(open(os.path.join(REPO, "src", "fizgig", "web", f), encoding="utf-8").read()
               for f in ("captions.py", "server.py"))
ck("the web layer never copies the encode system prompt", ENCODE_SYSTEM_DESCRIPTOR[:40] not in _web)

# --- 7b. the ~8 GB caption model makes way for a run -------------------------------------
_order = []
CAP.unload = lambda: _order.append("unload") or False
CAP.status = lambda: {"state": "idle"}
S.captions.unload, S.captions.status = CAP.unload, CAP.status


def _plan_spy(*a, **k):
    _order.append("plan")
    raise RuntimeError("stop here")


S._plan = _plan_spy
with S._lock:
    S._run.update(proc=None, state="idle")
try:
    S._start("krea2", {}, imgdir)
except RuntimeError:
    pass
ck("a training start unloads the caption model BEFORE planning the run", _order[:2] == ["unload", "plan"], _order)
S.captions.status = lambda: {"state": "running"}
_order.clear()
r = S._start("krea2", {}, imgdir)
ck("  and refuses while captions are being made",
   isinstance(r, dict) and "Captions are being made" in str(r.get("error")) and "plan" not in _order, r)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
