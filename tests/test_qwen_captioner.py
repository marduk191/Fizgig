"""Qwen3-VL captioning on the Captions tab — headless verification (no GPU, no model load)."""
import json
import os
import shutil
import sys
import tempfile

os.environ["FIZGIG_NO_PERSIST"] = "1"
# Repo root, derived from this file's location -- was hardcoded to one machine's
# absolute path, which made the whole suite unrunnable anywhere else.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import tkinter as tk
import lora_trainer_gui as G
from fizgig.krea2.embedder import (CAPTION_TASKS, DEFAULT_CAPTION_TASK,
                                   ENCODE_SYSTEM_DESCRIPTOR, CAPTION_INSTRUCTION,
                                   SUBJECT_RULE, _strip_caption_preamble)

G.LAST_USED_FILE = os.path.join(os.environ["TEMP"], "nope", ".last_used.json")

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


# A stand-in "text encoder file" — only its existence is checked by the gate.
TMP = tempfile.mkdtemp()
FAKE_TE = os.path.join(TMP, "qwen3vl_4b_bf16.safetensors")
open(FAKE_TE, "wb").write(b"0")

root = tk.Tk()
root.withdraw()
g = G.LoRATrainerGUI(root)

# --- 1. gating on the file, not on model family -----------------------------------------
g.prefs_vars["krea2_text_encoder"].set("")
root.update()
ck("no TE path -> 3 Florence models only", g._caption_model_values() == G.FLORENCE_MODELS,
   g._caption_model_values())

g.prefs_vars["krea2_text_encoder"].set(FAKE_TE)
root.update()
vals = list(g.caption_model_combo.cget("values"))
ck("TE path set -> Qwen3-VL appears without restart", G.QWEN_CAPTION_MODEL in vals, vals)
ck("  ...and it is offered regardless of model family (Klein selected)",
   g.architecture_var.get().startswith("Flux 2 Klein") and G.QWEN_CAPTION_MODEL in vals)

# --- 2. task list swaps with the model ---------------------------------------------------
g.caption_model_var.set(G.QWEN_CAPTION_MODEL)
g._on_caption_model_changed()
root.update()
qwen_tasks = list(g.caption_task_combo.cget("values"))
ck("Qwen selected -> every shipped preset + Custom",
   len(qwen_tasks) == len(CAPTION_TASKS) + 1 and qwen_tasks[-1] == G.QWEN_CUSTOM_TASK,
   qwen_tasks)
ck("  the style preset is offered", CAPTION_TASKS["style"][0] in qwen_tasks, qwen_tasks)
ck("  default task is the doctrine one",
   g.caption_task_var.get() == CAPTION_TASKS[DEFAULT_CAPTION_TASK][0], g.caption_task_var.get())
ck("  Edit instructions button shown", bool(g.caption_edit_instr_btn.winfo_manager()))
ck("  max tokens seeded from the task", g.caption_max_tokens_var.get() == "120",
   g.caption_max_tokens_var.get())

g.caption_model_var.set(G.FLORENCE_DEFAULT_MODEL)
g._on_caption_model_changed()
root.update()
ck("Florence selected -> task tokens back",
   list(g.caption_task_combo.cget("values")) == G.FLORENCE_TASKS)
ck("  Edit instructions button hidden", not g.caption_edit_instr_btn.winfo_manager())

# --- 3. instruction resolution -----------------------------------------------------------
# Isolate from the real prefs.json. FIZGIG_NO_PERSIST stops the GUI WRITING prefs, but it still
# READS them at startup — so a developer who has edited these presets in the app would see this
# section fail against their own instructions rather than the shipped ones.
g.prefs.pop("caption_qwen_instructions", None)
g.prefs.pop("caption_qwen_instruction", None)

g.caption_model_var.set(G.QWEN_CAPTION_MODEL)
g._on_caption_model_changed()
for key, (label, instr, tok) in CAPTION_TASKS.items():
    g.caption_task_var.set(label)
    ck(f"  task '{key}' resolves to its own instruction",
       g._resolve_caption_instruction() == instr)

g.prefs["caption_qwen_instructions"] = {"custom": "MY CUSTOM INSTRUCTION"}
g.caption_task_var.set(G.QWEN_CUSTOM_TASK)
ck("Custom… resolves to the saved prefs instruction",
   g._resolve_caption_instruction() == "MY CUSTOM INSTRUCTION")
g.prefs.pop("caption_qwen_instructions", None)
ck("Custom… with nothing saved falls back to the default task's text",
   g._resolve_caption_instruction() == CAPTION_TASKS[DEFAULT_CAPTION_TASK][1])

# --- 3b. each preset is independently editable -------------------------------------------
_TR, _SH, _EX = (CAPTION_TASKS["training"][0], CAPTION_TASKS["short"][0],
                 CAPTION_TASKS["exhaustive"][0])
g.prefs["caption_qwen_instructions"] = {"training": "MY TRAINING", "exhaustive": "MY EXHAUSTIVE"}
ck("two presets hold different edits at once",
   g._caption_instruction_for_task(_TR) == "MY TRAINING"
   and g._caption_instruction_for_task(_EX) == "MY EXHAUSTIVE")
ck("  an unedited preset keeps its shipped text",
   g._caption_instruction_for_task(_SH) == CAPTION_TASKS["short"][1])
ck("  edited flag is per preset",
   g._caption_task_is_edited(_TR) and not g._caption_task_is_edited(_SH))
ck("  builtin_only always returns the shipped text",
   g._caption_instruction_for_task(_TR, builtin_only=True) == CAPTION_TASKS["training"][1])

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
ck("  'style' has budget to finish its sentences", CAPTION_TASKS["style"][2] >= 160,
   CAPTION_TASKS["style"][2])
# Nothing is appended to a style caption, so the whole look rides on the trigger word the GUI
# prepends — which lands in front of the caption, where _strip_caption_preamble looks. It must
# leave a real caption alone and only remove an actual preamble.
ck("  a captioned style line survives preamble stripping",
   _strip_caption_preamble("a red car parked on a street") == "a red car parked on a street")
ck("  ...whereas a LEADING 'the image is a' would have been stripped",
   not _strip_caption_preamble("the image is a watercolour of a red car").startswith("the image"))
# 'short' is excluded: it is a single clause (subject, action, setting) and never asked for
# lighting in the first place — nothing to preserve there.
for _k in ("training", "detailed", "exhaustive"):
    ck(f"  identity preset '{_k}' still asks for lighting", "lighting" in CAPTION_TASKS[_k][1])

# auto-recaption maps attempt 1 -> Training caption, attempt 2 -> Exhaustive detail.
# Never "whatever the tab is set to".
ck("attempt 1 uses the training override", g._caption_overrides().get("training") == "MY TRAINING")
ck("attempt 2 uses the exhaustive override",
   g._caption_overrides().get("exhaustive") == "MY EXHAUSTIVE")
g.caption_task_var.set(_SH)
ck("  selecting another task doesn't change what the trainer gets",
   g._caption_overrides().get("training") == "MY TRAINING"
   and g._caption_overrides().get("exhaustive") == "MY EXHAUSTIVE")
g.prefs["caption_qwen_instructions"] = {"short": "ONLY SHORT EDITED"}
ck("  editing only a non-recaption preset sends nothing to the trainer",
   not g._caption_overrides().get("training") and not g._caption_overrides().get("exhaustive"))

# legacy single-slot pref migrates into Custom and doesn't leak into the presets
g.prefs.pop("caption_qwen_instructions", None)
g.prefs["caption_qwen_instruction"] = "OLD SINGLE SLOT"
ck("legacy single slot migrates into Custom",
   g._caption_instruction_for_task(G.QWEN_CUSTOM_TASK) == "OLD SINGLE SLOT")
ck("  and does not leak into the presets",
   g._caption_instruction_for_task(_TR) == CAPTION_TASKS["training"][1])
g.prefs.pop("caption_qwen_instruction", None)
g.caption_task_var.set(_TR)

# --- 4. the model choice reaches the captioning worker ------------------------------------
# Captioning moved out of the GUI process in #93: _generate_ai_caption and its in-process
# generate_qwen_caption / generate_florence_caption router are gone, replaced by a persistent
# batch_caption subprocess. The routing decision still exists, it just travels as JSON now — so
# assert it where it actually lives, in the worker config and the cache key that decides whether
# a running worker can be reused or has to be restarted.
import json as _json  # noqa: E402

g.caption_model_var.set(G.QWEN_CAPTION_MODEL)
_qwen_cfg = _json.load(open(g._write_caption_worker_config(), encoding="utf-8"))
_qwen_key = g._caption_worker_config_key()
g.caption_model_var.set(G.FLORENCE_DEFAULT_MODEL)
_flor_cfg = _json.load(open(g._write_caption_worker_config(), encoding="utf-8"))
_flor_key = g._caption_worker_config_key()

ck("worker config: Qwen selected -> backend qwen",
   _qwen_cfg.get("backend") == "qwen", _qwen_cfg)
ck("worker config: Florence selected -> backend florence, and names the model",
   _flor_cfg.get("backend") == "florence"
   and _flor_cfg.get("florence_model") == G.FLORENCE_DEFAULT_MODEL, _flor_cfg)
# The key is what stops a Qwen worker being handed a Florence job (and vice versa) — if it did
# not change with the backend, switching captioner would silently reuse the wrong model.
ck("switching captioner changes the worker key, forcing a restart",
   _qwen_key != _flor_key and _qwen_key[0] == "qwen" and _flor_key[0] == "florence",
   (_qwen_key, _flor_key))

# --- 5. trigger word still prepended, now across the subprocess boundary ------------------
# save_caption_with_trigger went with the in-process captioner in #93. The trigger now crosses
# into batch_caption as a job field and is applied there, so both halves get asserted: the GUI
# must PUT it in the job, and the worker must APPLY it in the same format as before. Testing only
# one half would let a rename on either side pass while captions silently lost their trigger.
imgdir = tempfile.mkdtemp()
img = os.path.join(imgdir, "a.png")
open(img, "wb").write(b"0")

g.caption_trigger_var.set("zwxbsp")
_job_path, _ = g._write_caption_job([img])
_job = _json.load(open(_job_path, encoding="utf-8"))
ck("GUI half: the trigger reaches the worker job", _job.get("trigger") == "zwxbsp", _job)
g.caption_trigger_var.set("")
_job_path, _ = g._write_caption_job([img])
ck("  and an empty trigger travels as empty, not as a stray value",
   _json.load(open(_job_path, encoding="utf-8")).get("trigger") == "")

from fizgig.scripts.batch_caption import _write_caption as _wc  # noqa: E402

_wc(img, "a woman viewed from behind", "zwxbsp")
txt = open(os.path.splitext(img)[0] + ".txt", encoding="utf-8").read()
ck("worker half: trigger prepended", txt == "zwxbsp, a woman viewed from behind", txt)
_wc(img, "a woman viewed from behind", "")
txt = open(os.path.splitext(img)[0] + ".txt", encoding="utf-8").read()
ck("  no trigger -> no stray leading comma", txt == "a woman viewed from behind", txt)

# --- 6. training guard -------------------------------------------------------------------
class _FakeProc:
    def poll(self):
        return None          # still running


g.current_process = _FakeProc()
g.caption_model_var.set(G.FLORENCE_DEFAULT_MODEL)
ck("Florence not blocked during training", g._caption_model_blocked_by_training() is False)
g.current_process = None

# --- 7. encode system prompt is read from the module, never duplicated -------------------
src = open(os.path.join(REPO, "lora_trainer_gui.py"), encoding="utf-8").read()
ck("GUI imports the encode system prompt rather than copying it",
   "ENCODE_SYSTEM_DESCRIPTOR" in src and "Describe the image by detailing" not in src)

# --- 7b. the ~8 GB captioner is released when a heavy engine needs the VRAM ---------------
# The model lives in the batch_caption SUBPROCESS now (#93), so "release" means stopping that
# worker -- and the policy deliberately changed with it. It no longer dies on leaving the
# Captions tab (it stays warm so Regenerate is fast); it is released on ENTERING a heavy-engine
# tab (Repair Studio / Explorer / Royale, 10-20 GB each), on Unload, on Start Training, and at
# close. Asserting the old "released on leaving Captions" would now be pinning a behaviour
# upstream removed on purpose.
class _FakeProc:
    """Stands in for a live worker -- _caption_worker_alive only calls poll()."""

    def poll(self):
        return None          # still running


class _Ev:
    pass


_stopped = []
g._stop_caption_worker_async = lambda cb, **kw: (_stopped.append(kw), cb())

g.caption_process = _FakeProc()
g._captioning_running = False
ck("a live worker reads as alive", g._caption_worker_alive() is True)

# Tab labels carry step numbers ("3. Captions"), so pick the plain tab as "any tab that is NOT a
# heavy engine" rather than matching literal names that renumbering would break.
_HEAVY_NAMES = ("Repair Studio", "LoRA the Explorer", "LoRA Royale")
_heavy = _plain = None
for _tab_id in g.notebook.tabs():
    _t = g.notebook.tab(_tab_id, "text")
    if _t in _HEAVY_NAMES:
        _heavy = _heavy or _tab_id
    else:
        _plain = _plain or _tab_id
ck("a heavy-engine tab and a plain tab exist to switch between",
   _heavy is not None and _plain is not None,
   [g.notebook.tab(t, "text") for t in g.notebook.tabs()])


def _switch_to(tab_id):
    """Drive the REAL <<NotebookTabChanged>> wiring rather than calling the handler by hand --
    selecting a tab fires it once on its own, so an extra manual call double-counts."""
    g.notebook.select(_plain)
    root.update()
    _stopped.clear()
    g.notebook.select(tab_id)
    root.update()


g.caption_process = _FakeProc()
g._captioning_running = False
_switch_to(_heavy)
ck("released on entering a heavy-engine tab", len(_stopped) == 1, _stopped)
ck("  released hard, not gracefully -- the VRAM is wanted now",
   bool(_stopped) and _stopped[0].get("graceful") is False, _stopped)

g.caption_process = _FakeProc()
g._captioning_running = True
_switch_to(_heavy)
ck("  kept while a captioning job is in flight", _stopped == [], _stopped)
g._captioning_running = False

g.caption_process = None
ck("no worker -> not alive", g._caption_worker_alive() is False)
_switch_to(_heavy)
ck("  no-op when no worker is running", _stopped == [], _stopped)
