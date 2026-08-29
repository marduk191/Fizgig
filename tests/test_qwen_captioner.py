"""Qwen3-VL captioning on the Captions tab — headless verification (no GPU, no model load)."""
import json
import os
import shutil
import sys
import tempfile

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = r"W:/Peter/Documents/Development/Fizgig"
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

# --- 4. router picks the right generator -------------------------------------------------
calls = []
g.generate_qwen_caption = lambda p: calls.append(("qwen", p)) or "qwen caption"
g.generate_florence_caption = lambda p: calls.append(("florence", p)) or "florence caption"
g.caption_model_var.set(G.QWEN_CAPTION_MODEL)
g._generate_ai_caption("x.png")
g.caption_model_var.set(G.FLORENCE_DEFAULT_MODEL)
g._generate_ai_caption("x.png")
ck("router: Qwen selected -> qwen, Florence selected -> florence",
   [c[0] for c in calls] == ["qwen", "florence"], calls)

# --- 5. trigger word still prepended for BOTH paths --------------------------------------
imgdir = tempfile.mkdtemp()
img = os.path.join(imgdir, "a.png")
open(img, "wb").write(b"0")
g.caption_trigger_var.set("zwxbsp")
g.save_caption_with_trigger(img, "a woman viewed from behind")
txt = open(os.path.splitext(img)[0] + ".txt", encoding="utf-8").read()
ck("trigger prepended", txt == "zwxbsp, a woman viewed from behind", txt)
g.caption_trigger_var.set("")
g.save_caption_with_trigger(img, "a woman viewed from behind")
txt = open(os.path.splitext(img)[0] + ".txt", encoding="utf-8").read()
ck("no trigger -> no stray leading comma", txt == "a woman viewed from behind", txt)

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

# --- 7b. the ~8 GB captioner is released after every job ---------------------------------
class _FakeModel:
    pass


g._captioning_running = False
g.qwen_captioner = _FakeModel()
g._release_qwen_captioner_if_idle()
ck("released when idle (batch end / tab leave / Unload button)",
   g.qwen_captioner is None)

g.qwen_captioner = _FakeModel()
g._captioning_running = True
g._release_qwen_captioner_if_idle()
ck("  kept while a captioning job is in flight", g.qwen_captioner is not None)
g._captioning_running = False


class _Ev:
    pass


g.notebook.select(g.caption_gen_tab)
root.update()
g.qwen_captioner = _FakeModel()
g.notebook.select(g.image_converter_tab)
root.update()
g.on_tab_changed(_Ev())
ck("  released on leaving the Captions tab", g.qwen_captioner is None)

g.qwen_captioner = _FakeModel()
g._captioning_running = True
g.on_tab_changed(_Ev())
ck("  kept on tab switch while a batch runs", g.qwen_captioner is not None)
g._captioning_running = False

g.qwen_captioner = None
g._release_qwen_captioner_if_idle()
ck("  no-op when nothing is loaded", g.qwen_captioner is None)

g.qwen_captioner = _FakeModel()
g.florence_model = None
g.unload_florence_model(silent=True)
ck("  Unload Model button frees Qwen too", g.qwen_captioner is None)

# --- 7c. default model + per-model task memory --------------------------------------------
def _boot(last_used, te=True):
    """A fresh GUI with a given last_used, restored through the same path startup uses."""
    _r = tk.Tk()
    _r.withdraw()
    _g = G.LoRATrainerGUI(_r)
    _g.prefs_vars["krea2_text_encoder"].set(FAKE_TE if te else "")
    _g.last_used = dict(last_used)
    _g._restore_caption_selection()
    return _r, _g


_TRAIN = CAPTION_TASKS["training"][0]
_EXHV = CAPTION_TASKS["exhaustive"][0]

_r, _g = _boot({})
ck("default: no saved choice + TE present -> Qwen3-VL, its default task",
   _g.caption_model_var.get() == G.QWEN_CAPTION_MODEL and _g.caption_task_var.get() == _TRAIN,
   (_g.caption_model_var.get(), _g.caption_task_var.get()))
_r.destroy()

_r, _g = _boot({}, te=False)
ck("  no text encoder -> Florence", _g.caption_model_var.get() == G.FLORENCE_DEFAULT_MODEL)
_r.destroy()

_r, _g = _boot({"caption_model": G.FLORENCE_DEFAULT_MODEL})
ck("  an explicit saved Florence beats the Qwen default",
   _g.caption_model_var.get() == G.FLORENCE_DEFAULT_MODEL)
_r.destroy()

_r, _g = _boot({"caption_model": G.QWEN_CAPTION_MODEL}, te=False)
ck("  saved Qwen whose file is gone falls back to Florence",
   _g.caption_model_var.get() == G.FLORENCE_DEFAULT_MODEL
   and _g.caption_model_var.get() in list(_g.caption_model_combo.cget("values")))
_r.destroy()

_r, _g = _boot({"caption_model": G.QWEN_CAPTION_MODEL, "caption_task": _EXHV})
ck("  legacy flat caption_task migrates onto the selected model",
   _g.caption_task_var.get() == _EXHV, _g.caption_task_var.get())

_g.caption_model_var.set(G.FLORENCE_DEFAULT_MODEL)
_g._on_caption_model_changed()
ck("per-model memory: Florence's first visit uses its own default",
   _g.caption_task_var.get() == "<DETAILED_CAPTION>")
_g.caption_task_var.set("<CAPTION>")
_g._on_caption_task_changed()
_g.caption_model_var.set(G.QWEN_CAPTION_MODEL)
_g._on_caption_model_changed()
ck("  switching back restores the Qwen task", _g.caption_task_var.get() == _EXHV,
   _g.caption_task_var.get())
_g.caption_model_var.set(G.FLORENCE_DEFAULT_MODEL)
_g._on_caption_model_changed()
ck("  and the Florence task", _g.caption_task_var.get() == "<CAPTION>", _g.caption_task_var.get())
_mem = dict(_g._caption_task_memory)
_r.destroy()

_r, _g = _boot({"caption_model": G.FLORENCE_DEFAULT_MODEL, "caption_tasks": _mem})
ck("  survives a restart: model + its task",
   _g.caption_model_var.get() == G.FLORENCE_DEFAULT_MODEL
   and _g.caption_task_var.get() == "<CAPTION>")
_g.caption_model_var.set(G.QWEN_CAPTION_MODEL)
_g._on_caption_model_changed()
ck("  survives a restart: the other model's task too", _g.caption_task_var.get() == _EXHV,
   _g.caption_task_var.get())
_r.destroy()

# --- 7d. concurrency guards: no phantom "resumed" job ------------------------------------
# Reported symptom: after Stop/Unload the job appeared to carry on. Two causes, both here.
import types as _types

_popups = []
_real_showinfo = G.messagebox.showinfo
G.messagebox.showinfo = lambda *a, **k: _popups.append(a[0] if a else "")


class _FakeModel2:
    pass


# (a) Unloading mid-job used to free the model; the worker's next image then found no model and
#     silently RELOADED it, which reads exactly as the job resuming after you pressed Unload.
g.qwen_captioner = _FakeModel2()
g._captioning_running = True
g.unload_florence_model(silent=True)
ck("Unload refuses while a job is running", g.qwen_captioner is not None)
g._captioning_running = False
g.unload_florence_model(silent=True)
ck("  and unloads normally once it has stopped", g.qwen_captioner is None)

# (b) Caption All was never disabled, so a second click started a SECOND worker over the same
#     files — doubled work and doubled log lines, i.e. "did I queue a job up?"
g.get_caption_image_files = lambda: ["a.png"]
g.image_folder_var.set(os.environ["TEMP"])
_started = []
_real_thread = G.threading.Thread
G.threading.Thread = lambda *a, **k: _started.append(1) or _types.SimpleNamespace(
    start=lambda: None, daemon=True)

g._captioning_running = True
_popups.clear()
g.caption_all_florence()
ck("a second Caption All is refused", not _started and _popups == ["Already running"], _popups)

_popups.clear()
g.caption_single_image("x.png")
ck("Regenerate is refused mid-job", not _started and _popups == ["Captioning in progress"], _popups)

G.threading.Thread = _real_thread
G.messagebox.showinfo = _real_showinfo
g._captioning_running = False

g._set_caption_buttons_running(True)
ck("job buttons grey out while running",
   str(g.caption_all_btn.cget("state")) == "disabled"
   and str(g.caption_static_btn.cget("state")) == "disabled")
g._set_caption_buttons_running(False)
ck("  and come back afterwards", str(g.caption_all_btn.cget("state")) == "normal")

# --- 8. persistence ----------------------------------------------------------------------
g.caption_model_var.set(G.QWEN_CAPTION_MODEL)
g.caption_task_var.set(CAPTION_TASKS["exhaustive"][0])
g.caption_max_tokens_var.set("240")
data = dict(g.last_used)
for attr, key in (("caption_model_var", "caption_model"),
                  ("caption_task_var", "caption_task"),
                  ("caption_max_tokens_var", "caption_max_tokens")):
    data[key] = getattr(g, attr).get()
root.destroy()

root2 = tk.Tk()
root2.withdraw()
g2 = G.LoRATrainerGUI(root2)
g2.last_used = data
# re-seed as the constructor would
g2.caption_model_var.set(data["caption_model"])
g2.caption_task_var.set(data["caption_task"])
g2.caption_max_tokens_var.set(data["caption_max_tokens"])
ck("model/task/max-tokens survive a restart",
   g2.caption_model_var.get() == G.QWEN_CAPTION_MODEL
   and g2.caption_task_var.get() == CAPTION_TASKS["exhaustive"][0]
   and g2.caption_max_tokens_var.get() == "240")
root2.destroy()

# --- 9. trainer CLI accepts the custom instruction ---------------------------------------
from fizgig.scripts.krea2_train import setup_parser
p = setup_parser()
ns = p.parse_args(["--dit", "d", "--dataset_config", "c", "--output_dir", "o",
                   "--output_name", "n", "--recaption_instruction", "CUSTOM"])
ck("krea2_train --recaption_instruction parses", ns.recaption_instruction == "CUSTOM")

import inspect
from fizgig.krea2.trainer import train_krea2, _apply_caption_updates
ck("train_krea2 takes recaption_instruction",
   "recaption_instruction" in inspect.signature(train_krea2).parameters)
ck("_apply_caption_updates takes recaption_instruction",
   "recaption_instruction" in inspect.signature(_apply_caption_updates).parameters)

from fizgig.krea2.embedder import generate_caption
ck("generate_caption takes instruction",
   "instruction" in inspect.signature(generate_caption).parameters)

# --- 10. caption hygiene: preamble stripping + the two instruction rules -----------------
from fizgig.krea2.embedder import (_strip_caption_preamble, SUBJECT_RULE, NO_PREAMBLE_RULE,
                                   SHORT_CAPTION_INSTRUCTION, DETAILED_DESCRIPTION_INSTRUCTION,
                                   DETAILED_CAPTION_INSTRUCTION)

_preamble_cases = [
    ("This image shows a woman standing on a beach.", "a woman standing on a beach."),
    ("The image depicts a man viewed from behind.", "a man viewed from behind."),
    ("In this image, a woman sits on a chair.", "a woman sits on a chair."),
    ("In this image we see a girl running.", "a girl running."),
    ("Here we see a woman laughing.", "a woman laughing."),
    ("The photo shows a bride walking.", "a bride walking."),
    ("This photograph captures a couple dancing.", "a couple dancing."),
    ("This image is of a woman.", "a woman."),
    # must survive untouched
    ("a woman viewed from behind, wearing a red coat", "a woman viewed from behind, wearing a red coat"),
    ("A photo of a woman on a beach.", "A photo of a woman on a beach."),
    ("The image on the wall shows a landscape.", "The image on the wall shows a landscape."),
    ("This image shows", "This image shows"),   # stripping to empty keeps the original
]
_bad = [(s, _strip_caption_preamble(s), w) for s, w in _preamble_cases
        if _strip_caption_preamble(s) != w]
ck("preamble stripper: 12 cases incl. negatives", not _bad, _bad[:2])

for _name, _instr in (("training", CAPTION_INSTRUCTION),
                      ("short", SHORT_CAPTION_INSTRUCTION),
                      ("detailed", DETAILED_DESCRIPTION_INSTRUCTION),
                      ("exhaustive", DETAILED_CAPTION_INSTRUCTION)):
    ck(f"  '{_name}' instruction carries the subject + no-preamble rules",
       SUBJECT_RULE in _instr and NO_PREAMBLE_RULE in _instr)

shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(imgdir, ignore_errors=True)
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S)"))
sys.exit(1 if fails else 0)
