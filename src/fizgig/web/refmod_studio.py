"""The web UI's RefMod Toolkit (the desktop tab's): try RefMods and ReFLoRAs on MiniMax H3 before ComfyUI - a still or a
short clip with them and the same seed without them, side by side. The renders run on the GPU worker
(fizgig.web.studio refmod_render); everything here is CPU work in the server, without torch - the folder scan and each
mod's header (fizgig.minimax.refmod_meta), the job the worker renders, the ComfyUI readout, the curve presets and
setups (the desktop's files), the saved previews and strips. Bake and Save as ReFLoRA write tensors, so they run in a
short process of their own (fizgig.scripts.refmod_file)."""

import json
import os
import re
import shutil
import subprocess
import sys
import threading

from fizgig.minimax import refmod_meta as rm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRESETS_DIR = os.path.join(__import__("fizgig.user_data", fromlist=["x"]).PRESETS_DIR, "refmod_studio")
CURVES_DIR = os.path.join(PRESETS_DIR, "curves")                       # and its curve presets

SWEEPS = ("Retention 0 · 0.15 · 0.4 · 0.7 · 1", "Seeds ×4", "Frame-curve directions ×5", "Step-curve directions ×5",
          "Copies 1 · 2 · 3 · 4")
COMPARE = (("No mod (LoRA alone)", "nomod"), ("No LoRA (mods alone)", "nolora"), ("Neither (base model)", "neither"))

_lock = threading.Lock()
_cache = {}                  # (path, size, mtime) -> the file's references (headers)


def families():
    from fizgig.families.registry import workbench_families
    return workbench_families("refmod")


def info(desc, last_used):
    """The page's choices: the model's pickers (its description's video_controls), the pack's curves and limits."""
    saved = last_used.get("refmod_studio") if isinstance(last_used.get("refmod_studio"), dict) else {}
    vc = desc.video_controls if desc is not None else None
    pick = {"models": [c.label for c in (vc.refmod_checkpoints or vc.checkpoints)],
            "lengths": [n for n, _ in (vc.refmod_lengths or vc.lengths)], "sizes": [str(s) for s in vc.sizes],
            "size": [str(s) for s in vc.refmod_size], "bases": [n for n, _ in vc.bases],
            "sweep_frames": vc.sweep_frames} if vc else {}
    return {"families": [{"key": d.key, "name": d.display_name} for d in families()],
            "family": desc.key if desc is not None else "", **pick, "sweeps": list(SWEEPS),
            "compare": [c for c, _k in COMPARE], "directions": list(rm.CURVE_DIRECTIONS), "shapes": list(rm.CURVE_SHAPES),
            "retention_presets": [list(p) for p in rm.RETENTION_PRESETS], "frame_curve": list(rm.DEFAULT_FRAME_CURVE),
            "step_curve": list(rm.DEFAULT_STEP_CURVE), "max_rows": rm.MAX_ROWS, "max_copies": rm.MAX_COPIES,
            "token_cap": rm.NODE_TOKEN_CAP, "none": rm.NONE_MOD, "desktop": saved,
            "curve_presets": curve_presets(), "setups": setups()}


# ---- the mods ----------------------------------------------------------------------------------------------------
def _refs_of(path):
    try:
        st = os.stat(path)
    except OSError:
        return []
    key = (path, st.st_size, st.st_mtime)
    with _lock:
        hit = _cache.get(key)
    if hit is None:
        hit = rm.read_refmod_metas(path)
        with _lock:
            _cache[key] = hit
    return hit


def scan(folder, lora=""):
    """{mods: [meta + info line], audio: n left out, status}: the folder's visual RefMods by name, a ReFLoRA's own
    references after them (named "(ReFLoRA)" when a folder mod has the name)."""
    mods, audio = {}, 0
    if folder and os.path.isdir(folder):
        for fn in sorted(os.listdir(folder), key=str.lower):
            if not fn.lower().endswith(".safetensors"):
                continue
            for m in _refs_of(os.path.join(folder, fn)):
                if str(m.get("kind", "")) == "audio":
                    audio += 1
                else:
                    mods[m["name"]] = m
    if lora and os.path.isfile(lora) and rm.is_reflora(lora):
        for m in _refs_of(lora):
            if str(m.get("kind", "")) == "audio":
                continue
            if any(os.path.normcase(x["path"]) == os.path.normcase(m["path"]) and x.get("tensor_key") == m.get("tensor_key")
                   for x in mods.values()):
                continue
            name = m["name"] if m["name"] not in mods else f"{m['name']} (ReFLoRA)"
            mods[name] = dict(m, name=name)
    n = len(mods)
    status = (f"{n} RefMod{'s' if n != 1 else ''} in {folder or '(no folder)'}."
              + (f"  {audio} audio mod{'s' if audio != 1 else ''} not listed (bundled or standalone): the Studio renders "
                 f"visual mods only, audio mods play in ComfyUI." if audio else ""))
    return {"mods": [dict(m, info=rm.describe_meta(m)) for m in mods.values()], "audio": audio, "status": status}


def lora_info(path, folder):
    """The LoRA card's line, and a ReFLoRA's references (their picker names) for the rows."""
    path = (path or "").strip()
    if not path:
        return {"line": "No LoRA: the render uses the mods alone.", "refs": [], "ok": True}
    if not os.path.isfile(path):
        return {"line": f"File not found: {path}", "refs": [], "ok": False}
    if rm.is_reflora(path):
        names = [m["name"] for m in scan(folder, path)["mods"]
                 if os.path.normcase(os.path.abspath(m["path"])) == os.path.normcase(os.path.abspath(path))]
        n = len(names)
        return {"line": f"Opened ReFLoRA {os.path.basename(path)}: its LoRA, and its {n} RefMod{'s' if n != 1 else ''}, "
                        f"now the rows in the Mods card. Press Render: the left picture is the ReFLoRA as ComfyUI runs it.",
                "refs": names, "ok": True, "reflora": True}
    return {"line": f"LoRA {os.path.basename(path)}: renders together with the mods in the Mods card.", "refs": [],
            "ok": True}


# ---- one render --------------------------------------------------------------------------------------------------
def _f(x, d, lo=None, hi=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        v = d
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def _i(x, d):
    try:
        return int(float(str(x).strip()))
    except (TypeError, ValueError):
        return d


def _curve(c, d):
    c = list(c or d)
    return (c[0] if c[0] in rm.CURVE_DIRECTIONS else d[0], c[1] if c[1] in rm.CURVE_SHAPES else d[1],
            round(_f(c[2], 1.0, 0.0, 1.0), 2))


def _meta_rows(rows, metas):
    """[(page row, MetaRow)] for the rows whose mod is in the folder."""
    out = []
    for r in rows or []:
        m = metas.get(str(r.get("mod")))
        if m is None:
            continue
        out.append((r, rm.MetaRow(m, _f(r.get("value"), 1.0, 0.0, 1.0), _i(r.get("copies"), 1),
                                  bool(r.get("on", True)), m["name"])))
    return out


def tokens(rows, metas, retention):
    if _f(retention, 1.0) <= 0:
        return 0
    return sum(int(m.meta.get("tokens", 0)) * m.copies for _r, m in _meta_rows(rows, metas)
               if m.enabled and m.value > 0)


def ref_map(rows, metas):
    return rm.reference_map(rm.numbered_entries([m for _r, m in _meta_rows(rows, metas)]))


def job(b, metas, audio_ok, vc, overrides=None):
    """Everything one render needs (the desktop's _rms_job): the worker's job, the ComfyUI readout, the comparison and
    the key the kept comparison render is good for. `overrides` patch one dial (a sweep item)."""
    ov = overrides or {}
    rows = [dict(r) for r in (b.get("rows") or [])]
    if "copies" in ov:
        for r in rows:
            r["copies"] = ov["copies"]
    paired = _meta_rows(rows, metas)
    retention = _f(ov.get("retention", b.get("retention")), 1.0, 0.0, 1.0)
    fc = _curve(ov.get("frame_curve", b.get("frame_curve")), rm.DEFAULT_FRAME_CURVE)
    sc = _curve(ov.get("step_curve", b.get("step_curve")), rm.DEFAULT_STEP_CURVE)
    step_on = bool(ov.get("step_on", b.get("step_on")))
    scramble = _i(b.get("scramble"), -1)
    numbered = bool(b.get("numbered"))
    steps = max(1, min(40, _i(b.get("steps"), 6)))
    turbo = _f(b.get("turbo"), 0.75, 0.0, 1.5)
    w, h = _i(b.get("width"), 768), _i(b.get("height"), 768)
    w, h = max(256, w // 32 * 32), max(256, h // 32 * 32)
    lengths = dict(vc.refmod_lengths or vc.lengths)
    frames = int(ov.get("frames", lengths.get(str(b.get("frames")), 1)))     # the length's label (desktop key)
    seed = int(ov.get("seed", _i(b.get("seed"), 300)))
    prompt = str(b.get("prompt") or "").strip()
    sound = bool(b.get("sound", True)) and audio_ok
    readout = rm.comfy_readout([m for _r, m in paired], retention=retention, frame_curve=fc, scramble_seed=scramble,
                               step_curve=sc, step_on=step_on)
    refmap = rm.reference_map(rm.numbered_entries([m for _r, m in paired]))
    if numbered:                                 # the Text Encode path: the loader paragraphs, then that node
        retention, step_on = 1.0, False
        readout = (readout.split("\nApply RefMods")[0].rstrip()
                   + "\n\nH3 RefMod Text Encode\n  mods: the loader above; prompt: this prompt; connect its "
                     "conditioning to the sampler — no Apply node, no Step Curve node on this path\n  reference map: "
                   + ("  ".join(refmap) or "none")
                   + "\n  (the same numbering the node gives these mods: loader order, every copy, pictures, videos "
                     "and bundled audio counted apart; row strengths only)")
    lora = str(b.get("lora") or "").strip()
    lora = lora if lora and os.path.isfile(lora) else ""
    compare = dict(COMPARE).get(str(b.get("compare")), "nomod") if lora else "nomod"
    if lora:
        readout += (f"\n\nLoRA: {os.path.basename(lora)} at 1.0"
                    + (f" — a ReFLoRA: Load H3 RefLoRA (ComfyUI-MiniMaxH3RefLoRA) gives the patched model and its "
                       f"references from this one file (refmod_retention {retention:.2f})"
                       if rm.is_reflora(lora) else " (any LoRA loader, beside the RefMod loader above)"))
    base_key = json.dumps([prompt, seed, w, h, frames, steps, turbo, str(b.get("base")), str(b.get("model")), sound,
                           lora, compare, readout if compare == "nolora" else ""])
    eff = [(m.name, min(1.0, m.value) * (1.0 if numbered else retention)) for _r, m in paired
           if m.enabled and m.value > 0]
    describe = ", ".join(f"{n}@{s:.2f}" for n, s in eff) or "no mods"
    desc = f"{describe} · retention {retention:.2f} · frame {fc[0]}/{fc[1]}/{fc[2]:.2f}"
    if step_on:
        desc += f" · step {sc[0]}/{sc[1]}/{sc[2]:.2f}"
    desc += f" · seed {seed} · {w}×{h}×{frames} · {steps} steps · Turbo {turbo:g}"
    work = {"rows": [{"meta": m.meta, "value": m.value, "copies": m.copies, "on": m.enabled} for _r, m in paired],
            "retention": retention, "frame_curve": list(fc), "step_curve": list(sc), "step_on": step_on,
            "scramble": scramble, "numbered": numbered, "prompt": prompt, "seed": seed, "width": w, "height": h,
            "frames": frames, "steps": steps, "turbo": turbo, "sound": sound if frames > 1 else False,
            "early": bool(b.get("early", True)), "label": ov.get("label") or desc}
    return {"job": work, "readout": readout, "ref_map": refmap, "compare": compare, "lora": lora,
            "base_key": base_key, "describe": desc, "active": bool(eff)}


def sweep_items(kind, b):
    """(label, overrides) per chip for a sweep (the desktop's _rms_sweep_items_for)."""
    if kind.startswith("Retention"):
        return [(f"retention {v:g}", {"retention": v}) for v in (0.0, 0.15, 0.4, 0.7, 1.0)]
    if kind.startswith("Seeds"):
        s0 = _i(b.get("seed"), 300)
        return [(f"seed {s0 + i}", {"seed": s0 + i}) for i in range(4)]
    if kind.startswith("Frame-curve"):
        fc = _curve(b.get("frame_curve"), rm.DEFAULT_FRAME_CURVE)
        return [(d, {"frame_curve": (d, fc[1], fc[2])}) for d in rm.CURVE_DIRECTIONS]
    if kind.startswith("Step-curve"):
        sc = _curve(b.get("step_curve"), rm.DEFAULT_STEP_CURVE)
        return [("step off", {"step_on": False})] + [(f"step {d}", {"step_on": True, "step_curve": (d, sc[1], sc[2])})
                                                     for d in rm.CURVE_DIRECTIONS if d != "constant"]
    return [(f"copies {c}", {"copies": c}) for c in (1, 2, 3, 4)]


def prompt_hints(rows, metas):
    seen, out = set(), []
    for r, m in _meta_rows(rows, metas):
        if m.enabled and m.meta["path"] + str(m.meta.get("tensor_key")) not in seen:
            seen.add(m.meta["path"] + str(m.meta.get("tensor_key")))
            out.append(m.meta)
    return rm.prompt_hint(out)


# ---- curve presets, graph presets, setups (the desktop's files) ---------------------------------------------------
def _safe(name):
    s = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in str(name or "")).strip()
    if not s:
        raise ValueError("Give it a name.")
    return s


def curve_presets():
    try:
        return sorted(f[:-5] for f in os.listdir(CURVES_DIR) if f.lower().endswith(".json"))
    except OSError:
        return []


def curve_preset_save(name, frame, step, step_on, retention, overwrite=False):
    name = _safe(name)
    p = os.path.join(CURVES_DIR, name + ".json")
    if os.path.exists(p) and not overwrite:
        return {"exists": True, "name": name}
    fc, sc = _curve(frame, rm.DEFAULT_FRAME_CURVE), _curve(step, rm.DEFAULT_STEP_CURVE)
    os.makedirs(CURVES_DIR, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"frame": dict(zip(("direction", "shape", "value"), fc)),
                   "step": dict(on=bool(step_on), **dict(zip(("direction", "shape", "value"), sc))),
                   "retention": _f(retention, 1.0, 0.0, 1.0)}, f, indent=2)
    return {"ok": True, "name": name, "presets": curve_presets()}


def curve_preset_load(name):
    with open(os.path.join(CURVES_DIR, _safe(name) + ".json"), encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    fc, sc = d.get("frame") or {}, d.get("step") or {}
    if fc.get("direction") in rm.CURVE_DIRECTIONS and fc.get("shape") in rm.CURVE_SHAPES:
        out["frame_curve"] = [fc["direction"], fc["shape"], float(fc.get("value", 1.0))]
    if sc.get("direction") in rm.CURVE_DIRECTIONS and sc.get("shape") in rm.CURVE_SHAPES:
        out["step_curve"] = [sc["direction"], sc["shape"], float(sc.get("value", 1.0))]
        out["step_on"] = bool(sc.get("on", False))
    if "retention" in d:
        out["retention"] = float(d["retention"])
    return out


def curve_preset_delete(name):
    p = os.path.join(CURVES_DIR, _safe(name) + ".json")
    if os.path.isfile(p):
        os.remove(p)
    return {"ok": True, "presets": curve_presets()}


def graph_presets(folder):
    """The pack's graph presets (<mods folder>/graph_presets/*.png, legacy .json) that hold a curve."""
    d = os.path.join(folder or "", "graph_presets")
    if not folder or not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d), key=str.lower):
        if f.lower().endswith((".png", ".json")):
            spec = rm.read_graph_preset_png(os.path.join(d, f))
            if spec is not None:
                out.append({"name": os.path.splitext(f)[0], "path": os.path.join(d, f), "curve": list(spec)})
    return out


def graph_preset(path):
    spec = rm.read_graph_preset_png(path)
    if spec is None:
        raise ValueError("No curve found in that file (expected the pack's 'graph' text chunk with direction / shape "
                         "/ value).")
    return list(spec)


def setups():
    try:
        return sorted((f[:-5] for f in os.listdir(PRESETS_DIR) if f.lower().endswith(".json")), key=str.lower)
    except OSError:
        return []


def setup_save(name, data, overwrite=False):
    name = _safe(name)
    p = os.path.join(PRESETS_DIR, name + ".json")
    if os.path.exists(p) and not overwrite:
        return {"exists": True, "name": name}
    os.makedirs(PRESETS_DIR, exist_ok=True)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(p + ".tmp", p)
    return {"ok": True, "name": name, "setups": setups()}


def setup_load(name):
    with open(os.path.join(PRESETS_DIR, _safe(name) + ".json"), encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("Not a RefMod Toolkit setup.")
    return d


# ---- saving ------------------------------------------------------------------------------------------------------
def _free(path):
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"


def _folder(d):
    d = (d or "").strip()
    if not d:
        raise ValueError("Choose a folder to save into.")
    os.makedirs(d, exist_ok=True)
    return d


def save_item(item, folder, stem):
    """A render kept: its clip (MP4 with its sound) or, for a still, the PNG - never over a file."""
    src = item.get("mp4") if item.get("mp4") and os.path.isfile(item["mp4"]) else item.get("png")
    if not src or not os.path.isfile(src):
        raise ValueError("Nothing rendered yet.")
    out = _free(os.path.join(_folder(folder), stem + os.path.splitext(src)[1]))
    shutil.copyfile(src, out)
    return out


def save_strip(chips, folder, kind):
    """The sweep as one labelled contact sheet (each clip's middle frame)."""
    from PIL import Image, ImageDraw
    if not chips:
        raise ValueError("No sweep to save - render one first.")
    tiles = []
    for c in chips:
        with Image.open(c["png"]) as im:
            tiles.append((c.get("label", ""), im.convert("RGB")))
    tw, th = max(t.width for _l, t in tiles), max(t.height for _l, t in tiles)
    band = 28
    sheet = Image.new("RGB", (tw * len(tiles), th + band), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (label, t) in enumerate(tiles):
        sheet.paste(t, (i * tw, 0))
        draw.text((i * tw + 6, th + 7), label, fill=(230, 230, 230))
    out = _free(os.path.join(_folder(folder), f"refmod_sweep_{(kind or 'strip').split(' ')[0].lower()}.png"))
    sheet.save(out)
    return out


def _file_job(op):
    """Bake / Save as ReFLoRA in a short process of its own (they read and write tensors; the server stays torch-free)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.path.join(BASE_DIR, "src") + os.pathsep + BASE_DIR)
    kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    p = subprocess.run([sys.executable, "-m", "fizgig.scripts.refmod_file"], input=json.dumps(op), text=True,
                       capture_output=True, encoding="utf-8", errors="replace", cwd=BASE_DIR, env=env, timeout=600, **kw)
    line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("@@R ")), None)
    if line is None:
        raise ValueError((p.stderr or p.stdout or "The file job failed.").strip().splitlines()[-1])
    res = json.loads(line[4:])
    if res.get("error"):
        raise ValueError(res["error"])
    return res


def bake(row, metas, retention, frame_curve, name):
    """One row's mod with its strength × the master Strength, its copies (repeated frames) and the frame curve folded
    in, as a new mod beside the source (never over a file)."""
    m = metas.get(str(row.get("mod")))
    if m is None:
        raise ValueError("Pick a mod in that row.")
    strength = _f(row.get("value"), 1.0, 0.0, 1.0)
    retention = _f(retention, 1.0, 0.0, 1.0)
    if strength <= 0:
        raise ValueError("That row's strength is 0 - nothing to bake.")
    if retention <= 0:
        raise ValueError("The master Strength (Retention) is 0 - nothing to bake.")
    name = re.sub(r'[\\/:*?"<>|]+', "_", (name or "").strip()) or f"{m['name']}_studio"
    if name.lower().endswith(".safetensors"):
        name = name[:-len(".safetensors")]
    folder = os.path.dirname(m["path"])
    stem = os.path.splitext(_free(os.path.join(folder, name + ".safetensors")))[0]
    fc = _curve(frame_curve, rm.DEFAULT_FRAME_CURVE)
    res = _file_job({"op": "bake", "meta": m, "strength": strength, "retention": retention, "curve": list(fc),
                     "copies": _i(row.get("copies"), 1), "out": stem})
    eff = strength * retention
    copies = _i(row.get("copies"), 1)
    return {"path": res["path"], "message": (
        f"Wrote {res['path']}. Loads at strength 1.0 in any loader as {m['name']} at {eff:.2f}"
        f"{' with the frame curve' if fc[0] != 'constant' or fc[1] != 'linear' or fc[2] < 1 else ''}"
        f"{f', {copies} copies' if copies > 1 else ''}. Not baked (they act at render time): Shuffle and the step "
        f"curve.")}


def save_reflora(lora, rows, metas, retention, name):
    """The LoRA + the active rows' references (each once, in row order) as one ReFLoRA beside the LoRA."""
    lora = (lora or "").strip()
    if not lora or not os.path.isfile(lora):
        raise ValueError("Open a LoRA in the LoRA or ReFLoRA card first - a ReFLoRA is that LoRA with the RefMods of "
                         "your ticked Mods rows.")
    refs, seen = [], set()
    for r, m in _meta_rows(rows, metas):
        if not m.enabled or m.value <= 0:
            continue
        key = (os.path.normcase(m.meta["path"]), m.meta.get("tensor_key", "latent"))
        if key not in seen:
            seen.add(key)
            refs.append([m.meta["path"], m.meta.get("tensor_key", "latent")])
    if not refs:
        raise ValueError("No active mod row - tick a row with a mod and a strength above 0.")
    name = re.sub(r'[\\/:*?"<>|]+', "_", (name or "").strip()) or os.path.splitext(os.path.basename(lora))[0] + "_reflora"
    if not name.lower().endswith(".safetensors"):
        name += ".safetensors"
    out = _free(os.path.join(os.path.dirname(lora), name))
    res = _file_job({"op": "reflora", "lora": lora, "refs": refs, "out": out,
                     "name": os.path.splitext(os.path.basename(out))[0]})
    n = res["references"]
    return {"path": out, "message": (
        f"Saved {os.path.basename(out)}: {os.path.basename(lora)} ({res['lora_keys']} tensors) and {n} "
        f"RefMod{'s' if n != 1 else ''} ({', '.join(res['names'])}). In ComfyUI put it in models/loras and load it with "
        f"Load H3 RefLoRA (ComfyUI-MiniMaxH3RefLoRA) at refmod_retention {_f(retention, 1.0, 0.0, 1.0):.2f}; a plain "
        f"LoRA loader still reads its LoRA. Row strengths, copies and curves are node settings - ComfyUI settings lists "
        f"them.")}
