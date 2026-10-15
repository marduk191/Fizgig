"""The launch plan for a described family: plain settings in, everything a run needs out.

One place turns the settings of a run into what starts it - the dataset TOML, the cache and training commands in
order, the prompt files the previews read, and the problems that would stop it - so every front end that starts a
run (the desktop app, a web UI, a queue, a command line) launches it the same way.

`inputs` is a plain dict: the run's settings (the same keys the desktop app saves: LORA_NAME, NETWORK_DIM,
FAMILY_EDIT, ...) plus the values the app reads from its other tabs, all lower case:

    python, repo_dir            interpreter and Fizgig checkout the commands run from
    models                      {pref_key: path} for the family's model files (Preferences)
    image_folder, caption_ext   the training folder (Start tab) and its caption extension
    batch_size, megapixels      Dataset settings
    enable_bucket, no_upscale
    cache_root                  the cache folder from Preferences (each dataset gets its own folder inside it)
    blocks_swap                 the Block Swap box as typed ("Auto (detect from GPU)", "8", ...)
    enable_cache, resuming      whether the cache stages run; a resume (or fine-tune continuation) skips them,
    recache                     unless its picture size changed (Target Megapixels): then they run, and keep what fits
    loss_watch                  {"detect", "per_image_lr", "warmup", "recaption": bool}
    captioner, caption_trigger, caption_overrides   auto-recaption: model path, trigger word, edited instructions
    samples                     {"enabled", "every", "width", "height", "steps", "cfg", "negative", "seed",
                                 "at_first": ..., "prompts": [lines as typed on the Samples tab]}
    samples_dir                 where preview prompt files go (the output folder's sample/)
    edit_caption                the edit instruction typed in the Edit card

Switches (FAMILY_EDIT, FAMILY_SLIDER, ADAPTIVE_LR, SAVE_STATE, enable_cache, samples["enabled"], loss_watch[...],
...) must be real booleans: the string "false" counts as on. Numbers may be strings or numbers. The run's
settings the commands need are LORA_NAME, LORA_OUTPUT_DIR, DATASET_CONFIG (the path the plan's TOML is written
to), NETWORK_DIM, NETWORK_ALPHA, LEARNING_RATE, MAX_TRAIN_EPOCHS, SAVE_EVERY_N_EPOCHS, SEED; the number checks
also read LORA_LR_RATIO, GRADIENT_ACCUMULATION, MAX_GRAD_NORM (and ADAPTIVE_LR_MIN / _MAX with
Adaptive LR on, LOKR_FACTOR with LoKR). A missing one is a problem, never a crash.

plan() returns problems only when there are any: then it has no stages and no files, and nothing may be written or
started. Otherwise the caller creates plan.dirs, writes plan.files (the dataset TOML first), then runs the stages in
order. What the desktop app also does at Start, which a caller has to do itself: refuse to start while a run is
active; delete a stale <LORA_OUTPUT_DIR>/.pause_requested (the trainer would stop after one epoch) and a stale
.sample_override.json (it changes the previews); warn when the output drive is nearly full, or when a resume is
already at Max Train Epochs.
"""
import os
import re
from dataclasses import dataclass, field

PRECISION_LABELS = {"auto": "Auto (fits your free VRAM)", "bf16": "bf16 (full precision)",
                    "int8": "INT8 (8-bit, fastest)", "nf4": "4-bit NF4 (smallest)",
                    "hqq": "4-bit HQQ (more accurate 4-bit)"}
PAIR_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class Stage:
    name: str
    cmd: list


@dataclass
class LaunchPlan:
    dataset_toml: str = ""                           # written to inputs["DATASET_CONFIG"] (also in files)
    stages: list = field(default_factory=list)      # Stage, in the order they run
    dirs: list = field(default_factory=list)        # folders to create before the stages run
    files: list = field(default_factory=list)       # (path, text) to write before the stages run
    console: list = field(default_factory=list)     # lines to show the user
    problems: list = field(default_factory=list)    # anything here and the run must not start


def _s(x):
    """A value from the inputs as trimmed text ("" for None) - they may arrive as numbers."""
    return "" if x is None else str(x).strip()


def _model(inputs, desc, role):
    return desc.model_path(role, (inputs.get("models") or {}).get)


def edit_on(desc, inputs):
    return bool(desc.edit_training and inputs.get("FAMILY_EDIT"))


def slider_on(desc, inputs, source=None):
    """Slider mode for a family that offers it; source "pairs" / "prompts" narrows it to one kind."""
    on = bool(desc.slider_training and inputs.get("FAMILY_SLIDER"))
    if on and source:
        offered = tuple(getattr(desc, "slider_sources", ("pairs", "prompts"))) or ("pairs", "prompts")
        have = inputs.get("FAMILY_SLIDER_SOURCE") or "pairs"
        on = (have if have in offered else offered[0]) == source      # a family with one source always uses it
    return on


def ft_on(desc, inputs):
    """A full fine-tune of the base model, for a family whose driver offers one (description.finetune)."""
    return bool(desc.finetune and inputs.get("FAMILY_FT"))


# ---- a paused run's continuation point (both apps' Resume) -------------------------------------------------------
STATE_MARKERS = ("training_state.json", "random_states_0.pkl")   # written last: a state folder without one is a
                                                                  # half-written save


def latest_state(out_dir, name):
    """The run's newest complete state folder (<name>-NNNNNN-state with its commit marker), or ""."""
    try:
        found = [d for d in os.listdir(out_dir) if re.fullmatch(re.escape(name) + r"-\d{6}-state", d)
                 and any(os.path.isfile(os.path.join(out_dir, d, m)) for m in STATE_MARKERS)]
    except OSError:
        return ""
    return os.path.join(out_dir, max(found)) if found else ""


def latest_ft_checkpoint(out_dir, name):
    """A fine-tune's newest checkpoint (<name>-NNNNNN.safetensors - a fine-tune has no state folders), or ""."""
    try:
        found = [f for f in os.listdir(out_dir) if re.fullmatch(re.escape(name) + r"-\d{6}\.safetensors", f)]
    except OSError:
        return ""
    return os.path.join(out_dir, max(found)) if found else ""


def ft_continuation(path):
    """(next_window, epochs_done, n_windows) from a fine-tune checkpoint's metadata (the trainers stamp them); the
    -NNNNNN in the file name wins for the epochs. Anything missing reads as 0."""
    import json
    import struct
    md = {}
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            md = json.loads(f.read(n)).get("__metadata__") or {}
    except (OSError, ValueError, struct.error):
        pass
    def num(k):
        try:
            return int(float(md.get(k) or 0))
        except (TypeError, ValueError):
            return 0
    epochs = num("fizgig_ft_epochs_done")
    m = re.search(r"-(\d{6})\.safetensors$", os.path.basename(path or ""))
    if m:
        epochs = int(m.group(1))
    return num("fizgig_next_start_window"), epochs, num("fizgig_ft_n_windows")


def ft_rotations_left(path, rotations, rotate_every):
    """What a family fine-tune paused at path has left of `rotations` (whole rotations of its cycle), and the
    continuation's FAMILY_FT_CONTINUE - as the desktop's Resume works it out."""
    nxt, done, nw = ft_continuation(path)
    per = max(1, _ft_int({"x": rotate_every}, "x", 1))
    rot = max(1, _ft_int({"x": rotations}, "x", 1))
    done_rot = round(done / (nw * per)) if nw else 0
    return rot - done_rot, done_rot, {"checkpoint": path, "start_window": nxt, "epochs_done": done}


def ft_reg_dir(desc, inputs):
    """A fine-tune's regularisation folder, when fine-tune is on and one is set (else "")."""
    return _s(inputs.get("FAMILY_FT_REG_DIR")) if ft_on(desc, inputs) and getattr(desc, "ft_regularisation", True)         else ""


def _ft_reg_mult(inputs):
    try:
        return max(0.0, float(str(inputs.get("FAMILY_FT_REG_MULT", "") or 0.2)))
    except ValueError:
        return None


def _ft_int(inputs, key, default):
    try:
        return max(1, int(float(str(inputs.get(key, "") or default))))
    except ValueError:
        return None


def area_blocks(desc, inputs):
    """The block ids the Model Area to Train setting trains (FAMILY_TRAIN_AREA; "Custom" = FAMILY_TRAIN_BLOCKS), or
    [] for every block."""
    name = _s(inputs.get("FAMILY_TRAIN_AREA"))
    if name == "Custom":
        return [b for b in (inputs.get("FAMILY_TRAIN_BLOCKS") or []) if b]
    area = next((a for a in desc.train_areas if a[0] == name), None)
    return list(area[1]) if area else []


def timestep_flags(inputs):
    """The Timestep Range (MIN/MAX_TIMESTEP, 0-1000 as the Training tab shows it) as the trainer's 0-1 window."""
    out = []
    for key, flag in (("MIN_TIMESTEP", "--min_timestep"), ("MAX_TIMESTEP", "--max_timestep")):
        try:
            v = float(_s(inputs.get(key)))
        except ValueError:
            continue
        out += [flag, f"{min(max(v, 0.0), 1000.0) / 1000.0:g}"]
    return out


def caches(desc, inputs):
    """Whether the cache stages run: not on a resume (the cache is already built) unless it trains at a different
    size from now on (recache: the latents that still fit are kept), not for a prompt slider (no photos), not with
    Enable Cache off."""
    resuming = inputs.get("resuming") or bool(_s(inputs.get("RESUME_TRAINING")))
    return bool(inputs.get("enable_cache", True) and (not resuming or inputs.get("recache"))
                and not slider_on(desc, inputs, "prompts"))


# ---------------------------------------------------------------------------------------------------- pairs
def pairs(after_dir, before_dir, exts=PAIR_EXTS):
    """(photos without a partner, photos with more than one, first partner's path) for an edit's or a slider's
    two folders, matched the way the dataset loader matches them: a partner of photo.png is photo.<ext> or
    photo_<anything>.<ext>, ignoring case. exts: what counts as an item (a clip family's sliders add .mp4)."""
    befores = sorted(f for f in os.listdir(before_dir) if os.path.splitext(f)[1].lower() in exts)
    missing, multiple, first = [], [], None
    for f in sorted(os.listdir(after_dir)):
        b, e = os.path.splitext(f)
        if e.lower() not in exts:
            continue
        m = [x for x in befores if x.casefold().startswith(b.casefold() + ".")    # IMG_1 pairs with img_1
             or x.casefold().startswith(b.casefold() + "_")]
        if not m:
            missing.append(f)
        elif len(m) > 1:
            multiple.append(f)
        first = first or (os.path.join(before_dir, m[0]) if m else None)
    return missing, multiple, first


def pair_problems(after_dir, before_dir, caption_ext=".txt", exts=PAIR_EXTS):
    """(photos in after_dir without a caption, pairs whose shapes differ - photos only; a clip pair's size and
    length are checked when it is cached)."""
    from PIL import Image
    befores = sorted(f for f in os.listdir(before_dir) if os.path.splitext(f)[1].lower() in exts)
    uncaptioned, shapes = [], []
    for f in sorted(os.listdir(after_dir)):
        b, e = os.path.splitext(f)
        if e.lower() not in exts:
            continue
        if not os.path.exists(os.path.join(after_dir, b + caption_ext)):
            uncaptioned.append(f)
        m = [x for x in befores if x.casefold().startswith(b.casefold() + ".")
             or x.casefold().startswith(b.casefold() + "_")]
        if len(m) == 1 and e.lower() in PAIR_EXTS:
            try:
                with Image.open(os.path.join(after_dir, f)) as ia, Image.open(os.path.join(before_dir, m[0])) as ib:
                    ra, rb = ia.width / ia.height, ib.width / ib.height
                if abs(ra - rb) / ra > 0.02:
                    shapes.append(f)
            except OSError:
                pass
    return uncaptioned, shapes


def edit_instruction(inputs):
    """The edit instruction for previews: the typed instruction, else the first edited photo's caption file."""
    text = _s(inputs.get("edit_caption"))
    if text:
        return text
    folder = _s(inputs.get("image_folder"))
    ext = inputs.get("caption_ext") or ".txt"
    try:
        for f in sorted(os.listdir(folder)):
            b, e = os.path.splitext(f)
            cap = os.path.join(folder, b + ext)
            if e.lower() in PAIR_EXTS and os.path.exists(cap):
                with open(cap, encoding="utf-8") as fh:
                    line = fh.read().strip().splitlines()
                if line and line[0].strip():
                    return line[0].strip()
    except OSError:
        pass
    return ""


# the Training tab's window-size choice -> the planner's cap on parts per window (0 = as many as fit)
FT_WINDOW_SIZES = (("Auto - as few windows as fit", 0), ("At most 3 parts per window", 3),
                   ("At most 2 parts per window", 2), ("One part per window (most headroom)", 1))


def ft_max_parts(inputs):
    return dict(FT_WINDOW_SIZES).get(_s(inputs.get("FAMILY_FT_MAX_PARTS")), 0)


# ---------------------------------------------------------------------------------------------------- problems
def problems(desc, inputs):
    """What the family's own checks refuse: an edit's or a slider's pairs, a prompt slider's prompts, and every
    required model file (plus the training adapter while it is on) set and on disk."""
    errors = []
    folder = _s(inputs.get("image_folder"))
    if ft_on(desc, inputs):
        for key, label in (("FAMILY_FT_ROTATIONS", "Rotations"), ("FAMILY_FT_SAVE_EVERY", "Save every"),
                           ("FAMILY_FT_ROTATE_EVERY", "Epochs per window")):
            if _ft_int(inputs, key, 1) is None:
                errors.append(f"Fine-tune: {label} must be a whole number of 1 or more")
        reg = ft_reg_dir(desc, inputs)
        if reg and not os.path.isdir(reg):
            errors.append(f"Fine-tune: the regularisation folder {reg} does not exist - clear the box to train "
                          f"without one")
        elif reg and _ft_reg_mult(inputs) is None:
            errors.append("Fine-tune: the regularisation LR multiplier must be a number")
    ext = inputs.get("caption_ext") or ".txt"
    if edit_on(desc, inputs) and slider_on(desc, inputs):
        errors.append("Edit LoRA and Slider are both on - a run is one kind of LoRA: pick Edit or Slider")
    if edit_on(desc, inputs):
        before = _s(inputs.get("FAMILY_EDIT_DIR"))
        ref = _s(inputs.get("FAMILY_EDIT_REF"))
        if not before or not os.path.isdir(before):
            errors.append("Edit LoRA is on: set the Originals folder (Training tab, Edit pairs)")
        elif os.path.isdir(folder):
            missing, multiple, _ = pairs(folder, before)
            if missing:
                errors.append(f"Edit LoRA: {len(missing)} edited photo(s) have no original with the same file "
                              f"name in {before} (e.g. {', '.join(missing[:3])})")
            if multiple:
                errors.append(f"Edit LoRA: {len(multiple)} edited photo(s) match more than one original (e.g. "
                              f"{', '.join(multiple[:3])}) - keep one original per edited photo, with the "
                              f"same file name")
            uncaptioned, shapes = pair_problems(folder, before, ext)
            if uncaptioned:
                errors.append(f"Edit LoRA: {len(uncaptioned)} edited photo(s) have no caption (e.g. "
                              f"{', '.join(uncaptioned[:3])}) - type the edit and press Write captions")
            if shapes:
                errors.append(f"Edit LoRA: {len(shapes)} edited photo(s) have a different crop or shape from their "
                              f"original (e.g. {', '.join(shapes[:3])}) - export both at the same crop")
        if ref and not os.path.isfile(ref):
            errors.append(f"Edit LoRA test photo for previews does not exist: {ref}")
    if slider_on(desc, inputs, "pairs"):
        other = _s(inputs.get("FAMILY_SLIDER_DIR"))
        _ex = PAIR_EXTS | ({".mp4"} if "clip" in desc.media else set())    # H3: clip pairs too
        if not other or not os.path.isdir(other):
            errors.append("Slider: set the -1 end folder (Training tab, Slider)")
        elif os.path.isdir(folder):
            missing, multiple, _ = pairs(folder, other, _ex)
            if missing:
                errors.append(f"Slider: {len(missing)} +1 photo(s) have no -1 photo with the same file name in "
                              f"{other} (e.g. {', '.join(missing[:3])})")
            if multiple:
                errors.append(f"Slider: {len(multiple)} +1 photo(s) match more than one -1 photo (e.g. "
                              f"{', '.join(multiple[:3])}) - keep one per photo, with the same file name")
            uncaptioned, shapes = pair_problems(folder, other, ext, _ex)
            if uncaptioned:
                errors.append(f"Slider: {len(uncaptioned)} photo(s) have no caption (e.g. "
                              f"{', '.join(uncaptioned[:3])}) - type what both ends share and press Write "
                              f"captions")
            if shapes:
                errors.append(f"Slider: {len(shapes)} photo(s) have a different crop or shape at the two ends "
                              f"(e.g. {', '.join(shapes[:3])}) - use the same framing for both")
    if slider_on(desc, inputs, "prompts"):
        for k, what in (("FAMILY_SLIDER_BASE", "what the picture is"), ("FAMILY_SLIDER_POS", "what the +1 end "
                        "adds"), ("FAMILY_SLIDER_NEG", "what the -1 end adds")):
            if not _s(inputs.get(k)):
                errors.append(f"Slider from prompts: fill in {what}")
        try:
            if float(inputs.get("FAMILY_SLIDER_GUIDANCE")) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("Slider push strength must be a number above 0")
    need = [f for f in desc.model_files if f.required]
    if desc.training_adapter and inputs.get("FAMILY_TRAINING_ADAPTER", True):
        need += [f for f in desc.model_files if f.pref_key == desc.training_adapter]
    for f in need:
        path = (inputs.get("models") or {}).get(f.pref_key, "")
        if not path:
            errors.append(f"{f.label} path is empty (set it on the Preferences tab)")
        elif not os.path.exists(path):
            errors.append(f"{f.label} file does not exist: {path}")
    return errors


# ---------------------------------------------------------------------------------------------------- commands
def dataset_media(folder):
    """The media kinds in a training folder: {"photo", "clip", "voice"} (the dataset layer's extensions)."""
    from fizgig.dataset.image_dataset import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    kinds = set()
    try:
        names = os.listdir(folder) if folder and os.path.isdir(folder) else []
    except OSError:
        names = []
    img, vid, aud = ({e.lower() for e in x} for x in (IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, AUDIO_EXTENSIONS))
    for n in names:
        ext = os.path.splitext(n)[1].lower()
        kinds |= {"photo"} if ext in img else {"clip"} if ext in vid else {"voice"} if ext in aud else set()
    return kinds


def option_applies(opt, media, values=None, options=(), finetune=None):
    """Whether a FamilyOption is live for a dataset with these media kinds: a clip-only option without clips, or a
    mixed-dataset one on a dataset that is not mixed, is neither shown nor sent (#136: a retired category restored
    from a mixed run must not reach a photo-only run)."""
    if opt.show_if_media and opt.show_if_media not in media:
        return False
    if finetune is not None and opt.mode and (opt.mode == "finetune") != bool(finetune):
        return False
    if opt.mixed_only and not ("voice" in media and media & {"photo", "clip"}):
        return False
    if opt.requires and values is not None:
        key, eq, want = opt.requires.partition("=")
        have = values.get(key, "")
        if eq:
            o = next((x for x in options if x.key == key), None)
            if (o.pick(have) if o is not None else have) != (o.pick(want) if o is not None else want):
                return False
        elif str(have) not in ("1", "True", "true"):
            return False
    return True


def option_tokens(desc, inputs, missing=None):
    """(trainer args, cache --aux args) from the family's FamilyOption values (inputs["FAMILY_OPTIONS"]: key ->
    value; an unset key takes the option's default). "pref:KEY" becomes that Preferences file's path; `missing`
    (a list) collects (pref key, option) for every such file that is not set or not on disk."""
    vals = inputs.get("FAMILY_OPTIONS") or {}
    models = inputs.get("models") or {}

    def path(v):
        """'pref:KEY' -> the file; 'pref:KEY[OPT=LABEL|OPT2=1->ALT]' -> ALT's file when any condition holds."""
        if not v.startswith("pref:"):
            return v
        key, _, cond = v[5:].partition("[")
        if cond:
            tests, _, alt = cond.rstrip("]").partition("->")
            for t in tests.split("|"):
                ok, _, want = t.partition("=")
                o = next((x for x in desc.options if x.key == ok), None)
                have = vals.get(ok, o.default if o else "")
                if o is not None and o.kind == "choice":
                    have, want = o.pick(have), o.pick(want)
                if str(have) == want or (want == "1" and str(have) in ("True", "true")):
                    key = alt
                    break
        p = models.get(key, "")
        if missing is not None and (not p or not os.path.exists(p)):
            missing.append((key, cur[0]))
        return p

    train, aux = [], []
    cur = [None]                        # the option being resolved (for `missing`)
    media = dataset_media(_s(inputs.get("image_folder")))
    for opt in desc.options:
        if not option_applies(opt, media, vals, desc.options, finetune=ft_on(desc, inputs)):
            continue
        cur[0] = opt
        for tok in opt.resolve(vals.get(opt.key, opt.default)):
            if tok.startswith("aux:"):
                k, _, v = tok[4:].partition("=")
                aux += ["--aux", f"{k}={path(v)}"]
            elif tok.startswith("--"):
                k, eq, v = tok.partition("=")
                train += [k, path(v)] if eq else [k]
            else:
                k, _, v = tok.partition("=")
                train += ["--family_option", f"{k}={path(v)}"]
    return train, aux


def _dedupe_flags(cmd):
    """A flag an option sets (--dit, --training_adapter) replaces the one the generic builder put earlier: the last
    occurrence of a valued flag wins and the earlier pair goes."""
    valued = {"--dit", "--training_adapter", "--train_blocks", "--ema_decay", "--optimizer_type"}
    out, seen = [], set()
    i = len(cmd) - 1
    while i >= 0:
        tok = cmd[i]
        if i > 0 and cmd[i - 1] in valued:
            if cmd[i - 1] not in seen:
                out += [tok, cmd[i - 1]]
                seen.add(cmd[i - 1])
            i -= 2
            continue
        out.append(tok)
        i -= 1
    return out[::-1]


def cache_command(desc, inputs, stage):
    """families/cache.py: latents with the family's VAE, text with its text encoder."""
    model = _model(inputs, desc, "vae" if stage == "latents" else "text_encoder")
    cmd = [inputs["python"], os.path.join(inputs["repo_dir"], desc.cache_script), "--family", desc.key,
           "--stage", stage, "--dataset_config", inputs["DATASET_CONFIG"], "--model", model]
    cmd += option_tokens(desc, inputs)[1]
    if stage == "latents":
        cmd.append("--skip_existing")   # validated against the current bucket
    if slider_on(desc, inputs, "pairs"):
        cmd.append("--slider")          # the control folder is the -1 end: its latents, captions encoded plainly
    return cmd


def _state_flags(st):
    """Save-state flags. Keep-N is clamped to >= 1 here as well as in the trainer: a blank or zero box must never
    reach a prune that would take the state just written with it."""
    flags = []
    if st.get("SAVE_STATE", True):
        flags.append("--save_state")
    if st.get("SAVE_STATE_ON_TRAIN_END", True):
        flags.append("--save_state_on_train_end")
    if flags:
        try:
            keep_n = max(1, int(str(st.get("KEEP_LAST_N_STATES", 2)).strip()))
        except (TypeError, ValueError):
            keep_n = 2
        flags += ["--keep_last_n_states", str(keep_n)]
    return flags


def train_command(desc, inputs, plan):
    """families/train.py. Model paths from the description's roles, training settings from the run's settings,
    previews from the sample settings. Preview prompt files and console lines go into `plan`."""
    st = inputs
    cmd = [st["python"], os.path.join(st["repo_dir"], desc.train_script), "--family", desc.key,
           "--dit", _model(st, desc, "dit"),
           "--dataset_config", st["DATASET_CONFIG"],
           "--output_dir", st["LORA_OUTPUT_DIR"], "--output_name", st["LORA_NAME"],
           "--network_dim", str(st["NETWORK_DIM"]), "--network_alpha", str(st["NETWORK_ALPHA"]),
           "--learning_rate", str(st["LEARNING_RATE"]), "--max_train_epochs", str(st["MAX_TRAIN_EPOCHS"]),
           "--save_every_n_epochs", str(st["SAVE_EVERY_N_EPOCHS"]), "--seed", str(st["SEED"])]
    cmd += _state_flags(st)
    if _s(st.get("RESUME_TRAINING")) and not ft_on(desc, st):
        cmd += ["--resume", _s(st["RESUME_TRAINING"])]
    if desc.training_adapter and st.get("FAMILY_TRAINING_ADAPTER", True):
        cmd += ["--training_adapter", (st.get("models") or {}).get(desc.training_adapter, "")]
    ctx = _s(st.get("CONTEXT_LORA_PATH"))
    if ctx and not ft_on(desc, st):         # a fine-tune trains the base itself: there is no LoRA to stack on
        cmd += ["--context_lora_path", ctx,
                "--context_lora_strength", _s(st.get("CONTEXT_LORA_STRENGTH") or "1.0") or "1.0"]
    if st.get("ADAPTIVE_LR") and desc.adaptive_lr:
        cmd += ["--adaptive_lr",
                "--adaptive_lr_min", str(st.get("ADAPTIVE_LR_MIN", "1e-4")).split(" ")[0],
                "--adaptive_lr_max", str(st.get("ADAPTIVE_LR_MAX", "2e-4")).split(" ")[0]]
    else:
        sched = _s(st.get("LR_SCHEDULER") or "constant") or "constant"
        if sched != "constant":
            cmd += ["--lr_scheduler", sched]
        try:
            if int(float(str(st.get("LR_WARMUP_STEPS", "") or 0))) > 0:
                cmd += ["--lr_warmup_steps", str(int(float(st["LR_WARMUP_STEPS"])))]
        except ValueError:
            pass
    if desc.compiles:
        cb = str(st.get("COMPILE_BLOCKS", "auto") or "auto").lower()
        if cb in ("auto", "on", "off", "outside"):
            cmd += ["--compile_blocks", cb]
    try:
        if int(float(str(st.get("GRADIENT_ACCUMULATION", "") or 1))) > 1:
            cmd += ["--gradient_accumulation_steps", str(int(float(st["GRADIENT_ACCUMULATION"])))]
    except ValueError:
        pass
    try:
        if abs(float(str(st.get("MAX_GRAD_NORM", "") or 1.0)) - 1.0) > 1e-9:
            cmd += ["--max_grad_norm", str(float(st["MAX_GRAD_NORM"]))]
    except ValueError:
        pass
    if len(desc.precisions) > 1:
        lab = str(st.get("FAMILY_PRECISION", "") or "")
        labels = {**PRECISION_LABELS, **desc.precision_labels}
        prec = next((k for k, v in labels.items() if v == lab), "auto")
        cmd += ["--precision", prec if prec == "auto" or prec in desc.precisions else "auto"]
    if ft_on(desc, st):
        # the length and cadence in whole rotations; a continuation starts from the paused run's checkpoint
        cmd += ["--finetune", "--ft_rotations", str(_ft_int(st, "FAMILY_FT_ROTATIONS", 10)),
                "--ft_save_every_rotations", str(_ft_int(st, "FAMILY_FT_SAVE_EVERY", 1)),
                "--ft_rotate_every", str(_ft_int(st, "FAMILY_FT_ROTATE_EVERY", 1))]
        if ft_max_parts(st):
            cmd += ["--ft_max_parts", str(ft_max_parts(st))]
        if st.get("FAMILY_FT_FUSED", True):
            cmd.append("--ft_fused_backward")
        if ft_reg_dir(desc, st):
            cmd += ["--reg_lr_multiplier", f"{_ft_reg_mult(st):g}"]
        cont = st.get("FAMILY_FT_CONTINUE") or {}
        if cont.get("checkpoint"):
            cmd += ["--ft_start_window", str(int(cont.get("start_window", 0))),
                    "--ft_epochs_done", str(int(cont.get("epochs_done", 0)))]
    # per-image loss watch (families/loss_watch.py), batch size 1 only
    try:
        bs1 = int(str(st.get("batch_size", 1)).strip() or 1) <= 1
    except ValueError:
        bs1 = True
    lw = (st.get("loss_watch") or {}) if desc.loss_watch else {}
    watch = [("detect", "--log_per_image_loss"), ("per_image_lr", "--per_image_lr"), ("warmup", "--warmup_look_outliers")]
    can_recaption = bool(st.get("captioner"))   # every family captions with the shared captioner
    if can_recaption:
        watch.append(("recaption", "--auto_recaption"))
    if bs1:
        cmd += [flag for key, flag in watch if lw.get(key)]
        if can_recaption and lw.get("recaption"):
            cmd += ["--captioner", st.get("captioner", "")]
            trig = _s(st.get("caption_trigger"))
            if trig and trig.lower() != "trigger_word":
                cmd += ["--trigger_word", trig]      # leads each AI caption, as the Captions tab writes it
            ovr = st.get("caption_overrides") or {}
            for key, flag in (("training", "--recaption_instruction"), ("exhaustive", "--recaption_instruction_detailed")):
                instr = str(ovr.get(key, "") or "").strip()
                if instr:
                    cmd += [flag, instr]
    elif any(lw.get(key) for key, _ in watch):
        plan.console.append("[loss-watch] per-image features skipped - they need Batch Size 1.\n")
    if any(lw.get(key) for key, _ in watch) and "--text_encoder" not in cmd:
        cmd += ["--text_encoder", _model(st, desc, "text_encoder")]   # caption repair re-encodes
    if ("lokr" in desc.network_types and str(st.get("NETWORK_TYPE", "")).startswith("LoKR")
            and not slider_on(desc, st)):        # a slider is always a plain LoRA
        cmd += ["--network_type", "lokr", "--lokr_factor", str(st.get("LOKR_FACTOR", 8))]
    raw_swap = str(st.get("blocks_swap") or "").strip()
    if raw_swap.lower().startswith("auto"):
        cmd += ["--blocks_to_swap", "-1"]            # the trainer sizes it for the precision that runs
    else:
        m = re.match(r"\d+", raw_swap)
        if m and int(m.group()) > 0:
            cmd += ["--blocks_to_swap", m.group()]
    if desc.ema_default:
        ema = str(st.get("FAMILY_EMA", "") or desc.ema_default).split(" ")[0]
        if ema != "Off":
            cmd += ["--ema_decay", ema]
    if str(st.get("OPTIMIZER_TYPE", "") or "").strip():
        cmd += ["--optimizer_type", str(st["OPTIMIZER_TYPE"]).strip()]
    if str(st.get("OPTIMIZER_ARGS", "") or "").strip():
        cmd += ["--optimizer_args", str(st["OPTIMIZER_ARGS"]).strip()]
    for key, flag in (("METADATA_TITLE", "--metadata_title"), ("METADATA_AUTHOR", "--metadata_author"),
                      ("METADATA_DESCRIPTION", "--metadata_description"),
                      ("METADATA_LICENSE", "--metadata_license"), ("METADATA_TAGS", "--metadata_tags"),
                      ("METADATA_THUMBNAIL", "--metadata_thumbnail")):
        val = str(st.get(key, "") or "").strip()
        if val:
            cmd += [flag, val]
    trig = _s(st.get("METADATA_TRIGGER_PHRASE")) or _s(st.get("caption_trigger"))
    if trig and trig.lower() != "trigger_word":
        cmd += ["--metadata_trigger_phrase", trig]
    cmd += _preview_flags(desc, st, plan, cmd)
    if (desc.identity_blocks and st.get("FAMILY_FAST_ID")
            and not (slider_on(desc, st) or edit_on(desc, st) or ft_on(desc, st))):
        cmd += ["--train_blocks", ",".join(desc.identity_blocks)]
    elif desc.train_areas and not (slider_on(desc, st) or ft_on(desc, st)):
        blocks = area_blocks(desc, st)
        if blocks:
            cmd += ["--train_blocks", ",".join(blocks)]
    if desc.train_areas and not ft_on(desc, st):
        cmd += timestep_flags(st)
    if slider_on(desc, st):
        if desc.slider_ultra_blocks and st.get("FAMILY_SLIDER_ULTRA"):
            cmd += ["--train_blocks", ",".join(desc.slider_ultra_blocks)]
        if slider_on(desc, st, "prompts"):
            base = _s(st.get("FAMILY_SLIDER_BASE"))
            cmd += ["--slider_prompts", base, f"{base} {_s(st.get('FAMILY_SLIDER_POS'))}",
                    f"{base} {_s(st.get('FAMILY_SLIDER_NEG'))}",       # the user's own comma, if any
                    "--slider_guidance", _s(st.get("FAMILY_SLIDER_GUIDANCE")) or f"{desc.slider_guidance:g}"]
            try:        # the practice pictures (and so the training) at Target Megapixels, a square on the 16 px grid
                side = int((float(st.get("megapixels")) * 1_000_000) ** 0.5) // 16 * 16
                if side >= 256:
                    cmd += ["--slider_bank_res", str(side)]
            except (TypeError, ValueError):
                pass
            for flag, role in (("--text_encoder", "text_encoder"), ("--vae", "vae")):
                if flag not in cmd:          # the practice images and the three prompts need both
                    cmd += [flag, _model(st, desc, role)]
        else:
            cmd.append("--slider_pairs")
    if desc.options:
        cmd = _dedupe_flags(cmd + option_tokens(desc, st)[0])
    cont = (st.get("FAMILY_FT_CONTINUE") or {}) if ft_on(desc, st) else {}
    if cont.get("checkpoint"):       # after the options: a model option's own --dit (H3's reference base) is what the
        cmd[cmd.index("--dit") + 1] = cont["checkpoint"]     # paused run started from; its checkpoint goes on
    return cmd


def _preview_flags(desc, st, plan, cmd):
    """The preview flags, and the prompt file they read (added to plan.files)."""
    sm = st.get("samples") or {}
    out = []
    if not sm.get("enabled"):
        return out
    sp = desc.preview_speed()
    speed_path = (st.get("models") or {}).get(sp.pref_key, "") if sp and sp.pref_key else ""
    turbo_steps = None
    if speed_path and os.path.exists(speed_path):
        dflt = desc.preview_speed_defaults()[1]
        try:
            if desc.samples_turbo_pace:        # H3's row: N steps at M% (the Steps box is the plain-model count)
                ts = float(str(st.get("FAMILY_TURBO_PACE", "") or round(100 * dflt))) / 100.0
                turbo_steps = str(st.get("FAMILY_TURBO_STEPS", "") or "").strip() or str(desc.preview_speed_defaults()[0])
            else:
                ts = float(str(st.get("FAMILY_TURBO_STRENGTH", "") or dflt))
        except ValueError:
            ts = dflt
        if ts > 0:                              # strength 0 = previews without the turbo: don't load it
            out += ["--speed_lora", speed_path]
            if abs(ts - dflt) > 1e-9:
                out += ["--speed_lora_strength", f"{max(0.0, min(2.0, ts)):g}"]
        else:
            turbo_steps = None
    samples_dir = st.get("samples_dir") or os.path.join(_s(st.get("LORA_OUTPUT_DIR")), "sample")
    prompts = None
    if edit_on(desc, st):              # edit previews: the edit instruction, not the Samples-tab prompts
        instr = edit_instruction(st)
        if instr:
            prompts = os.path.join(samples_dir, f"{desc.key}_edit_prompt.txt")
            plan.dirs.append(samples_dir)
            plan.files.append((prompts, instr + "\n"))
    elif slider_on(desc, st):          # slider previews: the dial on its own picture, not the Samples tab
        own = (st.get("FAMILY_SLIDER_BASE") if slider_on(desc, st, "prompts")
               else st.get("FAMILY_SLIDER_CAPTION"))
        prompts = os.path.join(samples_dir, f"{desc.key}_slider_prompt.txt")
        plan.dirs.append(samples_dir)
        plan.files.append((prompts, (str(own or "").strip() or "a photo") + "\n"))   # the trainer reads the captions
    else:
        lines = []
        for raw in sm.get("prompts") or []:
            ln = raw.strip()
            if not ln or ln.startswith("#"):
                continue
            ln = ln.split(" --")[0].strip()      # drop Klein-style inline flags
            if ln:
                lines.append(ln)
        plan.dirs.append(samples_dir)            # the sample folder exists either way
        if lines or _sample_image(desc, sm):     # a reference alone previews too ('generate from this picture')
            prompts = os.path.join(samples_dir, f"{desc.key}_prompts.txt")
            plan.files.append((prompts, "\n".join(lines) + "\n"))
    every = str(sm.get("every") or "").strip()
    if every.isdigit() and int(every) > 0 and slider_on(desc, st):
        # a slider's preview is the dial itself: one at least with every checkpoint (a family's long-run preview
        # cadence - ACE-Step's 40 - would leave a 20-epoch slider with only its epoch-0 preview)
        try:
            every = str(min(int(every), max(1, int(float(st.get("SAVE_EVERY_N_EPOCHS"))))))
        except (TypeError, ValueError):
            pass
    if prompts and every.isdigit() and int(every) > 0:
        out += ["--sample_prompts", prompts, "--sample_every_n_epochs", every,
                "--vae", _model(st, desc, "vae"),
                "--text_encoder", _model(st, desc, "text_encoder"),
                "--sample_width", str(sm.get("width") or "").strip() or str(desc.preview_width),
                "--sample_height", str(sm.get("height") or "").strip() or str(desc.preview_height)]
        if turbo_steps and turbo_steps.isdigit():
            out += ["--sample_steps", turbo_steps]
        elif str(sm.get("steps") or "").strip():
            out += ["--sample_steps", str(sm["steps"]).strip()]
        try:
            cfg = float(str(sm.get("cfg") or "").strip() or desc.preview_cfg)
        except ValueError:
            cfg = desc.preview_cfg
        out += ["--sample_cfg_scale", str(cfg)]
        neg = str(sm.get("negative") or "").strip()
        if neg and cfg > 1.0:
            out += ["--sample_negative", neg]
        try:
            out += ["--sample_seed", str(int(str(sm.get("seed")).strip()))]
        except ValueError:
            pass
        if sm.get("at_first"):
            out.append("--sample_at_first")
        strengths = [float(x) for x in (sm.get("strengths") or ()) if 0 < float(x) != 1.0]
        if strengths and not ft_on(desc, st) and not slider_on(desc, st) and not edit_on(desc, st):
            # each preview again at these LoRA strengths (a standard LoRA: a fine-tune has no strength to turn down)
            out += ["--sample_strengths", ",".join(f"{x:g}" for x in strengths)]
        if not edit_on(desc, st) and not slider_on(desc, st) and _sample_image(desc, sm):
            # the Samples tab's picture: seen through the vision path, or edited by every preview prompt
            out += ["--sample_image" if desc.reference_kind == "vision" else "--sample_reference",
                    _sample_image(desc, sm)]
        if edit_on(desc, st):
            ref = _s(st.get("FAMILY_EDIT_REF"))
            if not ref:
                try:
                    ref = pairs(_s(st.get("image_folder")), _s(st.get("FAMILY_EDIT_DIR")))[2] or ""
                except OSError:
                    ref = ""
            if ref:
                out += ["--sample_reference", ref]
        ck = desc.preview_checkpoint()
        if (desc.train_preview_checkpoint and ck and sm.get("checkpoint") and not slider_on(desc, st)
                and not ft_on(desc, st)):      # a fine-tune previews the model it trains, not the Distilled
            path = _s((st.get("models") or {}).get(ck[0].pref_key))
            if path and os.path.exists(path):        # Klein's "Use Distilled model for samples"
                out += ["--preview_checkpoint", path,
                        "--preview_checkpoint_cache", _s(sm.get("checkpoint_cache")) or "auto"]
                if sm.get("int8"):
                    out.append("--preview_int8")
    return out


def _sample_image(desc, sm):
    """The Samples tab's reference image, for families whose previews take one (reference_kind), when it exists."""
    ref = _s(sm.get("reference")) if desc.reference_kind else ""
    return ref if ref and os.path.isfile(ref) else ""


# ---------------------------------------------------------------------------------------------------- dataset
def cache_dir_for(cache_root, image_dir):
    """`<cache_root>/<folder name>-<hash of full path>`: one cache folder per image folder. The trainer builds its
    item list by globbing the cache folder, so two datasets sharing one would train on each other's leftovers; the
    hash keeps it stable per folder and unique across same-named folders (case and trailing slash normalised)."""
    import hashlib
    norm = image_dir.lower().replace("\\", "/").rstrip("/")
    h = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]
    nm = "".join(c if (c.isalnum() or c in "-_") else "_"
                 for c in os.path.basename(image_dir.rstrip("/\\"))) or "dataset"
    return os.path.join(cache_root, f"{nm}-{h}")


def dataset_toml(desc, inputs):
    """The dataset TOML for the run: the training folder at Target Megapixels (a square of that area on the
    16-pixel grid), an edit's originals or a slider's -1 end as its control folder, and its own cache folder.
    Raises ValueError on a Target Megapixels that is not a number. Batch size is always 1 (one image a step)."""
    mp = float(inputs.get("megapixels"))
    if mp <= 0:
        raise ValueError(f"Target Megapixels {mp}")
    side = int((mp * 1_000_000) ** 0.5) // 16 * 16
    batch = 1           # every family trains one image at a time (families/train.py); Gradient Accumulation for more
    folder = _s(inputs.get("image_folder"))
    extra = [_s(f) for f in (inputs.get("extra_folders") or []) if _s(f)] if desc.multi_concept else []
    lines = ["[general]", f"resolution = [{side}, {side}]",
             f'caption_extension = "{_s(inputs.get("caption_ext", ".txt"))}"',
             f"batch_size = {batch}", "num_repeats = 1",
             f"enable_bucket = {'true' if inputs.get('enable_bucket', True) else 'false'}",
             f"bucket_no_upscale = {'true' if inputs.get('no_upscale', True) else 'false'}"]
    try:
        cmp_ = float(inputs.get("clip_megapixels") or 0)
    except (TypeError, ValueError):
        cmp_ = 0.0
    if cmp_ > 0 and "clip" in desc.media and any("clip" in dataset_media(f) for f in [folder] + extra):
        lines.append(f"clip_megapixels = {cmp_:g}")      # clips cache and train at their own size
    lines += ["", "[[datasets]]", f'image_directory = "{folder.replace(chr(92), "/")}"']
    if edit_on(desc, inputs):
        lines.append(f'control_directory = "{_s(inputs.get("FAMILY_EDIT_DIR")).replace(chr(92), "/")}"')
    elif slider_on(desc, inputs, "pairs"):
        lines.append(f'control_directory = "{_s(inputs.get("FAMILY_SLIDER_DIR")).replace(chr(92), "/")}"')
    root = _s(inputs.get("cache_root"))
    if root and folder:
        lines.append(f'cache_directory = "{cache_dir_for(root, folder).replace(chr(92), "/")}"')
    for f in extra:                      # Multi Concept: each subject its own block and cache folder
        lines += ["", "[[datasets]]", f'image_directory = "{f.replace(chr(92), "/")}"']
        if root:
            lines.append(f'cache_directory = "{cache_dir_for(root, f).replace(chr(92), "/")}"')
    reg = ft_reg_dir(desc, inputs)
    if reg:                              # a fine-tune's regularisation set: its own block (and cache folder)
        lines += ["", "[[datasets]]", f'image_directory = "{reg.replace(chr(92), "/")}"']
        if root:
            lines.append(f'cache_directory = "{cache_dir_for(root, reg).replace(chr(92), "/")}"')
        lines.append("is_reg = true")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- the plan
def multi_concept_problems(desc, inputs):
    """Multi Concept ticked (inputs["multiconcept"], the web form's): each extra subject folder becomes its own
    [[datasets]] block, so it has to be there, be its own folder and carry its own captions (the desktop's checks)."""
    if not (desc.multi_concept and inputs.get("multiconcept")):
        return []
    norm = lambda f: f.lower().replace("\\", "/").rstrip("/")
    seen, out = {norm(_s(inputs.get("image_folder")))}, []
    extra = [_s(f) for f in (inputs.get("extra_folders") or []) if _s(f)]
    if not extra:
        out.append("Multi Concept is on but no second subject folder is set — pick one, or turn the mode off.")
    ext = _s(inputs.get("caption_ext")) or ".txt"
    for f in extra:
        if norm(f) in seen:
            out.append(f"Multi Concept: {f} is the same folder as another subject — each needs its own folder.")
            continue
        seen.add(norm(f))
        if not os.path.isdir(f):
            out.append(f"Multi Concept: folder does not exist: {f}")
        elif not any(fn.lower().endswith(ext.lower()) for fn in os.listdir(f)):
            out.append(f"Multi Concept: no {ext} captions in {f}. Caption both folders before training — each "
                       f"subject needs its own trigger word in every caption, or they will blend.")
    return out


def automagic(inputs):
    return _s(inputs.get("OPTIMIZER_TYPE")).lower() == "automagic3"


def start_problems(desc, inputs):
    """Everything Start refuses, in the order the desktop app lists it: the shared checks (fizgig.families.checks)
    with the family's own (problems) in their place. The dataset TOML is written by the plan itself, so only its
    path is required here."""
    from fizgig.families import checks
    v = inputs
    out = checks.learning_rate(v)
    out += checks.network(v, lokr="lokr" in desc.network_types and str(v.get("NETWORK_TYPE", "")).startswith("LoKR"))
    out += checks.numbers(v)
    out += checks.dataset_config(v.get("DATASET_CONFIG"), must_exist=False)
    out += problems(desc, inputs)
    out += checks.learning_rate_range(v)
    out += multi_concept_problems(desc, v)
    if automagic(v) and not v.get("ADAPTIVE_LR") and not ft_on(desc, v):
        try:
            lr = float(v.get("LEARNING_RATE"))
        except (TypeError, ValueError):
            lr = 0.0
        if lr > 1e-3:
            out.append(f"Automagic v3 starts at the Learning Rate you give and finds its own rate: {lr:g} is an AdamW "
                       f"number. Use 1e-6 (its default) or at most 0.001.")
    out += checks.context_lora(v)
    raw_swap = str(v.get("blocks_swap") or "").strip()
    m = re.match(r"\d+", raw_swap)
    swap = 0 if raw_swap.lower().startswith("auto") or not m else int(m.group())   # Auto always fits
    out += checks.run(v, blocks_swap=swap, swap_max=max(0, desc.n_blocks - 2), arch_label=desc.gui_label,
                      name_error=checks.tidy_name(v.get("LORA_NAME"))[1])
    # a Preferences file an option asks for (H3's ref2va base, an Ostris adapter): set and on disk
    miss = []
    option_tokens(desc, inputs, missing=miss)
    for key, opt in dict.fromkeys(miss):
        row = next((f.label for f in desc.model_files if f.pref_key == key), key)
        p = (inputs.get("models") or {}).get(key, "")
        why = f"'{opt.label}' on the Training tab needs it" if opt is not None and opt.label else "a setting needs it"
        out.append(f"{row} is {'empty' if not p else 'not where Preferences says (' + p + ')'} — {why}. Set it on "
                   f"the Preferences tab (its Download link, or the download button there).")
    # a typed block range (an option with counts_blocks) that does not parse, caught before any model loads
    vals = inputs.get("FAMILY_OPTIONS") or {}
    media = dataset_media(_s(v.get("image_folder")))
    for opt in desc.options:
        if not opt.counts_blocks or not option_applies(opt, media, vals, desc.options, finetune=ft_on(desc, inputs)):
            continue
        spec = str(vals.get(opt.key, opt.default) or "all").split("·")[0].strip() or "all"
        if spec.lower() != "all":
            from fizgig.utils.block_spec import parse_block_spec
            try:
                parse_block_spec(spec, opt.counts_blocks)
            except ValueError as e:
                out.append(f"{opt.label}: {e}")
    from fizgig.dataset.image_dataset import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    exts = {e.lower() for e in IMAGE_EXTENSIONS}
    if "clip" in desc.media:
        exts |= {e.lower() for e in VIDEO_EXTENSIONS}       # the clip formats the dataset layer reads
    if "voice" in desc.media:
        exts |= {e.lower() for e in AUDIO_EXTENSIONS}
    out += checks.training_folder(_s(v.get("image_folder")), _s(v.get("caption_ext", ".txt")),
                                  check_captions=not slider_on(desc, inputs, "prompts"), media_exts=exts)
    out += clip_problems(desc, _s(v.get("image_folder")))
    return out


_CLIP_CHECKED = {}          # (path, mtime, size) -> problem: a Start re-check does not probe an unchanged clip again


def clip_problems(desc, folder):
    """Every clip in the training folder that is off the family's ClipSpec, with the reason - before anything
    caches, so a wrong frame rate or frame count is refused at Start, not found an hour into caching. Plus the
    driver's own media checks (media_problem) for anything the spec cannot say."""
    if not ({"clip", "voice"} & set(desc.media)) or not folder or not os.path.isdir(folder):
        return []
    from fizgig.families import clips
    from fizgig.dataset.image_dataset import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
    vid = {e.lower() for e in VIDEO_EXTENSIONS} if "clip" in desc.media else set()
    aud = {e.lower() for e in AUDIO_EXTENSIONS} if "voice" in desc.media else set()
    driver, out = None, []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        ext = os.path.splitext(name)[1].lower()
        if ext not in vid | aud or not os.path.isfile(path):
            continue
        st = os.stat(path)
        key = (path, st.st_mtime, st.st_size)
        if key not in _CLIP_CHECKED:
            why = clips.problem(path, desc.clip_spec, desc.display_name) if ext in vid and desc.clip_spec else ""
            if not why and desc.driver:
                if driver is None:
                    driver = desc.load_driver()
                why = driver.media_problem(path)
            _CLIP_CHECKED[key] = why
        if _CLIP_CHECKED[key]:
            out.append(_CLIP_CHECKED[key])
    return out


def plan(desc, inputs):
    """The whole launch: problems first (a plan with problems must not start), then the stages in order."""
    from fizgig.families.checks import tidy_name
    inputs = dict(inputs)
    folder = _s(inputs.get("image_folder"))
    if folder and not os.path.isabs(folder):
        inputs["image_folder"] = os.path.abspath(folder)     # the subprocesses run from the Fizgig folder
    inputs["LORA_NAME"] = tidy_name(_s(inputs.get("LORA_NAME")))[0]
    if automagic(inputs):            # it owns the rate: Adaptive LR, per-image LR and the look warm-up are off
        inputs["ADAPTIVE_LR"] = False
        inputs["loss_watch"] = {**(inputs.get("loss_watch") or {}), "per_image_lr": False, "warmup": False}
    p = LaunchPlan(problems=start_problems(desc, inputs))
    if not p.problems:
        try:
            p.dataset_toml = dataset_toml(desc, inputs)
        except (TypeError, ValueError):
            p.problems.append("Target Megapixels (Dataset) must be a number above 0")
    missing = [k for k in ("python", "repo_dir", "DATASET_CONFIG", "LORA_OUTPUT_DIR", "NETWORK_DIM", "NETWORK_ALPHA",
                           "LEARNING_RATE", "MAX_TRAIN_EPOCHS", "SAVE_EVERY_N_EPOCHS", "SEED") if not _s(inputs.get(k))]
    if missing and not p.problems:
        p.problems.append(f"Missing settings: {', '.join(missing)}")
    if p.problems:
        return p                        # nothing to write or run
    p.files.append((inputs["DATASET_CONFIG"], p.dataset_toml))
    train = train_command(desc, inputs, p)
    if caches(desc, inputs):
        p.stages += [Stage("Cache Preparation", cache_command(desc, inputs, "latents")),
                     Stage("Text Encoder Caching", cache_command(desc, inputs, "text"))]
    p.stages.append(Stage("Training", train))
    return p
