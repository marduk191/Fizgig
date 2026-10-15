"""MiniMax H3 RefMod: the dataset's references made into a RefMod file (the community ComfyUI-MiniMaxH3Mod format)
whose latent is OPTIMISED against the frozen H3 base rather than just VAE-encoded. Never a LoRA run, no previews.

Shared by the desktop app's "MiniMax H3 RefMod" entry and the web UI's, so the two build the same run: the card's
options and presets, the label -> command-line value rules, and the stages (latent caching, the references' own pass
at Target MP, text caching unless it is a plain encode, then the maker, scripts/minimax_refmod.py). RefMod runs on
MiniMax H3's files (its description's model files and its Training Base option).
"""
import os

KEY = "minimax_refmod"
LABEL = "MiniMax H3 RefMod"
LORA_NAME_SUFFIX = "refmod"
BLOCKS_SWAP_MAX = 40

REFMOD_GRID_OPTIONS = ["Full reference (recommended — carries the face)", "32×32 (256 tokens)",
                       "16×16 (64 tokens, concept-level)", "8×8 (16 tokens, stackable)"]
# Steps: the optimisation against the frozen H3 base — the thing no other RefMod maker does.
# Measured 9-10 Sep 2026 (mbacc photos, ref2va, ArcFace vs the dataset): 200 steps at the
# tested rate, pull and noise window took portraits 66 -> 70 and off-dataset scenes 53 -> 59,
# ahead on every prompt. 0 writes the plain encode, the same file the node pack's extractor makes.
REFMOD_STEP_OPTIONS = ["0 (plain encode — what the node pack makes)", "200 (optimised against H3 — recommended)",
                       "500", "1000"]
REFMOD_REFS_OPTIONS = ["4", "8", "16", "all"]
# How a clip in the dataset enters the mod (16 Sep 2026): its sharpest-face still (one frame,
# the identity choice) or every latent frame as motion (the node pack's video reference).
REFMOD_CLIPS_OPTIONS = ["as their sharpest still", "as motion (all their frames)"]
# The pack's concept types (its Extract node's list). Written into the file with the description:
# together they are the mod's HINT — what RefMod Toolkit's '+ mod hints' and the pack's Loader
# emit into the prompt ("identity: a ginger woman with messy hair").
REFMOD_CONCEPT_OPTIONS = ["identity", "style", "pose_motion", "clothing", "background", "generic"]
# Token cap (16 Sep 2026): the pack's extractor rule for CLIPS AS MOTION only — near-duplicate
# frames first, then an even resample to what fits. Photos are never touched (Peter: "that's not
# what it's for"); a cap with no motion clips does nothing. Off by default.
REFMOD_TOKEN_CAP_OPTIONS = ["off (every clip frame kept)", "5,120 (the pack's default)", "8,192 (the library's files)",
                            "16,384"]
# Audio mod (16 Sep 2026): the pack's audio RefMod, a plain encode of the folder's sound through
# the H3 audio VAE, written as a second file <name>_audio.safetensors. Off by default.
REFMOD_AUDIO_OPTIONS = ["off: the visual mod only",
                        "on: one file — the visual mod plus an audio mod of the folder's sound",
                        "on: two files — the visual mod, and an audio mod of the folder's sound beside it"]
REFMOD_AUDIO_CONCEPT_OPTIONS = ["voice", "singing", "music_style", "sound_fx", "ambience"]
REFMOD_AUDIO_SECONDS_OPTIONS = ["10", "20", "30", "60"]
REFMOD_MP_OPTIONS = ["0.25", "0.37", "0.5", "0.75", "1.0"]       # the card's Target MP
REFMOD_DEFAULTS = {
    "MINIMAX_REFMOD_REFS": "16",
    "MINIMAX_REFMOD_STEPS": REFMOD_STEP_OPTIONS[1],
    "MINIMAX_REFMOD_CLIPS": REFMOD_CLIPS_OPTIONS[0],
    "MINIMAX_REFMOD_DESC": "",
    "MINIMAX_REFMOD_CONCEPT": REFMOD_CONCEPT_OPTIONS[0],
    "MINIMAX_REFMOD_TOKEN_CAP": REFMOD_TOKEN_CAP_OPTIONS[0],
    "MINIMAX_REFMOD_AUDIO": REFMOD_AUDIO_OPTIONS[0],
    "MINIMAX_REFMOD_AUDIO_CONCEPT": REFMOD_AUDIO_CONCEPT_OPTIONS[0],
    "MINIMAX_REFMOD_AUDIO_SECONDS": "30",
}
TRAIN_MP = 0.25        # the optimiser's stills: the measured recipe, whatever Target MP says
TOKEN_CAP = 5120       # the node pack's Extract default max_tokens


def refmod_audio_on(label) -> bool:
    """'on: …' -> True; 'off: …', blank or anything else -> False."""
    return str(label or "").strip().lower().startswith("on")


def refmod_audio_value(label) -> str:
    """The CLI's --audio: 'bundle' for the one-file choice, 'folder' for two files, else 'off'."""
    t = str(label or "").strip().lower()
    if not t.startswith("on"):
        return "off"
    return "folder" if "two files" in t else "bundle"


def refmod_token_cap_value(label) -> str:
    """'5,120 (the pack's default)' -> '5120'; 'off …', blank or unparsable -> '0'."""
    head = str(label or "").strip().split(" ")[0].replace(",", "")
    try:
        return str(max(0, int(head)))
    except ValueError:
        return "0"


def refmod_clips_value(label) -> str:
    """'as motion (all their frames)' -> 'motion'; anything else -> 'still'."""
    return "motion" if str(label or "").strip().lower().startswith("as motion") else "still"


def refmod_num(raw, default: str, kind=float) -> str:
    """A free-text number field -> the string the CLI gets; the measured default when blank or
    unparsable (a typo must not silently become 0)."""
    s = str(raw if raw is not None else "").strip()
    try:
        v = kind(s)
        return str(int(v)) if kind is int else s
    except (TypeError, ValueError):
        return default


def refmod_grid_value(label) -> str:
    """'16×16 (256 tokens, recommended)' -> '16'; 'Full reference ...' -> 'full'."""
    s = str(label or "").strip()
    if not s or s.lower().startswith("full"):
        return "full"
    return s.split("×")[0].split("x")[0].strip() or "16"


def refmod_steps_value(label) -> str:
    """'200 (recommended)' -> '200'; a typed '75' -> '75'; junk -> the measured 200."""
    s = str(label or "").strip().split(" ")[0]
    return s if s.isdigit() else "200"


def minimax_train_base(raw):
    """Training Base label -> canonical base key. Anything unrecognised is the fl2va default."""
    return "ref2va" if "ref2va" in str(raw or "").lower() else "fl2va"


# The base every RefMod built-in starts from: the H3 Fast preset's values as they stood (3 Oct 2026), written out so
# RefMod no longer reads the old trainer's preset table. RefMod trains no LoRA - the LoRA keys are inert for it; the
# dataset ones (megapixels, clip stills, distillation off) are what its caching reads.
_REFMOD_PRESET_BASE = {
    "SEED": 42, "DATASET_MEGAPIXELS": "0.25",
}

REFMOD_BUILT_IN_PRESETS = {
    # The community recipe (16 Sep 2026): the node pack's own extractor path — encode mode (no
    # optimisation), full-resolution references, under its 8 192-token cap (8 references at
    # 1 MP ~ 8 000 tokens). Same photos as the Fizgig recipe, for a fair comparison.
    "✨ RefMod — community recipe (plain encode, 8 refs at 1 MP)": {
        **_REFMOD_PRESET_BASE,
        "MINIMAX_REFMOD_GRID": REFMOD_GRID_OPTIONS[0],
        "MINIMAX_REFMOD_REFS": "8",
        "MINIMAX_REFMOD_STEPS": REFMOD_STEP_OPTIONS[0],
        "MINIMAX_REFMOD_CLIPS": REFMOD_CLIPS_OPTIONS[0],
        "MINIMAX_REFMOD_TOKEN_CAP": REFMOD_TOKEN_CAP_OPTIONS[0],
        "MINIMAX_REFMOD_AUDIO": REFMOD_AUDIO_OPTIONS[0],
        "MINIMAX_REFMOD_AUDIO_CONCEPT": REFMOD_AUDIO_CONCEPT_OPTIONS[0],
        "MINIMAX_REFMOD_AUDIO_SECONDS": "30",
        "MINIMAX_REFMOD_CONCEPT": "identity",
        "DATASET_MEGAPIXELS": "1.0",
    },
    # Measured 10 Sep 2026: the Full-canvas mod with 16 references is the strongest file
    # (portraits 70 vs 60 for 8), and 200 optimisation steps add ~5 on top. A companion LoRA
    # in the same file was tried in eleven configurations and never lifted likeness beyond
    # noise while dropping portraits ~10 (pose drift) — removed 14 Sep 2026.
    "✨ RefMod — Fizgig recipe (Full reference, 16 refs at 1 MP, optimised)": {
        **_REFMOD_PRESET_BASE,
        "MINIMAX_REFMOD_GRID": REFMOD_GRID_OPTIONS[0],
        **REFMOD_DEFAULTS,
        # 1 MP references (the file's pixels); the optimiser's stills are cached at 0.25 MP
        # regardless (Peter, 16 Sep 2026)
        "DATASET_MEGAPIXELS": "1.0",
    },
    # The same recipe at 0.5 MP references: half the tokens per reference at generation
    # (~8,000 for 16, the library's budget) for a lighter file (Peter, 16 Sep 2026).
    "✨ RefMod — Fizgig recipe lite (Full reference, 16 refs at 0.5 MP, optimised)": {
        **_REFMOD_PRESET_BASE,
        "MINIMAX_REFMOD_GRID": REFMOD_GRID_OPTIONS[0],
        **REFMOD_DEFAULTS,
        "DATASET_MEGAPIXELS": "0.5",
    },
    # Style (16 Sep 2026): a look, not a person — plain encode of EVERY still (the optimiser's
    # loss rewards reproducing the stills themselves, which on a style set means their subjects
    # and framing; untested there, so Steps stays 0 and anyone can type a number). Community =
    # the pack's concept recipe, the pooled 16x16 grid (stacks under the cap, carries palette
    # and feel); high fidelity = Full at 0.25 MP (keeps texture, grain, brushwork).
    "✨ RefMod — style, community (16×16 grid, all refs, plain encode)": {
        **_REFMOD_PRESET_BASE,
        "MINIMAX_REFMOD_GRID": REFMOD_GRID_OPTIONS[2],
        "MINIMAX_REFMOD_REFS": "all",
        "MINIMAX_REFMOD_STEPS": REFMOD_STEP_OPTIONS[0],
        "MINIMAX_REFMOD_CLIPS": REFMOD_CLIPS_OPTIONS[0],
        "MINIMAX_REFMOD_TOKEN_CAP": REFMOD_TOKEN_CAP_OPTIONS[0],
        "MINIMAX_REFMOD_AUDIO": REFMOD_AUDIO_OPTIONS[0],
        "MINIMAX_REFMOD_AUDIO_CONCEPT": REFMOD_AUDIO_CONCEPT_OPTIONS[0],
        "MINIMAX_REFMOD_AUDIO_SECONDS": "30",
        "MINIMAX_REFMOD_CONCEPT": "style",
        "DATASET_MEGAPIXELS": "0.25",
    },
    # High fidelity means it (Peter, 16 Sep 2026): 1 MP references, ~1 000 tokens a still, so 8 of
    # them — the library's budget; the token readout warns anyone who takes it further.
    "✨ RefMod — style, high fidelity (Full at 1 MP, 8 refs, plain encode)": {
        **_REFMOD_PRESET_BASE,
        "MINIMAX_REFMOD_GRID": REFMOD_GRID_OPTIONS[0],
        "MINIMAX_REFMOD_REFS": "8",
        "MINIMAX_REFMOD_STEPS": REFMOD_STEP_OPTIONS[0],
        "MINIMAX_REFMOD_CLIPS": REFMOD_CLIPS_OPTIONS[0],
        "MINIMAX_REFMOD_TOKEN_CAP": REFMOD_TOKEN_CAP_OPTIONS[0],
        "MINIMAX_REFMOD_AUDIO": REFMOD_AUDIO_OPTIONS[0],
        "MINIMAX_REFMOD_AUDIO_CONCEPT": REFMOD_AUDIO_CONCEPT_OPTIONS[0],
        "MINIMAX_REFMOD_AUDIO_SECONDS": "30",
        "MINIMAX_REFMOD_CONCEPT": "style",
        "DATASET_MEGAPIXELS": "1.0",
    },
}


# ---- the run's rules -------------------------------------------------------------------------------------------
def plain_encode(steps_label) -> bool:
    """Steps 0: a plain encode of the references. Nothing trains, so no captions and no text-encoder caches."""
    return refmod_steps_value(steps_label) == "0"


def ref_pass_needed(steps_label, megapixels) -> bool:
    """Steps above 0 and Target MP other than 0.25: the references need their own cache pass at Target MP (the
    dataset's caches are the optimiser's, at 0.25)."""
    if plain_encode(steps_label):
        return False
    try:
        return abs(float(megapixels) - TRAIN_MP) > 1e-6
    except (TypeError, ValueError):
        return False


def toml_megapixels(steps_label, megapixels):
    """The dataset TOML's megapixels: the optimiser's 0.25 when it optimises, else Target MP."""
    return megapixels if plain_encode(steps_label) else TRAIN_MP


def ref_cache_dirs(folders, cache_root, cache_dir_for):
    """The '-refs' sibling of each dataset folder's cache folder (<cache root>/<folder>-<hash>, or the folder itself
    with no cache root). cache_dir_for(root, folder) is launch.cache_dir_for."""
    return [(cache_dir_for(cache_root, f) if cache_root else f).rstrip("/\\") + "-refs" for f in folders if f]


def token_estimate(grid_label, refs_label, megapixels):
    """(tokens per reference, total or None for 'all') for the card's readout. Full: a reference at Target MP is
    (W/16/2)*(H/16/2) tokens - MP*1e6/1024; a pooled grid g is about (g/2)^2."""
    g = refmod_grid_value(grid_label)
    try:
        mp = float(str(megapixels or "0.25").split(" ")[0])
    except ValueError:
        mp = 0.25
    per = int(round(mp * 1e6 / 1024.0)) if g == "full" else max(1, (int(g) // 2) ** 2)
    r = str(refs_label or "8").strip().lower()
    if r.startswith("all"):
        return per, None
    try:
        return per, per * max(1, int(float(r)))
    except ValueError:
        return per, per * 8


def token_text(grid_label, refs_label, megapixels):
    """(the card's readout line, over the pack's cap)."""
    per, total = token_estimate(grid_label, refs_label, megapixels)
    if total is None:
        return (f"≈ {per:,} tokens per reference at generation (the pack's {TOKEN_CAP:,} default fits "
                f"{max(1, TOKEN_CAP // per)})"), False
    over = total > TOKEN_CAP
    return (f"≈ {total:,} tokens at generation ({per:,} per reference), "
            f"{'above' if over else 'under'} the pack's {TOKEN_CAP:,} default"), over


# ---- commands ---------------------------------------------------------------------------------------------------
def _script(repo_dir, name):
    return os.path.join(repo_dir, "src", "fizgig", "scripts", name)


def cache_latents_command(python, repo_dir, dataset_config, vae, audio_vae, steps_label):
    """The H3 latent caching a RefMod run uses: --skip_existing (the skip checks the bucket, so it is safe on
    latents), the audio VAE whenever it is set (the script loads it only for clips), --clip_still (each clip lends its
    sharpest face), and --captions_optional on a plain encode (nothing trains)."""
    cmd = [python, _script(repo_dir, "minimax_cache_latents.py"), "--dataset_config", dataset_config,
           "--vae", vae, "--skip_existing"]
    if audio_vae:
        cmd += ["--audio_vae", audio_vae]
    cmd += ["--clip_still"]
    if plain_encode(steps_label):
        cmd += ["--captions_optional"]
    return cmd


def ref_cache_command(latents_cmd, megapixels):
    """The references' own pass at Target MP into the '-refs' sibling folder ('=' form: a bare '-refs' reads as a
    flag)."""
    return latents_cmd + ["--megapixels", str(megapixels).strip(), "--cache_suffix=-refs"]


def cache_text_command(python, repo_dir, dataset_config, text_encoder):
    return [python, _script(repo_dir, "minimax_cache_text.py"), "--dataset_config", dataset_config,
            "--text_encoder", text_encoder]


def maker_command(python, repo_dir, st, dit, ref_dit, audio_vae, ref_dirs=(), caching=True, console=None):
    """minimax_refmod.py: builds the mod from the caches and, with Steps above 0, optimises its latent against the
    frozen H3 base. st holds the card's labels (MINIMAX_REFMOD_*), MINIMAX_TRAIN_BASE ("ref2va"/"fl2va"), SEED,
    BLOCKS_SWAP ("auto" or a number), DATASET_CONFIG, LORA_OUTPUT_DIR, LORA_NAME and DATASET_MEGAPIXELS. ref_dirs:
    ref_cache_dirs() for the run; with caching off only one that exists is passed (console(line) says so)."""
    d = REFMOD_DEFAULTS
    refs = str(st.get("MINIMAX_REFMOD_REFS", d["MINIMAX_REFMOD_REFS"]) or "16").strip().lower()
    refs_arg = "10000" if refs.startswith("all") else refmod_num(refs, "16", int)
    steps = refmod_steps_value(st.get("MINIMAX_REFMOD_STEPS", d["MINIMAX_REFMOD_STEPS"]))
    ref2va = st.get("MINIMAX_TRAIN_BASE") == "ref2va"
    concept = str(st.get("MINIMAX_REFMOD_CONCEPT", "") or "").strip()
    cmd = [
        python, _script(repo_dir, "minimax_refmod.py"),
        "--dit", ref_dit if (ref2va and ref_dit) else dit,
        "--dataset_config", st["DATASET_CONFIG"],
        "--output_dir", st["LORA_OUTPUT_DIR"],
        "--output_name", st["LORA_NAME"],
        "--grid", refmod_grid_value(st.get("MINIMAX_REFMOD_GRID")), "--steps", steps, "--max_refs", refs_arg,
        "--clips", refmod_clips_value(st.get("MINIMAX_REFMOD_CLIPS", d["MINIMAX_REFMOD_CLIPS"])),
        "--concept_type", concept if concept in REFMOD_CONCEPT_OPTIONS else d["MINIMAX_REFMOD_CONCEPT"],
        "--seed", str(st.get("SEED", 42) or 42),
        "--base_model", "ref2va" if ref2va else "fl2va",
    ]
    desc = str(st.get("MINIMAX_REFMOD_DESC", "") or "").strip()
    if desc:
        cmd += ["--description", desc]
    cap = refmod_token_cap_value(st.get("MINIMAX_REFMOD_TOKEN_CAP", d["MINIMAX_REFMOD_TOKEN_CAP"]))
    if cap != "0":
        cmd += ["--token_cap", cap]
    if refmod_audio_on(st.get("MINIMAX_REFMOD_AUDIO", d["MINIMAX_REFMOD_AUDIO"])):
        ac = str(st.get("MINIMAX_REFMOD_AUDIO_CONCEPT", "") or "").strip()
        cmd += ["--audio", refmod_audio_value(st.get("MINIMAX_REFMOD_AUDIO", d["MINIMAX_REFMOD_AUDIO"])),
                "--audio_max_seconds", refmod_num(st.get("MINIMAX_REFMOD_AUDIO_SECONDS"), "30", float),
                "--audio_concept", ac if ac in REFMOD_AUDIO_CONCEPT_OPTIONS else d["MINIMAX_REFMOD_AUDIO_CONCEPT"]]
        if audio_vae:
            cmd += ["--audio_vae", audio_vae]
    bs = str(st.get("BLOCKS_SWAP", "auto") or "auto").strip()
    cmd += ["--blocks_to_swap", "auto" if bs.lower().startswith("auto") else bs]
    # NF4 always (Peter, 16 Sep 2026): 10.5 GB resident, so the step fits without recompute on 32 GB and without
    # streaming on 16/24 GB; the precision box is H3 LoRA/FT's and does not apply here.
    cmd += ["--base_quant", "nf4"]
    if ref_pass_needed(st.get("MINIMAX_REFMOD_STEPS", d["MINIMAX_REFMOD_STEPS"]), st.get("DATASET_MEGAPIXELS")):
        # With caching off the reference pass never runs, so only a '-refs' folder that already exists (an earlier
        # run's) can be used; otherwise the references come from the dataset caches and the console says so.
        for rd in ref_dirs:
            if caching or os.path.isdir(rd):
                cmd += ["--ref_cache_dir", rd]
            elif console is not None:
                console(f"[refmod] caching is off and {rd} does not exist — the references come from the dataset "
                        f"caches (0.25 MP) this run; tick Enable Cache for a Target MP reference pass\n")
    # No previews (Peter, 16 Sep 2026): a mod is judged in RefMod Toolkit (the script keeps its --sample_* flags for
    # command-line sweeps).
    return cmd


# ---- the run (the web UI's; the desktop app builds the same through these pieces) ---------------------------------
SAFE_NUMBERS = {"NETWORK_DIM": "16", "NETWORK_ALPHA": "16", "LEARNING_RATE": "1e-4", "MAX_TRAIN_EPOCHS": "1",
                "SAVE_EVERY_N_EPOCHS": "1", "LORA_LR_RATIO": "1", "GRADIENT_ACCUMULATION": "1", "MAX_GRAD_NORM": "1.0"}


def dataset_toml(folder, megapixels, caption_ext, cache_root, clip_megapixels, cache_dir_for):
    """The dataset TOML the desktop writes for RefMod: one folder, a square of toml_megapixels on the 16 grid,
    clip_megapixels when the folder holds a clip, its own cache folder."""
    side = int((float(megapixels) * 1_000_000) ** 0.5) // 16 * 16
    lines = ["[general]", f"resolution = [{side}, {side}]", f'caption_extension = "{caption_ext}"', "batch_size = 1",
             "num_repeats = 1", "enable_bucket = true", "bucket_no_upscale = true"]
    try:
        cmp_ = float(clip_megapixels or 0)
    except (TypeError, ValueError):
        cmp_ = 0.0
    try:
        has_clip = any(f.lower().endswith(".mp4") for f in os.listdir(folder))
    except OSError:
        has_clip = False
    if cmp_ > 0 and has_clip:
        lines.append(f"clip_megapixels = {cmp_:g}")
    lines += ["", "[[datasets]]", f'image_directory = "{folder.replace(chr(92), "/")}"']
    if cache_root:
        lines.append(f'cache_directory = "{cache_dir_for(cache_root, folder).replace(chr(92), "/")}"')
    return "\n".join(lines) + "\n"


def start_problems(st, models, image_folder, caption_ext=".txt"):
    """What Start refuses for a RefMod run, in the desktop's order and words. st: the card's labels, LORA_NAME,
    LORA_OUTPUT_DIR, DATASET_CONFIG, BLOCKS_SWAP, MINIMAX_TRAIN_BASE; models: {pref key: path}."""
    from fizgig.families import checks
    v = {**SAFE_NUMBERS, **st}
    out = checks.dataset_config(v.get("DATASET_CONFIG"), must_exist=False)
    for key, label in (("minimax_dit", "MiniMax H3 DiT"), ("minimax_vae", "MiniMax H3 Video VAE"),
                       ("minimax_text_encoder", "Qwen3-VL-32B text encoder")):
        path = models.get(key, "")
        if not path:
            out.append(f"{label} path is empty (set it on the Preferences tab)")
        elif not os.path.exists(path):
            out.append(f"{label} file does not exist: {path}")
    if refmod_audio_on(st.get("MINIMAX_REFMOD_AUDIO")):
        avae = models.get("minimax_audio_vae", "")
        if not avae:
            out.append("Audio mod is on but the MiniMax H3 Audio VAE path is empty — set it on the Preferences tab, or "
                       "set Audio mod to off")
        elif not os.path.exists(avae):
            out.append(f"MiniMax H3 Audio VAE file does not exist: {avae}")
        try:
            secs = float(refmod_num(st.get("MINIMAX_REFMOD_AUDIO_SECONDS", "30"), "30", float))
        except (TypeError, ValueError):
            secs = 0.0
        if not secs > 0:
            out.append("Audio mod: 'Up to' must be a number of seconds above 0")
    if st.get("MINIMAX_TRAIN_BASE") == "ref2va" and not models.get("minimax_ref_dit"):
        out.append("Training Base is set to Reference (ref2va) but 'DiT (reference)' is empty on the Preferences tab. "
                   "Easiest fix: on the Preferences tab tick 'Include the reference DiT (+21 GB)' and hit '⬇ Download "
                   "models for me' — it fetches the model and fills the path in for you. Or switch Training Base back "
                   "to First/last frame.")
    bs = str(v.get("BLOCKS_SWAP") or "auto").strip()
    swap = 0 if bs.lower().startswith("auto") else (int(bs) if bs.isdigit() else None)
    name_error = checks.tidy_name(v.get("LORA_NAME"))[1]
    out += checks.run(v, blocks_swap=swap, swap_max=BLOCKS_SWAP_MAX, arch_label=LABEL, name_error=name_error)
    from fizgig.dataset.image_dataset import IMAGE_EXTENSIONS
    out += checks.training_folder(image_folder, caption_ext, check_captions=not plain_encode(st.get("MINIMAX_REFMOD_STEPS")),
                                  media_exts={e.lower() for e in IMAGE_EXTENSIONS} | {".mp4"})
    return out


def plan(st, models, *, python, repo_dir, image_folder, cache_root, cache_dir_for, caching=True, caption_ext=".txt",
         clip_megapixels="0.25"):
    """The RefMod launch as a launch.LaunchPlan: problems first; then the dataset TOML, latent caching, the
    references' own pass at Target MP when it optimises at another size, text caching unless it is a plain encode,
    then the maker. caching False runs the maker alone (the desktop's Enable Cache untick)."""
    from fizgig.families.checks import tidy_name
    from fizgig.families.launch import LaunchPlan, Stage
    st = dict(st)
    if image_folder and not os.path.isabs(image_folder):
        image_folder = os.path.abspath(image_folder)
    st["LORA_NAME"] = tidy_name(str(st.get("LORA_NAME") or ""))[0]
    p = LaunchPlan(problems=start_problems(st, models, image_folder, caption_ext))
    if p.problems:
        return p
    steps = st.get("MINIMAX_REFMOD_STEPS", REFMOD_DEFAULTS["MINIMAX_REFMOD_STEPS"])
    mp = st.get("DATASET_MEGAPIXELS") or "0.25"
    try:
        p.dataset_toml = dataset_toml(image_folder, toml_megapixels(steps, mp), caption_ext, cache_root,
                                      clip_megapixels, cache_dir_for)
    except (TypeError, ValueError):
        p.problems.append("Target MP must be a number above 0")
        return p
    p.files.append((st["DATASET_CONFIG"], p.dataset_toml))
    p.dirs.append(st["LORA_OUTPUT_DIR"])
    refs = ref_cache_dirs([image_folder], cache_root, cache_dir_for) if ref_pass_needed(steps, mp) else ()
    maker = maker_command(python, repo_dir, st, models.get("minimax_dit", ""), models.get("minimax_ref_dit", ""),
                          models.get("minimax_audio_vae", ""), refs, caching, p.console.append)
    if caching:
        latents = cache_latents_command(python, repo_dir, st["DATASET_CONFIG"], models.get("minimax_vae", ""),
                                        models.get("minimax_audio_vae", ""), steps)
        p.stages.append(Stage("Cache Preparation", latents))
        if ref_pass_needed(steps, mp):
            p.console.append(f"Encoding the references at {mp} MP after the latent caching.\n")
            p.stages.append(Stage("Reference Caching", ref_cache_command(latents, mp)))
        if plain_encode(steps):
            p.console.append("Plain encode — no captions needed, skipping text encoder caching.\n")
        else:
            p.stages.append(Stage("Text Encoder Caching", cache_text_command(python, repo_dir, st["DATASET_CONFIG"],
                                                                             models.get("minimax_text_encoder", ""))))
    p.stages.append(Stage("Training", maker))
    return p
