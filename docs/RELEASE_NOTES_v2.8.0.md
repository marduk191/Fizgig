# Fizgig v2.8.0 — The Reliability Release

This is the largest release since Krea 2 landed — and most of it is invisible: **over 60 real
bugs found, fixed, and verified**, from an adversarial line-by-line audit of the whole app,
with every fix validated by tests and real training runs (including live pause/resume cycles on
both model families). Plus a genuine Krea 2 speed jump, smarter VRAM planning, and the sample
controls Krea 2 was missing.

---

## 🚀 Krea 2 got fast

- **INT8 W8A8 auto strategy** — with Blocks Swap and 4-bit Base on **Auto**, Fizgig now picks
  the best of INT8 (fastest, ~7× more accurate than NF4, exact gradients), NF4, or fp8 from
  your **free** VRAM — read fresh at launch, not from a stale snapshot.
- **torch.compile** — the transformer blocks compile automatically on longer runs: roughly
  **2× faster steady-state steps** on the INT8 path after a one-off warm-up. triton now
  installs with the requirements; on Windows you also need the MSVC C++ Build Tools — the
  installer and updater check and print the **direct download link**
  ([aka.ms/vs/17/release/vs_BuildTools.exe](https://aka.ms/vs/17/release/vs_BuildTools.exe),
  "Desktop development with C++" workload).
- **Shape-aware VRAM budget** — the strategy now accounts for your actual run shape. Measured:
  each extra image of batch size costs **~2.4 GB** (the old budget's blind spot — batch 2
  could sail through the check and OOM), resolution is nearly free at batch 1, rank is minor.

**Where this lands:** with INT8 + compile, Fizgig's per-step speed is now **approximately on a
par with OneTrainer**. And raw step speed isn't the whole race — Fizgig's real-time dataset
intelligence (the per-image loss watch you can enable with one click on the Training tab:
detect problem images, throttle stuck ones, auto-recaption) showed **faster likeness and a
higher final ceiling in matched-epoch A/B runs**. Same step speed, smarter steps — so
wall-clock time to a *good* LoRA should now favour Fizgig.

## 🎛 Krea 2 sample controls, wired

The Samples tab's **Steps** and **Sample at Start** now actually reach the Krea 2 trainer —
Sample at Start renders an epoch-0 baseline before training begins. **Metadata**
(title/author/description/license/tags) is now recorded in saved Krea 2 LoRAs as
`modelspec.*` keys.

## 🧠 Adaptive LR: two knobs, not three

With Adaptive LR on, the Learning Rate box is now **ignored and greyed out** — you choose the
Min/Max window and the run starts at its geometric midpoint (1e-4 & 4e-4 → 2e-4), with the
watcher owning the LR from there. Previously the LR box silently set the starting LR while
Min/Max looked authoritative. Resumed runs keep their mid-flight LR.

## 🛠 The reliability overhaul (highlights of 60+)

**Silently-wrong-training fixes**
- "Model Area to Train" now actually restricts training — the block presets previously trained the full model (retrain targeted LoRAs to get the real behaviour)
- Resume + gradient accumulation ≥ 2 trained *nothing* (checkpoints still wrote normally) — fixed and verified live
- `sigma` timestep sampling used min/max as indices into a descending schedule — a 0–400 "late" window actually selected 1000–600 (CLI/TOML runs; GUI runs were unaffected)
- OneTrainer/ai-toolkit LoRAs with `alpha = rank/2` loaded with their attention at **2× strength** (QKV fusion discarded per-slot alphas)
- The per-image loss watch now restores **faithfully** on resume — verified state-identical against live runs (incorrigibility applied at the right boundary, multi-fix caption history, exclusions purged-not-pardoned, LR scheduler position exact)

**Data-loss fixes**
- Image Prep could **destroy a photo**: `photo.jpg` + `photo.png` in one folder overwrote the unrelated `.png` then deleted the `.jpg`. Collisions now divert to `_2.png` with a log line; in-place saves are atomic
- Look Filter baseline protection never worked on Windows (path-separator mismatch) — baselines could be auto-suggested and moved out of the dataset
- `prefs.json` / `last_used.json` writes are atomic — a crash mid-write no longer blanks your model paths
- Find & Replace no longer writes literal newlines into captions when the replacement contains backslashes

**Runs that no longer die or lie**
- Closing the window during training used to **orphan the trainer** (14–20 GB of VRAM, only Task Manager could stop it) — now it asks, then stops the whole tree
- A stale pause flag no longer truncates the next run to one epoch; the trainer consumes it itself
- Klein preview failures no longer kill an hours-long run (Krea 2 parity)
- A typo in any numeric field now names the field instead of making Start silently do nothing
- Settings shown on screen now match what runs: arch switches preserve your Timestep/Samples config, presets restore their saved timesteps, a finished run clears its resume path, and the OOM popup no longer *changes your training config as a side effect of reporting an error*

**Tool-output fixes**
- Loading a donor in Repair Studio no longer doubles every shared block's rank with dead zero-weights before you touch a slider
- Extraction: zero-layer results are an error (not "Extraction complete!"), Fast presets no longer demand models they never load, `--source_multiplier` works on all three paths, outputs never silently overwrite
- Repaired LoRAs no longer ship the source's content hash and pre-bake rank in their metadata
- The gallery serves the right run's checkpoint in reused output folders, and Chinese sample filenames no longer break its refresh

## 🧹 Removed

- **prodigy, came, and adafactor** optimizers — the "manage their own LR" family. They conflict
  with Adaptive LR by design and produced real failures (prodigy silently became AdamW at
  lr=1.0 when its package was missing; adafactor crashed against the adaptive watcher). Saved
  configs fall back to `adamw8bit` cleanly. Power users can still reach anything via the
  `module.path.ClassName` form.
- 144 lines of dead settings code.

## 💬 Quality of life

- The slow start of the first two epochs (kernel planning, cache warm-up) now announces itself
  in the console every ~30 s — "Nothing is stuck; full speed arrives from epoch 3."
- The 4-bit Base control is an explicit **Auto / On / Off** dropdown — Auto hands the choice to
  the memory strategy; an explicit choice is never overridden.
- The per-image watch toggles grey out at Batch Size > 1 (a batch-mean isn't a per-image
  signal), with a note explaining why.

## Upgrading

Run `update_fizgig.bat` (or `git pull` + reinstall requirements). Caches, prefs, and presets
carry over; old presets that referenced removed optimizers or withdrawn options are sanitised
on load with a console note.

**One new (optional) Windows dependency for the compile speedup: the MSVC C++ Build Tools.**
triton installs automatically with the requirements, but torch.compile also needs a C++
compiler on Windows. Both the installer and `update_fizgig.bat` check for it and, if it's
missing, print the exact installer link —
**[aka.ms/vs/17/release/vs_BuildTools.exe](https://aka.ms/vs/17/release/vs_BuildTools.exe)**,
tick the **"Desktop development with C++"** workload (or run the winget one-liner they print).
Without it, everything works exactly as before — you just don't get the ~2× compiled step
speed on Krea 2.
