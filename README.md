<h1 align="center">Fizgig — Klein 9B & Krea 2 LoRA Studio</h1>

<p align="center">
  <strong>Fix broken LoRAs without retraining. Remix any LoRA into new variations in seconds.</strong><br>
  A train · repair · explore workbench built end-to-end for <strong>Flux 2 Klein 9B</strong> and <strong>Krea 2</strong>.
</p>

<p align="center">
  <a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>
</p>

<p align="center">
  <a href="https://youtu.be/sH-kGR8yzBU"><img src="assets/hero.png" alt="Fizgig LoRA Studio — now with Krea 2 support" width="600"></a><br>
  <em>Watch the full walkthrough on YouTube</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/models-Klein%209B%20%2B%20Krea%202-blue?style=for-the-badge" alt="Klein 9B + Krea 2">
</p>

> **🎉 New — Krea 2.** Fizgig now supports a **second, fully native model family**: **Krea 2 (12.9B)**. The whole workbench works with it — Repair Studio, Explorer, Royale, Profiler, Extract — plus Context LoRA, Adaptive LR, Pause/Resume, 4-bit (NF4) low-VRAM training, and the live sample override. [Details below ↓](#krea-2--second-model-family)

> **🆕 Newest — the run that looks after itself.** The Krea 2 trainer now **curates your dataset live**: it detects problem images from their loss alone, throttles them, has the text encoder *look at* the stuck ones and rewrite their captions, warms in unusual angles gently, and **tells you the best epoch** when the run plateaus. [Details ↓](#the-trainer-curates-your-dataset-while-it-trains-krea-2-experimental) Around it, two new instruments for **both families**: the Image Prep tab's **Look Consistency Filter** pre-filters off-look images by face-embedding score, and the **sample gallery** now scores every sample's likeness against your own photos live during training — with a Royale-style **Training Run Visualiser** to scrub and export the run. [Details ↓](#the-sample-gallery-is-an-instrument-both-families)

---

## What Fizgig is

Every trainer makes LoRAs. Fizgig is built around what you do with them **afterwards** — and that's the part nobody else has.

- **Fix** a baked LoRA block-by-block, no retraining — overbaked identity, crushed style, drag a slider, save a new `.safetensors`.
- **Explore** new variations like a game — the app proposes mutations, you pick favourites, the LoRA evolves through selection.
- **Find** the best LoRA by eye — **LoRA Royale** renders every epoch of a run (or any folder of LoRAs) on one seed; crossfade to the one that *feels* right.
- **Share** what you made — LoRA Royale exports the epoch morph, or travels a single LoRA through seeds, prompts, or strength, as a looping MP4/GIF made to share.
- **Profile** exactly which blocks carry identity, style, and detail — so you know what to touch before you touch it.

Under that workbench sits a fast, light trainer tuned for its models — and tuned to **fit your GPU**, not a datacenter's. Because everything is built natively for Klein 9B and Krea 2 instead of bolted onto a dozen models, Fizgig can do things the generalists can't:

- **Big models on modest cards.** A full **Klein 9B** LoRA trains on a **16 GB card** — and the 12.9B **Krea 2** trains on a **10–12 GB card** thanks to the 4-bit (NF4) base (~8 GB resident, QLoRA-style: the base is 4-bit, your LoRA still trains in bf16 on top). Block swap and previews **size themselves to your VRAM automatically** — nothing to configure — and if a preview can't fit, it steps aside so **training keeps running and saving**. You don't need a 4090 to train on the newest 12.9B model.
- **A workbench nobody else has.** Repair broken LoRAs block-by-block, evolve new ones like a game, and crossfade every epoch of a run to find the sweet spot — then the tools **read each other's output** (profile → repair → explore → compare, one closed loop).
- **It just works on your files.** Loads kohya / PEFT / OneTrainer / AI-Toolkit / LyCORIS, auto-converted; saves kohya `.safetensors` that drop straight into ComfyUI.

**📣 Help map Krea 2 — [open an issue](https://github.com/shootthesound/Fizgig/issues).** Krea 2's per-block roles — which blocks carry **identity, style, and detail** — aren't charted yet, which is why the colour-coded sliders and layer-targeting presets are Klein-only for now. The **Profiler** is the instrument for finding them: spot a pattern, share it in **[GitHub Issues](https://github.com/shootthesound/Fizgig/issues)**, and it directly drives the colour-coding and finer layer targeting coming to Krea 2's presets and Repair Studio.

**Free and open source.** A good first run is the **✨ Old Reliable** preset on the Training tab — then try **✨ Old Reliable · Flavour 8** (rank 8). Much of the old rank-16 instinct predates models this size; on Klein 9B, rank 8 is often plenty.

---

## The workbench

The reason to use Fizgig. Each tool works on a trained run's output **or any Klein LoRA you've downloaded** — and they hand off to each other.

### Repair Studio
Thirty-two live sliders — one per transformer block — with a side-by-side Distilled preview that updates instantly. **Turbo Preview** — activation caching for a live LoRA-tweaking UI, which no other LoRA tool does — caches per-block outputs and prompt encodings across the denoising steps, so late-block edits redraw up to **97% faster**; the baked save is always exact, Turbo or not. Quick-set buttons on every slider (`[0]` `[1]` `[±]` `[⚖]`); **Balance** holds the combined primary + donor weight at 1.0 per block, ideal for cross-fading two LoRAs. Optional donor-LoRA blending mixes blocks from a second LoRA via rank concatenation. Previews can be conditioned on a **reference image** (Klein is an edit model), so you see how your LoRA edits a real photo. Click a preview to pop it into a resizable window. Browse a new LoRA and it auto-swaps — no manual reset. Saves a baked `.safetensors` that works in ComfyUI at strength 1.0.

### LoRA the Explorer
Evolutionary discovery. The app mutates blocks and shows four variants — pick a favourite and it becomes the new baseline. **Freeze Tweaked Blocks** locks what you like so future mutations only touch the rest. A **Structure** slider sets how far the composition anchor drifts each round; seed cycling checks variants across seeds. Found a direction you love? **Refine this baseline in Repair Studio** sends all 32 slider values straight over — and Repair Studio sends state back the same way. Discover → refine → discover, in a loop.

### LoRA Royale
Find the best LoRA the human way — then turn the winner into share-ready clips. Point it at a training output folder and it renders **every epoch on one fixed seed** (Distilled 4-step), with a **crossfade slider** that blends smoothly between consecutive epochs — drag until it looks best and stop. A thumbnail grid sits below; click any epoch to jump there. Drop in a **reference image** (Klein is an edit model) and every epoch edits the same photo. An optional **likeness score** (InsightFace ArcFace, CPU — no extra VRAM) rates each epoch against a training shot — **close-up headshots included** (it pads-and-retries so a face that fills the frame still detects) — and flags the closest in gold with one-click **Jump to best**. **Export likeness clip** turns it into a share-ready, side-by-side *subject vs each epoch* comparison with the score burned in, morphing epoch by epoch. **Promote** copies the winner to a clean `.safetensors`. Not a training run? Point it at **any folder of LoRAs** and it compares them by name — or flip to **Single-LoRA mode** to run everything below on one downloaded LoRA, no folder required.

**Comparison sheet** builds the share image people actually post for a new LoRA — one row per prompt, columns for *without LoRA* vs *with LoRA* (or one column per epoch), the same seed across a row so only the LoRA changes, headers and captions drawn in, saved as a single PNG. The no-LoRA column also strips your trigger word, so the baseline isn't fed a token the base model has never seen.

Because the morph *is* the magic, the payoff is four **travel** tools that each render a sequence you **scrub to review and only save if you like it** — as a looping MP4 or GIF, re-saveable in either format without re-rendering, with an optional **deflicker** pass (the timelapse trick DaVinci uses) for flicker-free clips. **Export the morph** saves the whole epoch sweep, a face resolving epoch by epoch — or **Save all stills** dumps every rendered epoch to a folder as full-res PNGs (the renders otherwise live only in memory). **Seed travel** slerps through a journey of seeds to show the LoRA's range. **Prompt travel** interpolates the text embedding through waypoints — Time of day, Season, Age, Era, or your own words — so one subject flows through the change; pick a **Preset + Subject** and it writes the prompt for you. And **LoRA strength travel** ramps the LoRA from 0 (base model) to full and beyond, so you literally *watch the effect fade in*. Every travel can be anchored to a reference to hold the subject steady, with interpolation and seed-drift knobs for a smooth, brightness-even result. (The epoch morph shows the LoRA *learning*; the travels show what it can *do*.)

### Profiler
A per-block activation profile with a colour-coded, five-bucket HTML report — which blocks carry style, identity, and detail signal, and where they overlap. Writes a JSON sidecar that Repair Studio reads automatically, showing the findings inline when you load the same LoRA. Krea 2 LoRAs get a weight-only per-block report (no models loaded) — the instrument for the community block-mapping effort below.

### Extract
Distil any Klein or Krea 2 LoRA to a lower rank — Klein with block and timestep targeting, Krea 2 via weight-only SVD. Fast presets run pure weight SVD with no GPU models loaded; activation-weighted presets (Klein) use forward passes for better accuracy. Supports PEFT and LyCORIS (LoKR / LoHa) sources. Expect roughly **5 minutes for a full-model Klein LoRA and ~25 for Krea 2** (its 264 modules are 6144-wide) — a long quiet stretch mid-extract is normal, not a hang.

---

## Krea 2 — second model family

Krea 2 is a from-scratch **native** port — no external tooling at runtime: a 12.9B single-stream MMDiT, the Qwen-Image VAE, and a Qwen3-VL-4B text encoder. **Train on the RAW model** (fp8, ~14 GB resident) and **preview on the fp8 Turbo** (8-step, CFG-free) with your live LoRA applied. Pick it from the **Base Model selector** at the top of the Training tab.

Everything works on Krea 2: **all five workbench tools** (Profiler, Extract, Repair Studio, Explorer, Royale), plus **Pause/Resume** (full state), **Context LoRA**, **Adaptive LR**, **reference images** (through the text encoder's vision path — "prompt from a picture"), and the live sample override. A few Training-tab controls are hidden for Krea 2 for now (not removed): per-block Model-Area targeting (no Krea 2 block map yet), the Timestep section, and the FP8-Scaled / FP8-TE / Gradient-Checkpointing toggles.

> **📣 Help map Krea 2's blocks — [open an issue](https://github.com/shootthesound/Fizgig/issues).** The colour-coded sliders, Model-Area targeting, and block-aware presets are Klein-only right now because Krea 2's per-block roles (which blocks carry **identity**, **style**, and **detail**) aren't mapped yet. The **Profiler**'s weight-only report is the instrument for discovering them. If you find patterns — a block that clearly drives identity, a range that governs style — please share your findings in the **[GitHub Issues](https://github.com/shootthesound/Fizgig/issues)**. Community block-discovery is what will drive the colour-coding and finer layer targeting coming to the Krea 2 presets and Repair Studio.

**Runs on smaller cards, and adapts to yours.** Krea 2 is a bigger model than Klein, but the low-VRAM paths are wired — and with everything on **Auto**, Fizgig plans the whole run for you:

- **Auto memory strategy** — leave Blocks Swap and 4-bit Base on **Auto** and Fizgig picks the best of **INT8 W8A8** (fastest, near-exact — the default wherever it fits), **NF4 4-bit**, or fp8 from your *free* VRAM — budgeted for your actual run shape (batch size is the big cost: ~2.4 GB per extra image, measured). The console explains what it chose and why.
- **torch.compile speedup** — on longer runs the transformer blocks compile automatically (needs the MSVC C++ Build Tools on Windows; triton installs with the requirements). Roughly 2× faster steady-state steps on the INT8 path after a one-off warm-up — putting per-step speed **approximately on a par with OneTrainer**. Combine that with the real-time dataset intelligence below (which showed faster likeness and a higher ceiling in matched-epoch A/Bs) and time-to-a-*good*-LoRA should now favour Fizgig: same step speed, smarter steps.
- **4-bit (NF4) base** — the base trains frozen at ~5.6 GB (base + LoRA ~8.3 GB), so a full Krea 2 LoRA fits a **10–12 GB card with no block swap** — QLoRA-style, the LoRA still trains in bf16 on top. Auto picks it when it's the right call; the *4-bit Base* dropdown forces it On/Off.
- **VRAM-adaptive previews & workbench** — the fp8 Turbo (used for in-training previews *and* Repair Studio / Explorer / Royale) auto-sizes its own block swap to your GPU, so the tools fit smaller cards without you tuning anything.
- **Previews never crash a run** — if the Turbo preview can't fit even swapped, previews auto-disable and **training keeps going and saving**; evaluate the LoRA in ComfyUI. (With 4-bit, the base even parks off the GPU during each preview, so the two coexist.)

Krea 2 trains real, ComfyUI-compatible LoRAs, and its training recipe is verified against the reference implementation — same noised/target flow-matching, `krea2_shift` timestep sampling, and gradient clipping.

### The trainer curates your dataset while it trains (Krea 2, experimental)

Four Training-tab toggles turn a run into a live dataset curator — no other trainer does any of this:

- **Detect problem images** — every image's loss is tracked across epochs, normalized for the random noise level each step draws (raw per-step loss mostly ranks the dice roll, not the image). Images that stay hard **without improving** get flagged in the console and in the live **Problem Images window** (thumbnails, verdicts, per-image trends, auto-refreshing every epoch). In real runs the top flags were all caption/image mismatches — e.g. from-behind shots whose captions never said so. The detector finds them from the loss trajectory alone.
- **Per-image adaptive LR** — flagged images are throttled (suspects ×0.7 from ~epoch 3, confirmed-stuck ×0.5 escalating toward ×0.1) so one bad caption can't keep yanking the weights all run, while fully-mined images ease off to prevent overbake and consistently-healthy learned images get a gentle ×1.1 boost. In matched-epoch A/Bs this gave faster likeness *and* a higher final ceiling — with real skin texture where the untreated run went plastic.
- **Auto-recaption stuck images** — the same Qwen3-VL that conditions training *looks at* each confirmed-stuck image between epochs, rewrites its caption from what's actually visible (appending your trigger word if set), re-encodes it, and gives the image a fresh start. A second attempt goes exhaustive-detail; still stuck after two means the image is **excluded** for the rest of the run — so the loss average stops carrying its permanent error term — and the exclusion is remembered per-dataset (`fizgig_excluded.json`, travels with your images). Fix the caption and it's automatically re-admitted.
- **Warm up look outliers** — real-but-unusual images (tight angles, profiles, occlusion) that the Look Consistency Filter scored as outliers keep their unique information but **ease in at ×0.4 LR**, ramping to full over the first ~4 epochs — refining the identity instead of fighting it while it forms — and release to full LR early the moment they start improving. Prior-then-evidence: the face-embedding score covers exactly the epochs before the loss watch has a trend to act on.

You can also edit any caption yourself mid-run from the Problem Images window — the trainer re-encodes it at the next epoch boundary, no restart. Once nothing is improving any more, the watch tells you you're **done**: a plateau banner with a best-checkpoint estimate and a suggested epoch window to scrub in LoRA Royale — and it's honest about certainty, distinguishing a *provisional* plateau (images still being adjudicated that may give the run a second wind) from a *confirmed* one. And pausing or restarting loses nothing: a **resumed run replays its own loss log** to restore every verdict, trend, and exclusion exactly where they were.

---

## Training

The foundation: fast, light, and tuned for one model.

- **Proven presets** for rank 4–16, single subject through multi-character — or roll your own.
- **Context LoRA** — load an existing LoRA as a frozen *active* layer so the new one learns to coexist. Train a face on top of a style and they stop fighting at inference; train an outfit on top of a character and the clothes drape correctly. No other trainer does this.
- **Distilled training samples** — 4-step previews that match ComfyUI output closely (a separate Distilled DiT, ComfyUI Euler Simple schedule). On by default; toggle on the Samples tab. On tight cards the sample model auto-swaps its own blocks by VRAM so 4-step previews keep working on 16 GB. On 24 GB+ it stays resident and is cached in system RAM between epochs (RAM-checked, saves ~3–4 s/epoch).
- **Reference-conditioned samples** — Klein is an edit model, so previews can *edit* a reference photo instead of generating from scratch. Auto-resized to ~0.20 MP so it can't OOM; works on Base and Distilled samples.
- **Adaptive LR** — a bi-directional plateau tracker that probes up on steady loss descent and pulls down (with optional weight rollback) on plateau, heavy gradient clipping, or weight-norm runaway. Two knobs, not three: you set the **Min/Max window** and the run starts at its geometric midpoint — the Learning Rate box greys out and is ignored while adaptive is on.
- **fp8 Base training** — the fp8 Base DiT stays resident at ~9.6 GB instead of dequantising to ~18 GB, so a full 9B LoRA trains in ~14 GB and fits a 16 GB card — lossless, no quality cost. Automatic (Fizgig detects the pre-quantised file), no flag.
- **Gradient checkpointing toggle** — on by default (it's what fits a 9B LoRA on 16 GB). Turn it **off** on a 24 GB+ card for meaningfully faster steps. A VRAM-aware warning fires if you switch it off on a card that can't spare the activation memory.
- **Pause / Resume** — graceful epoch-boundary pause that frees your GPU mid-run and resumes with full optimizer state and no quality regression. Fire up Rocket League, come back, carry on.
- **Model Area targeting** — train only Identity, Style, or Detail blocks, or the full model.
- **Auto VRAM management** — block swap auto-detects from GPU VRAM; OOM detection tells you exactly what to change. Supports bf16 and fp8 Base DiT, with block swap.
- **Per-dataset caches, cross-checked** — every dataset gets its own cache folder, and the trainer verifies each cached item against the images actually in your folder before training. Deleted an image? It's gone from the run. Switched datasets? The old one can never leak in.
- **Diffusers LoRA support** — OneTrainer LoRAs with split Q/K/V keys are auto-fused on load.

> **A note on Base previews:** the default Distilled 4-step previews track ComfyUI closely, including with a Context LoRA active. Only **Base multi-step** previews (Distilled toggled off) can look softer than the deployed LoRA — they come from a mid-training fp8 checkpoint, so colours and detail can be slightly off even when the LoRA is excellent. Judging from Base previews? Confirm final quality in ComfyUI.

### Live status bar
A bottom bar with stacked **VRAM and system-RAM gauges** (smooth gradient fills, plus a per-run peak marker so you can see how high a run pushed memory). VRAM is read at the device level, so it catches other apps holding the GPU too. A top-right **IDLE / BUSY** light shows at a glance whether the app is working. Hide or show the whole bar with one click; it remembers.

Beside it sits a **live sample override** — tick it to set a prompt, seed, width/height, and optional reference image for the *next* samples, mid-run, no restart. The text encoder only re-runs when the prompt text changes, so seed / resolution / reference tweaks are instant.

### The sample gallery is an instrument (both families)

The browser gallery of training samples now *measures* the run instead of just showing it — on Klein and Krea 2 alike:

- **Live likeness scoring** — pick the **3 dataset photos** that best nail the look, and every sample gets a colour-coded likeness badge (ArcFace face embeddings averaged across all three baselines — one photo would bias every score with its own angle and lighting). Scoring runs on **CPU with zero impact on training speed**, newest samples first, and keeps up live as each epoch's previews land. A **trend chart** plots per-epoch average likeness for the current run with the best epoch highlighted — an objective likeness-vs-epoch curve, *while the run is still going*. (It measures identity likeness only — overbake and skin texture still need your eyes.)
- **Training Run Visualiser** — scrub the current run epoch by epoch, Royale-style: a slider carousel per sample prompt, play/pause with ping-pong looping, likeness score inline, and share-ready export — a WebM clip with the epoch ticker and Fizgig tag burned in, or full-res PNG frames. It's a taste of the **LoRA Royale** tab, right in the browser.

### Dataset prep
- **Florence-2 AI captioning** — bulk-generate detailed captions in one click.
- **Bilingual captions** — optionally append Chinese via Helsinki-NLP. Klein's Qwen3 text encoder has deep Chinese training, so bilingual captions act as text-level data augmentation, improving visual quality without changing loss. In a controlled A/B (same data, seed, and hyperparameters — captions the only change) the loss curves stayed within ±0.001/epoch, yet the bilingual run produced visibly more skin detail and faster visual convergence.
- **Image Prep** — batch resize, PNG conversion, and InsightFace face-crop derivatives, with optional **gender targeting** (largest male/female face) so it locks onto your subject in group shots. Pairing a tight crop with a full shot adds a lot to a character dataset. Training defaults to ~512² (0.25 MP) and resizes in-cache, so any resolution or aspect ratio just works — nothing has to be square or pre-sized.
- **Look Consistency Filter** — the final prep stage, built for **synthetic-heavy datasets**: the subtly off-look near-misses that drag a likeness down are *easy* for the model to reconstruct, so a loss curve never sees them — but face-embedding distance does. Pick the **3 images that best nail the look** and every image is scored against all three, averaged (close-up faces included — detection pads-and-retries). Worst matches surface first with colour-coded verdicts; mark drifters by click or let **Auto-Suggest** flag the statistical outliers, then move them out of the dataset in one go (to a subfolder — nothing is deleted, and moving them back re-admits them). The scores save with your dataset and drive the trainer's **look-outlier warm-up**.

### Compatibility
Loads kohya, PEFT, OneTrainer (OMI + legacy), AI-Toolkit, and LyCORIS (LoKR / LoHa) — all auto-converted on load. LoKR and LoHa run **natively at inference** — no pre-conversion — anywhere in the app: as a primary or donor in Repair Studio, in the Profiler, in Extract, even as a Context LoRA. **Bake** materialises them to a standard LoRA via GPU-accelerated SVD. Output is kohya-style `.safetensors` that drop straight into ComfyUI Klein nodes. Every tab links to the relevant section of the walkthrough video.

---

## Requirements

- **GPU** — NVIDIA RTX 30 / 40 / 50-series. **16 GB+ VRAM** recommended (24 GB+ comfortable). The fp8 Base's VRAM savings apply on every supported card.
- **NVIDIA driver** — 555+ on Windows, 550+ on Linux (for the CUDA 12.8 PyTorch wheels).
- **OS** — Windows 10 / 11 or Linux. macOS handles captioning and image prep, but training needs CUDA.
- **Python** — 3.10, 3.11, 3.12, or 3.13.
- **Disk** — ~10 GB for the venv, plus ~40 GB for model files.
- **Visual Studio Build Tools** (Windows only) — needed to compile InsightFace, and for the **torch.compile training speedup**. Direct installer (no hunting on the MS site): **[aka.ms/vs/17/release/vs_BuildTools.exe](https://aka.ms/vs/17/release/vs_BuildTools.exe)** — tick the **"Desktop development with C++"** workload. The installer and `update_fizgig.bat` detect it and print this link if it's missing; without it everything still works, you just skip the compile speedup. (triton, compile's other dependency, installs automatically with the requirements.)

---

## Install

Clone the repo (or download the ZIP via the green **Code** button and extract):

```bash
git clone https://github.com/shootthesound/Fizgig.git
cd Fizgig
```

**Windows (one-click)** — double-click `install_fizgig.bat`. It creates a venv, installs CUDA 12.8 PyTorch and all dependencies, pre-downloads the InsightFace models, and verifies CUDA is visible to PyTorch. Launch with `run_fizgig.bat`; update later with `update_fizgig.bat`.

**Linux / macOS:**

```bash
python install_fizgig.py
chmod +x run_fizgig.sh
./run_fizgig.sh
```

Three small models auto-download on first use: InsightFace `buffalo_l` (~300 MB, during install), Florence-2 (~500 MB–1.5 GB, first AI caption), and Helsinki-NLP `opus-mt-en-zh` (~300 MB, first bilingual translation).

---

## Model downloads (you provide)

Fizgig doesn't bundle weights — they're large and licensing varies. Each row in the **Preferences** tab has a **Download** link to the right HuggingFace page. You only need the family you're using.

### Klein 9B

| Model | File | Size | Source |
|---|---|---|---|
| **Base DiT (fp8) — recommended** | `flux-2-klein-base-9b-fp8.safetensors` | ~9.5 GB fp8 | [black-forest-labs/FLUX.2-klein-base-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8) |
| Base DiT (bf16) | `flux-2-klein-base-9b.safetensors` | ~17 GB bf16 | [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) |
| Distilled DiT | `flux-2-klein-9b-fp8.safetensors` | ~9 GB fp8 | [black-forest-labs/FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) |
| VAE / AE | `ae.safetensors` | ~320 MB | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors) (from root, **not** the `vae/` subfolder) |
| Text Encoder | `qwen_3_8b.safetensors` | ~15 GB | [Comfy-Org/vae-text-encorder-for-flux-klein-9b](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b.safetensors) |

Training runs on the **Base DiT**, and the **fp8 version is recommended on every GPU**: same training quality at roughly half the VRAM (resident at ~9.6 GB, so a 9B LoRA trains in ~14 GB and fits a 16 GB card).

The VRAM savings and quality are the same across all supported cards (RTX 30 / 40 / 50-series) — fp8 Base is worth it on every GPU.

It's all automatic — Fizgig detects pre-quantised files and the right path for your GPU, so you never need to touch the "FP8 Base" checkbox (the bf16 version works too if you prefer). The **Distilled DiT** powers the fast 4-step previews — on by default during training, and always used in the Profiler, Repair Studio, and Explorer — so grab both if you'll use the workbench.

### Krea 2

All four files live in the one [**Comfy-Org/Krea-2**](https://huggingface.co/Comfy-Org/Krea-2) repo.

| Model | File | Size | Source |
|---|---|---|---|
| **RAW DiT (bf16) — training** | `krea2_raw_bf16.safetensors` | ~26 GB bf16 | [Comfy-Org/Krea-2 → diffusion_models](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_bf16.safetensors) |
| **Turbo DiT (fp8) — previews / workbench** | `krea2_turbo_fp8_scaled.safetensors` | ~13 GB fp8 | [Comfy-Org/Krea-2 → diffusion_models](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors) |
| Qwen-Image VAE | `qwen_image_vae.safetensors` | ~250 MB | [Comfy-Org/Krea-2 → vae](https://huggingface.co/Comfy-Org/Krea-2/blob/main/vae/qwen_image_vae.safetensors) |
| Text Encoder (bf16) | `qwen3vl_4b_bf16.safetensors` | ~8 GB bf16 | [Comfy-Org/Krea-2 → text_encoders](https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_bf16.safetensors) |

Training runs on the **RAW DiT**; previews and the whole workbench run on the **fp8 Turbo** — grab both if you'll train *and* use the tools. The text encoder must be the **bf16** Qwen3-VL-4B (the fp8 ComfyUI variant can't run the vision path used for reference images, or training). On smaller cards, the **4-bit (NF4)** toggle shrinks the RAW base to ~5.6 GB so it fits 10–12 GB GPUs.

---

## VRAM guidance

### Klein 9B

**Inference tools** (Profiler / Repair Studio / Explorer / Extract) on Distilled 4-step:

| Block Swap | Min VRAM | Notes |
|---|---|---|
| 0 | 24 GB+ | No swap — fastest |
| 4 | 20 GB | Light swap |
| 8 | 16 GB | Moderate swap |
| 12 | 14 GB | Aggressive swap |
| 16 | 12 GB | Maximum swap — slower, but fits |

**Training** — the fp8 Base DiT stays resident at ~9.6 GB (not dequantised to bf16), so a 9B LoRA fits comfortably in **16 GB** — around 14 GB observed at block-swap 0 with a Context LoRA active, a little less without. VRAM scales with resolution and batch size; raise block swap to fit smaller cards.

**Smaller cards — 4-bit (NF4) base.** fp8 training needs ~14 GB: it fits a 16 GB card with no swap, but a **10–12 GB card has to block-swap**, paying a PCIe-transfer penalty every step. The opt-in **4-bit (NF4) base** mode (the *4-bit Base* toggle in Memory & FP8 / FP4) quantizes the frozen base to 4-bit — halving DiT VRAM to ~5.6 GB so a full 9B LoRA trains in **~7.5 GB**, which fits 10–12 GB cards with **no swap at all** (and so beats fp8-with-swap on those cards). The LoRA still trains in bf16 on top, QLoRA-style, and the base loads layer-by-layer so the card never holds the whole model. It's a lower-precision base, so it's a slight quality trade — always check the output in ComfyUI — and **16 GB+ cards should stick with fp8** (same quality, no swap).

**DiT Block Swap (inference)** in Preferences applies only to the workbench tools. Training has its own separate block-swap setting, and its Distilled samples auto-swap by VRAM — so this preference never touches a training run. On first launch Fizgig auto-detects your VRAM and picks a sensible default; once you choose a value, your choice sticks.

### Krea 2

Krea 2 is a bigger model, so the numbers differ — but Fizgig **auto-sizes block swap to your card** for both training and the workbench, so there's nothing to tune:

- **Training** runs on the RAW fp8 base (~14 GB resident); block swap auto-detects from VRAM (32 GB → none, scaling up to maximum on sub-16 GB cards). The **4-bit (NF4)** toggle drops the base to ~5.6 GB (base + LoRA ~8.3 GB), fitting a **10–12 GB card with no swap**.
- **Previews & workbench** (Repair Studio / Explorer / Royale / in-training previews) run on the fp8 Turbo, which peaks ~22.6 GB unswapped — heavier than Klein's Distilled, so Fizgig auto-swaps it to fit your GPU (≈17 GB at swap 12; 16 GB cards swap enough to fit). If a preview still can't fit, it **auto-disables and training keeps running and saving** — evaluate the LoRA in ComfyUI. With 4-bit, the base even parks off the GPU during each preview so the two coexist.

---

## INT8 fast inference (on by default)

Previews and the whole workbench (Repair Studio / Explorer / LoRA Royale, plus in-training previews) run an **INT8 (W8A8)** matmul instead of fp8 — faster, at **near-identical quality**, on **both Klein and Krea 2**. Key points:

- It **only affects previews** — your **saved LoRA is always exact**, INT8 or not. It changes what you *see* while working, never what you *ship*.
- It's a **speed** knob, not a memory one: int8 is 8-bit like fp8, so **same VRAM**. It also **stacks with block swap**, so it helps small cards too.
- The win **varies by GPU**: **biggest on RTX 30-series** (which have no fast fp8 tensor cores), modest on 40/50-series where fp8 is already fast. Measured so far: ~1.19× vs fp8 on a 5090, larger on a 3090.

It's **on by default**; flip **INT8 fast inference** off in **Preferences → Inference Performance** to fall back to fp8.

---

## Getting started

Launch Fizgig and work left-to-right through the numbered tabs:

1. **Start** — set your training image folder. If model paths aren't configured, a prompt points you to Preferences.
2. **Image Prep** (optional) — resize, PNG-convert, or face-crop your images; finish with the **Look Consistency Filter** to weed out off-look images before they train.
3. **Captions** — write trigger-word captions or generate with Florence-2; optionally translate to bilingual English + Chinese.
4. **Samples** — configure the preview prompts that render during training (Distilled 4-step on by default).
5. **Training** — pick a preset, tune, click **Start Training**.

The unnumbered tabs are the post-training workbench — and work on any Klein LoRA you've downloaded: **Profiler**, **Repair Studio**, **LoRA the Explorer**, **LoRA Royale**, **Extract**, and **Preferences** (model paths, output directories, inference block-swap preset, default Browse folders).

---

## Headless / CLI training

Everything the trainer does is also available from the command line — the GUI is a front-end over the scripts in `src/fizgig/scripts/`, so the CLI is always feature-complete: adaptive LR, the full per-image loss watch, auto-recaptioning, Context LoRA, pause/resume. Works on Windows and Linux alike, including display-less boxes. See **[docs/CLI.md](docs/CLI.md)** for the pipeline, dataset config format, and worked examples for both Klein 9B and Krea 2.

---

## Support the project

If Fizgig saves you time or helps you make better LoRAs, consider supporting development:

<a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>

---

## License

Fizgig is open source under the **[Apache License 2.0](LICENSE)** — free to use, modify, and redistribute, including commercially, with attribution and no warranty. It includes third-party components under compatible permissive licenses (musubi-tuner — Apache-2.0; ai-toolkit — MIT; Diffusers / FLUX — Apache-2.0); see **[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)**.

Copyright © 2026 Peter Neill.

Model weights are **not** covered by this license — each model carries its own terms from its publisher (see the Download links in Preferences).
