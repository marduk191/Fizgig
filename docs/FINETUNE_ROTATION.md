# Rotating-Block Full Fine-Tuning (Krea 2)

Branch: `experiment/ft-rotation` — not merged, not pushed. Master is untouched.

Trains the **Krea 2 base model itself** instead of a LoRA, on a single consumer GPU, by keeping
most of the model fp8-frozen and rotating which slice is trainable. First real runs are done and
the results are strong enough to be worth pursuing properly.

---

## Why

A naive full fine-tune of Krea 2 (12.9B) needs roughly **78 GB** — bf16 weights, gradients and
Adam state all at once. That's rented-A100 territory.

The bet is that removing LoRA's rank bottleneck matters. A LoRA constrains every update to a
low-rank subspace, so concepts compete for the same handful of directions and the result behaves
more like a filter applied over the model's outputs — which is why LoRAs tend to drag pose,
framing and lighting toward the training distribution along with the likeness. Full-rank updates
can change the model's internal representation of a concept, so it composes with what the model
already knows.

---

## How it works

**Rotating windows.** Only part of the model is trainable at a time; the window advances each
epoch. Gradients and optimizer state only ever exist for the active slice, which is what makes it
fit. Over a full cycle every weight is trained. This is LISA (Layerwise Importance Sampled AdamW)
applied to a diffusion DiT rather than an LLM — as far as we know, untried in this setting.

**The bf16 master copy is the critical design decision.** A CPU-resident bf16 copy of every
trainable weight is the source of truth. Blocks activate *from* it and write back *to* it, so the
lossy fp8 copy on the GPU is only ever used for the frozen forward pass. Training never
round-trips through fp8 — which would quantise away exactly the small updates we're trying to
learn. Costs ~24 GB of system RAM.

**Two window modes:**

- **component** (default) — attention across all 28 blocks, then each MLP matrix in turn. Four
  windows per cycle. Every window spans the model's full depth, so a concept is learned by every
  layer at once rather than by one depth slice at a time.
- **block** — contiguous slices of blocks. Fewer windows, but each trains only part of the depth.

Component sizes are balanced by parameter count: a block is 30% attention / 70% MLP, so a naive
`("attn", "mlp")` split makes the MLP window 2.3× the attention one — measured, and it OOM'd.
Splitting the MLP into its three matrices gives four windows of ≤30% each.

**txtfusion is always-on.** It sits outside `dit.blocks`, so rotation would never have reached
it — yet it's the stack that fuses the text embeddings, where prompt-to-concept binding happens.
It's small (3072-wide text path vs 6144-wide main blocks) so it stays trainable throughout.

**Optimizer-in-backward.** Each trainable tensor gets a single-parameter optimizer stepped from a
`post_accumulate_grad_hook`, so a gradient is consumed and freed the moment it lands and the whole
window's gradients never coexist. Worth 5.2 GB. Costs global gradient clipping (impossible when
grads are freed on arrival) and gradient accumulation (nothing left to accumulate).

**Adafactor** rather than AdamW: its factored state is ~10× smaller, which is part of what keeps
this inside 32 GB.

---

## Measured (RTX 5090, 32 GB, 36-image dataset at 0.25 MP)

> **These are the fp8-era baselines.** The shipped fine-tune default is now a **4-bit NF4**
> frozen base, which changes the picture: full-depth component windows peak ~16 GB at a
> 24 GB budget (~21-23 GB uncapped on 32 GB) at the same ~1.0 s/it, and streaming reaches
> 16 GB cards. Current numbers: README fine-tuning section + EVAL_full_model_h3_video.md.

| Config | Peak VRAM | s/it | Windows/cycle | Time per full cycle |
|---|---|---|---|---|
| 4 blocks resident | 24.8 GB | 0.93 | 7 | 234 s |
| 8 blocks resident | 24.2 GB | 1.01 | 4 | 145 s |
| 14 blocks resident | 27.5 GB | 1.16 | 2 | 84 s |
| 18 blocks resident | 29.5 GB | — | 2 | — |
| **component (default)** | **30.1 GB** | **1.03** | **4** | **148 s** |
| 12 blocks + streaming | 26.5 GB | 2.99 | 3 | 323 s |
| 18 blocks + streaming | 29.5 GB | 2.64 | 2 | 190 s |

**Fine-tuning is faster per step than LoRA training.** Counterintuitive, but: the trainable window
holds real bf16 weights, so it skips the fp8 dequantisation that materialises a bf16 copy of every
weight matrix on every forward; there are no LoRA down/up matmuls (528 extra small, latency-bound
kernels per forward plus their backward); and the optimizer steps ~65 large tensors instead of
~528 small ones. Reasoned from the code, not profiled.

---

## Block streaming — built, and not worth it at 32 GB

`RotationOffloader` pins the trainable window on GPU and streams every other block from CPU
just-in-time, with async prefetch on a dedicated CUDA stream. The stock offloader can't express
this: it keeps a fixed contiguous *prefix* resident and pairs each eviction with a matching load,
whereas rotation needs an arbitrary resident set that *moves*.

It works, and it was built on a wrong assumption. I expected the fp8 base to be the binding
constraint; optimizer-in-backward freed enough that it never was. Simply raising the resident
block count wins outright — 14 resident covers the model in 2 windows at 84 s/cycle, versus
323 s/cycle for 12 blocks with streaming.

**It stays as opt-in** (`--blocks_to_swap`), because it's the right tool for 24 GB and 16 GB
cards where residency *is* the constraint. Just not the default here.

Gradient checkpointing recomputes a block's forward *inside* its backward, so blocks are pulled
back on a backward **pre**-hook — evicting after the forward alone leaves the recompute pointing
at CPU weights.

---

## Results so far

First overnight run (single subject, real name as the identity token, detailed captions, no
regularisation images):

- Quality reported as **very good**, and notably **better generalisation than the equivalent
  LoRA** — which is the predicted symptom of removing the rank bottleneck rather than a lucky run.
- **Limited class drift** despite no regularisation set, presumably because a distinctive name
  absorbed the identity, detailed captions gave everything else somewhere to live, and 40 epochs
  at 1e-5 is gentle.

Not yet established: which checkpoint peaked, whether 1024 generation shows detail softening from
training at 0.25 MP, and a matched A/B against a LoRA.

---

## MiniMax H3 — the recommended recipe (25 Aug, settled by matched A/Bs)

H3 rotation FT is fully ported (`H3NF4Rotator` in `src/fizgig/minimax/rotation_ft.py`) and
the recipe below is not a guess — every element was chosen by a matched real-run comparison
on valid checkpoints:

**Component windows + Optimised Likeness Learning ON + LR 1e-4.** One checkbox each:

- **Component windows** (the only mode since 24 Aug — the old 4/6/8-block windows are
  removed; they never matched component's likeness speed): each window is one matmul
  (qkv / out / fc1 / fc2) across every block, so a concept trains at full model depth every
  epoch — 4 windows per cycle, whatever the block span. The frozen trunk runs as NF4
  during training; the saved checkpoint is still exact int8 and deploys in ComfyUI like
  the base model.
- **Optimised Likeness Learning ON** (the default) — per-modality routing: photos feed the
  identity blocks (20-49), voice feeds the audio zone (34-49), clips train the full model.
  The cycle tightens automatically to the union of what the dataset actually trains
  (photos-only → 20-49; audio-only → 34-49; photos+voice → 20-49 with voice confined to
  its zone per step; add clips and the cycle spans the full model with photos and voice
  each still confined). Matched 64-epoch A/B against full-model training: **vastly better
  looking and vastly better prompt adherence** — full-model photo gradients bend the 0-19
  trunk (composition, motion, prompt binding) toward photo reconstruction. Bonus on
  photo-led datasets: 30-block cycles (faster epochs, ~23 GB master, peaks that brush a
  24 GB card). Untick only for style/scene fine-tunes, where the trunk IS the target —
  voice still routes to its zone either way. An explicit Blocks range wins over all
  routing. (An earlier "better without it" verdict was an artifact of the likeness-save
  bug below — it did not survive the fix.)
- **Voice → 34-49, always** (24 Aug A/B): an audio-only fine-tune at 34-49 learns the
  voice cleanly; the same audio at 20-49 measurably **corrupted the visual blocks** —
  audio gradients do real damage outside the audio zone (core 38-48, shoulder 34-37; see
  RESEARCH_h3_block_map.md). This is why the routing is unconditional.
- **Per-modality run length** (`Finish one category early`, now live under FT): a mixed
  dataset's smaller category can finish (or start to overbake) well before the larger one
  — stop photos & clips at epoch N and let the voice keep refining its own blocks, or the
  reverse. Under FT the retired category stops outright (no anchor mode) and the stop
  lands on a rotation-cycle boundary — epochs snap UP to the next multiple of the cycle,
  so every window sees the identical data mix for equal passes before the mix changes.
- **Stochastic-rounding saves** (automatic, not a setting): trained tensors re-encode to
  int8 with stochastic rounding at save. Nearest rounding is biased back to the base's
  codes, so an FT's 0.2-1% deltas — below the int8 grid step — used to round HOME: previews
  showed full likeness while every saved checkpoint deployed without it. Measured
  direction-cosine of the saved delta: 0.02 nearest → 0.36 stochastic at a 0.3% delta,
  ~10× the weight-space SNR of the NF4 trunk previews render through. Field-validated in
  ComfyUI the same day.
- **Memory**: the bf16 master auto-selects RAM or disk (`--finetune_master`, auto = disk
  when the master would eat >40% of available RAM). Disk mode measured a **3.8 GB** trainer
  working set vs 90.7 GB for the RAM path — full-model FT fits 64 GB boxes easily.
- Previews follow checkpoint saves (24 Aug): one preview per saved checkpoint, plus the
  final one — every sample in the gallery maps to a file you can deploy. Saves are full
  ~21 GB int8 checkpoints, cadence snapped to the cycle (so previews keep the
  equal-training honesty); rendered via a deactivate/reactivate bracket; continuation is
  `--dit <checkpoint> --finetune_start_window N` (printed at every save).
- **Pause / Resume** (GUI, both families): Pause saves a full checkpoint at the epoch
  boundary — even between save-cadence epochs — and exits to free the GPU. Resume
  relaunches from that checkpoint: the rotation cycle picks up at its stamped window,
  the run trains only the epochs that remain of the original total, and checkpoint
  numbering continues where it left off (nothing gets overwritten). The same works by
  hand with the printed `--dit <checkpoint> --finetune_start_window N` command.
  Category-retirement epochs count on the same calendar: pause a mixed run, set the
  stop epoch to the current epoch (or lower), and Resume to finish it voice-only or
  visual-only.

---

## Usage

**GUI** — Training tab, Krea 2 only: tick **⚗ Fine-tune the BASE MODEL instead of training a
LoRA**. That applies the whole recipe in one go and prints what it changed:

| Setting | Value | Why |
|---|---|---|
| Learning rate | 1e-5 | LoRA rates (1e-4+) destroy a base model |
| Max train epochs | 40 | 10 full cycles |
| Save every N epochs | 4 | one per cycle, so checkpoints compare like-for-like |
| Window mode | component | every window spans full depth |
| Free gradients | on | ~5 GB |
| Adaptive LR | off | rotation boundaries read as instability |
| LR scheduler | constant | |
| Grad accumulation / clip | 1 / 0 | required by fused backward |

Checkpoints go to the **Output Directory** — each is a full ~26 GB model, so point it somewhere
with room (a ComfyUI `models/unet` folder is the natural home). Test them as ordinary Krea 2
models.

**CLI:**

```bash
python src/fizgig/scripts/krea2_train.py \
  --dataset_config my_dataset.toml \
  --dit /models/Krea-2-raw.safetensors \
  --output_dir /big-drive/ft_run --output_name my_subject \
  --learning_rate 1e-5 --max_train_epochs 40 --save_every_n_epochs 4 \
  --finetune_rotation 14 --finetune_rotation_mode component \
  --finetune_fused_backward
```

`--finetune_rotation` is the master switch (any value > 0). In component mode the number is
ignored; in block mode it's the blocks per window.

---

## What is and isn't compatible

**Works:** the per-image loss watch and per-image adaptive LR (loss scaling happens before
backward, upstream of where fused backward consumes gradients), auto-recaption, look-outlier
warm-up. Verified: 36/36 images tracked with real verdicts under a component-mode run.

**Auto-disabled, with a log line:** adaptive LR, block swap unless explicitly asked for, 4-bit.
In-training previews RUN under the fine-tune (27 Aug, both families): the save cadence snaps to
rotation-cycle boundaries and previews follow the checkpoint saves, rendered on the training DiT
via a deactivate/reactivate bracket with the Turbo LoRA applied fresh each time — the same
pattern H3 field-proved. Krea 2 needs the Turbo LoRA configured (the standalone Turbo checkpoint
is a different model and cannot show fine-tuned weights); without it previews stay off with a
log line saying why.

**Caveat on the loss watch.** It was designed against a model of constant capacity. Under rotation
the trainable set changes every epoch, so loss carries a 4-epoch periodic component — an
attention epoch and an MLP epoch don't fit equally well (observed: 0.0796 → 0.0775 → 0.0948 →
0.0760, where the spike is `mlp.up`, not the data). Residuals are computed relative to
per-timestep-bucket means so a uniform shift largely cancels, which is why verdicts look sane —
but trust a `stuck` flag that persists across a full cycle over one that appears and vanishes
within four epochs. This is exactly why adaptive LR is disabled: it reads epoch-to-epoch loss
directly and would have called that spike instability and rolled weights back.

---

## Continuing a run

**Do not use Pause/Resume.** The state directory saves the LoRA network and optimizer, but in
fine-tune mode the LoRA is inert and all the training lives in the base weights, which aren't in
there. Resuming would silently restart from the original RAW model.

**Instead, swap the base.** The output is a complete, structurally identical Krea 2 checkpoint
(all 430 keys, same names, bf16, untrained tensors byte-identical to RAW), so point Preferences →
Krea 2 RAW DiT at a checkpoint and train again. Latent and text caches stay valid — they depend on
the VAE and text encoder, not the DiT. Use a new output name, and drop the LR (5e-6) for a
refinement pass.

*Known gap: `--resume` should refuse to run in fine-tune mode rather than silently doing the wrong
thing.*

---

## Captioning for a base-model fine-tune

Different from LoRA practice, and it matters more:

- **Use a distinctive real name, not a rare token.** The method can genuinely rebind a name, and
  a name carries useful semantic scaffolding. Trigger goes **first** now, in both the Captions tab
  and auto-recaption (they disagreed before this branch).
- **Never train into a generic class word.** Using `woman` as the trigger would overwrite the
  model's concept of "woman" globally, with no strength dial to back it off.
- **Caption what varies, omit what's constant about the subject.** Clothing, background, lighting,
  pose, viewpoint, framing — yes. Facial structure and eye colour — no: caption them and they bind
  to those words rather than to the name.
- **Regularisation images are the proper fix for class drift** (~50% of dataset size, generated by
  the base model itself so you anchor its own prior rather than teaching new content, captioned
  with the class only). Not yet supported in the GUI; possible today by hand-editing a second
  `[[datasets]]` block into the TOML with its own `cache_directory`.

---

## What SVD extraction and the shortcut experiments showed (27 Jul)

The delta from a 60-epoch component-rotation run on three subjects (two of them women, i.e. one
within-class pair) was extracted to LoRA at several ranks with the `comfyui-model-diff-to-lora`
node, and the resulting subspaces measured against each other. Four results, all reproducible
from checkpoints on disk with no training.

**1. SVD extraction works, and works at low rank.** Ranks 8 / 16 / 32 / 64 / 128 all preserve
*separation* -- the three identities never blend. What degrades below 64 is *likeness* precision,
smoothly. Rank 64 is perceptually perfect and about half a gigabyte; rank 32 is very close.

Identity **placement** is a coarse, high-energy move that lands in the top few singular directions
and survives brutal truncation. Likeness **precision** is fine detail spread through the tail.
That asymmetry is why separation is the robust part.

**2. The delta is not low-rank, and energy is a poor proxy for quality.** The perceptually perfect
rank-64 extraction captures only **~28%** of the delta's Frobenius energy (confirmed two ways:
from the shipped LoRA's own up-matrices, and from a fresh decomposition -- 27.94% vs 27.78%).
Nearly three-quarters of the update's magnitude is discardable without visible loss. The update is
a small amount of highly structured learning inside a large amount of diffuse drift. Do not use
energy capture to judge an extraction.

**3. The good subspace is built late, and cannot be bought early.** Measuring each epoch's top-64
basis against the known-good rank-64 basis (mean squared cosine of principal angles, plus a count
of good directions individually captured at cos^2 > 0.5):

| basis from | overlap | good directions captured |
|---|---|---|
| random | 1.8% | -- |
| epoch 4 | 20.9% | 2.7 of 64 |
| epoch 12 | 31.7% | 6.6 |
| epoch 20 | 41.9% | 15.3 |
| epoch 32 | 57.6% | 47.2 |
| epoch 40 | 67.2% | 55.2 |

Still climbing at epoch 40, with a sharp consolidation between 20 and 32 -- consistent with
three-character separation only appearing around epoch 60. All three component groups (attn / mlp
/ txtfusion) track within a point of each other, so no per-component strategy is needed.

**Width does not substitute for time.** Widening the early basis saturates: excess coverage over a
same-width random subspace goes 19.1 -> 24.6 -> 27.9 -> 28.0 points at ranks 64 -> 128 -> 256 ->
512. Everything past rank 256 is luck. A rank-512 basis at epoch 4 captures 24.8 of the good 64; a
rank-**64** basis at epoch 40 captures 55.2. Eight times the width, taken early, gets less than
half as much as simply waiting.

**4. A high-LR "fast-forward" probe makes it worse, not better.** Three 4-epoch probes from the
same base on the same dataset, identical but for LR:

| probe | delta norm | good directions captured |
|---|---|---|
| 1x (5e-5) | 2.38 | 1.8 |
| 5x (2.5e-4) | 13.50 | 1.5 |
| 20x (1e-3) | 37.19 | 0.3 |
| *epoch 40 of the real run* | *7.27* | *58.1* |

The 5x probe travelled nearly **twice as far** as epoch 40 and the 20x probe **five times** as far,
yet captured essentially none of the right directions. Distance travelled is not what produces
them. High LR does not advance along the trajectory -- it goes somewhere else.

(Control: a 1x reproduction of the original epoch 4 travelled 2.38 against the original's 2.41,
confirming the rig. Their direction sets differed slightly -- 1.8 vs 2.7 hits -- so even two runs
at identical LR disagree about which directions form early. Early structure is not a stable
target.)

**5. Per-layer rank allocation is not worth doing.** A standing suggestion was to truncate by
*energy fraction* rather than fixed rank -- give layers carrying facial detail more directions and
coarse-placement layers fewer, at the same file size. Measured on the epoch-60 delta (145 matrices,
every allocation held to the exact parameter budget of flat rank 64), residual energy
`sum||d - d_hat||^2 / sum||d||^2`:

| allocation | residual | vs flat |
|---|---|---|
| flat r64 | 73.37% | -- |
| energy fraction (tau = 24.6%) | 75.24% | **-1.86** |
| relative movement (||d||/||W||) | 74.34% | -0.96 |
| water-filling | 72.22% | **+1.15** |

Both heuristics are *worse* than flat. Water-filling -- greedily taking the best marginal
energy-per-parameter, which is provably optimal here because cumulative energy is concave in rank
-- gains only **1.15 points**. That is the ceiling, not a result: no allocation rule can beat it.

Flat rank is near-optimal because there is nothing to exploit. Energy captured at rank 64 per layer
runs min 16.2% / median 26.4% / max 70.0%, **sd 9.0pp** -- every layer has the same heavy-tailed
shape. The diffuse drift from finding 2 is not concentrated in particular layers; it is uniform
across all of them.

Two smaller notes. The two heuristics **anti-correlate (-0.52)** -- they disagree about which layers
deserve rank and both lose to flat, so neither tracks anything real. And the energy rule **starves
block 0** (r3-r7 against flat 64, its spectrum decaying fast enough to hit the threshold at once)
while spending r142 on mid-block `mlp.down`; as the composition anchor, block 0 is likely the worst
place to take rank from, which the energy metric cannot see.

Finally, the 1.15-point ceiling is measured on an axis already known not to predict quality: the
perceptually perfect extraction sits at 28% energy, and the span between perfect and nothing is ~73
points. Flat rank stays.

(Per-layer rank is still *supported* -- `modules_dim`/`modules_alpha` are read per module, and
Repair Studio's donor blending emits such files. This finding is about allocating a fixed budget,
not about the format.)

---

### What this means

Three independent attempts to shortcut the full-rank trajectory failed: training directly at rank
128 (mush), freezing a wide early basis (saturates far short), and a high-LR probe (actively
worse). Taken together with a rank-128 *trained* LoRA failing where a rank-64 *extracted* one is
perfect, the claim sharpens considerably:

> The limitation of low-rank adaptation here is **optimisation, not expressivity** -- 64 directions
> suffice to *hold* a solution that 128 trainable directions cannot *find*. And the full-rank
> trajectory that produces those directions appears **irreducible**: they are constructed by slow
> cumulative optimisation, not discovered by capacity, width, or step size.

So fine-tune-then-extract is the correct architecture, not a stepping stone -- the "wasteful"
full-rank phase is the mechanism, and the extraction is lossless enough at rank 64 to be free.

**Direct-to-LoRA (DTL) is closed.** The idea was to keep the update in factored form as it is
produced (a frozen basis with linearly-trained coefficients, GaLore-style, emitting an adapter
natively) to reach 16 GB cards. Every variant needs a usable basis before the run, and results 3
and 4 show no way to obtain one: any basis good enough only exists after ~40 epochs of the very
fine-tune it was meant to replace. Recorded here so it is not re-derived.

---

## Next

1. ~~**SVD the delta into a LoRA.**~~ **Answered** -- see above. It survives, and at rank 64 it is
   perceptually perfect. Extraction should become a first-class step: a GUI button that takes the
   run's final checkpoint and emits the LoRA directly, rather than sending users to a ComfyUI node.
2. ~~**Regularisation-images support** in the GUI.~~ **Done** — optional folder + fixed LR
   multiplier beside the fine-tune toggles, written as a second `is_reg` dataset block so the
   cache scripts pick it up unchanged. Fine-tune only; a LoRA update is rank-bounded and cannot
   drift the general representation the way this can.
3. ~~**VRAM sweep — what actually fits a 4090 and a 16 GB card.**~~ **Answered — see below.**

<details><summary>original plan</summary>

**VRAM sweep — what actually fits a 4090 and a 16 GB card.** Nothing measured fits
   either: the lowest config on record is 24.2 GB (8 blocks resident) against ~22.4 GB usable on a
   4090 and ~14.5 GB on a 16 GB card. Run each candidate ~30 steps and read
   `torch.cuda.max_memory_allocated()` — no full run, no checkpoints. Candidates: block mode at 4
   and 8 resident with streaming on (the 24 GB targets), 2 and 4 resident with streaming to find
   the floor, and component mode with streaming since that is the quality-preferred window shape.

   Worth doing rather than reasoning about: the analytic model (fp8 base + bf16 window + factored
   optimizer state + checkpointed activations) lands ~20 GB for component mode, which measures
   30.1 — a 10 GB gap that says something substantial is unmodelled. The measured table is also
   **non-monotonic** (4 blocks resident = 24.8 GB, 8 blocks = 24.2 GB), which is backwards for
   weights and suggests activations or fragmentation dominate. Extrapolating from either would be
   a guess.

   Structural note that does hold: **16 GB is unreachable without streaming.** The frozen base
   alone is ~11.3 GB of fp8 blocks plus ~0.6 GB of bf16 norms/embeddings/IO before any trainable
   weight or activation exists. (Still true as stated, but read it as a statement about **fp8**:
   an NF4 trunk is 6.08 GB, and 16 GB was reached that way on 27 Aug 2026 — see below.)

</details>

   **Measured (5090, 130 steps — past a rotation boundary, which matters enormously: component's
   peak IS the window switch and a 30-step probe reads 20.44 GB instead of 27.67).** Peak reserved,
   excluding the ~0.5-1 GB CUDA context:

   | config | peak | load | training | 4090 (~22.4) | 16 GB (~14.5) |
   |---|---|---|---|---|---|
   | component | **27.67 GB** | 19.54 | 27.67 | no | no |
   | block 8 + streaming | 20.71 GB | 18.33 | 20.71 | **yes** | no |
   | block 4 + streaming | 18.70 GB | -- | -- | **yes** | no |
   | block 2 + streaming | **17.62 GB** | -- | -- | **yes** | no |

   - **24 GB cards: yes, comfortably.** Block mode with streaming has room to spare.
   - **16 GB: no, in any configuration.** The floor is 17.62 GB with the most aggressive window
     and streaming already on. That leaves the txtfusion-only mode below as the only route.
     **SUPERSEDED (27 Aug 2026) — 16 GB is reached, and the escape was the assumption rather
     than the arithmetic.** Every figure in this table is an **fp8** base, which was the only
     frozen trunk Krea 2 FT allowed (4-bit was refused outright). An **NF4** trunk is 6.08 GB
     packed against fp8's ~13, which moves the whole table down: component mode with streaming
     now measures **8.4–11.0 GB peak** across 12 depth-split windows under a hard 15.9 GB cap,
     ~2.8 s/it, with pause/resume verified across a rotation. The "structural note that does
     hold" above holds only for its own precision — it reasoned from an 11.3 GB fp8 base as if
     the base size were fixed. See docs/DESIGN_component_ft_24gb.md for the current tiers.
   - **The peak is TRAINING, not load**, in both modes (load is 18-19.5 GB either way). So
     changing how the model is brought in -- e.g. sourcing frozen blocks from a pre-quantised
     fp8 file rather than quantising RAW -- cannot help. The lever is the trainable window and
     activations.
   - **Streaming cannot help component mode at all.** Every block holds a trainable slice, so
     there is nothing to stream out; the trainer disables it and says so. Component's floor is
     fixed at its no-stream figure.
   - Peaks are monotonic in resident blocks here (17.62 -> 18.70 -> 20.71), unlike the older
     table. The earlier non-monotonic reading was probably fragmentation noise, since these were
     taken with `expandable_segments` on.

   **Shipped as an Auto window mode** (`--finetune_rotation_mode auto`, now the default; "Auto (by
   VRAM)" in the GUI). Resolved in the trainer at launch from FREE VRAM, so headless runs get it
   and another app holding the card is accounted for. Tiers add ~2 GB over the measured peak for
   the CUDA context and headroom: >=29.5 component, >=22.5 block-8, >=20.5 block-4, >=19.5
   block-2, below that a plain warning that it will not fit. The console states the pick, the
   measured peak, and -- when it drops to block mode -- that block mode is **not yet
   quality-tested**, since every good result so far came from component runs.

4. **txtfusion-only window mode** (`--finetune_rotation_mode txtfusion`) — freeze all 28 blocks,
   train only the text-fusion stack, to bind a *name* to a concept the model can already render
   (the motivating case: a named pose, e.g. a YMCA letter shape, letting the frozen model assemble
   the body from its own knowledge of body positions).

   **Cheap, and possibly the small-card answer.** txtfusion is **343 M params — 2.7 % of the
   model** (two 172 M stacks: `layerwise_blocks` + `refiner_blocks`). Frozen blocks stay fp8
   (~11.3 GB), trainable weights are 0.64 GB bf16, Adafactor state is negligible. With no large
   bf16 rotation window, this is by far the lightest mode and is the most likely to fit 16 GB —
   pairs naturally with the sweep above.

   **Half-wired already:** txtfusion is *always* trainable during rotation
   (`rotator.activate_always`) because it sits outside `dit.blocks`. What is missing is a mode that
   trains only it — `RotationSchedule` clamps to at least one block window.

   **The idea is a higher-capacity textual inversion:** TI optimises one embedding vector, this
   optimises the whole text→conditioning transform. The text encoder stays frozen, so a rare token
   keeps a fixed embedding and txtfusion learns the mapping — cleaner than TI, which fights both at
   once.

   **Risks, in order:**
   - **Damage is global and permanent.** txtfusion processes *every* prompt. A block LoRA can be
     switched off and a token embedding only fires on its token; this cannot. Overfitting degrades
     text understanding for everything. Regularisation images move from optional to near-mandatory.
   - **343 M params against ~20 images is a lot of capacity** — expect to want a lower LR and fewer
     epochs than a full fine-tune, or it encodes image detail rather than a clean concept.
   - **Unknown whether pose is expressible there at all.** Pose is spatial; if assembling the shape
     needs computation in the main blocks, conditioning alone will not reach it. Krea 2's block
     roles are unmapped, so this *is* the experiment.

   Caption everything except the concept in detail, so the token has nothing else to absorb.

5. **ReLoRA** -- train a rank-16 LoRA, merge it into the bf16 master, reset, repeat, accumulating
   effective rank at LoRA memory cost. Still the only untested route to smaller cards, but the
   evidence above lowers its odds: each cycle's *search* stays rank-constrained, and the good
   directions took ~40 epochs of unconstrained trajectory to form. If tried, expect it to need a
   long full-rank warmup -- which is the memory cost it was meant to avoid.
6. **Make the loss watch rotation-aware** — compare residuals against the same window position
   rather than the previous epoch. Only if the cyclic noise proves to matter.
7. **Guard `--resume`** in fine-tune mode.
8. **Two-machine pipeline split** — see [`DISTRIBUTED_FINETUNE.md`](DISTRIBUTED_FINETUNE.md).
   Costed, nothing built. Splitting the block list across two machines is the only remaining
   attack on **component** mode, which streaming provably cannot help. A 12/16 split models at
   12.68 GB / ~18 GB, i.e. component mode on a **24 GB + 16 GB pair** — the mode every good
   result came from, on hardware that today gets block mode at best. The wire is not the
   obstacle: one split ships 38 MB/step at 512 px, which plain gigabit already carries.

---

## Commits

| | |
|---|---|
| `42b0825` | rotation mechanics (schedule, block rotator, bf16 master) |
| `44f56a7` | fp8 fix: scale_weight created on-device in a compute dtype |
| `4dec740` | wire rotation into the trainer |
| `2a0225d` | txtfusion always-on |
| `0b00583` | optimizer-in-backward (−5.2 GB) |
| `04dc0ec` | rotation-aware block swap |
| `7827fa3` | async prefetch (4.55 → 2.99 s/it) |
| `5d8430d` | component (sub-block) rotation |
| `740184b` `03fa82c` `a16f4b9` `74a6f92` | GUI controls, one-click recipe, defaults |
| `9d0c0ad` | trigger word leads the caption |
| `ab3cca2` | fix: fine-tune never engaged (read Tk vars, not settings) |
| `61f6cda` | fix: resolve fine-tune mode before preview setup |

Two bugs shared a root cause worth remembering: **setup running before the decision that
invalidates it**, and **reading a flag from a dict that is only populated elsewhere**. Both were
invisible in isolation and only surfaced when the feature was exercised the way a user actually
reaches it — a headless test that set `self.settings` by hand passed while the real path was
broken.
