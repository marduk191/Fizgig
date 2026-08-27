# Full-model H3 fine-tuning (the video path) — evaluation, 27 Aug 2026

**Read before building anything. No routing or planner code changed on the strength of
this note.** It answers the five questions parked in the small-card gate, in leverage
order, and separates what is measured from what is still reasoning.

## Why this is not a niche path

`plan_ft_modality_routing` (`src/fizgig/minimax/trainer.py:395`) does:

```python
if n_clip:
    union |= full
```

There is no `clip_blocks` parameter — `photo_blocks` and `audio_blocks` both exist, clips
have no equivalent. So **one video clip anywhere in the dataset makes the cycle span all
50 blocks**, whatever likeness is set to. Video is H3's headline feature, so the heavy
path is the path most H3 fine-tunes will take, and "full-model is the fallback" understates
it.

## What is now measured

All under real allocator caps (`FIZGIG_SIM_VRAM_GB`, which caps the allocator as well as
the planner), all on **496 px stills**:

| config | windows | peaks | speed |
|---|---|---|---|
| likeness (20-49), 24 GB | 5 | 19.6 / 21.5 / 19.1 / 21.4 / 21.5 GB | ~1.3–1.5 it/s |
| likeness (20-49), 16 GB | 9 | 8.8 – 12.3 GB | ~1.0–1.1 it/s |
| **full model, 24 GB** | 8 | **18.7 / 18.7 / 17.3 / 18.4 GB** | ~1.55 it/s |
| **full model, 16 GB** | 16 | **11.6 GB** (13 resident / 37 streamed) | ~1.1 it/s |

The overhead constant (`_FT_OVERHEAD_GB = 14.5`) predicted full-model epoch 1 at 18.85
against 18.7 measured — accurate to 0.1 GB **for stills**.

## Q3 — boundary retention: RESOLVED, and it was the whole problem

This was listed as "becomes load-bearing at full-model scale". It is now fixed at the
cause. Every rotation was carrying **two** windows: rebinding `lin.weight` never freed the
outgoing bf16, because a C++-side autograd referrer keeps its storage alive and it is no
longer a module parameter for gc or the park walk to find. Fix (`a23a325`) is
`park_dit_partial`'s own trick — take the storage, drop the tensor, `resize_(0)` — in both
rotators' `_deactivate_targets`, guarded so it never frees a storage the re-encoded weight
still shares.

Consequences worth carrying forward:

* Full-model H3 at 24 GB went from **OOM at the epoch-2 boundary** to running, with
  epoch 2 identical to epoch 1 for an identically sized window (18.7 / 18.7) — which is
  what it should always have been.
* The `(usable − overhead) / 2` budgeting that was on the table is **not needed**, and was
  deliberately not applied: it would have turned the likeness plan from 5 windows into 9,
  ~1.8× the compute, to work around a bug rather than fix it.
* **The 29.2 GiB peak measured on the 5090 was pre-fix.** Any full-model number quoted
  from before 27 Aug is inflated by roughly one window. Don't plan against it.

## Q1 — do clips actually need all 50 blocks? (highest leverage, and it needs NO code)

The `if n_clip: union |= full` line is a placeholder, not a finding — the docstring says
"clips -> full model for now" and it was never measured. Compare the voice zone (34-49),
which came from a real A/B and cut that path's cost enormously.

**The experiment is already runnable today.** `restrict_patterns_to_blocks` exists for
exactly this, and its docstring says so: H3 is 50 identical blocks with no published map,
so "training a subset is an experiment, not a recipe — this exists to make that experiment
cheap to run". An explicit `--finetune_blocks` subset is returned verbatim and wins over
all routing, which is precisely how the 34-49 voice rule was found.

So the shape is: same clip dataset, matched epochs, `--finetune_blocks` at a few candidate
ranges vs full 50, judged by eye. If a clip zone exists it shrinks the cycle, the CPU
master, and the VRAM simultaneously, and every question below gets smaller. **Nothing to
build; it needs Peter's video material and GPU time, not engineering.**

## Q2 — the VRAM model is calibrated on stills and will not transfer (the gating unknown)

Every constant above came from 496 px still runs. A 22-frame clip is on the order of 30×
the tokens, and **activations, not window bf16, will dominate** — which means the
overhead term stops being a constant and becomes resolution- and frame-count dependent.
The FT planner has no activation term at all; the LoRA-side planner does (`_ACT_GB_CKPT ×
mp/0.25`), and the FT side would need an analogue.

**This is unmeasured and I did not measure it.** No clip dataset is cached on this box
(`h3_mixed.toml` is stills + voice), so it needs Peter's source video and a cache pass —
not a cheap measurement, and not one worth faking with a synthetic clip.

**Consequence for the table above: the full-model rows are stills numbers.** They say the
50-block cycle fits 24 GB and 16 GB *for stills*; they do **not** license a "video fine-tunes
on 16 GB" claim, and the README should not make one until a clip run is measured. Take
per-window peaks on a real clip dataset before any clip tier is stated publicly.

## Q4 — the freed-NF4-share term is unmodelled (cheap, worth doing with Q2)

The design note's formula subtracts the active component's freed NF4 share;
`plan_component_windows` does not. Fat windows (fc1) are therefore over-predicted by ~3 GB
and over-split. It is a few lines, it costs nothing at runtime, and it matters most exactly
where full-model hurts. **Recommend bundling it with the Q2 activation term** — both are
edits to the same budgeting expression, and doing them together means one recalibration
pass instead of two.

## Q5 — cycle-length ergonomics

More splits = longer cycle = a higher *minimum useful run*, because every weight must train
at least once. Full-model at 24 GB is 8 windows; at 16 GB it is 16. At rotate-every 1 that
is a 16-epoch floor before the model is even once evenly trained.

This is real but mild: video FT needs roughly 30× the steps of a LoRA run anyway (Peter's
own figure), so a 16-epoch cycle sits comfortably inside a real run. The one concrete gap
is that **the GUI still estimates the cycle from a 32 GB 4-window baseline**, which will
under-report badly on a small card. The trainer's own snap is authoritative and prints the
real plan; the GUI estimate should either follow it or say it is a 32 GB figure.

## Recommendation

1. **Run the Q1 block-zone experiment first.** Zero build cost, and a positive result
   shrinks every other problem here. It is the only item where an afternoon of GPU time
   could remove the need for engineering entirely.
2. **Then measure Q2 on a real clip dataset** before claiming any video VRAM tier. Until
   that exists, treat the full-model rows above as stills-only and say so.
3. **Then do Q4 + the Q2 activation term together**, as one budgeting change with one
   recalibration.
4. **Q3 is closed.** Q5 needs only the GUI estimate corrected, which is cosmetic.

Nothing above is blocked on code. The gating input is Peter's video material.
