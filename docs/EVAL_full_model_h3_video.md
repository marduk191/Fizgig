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

*Status after the 28 Aug Q2 session: still open, deliberately.* The Q2 measurement
dataset is 8 identical copies of ONE clip — overfit-one shaped, so a quality A/B on it
would say nothing about generalisation. Needs a real multi-clip dataset. Note for that
session: on a 5090 the resident full-model plan only fits ≤22-frame clips (Q2), so
either run the A/B at 22 frames or wait for the planner activation term to land.

## Q2 — MEASURED (28 Aug 2026): the activation term is real, linear, and plan-independent

Measured on a real Gizmo-spec clip (2144×3808 portrait, 24 fps, with sound), duplicated
8× and cut to grid lengths, at 0.25 MP (latent 24×40 spatial, audio targets cached).
Full-model FT, `--finetune_rotation 1`, one full window cycle per run, on the 5090 —
no-sim for the 32 GB tier, `FIZGIG_SIM_VRAM_GB` (allocator-capped) for 24/16.
Runs + logs: `Desktop/fizgig-ft-runs/clipq2/` (RESULTS.md there has the full tables).

**The activation term: ~0.145 GB per latent frame** (grid `latent = 5n+2`, frames
`= 17n+5`), i.e. **~0.9 GB per second of 24 fps clip at 0.25 MP**. Linear across the
measured range (7 → 17 latent frames, extrapolates cleanly to the 37-frame failures),
consistent across windows, and **plan-independent**: it is per-step activation memory,
so splitting or streaming windows does not reduce it (proved by the sim-24 124-frame
run dying in the *forward* at 19.1 GiB live with small split windows). The spike lives
in the checkpoint-segment recompute during backward for the resident plans, and in the
forward itself once windows are small.

**The measured tier map (full-model, 0.25 MP, audio on):**

| tier | plan | 22 fr (0.9 s) | 56 fr (2.3 s) | 124 fr (5.2 s) |
|---|---|---|---|---|
| 32 GB resident | 4 windows | **PASS** (peaks 23.4/17.3/28.5/20.8) | FAIL (fc1, e3) -> **PASS post-fix** (5-window split, max 24.9) | FAIL (qkv, e1 backward) -> honest refusal post-fix |
| 24 GB | 8-window split | *predicted pass* | **PASS** (max 21.2 / 23.88) | FAIL (step 0, forward) |
| 16 GB | streamed, 16 windows | **PASS** (max 11.6 / 15.9 — identical to stills) | **PASS** (steady 10.1–11.1) | *predicted ~16.0, at the line* |

Loss fell normally in every passing run (1.15 → 0.50 over a cycle); the a23a325
orphan-release fix and the ring-aware defrag both hold on clips (no boundary
retention, all rescopes clean). Clip cost at passing lengths is **time, not VRAM**:
~2.1–3.2 s/it at 22 frames, ~5.4–5.9 s/it at 56, vs ~0.7–1.5 s/it stills.

**Product consequence (the actionable one):** the FT planner has no activation term,
so a 32 GB card with >22-frame clips gets the resident plan and OOMs — and
`FIZGIG_SIM_VRAM_GB` is not a workaround, because it caps the allocator along with the
planner. **At 0.25 MP there is currently no configuration that full-model-trains 5 s
clips on ≤32 GB.** The fix is exactly the Q4 bundle below: give `plan_component_windows`
an activation term (`~0.145 GB × latent_frames × mp/0.25`, from the largest cached item)
so it downshifts to split/streamed plans — and refuses honestly when even streaming
cannot fit. **BUILT 29 Aug (3706893), Peter's word given, and gate-run measured**:
`ft_clip_activation_gb` (0.145/latent-frame + a 2.0 GB fragmentation margin, clip
datasets only; the probe keys on the cache header's `audio_only` discriminator so voice
placeholders cannot trip it). 32 GB x 56-frame now plans a 5-window resident split and
PASSES (peaks 24.9/18.7/22.5/23.2/22.5 — the fc1 epoch that OOM'd at ~30.5 peaked 22.5);
32 GB x 22-frame deliberately goes 4->5 windows (fc1 headroom 2.0 -> ~9 GB); 16 GB x
124-frame now refuses up front with the cut-to-2.3 s remedy instead of OOMing mid-run.
Stills plans are bit-identical (act = 0); Krea 2 and every LoRA path untouched.

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

## Recommendation (updated 28 Aug after the Q2 session)

1. **Q2 is measured.** The activation term is ~0.145 GB/latent-frame at 0.25 MP,
   plan-independent, and the tier map above is real. The headline: 22-frame clips fit
   every tier down to 16 GB; 2.3 s clips fit 24 GB; 5 s clips fit nothing ≤32 GB today.
2. **The Q4 + activation-term planner change is now the gating item** — without it, a
   32 GB card with normal-length clips picks a plan measured to OOM. One budgeting
   change, calibrated by this session's numbers. Needs Peter's go-ahead.
3. **Q1 (block zones for clips) stays next after that**, and needs a real multi-clip
   dataset — the Q2 duplicate-clip set is overfit-one shaped and can't judge quality.
4. **Q3 is closed.** Q5 needs only the GUI estimate corrected, which is cosmetic.

The gating input is no longer measurement — it is the planner decision (2) and more
source video for (3).
