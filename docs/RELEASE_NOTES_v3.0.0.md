# Fizgig 3.0 — LoKR training for Krea 2

Fizgig now trains **LoKR** (LyCORIS Kronecker) on Krea 2 — the parametrization the community
rates highest for character likeness. Pick it from the new **Network Type** dropdown on the
Training tab; standard LoRA remains the default.

## Why you'll want to try it

Same dataset, same settings, LoKR vs standard LoRA in our validation runs:

- **The highest likeness we have ever measured** with Fizgig's own ArcFace scorer — beating
  every previous run on the same instrument.
- **Noticeably more natural skin and light** — less of the shiny, over-rendered look, visible
  from the earliest epochs and holding through convergence.

Where LoRA squeezes learning through a thin low-rank slice, LoKR covers the whole weight matrix
with structure. Identity seems to live in exactly the kind of change that captures. The
straightforward trade, from our testing: **LoKR is slightly higher quality, LoRA is slightly
faster** — the hint next to the dropdown says the same.

## How it works

- **One dial.** The **Factor** field replaces rank and alpha. Lower factor = more capacity and
  bigger files (factor 8 ≈ 400 MB, factor 16 ≈ 100 MB). 8 is the validated default.
- **Everything intelligent still applies** — per-image loss watch, adaptive LR, auto-recaption,
  Context LoRA (any format context under any network type), pause/resume, live previews. No
  other LoKR trainer has any of that.
- **Straight into ComfyUI.** Output is standard LyCORIS format; epoch checkpoints too.
- **Klein is unchanged** and trains standard LoRA as always.

Headless: `--network_type lokr --lokr_factor 8` on `krea2_train.py`.

## Repair Studio + LoRA the Explorer: LoKR in, LoKR out

Editing a LoKR (or LoHa) no longer converts it to standard LoRA on save. Slider changes bake
**losslessly** into the Kronecker factors — a saved file with no edits is byte-identical to the
original. The only case that still converts is donor-blending a block, where it's mathematically
unavoidable, and the save dialog tells you exactly which blocks were. The load-time conversion
popups are gone, and LyCORIS files load faster.

## Also in 3.0

- **Resolution changes re-cache correctly** — cached latents from a different Target Megapixels
  setting are detected and refused with a clear "re-run cache preparation" message instead of
  silently training at the old resolution.
- **Crash fix** — clicking the IDLE/BUSY light (or having the console log open) during model
  loads or background work could hard-crash the app. Fixed everywhere.
- **LoRA the Explorer** shows the baseline image the moment it renders, and each variant appears
  as it completes — no more waiting for the full set.
- Training runs at below-normal priority so your desktop stays smooth (see the README note on
  Hardware-accelerated GPU scheduling if you still see stutter).

---

**Upgrading:** run `update_fizgig.bat`, or `git pull`.
