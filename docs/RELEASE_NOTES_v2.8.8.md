# Fizgig v2.8.8 — FP8 Base hidden for Krea 2

**FP8 Base** is no longer shown on the Training tab when Krea 2 is selected. Unticking it was
never a useful option there — it was a guaranteed out-of-memory error.

The checkbox asked the trainer for a **bf16 base**: 25.8 GB of weights on their own, ~28 GB in
total, which no consumer card can hold. Three things made it worse than merely useless:

- **The automatic block-swap planner never saw it.** The plan came out identical whether the box
  was ticked or not, so the run received a swap count sized for fp8/INT8/NF4 and then loaded a
  model twice that size.
- **Unticking it silently cancelled INT8**, undoing the faster path the planner had just chosen.
- **There's no card where it's the right call.** The base is frozen during LoRA training and your
  LoRA trains in bf16 either way, so the precision you'd gain is negligible.

Krea 2's base precision is chosen on the **4-bit Base** control — *Auto* / *On* / *Off*, giving
NF4 / INT8 / fp8 as appropriate — and the planner accounts for all of those properly.

Klein is unaffected: it uses different flags and keeps the checkbox. The `--no_fp8` option
remains available on the command line for anyone with the hardware to use it.

## Also in this line of fixes

This is the third issue in the same pair of controls in two days, all now closed:

- [v2.8.6](https://github.com/shootthesound/Fizgig/releases/tag/v2.8.6) — block swap planned for
  a quantisation the user had overridden (the 16 GB out-of-memory report), plus lighter,
  measured swap counts.
- [v2.8.7](https://github.com/shootthesound/Fizgig/releases/tag/v2.8.7) — 4-bit *Off*
  accidentally disabling INT8 on 20 GB+ cards.
- **v2.8.8** — this one.

If you train Krea 2, leave **Blocks Swap** and **4-bit Base** on *Auto* and Fizgig will pick the
fastest configuration that fits your card.
