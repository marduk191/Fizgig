# Fizgig v4.3.1

A maintenance release — every fix in it came from the community finding, diagnosing, or
building the solution.

## Captioning no longer slows down your next training run

AI captioning (Florence-2 and Qwen3-VL) now runs in its own worker process instead of inside
the GUI. Previously, captioning could leave GPU memory held by the app even after "unload",
and a training run started afterwards would silently plan around the missing VRAM — up to 4×
slower steps until you restarted Fizgig. Now the memory genuinely comes back: the worker
releases automatically when you start training, open Repair Studio / Explorer / Royale, press
Unload, or close the app — and stays warm in between, so repeated single-image Regenerates
are instant.

Diagnosed, measured, and built by **@scryptio**
([#90](https://github.com/shootthesound/Fizgig/issues/90) →
[#93](https://github.com/shootthesound/Fizgig/pull/93)), validated on AMD and NVIDIA.

## 12 GB cards: two crash fixes

From a great RTX 5070 field report by **u/mabseyuk**, who proved MiniMax H3 LoRA training
is otherwise fully stable at 12 GB:

- **Checkpoint saves no longer crash on low memory.** The optional model-hash metadata
  computed at save time could raise MemoryError at the exact moment RAM was tightest —
  killing the run before the checkpoint was written. The hashes are now skipped gracefully
  when memory is short; the checkpoint always saves.
- **No more out-of-memory after previews.** On tight cards, the epoch preview could fragment
  GPU memory badly enough that the next training step failed. Fizgig now detects this and
  compacts memory after the preview — takes seconds, and only runs on cards that need it.

## AMD: gfx12 workaround removed

The RDNA4 batched-GEMM workaround shipped in v4.3.0 is gone: **@0xDELUXA** measured that it
no longer helps on current stacks (and actively hurts the raw batched path), with careful
methodology including control runs. The launchers also clear the old variable from existing
installs, so nothing lingers. Thanks also to @scryptio for re-examining the original
measurement.

## Also

- The README's Korean translation entry now uses **@ssain3d-lgtm**'s own wording — including
  한국어, so Korean users can actually spot it.
