# Fizgig v2.9.0 — One honest control for Krea 2 memory

The **4-bit Base** dropdown is now **Base precision**, and it finally says what it does.

## The problem

The old control offered *Auto / On / Off*, and "Off" meant "don't use 4-bit" — after which, on
a capable card, you silently got **INT8**. That was never wrong exactly, but INT8 appeared
nowhere in the interface: Fizgig's fastest and most accurate quantisation could only be reached
by leaving a differently-named dropdown alone.

Three bugs in three days came out of that ambiguity, including one where we read "Off" as "no
quantisation" ourselves and briefly disabled INT8 for everyone with a big card.

## The fix

One dropdown, every option named, each one properly planned for:

| Base precision | What it does |
|---|---|
| **Auto (recommended)** | Picks the fastest option that fits your free VRAM, and sizes block swap to match |
| **INT8 — 8-bit, fastest** | Fastest and ~7× more accurate than 4-bit; needs ~18 GB free |
| **4-bit NF4 — smallest** | ~5.6 GB base, fits 10–12 GB cards with no swap, slight quality cost |
| **fp8 — least compressed** | Needs the most VRAM, so blocks are swapped to fit |

What you pick explicitly is now what the memory planner budgets for — block swap is sized for
the option that will actually run, not for one the planner would have preferred.

**Auto remains the default**, and all three Krea 2 presets use it. If you've never touched this
setting, nothing changes for you.

### If you had it set to *Off*

You'll land on **INT8**, because that's what *Off* has actually been doing since
[v2.8.7](https://github.com/shootthesound/Fizgig/releases/tag/v2.8.7). Your existing behaviour
is preserved; it just has a name now. On a 16 GB card that means INT8 with 8 swapped blocks
rather than fp8 with 14 — same fit, fewer transfers per step, faster.

## Smaller things in the same area

- **The section is now "Memory & Precision (INT8 / FP8 / NF4)"**, so INT8 is visible even when
  the section is collapsed.
- **Base precision is Krea 2 only.** It used to appear under Klein as well, which has no INT8
  path and no automatic strategy.
- **Asking for INT8 on a card without int8 tensor cores** now falls back to fp8 instead of
  starting a run that can't work.
- **fp8 is labelled "least compressed" rather than "no quantisation"** — Krea 2's fp8 path is
  itself a dynamic quantisation, so the old wording was simply inaccurate.
