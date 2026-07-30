# Fizgig v2.8.7 — Hotfix: INT8 restored when 4-bit is off

**Update if you're on v2.8.6 and have a 20 GB or larger card.**

v2.8.6 fixed the block-swap planner, but in doing so it made one change in error: setting the
**4-bit Base** control to *Off* was treated as "use no quantisation at all", which switched off
**INT8** as well as NF4.

That was wrong. The control is labelled *4-bit Base*, so turning it off is a vote against
**4-bit (NF4)** — not against INT8, which is 8-bit, faster than NF4, roughly 7× more accurate,
and trains with exact gradients. The result was that anyone with 20 GB+ of free VRAM and 4-bit
set to *Off* quietly lost the fastest path and fell back to plain fp8. Nothing broke and no run
failed — it was simply slower than it should have been.

**Fixed.** With 4-bit *Off*, Fizgig now uses INT8 wherever it fits and the card supports it, and
fp8 otherwise. NF4 is still never applied when you've said *Off*.

| 4-bit control | 16 GB card | 24 GB card |
|---|---|---|
| Auto | NF4, no swap | INT8, no swap |
| **Off** | **fp8 + 14 swapped blocks** | **INT8, no swap** |
| On | NF4, no swap | NF4, no swap |

Everything else from [v2.8.6](https://github.com/shootthesound/Fizgig/releases/tag/v2.8.6) is
unchanged — the 16 GB out-of-memory fix and the lighter, measured swap counts are all still in
place.
