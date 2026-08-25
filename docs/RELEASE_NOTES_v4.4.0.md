# Fizgig v4.4.0 — MiniMax H3 training comes to 12 GB cards

## Two streamers from u/mabseyuk, picked automatically

**u/mabseyuk** — whose 5070 field reports drove the 12 GB fixes in v4.3.1 — built and
field-proved the two pieces that make MiniMax H3 practical on a 12 GB card, and both now
ship in Fizgig:

**Block streaming for the 4-bit base.** Parked DiT blocks now stream host-to-device
through a prefetching ring instead of crossing PCIe both ways every step. On his 5070,
training went from **12–14 seconds per step to about 1 second** — a full 2,800-step run
in under 50 minutes. The output is identical to the unstreamed path: same losses, same
gradients, verified bit-for-bit.

**Text-encoder streaming for caption caching.** Caching captions on a 12 GB card used to
mean the text encoder crawling in system RAM at minutes per image. It now streams its
layers through the card two at a time — **~235 s/image down to ~2.8 s per batch**, using
under 2 GB of VRAM, with bit-identical embeddings.

There's nothing to configure: the VRAM planner reads your card at launch and picks the
right transport, the same way it already picks base precision and block swap. Cards that
never needed streaming behave exactly as before. (For field debugging only:
`FIZGIG_NO_NF4_H2D=1` and `FIZGIG_NO_TE_H2D=1` switch the streamers off.)

## The VRAM planner tells the truth about tight fits

- **12 GB cards no longer get handed a doomed int8 plan.** Auto now knows the tested
  floor for int8 streaming and drops to the 4-bit base below it, instead of planning a
  run that crashed before step one.
- **A plan that can't fit says so.** With heavy video clips, even maximum block swap
  can leave a card genuinely short — previously you found that out from an
  out-of-memory error at the first training step. The planner now says up front how many
  GB short the plan is and what to change (lower Target Megapixels, shorten the heaviest
  clips).
- **Video-tier plans carry a safety margin.** Reported after a 4090 owner hit training
  OOMs: plans at video-scale token loads were running closer to the edge than stills
  plans. Near-the-edge int8 plans now fall back to the smaller 4-bit base instead of
  gambling. Stills plans are unchanged to the byte.

## Clearer reference (vision) caching on small-RAM machines

- A machine that can't hold the reference encoder's staging in system RAM now gets a
  clear explanation up front — with your options — instead of a misleading
  "CUDA out of memory" minutes later with headroom still free on the card
  ([#95](https://github.com/shootthesound/Fizgig/issues/95)). `FIZGIG_TE_RAM_OK=1`
  attempts it anyway.
- On boxes where RAM is merely tight (not impossible), pinned-memory setup now degrades
  gracefully to ordinary RAM instead of crashing the caching pass
  ([#94](https://github.com/shootthesound/Fizgig/issues/94)).

## References off now means off

If you cached a dataset with reference distillation and later re-ran it without,
leftover teacher-pairing files could silently keep distillation active. Caching without
references now clears them, and says so in the console.
