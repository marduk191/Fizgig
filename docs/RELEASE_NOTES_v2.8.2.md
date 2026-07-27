# Fizgig v2.8.2 — Long runs stay fast

One fix, and it's a good one: **training no longer slows down as a run goes on**.

## 🐌 → 🚀 The slowdown that had no obvious cause

On long runs — especially on cards with little VRAM headroom — per-step time crept upward and
never recovered, in one measured case roughly **doubling** by two-thirds of the way through a
run. Nothing in the usual telemetry explained it: the GPU sat at 90% utilisation, full clocks,
73°C, and well under its power limit, with no throttling of any kind.

The cause was **memory fragmentation**. PyTorch's default allocator hands out GPU memory in
fixed-size chunks, and training repeatedly allocates and frees large tensors of *different*
sizes. Near a full card, the free space gradually stops being usable in the shapes the next step
needs, and the allocator falls back to slow, synchronising calls to the driver — progressively
more of them as the run continues.

Fizgig now enables PyTorch's **expandable memory segments**, which let a memory region grow and
shrink instead of being carved into fixed blocks. **Confirmed on a real run to remove the
slowdown entirely.**

Applied everywhere it matters, so it doesn't depend on how you launch:

- **Training from the GUI** — Klein and Krea 2 alike.
- **Headless / CLI training** — previously missed out; now covered.
- **The workbench** — Repair Studio, LoRA the Explorer and LoRA Royale run inside the app and
  churn memory hard (a render per slider move, four per Explorer round, a swap per LoRA), so they
  get it too.

Nothing to configure. If you ever want the old behaviour for comparison, set the environment
variable `FIZGIG_NO_EXPANDABLE=1`.

**Who benefits most:** anyone running close to their card's limit, and anyone doing long runs.
If your steps were fine at the start and sluggish by the end, this was why.

---

**For the curious:** the reason this went unnoticed for so long is that nothing points at it. GPU
utilisation, clock speed, temperature and power draw all look perfectly healthy while it happens
— and the progress bar's own "s/it" is a running average over the whole run, so it drifts upward
by itself and can't distinguish "steps got slower" from "time was spent not stepping". The only
tell is memory occupancy sitting near the card's limit.
