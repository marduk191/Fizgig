# Fizgig v2.8.2 — Long runs stay fast

One fix, and it's a good one: **training no longer slows down as a run goes on** — on Windows and
Linux alike, via a different allocator on each.

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

**On Windows, expandable segments don't exist** — PyTorch's CUDA allocator rejects the option
outright and silently falls back to the default allocator, logging `expandable_segments not
supported on this platform`. So Windows users got the warning and none of the fix.

Windows now gets **CUDA's stream-ordered allocator** (`cudaMallocAsync`) instead, which solves
the same problem a different way: the driver manages one growable pool rather than PyTorch
carving fixed segments, so a freed block of the wrong shape no longer strands memory.

Measured on a fragmentation reproduction at ~5% free headroom, 200 iterations, on both an RTX
3060 and an RTX 5090:

| | default allocator | `cudaMallocAsync` |
|---|---|---|
| Allocator retries | 9 | **0** |
| Worst-case step | 84 ms | **3.8 ms** |
| Median step | 2.9 ms | 2.8 ms |

The retry count is the whole story. A retry is a synchronising `cudaFree`/`cudaMalloc` round trip
costing roughly **30x a normal step**, and they get more frequent as free space fragments — which
is exactly the creeping slowdown. Median step time is unchanged, so this costs nothing when the
card isn't under pressure.

Validated on a full Krea 2 training run with block swap (20 of 28 blocks — the heaviest
memory-churn path there is), step time flat start to finish, LoRA output verified intact.

Two other allocator settings were tried and rejected: `max_split_size_mb` made everything ~20x
slower, and `garbage_collection_threshold` changed nothing.

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
