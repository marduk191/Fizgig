# Fizgig v2.8.6 — Block swap auto, fixed and faster

One fix, and it matters if you train Krea 2 on a 16 GB card.

## 🧠 Auto block swap planned for the wrong model

Reported by **@jowala** (thanks again — this is the second real bug from that thread): CUDA
out-of-memory on **Auto** with a 16 GB card running Krea 2 in fp8, cured by manually setting
swap to 16.

The cause: Fizgig plans quantisation and block swap **together**, then applied only half the
plan. If you set the **4-bit Base** control to *Off* — i.e. "run fp8" — your choice was
respected, but the swap count handed to the run was the one calculated for **NF4**, a model
less than two-thirds the size. fp8 then tried to train with no swap at all and ran out of
memory immediately.

Now the quantisation you pin is fed into the planner, so swap is always sized for the model
that will actually run. The same gap applied to **INT8**, which had no GUI control and so
ignored an explicit *Off* entirely — that's fixed too, and *Off* now means what it says.

## ⚡ ...and swap is no longer over-prescribed

While fixing it, the swap counts themselves turned out to be too heavy. They came from coarse
VRAM tiers; they now come from a measured curve. On the real trainer with a real dataset,
constrained to a genuine 14.5 GB budget:

| Blocks swapped | Peak VRAM | Headroom |
|---|---|---|
| 12 | 13.66 GB | 0.84 GB — too tight |
| **14** | **12.88 GB** | **1.62 GB — what Fizgig now picks** |
| 16 | 12.08 GB | 2.42 GB |
| 20 | 10.28 GB | 4.22 GB — what the old table gave you |

Every swapped block is a PCIe round-trip on **every step**, so the old table was costing 16 GB
users six blocks of speed for memory they never needed. A stale internal constant was part of
it — the fp8 footprint was recorded 1 GB light, from a whole-GPU reading rather than a
controlled one; it's now measured directly.

Nothing to change on your side: leave **Blocks Swap** on *Auto* and it plans correctly for
whatever quantisation you've chosen.

## 🔍 Also: compile now tells you when it isn't running

torch.compile and block swap can't coexist — compiled graphs assume weights stay put, and
swapping moves them every step — so the trainer has always ignored compile when swap is
active. But the GUI still displayed "On", so it looked like it was working. It now says so
plainly in the console when it's skipped.
