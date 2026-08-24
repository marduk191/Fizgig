# Two-Machine Fine-Tuning (Krea 2) — feasibility

Branch: `experiment/ft-rotation`. **Nothing here is built.** This is a costed feasibility study
for splitting a rotating-block fine-tune across two machines over gigabit ethernet.

Companion to [`FINETUNE_ROTATION.md`](FINETUNE_ROTATION.md), which covers the single-machine
trainer this would extend.

---

## The question

Can a Krea 2 full fine-tune be spread over two machines, using the NVENC activation-compression
work in `torch-nvenc-compress` as the wire?

**Verdict: yes, via pipeline parallelism — and the interesting part is that the codec is not what
makes it work.** The wire was never the binding constraint at Fizgig's usual resolutions. VRAM
was. Splitting the block list is a direct attack on VRAM, and it reaches the one configuration
that block streaming provably cannot help.

---

## Data parallelism is dead — worth killing first

Both machines train the full model on different images and sync gradients each step. The sync is
the size of the trainable window. In component mode that window is attention across all 28
blocks: **3.70 GB of bf16 gradients per step.**

| | per step |
|---|---|
| gigabit, uncompressed | ~33 s |
| gigabit, at the codec's 6.1× lossless ratio | ~5.4 s |

Against a ~1.0 s/it step time. Compression doesn't rescue this — it's two orders of magnitude
out, and no plausible ratio closes it. **Pipeline parallelism is the only route.**

---

## What actually crosses the wire

A detail in `SingleStreamDiT` decides this: the block stream is a single tensor. Text is
concatenated in before block 0 (`combined = torch.cat((img, context), dim=1)`, padded to
`max_length=512`), and RoPE `freqs` is recomputed from positions rather than carried. So one
pipeline split point ships exactly one `[1, T, 6144]` bf16 tensor forward, and its gradient back.

| Resolution | Tokens (img + 512 txt) | Per step, both directions | Real gigabit (~112 MB/s) | At 6.1× |
|---|---|---|---|---|
| 512 px | 1536 | 38 MB | 0.34 s | 0.06 s |
| 768 px | 2816 | 69 MB | 0.62 s | 0.10 s |
| 1024 px | 4608 | 113 MB | 1.0 s | 0.17 s |

Against ~1.0 s/it, **uncompressed gigabit is already viable at 512–768 px** — a 34–62 % step
penalty, most of which `gradient_accumulation_steps` (already in the trainer) hides behind a
GPipe-style schedule. At 1024 px the wire costs as much as the compute and compression becomes
load-bearing rather than optional.

---

## Per-stage VRAM

Component mode, 512 px, gradient checkpointing on, optimizer-in-backward. Derived from real key
shapes in the RAW checkpoint header — **modelled, not measured on two machines.**

Per block: 434 M params → 0.43 GB fp8 frozen. The largest component window is **attention** at
0.26 GB bf16 per block (`attn` bundles wq/wk/wv/wo/gate = 132 M; each MLP matrix is 100.7 M) —
so attention, not MLP, sets the window peak.

| blocks on stage | fp8 frozen | window bf16 | acts | steady | peak (rotation boundary) |
|---|---|---|---|---|---|
| 8 | 3.47 | 2.11 | 0.15 | 6.64 | **8.75 GB** |
| 10 | 4.34 | 2.64 | 0.19 | 8.07 | **10.72 GB** |
| **12** | 5.21 | 3.17 | 0.23 | 9.51 | **12.68 GB** |
| 14 | 6.08 | 3.70 | 0.26 | 10.94 | 14.64 GB |
| 16 | 6.95 | 4.23 | 0.30 | 12.38 | 16.60 GB |
| 20 (+txtfusion) | 8.68 | 5.28 | 0.38 | 16.62 | 21.90 GB |
| 28 (single machine) | 12.16 | 7.40 | 0.53 | 22.36 | 29.76 GB |

**Calibration:** the model predicts 29.76 GB for the 28-block single-machine case against the
**measured 27.67 GB** — it overstates by ~7.5 %. Conservative is the right direction for a fit
question, so the figures above are used as-is rather than scaled down.

The peak is the rotation boundary in every row, same as single-machine: both the outgoing and
incoming windows are briefly resident.

**txtfusion (343 M, 1.37 GB resident) lives on stage 0 only** — it runs before the blocks and is
held always-on rather than rotated.

---

## Per-stage system RAM

The CPU bf16 master copy dominates, and scales linearly with blocks assigned (0.87 GB/block).

| blocks on stage | VRAM peak | bf16 master | + torch/staging + OS |
|---|---|---|---|
| 8 | 8.75 GB | 6.95 GB | ~14 GB |
| 10 | 10.72 GB | 8.68 GB | ~16 GB |
| 12 | 12.68 GB | 10.42 GB | ~17.5 GB |

**On a 16 GB-VRAM laptop the binding constraint is system RAM, not VRAM.** 32 GB comfortable at
any split, 24 GB fine at 8–10 blocks, 16 GB unworkable.

**Load-time footprint is worse than steady state.** `_build_bf16_master` calls
`load_file(raw_path)` on the whole 25.8 GB RAW checkpoint and clones out the keys it wants. It's
mmap'd so it isn't 26 GB committed, but the machine must hold the full RAW file locally and page
through all of it to extract its share. **Pre-sharding the checkpoint per stage** is the fix —
each machine opens only its own blocks, and the full model never needs to exist on the smaller
machine's disk. Worth doing before any two-machine attempt.

---

## What this would unlock

A **12/16 split puts the small stage at 12.68 GB and the large stage at ~18 GB.** Both under
22.4 GB.

That is the point of the exercise. Today:

| | today | with a 12/16 split |
|---|---|---|
| component mode (quality-preferred) | 32 GB card only | **24 GB + 16 GB pair** |
| 24 GB card | block mode only — never quality-tested | component mode |
| 16 GB card | nothing fits (floor 17.62 GB) | works as the small stage |

Component mode is exactly the configuration **frozen-block streaming provably cannot help** —
every block holds a trainable slice, so there's nothing to stream out. Splitting the block list
attacks it from the only remaining direction. And every good fine-tune result so far came from
component runs, so this reaches the mode that matters rather than the one that merely fits.

The pairing is also common hardware rather than an exotic rig, which is what would make it worth
building.

---

## Where the codec is and isn't load-bearing

Being straight about this, because the framing in `torch-nvenc-compress` points the other way:

- **512–768 px, batch 1, single split:** not needed. Uncompressed gigabit already works.
- **1024 px+, or multiple split points:** load-bearing. The wire matches compute time
  uncompressed.
- **Long-wire (rented GPU over broadband):** load-bearing, and this is the strongest fit.

That last case deserves flagging upstream. `torch-nvenc-compress` frames its hybrid local-cloud
scenario (Test 6) as **inference**, but inference is where round-trip latency hurts most.
**Training is far more forgiving:** a fine-tune is a multi-hour background job, 100 ms of RTT
against a ~1 s step is 10 % before grad accumulation hides it, and there is no interactive user
waiting. "Rent a GPU for half your fine-tune" is a better fit for that primitive than the
inference equivalent, using the same code.

---

## Open questions, in the order that kills this earliest

1. **Do gradients compress like activations?** The 6.1× lossless figure is for **activations**.
   What crosses backward is ∂L/∂x, a different distribution, and the gradient test in that repo
   is unrun. Cheap to answer without a training loop: hook a real run, dump grad tensors at a
   block boundary, push them through the existing PCA + NVENC pipeline, plot the same Pareto.
   An afternoon's work, and it gates the 1024 px and long-wire cases entirely.

2. **Does lossy compression damage a training run, and would we notice?** In inference the error
   is one-shot; in training it enters the gradient signal every step for thousands of steps.
   **Loss will not detect this** — the bilingual-caption A/B established that loss curves are
   blind to quality differences that matter (±0.001 per epoch, visibly better output). An A/B on
   loss would return a false pass. Fizgig has the right instruments for once: gallery likeness
   scoring and the per-image loss watch measure the thing loss can't see. That makes this repo a
   better validation harness for the compression claim than a generic training script.

3. **Fragmentation.** The budget above is static — parameter counts plus a calibrated overhead
   term. It doesn't model allocator fragmentation, which is exactly what caused the long-run
   slowdown (fixed with `expandable_segments:True`, already set globally). A 1.8 GB margin at 12
   blocks is thin. Probe before trusting 12 over 10.

4. **Pipeline bubble.** At batch 1 the two stages strictly alternate: each machine is idle half
   the time and step time is single-machine compute plus wire. **The win is VRAM, not speed** —
   it may in fact be slightly slower than one big card. Grad accumulation is what converts it
   into a throughput win (accum=4 across 2 stages → ~80 % utilisation), and it's already
   implemented.

5. **Heterogeneous stage balance.** A laptop 4090 is much slower than a desktop 5090. The split
   ratio has to balance *time*, not just VRAM, or the fast machine waits. Since VRAM, RAM and
   speed all pull on the same knob, there may be no ratio that satisfies all three.

6. **Rotation schedule sync.** Both stages must be in the same window at the same epoch or they
   train mismatched slices. Each stage owns its own master copy and optimizer state locally —
   **no cross-machine optimizer traffic at all** — so this is just a synchronised epoch counter.
   Cheap, but silent and catastrophic if wrong.

---

## What would need building

Roughly in dependency order:

1. Checkpoint pre-sharder (per-stage RAW shards) — also removes the load-time RAM spike.
2. Stage wrapper: a `SingleStreamDiT` that owns a block subrange and exposes send/recv at its
   boundary, with `autograd.Function` handling the backward hand-off.
3. Transport: plain TCP first, uncompressed. Codec swaps in behind the same interface later.
4. Synchronised rotation schedule + epoch barrier.
5. Checkpoint assembly: stage 1 ships its master half back once at run end (~10 GB, ~90 s on
   gigabit), stage 0 writes the full `.safetensors`.
6. Grad-accumulation-aware pipeline schedule (the speed win, not required for correctness).

Steps 1–5 are a working two-machine fine-tune with no compression at all. That is the honest
first milestone, and it's the one that proves or kills the VRAM claim.

---

## Provenance

VRAM and RAM figures: computed from the safetensors header of
`Krea-2-raw.safetensors` (8-byte length + JSON, no tensor data read, no GPU) — per-key shapes
summed by block and component, plus a 0.9 GB CUDA-context/allocator floor and one checkpointed
`[1, 1536, 6144]` block input per block. Calibrated against the measured 27.67 GB single-machine
component peak from the VRAM sweep in `FINETUNE_ROTATION.md`.

Wire figures: `T = (px/16)² + 512` tokens × 6144 × 2 bytes, doubled for the backward pass.

Compression ratios quoted from `torch-nvenc-compress/docs/findings.md` (LOO-validated 6.1× at
QP=10 / cos 0.991 on FLUX.2 Klein mid-block activations) — **an activation figure being applied
to a gradient question, which is open question 1 above.**

---

## Addendum (24 Aug 2026) — comfyui-mesh is the wire, and the numbers were pessimistic

Two updates that upgrade this study from feasibility to eventually-scheduled:

1. **The transport exists and is field-proven.** Peter''s
   [comfyui-mesh](https://github.com/shootthesound/comfyui-mesh) (Icarus client /
   Daedalus server) already does pipeline-parallel INFERENCE splits of FLUX.2 and
   LTX 2.3 over TCP with NVENC activation compression (3-10x), reconnection, and a
   measured 4.4 s/image for Klein 1024^2 across a 5090+4090 on gigabit. Training adds
   the backward leg (the boundary gradient — same size as the forward tensor, already
   in the tables above), an optimizer per half, and checkpoint assembly from two
   masters. Engineering, not physics.

2. **The wire numbers above were effectively measured on 100 Mbit** — the test machine
   turned out not to be linked at gigabit. True gigabit has ~10x the headroom this doc
   assumed: 768 px training is comfortable UNCOMPRESSED, and 1024 px no longer depends
   on the codec.

The compounding insight: **a mesh split composes with rotation.** Each machine hosts
half the blocks, so its frozen trunk AND its share of the active window halve together
— two 24 GB cards run full H3 component mode with margin, and two 32 GB cards could
plausibly hold the trunk in raw bf16, skipping NF4 and its trained-against-noise
context entirely (see the 24 Aug stochastic-rounding findings). Target shape: a Fizgig
server mode on the second machine, LAN discovery, auto-split. Eventually — after the
FT program ships.
