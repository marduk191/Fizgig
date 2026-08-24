# Component-mode H3 fine-tune on 24 GB (and maybe 16 GB) — design note

Status: **scheduled — part of the FT program** (Peter, 24 Aug 2026: "too good not to").
Route A is the deliverable for the program's release; Route B follows measurement.
Written the night the disk-backed master landed. FT-branch only until the program merges.

## Why 24 GB is currently out

Component mode's VRAM peak is set by ONE window: `mlp.fc1` across all 50 blocks is
15.4 GB of trainable bf16, and the measured full-model peaks per window are (5090,
0.25 MP, batch 1, fused backward, gradient checkpointing):

| window          | bf16 window | measured peak |
|-----------------|-------------|---------------|
| `attn.qkv_proj` | 8.7 GB      | 21.6 GB       |
| `attn.out_proj` | 2.9 GB      | 16.2 GB       |
| `mlp.fc1`       | 15.4 GB     | **24.7 GB**   |
| `mlp.fc2`       | 7.7 GB      | 18.8 GB       |

(Peak ≈ NF4 trunk 10.5 GB − the active component's freed NF4 share + the bf16 window +
~2.5 GB activations.) A 24 GB card has ~23 GB usable: three windows fit today, fc1 does
not, and qkv is marginal. System RAM is already solved — the disk-backed master
(`--finetune_master disk`) runs the whole thing at a ~4 GB working set.

## Route A — split the fat window by depth (full speed, ~20 lines)

Make the cycle 5 windows instead of 4: `fc1 @ blocks 0-24`, `fc1 @ blocks 25-49`
(~7.7 GB each) → fc1's peak drops to ~19 GB. If qkv's 21.6 proves tight on a real
24 GB desktop, split it the same way (6 windows).

- No offloading, no speed cost. Cycle grows to 5-6 epochs (save cadence follows
  automatically — the Save-every box already tracks the cycle).
- The trade: fc1 trains at half depth per window, mildly diluting the full-depth-per-
  window geometry that makes component mode learn likeness fast. Attention and fc2 keep
  full depth, so most of the win should survive — worth an A/B before it becomes a tier.
- Implementation: `RotationSchedule` component entries become (prefix, block-range)
  pairs; `_targets` already takes prefixes, so the range is a filter on its block loop.

## Route B — classic block swap under FT (Peter's offload idea)

Park N trunk blocks on CPU with the classic `.to()` parking swap (NOT the int8 H2D
streamer — that is what FT disarms; bnb NF4 weights already move packed, and training
params inside swapped blocks is standard musubi-style block swap).

The wrinkle in its favor: a component window spans EVERY block, so parking a block
parks its slice of the active window too — swap shrinks the frozen NF4 **and** the
trained bf16 residency at once. Roughly, with S of 50 blocks swapped:

```
peak ≈ (50-S)/50 × (10.5 trunk + window) + ~2 GB stream slots + ~2.5 GB activations
```

- Swap 20 → fc1 peak ≈ 20 GB → fits 24 GB.
- Swap ~35 → ≈ 13-14 GB → brushes **16 GB**. "Full fine-tune of a video model on a
  4080" — slow, but a sentence nobody has written.
- Cost: the PCIe tax. Krea measured classic swap at up to 4.4× step time on constrained
  cards; H3's blocks are smaller per step so likely kinder, but it needs measuring.
- Care points: the rotator's class-swapped Linears must survive `.to()` round-trips
  (Parameter identity changes on move → the fused per-tensor optimizers need the same
  rebind treatment the rotation sites got); the defrag round-trip and the swap must not
  fight over the same blocks.

## If built: auto-tiers

| free VRAM | config                                   | speed     |
|-----------|------------------------------------------|-----------|
| ~31 GB    | current 4-window cycle                   | full      |
| ~23 GB    | Route A split windows (5-6 per cycle)    | full      |
| ~15 GB    | Route A + Route B swap                   | 2-4×/step |

Route A alone is the 24 GB story and costs almost nothing; Route B is what stretches
to 16 GB. Both compose with the disk-backed master and stochastic-rounding saves
unchanged.
