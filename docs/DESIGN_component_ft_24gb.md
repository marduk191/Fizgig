# Component-mode H3 fine-tune on 24 GB (and 16 GB) — design note

Status: **BUILT AND FIELD-GATED (27 Aug 2026).** Gate results, all under real allocator
caps (`FIZGIG_SIM_VRAM_GB`), likeness config on H3:
  * **H3 24 GB — PASS.** The 5-window Route A plan exactly as designed below (only
    `mlp.fc1` splits), peaks 19.1–21.5 GB, full cycle + checkpoint.
  * **H3 16 GB — PASS.** 9 windows, 15 blocks resident / 35 streamed, peaks 8.8–12.3 GB,
    **~1.5× step time — far better than the 2–4× feared below**, and the bracket preview
    renders with the ring live. Needed three fixes at the rescope: drop the old ring ref
    before rebuilding (the CPU-staging doubling `enable_block_swap` already solved),
    evict-before-bind, and a **ring-aware defrag** (park the DiT at each rotation before
    clearing the offloader refs, so `empty_cache` can return whole segments).
  * **Krea 2 24 GB — PASS.** 8 windows, peaks 15.6–17.6 GB. Needed evict-before-allocate
    at setup: its offloader's constructor only RECORDS the resident set, so the full
    ~13 GB fp8 base was still on the card when the first window's bf16 landed.
  * **Krea 2 16 GB — PASS, via an NF4 frozen trunk.** On fp8 it FAILED on allocator
    fragmentation rather than arithmetic (7.35 GiB live vs 7.40 GiB stranded), because its
    `RotationOffloader` moves whole blocks with fresh allocations each time where H3's ring
    reuses flat buffers plus the boundary defrag. The fix was not to port the ring but to
    halve what the streamer moves: Krea 2 FT had REFUSED 4-bit outright, so its trunk was
    always ~13 GB of fp8. Lifting that gives a **6.08 GB packed trunk** and ~6.1 GB staged
    instead of 13.0 — 12 windows, peaks **8.7–11.0 GB** under a 15.9 GB cap, ~2.8 s/it,
    boundaries clean, and pause/resume verified across a rotation. Fragmentation never
    bit, because there is half as much to fragment. The checkpoint is unaffected: Krea 2
    saves bf16 straight from the CPU master, so NF4 is only the frozen forward CONTEXT
    (which is also why the residency re-encode can round to nearest — pinned: 10
    activate/deactivate cycles leave the master bit-identical).
    **Reachability caveat:** "Auto" resolves through a LoRA-shaped recommender with no FT
    awareness, which prefers INT8, and the FT branch then coerces INT8 → fp8 — so NF4 is
    never chosen automatically. The user must pin Base precision to 4-bit NF4. The trainer
    now warns when it lands on fp8 below ~20 GB free; changing the Auto policy is Peter's
    call, not done.
  * Constant corrected: `_FT_OVERHEAD_GB` 13.3 → **14.5**. The per-window table below is
    in **GiB** and was being subtracted as GB — a ~7% systematic underestimate. Krea 2's
    fp8 `_K2FT_OVERHEAD_GB = 16.0` is conservative but was left alone; it cannot be
    isolated from a single run. The NF4 path has its own pair
    (`_K2FT_NF4_OVERHEAD_GB` / `_K2FT_NF4_GB_PER_BLOCK`).
  * Both rotators' `_deactivate_targets` now RELEASE the outgoing bf16 by resizing its
    storage to 0 — rebinding `lin.weight` does not free it (a C++-side autograd referrer
    keeps it alive and it is no longer a module parameter for gc or the park walk to
    find), so every rotation was costing TWO windows. That, not the budget, is what failed
    full-model H3 at 24 GB. The release is guarded on the re-encoded weight not sharing
    that storage: freeing by storage frees every tensor sharing it, so an encoder that
    returned a view instead of a copy would be handed a zero-byte weight.

Route A shipped as designed
(depth-split windows, planner-selected: `plan_h3_ft_windows` in rotation_ft.py, 41 CPU
pins in tests/test_ft_small_cards.py). The 16 GB tier shipped NOT as Route B's classic
parking swap but as the stronger option that became possible after this note was
written: the **NF4 H2D ring generalized to an arbitrary streamed set** — depth-splitting
makes out-of-window blocks FULLY frozen again, which dissolves the ring-vs-rotator
conflict that forced swap off under FT. The ring rescopes at every rotation
(`_ft_rebuild_ring`), streamed blocks stage ~10.5 GB in RAM, and the resident window
rides `bind_block_packed_to` past the bnb re-quantize trap. GPU parity battery:
tests/test_ft_ring_scope.py. `FIZGIG_NO_FT_STREAM=1` is the kill-switch back to the
resident-only plan. Still unrun: pause/resume on the streamed tiers, and a ballast run
for genuine physical scarcity (the simulator caps this process, not the card).
Original note follows.

Route A is the deliverable for the program's release; Route B follows measurement.
Written the night the disk-backed master landed. FT-branch only until the program merges.

**Scope update (25 Aug)**: the fair-trial verdict made **likeness ON the recommended H3 FT
recipe** (vastly better output and prompt adherence than full-model), and the likeness
component run's measured window peaks — 18.4 / 19.0 / 20.3 / 22.8 GB — already brush a
24 GB card's usable ceiling with NO window splitting. So the recommended config may fit
24 GB today (needs one gate on a simulated 24 GB budget; 22.8 vs ~23 usable is tight), and
Route A's main customer shrinks to likeness-OFF runs (style/scene fine-tunes) plus safety
margin for the recommended path.

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
