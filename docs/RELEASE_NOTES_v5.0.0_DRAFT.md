# Fizgig v5.0.0 — DRAFT (do not publish)

<!-- DRAFT STATUS: builds as the FT release firms up. Before publishing:
     - DONE 29 Aug: NF4 quality signed off by Peter - every successful field run since 23 Aug ran the NF4 trunk (64-epoch fair-trial runs included) and the checkpoint is bf16 from the master regardless
     - DONE 29 Aug: 32 GB Krea 2 smoke green (NF4 default resolves flagless; 4 resident windows, peaks 22.6/20.8/20.8/20.8, ~1.0 s/it, bracket preview rendered, exit 0)
     - contributor credits for the cycle get added in the usual style -->

Fine-tune the base model itself — on the card you already have.

## Full fine-tuning graduates

Everything Fizgig trains has been a LoRA: an adapter riding on a frozen model. This
release trains the **model itself** — full-rank updates, no adapter, no rank
bottleneck — for both **Krea 2** (12.9B) and **MiniMax H3** (33B video), on a single
consumer GPU. One checkbox on the Training tab; the planner reads your free VRAM at
launch, picks the plan that fits, and prints what it chose.

**What can my card fine-tune?**

| Your card | Krea 2 — photos | MiniMax H3 — photos | H3 — voice | H3 — video clips (with sound) |
|---|---|---|---|---|
| **16 GB** | ✅ | ✅ | ✅ | ✅ up to **2.3 s** |
| **24 GB** | ✅ | ✅ | ✅ | ✅ up to **2.3 s** |
| **32 GB** | ✅ | ✅ | ✅ | ✅ up to **2.3 s** |

Clip lengths follow Gizmo's grid — 2.3 s is the 56-frame slot, so cut clips there and
everything fits. Longer clips currently need more than 32 GB. One clip anywhere in the
folder trains the whole model, so mixed photo + clip datasets use the clip column.
12 GB cards train LoRAs, not fine-tunes — 16 GB is the fine-tune floor. Every number
in that table comes from a measured run, not an estimate.

## What makes it fit (yes, really, 16 GB)

A naive full fine-tune of Krea 2 needs ~78 GB. Fizgig's trainer rotates a trainable
window through the model — every weight trains over a cycle, but gradients and
optimizer state only ever exist for the active slice — on top of a **4-bit NF4 frozen
base** (the fine-tune default: half the model held on the card, and on 24 GB it's
also ~3× faster than the fp8 base because the windows stay resident). Your saved
checkpoint is unaffected by any of this: it's written in bf16 from a master copy that
never passes through a quantiser.

If "a 33B video model fine-tuning on 16 GB" reads like a trick — the numbers are
measured, not projected: **8.8–12.3 GB peaks on a 16 GB card for H3, 8.4–11.0 GB for
Krea 2**, and the console prints your own run's peak every epoch so you can watch the
claim hold live. The full mechanism, the card tiers, and every "how do I" question:
**[docs/FINETUNE_HOWDOI.md](docs/FINETUNE_HOWDOI.md)**.

## When you're done

The built-in **Checkpoint to LoRA** utility diffs your fine-tune against the base and
extracts an ordinary, shareable LoRA at any rank — the quality of a full fine-tune, in
a file ComfyUI already knows how to use. Or keep the checkpoint: it's a valid training
base itself, and Pause / Resume works mid-fine-tune with everything carried over.

<!-- sections to add as the release firms up: intelligent-trainer features on FT
     (problem images, adaptive throttle), likeness-mode recommendation, RAM/disk
     requirements, upgrade notes, credits. MUST include the LR guidance from the README:
     fine-tune rates are much lower than LoRA rates (H3 3e-5 tested / 1e-4 destroys;
     Krea 2 may start at 1e-4 but best results are found lower), and the regularisation
     LR x multiplier is a dial worth experimenting with. -->
