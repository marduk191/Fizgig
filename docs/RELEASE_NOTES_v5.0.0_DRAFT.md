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

The quickest route on either family is genuinely five minutes: tick Fine-tune (the
learning rate, epochs and save cadence switch to fine-tune values by themselves — and on
Krea 2 Adaptive LR steps aside automatically), set your output drive, keep your trigger
tokens — done. Step-by-step recipes for both, and every other question:
**[docs/FINETUNE_HOWDOI.md](docs/FINETUNE_HOWDOI.md)**.

**What can my card fine-tune?**

| Your card | Krea 2 — photos | MiniMax H3 — photos | H3 — voice | H3 — video clips (with sound) |
|---|---|---|---|---|
| **16 GB** | ✅ | ✅ | ✅ | ✅ up to **2.3 s** |
| **24 GB** | ✅ | ✅ | ✅ | ✅ up to **2.3 s** |
| **32 GB** | ✅ | ✅ | ✅ | ✅ up to **2.3 s** |

Clip lengths follow Gizmo's grid — the **2.3 s (56-frame) slot is confirmed by measured
runs on every tier**, and **3.8 s is confirmed on 32 GB** even with video training the
whole model. The new **Restrict video to likeness blocks** tickbox (on by default with
likeness mode — in our tests it trains video just as well, and far lighter on VRAM)
extends the *expected* range to **5.2 s on 24 GB and 32 GB, 3.8 s on 16 GB** —
conservative arithmetic from the measured constants. Whole-model 5.2 s clips need more
than 32 GB (measured). With the restriction unticked, one clip anywhere in the folder
trains the whole model, so mixed photo + clip datasets use the clip column.
12 GB cards train LoRAs, not fine-tunes — 16 GB is the fine-tune floor. Every number
in that table comes from a measured run, not an estimate. Two honest caveats: fine-tuning
is **untested on AMD/ROCm** (every measured tier is NVIDIA), and **Krea 2 fine-tuning
realistically wants 48 GB+ of system RAM** (its ~24 GB master lives in RAM; H3's spills
to disk). The trainer says both out loud at launch where they apply.

## What makes it fit (yes, really, 16 GB)

A naive full fine-tune of MiniMax H3's 33B would need roughly 200 GB; even Krea 2's
12.9B needs ~78 GB. Fizgig's trainer rotates a trainable
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

## One number to respect: the learning rate

If you take a single thing from these notes: **fine-tuning wants much lower learning
rates than LoRA training.** Ticking Fine-tune sets a safe **1e-5** for you on both
families. On **MiniMax H3**, 3e-5 is the tested faster rate and the most you should ever
use — **1e-4 will destroy an H3 fine-tune** (measured, not folklore). On **Krea 2**
you're welcome to experiment up to 1e-4, but the best results are realistically found
lower. And if you use regularisation images, their **LR ×** multiplier is a real dial —
0.1–0.3 tethers the model's prior, higher trains them like a second subject set.

## When you're done

The built-in **Checkpoint to LoRA** utility diffs your fine-tune against the base and
extracts an ordinary, shareable LoRA at any rank — the quality of a full fine-tune, in
a file ComfyUI already knows how to use. It works very well: in our testing, rank 64 was
perceptually indistinguishable from the full checkpoint. Or keep the checkpoint: it's a
valid training base itself, and Pause / Resume works mid-fine-tune with everything
carried over.

## What we'll say, and what we won't

The numbers above are the cold, hard, measured facts — that's the release. What we can
add from our own tests, carefully: **multi-character and concept teaching seemed to land
at a much deeper level than LoRA training, with much better results.** The rest — how far
this actually goes — we're deliberately leaving for you to discover. Field reports
genuinely shape what gets built next.

A personal note on where this is at: I first got fine-tuning working on Krea 2 shortly
after its release, and I've been deliberately cautious about shipping it — first proving
it to myself, then refining it through the MiniMax H3 work. This is the point where it
needs the community to develop further. I don't expect every scenario to work perfectly
yet — but it works, and there's a solid foundation here to build on. I'm also aware this
technique is model-agnostic at heart — it opens the door to fine-tuning other models, and
I'm open to going there. But for that to happen it needs practical community support
around those models, so I have the time necessary to make it happen. — Peter

<!-- sections to add as the release firms up: intelligent-trainer features on FT
     (problem images, adaptive throttle), likeness-mode recommendation, RAM/disk
     requirements, upgrade notes, credits. -->
