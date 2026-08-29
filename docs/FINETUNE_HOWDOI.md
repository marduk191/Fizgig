# Full fine-tuning — "How do I…?"

The extended companion to the README's fine-tuning section, for **Krea 2** and
**MiniMax H3**. Everything here is measured or field-tested, not aspirational. If you're
new to fine-tuning and coming from LoRA training, read the first four answers in order —
they're the ones that save you a wasted overnight run.

---

## How do I start a fine-tune?

Tick **⚗ Fine-tune the BASE MODEL instead of training a LoRA** on the Training tab, leave
**Window** on **Auto (by VRAM)**, and press Start. That's genuinely it — the planner
measures the memory actually free on your card at launch, picks the plan that fits, and
prints what it chose and why. Everything else on this page is about making the run *good*
rather than making it *go*.

Before your first run, do two things:

1. **Set the Output Directory to a drive with room** (Training tab). Every save is a full
   checkpoint — ~26 GB on Krea 2, ~21 GB on H3 — and the box defaults to the same folder
   your LoRAs go to. Change it before you press Start; afterwards the only fix is moving
   20 GB files by hand.
2. **Check your learning rate** (next answer). This is the single most common way to ruin
   a fine-tune.

## What learning rate should I use?

**Lower than you're used to.** A LoRA nudges a small adapter riding on a frozen model; a
fine-tune moves the model's own weights. The rates you know from LoRA training land very
differently here.

- **MiniMax H3: 3e-5.** Tested, fast, reliable. **1e-4 will destroy an H3 fine-tune** —
  that's measured, not folklore.
- **Krea 2: 1e-5 is the safe recommendation.** You're welcome to *start experimenting* at
  1e-4 — it trains — but realistically the best results are found lower. Treat 1e-4 as
  the top of the experiment range, not the recipe. When a run looks almost right but
  slightly overcooked, the next move is a lower rate, not fewer epochs.

If you use regularisation images, their **LR ×** multiplier is part of the same tuning
space — see the regularisation answer below.

## What card do I need?

At the default training resolution:

| Your card | Krea 2 — photos | H3 — photos | H3 — voice | H3 — video clips (with sound) |
|---|---|---|---|---|
| **16 GB** | ✅ | ✅ | ✅ | ✅ up to **2.3 s** |
| **24 GB** | ✅ | ✅ | ✅ | ✅ up to **2.3 s** |
| **32 GB** | ✅ | ✅ | ✅ | ✅ up to **2.3 s** |

12 GB cards train **LoRAs**, not fine-tunes — 16 GB is the fine-tune floor. Every cell in
that table comes from a measured run.

## How does it fit on my card at all?

A naive full fine-tune of Krea 2 needs ~78 GB. Three things close the gap: **rotating
windows** (only one slice of the model is trainable at a time — over a full cycle every
weight trains, but gradients and optimizer state only ever exist for the active slice), a
**4-bit NF4 frozen base** (the fine-tune default — the frozen part of the model is held
at half size while the trainable window runs bf16), and a **CPU-resident bf16 master
copy** that is the source of truth, so your saved checkpoint never passes through a
quantiser regardless of what the card holds.

On 32 GB and 24 GB the full-depth windows stay resident at full speed (~1.0 s/it). On
16 GB the frozen blocks stream from system RAM — slower steps, same learning. The console
always prints the plan.

## How long should I train?

**At least one full cycle, or some weights never train at all.** A cycle is one pass of
the rotation — 4 epochs in the standard component plan, more when your card's plan splits
windows (the console prints your cycle length at launch).

On large datasets, budget generously: fine-tuning can take **~30× the steps you're used
to from LoRA runs**, because only one component of the model trains at a time where a
LoRA trains all of its weights every step. Checkpoint saves snap to cycle boundaries so
every save compares like-for-like.

## How do I fine-tune on video clips? (H3)

1. **Cut your clips with Gizmo** — it produces exactly the spec H3 trains on (24 fps, the
   right frame counts) from any footage.
2. **Keep clips at or under the 56-frame / 2.3 s slot.** That length trains on every card
   tier down to 16 GB — measured. Longer clips (3.8 s / 5.2 s) currently need more than
   32 GB; that ceiling is activation memory, not the model.
3. Mind the mix: **one clip anywhere in your folder trains the whole model** (that's what
   video needs), so a mixed photos + clips dataset runs at the clip tiers.

If your clips are too long for your card, the trainer tells you **before training
starts** — with the fix (cut to the 2.3 s slot, or lower Target Megapixels) — instead of
failing mid-run. Clips train with their sound automatically when the audio VAE is set.

## How do I fine-tune voices? (H3)

Drop voice recordings (or record a dataset with Gizmo's built-in recorder) into the
training folder like any other item. Voice stays confined to its measured blocks (34–49)
automatically, so audio gradients never touch the visual blocks. Mixed datasets work:
photos, clips and voice in one folder train one fine-tune in one run, each modality
confined to its zone.

The per-category **stop epoch** counts across Pause/Resume — pause a mixed run, set the
visual stop to the current epoch, Resume, and it finishes voice-only.

## Should I leave Optimised Likeness Learning on? (H3)

**Yes.** Matched runs came out clearly better on both look and prompt adherence than
full-model fine-tuning, and it shrinks the windows so every VRAM tier gets easier. The
exception is video: clips train the whole model regardless, so the checkbox mainly
matters for photo/voice datasets.

## What are regularisation images, and what does LR × do?

Full fine-tuning moves every weight, so a long run on a handful of subjects can drift the
model's whole notion of people — there's no low-rank bound to stop it like a LoRA has.
Point **Regularisation images** at a folder of ordinary **real photos** of the broader
class (not model output — anchoring a fine-tune to its own samples distils its artifacts
back in), captioned normally.

They train at the **LR ×** multiplier beside the folder box, and that multiplier is a
real dial worth experimenting with per dataset:

- **0.1–0.3** (default 0.2): a tether — the reg set nudges the model's prior back while
  your subject trains.
- **Toward 1.0**: the reg set trains like a second subject set — class-balanced training
  rather than a light anchor. A different, valid thing.
- Class drifting toward your subject? **Raise it a step.** Subject learning too slowly?
  **Lower it.**

## Where do my checkpoints go, and how much disk will this eat?

Checkpoints land in the Training tab's **Output Directory** — set it to a roomy drive
*before* the run. Sizes: **~26 GB per Krea 2 save, ~21 GB per H3 save**. Saving once per
4-epoch cycle over a 40-epoch run is **~260 GB**. A Pause writes an extra full checkpoint
on top of the regular cadence. Delete the intermediate saves you don't need once you've
picked your best epoch — each file is a complete, standalone checkpoint.

## How do I pause and resume?

Press **Pause Training**: the trainer finishes the epoch, writes a full checkpoint (even
between the regular save epochs), and exits cleanly. **Resume** continues it — rotation
window, checkpoint numbering and the remaining epoch count all carry over, so a resumed
run never overwrites an earlier save and the cycle stays evenly trained.

## How do I keep training a finished fine-tune?

Every save prints the exact continuation settings in the console — point the model path
(**--dit**) at your checkpoint and use the printed start-window. A finished fine-tune
checkpoint is a valid training base for its family, so you can chain runs indefinitely.

## How do I see previews during training?

Previews **ride the checkpoint saves** — one render per save plus the final one, each
sample the rehearsal of a checkpoint you can actually deploy. You need three things set:
**sample prompts** (Samples tab), a **sample cadence** (Sample every N epochs, or
sample-at-first), and on Krea 2 the **Turbo LoRA** (the standalone Turbo checkpoint can't
show fine-tuned weights, so previews render on the training DiT with the Turbo LoRA
applied fresh). If prompts are set but previews stay off, the console now tells you
exactly which ingredient is missing.

## How do I share a fine-tune? It's 26 GB!

Run **Checkpoint to LoRA** (`run_diff_to_lora.bat`) — it diffs your fine-tune against the
base model and extracts the difference as an ordinary kohya `.safetensors` at several
ranks at once. Measured on a three-subject fine-tune: **rank 64 was perceptually
indistinguishable from the full checkpoint** at ~0.5 GB, and even rank 8 kept the
identities cleanly separate. The result loads anywhere a normal LoRA does, ComfyUI
included.

## NF4 or fp8 Base precision?

**Leave it on the default (NF4).** It's what makes 16 GB fit at all, keeps 24 GB at full
speed, and its quality is field-proven — your saved checkpoint is written in bf16 from
the master copy either way, so the choice only affects the frozen context the trainable
window learns against. Pick **fp8** only if you have the VRAM to spare and want the more
accurate frozen context: on a 24 GB card it costs you depth-split, streamed windows at
~3× the step time, and it doesn't fit 16 GB at all.

## The trainer refused to start, or I hit CUDA out of memory — now what?

If it **refused up front**, read the message — it names the actual problem and the fix
(usually: close other GPU apps, shorten clips to the 2.3 s slot, or lower Target
Megapixels). A refusal is the trainer doing its job: it beats an OOM three hours in.

If you hit a genuine **mid-run OOM**, the usual suspects in order: something else grabbed
VRAM after launch (browsers on hardware acceleration are the classic), clips longer than
your card's tier, or Target Megapixels above the default. And know the Windows quirk:
when **system RAM** runs out, the error still arrives dressed as "CUDA out of memory"
with the GPU nearly empty — close other apps and retry before suspecting VRAM.

## What model files do I need?

The same training bases you already have — nothing new to download:

- **Krea 2** fine-tunes the **RAW bf16 model** (`krea2_raw_bf16.safetensors`, ~26 GB).
  The fp8 Turbo is the preview model and can't be fine-tuned.
- **MiniMax H3** fine-tunes the **pruned int8 checkpoint**
  (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`, ~21 GB) — the same file ComfyUI
  runs. The ~66 GB bf16 file works for LoRA training only; the trainer refuses it for
  fine-tuning with a clear message.

Plus system RAM for the bf16 master copy: ~24 GB on Krea 2, ~23–38 GB on H3 (H3's spills
to disk automatically when RAM is tight).

## Why fine-tune instead of training a LoRA?

A LoRA constrains every update to a low-rank subspace, so concepts compete for the same
handful of directions — which is why LoRAs tend to drag pose, framing and lighting toward
the training set along with the likeness. A full-rank update can change how the model
*represents* a concept, so it composes with what the model already knows. And with
Checkpoint to LoRA at the end, you don't have to choose between fine-tune quality and a
shareable file.
