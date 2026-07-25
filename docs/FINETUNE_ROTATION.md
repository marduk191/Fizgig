# Rotating-Block Full Fine-Tuning (Krea 2)

Branch: `experiment/ft-rotation` — not merged, not pushed. Master is untouched.

Trains the **Krea 2 base model itself** instead of a LoRA, on a single consumer GPU, by keeping
most of the model fp8-frozen and rotating which slice is trainable. First real runs are done and
the results are strong enough to be worth pursuing properly.

---

## Why

A naive full fine-tune of Krea 2 (12.9B) needs roughly **78 GB** — bf16 weights, gradients and
Adam state all at once. That's rented-A100 territory.

The bet is that removing LoRA's rank bottleneck matters. A LoRA constrains every update to a
low-rank subspace, so concepts compete for the same handful of directions and the result behaves
more like a filter applied over the model's outputs — which is why LoRAs tend to drag pose,
framing and lighting toward the training distribution along with the likeness. Full-rank updates
can change the model's internal representation of a concept, so it composes with what the model
already knows.

---

## How it works

**Rotating windows.** Only part of the model is trainable at a time; the window advances each
epoch. Gradients and optimizer state only ever exist for the active slice, which is what makes it
fit. Over a full cycle every weight is trained. This is LISA (Layerwise Importance Sampled AdamW)
applied to a diffusion DiT rather than an LLM — as far as we know, untried in this setting.

**The bf16 master copy is the critical design decision.** A CPU-resident bf16 copy of every
trainable weight is the source of truth. Blocks activate *from* it and write back *to* it, so the
lossy fp8 copy on the GPU is only ever used for the frozen forward pass. Training never
round-trips through fp8 — which would quantise away exactly the small updates we're trying to
learn. Costs ~24 GB of system RAM.

**Two window modes:**

- **component** (default) — attention across all 28 blocks, then each MLP matrix in turn. Four
  windows per cycle. Every window spans the model's full depth, so a concept is learned by every
  layer at once rather than by one depth slice at a time.
- **block** — contiguous slices of blocks. Fewer windows, but each trains only part of the depth.

Component sizes are balanced by parameter count: a block is 30% attention / 70% MLP, so a naive
`("attn", "mlp")` split makes the MLP window 2.3× the attention one — measured, and it OOM'd.
Splitting the MLP into its three matrices gives four windows of ≤30% each.

**txtfusion is always-on.** It sits outside `dit.blocks`, so rotation would never have reached
it — yet it's the stack that fuses the text embeddings, where prompt-to-concept binding happens.
It's small (3072-wide text path vs 6144-wide main blocks) so it stays trainable throughout.

**Optimizer-in-backward.** Each trainable tensor gets a single-parameter optimizer stepped from a
`post_accumulate_grad_hook`, so a gradient is consumed and freed the moment it lands and the whole
window's gradients never coexist. Worth 5.2 GB. Costs global gradient clipping (impossible when
grads are freed on arrival) and gradient accumulation (nothing left to accumulate).

**Adafactor** rather than AdamW: its factored state is ~10× smaller, which is part of what keeps
this inside 32 GB.

---

## Measured (RTX 5090, 32 GB, 36-image dataset at 0.25 MP)

| Config | Peak VRAM | s/it | Windows/cycle | Time per full cycle |
|---|---|---|---|---|
| 4 blocks resident | 24.8 GB | 0.93 | 7 | 234 s |
| 8 blocks resident | 24.2 GB | 1.01 | 4 | 145 s |
| 14 blocks resident | 27.5 GB | 1.16 | 2 | 84 s |
| 18 blocks resident | 29.5 GB | — | 2 | — |
| **component (default)** | **30.1 GB** | **1.03** | **4** | **148 s** |
| 12 blocks + streaming | 26.5 GB | 2.99 | 3 | 323 s |
| 18 blocks + streaming | 29.5 GB | 2.64 | 2 | 190 s |

**Fine-tuning is faster per step than LoRA training.** Counterintuitive, but: the trainable window
holds real bf16 weights, so it skips the fp8 dequantisation that materialises a bf16 copy of every
weight matrix on every forward; there are no LoRA down/up matmuls (528 extra small, latency-bound
kernels per forward plus their backward); and the optimizer steps ~65 large tensors instead of
~528 small ones. Reasoned from the code, not profiled.

---

## Block streaming — built, and not worth it at 32 GB

`RotationOffloader` pins the trainable window on GPU and streams every other block from CPU
just-in-time, with async prefetch on a dedicated CUDA stream. The stock offloader can't express
this: it keeps a fixed contiguous *prefix* resident and pairs each eviction with a matching load,
whereas rotation needs an arbitrary resident set that *moves*.

It works, and it was built on a wrong assumption. I expected the fp8 base to be the binding
constraint; optimizer-in-backward freed enough that it never was. Simply raising the resident
block count wins outright — 14 resident covers the model in 2 windows at 84 s/cycle, versus
323 s/cycle for 12 blocks with streaming.

**It stays as opt-in** (`--blocks_to_swap`), because it's the right tool for 24 GB and 16 GB
cards where residency *is* the constraint. Just not the default here.

Gradient checkpointing recomputes a block's forward *inside* its backward, so blocks are pulled
back on a backward **pre**-hook — evicting after the forward alone leaves the recompute pointing
at CPU weights.

---

## Results so far

First overnight run (single subject, real name as the identity token, detailed captions, no
regularisation images):

- Quality reported as **very good**, and notably **better generalisation than the equivalent
  LoRA** — which is the predicted symptom of removing the rank bottleneck rather than a lucky run.
- **Limited class drift** despite no regularisation set, presumably because a distinctive name
  absorbed the identity, detailed captions gave everything else somewhere to live, and 40 epochs
  at 1e-5 is gentle.

Not yet established: which checkpoint peaked, whether 1024 generation shows detail softening from
training at 0.25 MP, and a matched A/B against a LoRA.

---

## Usage

**GUI** — Training tab, Krea 2 only: tick **⚗ Fine-tune the BASE MODEL instead of training a
LoRA**. That applies the whole recipe in one go and prints what it changed:

| Setting | Value | Why |
|---|---|---|
| Learning rate | 1e-5 | LoRA rates (1e-4+) destroy a base model |
| Max train epochs | 40 | 10 full cycles |
| Save every N epochs | 4 | one per cycle, so checkpoints compare like-for-like |
| Window mode | component | every window spans full depth |
| Free gradients | on | ~5 GB |
| Adaptive LR | off | rotation boundaries read as instability |
| LR scheduler | constant | |
| Grad accumulation / clip | 1 / 0 | required by fused backward |

Checkpoints go to the **Output Directory** — each is a full ~26 GB model, so point it somewhere
with room (a ComfyUI `models/unet` folder is the natural home). Test them as ordinary Krea 2
models.

**CLI:**

```bash
python src/fizgig/scripts/krea2_train.py \
  --dataset_config my_dataset.toml \
  --dit /models/Krea-2-raw.safetensors \
  --output_dir /big-drive/ft_run --output_name my_subject \
  --learning_rate 1e-5 --max_train_epochs 40 --save_every_n_epochs 4 \
  --finetune_rotation 14 --finetune_rotation_mode component \
  --finetune_fused_backward
```

`--finetune_rotation` is the master switch (any value > 0). In component mode the number is
ignored; in block mode it's the blocks per window.

---

## What is and isn't compatible

**Works:** the per-image loss watch and per-image adaptive LR (loss scaling happens before
backward, upstream of where fused backward consumes gradients), auto-recaption, look-outlier
warm-up. Verified: 36/36 images tracked with real verdicts under a component-mode run.

**Auto-disabled, with a log line:** in-training previews (they apply a LoRA to the Turbo; there is
no LoRA here), adaptive LR, block swap unless explicitly asked for, 4-bit.

**Caveat on the loss watch.** It was designed against a model of constant capacity. Under rotation
the trainable set changes every epoch, so loss carries a 4-epoch periodic component — an
attention epoch and an MLP epoch don't fit equally well (observed: 0.0796 → 0.0775 → 0.0948 →
0.0760, where the spike is `mlp.up`, not the data). Residuals are computed relative to
per-timestep-bucket means so a uniform shift largely cancels, which is why verdicts look sane —
but trust a `stuck` flag that persists across a full cycle over one that appears and vanishes
within four epochs. This is exactly why adaptive LR is disabled: it reads epoch-to-epoch loss
directly and would have called that spike instability and rolled weights back.

---

## Continuing a run

**Do not use Pause/Resume.** The state directory saves the LoRA network and optimizer, but in
fine-tune mode the LoRA is inert and all the training lives in the base weights, which aren't in
there. Resuming would silently restart from the original RAW model.

**Instead, swap the base.** The output is a complete, structurally identical Krea 2 checkpoint
(all 430 keys, same names, bf16, untrained tensors byte-identical to RAW), so point Preferences →
Krea 2 RAW DiT at a checkpoint and train again. Latent and text caches stay valid — they depend on
the VAE and text encoder, not the DiT. Use a new output name, and drop the LR (5e-6) for a
refinement pass.

*Known gap: `--resume` should refuse to run in fine-tune mode rather than silently doing the wrong
thing.*

---

## Captioning for a base-model fine-tune

Different from LoRA practice, and it matters more:

- **Use a distinctive real name, not a rare token.** The method can genuinely rebind a name, and
  a name carries useful semantic scaffolding. Trigger goes **first** now, in both the Captions tab
  and auto-recaption (they disagreed before this branch).
- **Never train into a generic class word.** Using `woman` as the trigger would overwrite the
  model's concept of "woman" globally, with no strength dial to back it off.
- **Caption what varies, omit what's constant about the subject.** Clothing, background, lighting,
  pose, viewpoint, framing — yes. Facial structure and eye colour — no: caption them and they bind
  to those words rather than to the name.
- **Regularisation images are the proper fix for class drift** (~50% of dataset size, generated by
  the base model itself so you anchor its own prior rather than teaching new content, captioned
  with the class only). Not yet supported in the GUI; possible today by hand-editing a second
  `[[datasets]]` block into the TOML with its own `cache_directory`.

---

## Next

1. **SVD the delta into a LoRA.** Now the interesting question rather than a convenience: if the
   fine-tune generalises better *because* it's full-rank, does a rank-16 projection keep that or
   throw away the thing that made it good? Either answer is worth knowing — and if it survives,
   it's fine-tune quality in a shareable 100 MB file.
2. **Regularisation-images support** in the GUI (second dataset block + ratio).
3. **ReLoRA** — train a rank-16 LoRA, merge it into the bf16 master, reset, repeat. Accumulates a
   high effective rank at LoRA memory cost (~16-18 GB, so 24 GB cards and maybe 16 GB). Merging
   into the bf16 master rather than the fp8 base is what makes it viable. The most promising route
   to giving smaller cards most of what this branch achieves.
4. **Make the loss watch rotation-aware** — compare residuals against the same window position
   rather than the previous epoch. Only if the cyclic noise proves to matter.
5. **Guard `--resume`** in fine-tune mode.

---

## Commits

| | |
|---|---|
| `42b0825` | rotation mechanics (schedule, block rotator, bf16 master) |
| `44f56a7` | fp8 fix: scale_weight created on-device in a compute dtype |
| `4dec740` | wire rotation into the trainer |
| `2a0225d` | txtfusion always-on |
| `0b00583` | optimizer-in-backward (−5.2 GB) |
| `04dc0ec` | rotation-aware block swap |
| `7827fa3` | async prefetch (4.55 → 2.99 s/it) |
| `5d8430d` | component (sub-block) rotation |
| `740184b` `03fa82c` `a16f4b9` `74a6f92` | GUI controls, one-click recipe, defaults |
| `9d0c0ad` | trigger word leads the caption |
| `ab3cca2` | fix: fine-tune never engaged (read Tk vars, not settings) |
| `61f6cda` | fix: resolve fine-tune mode before preview setup |

Two bugs shared a root cause worth remembering: **setup running before the decision that
invalidates it**, and **reading a flag from a dict that is only populated elsewhere**. Both were
invisible in isolation and only surfaced when the feature was exercised the way a user actually
reaches it — a headless test that set `self.settings` by hand passed while the real path was
broken.
