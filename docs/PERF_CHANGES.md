# What changed, and why — the OneTrainer speed investigation

Branch `perf/benchmark-and-backends`, 30 commits. This is the change log; `PERF_ROADMAP.md` has the
detailed measurements and the things that were tried and rejected.

It started with a community thread claiming Fizgig trained Krea 2 more slowly than OneTrainer and
pegged the CPU (70–90% vs <5%). Both claims were true, and neither was caused by a missing kernel.

All numbers: RTX 5090, torch 2.10.0+cu128, 36 images at 0.25 MP, batch 1, measured with
`scripts/bench_train.py` on an otherwise idle GPU.

---

## Where it ended up

Krea 2, matched settings (INT8 W8A8, LoRA rank 16 / alpha 1, batch 1, 512px, 3 epochs, no offload):

| | s/step |
|---|---|
| Fizgig at the start of the day, 16 GB card | 3.09 |
| Fizgig at the start of the day, 32 GB card | 0.85 |
| **Fizgig now, default optimizer (`adamw8bit`)** | **0.306** |
| Fizgig now, `adamw` fused | 0.292 |
| Fizgig now, `adamw` unfused (matches OneTrainer's implementation) | 0.333 |
| OneTrainer, their preset minus its offloading default | 0.294 |

Read that honestly three ways, because all three are true:

- **At each side's own sensible defaults, we are within ~4%** (0.306 vs 0.294).
- **Matched exactly on optimizer implementation, OneTrainer leads ~13%** (0.333 vs 0.294). Our
  `adamw` defaults to CUDA-fused; theirs ships `fused: False`. Fused is worth 12.3% and they could
  enable it too.
- **With our fastest optimizer it is a wash** (0.292 vs 0.294).

The first framing is the one users experience. The second is the one an engineer should quote.

---

## The changes

### 1. Block swap was the whole 16 GB story
`_auto_krea2_blocks_swap` chose a swap count from VRAM, which handed 16 GB cards the worst possible
configuration: fp8 does not fit, so it swapped 20 of 28 blocks to CPU every step.

    fp8, no swap   0.85 s/it   20.1 GB   12.5% CPU
    fp8, swap 20   3.09 s/it   12.3 GB   49.9% CPU   <- what 16 GB cards got
    NF4, no swap   0.70 s/it   13.8 GB   14.0% CPU

Replaced by `utils/capabilities.py`, which probes the machine by running a real matmul (not by
consulting an sm-version table) and picks a *strategy*: INT8-no-swap > NF4-no-swap > fp8-no-swap >
swapping. It budgets from **free** VRAM, not the number on the box.

This is also the origin of the CPU complaint: swapping was the 4× CPU load.

### 2. The INT8 fast path was doing nothing
`modules/int8.py` stored weights pre-transposed as (K,N) with a comment explaining that this avoided
a per-call transpose. Backwards: `.t()` on a contiguous (N,K) tensor is a free view, and that layout
is what int8 tensor cores want — 0.131 ms vs 0.452 ms. "INT8 fast inference" is on by default and
used by Royale, Repair Studio and the Explorer, so users were paying int8's quantisation error for
no speed at all. Now 2.10× vs bf16.

### 3. INT8 training, and a row-count bug inside it
`--quant_int8 bf16|int8` for the frozen base. Faster *and* more accurate than the NF4 it replaces
(forward error 1.3e-02 vs 9.2e-02), at ~5 GB more, so it is auto-selected above ~17.7 GB free.

While chasing something else: `torch._int_mm` refuses M ≤ 16 and the code silently fell back to an
fp32 matmul, so the same weights gave different answers either side of 16 tokens. Now zero-pads the
rows and slices back — verified bit-identical at M = 1, 5, 8, 16, 17, 32.

### 4. A device sync inside every block
The attention trim decided whether it could shorten the sequence by reading `seqlens[0].item()` — a
CUDA→CPU sync — **per block**: 28 stalls per forward, 56 per step with gradient checkpointing.
Resolved once per forward now, in `AttentionParams.__post_init__` (Krea 2 and Klein).

### 5. Sequence shapes: 36 distinct → 12
The DiT pads the sequence to a multiple of 256 *explicitly to keep kernel shapes stable*, and the
trim immediately undid it, because the trimmed length carries each caption's own token count. Every
shape-planning backend (cuDNN, torch.compile) pays a one-time cost per distinct shape, and that cost
is flat in shape size — a 17-token shape costs as much to plan as a 1097-token one.

Attention now rounds the trim up to a multiple of 64 and masks the slack.
**Known perturbation, accepted deliberately:** short captions (≤ ~30 tokens) shift ~5e-03 relative on
bf16 and ~2e-02 under INT8. Not masking (padding the combined sequence instead is bit-exact) and not
INT8 (bf16 shows it too). The cause is still unknown. `FIZGIG_ATTN_TRIM_MULTIPLE=1` restores exact
lengths.

### 6. Attention backend, chosen per situation
cuDNN's SDPA is ~6% faster per step once warm but costs ~1.3 s per distinct shape to plan. Inference
holds one shape for a whole render, so it always wins there:

    Krea 2 1024 preview   5.15 s -> 4.40 s
    Klein  1024 preview   3.01 s -> 2.74 s

For training, `consider_training_backend()` decides at an epoch boundary — by then every shape has
been seen, so it is arithmetic rather than a guess. Verified switching mid-run: 0.611 s/step on the
default backend, then 0.569 warm on cuDNN.

### 7. torch.compile — 2.0× on INT8, and it auto-enables
`--compile_blocks auto|on|off` (Training tab: Auto / On / Off). Three things had to be fixed:

1. **`torch._dynamo.config.cache_size_limit` defaults to 8.** Past that, dynamo silently falls back
   to eager *permanently*, with nothing logged. A bucketed dataset exhausts 8 immediately, which is
   why the first attempt measured slower than no compile at all. Now 8192.
2. **A torch bug aborts inductor mid-run**: `Mod.eval` asserts on negative operands and inductor's
   tiling feeds it negative test values, giving `InductorError: AssertionError:
   -111500631004807/2000000000000000` and then a segfault. `modules/compile_util.py` wraps torch's
   own function and answers only the case it wrongly refuses.
3. **The gradient checkpoint sat outside the compiled region.** Moving it inside (`_CheckpointedBlock`)
   is worth 1.14× end to end.

Auto weighs the ~90 s warm-up against the run's real length (~600 steps to repay on INT8, ~1200 on
NF4, doubled for margin) and declines under block swap, without triton, or when INT8 + compile would
not fit.

**NF4 + compile fits a 16 GB card** — verified under a hard 13.5 GB cap (a 16 GB card minus ~2.4 GB
for Windows), zero OOM, 0.71 → 0.56 s/step. INT8 + compile needs ~24 GB.

### 8. Optimizer choice for Krea 2
Was hardcoded to AdamW8bit. Now seven families filtered to what is installed, plus any
`module.path.ClassName`, via `--optimizer_type` / `--optimizer_args`. See the safety notes below.

---

## What was measured and deliberately NOT shipped

- **Bucket-grouped batch ordering** (OneTrainer's `AspectBatchSorting`): no difference (0.7042 both).
  Behind `FIZGIG_BUCKET_ORDER=1`; off, because grouping correlates consecutive gradients.
- **NVFP4**: 4-bit and fast, but *less* accurate than NF4 (1.03e-01 vs 9.23e-02) — NF4's codebook is
  Gaussian-optimal, NVFP4's e2m1 is not. Parked.
- **fp8 `_scaled_mm`**: 4.2× on the matmul but needs fp8 activations; NF4 and INT8 both beat it.
- **`cudnn.benchmark = True`**: no measurable effect at all; the flag is no longer set.

---

## What is still open

- **No trained-LoRA validation for any of it.** INT8 (default above ~17.7 GB), int8 gradients, cuDNN,
  compile and the attention-trim perturbation are all speed-for-accuracy trades measured only as
  throughput and per-tensor error. The int8-vs-NF4 A/B has not been run.
- **The ~2% short-caption perturbation** from shape rounding is unexplained.
- **The remaining ~13%** against OneTrainer at matched optimizer settings is unexplained. Ruled out:
  the INT8 scaling formulation (identical once compiled). Still possible: torch 2.12 vs our 2.10.
- **Optimizer coverage**: see below.

---

## Optimizer safety

The plumbing is safe — availability-filtered, construction failure falls back to AdamW with a
warning, each family smoke-tested to construct and step a real parameter. The *choices* are not all
validated:

| optimizer | run end to end |
|---|---|
| `adamw8bit` (default) | extensively — the validated recipe |
| `adamw` | several 3-epoch runs |
| `lion8bit`, `ademamix8bit` | one epoch each |
| `pagedadamw8bit`, `pagedademamix8bit`, `adafactor` | never |
| `prodigy`, `came` | not installed here |

Three hazards worth knowing:

- **Learning rates do not transfer.** Lion applies the sign of the update and wants ~1/10 an AdamW
  LR; Prodigy estimates its own and wants `lr=1.0`. Currently a console warning, which is easy to
  miss — a GUI dialog would be better.
- **`adafactor` means different things per family.** Klein routes to `transformers.Adafactor` with
  `relative_step` handling and a dedicated scheduler; Krea 2 uses `torch.optim.Adafactor` and passes
  the LR straight through. Same name, different behaviour.
- **AdEMAMix's defaults are tuned for long runs**; on a short LoRA run it may simply underperform.
