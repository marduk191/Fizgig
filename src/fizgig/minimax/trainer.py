"""MiniMax H3 — image-only training core: flow-matching loss + timestep sampling.

The heart of the trainer, isolated so it's headless-testable with the tiny model (no GPU,
no 66 GB base, no 32 B text encoder). The full LoRA/rotating-FT wiring, caching and GUI come
later; this pins the maths of one training step.

Flow / sign convention (matched to ComfyUI's comfy/ldm/minimax/model.py):
  x0 = clean latent, noise ~ N(0,1), sigma in (0,1) the noise level.
  noised = (1 - sigma)*x0 + sigma*noise            (sigma 0 = clean, 1 = pure noise)
  t = 1 - sigma                                     the "cleanness" fed to the time embedder
  the DiT's raw video_out predicts (x0 - noise)     (the reference NEGATES it to get the
                                                     sampler's velocity noise - x0)
So the training target for the model's output is `x0 - noise`.
"""

import argparse
import contextlib
import gc
import logging
import math
import os
import random
import re
import sys
import time
from multiprocessing import Value

import torch
import torch.nn.functional as F

from fizgig.training.metadata import ARCHITECTURE_MINIMAX

logger = logging.getLogger(__name__)

VIDEO_SIGMA_SHIFT_TRAIN = 12.0     # H3's video shift — also the reference TRAINING density

# LoRA targets the transformer blocks' ATTENTION + MLP Linears (+ the 2-block text refiner).
# The fp32 patch/head IO layers are left alone (wrapping them clashes fp32-base vs bf16-adapter).
#
# `adaln_proj` is per-checkpoint (matching the reference trainer on the pruned build):
#   * FULL bf16 model ([96768, 2688]): EXCLUDED — the up-matrices are 96768-out (6x qkv),
#     soaked up the largest share of LoRA capacity, and ComfyUI's pruned inference builds
#     drop every adaln key anyway (~50% likeness until excluded, real run).
#   * PRUNED model ([96768, 8]): INCLUDED — deploy-consistent, and what ai-toolkit trains.
#     It carries ~45% of all weight movement in a matched reference epoch, and it is the
#     timestep-conditioned modulation, so starving it reads from outside as "the mid/low-noise
#     range never gets trained". Train it at the REQUESTED rank: capping to min(in,out)=8 cost
#     73% of its learning (see the no-cap note in networks/lora.py). An epoch-1 melt was once
#     pinned on these adapters (tests/diag_epoch1_ab.py) but the distortion predated adaln and
#     persisted without it — the real culprit was the training density (see sample_sigmas).
DEFAULT_INCLUDE_PATTERNS = [r"blocks\.\d+\.attn\..*", r"blocks\.\d+\.mlp\..*",
                            r"token_refiner\.blocks\..*"]
# NOTE: the per-block AdaLNs only — NOT `final_layer.adaln_proj`. The reference trains 258
# modules and we were training 259; the extra one was added here by symmetry, not by matching
# them. It also happened to carry our single highest per-element drift after a matched epoch
# (0.0133 vs their 0.0068 max), so it was contributing noise rather than capability.
PRUNED_INCLUDE_PATTERNS = DEFAULT_INCLUDE_PATTERNS + [r"blocks\.\d+\.adaln_proj\..*"]


def parse_block_spec(spec, num_blocks: int = None):
    """"3-12, 14-15, 22,27,31-33" -> [3,4,...,12,14,15,22,27,31,32,33].

    Ranges and singles, comma-separated, whitespace anywhere. Returns sorted unique indices.
    Raises ValueError on anything it cannot read — a typo here must stop the run, not silently
    train a different set of blocks than the one being tested.

    num_blocks, when given, bounds-checks: an out-of-range index would otherwise just match
    nothing and quietly shrink the experiment.
    """
    text = str(spec if spec is not None else "").strip()
    if not text:
        raise ValueError("no blocks given")
    out = set()
    for part in text.split(","):
        chunk = part.strip()
        if not chunk:
            continue                       # tolerate a trailing or doubled comma
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", chunk)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                raise ValueError(f"range runs backwards: {chunk!r}")
            out.update(range(lo, hi + 1))
        elif re.fullmatch(r"\d+", chunk):
            out.add(int(chunk))
        else:
            raise ValueError(f"cannot read {chunk!r} — use numbers and ranges, "
                             f"e.g. '3-12, 14-15, 22, 31-33'")
    if not out:
        raise ValueError("no blocks given")
    if num_blocks is not None:
        bad = sorted(i for i in out if i >= num_blocks)
        if bad:
            raise ValueError(f"block(s) {bad} do not exist — this model has {num_blocks} "
                             f"(0-{num_blocks - 1})")
    return sorted(out)


def format_block_spec(indices):
    """[3,4,5,7] -> "3-5,7" — the canonical form recorded in metadata and logged."""
    if not indices:
        return ""
    runs, start, prev = [], indices[0], indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    runs.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def restrict_patterns_to_blocks(patterns, block_spec, num_blocks: int = None):
    """Narrow `blocks.N.*` patterns to a block selection. Non-block patterns pass through.

    H3 is 50 IDENTICAL blocks with no published map of what each one does, so training a subset is
    an experiment, not a recipe — this exists to make that experiment cheap to run. The token
    refiner is deliberately never narrowed: it is text-side (where a trigger token gets shaped),
    it is 8 of 258 modules, and holding it constant keeps two selections comparable to each other
    rather than confounding the block question with a conditioning change.

    Applied ON TOP of the per-checkpoint pattern list rather than replacing it, so the pruned vs
    bf16 AdaLN decision stays in exactly one place.
    """
    idx = parse_block_spec(block_spec, num_blocks)
    alt = "|".join(str(i) for i in idx)
    out = []
    for p in patterns:
        if p.startswith(r"blocks\.\d+"):
            out.append(p.replace(r"blocks\.\d+", rf"blocks\.(?:{alt})", 1))
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# VRAM planner — resolves "auto" block swap + gradient checkpointing from the card's actual
# free VRAM and the run's real token load (bucket megapixels x batch). Simpler than Krea 2's:
# one quant mode (NF4), batch is 1, no preview co-residency.
# ---------------------------------------------------------------------------
# Measured anchors (5090, real 33B, rank 16, ~0.2 MP batch 1 — GPU validation pass, 4 Aug):
#   no swap, no ckpt : resident 17.6, step peak 22.7  (overhead ~5.1)
#   no swap, ckpt    : resident 17.5, step peak 18.3  (overhead 0.9 — and only ~+0.1 s/step)
#   swap 16 + ckpt   : resident 11.9 (0.34 GB/block), steady 12.8, step peak 19.3 — the swap
#                      path carries a ~7.4 GB backward transient (checkpoint recompute segments
#                      held by the engine), which the planner must budget on top of residency.
#
# Re-measured 6 Aug on the SHIPPED default (int8 base, LoKR factor 8 + adamw, AdaLN off), because
# those anchors were taken with a rank-16 LoRA on adamw8bit — an adapter of ~0.4 GB against the
# ~3.1 GB the defaults now carry, so the planner was budgeting for a run nobody does:
#   resident         : base 21.07 + LoKR weights 0.63 + fp32 Adam state 2.50 = 24.20 GB
#   0.23 MP  no ckpt : 29.18      |  ckpt: 24.39
#   0.50 MP  no ckpt : OOM (>31)  |  ckpt: 24.47
#   0.98 MP  no ckpt : OOM        |  ckpt: 24.56
# Two things fall out. Un-checkpointed really does scale hard (0.5 MP OOMs a 32 GB card, so
# forcing ckpt on there is correct), and CHECKPOINTED IS ALMOST FLAT — 1 MP costs 0.17 GB more
# than 0.23 MP, not four times as much. Hence _ACT_GB_CKPT below.
_RESIDENT_GB = 17.5          # full bf16 model, NF4 resident (measured 17.3-17.6)
# The PRUNED checkpoint drops the full-width AdaLN (~40% of the model's weight mass) for a curve
# table, so the same NF4 pass lands far smaller: ~20.1 B params quantized -> ~10.1 GB, plus the
# unquantized remainder. Estimated from the file's own tensor census, not yet GPU-measured, so
# it carries margin.
# MEASURED 6 Aug (was 11.0, estimated from the file's tensor census): the pruned checkpoint
# decoded and re-quantized to NF4 sits at 10.46 GB resident, and a checkpointed step peaks at
# 13.46 / 13.56 / 13.63 GB at 0.23 / 0.50 / 0.98 MP — flat in megapixels, exactly like int8.
# Un-checkpointed it is 18.27 / 23.52 / OOM. Now that Auto can CHOOSE this mode, the number it
# chooses against had to stop being a guess.
_RESIDENT_PRUNED_GB = 10.5
# int8 base (base_quant=int8, the reference's own storage): the 200 block linears stay 1 byte
# per param instead of NF4's 0.5, and the refiner/AdaLN load dense — ~19.3 + ~1.5 GB.
_RESIDENT_INT8_GB = 21.0
# int8 dequantizes a bf16 weight per matmul (fc1 is 28672x5376 = 308 MB). A few are live at
# once, but they are NOT retained for backward — _Int8RotLinearFn recomputes the weight in its
# own backward, so the cost is a handful of transients rather than one per layer. (Before that
# custom backward, autograd saved every one and a 0.25 MP run OOM'd the moment the planner
# turned checkpointing off: measured 0.45 GB of retained weight over 12 test linears against
# 0.12 GB now, and the real DiT has 200.)
_INT8_TRANSIENT_GB = 1.0
_PER_BLOCK_GB = 0.34         # one parked block's GPU share (measured: (17.5-11.9)/16)
_ACT_GB_NOCKPT = 5.5         # step overhead at 0.25 MP batch 1, no checkpointing (measured 4.98)
# Checkpointed memory is very nearly FLAT in megapixels — that is the whole point of recompute,
# and the old 2.0 (which then got multiplied by the MP scale) modelled it as growing four times
# faster than it does. Measured on the shipped default (int8 base, LoKR 8 + adamw, 6 Aug 2026),
# peak above the resident 24.20 GB:
#     0.23 MP  0.19 GB        0.50 MP  0.27 GB        0.98 MP  0.36 GB
# i.e. ~0.15 + 0.2 x scale. 0.5 keeps a wide margin at every size and still leaves the planner
# free to say "no swap" where the card genuinely fits — the old value invented 25 blocks of swap
# for a 1 MP run that actually peaks at 24.6 GB, costing ~4x the step time for nothing.
_ACT_GB_CKPT = 0.5           # step overhead at 0.25 MP batch 1, checkpointed (measured 0.19)
_SWAP_TRANSIENT_GB = 7.5     # extra backward-time peak whenever swap is active (measured 7.4 @ n=16)
_RESERVE_GB = 1.5            # display / allocator / fragmentation headroom
# Skipping checkpointing has to EARN it. Measured on H3, recompute costs ~0.1 s/step and saves
# ~5 GB — so choosing "no checkpointing" on a thin margin trades five gigabytes of headroom for
# a tenth of a second. Peter's 6 Aug run picked it with 0.37 GB of predicted margin (needed
# 32.13 of 32.5 GB free) and then ran at 4-6 s/step instead of ~1: on Windows the driver spills
# to system RAM rather than OOMing, so an over-tight plan does not fail, it just crawls, with
# nothing in the log to say why. The un-checkpointed peak is also the one that scales with
# megapixels, so a plan that barely fits at one bucket size will not fit at the next.
_NOCKPT_MARGIN_GB = 3.0      # extra headroom demanded before skipping recompute


def adapter_param_count(dit_path: str, include_patterns, network_type: str = "lora",
                        network_dim: int = 16, lokr_factor: int = 8,
                        train_blocks: str = None) -> int:
    """Trainable parameter count, read from the checkpoint HEADER — no model, no GPU.

    The VRAM plan runs before the DiT is built, so the shapes come from the safetensors header
    (which is just JSON at the front of the file). That keeps this exact rather than an
    architecture guess: it sees the real targeted Linears for whichever checkpoint is loaded,
    respects include_patterns and the Blocks to Train restriction, and works the same on the
    pruned and full builds.
    """
    import json
    import re as _re
    import struct
    try:
        with open(dit_path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
    except Exception:
        return 0

    pats = list(include_patterns or [])
    if train_blocks:
        n_blocks = len({int(m.group(1)) for k in hdr
                        for m in [_re.match(r"blocks\.(\d+)\.", k)] if m} or {0})
        pats = restrict_patterns_to_blocks(pats, train_blocks, n_blocks)
    if not pats:
        return 0
    rx = [_re.compile(p) for p in pats]

    total = 0
    for key, ent in hdr.items():
        if key == "__metadata__" or not key.endswith(".weight"):
            continue
        shape = ent.get("shape") or []
        if len(shape) != 2:                     # Linears only, as create_modules wraps
            continue
        name = key[:-len(".weight")]
        if not any(r.search(name) for r in rx):
            continue
        out_dim, in_dim = int(shape[0]), int(shape[1])
        if str(network_type).lower() == "lokr":
            from fizgig.networks.lora import factorization   # local: avoids a circular import
            a, _c = factorization(out_dim, int(lokr_factor))
            b, _d = factorization(in_dim, int(lokr_factor))
            total += a * b + _c * _d            # w1 (a,b) + w2 (c,d)
        else:
            total += int(network_dim) * (in_dim + out_dim)
    return total


def adapter_vram_gb(params: int, optimizer_type: str = "adamw8bit") -> float:
    """GB the adapter holds for the WHOLE run: bf16 weights + optimizer state.

    Not a rounding error at these sizes. LoKR factor 8 on H3 trains ~313 M parameters against a
    rank-16 LoRA's ~77 M, and the state dtype widens the gap again: fp32 Adam keeps two 4-byte
    moments per parameter where the 8-bit optimizers keep two 1-byte ones. LoKR + adamw is
    ~3.1 GB against ~0.4 GB for the rank-16 + adamw8bit configuration the original anchors were
    measured on — which is why planning without this term was planning for a run nobody does.

    Gradients are deliberately NOT counted here. They are transient, and fused AdamW frees them
    per parameter as it steps, so they never all coexist: measured, a checkpointed step peaks
    only 0.19 GB above this figure even though the gradients would be 0.63 GB if they were all
    live at once. They belong in the activation term's margin, not in the resident one.

    Verified against a real step (6 Aug 2026): base 21.07 + weights 0.63 + fp32 state 2.50 =
    24.20 GB resident, exactly what this returns for 313.1 M parameters on adamw.
    """
    key = (optimizer_type or "adamw8bit").lower()
    n_states = 1 if "lion" in key else 2        # Lion keeps momentum only
    state_bytes = (1 if "8bit" in key else 4) * n_states
    return params * (2 + state_bytes) / 1e9     # bf16 weight + optimizer state


def plan_base_quant(free_gb: float, pruned: bool, mp: float = 0.25, adapter_gb: float = 0.0):
    """Pick the base quantisation AND the swap plan together -> (mode, blocks_to_swap, ckpt, why).

    Choosing a swap count from VRAM alone, with the quantisation already fixed, produces the
    worst available outcome on mid-range cards: the int8 base is ~21 GB, so a 24 GB card cannot
    hold it and the planner parks 38 of 50 blocks on CPU — every one of them crossing PCIe every
    step, for roughly 4x the step time. The same file loaded 4-bit is ~11 GB and needs no swap at
    all. Krea 2 hit this exact failure and fixed it the same way (see _auto_krea2_strategy):
    quantisation and swap are one decision.

    Order of preference:
      1. int8, no swap  — the most accurate base (~0.17% error against the reference's own
                          storage) with no PCIe cost. Always preferred when it fits.
      2. 4-bit, no swap — trades base accuracy (~9.5% error) for keeping every block resident.
      3. 4-bit + swap   — 11 GB resident always parks fewer blocks than 21 GB would.

    The trade in step 2 is real and worth stating: a LoRA fitted on a 9.5%-perturbed base spends
    capacity correcting error that will not exist at inference, and it compounds with depth. It
    is chosen only when the alternative is most of the model crossing PCIe on every step.

    Only applies to a pruned int8 checkpoint — the bf16 file has no int8 weights to keep, so
    there is nothing to choose between.
    """
    if not pruned:
        n, c = plan_vram(free_gb, mp=mp, resident_gb=_RESIDENT_GB, adapter_gb=adapter_gb)
        return "nf4", n, c, "bf16 checkpoint — NF4 is the only option"

    i_swap, i_ckpt = plan_vram(free_gb, mp=mp, resident_gb=_RESIDENT_INT8_GB,
                               transient_gb=_INT8_TRANSIENT_GB, adapter_gb=adapter_gb)
    if i_swap == 0:
        return "int8", i_swap, i_ckpt, "int8 fits with no block swap — the most accurate base"

    n_swap, n_ckpt = plan_vram(free_gb, mp=mp, resident_gb=_RESIDENT_PRUNED_GB,
                               adapter_gb=adapter_gb)
    if n_swap == 0:
        return ("nf4", n_swap, n_ckpt,
                f"int8 would need {i_swap} of 50 blocks on CPU (~4x slower); 4-bit fits entirely "
                f"in VRAM, at ~9% more error in the frozen base")
    return ("nf4", n_swap, n_ckpt,
            f"neither fits outright — 4-bit parks {n_swap} blocks against int8's {i_swap}")


def plan_vram(free_gb: float, mp: float = 0.25, batch: int = 1, resident_gb: float = None,
              transient_gb: float = 0.0, adapter_gb: float = 0.0):
    """Pure planner: (blocks_to_swap, gradient_checkpointing) from free VRAM + token load.

    Token load scales the activation term linearly (tokens ∝ mp x batch). Checkpointing is
    preferred OFF (faster) when everything fits without it; forced ON whenever swap is needed
    (without recompute, autograd would pin every swapped block's weights through backward).
    Swap additionally budgets _SWAP_TRANSIENT_GB: the backward pass transiently holds
    recompute segments beyond the parked residency (measured, see anchors above)."""
    resident = _RESIDENT_GB if resident_gb is None else float(resident_gb)
    # adapter_gb is resident for the whole run (weights + grads + optimizer state), so it belongs
    # in the base, not the activation term — gradient checkpointing does not reduce it.
    base = resident + float(transient_gb) + float(adapter_gb)
    scale = max(0.25, float(mp)) / 0.25 * max(1, int(batch))
    # _NOCKPT_MARGIN_GB, not just _RESERVE_GB: see the note on the constant. Recompute is ~0.1 s
    # a step and worth ~5 GB, so skipping it on a thin margin is a bad trade in both directions.
    need_nockpt = base + _ACT_GB_NOCKPT * scale + _RESERVE_GB + _NOCKPT_MARGIN_GB
    if free_gb >= need_nockpt:
        return 0, False
    need_ckpt = base + _ACT_GB_CKPT * scale + _RESERVE_GB
    if free_gb >= need_ckpt:
        return 0, True
    deficit = need_ckpt + _SWAP_TRANSIENT_GB - free_gb
    blocks = min(40, int(deficit / _PER_BLOCK_GB + 0.999))
    return blocks, True


def is_pruned_checkpoint(path: str) -> bool:
    """Does this file carry the curve-table AdaLN? Reads only the safetensors header.

    Needed before the base loads, because the pruned build's NF4 residency is ~6 GB smaller and
    the swap planner would otherwise park blocks nobody needs parked."""
    import json
    import struct
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return "adaln_t_table" in json.loads(f.read(n))
    except Exception:
        return False


def read_sample_override(output_dir):
    """Live sample override written by the GUI to <output_dir>/.sample_override.json.

    Returns {prompt, seed, width, height} while active, else None. Unlike Krea 2 there is no
    ref_image: H3 is not an edit model, so a reference is meaningless here and a prompt is
    required for the override to count."""
    import json
    path = os.path.join(output_dir, ".sample_override.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        prompt = str(d.get("prompt", "")).strip()
        if not prompt:
            return None
        return {"prompt": prompt,
                "seed": int(d.get("seed", 1234)),
                "width": int(d.get("width", 768)),
                "height": int(d.get("height", 768))}
    except Exception:
        return None


def sample_sigmas(batch: int, device, shift=None, generator=None,
                  image_tokens: int = None) -> torch.Tensor:
    """Noise levels in (0,1) for training.

    shift=None (the default): sigma = 12u/(1+11u), u ~ uniform — H3's OWN training density.
    ai-toolkit's per-model defaults override the global 'sigmoid' with timestep_type='shift'
    through a scheduler configured shift=12 (ui options.tsx + their scheduler_config), so this
    is what MiniMax LoRAs are actually trained with there: median sigma ~0.92, ~57% of steps
    above 0.9, ~3% below 0.3. Training lives at the high-noise end, where each step nudges
    broad structure gently — which is why 1e-4 is a sane LR there and scorching at low shifts.
    (An earlier run here blamed shift-12 for poor likeness; that verdict was confounded —
    bf16 adaln was eating half the LoRA and being dropped at inference, and the pack had no
    audio rows yet. Withdrawn.)

    shift="sigmoid": UNSHIFTED logit-normal, sigma = sigmoid(N(0,1)), median 0.5 — the
    SD3/Flux-style density (ai-toolkit's GLOBAL default, but NOT its MiniMax one). Trains the
    mid/low-noise zone hard: at 1e-4 a 46-image epoch visibly overdrove the adapters
    (real-run finding, twice). A/B use only.

    shift="resolution": logit-normal with a resolution-dependent shift (~1.7 @768^2, median
    0.62 — Krea 2's mapping). Fizgig's original replacement density; same overdrive failure.

    shift=<float>: the uniform-u + shift map at any other value.
    """
    if shift is None:
        shift = VIDEO_SIGMA_SHIFT_TRAIN
    if shift == "sigmoid":
        return torch.sigmoid(torch.randn(batch, device=device, generator=generator))
    if shift == "resolution":
        tokens = float(image_tokens or 225)                       # ~0.25 MP default
        mu = 0.5 + (tokens - 256.0) * (1.15 - 0.5) / (6400.0 - 256.0)
        s = math.exp(mu)
        base = torch.sigmoid(torch.randn(batch, device=device, generator=generator))
    elif isinstance(shift, str) and shift.startswith("lognorm:"):
        # SHAPE, not amount. Same shift map, but a logit-normal base instead of a uniform one:
        # the mass piles up in the middle and thins at BOTH ends, where a uniform base has fat
        # tails. Krea 2 and Klein both draw logit-normal, so this is the one axis the numeric
        # ladder cannot reach — it only ever varies how much low-noise training there is, never
        # where the rest of the mass sits.
        s = float(shift.split(":", 1)[1])
        base = torch.sigmoid(torch.randn(batch, device=device, generator=generator))
    else:
        s = float(shift)
        base = torch.rand(batch, device=device, generator=generator)
    return (s * base) / (1.0 + (s - 1.0) * base)


def compute_loss(model, latent: torch.Tensor, text_embeds: torch.Tensor, *,
                 sigma: torch.Tensor = None, shift: float = None, generator=None,
                 noise: torch.Tensor = None):
    """One image-training step's loss.

    latent      : [1, 24, 1, H, W] clean VAE latent (x0).
    text_embeds : [1, L, text_dim] Qwen3-VL states.
    noise       : optional fixed noise (reproducible steps / tests); else sampled.
    Returns (loss, sigma_used) — MSE of the DiT's video_out against (x0 - noise).
    """
    if latent.shape[0] != 1:
        raise ValueError("MiniMax H3 image training is batch size 1")
    device = latent.device
    x0 = latent.float()
    # The DiT patchifies with patch_size (1, ph, pw), so the latent's H and W must be divisible by
    # the spatial patch. The dataset buckets on a 16-px step and the VAE is 16x, so a latent can be
    # odd (e.g. a 496-px bucket -> 31-px latent, not divisible by 2). Crop to the patch multiple
    # (drops at most one latent row/col = <=16 px of image edge) so patchify is exact and the target
    # (x0 - noise) stays the same shape as the model's prediction.
    _pt, _ph, _pw = getattr(model, "patch_size", (1, 2, 2))
    _H, _W = x0.shape[-2], x0.shape[-1]
    _Hc, _Wc = (_H // _ph) * _ph, (_W // _pw) * _pw
    if (_Hc, _Wc) != (_H, _W):
        x0 = x0[..., :_Hc, :_Wc].contiguous()
    if noise is None:
        noise = torch.randn(x0.shape, device=device, generator=generator, dtype=torch.float32)
    else:
        noise = noise.to(device=device, dtype=torch.float32)[..., :x0.shape[-2], :x0.shape[-1]]
    if sigma is None:
        # Resolution-aware auto schedule: token count from the (cropped) latent's patch grid.
        _tokens = (x0.shape[-2] // _ph) * (x0.shape[-1] // _pw)
        sigma = sample_sigmas(1, device, shift=shift, generator=generator, image_tokens=_tokens)
    s = sigma.reshape(1, 1, 1, 1, 1).to(torch.float32)

    noised = (1.0 - s) * x0 + s * noise
    t = (1.0 - sigma).to(device)
    pred = model(noised.to(latent.dtype), t, text_embeds)
    target = (x0 - noise).to(pred.dtype)
    return F.mse_loss(pred.float(), target.float()), float(sigma.reshape(-1)[0])


@contextlib.contextmanager
def lora_disabled(network):
    """Run the frozen BASE inside this block — every adapter's multiplier is temporarily 0.

    Every module type (LoRA, LoKR, LoHa) reads self.multiplier live in its forward and
    short-circuits on 0.0, so this needs no re-apply and no weight surgery. Restores whatever
    each module had, not a blanket 1.0 — a context LoRA rides at its own strength."""
    mods = list(getattr(network, "unet_loras", []))
    saved = [m.multiplier for m in mods]
    try:
        for m in mods:
            m.multiplier = 0.0
        yield
    finally:
        for m, v in zip(mods, saved):
            m.multiplier = v


def compute_distill_loss(model, network, latent, text_plain, *, text_ref, ref_latents,
                         text_token_tags=None, distill_weight=0.8, shift=None, generator=None,
                         noise=None, seed=0):
    """Reference distillation: teach the LoRA to behave, from text alone, as if it had been
    shown the reference photo.

    Two predictions of the SAME noised latent at the SAME timestep:
      teacher — frozen base, LoRA off, conditioning WITH the reference (vision blocks + ref rows)
      student — LoRA on, conditioning WITHOUT it
    loss = w * MSE(student, teacher) + (1 - w) * MSE(student, x0 - noise)

    The photo term is what keeps real photographic detail available: pure distillation caps the
    LoRA at exactly the teacher's habits and can never exceed them. The teacher term is what
    stops the run spending capacity on backgrounds and framing, because the target is no longer
    a particular photograph.

    Everything the two passes share is drawn ONCE — noise, timestep, and the audio silence rows.
    The audio rows especially: model.forward redraws them per call when not given, so letting
    each pass draw its own would put a different soundtrack under teacher and student and add
    pure noise to the very signal being distilled.
    """
    if latent.shape[0] != 1:
        raise ValueError("MiniMax H3 image training is batch size 1")
    device = latent.device
    x0 = latent.float()
    _pt, _ph, _pw = getattr(model, "patch_size", (1, 2, 2))
    _H, _W = x0.shape[-2], x0.shape[-1]
    _Hc, _Wc = (_H // _ph) * _ph, (_W // _pw) * _pw
    if (_Hc, _Wc) != (_H, _W):
        x0 = x0[..., :_Hc, :_Wc].contiguous()
    if noise is None:
        noise = torch.randn(x0.shape, device=device, generator=generator, dtype=torch.float32)
    else:
        noise = noise.to(device=device, dtype=torch.float32)[..., :x0.shape[-2], :x0.shape[-1]]

    _tokens = (x0.shape[-2] // _ph) * (x0.shape[-1] // _pw)
    sigma = sample_sigmas(1, device, shift=shift, generator=generator, image_tokens=_tokens)
    s = sigma.reshape(1, 1, 1, 1, 1).to(torch.float32)
    noised = ((1.0 - s) * x0 + s * noise).to(latent.dtype)
    t = (1.0 - sigma).to(device)

    # one soundtrack for both passes (see the docstring)
    audio_noise = None
    if getattr(model, "pack_audio_rows", False):
        from fizgig.minimax.model import AUDIO_CHANNELS, audio_latents_for_frames
        n_a = audio_latents_for_frames(1) * AUDIO_CHANNELS
        audio_noise = torch.randn(n_a, model.config.audio_latents_dim, device=device,
                                  generator=generator, dtype=torch.float32)

    with torch.no_grad(), lora_disabled(network):
        teacher = model(noised, t, text_ref, audio_noise, ref_latents=ref_latents,
                        text_token_tags=text_token_tags, seed=seed).float()
    student = model(noised, t, text_plain, audio_noise).float()

    w = float(distill_weight)
    loss = w * F.mse_loss(student, teacher.detach())
    if w < 1.0:
        loss = loss + (1.0 - w) * F.mse_loss(student, (x0 - noise).float())
    return loss, float(sigma.reshape(-1)[0])


# ---------------------------------------------------------------------------
# Adaptive LR — bi-directional plateau tracker (architecture-agnostic; a faithful port of the
# Klein/Krea 2 watcher). Stability signal is weight-norm growth (>30%), same as Krea 2 (the H3
# loop clips gradients but the watcher reads weight-norm growth, not the clip ratio).
# ---------------------------------------------------------------------------
class AdaptiveLR:
    """Each epoch boundary: probe UP x1.25 on steady loss descent (patience 2); reduce DOWN x0.5
    on loss plateau (patience ramp) or a stability signal. On a stability event it blends the LoRA
    weights 70/30 toward the previous epoch's snapshot and restores the optimizer state (kills bad
    Adam momentum). The CPU rollback snapshot is in-memory only; the streak/best_loss scalars are
    JSON round-trippable (kept for parity — this barebones trainer has no resume yet)."""

    BLEND = 0.7
    WEIGHT_GROWTH_THRESHOLD = 0.30

    def __init__(self, min_lr, max_lr):
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.best_loss = None
        self.good_streak = 0
        self.bad_streak = 0
        self.stability_streak = 0
        self.stability_triggered = False
        self.prev_weight_norm = None
        self.snapshot = None  # {"weights": {...cpu...}, "optim": cpu state} — not persisted

    def state_dict(self):
        return {"best_loss": self.best_loss, "good_streak": self.good_streak,
                "bad_streak": self.bad_streak, "stability_streak": self.stability_streak,
                "stability_triggered": self.stability_triggered,
                "prev_weight_norm": self.prev_weight_norm}

    def load_state_dict(self, d):
        if not d:
            return
        self.best_loss = d.get("best_loss")
        self.good_streak = int(d.get("good_streak", 0))
        self.bad_streak = int(d.get("bad_streak", 0))
        self.stability_streak = int(d.get("stability_streak", 0))
        self.stability_triggered = bool(d.get("stability_triggered", False))
        self.prev_weight_norm = d.get("prev_weight_norm")

    @staticmethod
    def _weight_norm(network):
        wn = 0.0
        with torch.no_grad():
            for p in network.parameters():
                if p.requires_grad:
                    wn += float(p.detach().float().norm().item()) ** 2
        return wn ** 0.5

    def _snapshot(self, network, optimizer):
        with torch.no_grad():
            weights = {n: p.detach().clone().to("cpu")
                       for n, p in network.named_parameters() if p.requires_grad}

        def _cpu(o):
            if isinstance(o, torch.Tensor):
                return o.detach().clone().to("cpu")
            if isinstance(o, dict):
                return {k: _cpu(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_cpu(v) for v in o]
            return o
        try:
            self.snapshot = {"weights": weights, "optim": _cpu(optimizer.state_dict())}
        except Exception:
            self.snapshot = {"weights": weights, "optim": None}

    def _rollback(self, network, optimizer):
        cur = dict(network.named_parameters())
        with torch.no_grad():
            for name, prev in self.snapshot["weights"].items():
                if name in cur and cur[name].requires_grad:
                    p = cur[name]
                    prev_d = prev.to(device=p.device, dtype=p.dtype)
                    p.copy_(self.BLEND * prev_d + (1.0 - self.BLEND) * p)
        if self.snapshot.get("optim") is not None:
            try:
                optimizer.load_state_dict(self.snapshot["optim"])
            except Exception:
                pass

    def epoch_boundary(self, epoch, current_loss, network, optimizer):
        """epoch is 0-indexed (global). epoch 0 arms the baseline; epoch >= 1 adjusts the LR."""
        if epoch == 0:
            self.best_loss = current_loss
            self.prev_weight_norm = self._weight_norm(network)
            logger.info(f"[adaptive_lr] epoch 1: loss={current_loss:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} | ARMED")
            self._snapshot(network, optimizer)
            return

        patience_up = 2
        patience_down = 2 if (self.stability_triggered or epoch == 1 or epoch >= 4) else 1
        cur_lr = optimizer.param_groups[0]["lr"]
        new_lr = cur_lr
        cur_wn = self._weight_norm(network)
        weight_growth = None
        if self.prev_weight_norm and self.prev_weight_norm > 0:
            weight_growth = (cur_wn - self.prev_weight_norm) / self.prev_weight_norm
        stability_reason = None
        if weight_growth is not None and weight_growth > self.WEIGHT_GROWTH_THRESHOLD:
            stability_reason = f"wnorm_Δ {weight_growth*100:+.0f}% > {self.WEIGHT_GROWTH_THRESHOLD*100:.0f}%"

        action, reason = "HOLD", ""
        if stability_reason is not None:
            self.stability_streak += 1
            stability_patience = 1 if not self.stability_triggered else 2
            if self.stability_streak >= stability_patience:
                candidate = max(cur_lr * 0.5, self.min_lr)
                note = ""
                if self.snapshot is not None:
                    self._rollback(network, optimizer)
                    note = f"; blended {int(self.BLEND*100)}/{int((1-self.BLEND)*100)} + optim restored"
                if candidate < cur_lr:
                    new_lr = candidate
                    action = "REDUCE+ROLLBACK" if self.snapshot is not None else "REDUCE"
                else:
                    action = "HOLD (floored)"
                reason = f"stability: {stability_reason}{note}"
                self.good_streak = self.bad_streak = self.stability_streak = 0
                self.stability_triggered = True
            else:
                action = "WAIT"
                reason = f"stability: {stability_reason}, streak {self.stability_streak}/{stability_patience}"
        elif self.best_loss is None or current_loss < self.best_loss:
            self.stability_streak = 0
            self.best_loss = current_loss
            self.good_streak += 1
            self.bad_streak = 0
            if self.good_streak >= patience_up:
                candidate = min(cur_lr * 1.25, self.max_lr)
                if candidate > cur_lr:
                    new_lr = candidate
                    action = "PROBE UP"
                    reason = f"loss improving, streak {self.good_streak}"
                else:
                    action = "HOLD (capped)"
                    reason = "loss improving, at max_lr"
                self.good_streak = 0
            else:
                reason = f"loss improving, streak {self.good_streak}/{patience_up}"
        else:
            self.stability_streak = 0
            self.bad_streak += 1
            self.good_streak = 0
            if self.bad_streak >= patience_down:
                candidate = max(cur_lr * 0.5, self.min_lr)
                if candidate < cur_lr:
                    new_lr = candidate
                    action = "REDUCE"
                    reason = f"loss plateau, streak {self.bad_streak}"
                else:
                    action = "HOLD (floored)"
                    reason = "loss plateau, at min_lr"
                self.bad_streak = 0
            else:
                reason = f"loss plateau, streak {self.bad_streak}/{patience_down}"

        if new_lr != cur_lr:
            # Respect a depth-split LR: each group carries its own lr_scale, so the watcher moves
            # the whole schedule up or down while KEEPING the ratio between groups. Writing new_lr
            # flat would silently undo the split on the first adaptive move.
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr * pg.get("lr_scale", 1.0)
        lr_str = f"{cur_lr:.2e}" if new_lr == cur_lr else f"{cur_lr:.2e}->{new_lr:.2e}"
        wn_str = f"{weight_growth*100:+.0f}%" if weight_growth is not None else "—"
        logger.info(f"[adaptive_lr] epoch {epoch + 1}: loss={current_loss:.4f} lr={lr_str} "
                    f"wnorm_Δ={wn_str} | {action} ({reason})")
        self.prev_weight_norm = cur_wn
        self._snapshot(network, optimizer)


# ---------------------------------------------------------------------------
# Full image-only training loop (NF4 base + LoRA) over the H3 caches.
# ---------------------------------------------------------------------------
class _Collator:
    """DataLoader batch_size is always 1 (the dataset batches internally by bucket)."""

    def __init__(self, shared_epoch, dataset):
        self.shared_epoch = shared_epoch
        self.dataset = dataset

    def __call__(self, examples):
        wi = torch.utils.data.get_worker_info()
        ds = wi.dataset if wi is not None else self.dataset
        ds.set_current_epoch(self.shared_epoch.value)
        return examples[0]


def _save_training_state(output_dir, output_name, network, optimizer, *, epoch, global_step,
                         dtype, extra=None):
    """Save a resumable training-state dir matching Klein/Krea 2 naming: <name>-<NNNNNN>-state/.

    NNNNNN is the number of COMPLETED epochs (= the next 0-indexed epoch to run). Holds the
    network weights in NATIVE state_dict naming (never the LyCORIS comfy-format rewrite — resume
    load_state_dict needs the module keys), the optimizer state, RNG, and a small JSON. The
    GUI's _detect_latest_state_dir finds the highest-numbered dir and passes it to --resume."""
    import json
    state_dir = os.path.join(output_dir, f"{output_name}-{epoch:06d}-state")
    os.makedirs(state_dir, exist_ok=True)
    network.save_weights(os.path.join(state_dir, "lora.safetensors"), dtype,
                         {"ss_architecture": ARCHITECTURE_MINIMAX,
                          "ss_network_module": "fizgig.minimax (state dir, native keys)"})
    torch.save(optimizer.state_dict(), os.path.join(state_dir, "optimizer.pt"))
    rng = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, os.path.join(state_dir, "rng.pt"))
    meta = {"epoch": epoch, "global_step": global_step}
    if extra:
        meta.update(extra)
    with open(os.path.join(state_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    logger.info(f"[state] saved -> {state_dir}")
    return state_dir


def _load_training_state(state_dir, network, optimizer, *, device):
    """Restore network + optimizer + RNG from a state dir. Returns (start_epoch, global_step, meta)."""
    import json
    from safetensors.torch import load_file
    # strict=False tolerates benign key drift, but if NOTHING matched the network silently stays
    # at its zero init and the run "succeeds" while training from scratch — then overwrites the
    # finished LoRA with a no-op. Refuse that outright.
    _incompat = network.load_state_dict(load_file(os.path.join(state_dir, "lora.safetensors")), strict=False)
    _missing = getattr(_incompat, "missing_keys", [])
    if _missing and len(_missing) >= len(network.state_dict()):
        raise RuntimeError(
            f"[state] {state_dir} matched none of this network's {len(network.state_dict())} keys — "
            f"refusing to resume into a zero-initialised network. The state was almost certainly "
            f"saved with a different config (rank/alpha/factor, network type, or target modules).")
    opt_path = os.path.join(state_dir, "optimizer.pt")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
    rng_path = os.path.join(state_dir, "rng.pt")
    if os.path.exists(rng_path):
        try:
            rng = torch.load(rng_path)
            torch.set_rng_state(rng["torch"].to("cpu", dtype=torch.uint8) if hasattr(rng["torch"], "to") else rng["torch"])
            if "cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            logger.warning("[state] RNG restore failed; continuing with fresh RNG", exc_info=True)
    meta_path = os.path.join(state_dir, "training_state.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    return int(meta.get("epoch", 0)), int(meta.get("global_step", 0)), meta


def _save_lora(network, path, network_dim, network_alpha, dtype, extra_metadata=None):
    is_lokr = getattr(network, "_network_type", "lora") == "lokr"
    if is_lokr:
        metadata = {
            "ss_network_module": "fizgig.minimax (lokr, transformer blocks)",
            "ss_lokr_factor": str(getattr(network, "_lokr_factor", "")),
            "ss_architecture": ARCHITECTURE_MINIMAX,
        }
    else:
        metadata = {
            "ss_network_module": "fizgig.minimax (lora_unet, transformer blocks)",
            "ss_network_dim": str(network_dim),
            "ss_network_alpha": str(network_alpha),
            "ss_architecture": ARCHITECTURE_MINIMAX,
        }
    if extra_metadata:
        metadata.update(extra_metadata)
    if is_lokr:
        # LyCORIS-standard keys (diffusion_model.<dotted>.lokr_*) — the format every ComfyUI
        # LoKR in the wild uses. Unlike Krea 2 (whose internal saves stay native for resume and
        # previews), MiniMax has neither, and every checkpoint's only consumer is ComfyUI — so
        # every LoKR save is comfy-format. Fizgig's own loader ingests both namings via
        # ensure_kohya_lora_state_dict.
        from fizgig.networks.lora import _precalculate_safetensors_hashes
        from safetensors.torch import save_file
        dotted = getattr(network, "_dotted_names", {})
        sd = {}
        for k, v in network.state_dict().items():
            mod, _, suffix = k.partition(".")
            path_dotted = dotted.get(mod)
            nk = f"diffusion_model.{path_dotted}.{suffix}" if path_dotted else k
            v = v.detach().clone().to("cpu")
            if dtype is not None:
                v = v.to(dtype)
            sd[nk] = v
        model_hash, legacy_hash = _precalculate_safetensors_hashes(sd, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash
        save_file(sd, path, metadata)
        return
    network.save_weights(path, dtype, metadata)


def train_minimax(
    dataset_config: str,
    output_dir: str,
    output_name: str,
    dit_path: str,
    *,
    network_dim: int = 16,
    network_alpha: float = 16,
    network_type: str = "lora",      # "lora" | "lokr" (Kronecker, full-matrix w2)
    lokr_factor: int = 8,            # LoKR only: w1 is ~factor x factor; dim/alpha unused
    learning_rate: float = 1e-4,
    max_train_epochs: int = 10,
    save_every_n_epochs: int = 0,
    # Resumable state dirs (network + optimizer + RNG + adaptive scalars). Pause saves state
    # regardless of these — they govern only the automatic per-checkpoint / end-of-run saves.
    save_state: bool = False,
    save_state_on_train_end: bool = False,
    keep_last_n_states: int = 2,
    resume_state_dir: str = None,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    optimizer_type: str = "adamw8bit",
    optimizer_args: str = "",
    caption_dropout: float = 0.05,
    base_quant: str = "auto",
    include_patterns: list = None,
    train_blocks: str = None,        # "14-37" = train only that block range (experiment)
    train_adaln: bool = True,        # False = drop adaln_proj from the targets (pruned only)
    distill: bool = False,           # reference distillation (references come from the dataset)
    distill_weight: float = 0.8,     # teacher share of the loss; the rest is the real photo
    slow_blocks: str = None,         # block spec trained at a reduced LR ("21-49")
    slow_block_lr_scale: float = 1.0,  # the multiplier applied to those blocks' LR
    quantize: bool = True,           # NF4 the base (QLoRA); False = bf16 base (needs ~66 GB VRAM)
    shift: float = None,             # None = auto resolution schedule (logit-normal); float = legacy
    blocks_to_swap="auto",           # "auto" | int — park the last N blocks on CPU between uses
    gradient_checkpointing="auto",   # "auto" | "on" | "off" — forced on when swap > 0
    adaptive_lr: bool = False,
    adaptive_lr_min: float = 1e-5,
    adaptive_lr_max: float = 4e-4,
    # In-training previews. Prompts come from the Samples tab; the text encoder is loaded ONCE
    # before the DiT (it must never be resident alongside it) and freed.
    sample_prompts: list = None,
    te_path: str = None,
    vae_path: str = None,
    sample_every_n_epochs: int = 0,
    sample_at_first: bool = False,
    # H3's native canvas: 768 short edge, 768*1344 pixel cap.
    sample_width: int = 768,
    sample_height: int = 768,
    # 28, matching the reference pipeline's default. 8 leaves the latent well off the
    # encoder's manifold, which is exactly where the decoder produces patchy output
    # (measured seam energy 4.0 on an off-manifold latent vs 1.05 on a real one).
    sample_steps: int = 28,
    sample_cfg_scale: float = 1.0,
    sample_negative: str = None,
    sample_seed: int = 42,
    # Output metadata (recorded in the saved LoRA).
    metadata_title: str = None,
    metadata_author: str = None,
    metadata_description: str = None,
    metadata_license: str = None,
    metadata_tags: str = None,
    metadata_trigger_phrase: str = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Native MiniMax H3 image-only LoRA training: bucketed dataloader over the H3 caches ->
    flow-matching loss -> optimizer -> save a ComfyUI-compatible LoRA. No samples, no preview."""
    from torch.utils.data import DataLoader

    from fizgig.dataset.config import (BlueprintGenerator, ConfigSanitizer,
                                       generate_dataset_group_by_blueprint, load_user_config)
    from fizgig.networks.lora import create_network
    from fizgig.training.optimizers import create_optimizer
    from fizgig.training.train_utils import LossRecorder
    from fizgig.training.metadata import build_metadata, resolve_title, ARCHITECTURE_MINIMAX
    from fizgig.minimax.loader import load_minimax_h3_dit
    from tqdm import tqdm
    import math

    torch.manual_seed(seed)
    user_include_patterns = include_patterns   # None -> resolved per checkpoint below
    # Parse the block selection NOW, before the 21 GB base streams in: a typo surfacing after
    # the load costs minutes and reads like a crash rather than a correction. Bounds-checking
    # waits until the model is up (that is when the real block count is known).
    if train_blocks:
        parse_block_spec(train_blocks)

    # ---- dataset (built from the caches the two cache scripts wrote) ----
    shared_epoch = Value("i", 0)
    user_config = load_user_config(dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        user_config, argparse.Namespace(), architecture=ARCHITECTURE_MINIMAX)
    group = generate_dataset_group_by_blueprint(
        blueprint.dataset_group, training=True, num_timestep_buckets=None, shared_epoch=shared_epoch)
    if group.num_train_items == 0:
        raise RuntimeError("No training items — run minimax_cache_latents then minimax_cache_text first.")
    logger.info(f"MiniMax H3 training: {group.num_train_items} items, {max_train_epochs} epochs")
    if shift is None:
        logger.info("[timesteps] shift-12 uniform map (median sigma ~0.92) — H3's own training "
                    "density, matching the reference trainer")
    elif shift == "sigmoid":
        logger.warning("[timesteps] UNSHIFTED logit-normal (median 0.5) — A/B mode. At 1e-4 this "
                       "overdrives adapters within an epoch on small datasets; the default "
                       "(omit --shift) is the reference recipe.")
    elif shift == "resolution":
        logger.warning("[timesteps] logit-normal + resolution shift (median ~0.62) — A/B mode; "
                       "same overdrive caveat as sigmoid.")
    elif isinstance(shift, str) and str(shift).startswith("lognorm:"):
        logger.info(f"[timesteps] logit-normal base at shift {str(shift).split(':', 1)[1]} — "
                    f"mid-concentrated spread at the requested low-noise share.")
    else:
        logger.info(f"[timesteps] explicit shift={shift} — uniform-u map.")

    # ---- VRAM plan: block swap + gradient checkpointing (before the base loads) ----
    _mp = 0.25
    try:
        _mp = max(w * h / 1e6 for ds in group.datasets for (w, h) in ds.batch_manager.bucket_resos)
    except Exception:
        pass
    _ckpt_req = str(gradient_checkpointing).lower()
    _base_mode = (base_quant if base_quant != "auto"
                  else ("int8" if is_pruned_checkpoint(dit_path) else "nf4"))
    if not quantize:
        _base_mode = "none"
    if str(blocks_to_swap).lower() == "auto":
        if torch.cuda.is_available() and quantize:
            _free_gb = torch.cuda.mem_get_info()[0] / 1e9
            _pruned = is_pruned_checkpoint(dit_path)
            # The adapter is NOT a rounding error and it is not fixed: LoKR 8 trains ~313 M
            # parameters against a rank-16 LoRA's ~75 M, and fp32 Adam state is 4x the 8-bit
            # one. Planning without it was planning for a configuration nobody runs — the
            # anchors were measured on rank-16 + adamw8bit (~0.45 GB) while the shipped default
            # is LoKR 8 + adamw (~3.8 GB). Shapes come from the checkpoint header, so this is
            # the real targeted module set for whichever file is loaded.
            _pat = PRUNED_INCLUDE_PATTERNS if _pruned else DEFAULT_INCLUDE_PATTERNS
            if not train_adaln:
                _pat = [p for p in _pat if "adaln" not in p]
            _ad_params = adapter_param_count(dit_path, _pat, network_type=network_type,
                                             network_dim=network_dim, lokr_factor=lokr_factor,
                                             train_blocks=train_blocks)
            _adapter = adapter_vram_gb(_ad_params, optimizer_type)

            if base_quant == "auto":
                _mode, n_swap, _ckpt_auto, _why = plan_base_quant(
                    _free_gb, _pruned, mp=_mp, adapter_gb=_adapter)
            else:
                # An explicit choice is never overridden — the plan is built AROUND it, or the
                # swap count would be sized for a quantisation that will not run.
                _mode = base_quant
                _res = (_RESIDENT_INT8_GB if _mode == "int8"
                        else _RESIDENT_PRUNED_GB if _pruned else _RESIDENT_GB)
                n_swap, _ckpt_auto = plan_vram(
                    _free_gb, mp=_mp, resident_gb=_res,
                    transient_gb=_INT8_TRANSIENT_GB if _mode == "int8" else 0.0,
                    adapter_gb=_adapter)
                _why = f"base precision pinned to {_mode} by the user"
            _base_mode = _mode
            _resident = (_RESIDENT_INT8_GB if _mode == "int8"
                         else _RESIDENT_PRUNED_GB if _pruned else _RESIDENT_GB)

            logger.info(f"[vram] auto plan: free {_free_gb:.1f} GB, largest bucket {_mp:.2f} MP, "
                        f"base ~{_resident:.0f} GB ({_mode}, {'pruned' if _pruned else 'bf16'}), "
                        f"adapter ~{_adapter:.1f} GB ({_ad_params/1e6:.0f} M params, "
                        f"{optimizer_type}) -> blocks_to_swap={n_swap}, "
                        f"checkpointing={'on' if _ckpt_auto else 'off'}")
            logger.info(f"[vram] base precision: {_mode} — {_why}")
            if _mode == "nf4" and _pruned and base_quant == "auto":
                # Say it plainly rather than quietly downgrading the base: this costs likeness,
                # and the user has a real alternative (train slower on int8, or free some VRAM).
                logger.warning(
                    "[vram] this run trains on a 4-bit base (~9% error) instead of the "
                    "checkpoint's own int8 (~0.17%). It is much faster here, but the LoRA spends "
                    "some capacity correcting quantization error that will NOT exist at "
                    "inference. To force the accurate base, set Base Precision to int8 — expect "
                    "block swap and a several-times-slower run — or close other GPU apps and "
                    "re-launch.")
            if n_swap > 0:
                logger.warning(
                    f"[vram] {n_swap} of 50 blocks will live on CPU and cross PCIe every step, "
                    f"which is several times slower. Lower Target Megapixels, or free VRAM, to "
                    f"avoid it.")
        else:
            n_swap, _ckpt_auto = 0, False
    else:
        n_swap = max(0, int(blocks_to_swap))
        _ckpt_auto = n_swap > 0
    use_ckpt = {"on": True, "off": False}.get(_ckpt_req, _ckpt_auto)
    if n_swap > 0 and not use_ckpt:
        logger.info("[vram] block swap needs gradient checkpointing (autograd would pin swapped "
                    "weights through backward) — forcing it on.")
        use_ckpt = True

    # ---- previews: encode the prompts BEFORE the DiT loads ----
    # Order matters more here than anywhere else in Fizgig: the Qwen3-VL-32B text encoder is
    # ~14 GB even at NF4, and the DiT is ~17 GB. They must never be resident together, so the
    # prompts are encoded once, up front, and the encoder is freed before the base streams in.
    do_previews = bool((sample_every_n_epochs or sample_at_first) and sample_prompts and te_path)
    encoded_prompts = encoded_negative = sample_dir = None
    if do_previews:
        from fizgig.minimax.sampling import encode_sample_prompts
        logger.info(f"[preview] pre-encoding {len(sample_prompts)} sample prompt(s) "
                    f"(the text encoder is freed before the DiT loads)...")
        try:
            encoded_prompts = encode_sample_prompts(te_path, sample_prompts, device=device,
                                                    quantize=quantize)
            if sample_negative and sample_cfg_scale and sample_cfg_scale > 1.0:
                encoded_negative = encode_sample_prompts(te_path, [sample_negative],
                                                         device=device, quantize=quantize)[0]
            sample_dir = os.path.join(output_dir, "sample")
            os.makedirs(sample_dir, exist_ok=True)
            # State the whole preview recipe once, up front — steps in particular, since too
            # few leaves the latent off-manifold and the decode patchy.
            logger.info(
                f"[preview] {sample_steps} steps @ {sample_width}x{sample_height}, "
                f"cfg {sample_cfg_scale:g}"
                f"{'' if sample_cfg_scale > 1.0 else ' (off — H3 is guidance-distilled)'}, "
                f"seed {sample_seed if sample_seed else 'random'}, "
                f"every {sample_every_n_epochs} epoch(s)"
                f"{', plus epoch 0' if sample_at_first else ''} — "
                f"{'full VAE decode' if vae_path else 'RGB approximation (no VAE path set)'}")
        except Exception as _e:
            logger.warning(f"[preview] prompt encoding failed ({type(_e).__name__}: {_e}) — "
                           f"previews disabled; training continues normally.")
            do_previews = False

    # ---- base (NF4-frozen) + trainable LoRA over the transformer blocks ----
    dit = load_minimax_h3_dit(dit_path, device=device, compute_dtype=dtype, quantize=quantize,
                              blocks_to_swap=n_swap, base_quant=base_quant)
    dit.requires_grad_(False)                                   # frozen base (QLoRA-style)
    if n_swap > 0:
        n_swap = dit.enable_block_swap(n_swap)                  # sets the JIT-move boundary
        logger.info(f"[vram] block swap active: last {n_swap} blocks parked on CPU "
                    f"(~{n_swap * 0.34:.1f} GB VRAM freed, packed NF4 in RAM)")
    if use_ckpt:
        dit.enable_gradient_checkpointing()
        logger.info("[vram] gradient checkpointing ON")
    # AdaLN targeting is per-checkpoint — see the pattern note at the top of this file.
    include_patterns = user_include_patterns or (
        PRUNED_INCLUDE_PATTERNS if dit.pruned_adaln else DEFAULT_INCLUDE_PATTERNS)
    # AdaLN is a pure function of the TIMESTEP — DiTBlock.forward calls adaln_proj(t_emb) and
    # nothing else, so its adapters cannot tell one subject from another. They can only reshape
    # how strongly each block fires at each noise level. On the pruned checkpoint they carry
    # ~45% of all weight movement in a matched epoch, which is a lot of a LoRA's capacity spent
    # somewhere structurally incapable of holding a face — hence the toggle. See
    # docs/MINIMAX_BLOCKS.md. No-op on the bf16 checkpoint, which never targets AdaLN.
    _adaln_on = bool(train_adaln) and dit.pruned_adaln
    if not train_adaln:
        _before = len(include_patterns)
        include_patterns = [p for p in include_patterns if "adaln" not in p]
        if len(include_patterns) < _before:
            logger.info("[base] EXPERIMENT: AdaLN adapters OFF. AdaLN sees only the timestep, so "
                        "it cannot encode identity — this frees the capacity it was taking. "
                        "Compare against the same run with it on.")
        else:
            logger.info("[base] AdaLN was not a target on this checkpoint; the toggle changes "
                        "nothing here.")
    _blocks_used = "all"
    if train_blocks:
        _n_blocks = len(dit.blocks)
        include_patterns = restrict_patterns_to_blocks(include_patterns, train_blocks, _n_blocks)
        _sel = parse_block_spec(train_blocks, _n_blocks)
        _blocks_used = format_block_spec(_sel)
        logger.info("[base] EXPERIMENT: training blocks %s only (%d of %d), text refiner "
                    "included. Nobody has mapped what H3's blocks do — judge this against a "
                    "full-model run on the same dataset, not on its own.",
                    _blocks_used, len(_sel), _n_blocks)
    logger.info("[base] %s checkpoint; LoRA targets: attention + MLP + token refiner%s",
                "pruned (curve-table AdaLN)" if dit.pruned_adaln else "full bf16",
                " + AdaLN (deploy-consistent on this build; rank caps at 8)"
                if dit.pruned_adaln else " (AdaLN excluded - dropped by pruned inference builds)")
    if network_type == "lokr":
        # LoKR (Kronecker) — same mechanism as Krea 2's: module_class swaps the parametrization
        # inside the identical scan/wrap machinery, so include_patterns (adaln exclusion) and the
        # NF4/Linear4bit base compose unchanged. dim/alpha are ignored; factor is the dial.
        from fizgig.networks.lora import LoKRModule
        logger.info(f"network: LoKR (Kronecker), factor {lokr_factor}, full-matrix w2 — "
                    "dim/alpha do not apply")
        network = create_network(None, "lora_unet", 1.0, network_dim, network_alpha, None, [], dit,
                                 include_patterns=include_patterns,
                                 module_class=LoKRModule, module_kwargs={"factor": int(lokr_factor)})
    else:
        network = create_network(None, "lora_unet", 1.0, network_dim, network_alpha, None, [], dit,
                                 include_patterns=include_patterns)
    network.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    network.requires_grad_(True)
    network.to(device=device, dtype=dtype)
    network._network_type = network_type
    network._lokr_factor = int(lokr_factor)
    # Dotted module paths for the LyCORIS-standard save (diffusion_model.<path>.lokr_*) — built
    # from the DiT itself with the same flattening create_modules used, so the reverse mapping is
    # exact even where module names contain underscores. isinstance covers bnb Linear4bit (an
    # nn.Linear subclass).
    network._dotted_names = {
        f"lora_unet_{name.replace('.', '_')}": name
        for name, m in dit.named_modules() if isinstance(m, torch.nn.Linear)
    }
    _n_targeted = len(network.unet_loras)
    if network_type == "lokr":
        logger.info(f"LoKR: {len(network.unet_loras)} modules wrapped (factor {lokr_factor})")
    else:
        logger.info(f"LoRA: {len(network.unet_loras)} modules wrapped (dim {network_dim}, alpha {network_alpha})")

    # How many Linears did the include_patterns actually TARGET? create_modules matches by class
    # NAME, so a quantized Linear stand-in that is not on that list is skipped in silence — which
    # once shipped a run training 58 of 258 modules with no error anywhere. Compare and refuse.
    import re as _re
    _targeted = [n for n, m in dit.named_modules()
                 if isinstance(m, torch.nn.Linear)
                 and any(_re.search(p, n) for p in include_patterns)]
    if len(network.unet_loras) < len(_targeted):
        _kinds = sorted({type(dit.get_submodule(n)).__name__ for n in _targeted})
        raise RuntimeError(
            f"only {len(network.unet_loras)} of {len(_targeted)} targeted Linears were wrapped — "
            f"the network builder matches by class name and one of {_kinds} is not on its list "
            f"(networks/lora.py, create_modules). Training now would silently learn a fraction "
            f"of the model.")
    _n_targeted = len(_targeted)
    logger.info(f"[network] {len(network.unet_loras)}/{_n_targeted} targeted Linears wrapped")

    params = list(network.get_trainable_params())

    # Adaptive LR ignores the Learning Rate box: it starts at the GEOMETRIC MIDPOINT of Min/Max
    # and the watcher owns the LR from there (matches Klein/Krea 2). Two knobs, not three.
    adaptive = AdaptiveLR(adaptive_lr_min, adaptive_lr_max) if adaptive_lr else None
    if adaptive:
        learning_rate = math.sqrt(adaptive_lr_min * adaptive_lr_max)
        logger.info(f"[adaptive_lr] ENABLED — start_lr={learning_rate:.3e} (geometric midpoint) "
                    f"min={adaptive_lr_min:.3e} max={adaptive_lr_max:.3e}; the Learning Rate box is ignored")

    # Weight-decay parity with the reference trainer: ai-toolkit's job template passes
    # optimizer_params weight_decay=1e-4; bitsandbytes' default is 0.01 (100x). Only applied
    # when the user hasn't set their own via Optimizer Args.
    if "weight_decay" not in (optimizer_args or "") and "adam" in optimizer_type.lower():
        optimizer_args = (optimizer_args + " weight_decay=1e-4").strip()

    # Depth-dependent LR. A perturbation injected at block 5 passes through 45 more blocks that
    # absorb and renormalize it; one injected at block 45 lands almost directly on the output. So
    # the same |dW| is far more disruptive the later it sits, and ONE learning rate is wrong by
    # construction — it is either too low for the early blocks or too high for the late ones.
    # Observed here: at 1e-4, blocks 0-20 train cleanly but slowly while anything past 20 wrecks
    # the samples (block swap ruled out — those runs recorded blocks_swapped=0).
    # Built AFTER the adaptive block above, so `learning_rate` is already the resolved start LR.
    _slow_used, _slow_n = "", 0
    opt_params = params          # the optimizer may get groups; `params` stays flat for clipping
    if slow_blocks and abs(float(slow_block_lr_scale) - 1.0) > 1e-9:
        _slow_idx = set(parse_block_spec(slow_blocks, len(dit.blocks)))
        _slow_ids = set()
        for _lora in network.unet_loras:
            _nm = _lora.lora_name
            if "token_refiner" in _nm:      # text-side, never part of the depth argument
                continue
            _m = re.search(r"blocks_(\d+)_", _nm)
            if _m and int(_m.group(1)) in _slow_idx:
                _slow_ids.update(id(p) for p in _lora.parameters())
        if _slow_ids:
            _slow = [p for p in params if id(p) in _slow_ids]
            _fast = [p for p in params if id(p) not in _slow_ids]
            _scaled = learning_rate * float(slow_block_lr_scale)
            # lr_scale rides along on the group so the adaptive watcher can move both groups
            # together without flattening them back to one rate.
            # NOTE: assign to opt_params, NOT params. `params` stays the flat tensor list because
            # clip_grad_norm_ iterates it every step and cannot take param-group dicts.
            opt_params = [{"params": _fast, "lr": learning_rate, "lr_scale": 1.0},
                          {"params": _slow, "lr": _scaled, "lr_scale": float(slow_block_lr_scale)}]
            _slow_used = format_block_spec(sorted(_slow_idx))
            _slow_n = len(_slow)
            logger.info("[lr] depth-split: blocks %s train at %.3e (x%g), the rest at %.3e "
                        "(%d of %d tensors slowed)", _slow_used, _scaled, slow_block_lr_scale,
                        learning_rate, _slow_n, len(_slow) + len(_fast))
        else:
            logger.warning("[lr] slow_blocks %r matched no trained modules — is it outside "
                           "Blocks to Train? Depth-split LR is not active.", slow_blocks)

    # eps_floor_8bit: H3-only. The 8-bit second moment underflows on this model's most structured
    # tensors and the update degrades to lr*m/eps — measured at ~100x the configured LR, which
    # presented as melted anatomy at epoch 1. The floor caps that. It is passed here and nowhere
    # else: Krea 2 has never shown the failure and keeps the library default.
    optimizer, optimizer_label = create_optimizer(optimizer_type, opt_params, learning_rate,
                                                  optimizer_args, eps_floor_8bit=True)
    logger.info(f"optimizer: {optimizer_label} @ lr={learning_rate:.3e}")

    # Caption dropout (reference default 0.05): swap in the cached empty-prompt embed for a
    # random ~5% of steps. The uncond file is written by minimax_cache_text next to the caches.
    uncond_text = None
    if caption_dropout and caption_dropout > 0:
        for _ds in group.datasets:
            _f = os.path.join(getattr(_ds, "cache_directory", "") or "",
                              f"uncond_{ARCHITECTURE_MINIMAX}_te.safetensors")
            if os.path.isfile(_f):
                from safetensors.torch import load_file as _lf
                uncond_text = _lf(_f)["hidden_states"].unsqueeze(0)      # (1, L, 5120)
                break
        if uncond_text is None:
            logger.warning("[caption_dropout] no uncond embed in the cache dirs (re-run text "
                           "caching to enable it) — dropout disabled for this run")
        else:
            logger.info(f"[caption_dropout] {caption_dropout:.2f} — empty-prompt embed loaded")

    # Reference distillation needs nothing at run start: each item's reference conditioning AND
    # that reference's latent both ride in from the cache, one slot picked at random per step.
    if distill:
        logger.info("[distill] reference distillation ON — teacher weight %.2f, photo %.2f. "
                    "References come from the dataset itself (each image paired with others by "
                    "the caching pass); no image is ever its own reference.",
                    distill_weight, 1.0 - distill_weight)

    collator = _Collator(shared_epoch, group)
    loader = DataLoader(group, batch_size=1, shuffle=True, collate_fn=collator, num_workers=0)
    try:
        steps_per_epoch = len(loader)
    except TypeError:
        steps_per_epoch = group.num_train_items

    os.makedirs(output_dir, exist_ok=True)
    pause_flag = os.path.join(output_dir, ".pause_requested")

    # ---- resume: restore network + optimizer + RNG + (epoch, step) + adaptive scalars ----
    from fizgig.training.train_utils import prune_state_dirs
    global_step = 0
    start_epoch = 0
    if resume_state_dir and os.path.isdir(resume_state_dir):
        start_epoch, global_step, _resume_meta = _load_training_state(
            resume_state_dir, network, optimizer, device=device)
        if adaptive:
            adaptive.load_state_dict(_resume_meta.get("adaptive_lr_state"))
        logger.info(f"[resume] from {resume_state_dir}: continuing at epoch "
                    f"{start_epoch + 1}/{max_train_epochs} (global_step {global_step})")
        if start_epoch >= max_train_epochs:
            # Pausing ON the last epoch exits before the final LoRA is written — Resume is what
            # completes it, so this fall-through writes the final file from the restored state.
            logger.warning(f"[resume] state is at epoch {start_epoch} of {max_train_epochs} — "
                           f"nothing left to train. Writing the final LoRA from the restored "
                           f"state. To train further, raise Max Train Epochs and resume again.")

    def _run_provenance():
        """What actually produced this LoRA — the facts you need to compare two of them.

        Added after an A/B where the file could not answer "was this the int8 base or NF4?",
        "how many modules were really wrapped?" or "how many steps?" — all of which changed the
        interpretation completely, and one of which (58 of 258 modules) had been a silent bug.
        A LoRA that cannot describe its own run is a measurement you have to take on trust."""
        try:
            _res = sorted({f"{w}x{h}" for ds in group.datasets
                           for (w, h) in ds.batch_manager.bucket_resos})
        except Exception:
            _res = []
        _dens = ("shift12" if shift is None else
                 shift if isinstance(shift, str) else f"shift{shift:g}")
        return {
            "ss_base_checkpoint": os.path.basename(dit_path),
            "ss_base_quant": _base_mode,
            "ss_lora_modules": str(len(network.unet_loras)),
            "ss_targeted_modules": str(_n_targeted),
            "ss_steps": str(global_step),
            "ss_epochs": str(max_train_epochs),
            "ss_learning_rate": f"{learning_rate:g}",
            "ss_optimizer": optimizer_label,
            "ss_timestep_density": _dens,
            "ss_train_blocks": _blocks_used,
            "ss_train_adaln": "1" if _adaln_on else "0",
            "ss_distill": "dataset" if distill else "off",
            "ss_distill_weight": (f"{distill_weight:g}" if distill else "0"),
            "ss_slow_blocks": _slow_used or "none",
            "ss_slow_block_lr_scale": (f"{slow_block_lr_scale:g}" if _slow_used else "1"),
            "ss_caption_dropout": f"{caption_dropout:g}" if uncond_text is not None else "0",
            "ss_max_grad_norm": f"{max_grad_norm:g}",
            "ss_bucket_resolutions": ",".join(_res),
            "ss_gradient_checkpointing": "1" if use_ckpt else "0",
            "ss_blocks_swapped": str(n_swap),
        }

    def _meta():
        md = build_metadata(
            None, ARCHITECTURE_MINIMAX, time.time(),
            title=(metadata_title if metadata_title is not None
                   else resolve_title(output_name, metadata_trigger_phrase)),
            author=metadata_author, description=metadata_description,
            license=metadata_license, tags=metadata_tags, trigger_phrase=metadata_trigger_phrase)
        md.update(_run_provenance())
        return md

    def _state_extra():
        return {"adaptive_lr_state": adaptive.state_dict()} if adaptive else None

    # Encoded override prompt, kept between epochs: re-encoding costs a TE load, so only redo it
    # when the prompt text actually changes.
    _ov_state = {"prompt": None, "enc": None}

    def _encode_override(prompt):
        """Encode one override prompt mid-run.

        The TE is ~14.5 GB and the int8 base ~21 GB, so unlike Krea 2 they cannot both be
        resident on a 32 GB card — the normal flow deliberately encodes every prompt BEFORE the
        DiT loads. To honour a live override we park the DiT on CPU for the duration, then
        restore it (and its block-swap split). That is a ~21 GB round trip, which is why the
        result is cached against the prompt text and only paid when you actually change it."""
        from fizgig.minimax.sampling import encode_sample_prompts
        _free = (torch.cuda.mem_get_info()[0] / 1e9) if torch.cuda.is_available() else 0.0
        _park = torch.cuda.is_available() and _free < 17.0     # TE + headroom
        if _park:
            logger.info(f"[sample override] parking the base on CPU to fit the text encoder "
                        f"({_free:.1f} GB free) — one-off for this prompt")
            dit.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
        try:
            return encode_sample_prompts(te_path, [prompt], device=device, quantize=quantize)
        finally:
            if _park:
                dit.to(device)
                if n_swap > 0:
                    dit.enable_block_swap(n_swap)   # restores the parked-block split
                gc.collect()
                torch.cuda.empty_cache()

    def _render_previews(epoch):
        """Render one still per prompt on the RESIDENT training DiT and write them where the
        samples gallery looks. The filename format is the gallery/likeness/Visualiser contract
        (parse_sample_filename in the GUI) — do not change it casually.

        The DiT never moves: only eval mode is toggled, and block swap's JIT .to() is already
        forward-safe, so there is no swap-mode dance like Krea 2 needs."""
        import time as _time
        import numpy as _np
        from PIL import Image
        from fizgig.minimax import sampling
        was_training = dit.training
        decoder = None
        try:
            dit.eval()
            if vae_path:
                # Loaded per preview and freed in the finally: the ViT3D decoder is ~4.85 GB and
                # would otherwise sit on top of the resident base for the whole run.
                from safetensors import safe_open as _safe_open
                from fizgig.minimax.vae import MiniMaxH3VideoVAEDecoder
                decoder = MiniMaxH3VideoVAEDecoder()
                with _safe_open(vae_path, framework="pt", device="cpu") as _f:
                    decoder.load_state_dict({k: _f.get_tensor(k) for k in _f.keys()}, strict=False)
                # bf16, NOT fp32: 2.4 B params is 4.8 GB vs 9.7 GB, and this sits on top of the
                # already-resident base. decode() follows the module dtype.
                decoder = decoder.to(device, dtype).eval()
            # Live override from the GUI, re-read every epoch so it can be turned on, changed or
            # switched off mid-run without touching the paused/resume path.
            _prompts, _w, _h = encoded_prompts, sample_width, sample_height
            _seed = sample_seed
            _ov = read_sample_override(output_dir)
            if _ov and not te_path:
                logger.warning("[sample override] a prompt is set but no --text_encoder is "
                               "configured, so it cannot be encoded — using the Samples tab.")
                _ov = None
            if _ov:
                if _ov["prompt"] != _ov_state["prompt"]:
                    _ov_state["enc"] = _encode_override(_ov["prompt"])
                    _ov_state["prompt"] = _ov["prompt"]
                _prompts, _w, _h, _seed = _ov_state["enc"], _ov["width"], _ov["height"], _ov["seed"]
                logger.info(f"[sample override] active — '{_ov['prompt'][:60]}' "
                            f"seed={_seed} {_w}x{_h}")

            _seed = _seed if _seed != 0 else random.randint(1, 2 ** 31 - 1)
            ts = _time.strftime("%Y%m%d%H%M%S")
            for i, txt in enumerate(_prompts):
                print(f"[preview] epoch {epoch}: prompt {i + 1}/{len(_prompts)} "
                      f"({_w}x{_h}, seed {_seed + i})", flush=True)
                lat = sampling.sample_image(
                    dit, txt.to(device, dtype),
                    width=_w, height=_h, steps=sample_steps,
                    cfg_scale=sample_cfg_scale,
                    uncond_embeds=(encoded_negative.to(device, dtype)
                                   if encoded_negative is not None else None),
                    seed=_seed + i, device=device, dtype=dtype, log_steps=True)
                if decoder is not None:
                    px = decoder.decode(lat.float())[0]          # [3, H, W] in [0, 1]
                    arr = (px.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                    img = Image.fromarray(arr)
                else:
                    # No VAE path configured — fall back to the 24ch->RGB linear approximation
                    # (a 1/16-scale rough look) rather than dropping previews entirely.
                    arr = sampling.latent_to_rgb(lat)
                    img = Image.fromarray(arr).resize((_w, _h), Image.NEAREST)
                print(f"[preview] decoded {_w}x{_h}", flush=True)
                img.save(os.path.join(
                    sample_dir, f"{output_name}_e{epoch:06d}_{i:02d}_{ts}_{_seed + i}.png"))
            logger.info(f"[preview] epoch {epoch}: wrote {len(_prompts)} sample(s) "
                        f"({sample_steps} steps, seed {_seed}) to {sample_dir}")
        finally:
            del decoder                                  # free the ~4.85 GB decoder immediately
            if was_training:
                dit.train()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- epoch loop ----
    loss_recorder = LossRecorder()
    if do_previews and sample_at_first and start_epoch == 0:
        try:
            _render_previews(0)
        except Exception as _e0:
            logger.warning(f"[preview] Sample at Start failed ({type(_e0).__name__}) — training "
                           f"continues; per-epoch previews will still be attempted.")
    progress_bar = tqdm(total=steps_per_epoch * max_train_epochs, initial=global_step,
                        desc="minimax-h3")
    for epoch in range(start_epoch, max_train_epochs):
        shared_epoch.value = epoch + 1
        network.train()
        for i, batch in enumerate(loader):
            latents = batch["latents"].to(device, dtype)           # (1, 24, H, W)
            if latents.dim() == 4:
                latents = latents.unsqueeze(2)                     # -> (1, 24, 1, H, W)
            text = batch["hidden_states"].to(device, dtype)        # (1, L, 5120)
            if uncond_text is not None and random.random() < caption_dropout:
                text = uncond_text.to(device, dtype)               # caption dropout step
            if distill and "ref_hidden_states" in batch:
                _rz = batch["ref_latent"].to(device, dtype)      # (1, 24, h, w) from the cache
                if _rz.dim() == 4:
                    _rz = _rz.unsqueeze(2)                       # -> (1, 24, 1, h, w)
                loss, _ = compute_distill_loss(
                    dit, network, latents, text,
                    text_ref=batch["ref_hidden_states"].to(device, dtype),
                    ref_latents=[_rz],
                    text_token_tags=batch["ref_token_tags"][0],
                    distill_weight=distill_weight, shift=shift, seed=seed)
            else:
                loss, _ = compute_loss(dit, latents, text, shift=shift)
            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            loss_recorder.add(epoch=epoch, step=i, loss=loss.item())
            progress_bar.set_postfix(avr_loss=f"{loss_recorder.moving_average:.4f}", refresh=False)
            progress_bar.update(1)

        logger.info(f"epoch {epoch + 1}/{max_train_epochs} done — avr_loss {loss_recorder.moving_average:.4f}")
        # Optimizer sanity: lora_up starts at zero and an Adam-family step is bounded by ~lr, so
        # after N steps no element can honestly exceed ~3*N*lr. When the 8-bit second moment
        # misbehaves (v quantized to zero -> update degrades to lr*m/eps) the drift blows through
        # that bound by orders of magnitude — caught here per epoch instead of per melted preview.
        try:
            _lr_now = optimizer.param_groups[0]["lr"]
            _drift = max((float(l.lora_up.weight.detach().abs().max())
                          for l in network.unet_loras if hasattr(l, "lora_up")), default=0.0)
            _bound = 3.0 * global_step * _lr_now
            if _drift > _bound:
                logger.warning(f"[drift] max|lora_up|={_drift:.4f} EXCEEDS the Adam bound "
                               f"~{_bound:.4f} ({global_step} steps @ lr={_lr_now:.1e}) — the "
                               f"optimizer is stepping far beyond the configured LR (8-bit "
                               f"state underflow?). Expect degraded samples.")
            else:
                logger.info(f"[drift] max|lora_up|={_drift:.4f} (bound ~{_bound:.4f} — healthy)")
        except Exception:
            pass
        if adaptive is not None:
            adaptive.epoch_boundary(epoch, loss_recorder.moving_average, network, optimizer)
        if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0 and (epoch + 1) < max_train_epochs:
            ckpt = os.path.join(output_dir, f"{output_name}-{epoch + 1:06d}.safetensors")
            _save_lora(network, ckpt, network_dim, network_alpha, dtype, _meta())
            logger.info(f"saved {ckpt}")
            if save_state:
                _save_training_state(output_dir, output_name, network, optimizer,
                                     epoch=epoch + 1, global_step=global_step,
                                     dtype=dtype, extra=_state_extra())
                prune_state_dirs(output_dir, output_name, keep_last_n_states)
        if do_previews and sample_every_n_epochs and (epoch + 1) % sample_every_n_epochs == 0:
            try:
                _render_previews(epoch + 1)
            except Exception as _pe:
                # Latch previews OFF for the rest of the run rather than re-failing (and
                # re-OOMing) every epoch. Training and checkpoints are never at risk.
                _oom = "out of memory" in str(_pe).lower()
                logger.warning(
                    f"[preview] epoch {epoch + 1} preview failed "
                    f"({'CUDA OOM' if _oom else type(_pe).__name__}); disabling previews for the "
                    f"rest of the run. Training continues and LoRAs still save normally.")
                do_previews = False
            network.train()
        if os.path.exists(pause_flag):
            # Pause = graceful epoch-end exit with FULL state (regardless of the save-state
            # toggles), so Resume continues exactly here — matching Klein/Krea 2. The final
            # LoRA is deliberately NOT written; Resume (or the natural run end) writes it.
            _save_training_state(output_dir, output_name, network, optimizer,
                                 epoch=epoch + 1, global_step=global_step,
                                 dtype=dtype, extra=_state_extra())
            try:
                os.remove(pause_flag)
            except OSError:
                pass
            progress_bar.close()
            logger.info(f"[pause] requested — state saved at epoch {epoch + 1}. Exiting cleanly.")
            sys.exit(0)

    progress_bar.close()
    final = os.path.join(output_dir, f"{output_name}.safetensors")
    _save_lora(network, final, network_dim, network_alpha, dtype, _meta())
    logger.info(f"saved final LoRA: {final}")
    if save_state_on_train_end and max_train_epochs > start_epoch:
        _save_training_state(output_dir, output_name, network, optimizer,
                             epoch=max_train_epochs, global_step=global_step,
                             dtype=dtype, extra=_state_extra())
        prune_state_dirs(output_dir, output_name, keep_last_n_states)
    try:
        os.remove(pause_flag)
    except OSError:
        pass
    return final
