"""Krea 2 LoRA training: full-model LoRA + flow-matching loss over a bucketed dataloader.

Trains on the RAW model. The LoRA wraps all 264 Linears (no layer-targeting presets yet — Krea2's
block semantics aren't mapped, so Identity/Style/Details presets come later). The base is frozen
(optionally fp8, QLoRA-style); only the LoRA trains in bf16. Uses Fizgig's bucketed multi-resolution
dataloader (same framework as Klein) over the krea2 latent/TE caches.
"""

import argparse
import gc
import json
import logging
import math
import os
import random
import sys
from multiprocessing import Value

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fizgig.dataset.config import (
    BlueprintGenerator,
    ConfigSanitizer,
    generate_dataset_group_by_blueprint,
    load_user_config,
)
from fizgig.krea2.utils import load_krea2_dit
from fizgig.krea2.sampling import gather_valid_text, prepare
from fizgig.networks.lora import create_network
from fizgig.training.metadata import ARCHITECTURE_KREA2
from fizgig.training.train_utils import LossRecorder

logger = logging.getLogger(__name__)


def _apply_context_lora(target, path, strength, *, device, dtype):
    """Load a context LoRA and apply it FROZEN + ACTIVE on `target` (the base DiT during
    training, or the Turbo at preview time). The context and the trainable/preview LoRA each
    wrap the forward and contribute additively; gradients never flow to the context. Returns
    the network so the caller can keep a reference (and free it after previews)."""
    from safetensors.torch import load_file
    from fizgig.networks.lora import create_network_from_weights, ensure_kohya_lora_state_dict
    # Normalize foreign formats (PEFT / diffusers / ComfyUI `diffusion_model.*`, LyCORIS) to
    # kohya keys so create_network_from_weights' lora_down scan finds the modules — without
    # this a diffusers-format context LoRA yields 0 modules. Mirrors Klein's load_lora.
    sd = ensure_kohya_lora_state_dict(load_file(path))
    net = create_network_from_weights(None, float(strength), sd, None, target, for_inference=True)
    net.apply_to(text_encoders=None, unet=target, apply_text_encoder=False, apply_unet=True)
    net.load_state_dict(sd, strict=False)
    net.to(device=device, dtype=dtype).eval()
    net.requires_grad_(False)
    return net


def load_dit_for_training(
    raw_path: str,
    *,
    network_dim: int = 32,
    network_alpha: float = 32,
    fp8_scaled: bool = True,
    quant_4bit: bool = False,
    blocks_to_swap: int = 0,
    gradient_checkpointing: bool = True,
    context_lora_path: str = None,
    context_lora_strength: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load the RAW DiT (frozen base, optionally fp8) and apply a trainable full-model LoRA.
    An optional frozen Context LoRA is applied to the base first, so the new LoRA learns to
    coexist with it (the context stays active during previews too).

    quant_4bit: QLoRA-style 4-bit (NF4) frozen base — halves DiT residency (~14 GB fp8 → ~5.6 GB)
    so a full LoRA trains on a 10-12 GB card with no block swap. Mutually exclusive with block
    swap (weights live in _nf4_packed, not .weight). Loads the base bf16 on CPU and NF4-quantizes
    the block Linears onto the GPU layer-by-layer (peak VRAM never holds the whole bf16 model).
    Reuses the same target/exclude keys as the fp8 path (`blocks.` minus mod./norm/txtfusion)."""
    if quant_4bit:
        # NF4 quantizes from bf16 (cleaner than fp8->NF4 double-quant), staged on CPU, and can't
        # coexist with block swap — force both here so callers can't misconfigure it.
        fp8_scaled = False
        blocks_to_swap = 0
        loading_device = "cpu"
    else:
        loading_device = "cpu" if blocks_to_swap > 0 else device
    dit = load_krea2_dit(raw_path, device=device, dtype=dtype, fp8_scaled=fp8_scaled,
                         loading_device=loading_device)
    dit.requires_grad_(False)  # frozen base (QLoRA-style)
    if quant_4bit:
        from fizgig.krea2.utils import KREA2_FP8_OPTIMIZATION_TARGET_KEYS, KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS
        from fizgig.modules.nf4 import apply_nf4_quantization
        n_q = apply_nf4_quantization(
            dit, target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
            exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS, compute_device=torch.device(device))
        dit.to(device)  # move the remaining (non-quantized) bf16 modules to the GPU
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"NF4 4-bit base active: {n_q} Linears quantized; DiT resident on {device}.")
    if gradient_checkpointing:
        dit.enable_gradient_checkpointing()

    # Context LoRA: frozen + active on the base BEFORE the trainable LoRA, so the trainable
    # one wraps the context-included forward (both additive; grads only flow to the trainable).
    if context_lora_path:
        logger.info(f"context LoRA: {os.path.basename(context_lora_path)} @ {context_lora_strength} (frozen, active)")
        _apply_context_lora(dit, context_lora_path, context_lora_strength, device=device, dtype=dtype)

    network = create_network(None, "lora_unet", 1.0, network_dim, network_alpha, None, [], dit)
    network.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    network.requires_grad_(True)
    network.to(device=device, dtype=dtype)
    return dit, network


def _get_lin_function(x1, y1, x2, y2):
    """Linear map through (x1,y1)-(x2,y2): f(x) = m*x + b. Used to schedule the flow shift `mu`
    from image-token count (musubi's get_lin_function)."""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


# Krea 2 resolution->mu schedule (musubi `krea2_shift`): token count maps to mu, shift = exp(mu).
# Endpoints match krea2 inference defaults (minres 256, maxres 1280 at align 16):
#   x1 = (256//16)**2 = 256, x2 = (1280//16)**2 = 6400, y1 = 0.5, y2 = 1.15.
_KREA2_MU = _get_lin_function(256, 0.5, 6400, 1.15)


def sample_krea2_timesteps(bsize: int, num_img_tokens: int, device, sigmoid_scale: float = 1.0) -> torch.Tensor:
    """Krea 2 'krea2_shift' timestep sampling — a faithful port of the musubi krea2_train recipe.

    The base t is **logit-normal** (sigmoid of a standard normal), so timesteps concentrate near the
    middle instead of being uniform. Uniform sampling (the old code) dumps far too much mass on the
    high-noise end, where the flow-matching velocity is intrinsically hard to predict — that inflates
    the loss AND skews the training signal away from the validated reference recipe. The shift is
    resolution-dependent (shift = exp(mu), mu from the image-token count), not a fixed 2.5.

        t_base = sigmoid(randn * sigmoid_scale)
        t      = (t_base * shift) / (1 + (shift - 1) * t_base)
    """
    mu = _KREA2_MU(num_img_tokens)
    shift = math.exp(mu)
    t = (torch.randn(bsize, device=device) * sigmoid_scale).sigmoid()
    return (t * shift) / (1.0 + (shift - 1.0) * t)


def compute_loss(dit, latent, hidden_states, attention_mask, *, shift=2.5, dtype=torch.bfloat16):
    """Flow-matching training loss for Krea 2.

    latent:        (B, 16, h, w)         — cached Qwen-Image VAE latent
    hidden_states: (B, seq, layers, dim) — cached Qwen3-VL multi-layer stack
    attention_mask:(B, seq) bool         — cached validity mask

    `shift` is kept for signature compatibility but no longer used: krea2_shift derives the flow
    shift from the image resolution (see sample_krea2_timesteps), matching the musubi reference.
    """
    device = next(p for p in dit.parameters()).device
    B = latent.shape[0]
    latent = latent.to(device=device, dtype=dtype)
    patch = dit.config.patch

    noise = torch.randn_like(latent)
    # krea2_shift: logit-normal base + resolution-dependent shift, over the image-token count
    # (latent grid // patch). Replaces the old uniform-u sampler that over-weighted high-noise t
    # and inflated the loss.
    num_img_tokens = (latent.shape[-2] // patch) * (latent.shape[-1] // patch)
    t = sample_krea2_timesteps(B, num_img_tokens, device)
    t_ = t.view(B, 1, 1, 1).to(dtype)
    noised = (1.0 - t_) * latent + t_ * noise
    target = noise - latent  # flow-matching velocity

    txt, txtmask = gather_valid_text(hidden_states.to(device=device, dtype=dtype), attention_mask.to(device))
    img_tokens, pos, mask = prepare(noised, txt.shape[1], patch, txtmask)
    target_tokens, _, _ = prepare(target, txt.shape[1], patch, txtmask)

    with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
        pred = dit(img=img_tokens, context=txt, t=t.to(dtype), pos=pos, mask=mask)
    # Return the mean drawn timestep alongside the loss so the passive per-image loss logger can
    # normalize for noise level (the caller ignores it when logging is off).
    return F.mse_loss(pred.float(), target_tokens.float()), float(t.mean().item())


class _Krea2Collator:
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
                         network_dim, network_alpha, dtype, extra=None):
    """Save a resumable training-state dir matching Klein's naming: <name>-<NNNNNN>-state/.

    NNNNNN is the number of COMPLETED epochs (= the next 0-indexed epoch to run). The dir
    holds the LoRA weights, the optimizer state, RNG states, and a small JSON. The GUI's
    _detect_latest_state_dir finds the highest-numbered one and passes it to --resume."""
    state_dir = os.path.join(output_dir, f"{output_name}-{epoch:06d}-state")
    os.makedirs(state_dir, exist_ok=True)
    _save_lora(network, os.path.join(state_dir, "lora.safetensors"), network_dim, network_alpha, dtype)
    if optimizer is not None:   # None under fused backward (per-parameter optimizers)
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
    from safetensors.torch import load_file
    network.load_state_dict(load_file(os.path.join(state_dir, "lora.safetensors")), strict=False)
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


class AdaptiveLR:
    """Bi-directional plateau LR tracker — a faithful port of Klein's adaptive_lr logic.

    Each epoch boundary: probe UP ×1.25 on steady loss descent (patience 2); reduce DOWN ×0.5
    on loss plateau (patience ramp) or a stability signal. On a stability event it blends the
    LoRA weights 70/30 toward the previous epoch's snapshot and restores the optimizer state
    (kills bad Adam momentum). Klein's stability signals are grad-clip ratio + weight-norm
    growth; krea2 has no grad clipping, so weight-norm growth (>30%) is the stability signal.

    State (streaks/best_loss/prev_weight_norm) is JSON round-trippable for pause/resume; the
    CPU rollback snapshot is in-memory only (too big to persist) — so the first post-resume
    epoch can't roll back, exactly as in Klein. Call epoch_boundary() at each epoch end."""

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
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
        lr_str = f"{cur_lr:.2e}" if new_lr == cur_lr else f"{cur_lr:.2e}->{new_lr:.2e}"
        wn_str = f"{weight_growth*100:+.0f}%" if weight_growth is not None else "—"
        logger.info(f"[adaptive_lr] epoch {epoch + 1}: loss={current_loss:.4f} lr={lr_str} "
                    f"wnorm_Δ={wn_str} | {action} ({reason})")
        self.prev_weight_norm = cur_wn
        self._snapshot(network, optimizer)


def _build_bf16_master(raw_path: str, dit) -> dict:
    """CPU bf16 copy of every fp8-patched block Linear — the source of truth for rotation.

    Read straight from the RAW file (which is bf16 on disk) rather than dequantizing the GPU
    copy: the GPU weights have already been through fp8, so dequantizing them would bake the
    quantization error into the master and we'd fine-tune a degraded model.
    """
    import torch.nn as _nn
    from safetensors.torch import load_file

    wanted = set()
    for bi, block in enumerate(dit.blocks):
        for name, m in block.named_modules():
            if isinstance(m, _nn.Linear) and hasattr(m, "scale_weight"):
                wanted.add(f"blocks.{bi}.{name}.weight")
    # txtfusion sits outside dit.blocks, so rotation never reaches it — but it's the stack
    # that fuses the text embeddings, so it's held always-on rather than left frozen.
    txtf = getattr(dit, "txtfusion", None)
    if txtf is not None:
        for name, m in txtf.named_modules():
            if isinstance(m, _nn.Linear) and hasattr(m, "scale_weight"):
                wanted.add(f"txtfusion.{name}.weight")

    sd = load_file(raw_path)          # mmap'd; we copy out only the keys we need
    master, missing = {}, []
    for key in sorted(wanted):
        t = sd.get(key)
        if t is None:
            missing.append(key)
            continue
        master[key] = t.to("cpu", dtype=torch.bfloat16).clone()
    del sd
    gc.collect()
    total_gb = sum(v.numel() * v.element_size() for v in master.values()) / 1e9
    logger.info("[ft-rotation] bf16 master: %d tensors, %.1f GB in CPU RAM%s",
                len(master), total_gb,
                f" ({len(missing)} keys missing from the RAW file — those stay frozen)" if missing else "")
    if missing:
        logger.warning("[ft-rotation] missing master keys, e.g. %s", missing[:3])
    return master


def _save_full_checkpoint(rotator, raw_path: str, path: str, extra_metadata=None):
    """Write the fine-tuned model: the RAW checkpoint with trained block weights replaced.

    Everything the rotator never touches (norms, embeddings, txtfusion, I/O layers) is copied
    through from the original, so the result is a complete, loadable Krea 2 checkpoint.
    """
    from safetensors.torch import load_file, save_file

    sd = load_file(raw_path)
    trained = rotator.master_state_dict()
    replaced = 0
    for k, v in trained.items():
        if k in sd:
            sd[k] = v.to(torch.bfloat16)
            replaced += 1
    meta = {"fizgig_finetune": "krea2-rotation", "fizgig_trained_tensors": str(replaced)}
    if extra_metadata:
        meta.update({str(k): str(v) for k, v in extra_metadata.items()})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_file(sd, path, metadata=meta)
    size_gb = os.path.getsize(path) / 1e9
    logger.info("[ft-rotation] saved full checkpoint (%d/%d tensors trained, %.1f GB) -> %s",
                replaced, len(sd), size_gb, path)
    del sd, trained
    gc.collect()


def _save_lora(network, path, network_dim, network_alpha, dtype, extra_metadata=None):
    metadata = {
        "ss_network_module": "fizgig.krea2 (lora_unet, all-Linear)",
        "ss_network_dim": str(network_dim),
        "ss_network_alpha": str(network_alpha),
        "ss_architecture": ARCHITECTURE_KREA2,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    network.save_weights(path, dtype, metadata)


# --- in-training previews (sample the fp8 Turbo with the live LoRA) -----------
def encode_sample_prompts(te_path, prompts, *, ref_image=None, vision_megapixels=1.0, device="cuda"):
    """Pre-encode the sample prompts once (Qwen3-VL), freeing the encoder afterwards.
    Returns a list of (txt, txtmask) on CPU, fed straight to sampling.sample at preview time.

    `ref_image` (a PIL image or path) routes a reference through Qwen3-VL's vision path so the
    samples become visually aware of it ('prompt from a picture' — Krea 2's reference mechanism)."""
    from fizgig.krea2.utils import load_krea2_text_encoder
    from fizgig.krea2 import sampling

    pil = None
    if ref_image:
        from PIL import Image
        pil = ref_image if hasattr(ref_image, "convert") else Image.open(ref_image)

    enc = load_krea2_text_encoder(te_path, dtype=torch.bfloat16, device=device)
    out = []
    for p in prompts:
        images = [[pil]] if pil is not None else None
        txt, txtmask, _, _ = sampling.encode_prompts(enc, [p], cfg=False,
                                                     images=images, vision_megapixels=vision_megapixels)
        out.append((txt.cpu(), txtmask.cpu()))
    del enc
    torch.cuda.empty_cache()
    return out


def _read_sample_override(output_dir):
    """Live sample override written by the GUI to <output_dir>/.sample_override.json.

    Returns {prompt, seed, width, height, ref_image} while active, else None. ref_image (if set)
    is routed through Qwen3-VL's vision path (Krea 2's reference mechanism)."""
    path = os.path.join(output_dir, ".sample_override.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        prompt = str(d.get("prompt", "")).strip()
        ref = str(d.get("ref_image", "")).strip()
        # Active on a prompt OR a reference — a reference with an empty prompt is a valid
        # 'generate from this picture' override (the Qwen3-VL vision path handles the rest).
        if prompt or ref:
            return {"prompt": prompt,
                    "seed": int(d.get("seed", 1234)),
                    "width": int(d.get("width", 1024)),
                    "height": int(d.get("height", 1024)),
                    "ref_image": ref}
    except Exception:
        pass
    return None


def _apply_caption_updates(output_dir, group, te_path, device, dit, blocks_to_swap, loss_watch, epoch,
                           *, auto_recaption=False, trigger_word=None, trigger_position="start",
                           recaptioned=None,
                           image_dir=None, caption_ext=".txt"):
    """Live caption repair (Problem Images window). Consume <output_dir>/loss_log/caption_updates.json
    ({item_key: new_caption}), re-encode each caption with Qwen3-VL, and OVERWRITE the item's
    text-embedding cache file — the collate re-reads that file from disk every step, so the very
    next epoch trains on the corrected caption. Also resets the image's loss-watch history (its
    stuck record reflects the old caption). Never raises into the training loop.

    auto_recaption: additionally re-caption CONFIRMED-STUCK images with the same Qwen3-VL (it's a
    full VLM with a real LM head — the captioner ships inside the training stack), appending
    "<trigger_word>, " (leading) when one is set. Max TWO attempts per image per run (`recaptioned` is a
    {key: attempts} dict): attempt 1 = standard caption; if the image re-confirms stuck after its
    history reset (~5-6 epochs later, i.e. the first caption demonstrably failed), attempt 2 =
    exhaustive-detail caption; after that it's permanently human-review. A manual edit already
    queued for a key always wins over the auto path. Both jobs share one DiT park + one
    text-encoder load.

    The GUI separately rewrites the .txt for manual edits; the auto path writes the .txt itself
    (image_dir + caption_ext from the dataset TOML) so fixes survive future re-caches. The 8 GB
    text encoder won't co-fit with the resident training DiT on smaller cards, so the DiT is
    parked on CPU around the encode (same dance as previews)."""
    path = os.path.join(output_dir, "loss_log", "caption_updates.json")
    updates = {}
    processing = path + ".processing"
    if os.path.exists(path):
        try:
            os.replace(path, processing)  # atomic claim — GUI edits during processing land in a fresh file
            with open(processing, encoding="utf-8") as f:
                updates = {str(k): str(v).strip() for k, v in json.load(f).items() if str(v).strip()}
        except Exception:
            logger.warning("[caption-fix] could not read caption_updates.json — skipping", exc_info=True)
            return

    # Auto-recaption candidates: confirmed stuck, not already handled this run, not manually
    # queued (the human's edit wins), and the source image must be findable on disk.
    auto_todo = []
    if auto_recaption and loss_watch is not None and image_dir and os.path.isdir(image_dir):
        confirmed = {k for k, v in loss_watch.verdicts.items() if v == "stuck"}
        for k in sorted(confirmed):
            attempts = recaptioned.get(k, 0) if recaptioned is not None else 0
            if k in updates or attempts >= 2:
                continue
            for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                p = os.path.join(image_dir, os.path.basename(k) + ext)
                if os.path.exists(p):
                    auto_todo.append((k, p, attempts + 1))
                    break

    if not updates and not auto_todo:
        if os.path.exists(processing):
            os.remove(processing)
        return
    if not te_path:
        logger.warning("[caption-fix] caption work is pending but no text encoder path was passed "
                       "(--text_encoder). Leaving the queue for a run with previews configured.")
        if updates:  # put the claim back (atomic + merged — the GUI may have queued more edits)
            try:
                newer = {}
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        newer = json.load(f)
                merged = {**updates, **newer}  # newer GUI edits win
                with open(path + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(merged, f, indent=2)
                os.replace(path + ".tmp", path)
                os.remove(processing)
            except Exception:
                pass
        return

    # item_key -> ItemInfo (training items come from the cache-driven path, so item_key is the
    # image basename without extension — same key the loss watch and the GUI use).
    items = {}
    for ds in group.datasets:
        bm = getattr(ds, "batch_manager", None)
        if bm is None:
            continue
        for bucket in bm.buckets.values():
            for it in bucket:
                items[str(it.item_key)] = it
    # todo entries: (key, ItemInfo, caption, attempt) — attempt 0 = manual edit, 1/2 = auto.
    todo = [(k, items[k], cap, 0) for k, cap in updates.items() if k in items]
    for k in updates:
        if k not in items:
            logger.warning(f"[caption-fix] '{k}' not found in the training set — skipped")
    auto_todo = [(k, p, a) for k, p, a in auto_todo if k in items]
    if not todo and not auto_todo:
        if os.path.exists(processing):
            os.remove(processing)
        return

    logger.info(f"[caption-fix] epoch boundary {epoch}: {len(todo)} manual edit(s), "
                f"{len(auto_todo)} stuck image(s) to auto-recaption...")
    dit.to("cpu")
    if getattr(dit, "_nf4_quantized", False):
        from fizgig.modules.nf4 import move_nf4_to_device
        move_nf4_to_device(dit, "cpu")
    gc.collect()
    torch.cuda.empty_cache()
    ok = False
    try:
        from fizgig.krea2.utils import load_krea2_text_encoder
        from fizgig.krea2.caching import encode_and_save_text
        from fizgig.krea2.embedder import generate_caption
        encoder = load_krea2_text_encoder(te_path, dtype=torch.bfloat16, device=device)

        # Auto-recaption: the SAME loaded VLM describes what's actually in the stuck image.
        # The trigger goes FIRST by default, matching what the Captions tab writes — a dataset
        # must not end up with the trigger leading on some images and trailing on others.
        # Leading is also the right call when the trigger is a real name (base-model
        # fine-tuning): the name is the subject, not an afterthought. trigger_position="end"
        # restores the weaker trailing claim, which suits a conditional trigger on a LoRA.
        # Attempt 2 (the first caption demonstrably failed) goes exhaustive-detail.
        for k, img_path, attempt in auto_todo:
            try:
                cap = generate_caption(encoder, img_path, detailed=(attempt >= 2))
                if trigger_word:
                    cap = (f"{cap}, {trigger_word}" if str(trigger_position) == "end"
                           else f"{trigger_word}, {cap}")
                cap_path = os.path.join(image_dir, os.path.basename(k) + caption_ext)
                try:
                    with open(cap_path, "w", encoding="utf-8") as f:
                        f.write(cap)
                except Exception:
                    logger.warning(f"[auto-recaption] could not write {cap_path} — the live run is "
                                   f"fixed but a future re-cache will use the old caption")
                todo.append((k, items[k], cap, attempt))
                logger.info(f"[auto-recaption] {os.path.basename(k)} (attempt {attempt}/2"
                            f"{', detailed' if attempt >= 2 else ''}): \"{cap[:110]}"
                            f"{'…' if len(cap) > 110 else ''}\"")
            except Exception:
                logger.warning(f"[auto-recaption] captioning failed for {os.path.basename(k)} — "
                               f"skipped (will retry next boundary)", exc_info=True)

        if not todo:
            del encoder
            if os.path.exists(processing):
                os.remove(processing)
            return
        for _, item, cap, _auto in todo:
            item.caption = cap
        for i in range(0, len(todo), 4):  # small chunks — captions pad to the longest in the batch
            encode_and_save_text(encoder, [item for _, item, _, _ in todo[i:i + 4]])
        del encoder
        ok = True
        # Mark auto-recaptioned keys only AFTER a successful encode — a failed boundary must be
        # allowed to retry them (their captions are re-queued in the failure path below).
        if recaptioned is not None:
            for k, _, _, attempt in todo:
                if attempt > 0:
                    recaptioned[k] = max(recaptioned.get(k, 0), attempt)
        if loss_watch is not None:
            for k, _, _, _ in todo:
                loss_watch.reset_key(k)
            # After the 2nd (detailed) AI caption, the benefit of the doubt is spent: if the
            # image re-confirms stuck, it goes STRAIGHT to the LR floor — no escalation ladder.
            # reset_key cleared any prior mark, so a manual human edit (attempt 0) restores hope.
            for k, _, _, attempt in todo:
                if attempt >= 2:
                    loss_watch.mark_incorrigible(k)
        # Ack for the GUI (row badge "caption re-encoded @ epoch N" / "AI re-captioned").
        applied_path = os.path.join(output_dir, "loss_log", "caption_updates_applied.json")
        applied = {}
        try:
            if os.path.exists(applied_path):
                with open(applied_path, encoding="utf-8") as f:
                    applied = json.load(f)
        except Exception:
            applied = {}
        for k, _, cap, attempt in todo:
            applied[k] = {"epoch": epoch, "caption": cap, "auto": attempt > 0, "attempt": attempt}
        # Atomic write — the GUI polls this file for the row badges.
        with open(applied_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(applied, f, indent=2)
        os.replace(applied_path + ".tmp", applied_path)
        if os.path.exists(processing):
            os.remove(processing)
        logger.info(f"[caption-fix] {len(todo)} caption(s) re-encoded — next epoch trains on the "
                    f"fixed text. Loss-watch history reset for: "
                    + ", ".join(os.path.basename(k) for k, _, _, _ in todo))
    except Exception:
        logger.warning("[caption-fix] re-encode failed — training continues on the old captions; "
                       "the edits stay queued and will be retried next epoch.", exc_info=True)
        # Put the claim back, merging any edits the GUI queued while we were processing. The
        # already-generated auto captions re-queue as if manual — no need to regenerate them.
        try:
            newer = {}
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    newer = json.load(f)
            auto_caps = {k: cap for k, _, cap, attempt in todo if attempt > 0}
            merged = {**updates, **auto_caps, **newer}  # newer GUI edits win
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)
            os.replace(path + ".tmp", path)
            if os.path.exists(processing):
                os.remove(processing)
        except Exception:
            pass
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        if blocks_to_swap > 0:
            dit.move_to_device_except_swap_blocks(torch.device(device))
            dit.switch_block_swap_for_training()
        else:
            dit.to(device)
        if getattr(dit, "_nf4_quantized", False):
            from fizgig.modules.nf4 import move_nf4_to_device
            move_nf4_to_device(dit, device)
        else:
            dit.to(device)
        if getattr(dit, "_nf4_quantized", False):
            from fizgig.modules.nf4 import move_nf4_to_device
            move_nf4_to_device(dit, device)
        dit.train()
    return ok


def sample_previews(turbo_path, ae, encoded_prompts, lora_sd, out_dir, epoch, *,
                    output_name="krea2", steps=8, cfg_scale=1.0, width=512, height=512,
                    seed=42, context_lora_path=None, context_lora_strength=1.0,
                    blocks_to_swap=0, int8=False, device="cuda"):
    """Load the (clean) pre-quant fp8 Turbo, apply the current LoRA LIVE (no merge -> no grid),
    and render each pre-encoded prompt. Turbo is freed afterwards.

    `blocks_to_swap` > 0 puts the Turbo on forward-only block swap so previews fit smaller cards
    (mirrors Klein's Distilled sample-model auto-swap). Order mirrors load_dit_for_training: load
    the base on CPU, apply the LoRA(s), then enable swap + place the resident blocks.

    Filenames follow the Fizgig samples-gallery pattern
    `{name}_e{epoch:06d}_{idx:02d}_{timestamp:14d}_{seed}.png` so the live preview gallery
    (which parses that exact format) picks them up — same as the Klein training path."""
    import datetime
    from fizgig.krea2.utils import load_krea2_dit
    from fizgig.networks.lora import create_network_from_weights
    from fizgig.krea2 import sampling

    _ld = "cpu" if blocks_to_swap > 0 else device
    turbo = load_krea2_dit(turbo_path, device=device, dtype=torch.bfloat16,
                           loading_device=_ld)  # prequant fp8 auto-detected
    if int8:
        # INT8 (W8A8) fast preview matmul — quantize the block Linears BEFORE the LoRA wraps them
        # (so the LoRA wraps the int8 forward) and before block swap (so the offloader stages int8).
        # Quantize on the load device so a swapped (CPU-loaded) model doesn't need the whole int8
        # model resident on GPU.
        from fizgig.modules.int8 import apply_int8_quantization
        from fizgig.krea2.utils import (KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
                                        KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS)
        apply_int8_quantization(turbo, target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
                                exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS,
                                compute_device=torch.device(_ld))
    # Context LoRA (frozen) goes on FIRST so previews match deployment: the trained LoRA runs
    # on top of the same context at the same strength it was trained with.
    ctx_net = None
    if context_lora_path:
        ctx_net = _apply_context_lora(turbo, context_lora_path, context_lora_strength,
                                      device=device, dtype=torch.bfloat16)
    net = create_network_from_weights(None, 1.0, lora_sd, None, turbo, for_inference=True)
    net.apply_to(text_encoders=None, unet=turbo, apply_text_encoder=False, apply_unet=True)
    # create_network_from_weights only builds the module STRUCTURE (sizes from dims/alphas);
    # the trained values must be loaded in, or the LoRA stays at its zero init (lora_up=0) and
    # contributes nothing — which made every epoch's preview identical. Mirrors the Klein path
    # (inference.py: apply_to -> load_state_dict(strict=False)).
    net.load_state_dict(lora_sd, strict=False)
    net.to(device=device, dtype=torch.bfloat16).eval()
    if blocks_to_swap > 0:
        from fizgig.krea2.offloading import BlockSwapConfig
        turbo.enable_block_swap(blocks_to_swap, BlockSwapConfig(torch.device(device), supports_backward=False))
        turbo.move_to_device_except_swap_blocks(torch.device(device))
        turbo.switch_block_swap_for_inference()
    turbo.eval()
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 14-digit timestamp
    paths = []
    for i, (txt, txtmask) in enumerate(encoded_prompts):
        with torch.no_grad():
            imgs = sampling.sample(turbo, ae, txt, txtmask, untxt=None, untxtmask=None,
                                   device=device, dtype=torch.bfloat16, width=width, height=height,
                                   steps=steps, cfg_scale=cfg_scale, mu=1.15, seed=seed + i)
        p = os.path.join(out_dir, f"{output_name}_e{epoch:06d}_{i:02d}_{ts}_{seed + i}.png")
        imgs[0].save(p)
        paths.append(p)
    del turbo, net, ctx_net
    torch.cuda.empty_cache()
    return paths


def train_krea2(
    raw_path: str,
    dataset_config: str,
    output_dir: str,
    output_name: str,
    *,
    network_dim: int = 32,
    network_alpha: float = 32,
    learning_rate: float = 1e-4,
    max_train_epochs: int = 10,
    save_every_n_epochs: int = 0,
    fp8_scaled: bool = True,
    quant_4bit: bool = False,
    blocks_to_swap: int = 0,
    shift: float = 2.5,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    # Effective batch = batch_size (1) x this. Grads accumulate over N micro-batches, then one
    # optimizer step. Per-image LR still applies per micro-batch (each image scales its own loss).
    gradient_accumulation_steps: int = 1,
    # LR schedule (step-level). Ignored when adaptive_lr is on — that watcher owns the LR.
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    lr_decay_steps: int = 0,
    lr_scheduler_num_cycles: int = 1,
    lr_scheduler_power: float = 1.0,
    # in-training previews (sample the fp8 Turbo with the live LoRA)
    sample_prompts: list = None,
    turbo_path: str = None,
    vae_path: str = None,
    te_path: str = None,
    sample_every_n_epochs: int = 0,
    sample_width: int = 512,
    sample_height: int = 512,
    sample_steps: int = 8,
    sample_seed: int = 42,
    sample_ref_image: str = None,
    preview_blocks_to_swap: int = 0,
    preview_int8: bool = False,
    log_per_image_loss: bool = False,
    per_image_lr: bool = False,
    auto_recaption: bool = False,
    warmup_look_outliers: bool = False,
    trigger_word: str = None,
    resume_state_dir: str = None,
    context_lora_path: str = None,
    context_lora_strength: float = 1.0,
    adaptive_lr: bool = False,
    adaptive_lr_min: float = 1e-5,
    adaptive_lr_max: float = 4e-4,
    # Rotating-block FULL fine-tune (experimental). >0 trains that many DiT blocks at a
    # time in bf16 while the rest stay fp8-frozen, rotating the window every N epochs.
    # No LoRA is trained in this mode — the output is a full model checkpoint.
    # Where the trigger word lands in auto-generated captions: "start" (matches the
    # Captions tab, and right for a real-name trigger) or "end" (weaker claim).
    trigger_position: str = "start",
    finetune_rotation: int = 0,
    finetune_rotate_every: int = 1,
    # "block" = contiguous depth slices; "component" = attn across ALL blocks, then
    # mlp — same VRAM, but every window spans the model's full depth.
    finetune_rotation_mode: str = "block",
    # Step each parameter's optimizer inside backward and free its grad immediately, so the
    # whole active window's gradients never coexist. Saves roughly the gradient footprint.
    finetune_fused_backward: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Native Krea 2 LoRA training: bucketed multi-resolution dataloader over the krea2 caches ->
    flow-matching loss -> AdamW -> save a ComfyUI-compatible LoRA. In-training Turbo previews +
    GUI wiring are layered on elsewhere."""
    torch.manual_seed(seed)

    shared_epoch = Value("i", 0)
    user_config = load_user_config(dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        user_config, argparse.Namespace(), architecture=ARCHITECTURE_KREA2)
    group = generate_dataset_group_by_blueprint(
        blueprint.dataset_group, training=True, num_timestep_buckets=None, shared_epoch=shared_epoch)
    if group.num_train_items == 0:
        raise RuntimeError("No training items — run the krea2 cache scripts first.")
    logger.info(f"Krea 2 training: {group.num_train_items} items, {max_train_epochs} epochs")

    # Preview setup: pre-encode prompts (frees the 8GB encoder) + load the VAE BEFORE the RAW DiT,
    # so the encoder never coexists with the resident base.
    do_previews = bool(sample_every_n_epochs and sample_prompts and turbo_path and vae_path and te_path)
    encoded_prompts = sample_ae = sample_dir = None
    if do_previews:
        from fizgig.krea2.vae_loader import load_vae
        logger.info(f"pre-encoding {len(sample_prompts)} sample prompt(s)"
                    f"{' with reference image' if sample_ref_image else ''}...")
        encoded_prompts = encode_sample_prompts(te_path, sample_prompts, ref_image=sample_ref_image, device=device)
        sample_ae = load_vae(vae_path, input_channels=3, device="cpu", disable_mmap=True)
        sample_dir = os.path.join(output_dir, "sample")

    ft_rotation = max(0, int(finetune_rotation or 0))
    ft_stream_frozen = False
    if ft_rotation:
        # Rotation owns the block weights: it swaps them between fp8-frozen and bf16-trainable
        # in place. Block swap moves whole blocks to CPU behind the offloader's back, and 4-bit
        # keeps weights packed in _nf4_packed — neither survives that swap, so both are off.
        if blocks_to_swap > 0:
            # Rotation brings its own swap policy (RotationOffloader): the trainable window is
            # pinned and every other block streams. The stock offloader can't express that —
            # it keeps a fixed contiguous prefix resident.
            logger.info("[ft-rotation] using rotation-aware block swap instead of the "
                        "fixed-prefix offloader (--blocks_to_swap value ignored).")
            blocks_to_swap = 0
            ft_stream_frozen = True
        if quant_4bit:
            logger.info("[ft-rotation] 4-bit base is incompatible with rotation "
                        "(weights live packed in _nf4_packed) — using fp8 instead.")
            quant_4bit = False
        if not fp8_scaled:
            logger.info("[ft-rotation] rotation needs the fp8-frozen base to fit — enabling fp8.")
            fp8_scaled = True
        if do_previews:
            # Previews render the Turbo with a LoRA applied; in FT mode there is no LoRA and
            # the trained weights live in the base itself. Sampling the fine-tuned model would
            # mean loading a second full checkpoint — out of scope for now.
            logger.info("[ft-rotation] in-training previews are disabled — evaluate saved "
                        "checkpoints in ComfyUI instead.")
            do_previews = False

    if quant_4bit and blocks_to_swap > 0:
        logger.info("[nf4] 4-bit base is incompatible with block swap (weights live in _nf4_packed) "
                    "— forcing blocks_to_swap=0.")
        blocks_to_swap = 0
    dit, network = load_dit_for_training(
        raw_path, network_dim=network_dim, network_alpha=network_alpha,
        fp8_scaled=fp8_scaled, quant_4bit=quant_4bit, blocks_to_swap=blocks_to_swap,
        context_lora_path=context_lora_path, context_lora_strength=context_lora_strength,
        device=device, dtype=dtype)
    if blocks_to_swap > 0 and not quant_4bit:
        from fizgig.krea2.offloading import BlockSwapConfig
        dit.enable_block_swap(blocks_to_swap, BlockSwapConfig(torch.device(device), supports_backward=True))
        dit.move_to_device_except_swap_blocks(torch.device(device))
        dit.switch_block_swap_for_training()
    dit.train()
    network.train()

    rotator = rot_schedule = None
    if ft_rotation:
        from fizgig.krea2.rotation import RotationSchedule, BlockRotator
        # The LoRA network stays created but frozen and zero-init, so it contributes nothing
        # to the forward. We're training the base weights themselves.
        network.requires_grad_(False)
        master = _build_bf16_master(raw_path, dit)
        rotator = BlockRotator(dit.blocks, master, key_prefix="blocks", device=device)
        if getattr(dit, "txtfusion", None) is not None:
            rotator.activate_always("txtfusion", dit.txtfusion)
        rot_schedule = RotationSchedule(len(dit.blocks), active=ft_rotation,
                                        rotate_every=finetune_rotate_every,
                                        mode=finetune_rotation_mode)
        if rot_schedule.mode == "component" and ft_stream_frozen:
            # Every block holds trainable Linears in component mode, so nothing can be
            # streamed out — there is no frozen block left to evict.
            logger.info("[ft-rotation] component mode trains part of every block — "
                        "block streaming disabled.")
            ft_stream_frozen = False
        if ft_stream_frozen:
            from fizgig.krea2.rotation import RotationOffloader
            # Injected as the DiT's offloader: the forward already calls wait_for_block /
            # submit_move_blocks_forward whenever blocks_to_swap is truthy, so no model change.
            # Window 0 here; the epoch loop re-pins to the correct window on the first
            # iteration (and on resume, since want != rotator.active triggers a rotation).
            dit.offloader = RotationOffloader(dit.blocks, torch.device(device),
                                              rot_schedule.active_at(0))
            dit.blocks_to_swap = 1
            logger.info("[ft-rotation] streaming frozen blocks from CPU — only the trainable "
                        "window stays resident.")
        logger.info("[ft-rotation] FULL FINE-TUNE — %s", rot_schedule.describe())
        if rot_schedule.cycle_epochs > max_train_epochs:
            logger.warning("[ft-rotation] a full cycle needs %d epochs but max_train_epochs=%d — "
                           "blocks after window %d will NEVER train this run.",
                           rot_schedule.cycle_epochs, max_train_epochs,
                           max_train_epochs // finetune_rotate_every)
        if adaptive_lr:
            # The watcher reads epoch-to-epoch loss movement as signal. Rotation changes which
            # weights are trainable at the boundary, so every rotation looks like a step change
            # and would trigger spurious reductions/rollbacks. Off for now.
            logger.info("[ft-rotation] adaptive LR disabled — rotation boundaries look like "
                        "instability to the plateau watcher.")
            adaptive_lr = False
    else:
        network.requires_grad_(True)

    def _make_optimizer(params_, quiet: bool = False):
        """Adafactor first in rotation mode: its factored state is ~10x smaller than Adam's,
        which is what keeps a full fine-tune inside 32 GB."""
        if ft_rotation:
            try:
                from transformers.optimization import Adafactor
                opt = Adafactor(params_, lr=learning_rate, scale_parameter=False,
                                relative_step=False, warmup_init=False)
                if not quiet:
                    logger.info("optimizer: Adafactor (rotation)")
                return opt
            except Exception as e:
                if not quiet:
                    logger.warning("Adafactor unavailable (%s) — falling back to AdamW8bit", e)
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(params_, lr=learning_rate)
            if not quiet:
                logger.info("optimizer: AdamW8bit")
            return opt
        except Exception:
            if not quiet:
                logger.info("optimizer: AdamW (bitsandbytes unavailable)")
            return torch.optim.AdamW(params_, lr=learning_rate)

    # ---- optimizer-in-backward (fused) ----
    # Normally every active parameter's gradient exists simultaneously at the peak of backward.
    # With this on, each parameter's optimizer steps the moment its grad is ready and the grad
    # is dropped, so only one parameter's gradient is live at a time.
    fused_backward = bool(finetune_fused_backward and ft_rotation)
    _fused = {"opts": {}, "handles": []}
    if finetune_fused_backward and not ft_rotation:
        logger.info("[fused-backward] only applies to rotation fine-tuning — ignored.")
    if fused_backward:
        if int(gradient_accumulation_steps or 1) > 1:
            logger.info("[fused-backward] incompatible with gradient accumulation (grads are "
                        "consumed and freed per parameter) — forcing accumulation to 1.")
        if max_grad_norm > 0:
            logger.info("[fused-backward] global grad-norm clipping needs all grads at once — "
                        "clipping is disabled in this mode.")

    def _attach_fused(params_):
        """One single-parameter optimizer per tensor, stepped from its grad hook."""
        for h in _fused["handles"]:
            h.remove()
        _fused["handles"].clear()
        _fused["opts"].clear()
        for p in params_:
            _fused["opts"][p] = _make_optimizer([p], quiet=True)

        def _hook(param):
            opt = _fused["opts"].get(param)
            if opt is not None:
                opt.step()
                opt.zero_grad(set_to_none=True)

        for p in params_:
            _fused["handles"].append(p.register_post_accumulate_grad_hook(_hook))
        logger.info("[fused-backward] %d per-parameter optimizers attached", len(params_))

    if ft_rotation:
        rotator.rotate_to(rot_schedule.active_at(0))
        params = rotator.trainable_params()
    else:
        params = list(network.get_trainable_params())
    if fused_backward:
        _attach_fused(params)
        optimizer = None            # stepping happens in the backward hooks
    else:
        optimizer = _make_optimizer(params)

    collator = _Krea2Collator(shared_epoch, group)
    loader = DataLoader(group, batch_size=1, shuffle=True, collate_fn=collator, num_workers=0)

    os.makedirs(output_dir, exist_ok=True)
    adaptive = AdaptiveLR(adaptive_lr_min, adaptive_lr_max) if adaptive_lr else None
    if adaptive:
        # The Min LR floor is authoritative over the LR box: starting below the declared floor is
        # contradictory, so the start is clamped UP to the floor (matches Klein).
        if learning_rate < adaptive_lr_min:
            logger.info(f"[adaptive_lr] starting LR {learning_rate:.3e} is below the Min LR floor — "
                        f"raising start to {adaptive_lr_min:.3e} (the floor overrides the LR box)")
            learning_rate = adaptive_lr_min
            for g in optimizer.param_groups:
                g["lr"] = adaptive_lr_min
        logger.info(f"[adaptive_lr] ENABLED — start_lr={learning_rate:.3e} "
                    f"min_lr={adaptive_lr_min:.3e} max_lr={adaptive_lr_max:.3e}")

    global_step = 0
    start_epoch = 0
    # Resume: restore LoRA + optimizer + RNG + (start_epoch, global_step) from a saved state dir.
    if resume_state_dir and os.path.isdir(resume_state_dir):
        start_epoch, global_step, _resume_meta = _load_training_state(resume_state_dir, network, optimizer, device=device)
        if adaptive:
            adaptive.load_state_dict(_resume_meta.get("adaptive_lr_state"))
            logger.info(f"[resume] adaptive_lr state restored: best_loss={adaptive.best_loss} "
                        f"streaks g/b/s={adaptive.good_streak}/{adaptive.bad_streak}/{adaptive.stability_streak} "
                        f"stability_triggered={adaptive.stability_triggered}")
        logger.info(f"[resume] from {resume_state_dir}: continuing at epoch {start_epoch + 1}/{max_train_epochs} "
                    f"(global_step {global_step})")
    try:
        steps_per_epoch = len(loader)
    except TypeError:
        steps_per_epoch = group.num_train_items

    # ---- Step-level LR scheduler (cosine / linear / warmup / ...) ----
    # Mutually exclusive with adaptive LR by design: both write optimizer.param_groups[*]["lr"],
    # so a live scheduler would stomp the watcher's epoch decisions every step. Adaptive wins
    # (same rule as Klein, whose GUI also forces "constant" when adaptive is on).
    accum_requested = max(1, int(gradient_accumulation_steps or 1))
    # Fused backward consumes and frees each grad as it lands, so there is nothing
    # left to accumulate across micro-batches.
    accum = 1 if fused_backward else accum_requested
    if accum > 1:
        logger.info(f"[grad_accum] {accum} micro-batches per optimizer step "
                    f"(effective batch {accum}); ~{max(1, steps_per_epoch // accum)} updates/epoch")

    _sched_total_steps = math.ceil(steps_per_epoch / accum) * max_train_epochs

    def _rebuild_scheduler(opt, position: int):
        """Build the configured schedule against `opt`, wound forward to `position`
        optimizer-steps. Used at startup, on resume, and after every rotation (which
        replaces the optimizer, so the old scheduler's parameter refs are dead)."""
        if adaptive or not lr_scheduler or lr_scheduler == "constant":
            return None
        from diffusers.optimization import get_scheduler
        kwargs = {}
        if lr_scheduler == "cosine_with_restarts":
            kwargs["num_cycles"] = int(lr_scheduler_num_cycles)
        elif lr_scheduler == "polynomial":
            kwargs["power"] = float(lr_scheduler_power)
        s = get_scheduler(lr_scheduler, opt,
                          num_warmup_steps=int(lr_warmup_steps or 0),
                          num_training_steps=_sched_total_steps, **kwargs)
        # These schedules are pure functions of the step count, so re-deriving the position
        # is exact and needs no persisted state. Setting last_epoch then stepping once lands
        # the LR exactly where `position` calls to step() would have.
        if position > 0:
            s.last_epoch = position - 1
            s.step()
        return s

    scheduler = None
    if adaptive:
        if lr_scheduler and lr_scheduler != "constant":
            logger.info(f"[lr_scheduler] '{lr_scheduler}' ignored — adaptive LR is enabled and owns the LR.")
    elif lr_scheduler and lr_scheduler != "constant":
        # global_step counts micro-batches; the schedule's position is optimizer steps.
        scheduler = _rebuild_scheduler(optimizer, global_step // accum)
        logger.info(f"[lr_scheduler] {lr_scheduler} — warmup {int(lr_warmup_steps or 0)} / "
                    f"{_sched_total_steps} total steps, start lr={optimizer.param_groups[0]['lr']:.3e}"
                    + (f" (resumed at step {global_step})" if global_step > 0 else ""))
    elif lr_warmup_steps:
        logger.info("[lr_scheduler] warmup steps ignored — LR scheduler is 'constant'.")

    pause_flag = os.path.join(output_dir, ".pause_requested")
    # Progress + loss display exactly as Klein: one continuous tqdm bar over all steps with
    # a smoothed avr_loss in the postfix (the raw per-step loss is very noisy — batch size 1
    # plus a random flow-matching timestep each step — so the moving average is the signal).
    loss_recorder = LossRecorder()
    # Per-image loss watcher (experiment). Three tiers, all sharing one class:
    #   env FIZGIG_PERIMAGE_LOSS_LOG=1  -> passive JSONL log only (offline study)
    #   log_per_image_loss (GUI toggle) -> JSONL + per-epoch stuck-image detection report
    #   per_image_lr (GUI toggle)       -> detection + per-image loss multiplier (throttle stuck,
    #                                      boost healthy learned; safe per-image LR at batch size 1)
    from fizgig.training.loss_logger import PerImageLossWatch, is_enabled as _loss_log_env
    # Fresh (non-resume) run: clear the previous run's loss-log artifacts so the GUI's Problem
    # Images window never shows stale verdicts (problem_images.json only gets rewritten after the
    # new run's warmup — or never, if the toggles are off this run). The pending caption queue is
    # stale too (the .txt fixes are already applied by the startup text re-cache). The research
    # JSONL is rotated, not deleted — appending would mix runs and corrupt offline analysis.
    if not (resume_state_dir and os.path.isdir(resume_state_dir)):
        _ll = os.path.join(output_dir, "loss_log")
        for _f in ("problem_images.json", "problem_images.json.tmp",
                   "caption_updates_applied.json", "caption_updates_applied.json.tmp",
                   "caption_updates.json", "caption_updates.json.processing"):
            try:
                os.remove(os.path.join(_ll, _f))
            except OSError:
                pass
        _jsonl = os.path.join(_ll, "per_image_loss.jsonl")
        if os.path.exists(_jsonl):
            import time as _time
            try:
                os.replace(_jsonl, _jsonl + "." + _time.strftime("%Y%m%d%H%M%S") + ".bak")
            except OSError:
                pass
    # The watch + auto-recaption need the source images + caption extension — pull them from the
    # dataset TOML (recursive: the keys live under [general] / [[datasets]] depending on config).
    # Also used to load/store <image_dir>/fizgig_excluded.json (exclusions travel with the dataset).
    recaptioned = {}   # key -> AI recaption attempts used (max 2; 2nd is the detailed pass)
    ar_image_dir, ar_caption_ext = None, ".txt"
    watch_enabled = (log_per_image_loss or per_image_lr or auto_recaption
                     or warmup_look_outliers or _loss_log_env())
    if watch_enabled:
        def _find_toml_key(d, key):
            if isinstance(d, dict):
                if key in d:
                    return d[key]
                for v in d.values():
                    r = _find_toml_key(v, key)
                    if r is not None:
                        return r
            elif isinstance(d, list):
                for v in d:
                    r = _find_toml_key(v, key)
                    if r is not None:
                        return r
            return None
        ar_image_dir = _find_toml_key(user_config, "image_directory")
        ar_caption_ext = _find_toml_key(user_config, "caption_extension") or ".txt"
        if not (ar_image_dir and os.path.isdir(ar_image_dir)):
            ar_image_dir = None
    if auto_recaption:
        if ar_image_dir:
            logger.info(f"[auto-recaption] ON — stuck images re-captioned by Qwen3-VL from "
                        f"{ar_image_dir}" + (f" (trigger: '{trigger_word}')" if trigger_word else ""))
        else:
            logger.warning("[auto-recaption] image_directory not found in the dataset config "
                           "— auto-recaption disabled")
            auto_recaption = False
    loss_watch = None
    if watch_enabled:
        loss_watch = PerImageLossWatch(output_dir, apply_lr=per_image_lr,
                                       write_jsonl=log_per_image_loss,
                                       dataset_dir=ar_image_dir, caption_ext=ar_caption_ext)
        # Reconcile persisted exclusions against the actual training set (prune entries for
        # images that left the dataset; refuse a file that would exclude everything).
        _dataset_keys = {str(it.item_key)
                         for ds in group.datasets
                         if getattr(ds, "batch_manager", None) is not None
                         for bucket in ds.batch_manager.buckets.values()
                         for it in bucket}
        loss_watch.preflight(_dataset_keys)
        logger.info(f"[loss-watch] per-image loss watch ON (per_image_lr={per_image_lr})")
        if warmup_look_outliers:
            # LR warm-up for Look Consistency Filter outliers (tight angles, unusual views):
            # they keep their unique information but ease in at x0.4 -> x1.0 over the first
            # epochs instead of fighting the forming identity core at full strength. Scores are
            # saved by the Image Prep tab's Look Filter into the dataset folder.
            _look_path = os.path.join(ar_image_dir or "", "fizgig_look_scores.json")
            try:
                with open(_look_path, encoding="utf-8") as _f:
                    _look = json.load(_f)
                _cut = _look.get("cutoff")
                _scores = _look.get("scores") or {}
                if _cut is None:
                    logger.warning("[look-warmup] no cutoff in fizgig_look_scores.json (too few "
                                   "scored faces) — warm-up disabled this run")
                else:
                    _outliers = {k for k, v in _scores.items()
                                 if isinstance(v, (int, float)) and v < float(_cut)}
                    # Outliers no longer in the dataset were most likely marked + moved to
                    # excluded_by_look/ in the Look Filter — that's the tool working, not an
                    # error. Warm up only what is actually being trained.
                    _gone = sorted(_outliers - _dataset_keys)
                    _keys = _outliers & _dataset_keys
                    if _gone:
                        logger.info(f"[look-warmup] {len(_gone)} scored outlier(s) not in the "
                                    f"dataset (moved/excluded via the Look Filter) — skipped: "
                                    + ", ".join(_gone[:8]) + ("…" if len(_gone) > 8 else ""))
                    if _keys:
                        loss_watch.set_warmup_keys(_keys)
                        logger.info(f"[look-warmup] {len(_keys)} look-outlier image(s) on LR "
                                    f"warm-up ×0.4→×1.0 over the first epochs (released early "
                                    f"on improvement): " + ", ".join(sorted(_keys)[:8])
                                    + ("…" if len(_keys) > 8 else ""))
                    else:
                        logger.info("[look-warmup] no look-outliers present in the dataset — "
                                    "nothing to warm up")
            except FileNotFoundError:
                logger.warning("[look-warmup] fizgig_look_scores.json not found in the dataset "
                               "folder — run the Look Consistency Filter (Image Prep tab, scan "
                               "with 3 baselines) first; warm-up disabled this run")
            except Exception as _e:
                logger.warning(f"[look-warmup] could not load look scores ({_e}) — warm-up "
                               f"disabled this run")
        if resume_state_dir and os.path.isdir(resume_state_dir) and start_epoch > 0:
            # Resumed run: rebuild the watch's history by replaying its own JSONL (it appends
            # across pause/resume). The applied-captions ledger supplies the reset/incorrigible
            # timeline: recaptioned images re-enter with post-fix history only, and images whose
            # 2 AI attempts are spent go back on the exclusion track instead of getting a free
            # third life. Also restores `recaptioned` so the max-2 attempt cap survives resume.
            _resets = {}
            try:
                with open(os.path.join(output_dir, "loss_log", "caption_updates_applied.json"),
                          encoding="utf-8") as _f:
                    for _k, _info in json.load(_f).items():
                        _att = int(_info.get("attempt", 0) or 0)
                        _auto = bool(_info.get("auto"))
                        if _auto:
                            recaptioned[_k] = max(recaptioned.get(_k, 0), _att)
                        _resets[_k] = (int(_info.get("epoch", 0) or 0), _att, _auto)
            except Exception:
                pass
            loss_watch.resume_from_jsonl(up_to_epoch=start_epoch, resets=_resets)
    progress_bar = tqdm(total=steps_per_epoch * max_train_epochs, initial=global_step,
                        desc="steps", smoothing=0)
    pending_accum = 0  # micro-batches backward'd since the last optimizer step
    for epoch in range(start_epoch, max_train_epochs):
        shared_epoch.value = epoch + 1
        if rotator is not None:
            want = rot_schedule.active_at(epoch)
            if want != rotator.active:
                # New window: swap the blocks, then rebuild the optimizer. The old optimizer's
                # state refers to tensors that no longer require grad, and Adam moments for the
                # outgoing window are meaningless to the incoming one.
                if ft_stream_frozen:
                    # Pin the incoming window BEFORE the weight swap: activate() reads from the
                    # master onto the GPU, and the outgoing window must rejoin the stream pool.
                    dit.offloader.set_resident(want)
                rotator.rotate_to(want)
                _new_params = rotator.trainable_params()
                if fused_backward:
                    # Hooks and per-parameter optimizers belong to the OLD window's tensors —
                    # rebuild them or the incoming blocks would never step.
                    _attach_fused(_new_params)
                else:
                    optimizer = _make_optimizer(_new_params)
                params = _new_params
                if scheduler is not None and not fused_backward:
                    # Re-attach the schedule to the new optimizer at the current position.
                    _pos = scheduler.last_epoch
                    scheduler = _rebuild_scheduler(optimizer, _pos)
                logger.info("[ft-rotation] epoch %d: training blocks %s", epoch + 1, want)
        for i, batch in enumerate(loader):
            # Excluded images (two failed AI recaptions, still stuck) are skipped ENTIRELY: no
            # forward, no gradient, and no loss recorded — avr_loss stops carrying their permanent
            # error term. Step accounting (bar + global_step) stays consistent for resume math.
            if loss_watch is not None and loss_watch.is_excluded(batch.get("item_keys")):
                loss_recorder.drop(step=i)  # the slot leaves avr_loss — no stale/zero padding
                global_step += 1
                progress_bar.update(1)
                continue
            loss, t_used = compute_loss(dit, batch["latents"], batch["hidden_states"], batch["attention_mask"],
                                        shift=shift, dtype=dtype)
            # Per-image LR: scale THIS step's gradient by the image's multiplier (throttle stuck
            # images, boost healthy learned ones). Raw loss is still what gets recorded/averaged below,
            # so avr_loss and the global adaptive-LR watcher see unscaled numbers.
            step_mult = loss_watch.multiplier(batch.get("item_keys")) if loss_watch is not None else 1.0
            # Divide by the accumulation count so N micro-batches AVERAGE into one update rather
            # than summing (which would scale the effective LR by N).
            _scaled = loss * step_mult if step_mult != 1.0 else loss
            (_scaled / accum if accum > 1 else _scaled).backward()
            pending_accum += 1
            if fused_backward:
                # The per-parameter hooks already stepped and freed each grad during backward.
                pending_accum = 0
            elif pending_accum >= accum:
                # Gradient clipping to match the musubi reference (max_grad_norm default 1.0). 0 disables.
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending_accum = 0
            global_step += 1
            loss_recorder.add(epoch=epoch, step=i, loss=loss.item())
            if loss_watch is not None:
                loss_watch.observe(epoch=epoch + 1, step=global_step,
                                   item_keys=batch.get("item_keys"), timestep=t_used, loss=loss.item())
            # refresh=False so only update(1) draws the bar — otherwise set_postfix AND update each
            # force a refresh, which a captured (non-tty) stderr logs as two lines per step (the
            # "187, 187, 188, 188" doubling). Training itself is one step per iteration.
            progress_bar.set_postfix(avr_loss=f"{loss_recorder.moving_average:.4f}", refresh=False)
            progress_bar.update(1)
        # Flush a partial accumulation group at the epoch boundary: the epoch-end work (adaptive
        # LR decisions, rollback snapshot, state save) must see a settled optimizer, and leftover
        # grads must not leak into the next epoch.
        if pending_accum > 0 and not fused_backward:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending_accum = 0
        logger.info(f"epoch {epoch + 1}/{max_train_epochs}  avr_loss={loss_recorder.moving_average:.4f}  step={global_step}"
                    + (f"  lr={optimizer.param_groups[0]['lr']:.3e}" if (scheduler is not None and optimizer is not None) else ""))

        # Adaptive LR: epoch-boundary plateau tracker (before save/preview so they reflect the
        # post-adjustment state). Uses the smoothed avr_loss as the signal, like Klein.
        if adaptive:
            adaptive.epoch_boundary(epoch, loss_recorder.moving_average, network, optimizer)

        # Per-image loss watch: reclassify images (stuck/learning/easy), refresh next epoch's
        # multipliers, write loss_log/problem_images.json (the GUI's Problem Images popup reads it).
        if loss_watch is not None:
            loss_watch.epoch_boundary(epoch + 1)

        # Live caption repair: apply caption edits queued from the Problem Images window, and
        # (when enabled) auto-recaption confirmed-stuck images with the same Qwen3-VL — both
        # re-encode in place so the next epoch trains on the fixed captions.
        _apply_caption_updates(output_dir, group, te_path, device, dit, blocks_to_swap,
                               loss_watch, epoch + 1,
                               auto_recaption=auto_recaption, trigger_word=trigger_word,
                               trigger_position=trigger_position,
                               recaptioned=recaptioned, image_dir=ar_image_dir,
                               caption_ext=ar_caption_ext)

        if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0 and (epoch + 1) < max_train_epochs:
            if rotator is not None:
                _save_full_checkpoint(rotator, raw_path,
                                      os.path.join(output_dir, f"{output_name}-{epoch + 1:06d}.safetensors"))
            else:
                _save_lora(network, os.path.join(output_dir, f"{output_name}-{epoch + 1:06d}.safetensors"),
                           network_dim, network_alpha, dtype)

        if do_previews and (epoch + 1) % sample_every_n_epochs == 0:
            from safetensors.torch import load_file
            tmp = os.path.join(output_dir, "_sample_lora.safetensors")
            _save_lora(network, tmp, network_dim, network_alpha, dtype)
            logger.info(f"rendering previews (epoch {epoch + 1}) on the fp8 Turbo...")
            # The preview loads the fp8 Turbo (~13 GB) on top of the resident training DiT
            # (~14 GB fp8) + the VAE — two full models won't fit (OOMs ~30 GB on a 32 GB card).
            # Park the training DiT on CPU for the preview, then restore it (and its block-swap
            # placement) before the next epoch. Costs one CPU<->GPU round-trip per preview.
            dit.to("cpu")
            if getattr(dit, "_nf4_quantized", False):
                # NF4's packed weights + quant state are plain attributes that .to("cpu") ignores
                # (~6 GB would stay on the GPU), so move them explicitly to free the VRAM the
                # preview needs — restored in the finally below.
                from fizgig.modules.nf4 import move_nf4_to_device
                move_nf4_to_device(dit, "cpu")
            gc.collect()
            torch.cuda.empty_cache()
            try:
                # Live sample override (GUI status-bar panel) — model-agnostic prompt/seed/res
                # for the next preview. Encoded here (after the training DiT is on CPU) so the
                # text encoder has room. No override -> the configured pre-encoded prompts.
                ov = _read_sample_override(output_dir)
                if ov:
                    logger.info(f"[sample override] active — '{ov['prompt'][:60]}' "
                                f"seed={ov['seed']} {ov['width']}x{ov['height']}"
                                f"{' +ref' if ov.get('ref_image') else ''}")
                    prev_enc = encode_sample_prompts(te_path, [ov["prompt"]],
                                                     ref_image=ov.get("ref_image") or None, device=device)
                    prev_w, prev_h, prev_seed = ov["width"], ov["height"], ov["seed"]
                else:
                    prev_enc, prev_w, prev_h, prev_seed = encoded_prompts, sample_width, sample_height, sample_seed
                # Seed 0 means "random": pick a fresh seed for this preview so 0 isn't a fixed seed
                # (each epoch's sample differs). Covers the Samples-tab field and a 0 in the override.
                if prev_seed == 0:
                    prev_seed = random.randint(1, 2**31 - 1)
                    logger.info(f"[sample] seed 0 -> random {prev_seed}")
                sample_previews(turbo_path, sample_ae, prev_enc, load_file(tmp), sample_dir, epoch + 1,
                                output_name=output_name, steps=sample_steps, width=prev_w,
                                height=prev_h, seed=prev_seed,
                                context_lora_path=context_lora_path, context_lora_strength=context_lora_strength,
                                blocks_to_swap=preview_blocks_to_swap, int8=preview_int8, device=device)
            except Exception as _prev_err:
                # A preview failure — almost always CUDA OOM (the ~13 GB Turbo + the Qwen3-VL
                # encoder won't fit alongside the parked training DiT on a small card) — must NEVER
                # kill the run. Training and LoRA saving are independent of previews, so we log,
                # disable previews for the rest of this run (so we don't re-OOM every sample epoch),
                # and carry on. The training DiT is restored in the finally below.
                _oom = "out of memory" in str(_prev_err).lower()
                logger.warning(
                    f"[preview] epoch {epoch + 1} preview failed "
                    f"({'CUDA OOM — this card is too small for the Turbo preview' if _oom else type(_prev_err).__name__}); "
                    f"disabling previews for the rest of the run. Training continues and LoRAs still save normally."
                )
                do_previews = False
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                if blocks_to_swap > 0:
                    # Re-establish the training placement (non-swap blocks -> GPU, swap blocks -> CPU).
                    dit.move_to_device_except_swap_blocks(torch.device(device))
                    dit.switch_block_swap_for_training()
                else:
                    dit.to(device)
                if getattr(dit, "_nf4_quantized", False):
                    # Restore the 4-bit packed weights + quant state to the GPU (they were parked
                    # on CPU above; .to(device) doesn't touch them). NF4 forces blocks_to_swap=0.
                    from fizgig.modules.nf4 import move_nf4_to_device
                    move_nf4_to_device(dit, device)
            dit.train()
            network.train()

        # Graceful pause (GUI wrote <output_dir>/.pause_requested): save a full resumable
        # state at this epoch boundary and exit cleanly so the GPU frees. The GUI detects the
        # clean exit, records the paused state, and offers Resume. Same contract as Klein.
        if os.path.exists(pause_flag):
            logger.info(f"[pause] requested — saving state at epoch {epoch + 1} and exiting cleanly")
            _save_training_state(output_dir, output_name, network, optimizer,
                                 epoch=epoch + 1, global_step=global_step,
                                 network_dim=network_dim, network_alpha=network_alpha, dtype=dtype,
                                 extra={"adaptive_lr_state": adaptive.state_dict()} if adaptive else None)
            try:
                os.remove(pause_flag)
            except Exception:
                pass
            progress_bar.close()
            logger.info("[pause] state saved — exiting (exit 0).")
            sys.exit(0)

    progress_bar.close()
    if loss_watch is not None:
        loss_watch.close()
    out = os.path.join(output_dir, f"{output_name}.safetensors")
    # Record the context LoRA in metadata so users know to pair it at the same strength at
    # inference (the trained LoRA is context-dependent — same contract as Klein).
    extra = None
    if context_lora_path:
        extra = {"ss_context_lora": os.path.basename(context_lora_path),
                 "ss_context_lora_strength": str(context_lora_strength)}
    if rotator is not None:
        _save_full_checkpoint(rotator, raw_path, out, extra_metadata=extra)
        logger.info(f"saved fine-tuned checkpoint -> {out}")
        return out
    _save_lora(network, out, network_dim, network_alpha, dtype, extra_metadata=extra)
    logger.info(f"saved final LoRA -> {out}")
    return out
