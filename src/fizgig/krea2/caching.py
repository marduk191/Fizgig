"""Krea 2 caching: VAE latents (Qwen-Image VAE) + text-encoder outputs (Qwen3-VL-4B).

Reuses Fizgig's dataset framework (ItemInfo + safetensors cache files); only the encode +
the cache contents are Krea2-specific:

* Latents: the Qwen-Image VAE wants a frame axis, so images are encoded as (B, C, 1, H, W)
  and normalized (`(raw - mean) / std`) by `encode_pixels_to_latents`. The 1-frame axis is
  squeezed for storage -> (C, H, W), like the Klein cache; the trainer re-adds nothing
  (patchify works on (B, C, H, W)).
* Text: Qwen3-VL returns the MULTI-LAYER hidden-state stack (seq, num_select_layers, dim)
  plus a validity mask. Both are stored (the layerwise fusion happens inside the DiT).
"""

import logging
import os
from typing import List

import torch
from safetensors.torch import save_file

from fizgig.dataset.image_dataset import ItemInfo, MemoryEfficientSafeOpen, dtype_to_str
from fizgig.training.metadata import ARCHITECTURE_KREA2

logger = logging.getLogger(__name__)


# --- cache writers -----------------------------------------------------------
def save_latent_cache_krea2(item_info: ItemInfo, latent: torch.Tensor,
                            control_latents: list = None) -> None:
    """Save a Krea 2 VAE latent (3-D: C, H, W) to the item's latent cache file.

    control_latents: optional list of (C, H, W) latents for the item's control images
    (paired-image training — e.g. the SOURCE frame of a temporal-displacement pair).
    Written as `latent_control_{i}_{H}x{W}` — the exact key format the dataset layer
    already maps to the `latents_control_{i}` batch key (image_dataset.py)."""
    assert latent.dim() == 3, f"latent must be 3-D (C,H,W), got {tuple(latent.shape)}"
    _, H, W = latent.shape
    sd = {f"latent_{H}x{W}": latent.detach().cpu().contiguous()}
    for i, ctrl in enumerate(control_latents or []):
        assert ctrl.dim() == 3, f"control latent must be 3-D (C,H,W), got {tuple(ctrl.shape)}"
        _, cH, cW = ctrl.shape
        sd[f"latent_control_{i}_{cH}x{cW}"] = ctrl.detach().cpu().contiguous()
    for key, value in sd.items():
        if torch.isnan(value).any():
            logger.warning(f"NaN in {key} for {item_info.item_key} - replaced with 0")
            value[torch.isnan(value)] = 0
    metadata = {
        "architecture": ARCHITECTURE_KREA2,
        "width": str(item_info.original_size[0]),
        "height": str(item_info.original_size[1]),
        "dtype": dtype_to_str(latent.dtype),
        "format_version": "2.0.0",
    }
    os.makedirs(os.path.dirname(item_info.latent_cache_path), exist_ok=True)
    save_file(sd, item_info.latent_cache_path, metadata=metadata)


def save_text_encoder_output_cache_krea2(
    item_info: ItemInfo, hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> None:
    """Save the Qwen3-VL multi-layer hidden stack (seq, layers, dim) + validity mask (seq,)."""
    sd = {
        "hidden_states": hidden_states.detach().cpu().contiguous(),
        "attention_mask": attention_mask.detach().cpu().to(torch.bool).contiguous(),
    }
    if torch.isnan(sd["hidden_states"]).any():
        logger.warning(f"NaN in hidden_states for {item_info.item_key} - replaced with 0")
        sd["hidden_states"][torch.isnan(sd["hidden_states"])] = 0
    metadata = {
        "architecture": ARCHITECTURE_KREA2,
        "caption1": item_info.caption,
        "dtype": dtype_to_str(hidden_states.dtype),
        "format_version": "2.0.0",
    }
    os.makedirs(os.path.dirname(item_info.text_encoder_output_cache_path), exist_ok=True)
    save_file(sd, item_info.text_encoder_output_cache_path, metadata=metadata)


def load_text_encoder_output_cache_krea2(path: str):
    """Return (hidden_states (seq, layers, dim), attention_mask (seq,) bool) from a cache file."""
    with MemoryEfficientSafeOpen(path) as f:
        return f.get_tensor("hidden_states"), f.get_tensor("attention_mask").to(torch.bool)


# --- batch encoders (called by the cache scripts) ----------------------------
def _encode_pixel_batch(vae, arrays) -> torch.Tensor:
    """(H,W,C) uint8 arrays -> (B, C, 1, H, W) latents through the Qwen-Image VAE."""
    contents = torch.stack([torch.from_numpy(a) for a in arrays], dim=0)
    contents = contents.permute(0, 3, 1, 2).unsqueeze(2)  # (B, C, 1, H, W)
    contents = contents / 127.5 - 1.0
    with torch.no_grad():
        return vae.encode_pixels_to_latents(contents.to(vae.device, dtype=vae.dtype))


def encode_and_save_latents(vae, batch: List[ItemInfo]) -> None:
    """Encode an image batch through the Qwen-Image VAE and save each latent.

    Control images (item.control_content — the source half of a training pair) are encoded
    through the same VAE and written alongside. Within a caching batch all items share the
    same control count and shapes (the bucketing batch key includes control dims), so control
    index i can be batch-encoded across items."""
    contents = []
    for item in batch:
        content = item.content
        content = content[0] if isinstance(content, list) else content  # (H, W, C)
        contents.append(content[..., :3])  # strip alpha — RGBA PNGs arrive as (H, W, 4)
    latents = _encode_pixel_batch(vae, contents)  # (B, C, 1, H, W)

    n_controls = len(batch[0].control_content or []) if batch else 0
    control_latents = []  # [i] -> (B, C, 1, H, W)
    for i in range(n_controls):
        ctrl_arrays = [item.control_content[i][..., :3] for item in batch]
        control_latents.append(_encode_pixel_batch(vae, ctrl_arrays))

    for b, item in enumerate(batch):
        latent = latents[b].squeeze(1) if latents[b].dim() == 4 else latents[b]  # (C, 1, H, W) -> (C, H, W)
        ctrls = [cl[b].squeeze(1) if cl[b].dim() == 4 else cl[b] for cl in control_latents]
        logger.info(f"latent cache: {item.item_key} -> {tuple(latent.shape)}"
                    + (f" + {len(ctrls)} control" if ctrls else ""))
        save_latent_cache_krea2(item, latent, control_latents=ctrls or None)


@torch.no_grad()
def encode_and_save_text(encoder, batch: List[ItemInfo]) -> None:
    """Encode a caption batch through the Qwen3-VL conditioner and save each (hiddens, mask)."""
    captions = [item.caption for item in batch]
    hiddens, mask = encoder(captions)  # (B, seq, layers, dim), (B, seq) bool
    for b, item in enumerate(batch):
        save_text_encoder_output_cache_krea2(item, hiddens[b], mask[b])
