"""MiniMax H3 caching: video-VAE latents (24-ch) + Qwen3-VL-32B text-encoder states.

Reuses Fizgig's dataset framework (ItemInfo + safetensors cache files, same as Krea 2); only
the encode + the cache contents are H3-specific:

* Latents: the H3 video VAE encodes a still image as one frame -> [1, 24, 1, H/16, W/16]. The
  1-frame axis is squeezed for storage -> (24, H/16, W/16), keyed `latent_{H}x{W}` so the shared
  collate maps it to the batch's `latents`. The trainer re-adds the frame axis before the DiT.
* Text: the H3 conditioning is the raw layer-50 Qwen3-VL output -> (L, 5120). Stored as
  `hidden_states` (L, 5120) plus an all-valid `attention_mask` (L,), matching the Krea 2 text
  cache keys so the same collate path applies (batch size 1 -> no padding needed).
"""

import logging
import os
from typing import List

import torch
from safetensors.torch import save_file

from fizgig.dataset.image_dataset import ItemInfo, MemoryEfficientSafeOpen, dtype_to_str
from fizgig.training.metadata import ARCHITECTURE_MINIMAX

logger = logging.getLogger(__name__)


# --- cache writers -----------------------------------------------------------
def save_latent_cache_minimax(item_info: ItemInfo, latent: torch.Tensor) -> None:
    """Save an H3 VAE latent (3-D: C, H, W) to the item's latent cache file."""
    assert latent.dim() == 3, f"latent must be 3-D (C,H,W), got {tuple(latent.shape)}"
    _, H, W = latent.shape
    sd = {f"latent_{H}x{W}": latent.detach().cpu().contiguous()}
    for key, value in sd.items():
        if torch.isnan(value).any():
            logger.warning(f"NaN in {key} for {item_info.item_key} - replaced with 0")
            value[torch.isnan(value)] = 0
    metadata = {
        "architecture": ARCHITECTURE_MINIMAX,
        "width": str(item_info.original_size[0]),
        "height": str(item_info.original_size[1]),
        "dtype": dtype_to_str(latent.dtype),
        "format_version": "2.0.0",
    }
    os.makedirs(os.path.dirname(item_info.latent_cache_path), exist_ok=True)
    save_file(sd, item_info.latent_cache_path, metadata=metadata)


def save_text_encoder_output_cache_minimax(item_info: ItemInfo, hidden_states: torch.Tensor) -> None:
    """Save the raw Qwen3-VL-32B layer-50 states (L, 5120) + an all-valid mask (L,)."""
    hidden_states = hidden_states.detach().cpu().contiguous()
    if torch.isnan(hidden_states).any():
        logger.warning(f"NaN in hidden_states for {item_info.item_key} - replaced with 0")
        hidden_states[torch.isnan(hidden_states)] = 0
    sd = {
        "hidden_states": hidden_states,
        "attention_mask": torch.ones(hidden_states.shape[0], dtype=torch.bool),
    }
    metadata = {
        "architecture": ARCHITECTURE_MINIMAX,
        "caption1": item_info.caption,
        "dtype": dtype_to_str(hidden_states.dtype),
        "format_version": "2.0.0",
    }
    os.makedirs(os.path.dirname(item_info.text_encoder_output_cache_path), exist_ok=True)
    save_file(sd, item_info.text_encoder_output_cache_path, metadata=metadata)


def load_text_encoder_output_cache_minimax(path: str):
    """Return (hidden_states (L, 5120), attention_mask (L,) bool) from a cache file."""
    with MemoryEfficientSafeOpen(path) as f:
        return f.get_tensor("hidden_states"), f.get_tensor("attention_mask").to(torch.bool)


# --- batch encoders (called by the cache scripts) ----------------------------
def encode_and_save_latents(vae, batch: List[ItemInfo]) -> None:
    """Encode an image batch through the H3 video VAE and save each 24-ch latent."""
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    contents = []
    for item in batch:
        content = item.content
        content = content[0] if isinstance(content, list) else content  # (H, W, C) uint8
        contents.append(torch.from_numpy(content))
    contents = torch.stack(contents, dim=0).permute(0, 3, 1, 2)  # (B, C, H, W)
    contents = contents / 127.5 - 1.0                            # -> [-1, 1]
    with torch.no_grad():
        latents = vae.encode(contents.to(device, dtype=dtype))  # (B, 24, 1, H/16, W/16)
    for b, item in enumerate(batch):
        latent = latents[b]                        # (24, 1, H/16, W/16)
        latent = latent.squeeze(1) if latent.dim() == 4 else latent  # -> (24, H/16, W/16)
        logger.info(f"latent cache: {item.item_key} -> {tuple(latent.shape)}")
        save_latent_cache_minimax(item, latent)


@torch.no_grad()
def encode_and_save_text(encoder, batch: List[ItemInfo]) -> None:
    """Encode a caption batch through the Qwen3-VL-32B TE and save each (L, 5120) state.

    One forward for the whole batch rather than one per caption: the nvfp4-resident encoder
    dequantizes 351 weights per FORWARD, so this is where the caching pass spends its time.
    encode_batch right-pads, which is exactly equivalent for a causal stack (verified in
    tests/diag_batch_encode.py) and memoizes, so a repeated caption costs nothing."""
    embeds = encoder.encode_batch([item.caption for item in batch])
    for item, hidden in zip(batch, embeds):
        save_text_encoder_output_cache_minimax(item, hidden[0])
