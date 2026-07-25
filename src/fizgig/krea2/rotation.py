"""Rotating-block fine-tuning for Krea 2 — full-rank updates on a consumer card.

Naive full fine-tuning of Krea 2 needs ~78 GB (bf16 weights + grads + Adam state).
This trains a *rotating slice* of the model instead: the DiT stays fp8-frozen
(~14 GB resident) and only N blocks at a time are swapped up to trainable bf16.
Over a cycle every block gets trained, but VRAM only ever holds grads and
optimizer state for the active slice.

The technique is LISA (Layerwise Importance Sampled AdamW) applied to a diffusion
DiT rather than an LLM — unproven in this setting, hence the experiment branch.

Precision: a CPU-resident bf16 **master copy** is the source of truth for every
trainable weight. Blocks activate FROM the master and write back TO it, so the
lossy fp8 copy on the GPU is only ever used for the frozen forward pass — training
never round-trips through fp8, which would quantize away exactly the small updates
we're trying to learn.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RotationSchedule:
    """Which block indices are trainable at a given epoch.

    `active` blocks train for `rotate_every` epochs, then the window advances.
    A full cycle (every block trained once) takes ceil(n_blocks / active) *
    rotate_every epochs — worth knowing when picking max_train_epochs, since
    fewer than one full cycle means part of the model never trains at all.
    """

    def __init__(self, n_blocks: int, active: int = 4, rotate_every: int = 1,
                 order: Optional[Sequence[int]] = None):
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive")
        self.n_blocks = int(n_blocks)
        self.active = max(1, min(int(active), self.n_blocks))
        self.rotate_every = max(1, int(rotate_every))
        self.order = list(order) if order is not None else list(range(self.n_blocks))
        if sorted(self.order) != list(range(self.n_blocks)):
            raise ValueError("order must be a permutation of range(n_blocks)")

    @property
    def n_windows(self) -> int:
        return (self.n_blocks + self.active - 1) // self.active

    @property
    def cycle_epochs(self) -> int:
        """Epochs needed for every block to have trained once."""
        return self.n_windows * self.rotate_every

    def window_at(self, epoch: int) -> int:
        """0-based window index for a 0-based epoch (wraps after a full cycle)."""
        return (int(epoch) // self.rotate_every) % self.n_windows

    def active_at(self, epoch: int) -> List[int]:
        w = self.window_at(epoch)
        start = w * self.active
        return sorted(self.order[start:start + self.active])

    def describe(self) -> str:
        return (f"{self.active} of {self.n_blocks} blocks per window, rotating every "
                f"{self.rotate_every} epoch(s) — {self.n_windows} windows, "
                f"{self.cycle_epochs} epochs per full cycle")


def _linears_with_scale(module: nn.Module) -> List[tuple]:
    """(qualified_name, linear) for every fp8-patched Linear under `module`."""
    out = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear) and hasattr(m, "scale_weight"):
            out.append((name, m))
    return out


class BlockRotator:
    """Swaps DiT blocks between fp8-frozen and bf16-trainable, in place.

    `master` maps the model-level weight key -> CPU bf16 tensor. It is the
    authoritative copy: activation reads from it, deactivation writes to it.
    """

    def __init__(self, blocks: nn.ModuleList, master: Dict[str, torch.Tensor],
                 key_prefix: str = "blocks", device: str = "cuda",
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                 quantization_mode: str = "block", block_size: int = 64):
        self.blocks = blocks
        self.master = master
        self.key_prefix = key_prefix
        self.device = device
        self.fp8_dtype = fp8_dtype
        self.quantization_mode = quantization_mode
        self.block_size = block_size
        self.active: List[int] = []
        self._patched_forward = {}      # id(linear) -> bound fp8 forward, for restore

    def _key(self, block_idx: int, linear_name: str) -> str:
        return f"{self.key_prefix}.{block_idx}.{linear_name}.weight"

    # ---- activation: fp8 (frozen) -> bf16 (trainable) ----
    def activate(self, block_ids: Iterable[int]) -> int:
        n = 0
        for bi in block_ids:
            block = self.blocks[bi]
            for lname, lin in _linears_with_scale(block):
                key = self._key(bi, lname)
                w = self.master.get(key)
                if w is None:
                    logger.warning("[rotation] no master weight for %s — leaving frozen", key)
                    continue
                # Stash the fp8 forward so deactivate() can put it back verbatim.
                self._patched_forward[id(lin)] = lin.__dict__.pop("forward", None)
                lin.weight = nn.Parameter(w.to(self.device, dtype=torch.bfloat16),
                                          requires_grad=True)
                if lin.bias is not None:
                    lin.bias = nn.Parameter(lin.bias.detach().to(torch.bfloat16),
                                            requires_grad=True)
                n += 1
        self.active = sorted(set(self.active) | set(block_ids))
        logger.info("[rotation] activated blocks %s (%d Linears now trainable)",
                    sorted(block_ids), n)
        return n

    # ---- deactivation: write back to master, re-quantize to fp8 ----
    def deactivate(self, block_ids: Iterable[int]) -> int:
        from fizgig.krea2.fp8_optimization_utils import (
            calculate_fp8_maxval, quantize_weight, fp8_linear_forward_patch,
        )
        max_value = calculate_fp8_maxval(4, 3)
        min_value = -max_value
        n = 0
        for bi in block_ids:
            block = self.blocks[bi]
            for lname, lin in _linears_with_scale(block):
                key = self._key(bi, lname)
                if key not in self.master:
                    continue
                trained = lin.weight.detach()
                # Master is the source of truth — save BEFORE the lossy re-quantize.
                self.master[key] = trained.to("cpu", dtype=torch.bfloat16).clone()
                q, scale = quantize_weight(key, trained.float(), self.fp8_dtype,
                                           max_value, min_value,
                                           quantization_mode=self.quantization_mode,
                                           block_size=self.block_size)
                lin.weight = nn.Parameter(q.to(self.device), requires_grad=False)
                sw = scale.to(self.device, dtype=lin.scale_weight.dtype).reshape(
                    lin.scale_weight.shape)
                lin.scale_weight.copy_(sw)
                if lin.bias is not None:
                    lin.bias.requires_grad_(False)
                saved = self._patched_forward.pop(id(lin), None)
                if saved is not None:
                    lin.forward = saved
                else:   # rebind a fresh fp8 forward
                    def _fwd(self_, x, _p=fp8_linear_forward_patch):
                        return _p(self_, x, False, None)
                    lin.forward = _fwd.__get__(lin, type(lin))
                n += 1
        self.active = [b for b in self.active if b not in set(block_ids)]
        logger.info("[rotation] deactivated blocks %s (%d Linears re-quantized)",
                    sorted(block_ids), n)
        return n

    def rotate_to(self, block_ids: Sequence[int]) -> None:
        """Make exactly `block_ids` trainable, deactivating whatever else is active."""
        want = set(block_ids)
        cur = set(self.active)
        if want == cur:
            return
        if cur - want:
            self.deactivate(sorted(cur - want))
        if want - cur:
            self.activate(sorted(want - cur))

    def trainable_params(self) -> List[nn.Parameter]:
        params = []
        for bi in self.active:
            for p in self.blocks[bi].parameters():
                if p.requires_grad:
                    params.append(p)
        return params

    def master_state_dict(self) -> Dict[str, torch.Tensor]:
        """The bf16 master, with any currently-active blocks flushed into it."""
        out = dict(self.master)
        for bi in self.active:
            for lname, lin in _linears_with_scale(self.blocks[bi]):
                key = self._key(bi, lname)
                if key in out:
                    out[key] = lin.weight.detach().to("cpu", dtype=torch.bfloat16).clone()
        return out
