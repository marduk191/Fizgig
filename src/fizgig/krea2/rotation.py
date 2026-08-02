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

import gc
import logging
from concurrent.futures import ThreadPoolExecutor
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
                 order: Optional[Sequence[int]] = None, mode: str = "block",
                 components: Sequence[str] = ("attn", "mlp.gate", "mlp.up", "mlp.down"),
                 start_window: int = 0):
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive")
        self.n_blocks = int(n_blocks)
        self.active = max(1, min(int(active), self.n_blocks))
        self.rotate_every = max(1, int(rotate_every))
        # A resumed FT run is a fresh process: without an offset it restarts the cycle at
        # window 0, retraining early windows and starving late ones. start_window shifts the
        # whole schedule so "resume for the last epoch" can mean "train the last component".
        self.start_window = int(start_window)
        self.order = list(order) if order is not None else list(range(self.n_blocks))
        if sorted(self.order) != list(range(self.n_blocks)):
            raise ValueError("order must be a permutation of range(n_blocks)")
        # Default components are balanced by parameter count: a block is 30% attention and
        # 70% MLP, so a naive ("attn", "mlp") split makes the MLP window 2.3x the attention
        # one — big enough to OOM. Splitting the MLP into its three matrices gives four
        # windows of <=30% each.
        # "block": windows are contiguous slices of blocks (depth-split).
        # "component": windows are sub-block components spanning EVERY block — attention
        # across the whole model, then MLP. Roughly the same VRAM (attention is about half
        # a block) but each window sees the model's full depth, so a concept is learned by
        # every layer at once instead of by one depth slice at a time.
        self.mode = "component" if str(mode).startswith("comp") else "block"
        self.components = list(components)

    @property
    def n_windows(self) -> int:
        if self.mode == "component":
            return len(self.components)
        return (self.n_blocks + self.active - 1) // self.active

    @property
    def cycle_epochs(self) -> int:
        """Epochs needed for every block to have trained once."""
        return self.n_windows * self.rotate_every

    def window_at(self, epoch: int) -> int:
        """0-based window index for a 0-based epoch (wraps after a full cycle)."""
        return ((int(epoch) // self.rotate_every) + self.start_window) % self.n_windows

    def active_at(self, epoch: int):
        """Block indices (block mode) or component name prefixes (component mode)."""
        w = self.window_at(epoch)
        if self.mode == "component":
            return [self.components[w]]
        start = w * self.active
        return sorted(self.order[start:start + self.active])

    def describe(self) -> str:
        if self.mode == "component":
            return (f"component windows {self.components} across all {self.n_blocks} blocks, "
                    f"rotating every {self.rotate_every} epoch(s) — {self.n_windows} windows, "
                    f"{self.cycle_epochs} epochs per full cycle (every window spans full depth)")
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


def _all_linears(module: nn.Module) -> List[tuple]:
    """(qualified_name, linear) for every Linear, quantized or not.

    Needed for always-on modules: the training fp8 pass only targets "blocks." and excludes
    txtfusion, so those Linears are plain bf16 and have no scale_weight to filter on.
    """
    return [(n, m) for n, m in module.named_modules() if isinstance(m, nn.Linear)]


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
        # (key_prefix, module) pairs held trainable for the WHOLE run, never rotated out.
        self.always: List[tuple] = []

    def _key(self, block_idx: int, linear_name: str) -> str:
        return f"{self.key_prefix}.{block_idx}.{linear_name}.weight"

    # ---- always-on modules (e.g. txtfusion) ----
    def activate_always(self, prefix: str, module: nn.Module) -> int:
        """Make every patched Linear under `module` trainable for the entire run.

        txtfusion lives outside dit.blocks, so rotation would never reach it — yet it's the
        stack that fuses the text embeddings, i.e. where prompt-to-concept binding is shaped.
        Leaving it frozen would cap exactly the behaviour a fine-tune is usually after. It's
        small (a 3072-wide text path vs the 6144-wide main blocks) so holding it trainable
        throughout costs little.
        """
        n_swapped = n_direct = 0
        for lname, lin in _all_linears(module):
            key = f"{prefix}.{lname}.weight"
            if hasattr(lin, "scale_weight"):
                # fp8-quantized: same swap as a rotating block, sourced from the master.
                w = self.master.get(key)
                if w is None:
                    logger.warning("[rotation] no master weight for %s — leaving frozen", key)
                    continue
                self._patched_forward[id(lin)] = lin.__dict__.pop("forward", None)
                lin.weight = nn.Parameter(w.to(self.device, dtype=torch.bfloat16), requires_grad=True)
                n_swapped += 1
            else:
                # Already bf16 — the training fp8 pass targets "blocks." and excludes
                # txtfusion, so these are the real weights and just need unfreezing.
                lin.weight.requires_grad_(True)
                n_direct += 1
            if lin.bias is not None:
                lin.bias.requires_grad_(True)
        self.always.append((prefix, module))
        logger.info("[rotation] always-on: %s (%d Linears trainable for the whole run"
                    "%s)", prefix, n_swapped + n_direct,
                    f"; {n_swapped} de-quantized, {n_direct} already bf16" if n_swapped else "")
        return n_swapped + n_direct

    # ---- targets ----
    def _targets(self, spec) -> List[tuple]:
        """(key, linear) pairs for a window spec.

        spec is either block indices (block mode) or component name prefixes such as
        "attn"/"mlp" (component mode), in which case the window spans EVERY block.
        """
        out = []
        if spec and isinstance(spec[0], str):
            for bi, block in enumerate(self.blocks):
                for lname, lin in _linears_with_scale(block):
                    if any(lname.startswith(c) for c in spec):
                        out.append((self._key(bi, lname), lin))
        else:
            for bi in spec:
                for lname, lin in _linears_with_scale(self.blocks[bi]):
                    out.append((self._key(bi, lname), lin))
        return out

    # ---- activation: fp8 (frozen) -> bf16 (trainable) ----
    def _activate_targets(self, targets) -> int:
        n = 0
        for key, lin in targets:
            w = self.master.get(key)
            if w is None:
                logger.warning("[rotation] no master weight for %s — leaving frozen", key)
                continue
            # Stash the fp8 forward so deactivate() can put it back verbatim.
            self._patched_forward[id(lin)] = lin.__dict__.pop("forward", None)
            lin.weight = nn.Parameter(w.to(self.device, dtype=torch.bfloat16), requires_grad=True)
            if lin.bias is not None:
                lin.bias = nn.Parameter(lin.bias.detach().to(torch.bfloat16), requires_grad=True)
            n += 1
        return n

    def _deactivate_targets(self, targets) -> int:
        from fizgig.krea2.fp8_optimization_utils import (
            calculate_fp8_maxval, quantize_weight, fp8_linear_forward_patch,
        )
        max_value = calculate_fp8_maxval(4, 3)
        min_value = -max_value
        n = 0
        for key, lin in targets:
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
            sw = scale.to(self.device, dtype=lin.scale_weight.dtype).reshape(lin.scale_weight.shape)
            lin.scale_weight.copy_(sw)
            if lin.bias is not None:
                lin.bias.requires_grad_(False)
            saved = self._patched_forward.pop(id(lin), None)
            if saved is not None:
                lin.forward = saved
            else:
                def _fwd(self_, x, _p=fp8_linear_forward_patch):
                    return _p(self_, x, False, None)
                lin.forward = _fwd.__get__(lin, type(lin))
            n += 1
        return n

    def activate(self, spec) -> int:
        spec = list(spec)
        n = self._activate_targets(self._targets(spec))
        # Keep self.active in step — trainable_params() and master_state_dict() read it.
        merged = list(self.active) + [x for x in spec if x not in self.active]
        self.active = sorted(merged) if merged and not isinstance(merged[0], str) else merged
        logger.info("[rotation] activated %s (%d Linears now trainable)", spec, n)
        return n

    def deactivate(self, spec) -> int:
        spec = list(spec)
        n = self._deactivate_targets(self._targets(spec))
        self.active = [x for x in self.active if x not in spec]
        logger.info("[rotation] deactivated %s (%d Linears re-quantized)", spec, n)
        return n

    def rotate_to(self, spec) -> None:
        """Make exactly `spec` trainable, deactivating whatever else is active."""
        spec = list(spec)
        if spec == list(self.active):
            return
        if self.active:
            self._deactivate_targets(self._targets(list(self.active)))
            # Hand the freed blocks back before allocating the next window: without this the
            # old window's bf16 tensors sit in the allocator's reserve and the new window
            # OOMs against its own predecessor.
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
        self._activate_targets(self._targets(spec))
        self.active = spec
        logger.info("[rotation] window -> %s", spec)

    def trainable_params(self) -> List[nn.Parameter]:
        """Active-window params PLUS the always-on modules. Rebuilt each rotation, so the
        always-on params must be re-included every time or they'd silently stop updating."""
        params, seen = [], set()
        for _key, lin in self._targets(list(self.active)):
            for p in (lin.weight, lin.bias):
                if p is not None and p.requires_grad and id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        for _prefix, module in self.always:
            for p in module.parameters():
                if p.requires_grad and id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        return params

    def master_state_dict(self) -> Dict[str, torch.Tensor]:
        """The bf16 master, with currently-active blocks and always-on modules flushed in."""
        out = dict(self.master)
        for key, lin in self._targets(list(self.active)):
            if key in out:
                out[key] = lin.weight.detach().to("cpu", dtype=torch.bfloat16).clone()
        for prefix, module in self.always:
            # Export unconditionally: unquantized always-on Linears were never in the master
            # (they live on the GPU in bf16 from the start), but they HAVE been trained.
            for lname, lin in _all_linears(module):
                out[f"{prefix}.{lname}.weight"] = (
                    lin.weight.detach().to("cpu", dtype=torch.bfloat16).clone())
        return out


# ---------------------------------------------------------------------------
# Rotation-aware block swap
# ---------------------------------------------------------------------------

class RotationOffloader:
    """Keeps the trainable window pinned on GPU and streams every other block from CPU.

    The stock offloader keeps a contiguous PREFIX of blocks resident and swaps the suffix,
    pairing each eviction with a matching load so residency stays constant. Rotation needs
    the opposite: an arbitrary resident set (the active window) that MOVES every rotation.
    Rather than rework that pairing, this streams non-resident blocks just-in-time.

    Implements the same small interface the DiT's forward expects, so it can be dropped in
    as `dit.offloader` with no changes to model.py.

    Gradient checkpointing recomputes a block's forward *inside* its backward, so blocks are
    pulled back to the GPU on a backward PRE-hook (before the recompute) and evicted on the
    post-hook — evicting on the forward pass alone would leave the recompute with weights on
    the wrong device.
    """

    def __init__(self, blocks: nn.ModuleList, device: torch.device, resident: Iterable[int],
                 debug: bool = False, prefetch: bool = True):
        self.blocks = blocks
        self.device = torch.device(device)
        self.cpu = torch.device("cpu")
        self.resident = set(int(i) for i in resident)
        self.forward_only = False
        self.debug = debug
        self.moves = 0                      # instrumentation: transfers per step
        self._handles = []
        # Async plumbing: one worker thread issuing copies on a dedicated CUDA stream, so a
        # block's transfer overlaps the previous block's compute instead of stalling on it.
        self.cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.prefetch = bool(prefetch and self.cuda)
        self._pool = ThreadPoolExecutor(max_workers=1) if self.prefetch else None
        self._stream = torch.cuda.Stream(device=self.device) if self.prefetch else None
        self._futures = {}                  # block idx -> Future[cuda.Event | None]
        self._register_hooks()

    # -- placement helpers -------------------------------------------------
    def _to(self, idx: int, dev: torch.device):
        from fizgig.krea2.offloading import weighs_to_device
        weighs_to_device(self.blocks[idx], dev)
        self.moves += 1

    def _move_task(self, idx: int, dev: torch.device, after):
        """Runs on the worker thread, issuing the copy on the transfer stream."""
        torch.cuda.set_device(self.device.index if self.device.index is not None
                              else torch.cuda.current_device())
        with torch.cuda.stream(self._stream):
            if after is not None:
                # Don't yank weights out from under kernels that are still queued.
                self._stream.wait_event(after)
            self._to(idx, dev)
            ev = torch.cuda.Event()
            ev.record(self._stream)
        return ev

    def _submit(self, idx: int, dev: torch.device, after=None):
        # Track the DIRECTION, not just that something is in flight: an in-flight eviction
        # must never be mistaken for a completed load (that race put weights on CPU while the
        # forward was still running, and only showed up as a device mismatch deep in fp8).
        pending = self._futures.get(idx)
        if pending is not None:
            if pending[0] == dev:
                return                  # same direction already queued
            self._await(idx)            # opposite direction: settle it first
        self._futures[idx] = (dev, self._pool.submit(self._move_task, idx, dev, after))

    def _await(self, idx: int):
        """Settle any in-flight move for `idx`. Returns the device it moved to, or None."""
        pending = self._futures.pop(idx, None)
        if pending is None:
            return None
        dev, fut = pending
        ev = fut.result()
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
        return dev

    def _ensure_gpu(self, idx: int):
        if idx in self.resident:
            return
        landed = self._await(idx)
        if landed is not None and landed.type == self.device.type:
            return                      # a prefetch already put it on the GPU
        self._to(idx, self.device)

    def _evict(self, idx: int):
        if idx in self.resident:
            return
        if self.prefetch:
            # Queue the eviction behind the compute that just used this block.
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream())
            self._submit(idx, self.cpu, after=done)
        else:
            self._to(idx, self.cpu)

    def _prefetch_next(self, idx: int):
        if not self.prefetch or idx < 0 or idx >= len(self.blocks) or idx in self.resident:
            return
        self._submit(idx, self.device)

    # -- rotation ----------------------------------------------------------
    def set_resident(self, resident: Iterable[int]):
        """Re-pin after a rotation: the incoming window goes to GPU and stays, the outgoing
        window rejoins the streamed pool."""
        for idx in list(self._futures):     # settle everything before re-planning
            self._await(idx)
        new = set(int(i) for i in resident)
        for idx in sorted(new - self.resident):
            self._to(idx, self.device)
        for idx in sorted(self.resident - new):
            self._to(idx, self.cpu)
        self.resident = new
        logger.info("[rotation-swap] resident blocks now %s (others stream from CPU)", sorted(new))

    # -- interface the DiT forward calls -----------------------------------
    def prepare_block_devices_before_forward(self, blocks):
        for idx in list(self._futures):
            self._await(idx)
        for i in range(len(blocks)):
            self._to(i, self.device if i in self.resident else self.cpu)
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        # Warm the first streamed block so step 0 isn't a stall.
        self._prefetch_next(0 if 0 not in self.resident else 1)

    def wait_for_block(self, index: int):
        self._ensure_gpu(index)
        self._prefetch_next(index + 1)      # overlap the next load with this block's compute

    def submit_move_blocks_forward(self, blocks, index: int):
        self._evict(index)

    def set_forward_only(self, forward_only: bool):
        self.forward_only = bool(forward_only)

    # -- backward ----------------------------------------------------------
    def _register_hooks(self):
        for i, block in enumerate(self.blocks):
            def pre_hook(module, grad_output, _i=i):
                self._ensure_gpu(_i)
                self._prefetch_next(_i - 1)   # backward walks blocks in reverse
                return None

            def post_hook(module, grad_input, grad_output, _i=i):
                self._evict(_i)
                return None

            self._handles.append(block.register_full_backward_pre_hook(pre_hook))
            self._handles.append(block.register_full_backward_hook(post_hook))

    def remove(self):
        for idx in list(self._futures):
            self._await(idx)
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
