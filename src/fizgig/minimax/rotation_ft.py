"""Rotating-window full fine-tune for MiniMax H3 — the int8-ConvRot adaptation.

Subclasses krea2's BlockRotator, overriding exactly the four quantization-coupled methods.
The Krea rotator swaps fp8-monkey-patched Linears; H3's frozen base is ConvRotInt8Linear,
which is a different animal in three ways this module exists to handle:

1. `weight` is a class-level @property (the dense TRUE-basis weight, computed on demand) and
   the module holds int8 `qdata` + fp32 `wscale` buffers instead of a Parameter. A Parameter
   assignment would land in `_parameters` but READS would keep resolving the property — so
   activation swaps the instance's `__class__` to nn.Linear (legal: ConvRotInt8Linear inherits
   it, no __slots__), which makes the property vanish and the stock F.linear forward take
   over. Deactivation swaps it back; the property and the activation-rotation forward return
   by themselves, nothing to stash.
2. There is no raw bf16 file for H3 (the int8 checkpoint IS the deployed ground truth), so
   the CPU master is built by DEQUANTIZING the int8 file per tensor — exact with respect to
   the function the reference deploys.
3. Write-back re-encodes with quantize_int8_convrot (rotate into the storage basis + per-row
   absmax scale). The master is updated FIRST, so training never round-trips through int8
   within a run; the one ~0.9%-class rounding happens per save, same error class the base's
   own storage carries.
"""

import logging
import os
from typing import Dict, List

import torch
import torch.nn as nn

from fizgig.krea2.rotation import BlockRotator
from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen, stream_save_file
from fizgig.minimax.convrot import (dequantize_int8_convrot, parse_comfy_quant,
                                    quantize_int8_convrot)

logger = logging.getLogger(__name__)


def _convrot_linears(module: nn.Module) -> List[tuple]:
    """(qualified_name, linear) for every ConvRot int8 Linear under `module`.

    Keyed on the qdata buffer, NOT the class name: while a window is active the instance's
    __class__ is nn.Linear, but the (emptied) qdata buffer entry survives — so an active
    module is still discoverable when rotate_to needs to deactivate it."""
    return [(n, m) for n, m in module.named_modules()
            if isinstance(m, nn.Linear) and hasattr(m, "qdata")]


class H3BlockRotator(BlockRotator):
    """BlockRotator for the int8-ConvRot MiniMax H3 base. See the module docstring."""

    def __init__(self, blocks: nn.ModuleList, master: Dict[str, torch.Tensor],
                 key_prefix: str = "blocks", device: str = "cuda"):
        super().__init__(blocks, master, key_prefix=key_prefix, device=device)
        # Model-level weight keys that were EVER trainable — the save path overlays exactly
        # these; untouched tensors copy through bit-exact from the source file instead of
        # round-tripping through the bf16 master (whose rounding could flip codes).
        self.touched: set = set()
        self._orig_class = {}            # id(linear) -> ConvRotInt8Linear subclass, for restore

    # ---- targets ------------------------------------------------------------------------
    def _targets(self, spec) -> List[tuple]:
        out = []
        if spec and isinstance(spec[0], str):
            for bi, block in enumerate(self.blocks):
                for lname, lin in _convrot_linears(block):
                    if any(lname.startswith(c) for c in spec):
                        out.append((self._key(bi, lname), lin))
        else:
            for bi in spec:
                for lname, lin in _convrot_linears(self.blocks[bi]):
                    out.append((self._key(bi, lname), lin))
        return out

    # ---- activation: int8 ConvRot (frozen) -> bf16 nn.Linear (trainable) ------------------
    def _activate_targets(self, targets) -> int:
        n = 0
        for key, lin in targets:
            w = self.master.get(key)
            if w is None:
                logger.warning("[h3-rotation] no master weight for %s — leaving frozen", key)
                continue
            self._orig_class[id(lin)] = type(lin)
            lin.__class__ = nn.Linear     # the weight @property vanishes with the class
            lin.weight = nn.Parameter(w.to(self.device, dtype=torch.bfloat16),
                                      requires_grad=True)
            # Free the codes but KEEP the buffer entries — ~0.39 GB reclaimed per block, and
            # the surviving qdata attribute is what keeps _convrot_linears finding this
            # module for deactivation.
            dev = lin.qdata.device
            lin.qdata = torch.empty(0, dtype=torch.int8, device=dev)
            lin.wscale = torch.empty(0, dtype=torch.float32, device=dev)
            if lin.bias is not None:
                lin.bias = nn.Parameter(lin.bias.detach().to(torch.bfloat16),
                                        requires_grad=True)
            self.touched.add(key)
            n += 1
        return n

    def _deactivate_targets(self, targets) -> int:
        n = 0
        for key, lin in targets:
            if key not in self.master:
                continue
            trained = lin.weight.detach()
            # Master is the source of truth — save BEFORE the lossy re-encode.
            self.master[key] = trained.to("cpu", dtype=torch.bfloat16).clone()
            q, s = quantize_int8_convrot(trained, getattr(lin, "rot", 256))
            del lin._parameters["weight"]          # back to the __init__ state
            lin.qdata = q.to(self.device)
            lin.wscale = s.to(self.device)
            if lin.bias is not None:
                lin.bias.requires_grad_(False)
            lin.__class__ = self._orig_class.pop(id(lin))
            n += 1
        return n

    # ---- always-on (token refiner) --------------------------------------------------------
    def activate_always(self, prefix: str, module: nn.Module) -> int:
        """H3's always-on analogue of txtfusion is the token refiner (+ condition_proj):
        UNQUANTIZED in the int8 checkpoint, so these are plain dense Linears that only need
        unfreezing. The ConvRot branch is a safety net — the base class's requires_grad_ on a
        @property result would be a silent no-op, so any quantized Linear that ever appears
        under an always-module must take the real swap."""
        n_swapped = n_direct = 0
        for lname, lin in [(n, m) for n, m in module.named_modules()
                           if isinstance(m, nn.Linear)]:
            key = f"{prefix}.{lname}.weight"
            if hasattr(lin, "qdata"):
                got = self._activate_targets([(key, lin)])
                n_swapped += got
            else:
                lin.weight.requires_grad_(True)
                self.touched.add(key)
                n_direct += 1
            if lin.bias is not None:
                lin.bias.requires_grad_(True)
        self.always.append((prefix, module))
        logger.info("[h3-rotation] always-on: %s (%d Linears trainable for the whole run%s)",
                    prefix, n_swapped + n_direct,
                    f"; {n_swapped} de-quantized, {n_direct} already dense" if n_swapped else "")
        return n_swapped + n_direct


def build_bf16_master_h3(int8_path: str, block_subset=None,
                         key_prefix: str = "blocks") -> Dict[str, torch.Tensor]:
    """The CPU bf16 master for an H3 rotation FT, dequantized from the int8 checkpoint.

    Streams tensor-by-tensor (never holds the file whole). block_subset limits the master to
    a --finetune_blocks selection — ~0.77 GB per block, so 20-49 costs ~23 GB instead of the
    full model's ~38.6. Keys match H3BlockRotator._key(): '<prefix>.<i>.<lname>.weight'."""
    allowed = None
    if block_subset is not None:
        allowed = tuple(f"{key_prefix}.{int(b)}." for b in block_subset)
    master: Dict[str, torch.Tensor] = {}
    n = 0
    with MemoryEfficientSafeOpen(int8_path) as f:
        keys = set(f.keys())
        for k in sorted(keys):
            if not k.endswith(".comfy_quant"):
                continue
            stem = k[: -len(".comfy_quant")]
            if not stem.startswith(f"{key_prefix}."):
                continue
            if allowed is not None and not stem.startswith(allowed):
                continue
            conf = parse_comfy_quant(f.get_tensor(k))
            dense = dequantize_int8_convrot(f.get_tensor(stem + ".weight"),
                                            f.get_tensor(stem + ".weight_scale"),
                                            conf, out_dtype=torch.bfloat16)
            master[stem + ".weight"] = dense.cpu()
            n += 1
    gb = sum(t.numel() * t.element_size() for t in master.values()) / 2 ** 30
    logger.info("[h3-rotation] bf16 master built: %d tensors, %.1f GB CPU RAM "
                "(dequantized from the int8 checkpoint — the deployed ground truth)", n, gb)
    if n == 0:
        raise RuntimeError(
            f"build_bf16_master_h3: no ConvRot tensors under '{key_prefix}.' in "
            f"{os.path.basename(int8_path)} — rotation FT needs the pre-quantized int8 "
            "checkpoint (the bf16 file has no comfy_quant markers and is not supported)")
    return master


def save_full_checkpoint_h3(rotator: H3BlockRotator, src_path: str, out_path: str,
                            extra_metadata: Dict[str, str] = None) -> str:
    """Write a full int8-ConvRot checkpoint: the source file with trained tensors overlaid.

    Trained ConvRot stems are re-encoded (quantize_int8_convrot) as they stream to disk —
    weight int8 + weight_scale reshaped to the SOURCE tensor's exact shape + comfy_quant
    copied verbatim — so the output loads everywhere the source does (ComfyUI, the loader,
    diff-to-LoRA). Trained dense tensors (the always-on refiner) overlay in the source
    dtype. Every untouched tensor copies through bit-exact. One tensor in memory at a time
    (stream_save_file); atomic via tmp + os.replace."""
    flushed = rotator.master_state_dict()
    touched_stems = {k[: -len(".weight")] for k in rotator.touched if k.endswith(".weight")}

    with MemoryEfficientSafeOpen(src_path) as src:
        src_meta = dict(src.metadata() or {})
        header = {k: src.header[k] for k in src.keys()}
        rot_by_stem = {}
        for stem in touched_stems:
            ck = stem + ".comfy_quant"
            if ck in header:
                conf = parse_comfy_quant(src.get_tensor(ck))
                rot_by_stem[stem] = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1

    meta = dict(src_meta)
    meta.update({"fizgig_finetune": "minimax-h3-rotation",
                 "fizgig_trained_tensors": str(len(rotator.touched)),
                 "fizgig_source_base": os.path.basename(src_path)})
    if extra_metadata:
        meta.update({str(k): str(v) for k, v in extra_metadata.items()})

    _DT = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
           "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
           "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}

    # Re-encode each trained stem ONCE: the weight producer computes (q, s) and parks the
    # scale for the weight_scale producer (source key order is not guaranteed). Scales are
    # [out, 1]-tiny, so a parked handful costs nothing. One reader handle serves every
    # untouched-tensor copy — the alternative (an open per tensor) re-parses the ~21 GB
    # file's header nine hundred times.
    _pending_scales: Dict[str, torch.Tensor] = {}
    _reader = {"f": None}

    def _encode(stem):
        return quantize_int8_convrot(flushed[stem + ".weight"], rot_by_stem.get(stem, 256))

    def make_producer(key):
        info = header[key]
        dt, shape = _DT[info["dtype"]], tuple(info["shape"])
        stem = None
        if key.endswith(".weight") and key[: -len(".weight")] in touched_stems:
            stem = key[: -len(".weight")]
            kind = "weight"
        elif key.endswith(".weight_scale") and key[: -len(".weight_scale")] in touched_stems:
            stem = key[: -len(".weight_scale")]
            kind = "scale"
        if stem is not None and dt == torch.int8 and kind == "weight":
            def _p(stem=stem, shape=shape):
                q, s = _encode(stem)
                _pending_scales[stem] = s
                return q.reshape(shape)
            return dt, shape, _p
        if stem is not None and kind == "scale":
            def _p(stem=stem, dt=dt, shape=shape):
                s = _pending_scales.pop(stem, None)
                if s is None:
                    s = _encode(stem)[1]
                return s.to(dt).reshape(shape)
            return dt, shape, _p
        if stem is not None and kind == "weight":
            # a touched DENSE tensor (always-on refiner): overlay in the source dtype
            def _p(key=key, dt=dt, shape=shape):
                return flushed[key].to(dt).reshape(shape)
            return dt, shape, _p

        def _p(key=key):
            return _reader["f"].get_tensor(key)
        return dt, shape, _p

    specs = {k: make_producer(k) for k in header}
    tmp = out_path + ".tmp"
    try:
        _reader["f"] = MemoryEfficientSafeOpen(src_path)
        try:
            stream_save_file(specs, tmp, metadata=meta)
        finally:
            _reader["f"].file.close()
        os.replace(tmp, out_path)
    except BaseException:
        # KeyboardInterrupt/SystemExit are exactly the truncation cases — never leave a
        # half-written file that LOOKS like a checkpoint.
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    logger.info("[h3-rotation] saved full checkpoint -> %s (%d trained tensors overlaid)",
                out_path, len(rotator.touched))
    return out_path
