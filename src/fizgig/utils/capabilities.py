"""What this machine can actually do, probed once and cached.

Fizgig used to pick memory settings from VRAM alone, which produced a bad outcome on 16 GB
cards: fp8 doesn't fit, so it fell back to swapping 20 of 28 blocks to CPU every step. Measured
on an RTX 5090 (Krea 2, 36 images @ 0.25 MP, batch 1):

    fp8, no swap    0.85 s/it   20.1 GB   12.5% CPU
    fp8, swap 20    3.09 s/it   12.3 GB   49.9% CPU     <- what 16 GB cards were getting
    NF4, no swap    0.70 s/it   13.8 GB   14.0% CPU

Block swap costs 4.4x the time and 4x the CPU, and NF4 fits the same card outright. So the
choice is a *strategy*, not a swap count — and it needs to know what the hardware supports,
because fp8 matmul is Ada+ while NF4 and int8 go back further.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Capabilities:
    has_cuda: bool = False
    device_name: str = "cpu"
    sm: tuple = (0, 0)
    vram_gb: float = 0.0        # card total, as reported
    vram_free_gb: float = 0.0   # actually available right now — what decisions must use
    fp8_matmul: bool = False       # torch._scaled_mm on fp8 — Ada (sm 8.9) and newer
    int8_matmul: bool = False      # torch._scaled_mm on int8 — NOT a thing; fp8-only API
    int8_matmul_train: bool = False  # torch._int_mm — the real int8 GEMM, Turing and newer
    cudnn_attention: bool = False  # PyTorch SDPA cuDNN backend
    flash_attn: bool = False       # the flash_attn package
    bitsandbytes: bool = False     # required for NF4
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        if not self.has_cuda:
            return "no CUDA device"
        flags = [f"sm_{self.sm[0]}{self.sm[1]}"]
        used = self.vram_gb - self.vram_free_gb
        vram = (f"{self.vram_free_gb:.1f} GB free of {self.vram_gb:.0f} GB"
                + (f" ({used:.1f} GB already in use)" if used > 1.0 else ""))
        for name, ok in (("fp8", self.fp8_matmul), ("int8", self.int8_matmul_train),
                         ("cuDNN-attn", self.cudnn_attention), ("flash", self.flash_attn),
                         ("nf4", self.bitsandbytes)):
            flags.append(f"{name} {'yes' if ok else 'no'}")
        return f"{self.device_name}, {vram} — " + " · ".join(flags)


def _probe_scaled_mm(dtype) -> bool:
    """Actually run a tiny _scaled_mm rather than trusting a compute-capability table."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        a = torch.zeros((16, 16), dtype=dtype, device="cuda")
        b = torch.zeros((16, 16), dtype=dtype, device="cuda").t()
        one = torch.ones((), dtype=torch.float32, device="cuda")
        torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        return True
    except Exception:
        return False


def _probe_int_mm() -> bool:
    """torch._int_mm is the int8 GEMM — a different API from _scaled_mm, which is fp8-only.
    Confusing the two is why int8 first looked unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        a = torch.zeros((32, 32), dtype=torch.int8, device="cuda")
        torch._int_mm(a, a.t().contiguous())
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def detect() -> Capabilities:
    caps = Capabilities()
    try:
        import torch
    except Exception:
        caps.notes.append("torch not importable")
        return caps

    if not torch.cuda.is_available():
        caps.notes.append("CUDA unavailable")
        return caps

    caps.has_cuda = True
    props = torch.cuda.get_device_properties(0)
    caps.device_name = props.name
    caps.sm = torch.cuda.get_device_capability(0)
    caps.vram_gb = props.total_memory / (1024 ** 3)
    try:
        # What is ACTUALLY available: a "16 GB" card reports ~15.9 GiB total, and a browser or
        # a running ComfyUI can be holding several more. Deciding from total would hand those
        # users a config that OOMs or silently falls back to swapping.
        free_b, _total_b = torch.cuda.mem_get_info(0)
        caps.vram_free_gb = free_b / (1024 ** 3)
    except Exception:
        caps.vram_free_gb = caps.vram_gb
        caps.notes.append("could not read free VRAM — using card total")

    caps.fp8_matmul = _probe_scaled_mm(torch.float8_e4m3fn)
    caps.int8_matmul = _probe_scaled_mm(torch.int8)     # expected False: _scaled_mm is fp8-only
    caps.int8_matmul_train = _probe_int_mm()

    try:    # cuDNN SDPA backend: present from PyTorch 2.5-ish, Ampere and newer
        from torch.backends.cuda import can_use_cudnn_attention  # noqa: F401
        caps.cudnn_attention = True
    except Exception:
        caps.cudnn_attention = hasattr(__import__("torch").backends.cuda, "cudnn_sdp_enabled")

    try:
        import flash_attn  # noqa: F401
        caps.flash_attn = True
    except Exception:
        pass

    try:
        import bitsandbytes  # noqa: F401
        caps.bitsandbytes = True
    except Exception:
        caps.notes.append("bitsandbytes missing — NF4 unavailable")

    return caps


# TRAINING-ONLY footprints at 0.25 MP, batch 1: the measured peaks (20.1 / 13.8 GB) were
# whole-GPU readings on a desktop already holding ~2.4 GB, so they overstate what training
# needs. Headroom then covers the user's own desktop plus allocator slack.
#
# The budget is FREE VRAM, not the number on the box: a "16 GB" card reports ~15.9 GiB total
# and may have several GB already held by a browser or a running ComfyUI. Deciding from total
# is how 16 GB cards ended up 4.4x slower — the config looked like it fit, and didn't.
_FP8_PEAK_GB = 17.7
_NF4_PEAK_GB = 11.4
# INT8 keeps the full 12.9B at one byte per weight, so it is ~5 GB above NF4 — measured 18.6 GB
# whole-GPU, ~16.2 GB training-only. It buys ~11% speed AND ~7x lower forward error than NF4
# (1.3e-02 vs 9.2e-02: 8-bit beats 4-bit), so it leads wherever it fits.
_INT8_PEAK_GB = 16.2
# Smaller than it looks: the budget is FREE VRAM, which already excludes whatever else is
# resident, so this only has to cover allocator slack and fragmentation.
_HEADROOM_GB = 1.5

# Run-shape terms, measured on a 5090 (36-image grid, 28 Jul 2026; whole-GPU peaks minus the
# ~1 GB desktop baseline; gradient checkpointing on, as the trainers force):
#   batch      +2.4 GB per extra image — flat across 0.25–1.05 MP, and by far the largest
#              term (the old single-constant budget's blind spot: batch 2 sailed through the
#              check and OOM'd).
#   resolution +0.15 GB from 0.25 → 1.05 MP at batch 1 (checkpointing absorbs it); budgeted
#              at 0.25 GB/MP for slack.
#   rank       +0.35 GB from r8 → r32 (~15 MB/rank); bases are measured AT rank 32.
_BATCH_GB_PER_IMAGE = 2.4
_RES_GB_PER_MP = 0.25
_RANK_GB_PER_RANK = 0.015


def estimate_krea2_peak(base_gb: float, mp: float = 0.25, batch: int = 1,
                        rank: int = 32) -> float:
    """Peak VRAM estimate for a Krea 2 run of this shape (base measured at 0.25 MP, b1, r32)."""
    return (base_gb
            + _BATCH_GB_PER_IMAGE * max(0, int(batch) - 1)
            + _RES_GB_PER_MP * max(0.0, float(mp) - 0.25)
            + _RANK_GB_PER_RANK * max(0, int(rank) - 32))


@dataclass
class MemoryStrategy:
    quant_4bit: bool
    blocks_to_swap: int
    reason: str
    quant_int8: str = ""     # "" | "bf16" — W8A8 base with exact bf16 gradients


def recommend_krea2_strategy(vram_gb: Optional[float] = None,
                             caps: Optional[Capabilities] = None,
                             mp: float = 0.25, batch: int = 1,
                             rank: int = 32) -> MemoryStrategy:
    """Pick quantisation + swap for Krea 2 training on this machine.

    Preference: INT8 no-swap > NF4 no-swap > fp8 no-swap > swapping.

    INT8 leads where it fits — faster than NF4 AND ~7x more accurate (8-bit vs 4-bit), with
    exact gradients, at the cost of ~5 GB. NF4 comes next because it measured faster than fp8 as
    well as smaller (fused bitsandbytes dequant, where the fp8 path materialises a bf16 copy of
    every weight per forward). Swapping is always last: 4.4x slower and 4x the CPU load.
    """
    caps = caps or detect()
    # Decide on FREE memory, not the number on the box — read it FRESH, not from the
    # lru_cached detect() snapshot: a GUI session that started while a browser held 6 GB
    # would otherwise plan every later run from that stale reading.
    vram = vram_gb
    if vram is None:
        try:
            import torch
            free_b, _ = torch.cuda.mem_get_info(0)
            vram = free_b / (1024 ** 3)
        except Exception:
            vram = caps.vram_free_gb or caps.vram_gb

    if not caps.has_cuda:
        return MemoryStrategy(False, 0, "no CUDA device — settings left alone")

    # INT8 first where it fits: faster than NF4 *and* far more accurate, with exact gradients.
    # Needs int8 tensor cores, which torch._int_mm requires — present from Turing, so this is
    # not Blackwell-only (unlike fp8 _scaled_mm, which needs sm_89+).
    _int8_need = estimate_krea2_peak(_INT8_PEAK_GB, mp, batch, rank)
    _nf4_need = estimate_krea2_peak(_NF4_PEAK_GB, mp, batch, rank)
    _fp8_need = estimate_krea2_peak(_FP8_PEAK_GB, mp, batch, rank)
    if caps.int8_matmul_train and vram >= _int8_need + _HEADROOM_GB:
        return MemoryStrategy(
            False, 0,
            f"INT8 W8A8, no block swap (~{_int8_need:.0f} GB needed at this run shape, {vram:.1f} GB free) — "
            "fastest measured, and ~7x more accurate than NF4 (8-bit vs 4-bit)",
            quant_int8="bf16")

    if caps.bitsandbytes and vram >= _nf4_need + _HEADROOM_GB:
        return MemoryStrategy(
            True, 0,
            f"NF4 4-bit, no block swap (~{_nf4_need:.0f} GB needed at this run shape, {vram:.1f} GB free) — "
            "fastest measured and leaves the most headroom")

    if vram >= _fp8_need + _HEADROOM_GB:
        return MemoryStrategy(
            False, 0, f"fp8, no block swap (~{_fp8_need:.0f} GB needed at this run shape, {vram:.1f} GB free)")

    if not caps.bitsandbytes:
        swap = 12 if vram >= 22 else (20 if vram >= 15 else 26)
        return MemoryStrategy(
            False, swap,
            f"fp8 with {swap} blocks swapped — bitsandbytes is missing, so NF4 (which would "
            "avoid swapping entirely and run ~4x faster) is unavailable. Install it.")

    # Below NF4's own footprint. NF4 CANNOT swap (the trainer force-zeroes blocks_to_swap
    # under 4-bit — weights live in _nf4_packed, not .weight), so the old "NF4 + swap"
    # recommendation here was a configuration that cannot exist: on a 12 GB card it was
    # the only reachable tier, leaving the auto path with no working configuration at all.
    # fp8 + heavy swap is the one combination that actually runs at this size.
    swap = 20 if vram >= 11 else 26
    return MemoryStrategy(
        False, swap,
        f"fp8 with {swap} blocks swapped — {vram:.1f} GB free is below what Krea 2 needs "
        "resident even at 4-bit, and NF4 can't block-swap, so fp8+swap is the only "
        "combination that fits (slow: ~4x the step time)")


# torch.compile decision. Warm-up is the whole story: compiling costs ~90 s up front (one plan per
# distinct sequence shape) and then saves per step, so it is a straight loss on a short run and a
# clear win on a long one. Measured per step on an RTX 5090, Krea 2, rank 16:
#
#     INT8   0.5917 -> 0.292   saves 0.300 s/step   break-even ~300 steps
#     NF4    0.7092 -> 0.556   saves 0.153 s/step   break-even ~590 steps
#
# Doubled for margin, as with the attention backend: at break-even there is nothing to win, and
# being wrong should cost a few percent rather than a run.
_COMPILE_WARMUP_S = 90.0
_COMPILE_SAVING_S = {"int8": 0.300, "nf4": 0.153}
_COMPILE_MARGIN = 2.0
# INT8 + compile peaked at 21.7 GB against 17.8 GB for INT8 alone. NF4 + compile is VRAM-neutral
# (12.9 GB vs 13.6 GB) and completes under a hard 15.5 GB cap, so it fits a 16 GB card.
_INT8_COMPILE_PEAK_GB = 20.0


def should_compile(total_steps: int, quant_4bit: bool, quant_int8: str,
                   blocks_to_swap: int, vram_gb: Optional[float] = None,
                   caps: Optional[Capabilities] = None) -> tuple:
    """Decide whether torch.compile pays for itself on this run. Returns (bool, reason)."""
    caps = caps or detect()
    vram = vram_gb if vram_gb is not None else (caps.vram_free_gb or caps.vram_gb)

    if blocks_to_swap:
        return False, "block swap is active — swapping moves weights between devices every step, " \
                      "which compiled graphs cannot tolerate"
    try:
        import triton  # noqa: F401
    except Exception:
        return False, "triton is not installed (pip install triton-windows on Windows)"

    kind = "nf4" if quant_4bit else ("int8" if quant_int8 else None)
    if kind is None:
        return False, "only measured for the quantised paths (NF4 / INT8); not enabled for fp8 or bf16"
    if kind == "int8" and vram < _INT8_COMPILE_PEAK_GB + _HEADROOM_GB:
        return False, (f"INT8 + compile peaks near {_INT8_COMPILE_PEAK_GB:.0f} GB and only "
                       f"{vram:.1f} GB is free — INT8 alone still fits, compile does not")

    needed = int(_COMPILE_WARMUP_S / _COMPILE_SAVING_S[kind] * _COMPILE_MARGIN)
    if total_steps < needed:
        return False, (f"{total_steps} steps is too short — compiling costs ~{_COMPILE_WARMUP_S:.0f} s "
                       f"up front and needs ~{needed} steps on the {kind.upper()} path to pay back")
    return True, (f"{total_steps} steps on the {kind.upper()} path — compile pays back within "
                  f"~{needed} steps and this run is longer")
