"""RefMod Toolkit maths: what the community node pack's Load H3 RefMods / Axis / Apply /
Step Curve nodes do to a reference latent between the file and the DiT, as Fizgig code.

Every formula here is the pack's (verified against its core.py / nodes.py, 15 Sep 2026, and
pinned by tests/test_refmod_apply_parity.py):

* strength = a lerp of the latent toward a heavily blurred copy of ITSELF
  (`s·z + (1−s)·blur8(z)`, refmod.blur_latent) — on-manifold weakening, never toward zero;
* retention = a master multiplier folded into every row's strength before the lerp;
* copies = the same latent repeated N times in the bundle (N× the tokens);
* the A/B axis = one signed value per row: negative picks A, positive picks B, |value| is the
  strength;
* the FRAME curve runs over the mod's own latent frames (`latent_t`), so an image mod (T = 1)
  has nothing for it to shape; the direction names say where the concept shows in the OUTPUT,
  which is the mirror of the envelope over the ref's timeline;
* the STEP curve re-mixes every reference latent once per denoising step from the same
  pristine/blurred pair, x = 1 − sigma (0 = first step, 1 = last).

One deliberate difference: the pack's `constant + linear @ 1.0` frame curve returns "no
curve" AND skips the flat strength on the way out, so a video mod under that (default-looking)
curve ignores retention and strength entirely. Here the flat strength always applies.
"""
from __future__ import annotations

import json
import math
import os
import random
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from fizgig.minimax.refmod import blur_latent, load_refmod, save_refmod, token_count
from fizgig.minimax.refmod_meta import (CURVE_DIRECTIONS, CURVE_SHAPES, DEFAULT_FRAME_CURVE,  # noqa: F401
                                        DEFAULT_STEP_CURVE, MAX_COPIES, MAX_ROWS, NODE_META_KEY, NODE_TOKEN_CAP,
                                        NONE_MOD, RETENTION_PRESETS, CurveSpec, _clamp01, _meta_from_shape,
                                        attach_bundle_audio, comfy_readout, entry_for, curve_is_flat, curve_strengths,
                                        curve_value_at, ease, prompt_hint, read_graph_preset_png,
                                        read_refmod_meta, read_refmod_metas, reference_map, scan_refmods,
                                        describe_meta)


# ─── one reference block ───────────────────────────────────────────────────────────────────

def ref_block(latent: torch.Tensor, strength: float,
              curve: Optional[CurveSpec] = None) -> Optional[torch.Tensor]:
    """The pack's H3RefMod.ref_block: the latent the DiT sees for one bundle entry at this
    strength (retention already folded in), with the frame curve across its own frames.
    None when the strength is <= 0 (the entry injects nothing)."""
    s = float(strength)
    if s <= 0.0:
        return None
    z = latent
    t = int(z.shape[2])
    per_frame = curve_strengths(curve, t) if curve is not None else None
    # Arithmetic exactly as the pack does it (bit-identical, see the parity test): both paths
    # mix in the latent's own dtype — the per-frame tensor is built in that dtype, the flat
    # path uses Python floats.
    if per_frame is not None:
        st = torch.tensor([_clamp01(s * v) for v in per_frame], dtype=z.dtype,
                          device=z.device).view(1, 1, t, 1, 1)
        return st * z + (1.0 - st) * blur_latent(z)
    if s < 1.0:
        return s * z + (1.0 - s) * blur_latent(z)
    return z


# ─── the bundle ────────────────────────────────────────────────────────────────────────────

class ModRow:
    """One Mods-card row. `b_latent`/`b_meta` set = an A/B axis row: `value` is signed
    (negative → A, positive → B, |value| = strength); otherwise `value` is the 0–1 strength."""

    def __init__(self, latent, meta, value: float = 1.0, copies: int = 1, enabled: bool = True,
                 b_latent=None, b_meta=None, name: str = "", b_name: str = ""):
        self.latent, self.meta = latent, meta
        self.b_latent, self.b_meta = b_latent, b_meta
        self.value = float(value)
        self.copies = max(1, min(MAX_COPIES, int(copies)))
        self.enabled = bool(enabled)
        self.name = name or str((meta or {}).get("name", ""))
        self.b_name = b_name or str((b_meta or {}).get("name", ""))

    @property
    def is_axis(self) -> bool:
        return self.b_latent is not None

    def resolve(self):
        """-> (latent, meta, strength, name) for the side this row picks, or None."""
        if not self.enabled:
            return None
        if self.is_axis:
            if abs(self.value) < 1e-6:
                return None
            if self.value > 0:
                return self.b_latent, self.b_meta, min(1.0, abs(self.value)), self.b_name
            return self.latent, self.meta, min(1.0, abs(self.value)), self.name
        if self.latent is None or self.value <= 0.0:
            return None
        return self.latent, self.meta, min(1.0, max(0.0, self.value)), self.name

    def tokens(self) -> int:
        r = self.resolve()
        return token_count(r[0]) * self.copies if r else 0


def loads_from_rows(rows: Sequence[ModRow]) -> List[Tuple[torch.Tensor, dict, float, str]]:
    """The Loader/Axis output: (latent, meta, strength, name) repeated `copies` times."""
    out = []
    for row in rows:
        r = row.resolve()
        if r is None:
            continue
        out.extend([r] * row.copies)
    return out


def scramble(items: list, seed: int) -> list:
    """The Apply node's scramble_seed: shuffle the bundle and keep a random 50–100% prefix.
    Off for seed < 0 or a single entry."""
    items = list(items)
    if int(seed) >= 0 and len(items) > 1:
        rng = random.Random(int(seed))
        rng.shuffle(items)
        keep = rng.randint(max(1, len(items) // 2), len(items))
        items = items[:keep]
    return items


def build_bundle_entries(rows: Sequence[ModRow], *, retention: float = 1.0,
                         curve: Optional[CurveSpec] = None, scramble_seed: int = -1):
    """build_bundle plus, per latent, an entry dict for the numbered-reference presentation:
    {"name", "kind" (image|video), "latent_t", "audio_members": [names]} in the same order."""
    factor = _clamp01(float(retention))
    loads = scramble(loads_from_rows(rows), scramble_seed)
    latents, describe, entries = [], [], []
    for latent, meta, strength, name in loads:
        eff = _clamp01(strength * factor)
        z = ref_block(latent, eff, curve)
        if z is None:
            continue
        latents.append(z)
        describe.append((name, eff))
        entries.append(entry_for(meta, name, int(latent.shape[2])))
    return latents, describe, attach_bundle_audio(entries)


def build_bundle(rows: Sequence[ModRow], *, retention: float = 1.0,
                 curve: Optional[CurveSpec] = None, scramble_seed: int = -1):
    """Everything the Apply node does before the DiT: -> (latents, describe) where latents
    is the list of reference latents to ride as condition rows (in order) and describe is a
    parallel list of (name, effective_strength) for the readout."""
    latents, describe, _entries = build_bundle_entries(rows, retention=retention, curve=curve,
                                                       scramble_seed=scramble_seed)
    return latents, describe


def bundle_tokens(rows: Sequence[ModRow]) -> int:
    return sum(r.tokens() for r in rows)


# ─── the step curve ────────────────────────────────────────────────────────────────────────

def step_schedule(spec: Optional[CurveSpec], latents: Sequence[torch.Tensor]
                  ) -> Optional[Callable[[int, int, float], List[torch.Tensor]]]:
    """The Step Curve node as a sampler hook: (step, n_steps, sigma) -> the reference latents
    for that step, `s·pristine + (1−s)·blur8(pristine)` with s = curve(1 − sigma). None when
    the curve is flat (the sampler then keeps its once-built list)."""
    if curve_is_flat(spec) or not latents:
        return None
    pristine = [z for z in latents]
    blurred = [blur_latent(z) for z in pristine]

    def at(step: int, n: int, sigma: float) -> List[torch.Tensor]:
        x = 1.0 - max(0.0, min(1.0, float(sigma)))
        s = curve_value_at(spec, x)
        if s >= 1.0:
            return pristine
        s = _clamp01(s)
        return [(s * p.float() + (1.0 - s) * b.float()).to(p.dtype) for p, b in zip(pristine, blurred)]
    return at


def bake(latent: torch.Tensor, meta: dict, *, strength: float, retention: float,
         curve: Optional[CurveSpec]):
    """Fold strength × retention and the frame curve into the latent so the result loads at
    1.0 anywhere. -> (latent, tags, note). Copies / scramble / the step curve are runtime-only
    and cannot bake (the note says so)."""
    eff = _clamp01(float(strength) * _clamp01(float(retention)))
    z = ref_block(latent, eff, curve)
    if z is None:
        raise ValueError("strength × retention is 0 — nothing to bake")
    tags = list(meta.get("tags", []) or [])
    desc = f"studio: strength {eff:.2f}"
    if curve is not None and curve_strengths(curve, int(latent.shape[2])) is not None:
        desc += f", curve {curve[0]}/{curve[1]}/{float(curve[2]):.2f}"
    tags.append(desc)
    note = ("Baked: strength × retention and the frame curve. Not baked (runtime only): "
            "copies, scramble, the step curve.")
    return z, tags, note


def save_baked(path_no_ext: str, latent: torch.Tensor, meta: dict, tags: list) -> str:
    return save_refmod(path_no_ext, latent, name=os.path.basename(path_no_ext),
                       mode=str(meta.get("mode", "encode")), pool=str(meta.get("pool", "")),
                       optimize_steps=int(meta.get("optimize_steps", 0) or 0),
                       source_shape=str(meta.get("source_shape", "")), tags=tags,
                       description=str(meta.get("description", "")),
                       concept_type=str(meta.get("concept_type", "identity")))


__all__ = ["CURVE_DIRECTIONS", "CURVE_SHAPES", "RETENTION_PRESETS", "DEFAULT_FRAME_CURVE",
           "DEFAULT_STEP_CURVE", "MAX_ROWS", "MAX_COPIES", "NONE_MOD", "NODE_TOKEN_CAP",
           "ease", "curve_value_at", "curve_strengths", "curve_is_flat", "ref_block", "ModRow",
           "loads_from_rows", "scramble", "build_bundle", "bundle_tokens", "step_schedule",
           "read_refmod_meta", "scan_refmods", "describe_meta", "prompt_hint", "comfy_readout",
           "bake", "save_baked", "read_graph_preset_png", "load_refmod", "token_count"]
