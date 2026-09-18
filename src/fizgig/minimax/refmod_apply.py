"""RefMod Studio maths: what the community node pack's Load H3 RefMods / Axis / Apply /
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

from fizgig.minimax.refmod import (NODE_META_KEY, NODE_TOKEN_CAP, blur_latent, load_refmod,
                                   save_refmod, token_count)

CURVE_DIRECTIONS = ("constant", "concept_at_start", "concept_at_middle",
                    "concept_at_end", "concept_at_ends")
CURVE_SHAPES = ("linear", "ease", "sigmoid", "tanh", "quadratic", "cubic",
                "exponential", "stair", "elastic", "bump", "dip")
RETENTION_PRESETS = (("Full", 1.0), ("Partial", 0.7), ("Attribute", 0.4), ("Weak", 0.15))
# The names mean what they say (the pack, v0.2.x): concept_at_start = full strength at the
# start of the timeline fading to nothing, concept_at_end = nothing rising to full. The default
# envelope is the one the pack ships — strong while the structure forms, released for the
# texture steps — which under these names is concept_at_start.
DEFAULT_FRAME_CURVE = ("concept_at_start", "ease", 1.0)
DEFAULT_STEP_CURVE = ("concept_at_start", "ease", 1.0)
MAX_ROWS = 8
MAX_COPIES = 10
NONE_MOD = "(none)"

CurveSpec = Tuple[str, str, float]


# ─── curves ─────────────────────────────────────────────────────────────────────────────────

def ease(shape: str, x: float) -> float:
    """The pack's easing table (elastic overshoots on purpose)."""
    if shape == "linear":
        return x
    if shape == "ease":
        return x * x * (3.0 - 2.0 * x)
    if shape == "sigmoid":
        return 1.0 / (1.0 + math.exp(-12.0 * (x - 0.5)))
    if shape == "tanh":
        return 0.5 * (math.tanh(8.0 * (x - 0.5)) + 1.0)
    if shape == "quadratic":
        return x * x
    if shape == "cubic":
        return x * x * x
    if shape == "exponential":
        return 2.0 ** x - 1.0
    if shape == "stair":
        return min(1.0, math.floor(x * 4) / 3.0)
    if shape == "elastic":
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        return 2.0 ** (-10.0 * x) * math.sin((x * 10.0 - 0.75) * (2.0 * math.pi / 3.0)) + 1.0
    if shape == "bump":
        return 1.0 - abs(2.0 * x - 1.0)
    if shape == "dip":
        return abs(2.0 * x - 1.0)
    return x


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def curve_value_at(spec: Optional[CurveSpec], x: float) -> float:
    """Strength multiplier of a (direction, shape, value) curve at progress x in [0, 1].
    1.0 (no weakening) for anything unrecognised — the pack's rule."""
    if not spec:
        return 1.0
    direction, shape, value = spec
    value = float(value)
    if direction == "constant":
        if shape == "linear":
            return value
        return _clamp01(value * ease(shape, x))
    y = ease(shape, x)
    if direction == "concept_at_end":
        return _clamp01(value * y)              # 0 -> value: the concept shows in the second half
    if direction == "concept_at_start":
        return _clamp01(value * (1.0 - y))      # value -> 0: the concept shows in the first half
    if direction == "concept_at_middle":
        return _clamp01(value * (1.0 - abs(2.0 * y - 1.0)))
    if direction == "concept_at_ends":
        return _clamp01(value * abs(2.0 * y - 1.0))
    return 1.0


def curve_strengths(spec: Optional[CurveSpec], t: int) -> Optional[List[float]]:
    """Per-frame multipliers for a mod with t latent frames, or None when the curve has
    nothing to shape (t <= 1, no spec, an unknown direction, or constant+linear — the
    caller then applies its flat strength)."""
    if t <= 1 or not spec:
        return None
    direction, shape, value = spec
    if direction not in CURVE_DIRECTIONS:
        return None
    if direction == "constant" and shape == "linear":
        # flat: 1.0 = no curve; below that the value multiplies every frame's strength
        value = float(value)
        return None if value >= 1.0 else [value] * t
    return [curve_value_at(spec, i / (t - 1)) for i in range(t)]


def curve_is_flat(spec: Optional[CurveSpec]) -> bool:
    """True when the curve changes nothing (what the Step Curve node's 'not connected' means)."""
    return (not spec) or (spec[0] == "constant" and spec[1] == "linear" and float(spec[2]) >= 1.0)


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
        m = meta or {}
        t = int(m.get("latent_t") or latent.shape[2])
        entries.append({"name": name, "kind": "video" if t > 1 else "image", "latent_t": t,
                        "path": str(m.get("path", "")), "bundle_audio": list(m.get("bundle_audio") or []),
                        "audio_members": []})
    # A bundle's audio members ride ONCE per bundle, after the last of that bundle's visual
    # entries, one label per visual copy present — the pack's loader expands a bundle slot as
    # all the visual copies, then all the audio copies (V V A A), and lists the audio once.
    seen = set()
    for i, e in enumerate(entries):
        key = e["path"]
        if not e["bundle_audio"] or not key or key in seen:
            continue
        seen.add(key)
        same = [j for j, o in enumerate(entries) if o["path"] == key]
        n = len(same)
        entries[same[-1]]["audio_members"] = [a for a in e["bundle_audio"] for _ in range(n)]
    return latents, describe, entries


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


# ─── files, folders, readouts ──────────────────────────────────────────────────────────────

def read_refmod_metas(path: str) -> List[dict]:
    """Every reference in a file (header only, no tensor read): one entry for a standalone
    mod, one per member for a bundle (each with `bundle_index` and `tensor_key`). Empty when
    the file is not a RefMod."""
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            raw = (f.metadata() or {}).get(NODE_META_KEY)
            if not raw:
                return []
            meta = json.loads(raw)
            if not isinstance(meta, dict):
                return []
            if str(meta.get("kind", "")) == "bundle":
                out = []
                keys = set(f.keys())
                for i, m in enumerate(meta.get("members") or []):
                    key = f"ref_{i}"
                    if not isinstance(m, dict) or key not in keys:
                        continue
                    entry = _meta_from_shape(dict(m), f.get_slice(key).get_shape(), path)
                    if entry is None:
                        continue
                    entry["bundle_index"] = i
                    entry["bundle_name"] = str(meta.get("name", ""))
                    entry["tensor_key"] = key
                    out.append(entry)
                _audio_names = [str(e.get("name", "")) for e in out if e.get("kind") == "audio"]
                for e in out:
                    if e.get("kind") != "audio":
                        e["bundle_audio"] = list(_audio_names)
                return out
            if "latent" not in f.keys():
                return []
            entry = _meta_from_shape(dict(meta), f.get_slice("latent").get_shape(), path)
            if entry is None:
                return []
            entry["tensor_key"] = "latent"
            return [entry]
    except Exception:
        return []


def read_refmod_meta(path: str) -> Optional[dict]:
    """The header only (no tensor read) — None when the file is not a RefMod. A bundle gives
    its first VISUAL member (the audio members are read_refmod_metas' business)."""
    entries = read_refmod_metas(path)
    for e in entries:
        if e.get("kind") != "audio":
            return e
    return entries[0] if entries else None


def _meta_from_shape(meta: dict, shape, path: str) -> Optional[dict]:
    """Fill the fields the Studio reads (kind, dims, tokens, name, path) from a tensor shape."""
    if len(shape) not in (4, 5):
        return None
    if len(shape) == 4:
        # an audio mod: [1, 32, 2, T], 2 tokens a latent frame
        if tuple(shape[:3]) != (1, 32, 2):
            return None
        meta["kind"] = "audio"
        meta.setdefault("latent_t", int(shape[3]))
        meta["latent_h"] = meta["latent_w"] = 0
        meta.setdefault("name", os.path.splitext(os.path.basename(path))[0])
        meta["tokens"] = 2 * int(meta["latent_t"])
        meta["path"] = path
        return meta
    meta.setdefault("latent_t", int(shape[2]))
    meta.setdefault("latent_h", int(shape[3]))
    meta.setdefault("latent_w", int(shape[4]))
    meta.setdefault("kind", "video" if int(shape[2]) > 1 else "image")
    meta.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    meta["tokens"] = int(meta["latent_t"]) * (int(meta["latent_h"]) // 2) * (int(meta["latent_w"]) // 2)
    meta["path"] = path
    return meta


def scan_refmods(folder: str, cache: Optional[dict] = None, include_audio: bool = False) -> List[dict]:
    """Every RefMod in a folder (non-recursive), by name. `cache` keyed on
    (path, size, mtime) skips re-reading unchanged headers. Audio mods (kind "audio", a
    [1, 32, 2, T] latent) are left out unless asked for: the Studio's renderer takes visual
    references only; they play in ComfyUI."""
    out = []
    if not folder or not os.path.isdir(folder):
        return out
    for fn in sorted(os.listdir(folder), key=str.lower):
        if not fn.lower().endswith(".safetensors"):
            continue
        p = os.path.join(folder, fn)
        try:
            st = os.stat(p)
        except OSError:
            continue
        key = (p, st.st_size, st.st_mtime)
        meta = cache.get(key) if cache is not None else None
        if meta is None:
            meta = read_refmod_metas(p)          # a bundle's members, or the one mod
            if not meta:
                continue
            if cache is not None:
                cache[key] = meta
        for m in (meta if isinstance(meta, list) else [meta]):
            if not include_audio and str(m.get("kind", "")) == "audio":
                continue
            out.append(m)
    return out


def describe_meta(meta: dict) -> str:
    """The Mods-card info line: `video · 44×16×16 · 2 816 tokens · optimised 200 · "…"`."""
    if not meta:
        return ""
    parts = [str(meta.get("kind", "?")),
             f"{meta.get('latent_t', '?')}×{meta.get('latent_h', '?')}×{meta.get('latent_w', '?')}",
             f"{int(meta.get('tokens', 0)):,} tokens".replace(",", " ")]
    # which H3 model it was tuned on (Fizgig writes the tag; the pack's Inspect node shows it too)
    for _t in meta.get("tags") or []:
        if str(_t).startswith("tuned on "):
            parts.append("for " + str(_t)[len("tuned on "):])
            break
    steps = int(meta.get("optimize_steps", 0) or 0)
    parts.append(f"optimised {steps}" if steps > 0 else "plain encode")
    d = str(meta.get("description") or "").strip()
    if d:
        parts.append(f'"{d}"')
    return " · ".join(parts)


def prompt_hint(metas: Sequence[dict]) -> str:
    """The Loader's prompt_hint: 'concept_type: description; …' over mods that have one."""
    bits = []
    for m in metas:
        d = str((m or {}).get("description") or "").strip()
        if d:
            bits.append(f"{(m or {}).get('concept_type', 'generic')}: {d}")
    return "; ".join(bits)


def reference_map(entries) -> List[str]:
    """The pack's Text Encode / Inspect reference map, one line per presented reference in
    bundle order — "<Picture n> = name", "<Video n> = name", "<Audio n> = name" — each kind
    counted on its own, copies with their own labels, bundled audio members after their
    visual member. `entries` is build_bundle_entries' third result."""
    out, counters = [], {"image": 0, "video": 0, "audio": 0}
    labels = {"image": "Picture", "video": "Video", "audio": "Audio"}
    for e in entries or []:
        kind = str(e.get("kind", "image"))
        if kind not in counters:
            kind = "image"
        counters[kind] += 1
        out.append(f"<{labels[kind]} {counters[kind]}> = {e.get('name', '')}")
        for a_name in e.get("audio_members") or []:
            counters["audio"] += 1
            out.append(f"<Audio {counters['audio']}> = {a_name}")
    return out


def comfy_readout(rows: Sequence[ModRow], *, retention: float, frame_curve: CurveSpec,
                  scramble_seed: int, step_curve: Optional[CurveSpec], step_on: bool) -> str:
    """The pack's widget values that reproduce this setup, one node per paragraph."""
    lines = []
    loader = [r for r in rows if r.enabled and not r.is_axis and r.latent is not None and r.value > 0]
    axis = [r for r in rows if r.enabled and r.is_axis and abs(r.value) >= 1e-6]
    if loader:
        lines.append("Load H3 RefMods")
        for i, r in enumerate(loader, 1):
            lines.append(f"  mod_{i} {r.name}   strength_{i} {min(1.0, r.value):.2f}   copies_{i} {r.copies}")
    if axis:
        lines.append("Load H3 RefMods (Axis)")
        for i, r in enumerate(axis, 1):
            lines.append(f"  mod_a_{i} {r.name}   mod_b_{i} {r.b_name}   value_{i} {max(-1.0, min(1.0, r.value)):+.2f}")
            if r.copies > 1:
                lines.append(f"    (the Axis node has no copies — this row was tested at ×{r.copies})")
    if not loader and not axis:
        lines.append("(no active mods)")
    d, s, v = frame_curve
    lines.append("Apply RefMods")
    lines.append(f"  retention {float(retention):.2f}   curve_direction {d}   curve_shape {s}   "
                 f"curve_value {float(v):.2f}   scramble_seed {int(scramble_seed)}")
    if step_on and step_curve and not curve_is_flat(step_curve):
        d2, s2, v2 = step_curve
        lines.append("Step Curve")
        lines.append(f"  curve_direction {d2}   curve_shape {s2}   curve_value {float(v2):.2f}")
    else:
        lines.append("Step Curve: not connected")
    hint = prompt_hint([r.resolve()[1] for r in rows if r.resolve()])
    if hint:
        lines.append(f"prompt_hint: {hint}")
    return "\n".join(lines)


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


def read_graph_preset_png(path: str) -> Optional[CurveSpec]:
    """The pack's graph preset: a PNG whose `graph` text chunk holds {direction, shape, value}
    (legacy: a .json with the same keys)."""
    try:
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        else:
            from PIL import Image
            with Image.open(path) as im:
                raw = (getattr(im, "text", None) or {}).get("graph") or im.info.get("graph")
            if not raw:
                return None
            d = json.loads(raw)
        direction, shape = str(d.get("direction")), str(d.get("shape"))
        if direction not in CURVE_DIRECTIONS or shape not in CURVE_SHAPES:
            return None
        return direction, shape, float(d.get("value", 1.0))
    except Exception:
        return None


__all__ = ["CURVE_DIRECTIONS", "CURVE_SHAPES", "RETENTION_PRESETS", "DEFAULT_FRAME_CURVE",
           "DEFAULT_STEP_CURVE", "MAX_ROWS", "MAX_COPIES", "NONE_MOD", "NODE_TOKEN_CAP",
           "ease", "curve_value_at", "curve_strengths", "curve_is_flat", "ref_block", "ModRow",
           "loads_from_rows", "scramble", "build_bundle", "bundle_tokens", "step_schedule",
           "read_refmod_meta", "scan_refmods", "describe_meta", "prompt_hint", "comfy_readout",
           "bake", "save_baked", "read_graph_preset_png", "load_refmod", "token_count"]
