"""Optimizer selection, shared by the trainers.

Krea 2 hardcoded `bnb.optim.AdamW8bit` — a good default, but the only choice, which is the one
place the community comparison against OneTrainer (~40 optimizers) landed a fair hit.

Two things matter for a LoRA and pull against the "more optimizers is better" instinct:

* **Optimizer state is tiny here.** Krea 2 trains 264 small factors, so AdamW's two moments cost
  ~tens of MB against a 13-19 GB base. Choosing an 8-bit or factored optimizer to save memory is
  nearly pointless — the base dominates. What the choice actually buys is *update behaviour*.
* **Learning rates are not comparable across families.** Lion's update is a sign, so it wants
  roughly a tenth of AdamW's LR. Handing someone a dropdown without saying that is how you
  produce a fried LoRA and a bug report, so `create_optimizer` warns loudly on the record when
  the LR looks wrong for the family. (Self-tuning families — Prodigy, CAME, Adafactor — were
  removed for exactly this class of failure; see the note above _CATALOG.)

Free-form `module.path.ClassName` is accepted too, so a user who pip-installs something exotic
does not need a Fizgig release to use it.
"""

from __future__ import annotations

import ast
import importlib
import logging

import torch

logger = logging.getLogger(__name__)


# name -> (import to test, one-line description shown in the GUI/CLI help)
# Removed by decision (2026-07-28): prodigy, came, adafactor — the "manage their own LR"
# family. They fight Adaptive LR by design (prodigy wants lr=1.0 as a multiplier, adafactor
# relative_step stores lr=None, came is an Adafactor variant), two of the three needed
# external packages, and both prodigy and adafactor caused real bugs (silent AdamW fallback
# at lr=1.0; TypeError against the adaptive watcher). Exotic optimizers remain available
# via the full module.path.ClassName form.
_CATALOG = {
    "adamw8bit":          ("bitsandbytes", "AdamW, 8-bit state (default — the validated recipe)"),
    "adamw":              (None,           "AdamW, fp32 state, CUDA-fused where available"),
    "pagedadamw8bit":     ("bitsandbytes", "AdamW8bit that pages state to CPU under pressure"),
    "ademamix8bit":       ("bitsandbytes", "AdEMAMix — second slow EMA, aimed at long runs"),
    "pagedademamix8bit":  ("bitsandbytes", "AdEMAMix8bit with CPU paging"),
    "lion8bit":           ("bitsandbytes", "Lion — sign updates; use ~1/10 the AdamW LR"),
}

DEFAULT_OPTIMIZER = "adamw8bit"


def available_optimizers() -> list[str]:
    """Catalog entries whose backing package is actually importable on this machine."""
    out = []
    for name, (module, _desc) in _CATALOG.items():
        if module is None:
            out.append(name)
            continue
        try:
            importlib.import_module(module)
            out.append(name)
        except Exception:
            pass
    return out


def describe(name: str) -> str:
    return _CATALOG.get(name.lower(), (None, "custom optimizer"))[1]


def parse_optimizer_args(raw: str) -> dict:
    """`"weight_decay=0.01 betas=0.9,0.99"` -> `{"weight_decay": 0.01, "betas": (0.9, 0.99)}`.

    Values go through `ast.literal_eval`, so tuples, bools and None all survive; anything that
    isn't a literal is kept as a plain string rather than being executed.
    """
    kwargs = {}
    for tok in (raw or "").split():
        if "=" not in tok:
            raise ValueError(f"optimizer arg {tok!r} is not key=value")
        key, value = tok.split("=", 1)
        try:
            kwargs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            kwargs[key] = value
    return kwargs


def _bnb(cls_name: str):
    import bitsandbytes as bnb
    return getattr(bnb.optim, cls_name)


def _warn_lr(name: str, lr: float) -> None:
    """A wrong-family LR is silent at step 1 and obvious only hours later. Say it now."""
    if name == "lion8bit" and lr > 5e-5:
        logger.warning("[optimizer] Lion applies the SIGN of the update, so it needs roughly a "
                       "TENTH of an AdamW LR. %.2e will likely overbake — try %.2e.", lr, lr / 10)
    elif name != "lion8bit" and lr > 1e-2:
        logger.warning("[optimizer] LR %.2e is very high for %s.", lr, name)


def create_optimizer(name: str, params, lr: float, args_str: str = "",
                     eps_floor_8bit: bool = False) -> tuple:
    """Build an optimizer. Returns `(optimizer, label)`; the label goes into LoRA metadata.

    Falls back to plain AdamW if the requested one cannot be constructed — a training run should
    not die at minute one over a dropdown, but the substitution is logged as a warning, never
    silently.

    `eps_floor_8bit` raises the 8-bit Adam family's eps to 1e-6. OFF by default: it is a
    MiniMax H3 workaround and every other family keeps the library default. See the note below.
    """
    name = (name or DEFAULT_OPTIMIZER).strip()
    kwargs = parse_optimizer_args(args_str)
    key = name.lower()
    _warn_lr(key, lr)

    # eps 1e-6, not the library defaults' 1e-8 (matches ai-toolkit, which passes eps=1e-6 to
    # every Adam-family optimizer). This is a REAL stability bound, not a nicety: the 8-bit
    # optimizers store the second moment blockwise-quantized, and for heavily structured
    # gradients the small v entries quantize to ZERO — the update then degrades to lr*m/eps.
    # Measured on a MiniMax H3 epoch (46 steps @ 1e-4, eps=1e-8): lora_up drift reached 0.81
    # against an Adam bound of ~0.005 — the optimizer was applying ~100x the configured LR to
    # the most structured tensors (adaln worst, fc1 next), which presented as melted anatomy
    # at epoch 1. eps=1e-6 caps that amplification two orders of magnitude lower. Explicit
    # "eps=..." in Optimizer Args still wins.
    # NOTE the two conditions. 8-BIT Adam family only, NOT full-precision adam/adamw: full
    # precision has no quantized state, so v is whatever it really is, and a 1e-6 floor there
    # would DAMP the tensors with genuinely small second moments — the ones converging on fine
    # detail — while looking like a stability measure.
    #
    # And OPT-IN per caller, never global. This began as a MiniMax fix (bafb4e6) applied to the
    # whole Adam family, which silently moved Krea 2's DEFAULT optimizer off the library eps and
    # shipped that way in v3.3.0. Krea 2 never had the failure this works around and never asked
    # for the change. A workaround for one model family does not get to alter another's defaults;
    # the caller that needs it asks for it. Explicit "eps=..." in Optimizer Args still wins.
    if eps_floor_8bit and "8bit" in key and "lion" not in key:
        kwargs.setdefault("eps", 1e-6)

    try:
        if key == "adamw8bit":
            opt = _bnb("AdamW8bit")(params, lr=lr, **kwargs)
        elif key == "pagedadamw8bit":
            opt = _bnb("PagedAdamW8bit")(params, lr=lr, **kwargs)
        elif key == "ademamix8bit":
            opt = _bnb("AdEMAMix8bit")(params, lr=lr, **kwargs)
        elif key == "pagedademamix8bit":
            opt = _bnb("PagedAdEMAMix8bit")(params, lr=lr, **kwargs)
        elif key == "lion8bit":
            opt = _bnb("Lion8bit")(params, lr=lr, **kwargs)
        elif key == "adamw":
            # Fused AdamW is one CUDA kernel over all 264 factors instead of a Python loop —
            # the "CUDA optimizer" the comparison thread was pointing at. Requires every param
            # on CUDA and floating point, which LoRA factors are.
            kwargs.setdefault("fused", torch.cuda.is_available())
            opt = torch.optim.AdamW(params, lr=lr, **kwargs)
        elif "." in name:
            module_path, cls_name = name.rsplit(".", 1)
            opt = getattr(importlib.import_module(module_path), cls_name)(params, lr=lr, **kwargs)
        else:
            raise ValueError(f"unknown optimizer {name!r} — use one of {available_optimizers()} "
                             "or a full module.path.ClassName")
    except Exception as e:
        logger.warning("[optimizer] could not create %s (%s) — falling back to AdamW", name, e)
        return torch.optim.AdamW(params, lr=lr), "adamw (fallback)"

    label = name + (f"({args_str.strip()})" if args_str.strip() else "")
    logger.info("optimizer: %s — %s", label, describe(key))
    return opt, label
