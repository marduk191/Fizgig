"""Extract a LoRA from the difference between two full checkpoints.

The companion to rotating-block fine-tuning: that produces a ~26 GB model, this turns the
difference from its starting base into an ordinary, shareable LoRA. Measured on a 3-subject
Krea 2 fine-tune, rank 64 is perceptually indistinguishable from the full checkpoint and
rank 8 still holds identity separation — see docs/FINETUNE_ROTATION.md.

Multi-rank is nearly free and is the point of the API: SVD returns singular values in
descending order, so a rank-r factorisation is a truncation of the same decomposition.
Asking for [16, 32, 64, 128] costs one SVD per layer, not four, and each result is
bit-identical to extracting that rank alone.

Architecture-agnostic: keys are flattened to the kohya convention
(``blocks.0.attn.wq.weight`` -> ``lora_unet_blocks_0_attn_wq``), which is what both Klein
and Krea 2 LoRA loaders expect.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Iterable, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Layers a LoRA meaningfully applies to: 2-D Linear weights, excluding norms/embeddings
# (1-D or not linear maps, and not what adapters target).
_SKIP_SUBSTRINGS = ("norm", "embed", "_scale", "modulation")


def is_lora_target(key: str, shape) -> bool:
    if not key.endswith(".weight") or len(shape) != 2:
        return False
    low = key.lower()
    return not any(s in low for s in _SKIP_SUBSTRINGS)


def lora_key(model_key: str) -> str:
    """blocks.0.attn.wq.weight -> lora_unet_blocks_0_attn_wq"""
    return "lora_unet_" + model_key[: -len(".weight")].replace(".", "_")


def factor_multi(delta: torch.Tensor, ranks: Iterable[int]) -> dict:
    """One SVD, sliced to every requested rank. Returns {rank: (up, down)} in fp16/CPU.

    Singular values are split evenly between the factors (sqrt on each side) so neither
    matrix carries the whole magnitude — the usual convention, and it keeps both halves in
    a sane numeric range for fp16 storage.
    """
    m, n = delta.shape
    cap = min(m, n)
    wanted = sorted({int(r) for r in ranks if int(r) >= 1})
    if not wanted or cap < 1:
        return {}
    top = min(max(wanted), cap)
    # NOTE: svd_lowrank returns V, not Vh — delta ~= U @ diag(S) @ V.T
    U, S, V = torch.svd_lowrank(delta, q=min(top + 8, cap), niter=4)
    out = {}
    for r in wanted:
        k = min(r, cap)
        s = torch.sqrt(S[:k])
        up = (U[:, :k] * s.unsqueeze(0)).to(torch.float16).cpu().contiguous()          # (m, k)
        down = (s.unsqueeze(1) * V[:, :k].T).to(torch.float16).cpu().contiguous()      # (k, n)
        out[r] = (up, down)
    return out


def extract_diff_loras(
    base_path: str,
    tuned_path: str,
    output_dir: str,
    ranks: Iterable[int],
    name: str = "extracted",
    device: Optional[str] = None,
    min_norm: float = 1e-4,
    progress: Optional[Callable[[int, int, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> list:
    """Diff two checkpoints and write one LoRA per rank. Returns the paths written.

    Streams tensor-by-tensor, so peak memory is a couple of layers regardless of how big
    the checkpoints are.
    """
    ranks = sorted({int(r) for r in ranks if int(r) >= 1})
    if not ranks:
        raise ValueError("no ranks requested")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    say = log or (lambda _m: None)
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    h_base = safe_open(base_path, framework="pt", device="cpu")
    h_tune = safe_open(tuned_path, framework="pt", device="cpu")
    base_keys, tune_keys = set(h_base.keys()), set(h_tune.keys())

    only_tuned = len(tune_keys - base_keys)
    if only_tuned:
        say(f"note: {only_tuned} key(s) exist only in the trained file and are ignored")

    keys = [k for k in sorted(base_keys & tune_keys)
            if is_lora_target(k, h_base.get_slice(k).get_shape())]
    if not keys:
        raise RuntimeError("no comparable 2-D weights found — are these the same architecture?")
    say(f"{len(keys)} candidate matrices, ranks {ranks}, device {dev}")

    sd = {r: {} for r in ranks}
    changed = skipped = 0
    total_norm = 0.0

    for i, k in enumerate(keys):
        if should_stop is not None and should_stop():
            say("cancelled")
            return []
        b = h_base.get_tensor(k).to(dev, torch.float32)
        d = h_tune.get_tensor(k).to(dev, torch.float32) - b
        del b
        nrm = d.norm().item()
        if nrm < min_norm:
            skipped += 1
            del d
            if progress:
                progress(i + 1, len(keys), k)
            continue
        total_norm += nrm
        lk = lora_key(k)
        for r, (up, down) in factor_multi(d, ranks).items():
            sd[r][f"{lk}.lora_up.weight"] = up
            sd[r][f"{lk}.lora_down.weight"] = down
            # alpha == rank -> scale 1.0, so up @ down reproduces the delta as-is.
            sd[r][f"{lk}.alpha"] = torch.tensor(float(min(r, min(d.shape))))
        changed += 1
        del d
        if progress:
            progress(i + 1, len(keys), k)

    if changed == 0:
        raise RuntimeError("the two checkpoints are identical (no weight moved) — nothing to extract")
    say(f"{changed} matrices changed, {skipped} unchanged, mean |delta| {total_norm / changed:.3f}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    written = []
    for r in ranks:
        path = os.path.join(output_dir, f"{name}_{stamp}_r{r}.safetensors")
        meta = {
            "ss_network_module": "networks.lora",
            "ss_network_dim": str(r),
            "ss_network_alpha": str(float(r)),
            "fizgig_extraction": "checkpoint_diff_svd",
            "fizgig_extraction_rank": str(r),
            "fizgig_extraction_rank_set": ",".join(str(x) for x in ranks),
            "fizgig_source_base": os.path.basename(base_path),
            "fizgig_source_tuned": os.path.basename(tuned_path),
            "fizgig_source_layers": str(changed),
        }
        save_file(sd[r], path, metadata=meta)
        written.append(path)
        say(f"saved r{r}: {os.path.basename(path)} "
            f"({os.path.getsize(path) / (1024 ** 2):.0f} MB)")

    say(f"done in {time.time() - t0:.0f}s")
    return written
