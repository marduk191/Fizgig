"""Stale-cache resolution guard is architecture-aware — headless, no GPU (issue #27).

Klein's FLUX.2 AE packs 2x2 space-to-channel after its /8 encoder, so Klein cache latents
are pixel/16; Krea 2's Qwen-Image VAE stores plain pixel/8. v3.0.0's guard compared both
against /8, which rejected every valid Klein cache — including one written seconds earlier
in the same run — ending in "No training items found". These tests pin the per-architecture
factor, that genuinely-stale caches are still caught, and the 992/496 aliasing trap
(992//16 == 496//8 == 62: a cross-factor "accept either" guard would wave stale caches
through — the factor must come from the architecture, never from what happens to match).

Run: venv/Scripts/python.exe tests/test_cache_reso_guard.py
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch
from safetensors.torch import save_file

from fizgig.dataset.image_dataset import ImageDataset, LATENT_SPATIAL_FACTOR

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


TD = tempfile.mkdtemp(prefix="reso_guard_")


def cache_with_latent(name, h, w):
    path = os.path.join(TD, name)
    save_file({f"latent_{h}x{w}": torch.zeros(4, h, w)}, path)
    return path


check = ImageDataset.latent_cache_matches_reso

ck("factors: klein 16, krea2 8",
   LATENT_SPATIAL_FACTOR == {"klein9b": 16, "krea2": 8}, LATENT_SPATIAL_FACTOR)

# The issue-27 case: a fresh Klein cache at the run's own bucket must PASS.
f = cache_with_latent("k1.safetensors", 31, 31)               # 496x496 bucket, packed /16
ck("klein fresh cache accepted (the #27 fix)", check(f, (496, 496), "klein9b") is True)
f = cache_with_latent("k2.safetensors", 23, 41)               # 656x368 bucket, packed /16
ck("klein non-square bucket accepted", check(f, (656, 368), "klein9b") is True)

# The guard's actual job: a Klein cache from a DIFFERENT Target Megapixels must FAIL.
f = cache_with_latent("k3.safetensors", 62, 62)               # written at a 992x992 bucket
ck("klein stale (992-run cache in a 496 run) rejected", check(f, (496, 496), "klein9b") is False)
f = cache_with_latent("k4.safetensors", 31, 31)               # written at a 496x496 bucket
ck("klein stale (496-run cache in a 992 run) rejected", check(f, (992, 992), "klein9b") is False)

# Krea 2 stores /8 — correct passes, stale fails.
f = cache_with_latent("q1.safetensors", 62, 62)
ck("krea2 fresh cache accepted", check(f, (496, 496), "krea2") is True)
ck("krea2 aliasing trap: 992-pixel bucket needs /8=124, not the /16 match",
   check(f, (992, 992), "krea2") is False)
f = cache_with_latent("q2.safetensors", 32, 56)               # 448x256 (w,h) -> latent (h/8, w/8)
ck("krea2 landscape accepted (set compare, orientation-proof)",
   check(f, (448, 256), "krea2") is True)

# Failure modes stay soft: unreadable/unknown -> None (caller decides), never a throw.
ck("unknown architecture -> None", check(f, (496, 496), "wan") is None)
ck("unreadable file -> None", check(os.path.join(TD, "missing.safetensors"),
                                    (496, 496), "klein9b") is None)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
