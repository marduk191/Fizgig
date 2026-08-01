"""Paired-image (edit-style / temporal-displacement) training path — CPU, tiny DiT.

The sequence is [noisy target @ RoPE frame 0 | clean source @ frame 1 | text], loss on target
tokens only — the krea2_edit ecosystem convention. These tests pin: the plain path unchanged,
the control path finite with gradients flowing, the source ACTUALLY read (different source →
different prediction), control-latent caching keys, and the timestep window.
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fizgig.krea2.model import SingleStreamDiT, SingleMMDiTConfig  # noqa: E402
from fizgig.krea2.trainer import compute_loss, sample_krea2_timesteps  # noqa: E402
from fizgig.krea2.sampling import patchify_block, prepare  # noqa: E402

FAILS = []


def ck(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


torch.manual_seed(0)
cfg = SingleMMDiTConfig(features=64, tdim=32, txtdim=48, heads=4, multiplier=4,
                        layers=2, patch=2, channels=16, txtlayers=2, txtheads=4, txtkvheads=4)
dit = SingleStreamDiT(cfg)
dit.train()
B, h, w, seq = 2, 16, 24, 8
latent = torch.randn(B, 16, h, w)
src = torch.randn(B, 16, h, w)
hid = torch.randn(B, seq, 2, 48)
mask = torch.ones(B, seq, dtype=torch.bool)

# --- 1. position ids ----------------------------------------------------------------------
tok, pos, m = patchify_block(latent, 2, frame=0.0)
ck("frame 0: axis-0 ids all zero", torch.all(pos[..., 0] == 0).item())
tok1, pos1, _ = patchify_block(latent, 2, frame=1.0)
ck("frame 1: axis-0 ids all one, h/w ids identical",
   torch.all(pos1[..., 0] == 1).item() and torch.equal(pos1[..., 1:], pos[..., 1:]))
p_tok, p_pos, p_mask = prepare(latent, seq, 2, mask)
ck("prepare() unchanged: image ids frame 0, text ids all-zero",
   torch.all(p_pos[:, :tok.shape[1], 0] == 0).item()
   and torch.all(p_pos[:, tok.shape[1]:] == 0).item())

# --- 2. loss paths ------------------------------------------------------------------------
loss0, _ = compute_loss(dit, latent, hid, mask, device="cpu", dtype=torch.bfloat16)
ck("plain path finite", torch.isfinite(loss0).item(), f"{loss0.item():.4f}")

loss1, _ = compute_loss(dit, latent, hid, mask, device="cpu", dtype=torch.bfloat16,
                        control_latent=src)
loss1.backward()
gsum = sum(p.grad.abs().sum().item() for p in dit.parameters() if p.grad is not None)
ck("control path finite, grads flow", torch.isfinite(loss1).item() and gsum > 0,
   f"loss={loss1.item():.4f} grad_sum={gsum:.1f}")

torch.manual_seed(7)
lA, _ = compute_loss(dit, latent, hid, mask, device="cpu", dtype=torch.bfloat16,
                     control_latent=src)
torch.manual_seed(7)
lB, _ = compute_loss(dit, latent, hid, mask, device="cpu", dtype=torch.bfloat16,
                     control_latent=src * 3)
ck("the source is READ: different source -> different prediction",
   abs(lA.item() - lB.item()) > 1e-9, f"|d|={abs(lA.item() - lB.item()):.2e}")

# --- 3. control-latent caching ------------------------------------------------------------
from fizgig.krea2.caching import save_latent_cache_krea2  # noqa: E402
from fizgig.dataset.image_dataset import ItemInfo  # noqa: E402
from safetensors import safe_open  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    item = ItemInfo("pairtest", "cap", (448, 256), (448, 256),
                    latent_cache_path=os.path.join(td, "pairtest_0448x0256_krea2.safetensors"))
    save_latent_cache_krea2(item, torch.randn(16, 32, 56),
                            control_latents=[torch.randn(16, 32, 56)])
    with safe_open(item.latent_cache_path, framework="pt") as f:
        keys = set(f.keys())
    ck("cache carries latent + control keys",
       "latent_32x56" in keys and "latent_control_0_32x56" in keys, sorted(keys))
    # master's reso-guard must keep reading the MAIN latent, not the control
    from fizgig.dataset.image_dataset import ImageDataset  # noqa: E402
    ck("reso-guard reads the main latent (control excluded)",
       ImageDataset.latent_cache_matches_reso(item.latent_cache_path, (448, 256), "krea2") is True)

# --- 4. timestep window -------------------------------------------------------------------
torch.manual_seed(1)
t = sample_krea2_timesteps(2000, 448, "cpu", min_timestep=0.4, max_timestep=1.0)
ck("window respected", t.min().item() >= 0.4 and t.max().item() <= 1.0,
   f"[{t.min():.3f}, {t.max():.3f}]")
t2 = sample_krea2_timesteps(2000, 448, "cpu")
ck("default window untouched", t2.min().item() < 0.1, f"min={t2.min():.3f}")

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
