"""Krea 2 training-loss paths through the driver — CPU, tiny DiT.

Krea 2 trains through the driver system now. Its native trainer (compute_loss,
sample_krea2_timesteps) was deleted, and with it the paired-image / temporal-displacement path
this file was written for: [noisy target @ frame 0 | clean source @ frame 1 | text]. That was
REMOVED, not moved - the driver refuses reference latents outright ("Krea 2 has no edit training").
So the checks that read the paired source are retired rather than faked, and what survives is
pinned against the driver: position ids, the plain loss, the refusal itself, the image-pair slider
loss that replaced motion weighting, control-latent caching keys, and the timestep window.
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fizgig.krea2.model import SingleStreamDiT, SingleMMDiTConfig  # noqa: E402
from fizgig.krea2.driver import Krea2Driver  # noqa: E402
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

# --- 2. loss paths (driver.loss_at: "the original compute_loss's arithmetic") ----------------
# The loss math needs no loaded model files, so the driver is built without its constructor.
D = Krea2Driver.__new__(Krea2Driver)
cond = {"hidden_states": hid, "attention_mask": mask}
noise = torch.randn(latent.shape)
t_mid = torch.full((B,), 0.5)

loss0 = D.loss_at(dit, latent, noise, t_mid, cond)
ck("plain path finite", torch.isfinite(loss0).item(), f"{loss0.item():.4f}")
loss0.backward()
gsum = sum(p.grad.abs().sum().item() for p in dit.parameters() if p.grad is not None)
ck("  grads flow", gsum > 0, f"grad_sum={gsum:.1f}")
dit.zero_grad()

# Paired training was removed for Krea 2. Pin that it is REFUSED, loudly: silently ignoring a
# reference would train an ordinary LoRA while the user believed they were training on pairs.
_refused = None
try:
    D.training_loss(dit, latent, cond, torch.Generator().manual_seed(0), refs=[src])
except RuntimeError as e:
    _refused = str(e)
ck("paired training is refused, not silently ignored",
   _refused is not None and "no edit training" in _refused, _refused)

# --- 2b. image-pair slider weighting (the successor to motion-weighted loss) ---------------
# Same contract the motion weight had: weights come from the CLEAN pair difference, are renormed
# to per-sample mean 1, and a pair with no difference degrades to uniform weights, so weighting can
# never change the loss of an identical pair.
lw0 = D.loss_at(dit, latent, noise, t_mid, cond)
lw7 = D.loss_at(dit, latent, noise, t_mid, cond, diff_ref=src, diff_weight=0.7)
ck("difference weight finite and CHANGES the loss when the pair differs",
   torch.isfinite(lw7).item() and abs(lw7.item() - lw0.item()) > 1e-9,
   f"|d|={abs(lw7.item() - lw0.item()):.2e}")
lw7.backward()
ck("weighted path grads flow",
   any(p.grad is not None and p.grad.abs().sum() > 0 for p in dit.parameters()))
dit.zero_grad()

ls0 = D.loss_at(dit, latent, noise, t_mid, cond)
ls7 = D.loss_at(dit, latent, noise, t_mid, cond, diff_ref=latent.clone(), diff_weight=0.7)
ck("identical pair: weighted == unweighted (uniform-degrade invariant)",
   abs(ls7.item() - ls0.item()) < 1e-5, f"|d|={abs(ls7.item() - ls0.item()):.2e}")

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

# --- 4. timestep window (driver._sample_t, one draw per call) ------------------------------
g = torch.Generator().manual_seed(1)
t = torch.cat([D._sample_t(48, g, 0.4, 1.0) for _ in range(2000)])
ck("window respected", t.min().item() >= 0.4 and t.max().item() <= 1.0,
   f"[{t.min():.3f}, {t.max():.3f}]")
g2 = torch.Generator().manual_seed(1)
t2 = torch.cat([D._sample_t(48, g2) for _ in range(2000)])
ck("default window untouched", t2.min().item() < 0.1, f"min={t2.min():.3f}")

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
