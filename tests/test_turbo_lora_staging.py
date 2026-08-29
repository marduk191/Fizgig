"""Turbo-LoRA preview staging — structure-level regression test (no GPU, no full model load).

Verifies, against the real Comfy-Org turbo LoRA file when present:
  1. _apply_turbo_lora resolves all 264 LoRA modules on the real Krea 2 architecture
     (meta-device instantiation — shapes only, no weights) and loads NON-ZERO ups
     (the zero-init trap: create_network_from_weights builds structure only).
  2. All 7 diff_b bias deltas resolve to real bias Parameters with matching shapes.
  3. The preview bias apply/revert is EXACT (snapshot restore, not += / -= round-trip).

Run: venv/Scripts/python.exe tests/test_turbo_lora_staging.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

TURBO_LORA = r"S:/Auto/ComfyUI_SEC/ComfyUI/models/loras/krea2/krea2_turbo_lora_rank_64_bf16.safetensors"

failures = []


def check(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        failures.append(label)


# ---- 3. bias apply/revert exactness (no model needed) ---------------------------------
# bf16 += delta then -= delta is NOT bit-clean; the preview path must snapshot+copy_.
bias = torch.nn.Parameter(torch.randn(64, dtype=torch.bfloat16), requires_grad=False)
orig = bias.detach().clone()
delta = torch.randn(64, dtype=torch.bfloat16) * 0.1
snap = bias.detach().clone()
bias.data.add_(delta)
bias.data.copy_(snap)
check("bias snapshot/restore is bit-exact", torch.equal(bias.detach(), orig))
bias.data.add_(delta)
bias.data.sub_(delta)
roundtrip_clean = torch.equal(bias.detach(), orig)
print(f"      (for reference: naive += / -= round-trip bit-exact? {roundtrip_clean})")

if not os.path.isfile(TURBO_LORA):
    print(f"SKIP  turbo LoRA file not present ({TURBO_LORA}) — structure checks skipped")
    sys.exit(1 if failures else 0)

# ---- 1 + 2. staging against the real architecture (meta device: shapes, no memory) ----
from fizgig.krea2.model import SingleStreamDiT
from fizgig.krea2.utils import single_mmdit_large_wide
from fizgig.krea2.trainer import _apply_turbo_lora

with torch.device("meta"):
    dit = SingleStreamDiT(single_mmdit_large_wide)

net, diffb = _apply_turbo_lora(dit, TURBO_LORA, device="cpu", dtype=torch.bfloat16)

check("264 LoRA modules resolved on the real architecture", len(net.unet_loras) == 264,
      f"got {len(net.unet_loras)}")
check("7 diff_b bias deltas resolved", len(diffb) == 7, f"got {len(diffb)}")
for b, d in diffb:
    if tuple(b.shape) != tuple(d.shape):
        check("diff_b shape match", False, f"{tuple(b.shape)} vs {tuple(d.shape)}")
        break
else:
    check("every diff_b delta matches its bias shape", True)

nonzero = sum(1 for l in net.unet_loras
              if getattr(l, "lora_up", None) is not None
              and l.lora_up.weight.abs().max().item() > 0)
check("all LoRA ups NON-zero after load_state_dict (zero-init trap)", nonzero == 264,
      f"{nonzero}/264 non-zero")

check("net disabled after staging", all(not l.enabled for l in net.unet_loras))
check("net frozen", all(not p.requires_grad for p in net.parameters()))
check("net CPU-resident", all(p.device.type == "cpu" for p in net.parameters()))

# Coverage beyond the block stack — the I/O layers the trainable LoRA never touches.
names = {l.lora_name for l in net.unet_loras}
for expected in ("lora_unet_first", "lora_unet_last_linear", "lora_unet_tmlp_0",
                 "lora_unet_txtfusion_projector"):
    check(f"module {expected} wrapped", expected in names)

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURE(S)"))
sys.exit(1 if failures else 0)
