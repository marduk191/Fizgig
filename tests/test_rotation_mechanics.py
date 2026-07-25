"""Rotation fine-tuning mechanics, on a synthetic fp8-patched model (fast, no 26GB load).

Proves the properties the real run depends on:
1. schedule covers every block exactly once per cycle
2. activate -> weights are bf16 + require grad; frozen blocks do not
3. a training step actually moves the active weights
4. deactivate -> trained values land in the CPU master (NOT the lossy fp8 copy)
5. reactivate -> the trained bf16 values come back EXACTLY (no fp8 round-trip)
6. the frozen fp8 forward still runs after deactivation
"""
import sys
import torch
import torch.nn as nn

sys.path.insert(0, r"W:\Peter\Documents\Development\Fizgig\src")
from fizgig.krea2.rotation import RotationSchedule, BlockRotator
from fizgig.krea2.fp8_optimization_utils import (
    calculate_fp8_maxval, quantize_weight, apply_fp8_monkey_patch,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FP8 = torch.float8_e4m3fn

# ---------- 1. schedule ----------
s = RotationSchedule(n_blocks=28, active=4, rotate_every=1)
print(s.describe())
seen = set()
for e in range(s.cycle_epochs):
    seen |= set(s.active_at(e))
assert seen == set(range(28)), "a full cycle must cover every block"
assert s.active_at(0) == [0, 1, 2, 3] and s.active_at(1) == [4, 5, 6, 7]
assert s.active_at(s.cycle_epochs) == s.active_at(0), "cycle must wrap"
s2 = RotationSchedule(28, active=5, rotate_every=2)   # ragged: 28/5 -> last window short
cov = set()
for e in range(s2.cycle_epochs):
    cov |= set(s2.active_at(e))
assert cov == set(range(28)), "ragged windows must still cover everything"
print("  schedule ok (incl. ragged windows + wrap)")

# ---------- build a toy 4-block model, fp8-quantized like the real loader ----------
class Blk(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, d, bias=False)
        self.mlp = nn.Linear(d, d, bias=False)
    def forward(self, x):
        return self.mlp(torch.relu(self.attn(x)))

D = 128
torch.manual_seed(0)
blocks = nn.ModuleList([Blk(D) for _ in range(4)]).to(DEV, torch.bfloat16)

master = {}
opt_sd = {}
maxv = calculate_fp8_maxval(4, 3)
for bi, blk in enumerate(blocks):
    for lname, lin in [("attn", blk.attn), ("mlp", blk.mlp)]:
        key = f"blocks.{bi}.{lname}.weight"
        master[key] = lin.weight.detach().to("cpu", torch.bfloat16).clone()
        q, sc = quantize_weight(key, lin.weight.detach().float(), FP8, maxv, -maxv,
                                quantization_mode="block", block_size=64)
        opt_sd[key] = q
        opt_sd[f"blocks.{bi}.{lname}.scale_weight"] = sc
        lin.weight = nn.Parameter(q.to(DEV), requires_grad=False)

holder = nn.Module(); holder.blocks = blocks
apply_fp8_monkey_patch(holder, opt_sd)
for bi, blk in enumerate(blocks):
    for lname, lin in [("attn", blk.attn), ("mlp", blk.mlp)]:
        # Assign (not copy_): register_buffer makes these on CPU and inherits the *fp8*
        # weight dtype, whereas the real loader replaces them via load_state_dict(assign=True)
        # with the file's scale in a compute dtype. Mirror that here.
        shape = lin.scale_weight.shape
        lin.scale_weight = (opt_sd[f"blocks.{bi}.{lname}.scale_weight"]
                            .to(DEV, torch.bfloat16).reshape(shape))

x = torch.randn(2, D, device=DEV, dtype=torch.bfloat16)
def fwd(t):
    for b in blocks:
        t = b(t)
    return t
base_out = fwd(x)
assert torch.isfinite(base_out).all(), "fp8 frozen forward broken"
print("  fp8 frozen forward ok")

# ---------- 2. activate ----------
rot = BlockRotator(blocks, master, key_prefix="blocks", device=DEV, fp8_dtype=FP8)
rot.activate([0, 1])
assert blocks[0].attn.weight.dtype == torch.bfloat16 and blocks[0].attn.weight.requires_grad
assert blocks[2].attn.weight.dtype == FP8 and not blocks[2].attn.weight.requires_grad
params = rot.trainable_params()
assert len(params) == 4, f"expected 4 trainable Linears, got {len(params)}"
print(f"  activated 2 blocks -> {len(params)} trainable tensors, rest still fp8")

# ---------- 3. a step must move them ----------
before = blocks[0].attn.weight.detach().clone()
frozen_before = blocks[2].attn.weight.detach().clone()
opt = torch.optim.AdamW(params, lr=1e-2)
loss = fwd(x).float().pow(2).mean()
loss.backward()
opt.step(); opt.zero_grad()
assert not torch.equal(before, blocks[0].attn.weight.detach()), "active weights did not move"
assert torch.equal(frozen_before, blocks[2].attn.weight.detach()), "frozen weights moved!"
print("  training step moves active blocks only")

# ---------- 4/5. deactivate -> master, reactivate -> exact ----------
trained = blocks[0].attn.weight.detach().to("cpu", torch.bfloat16).clone()
rot.deactivate([0, 1])
assert blocks[0].attn.weight.dtype == FP8, "did not re-quantize on deactivate"
assert torch.equal(master["blocks.0.attn.weight"], trained), "master did not capture trained values"
out_after = fwd(x)
assert torch.isfinite(out_after).all(), "fp8 forward broken after deactivate"
print("  deactivate -> master holds trained bf16, fp8 forward still runs")

rot.activate([0])
assert torch.equal(blocks[0].attn.weight.detach().cpu(), trained), \
    "reactivated weights lost precision — training round-tripped through fp8!"
print("  reactivate -> exact bf16 restore (no fp8 precision loss across rotations)")

# ---------- 6. rotate_to convenience ----------
rot.rotate_to([2, 3])
assert sorted(rot.active) == [2, 3]
assert blocks[2].attn.weight.requires_grad and not blocks[0].attn.weight.requires_grad
assert torch.isfinite(fwd(x)).all()
sd = rot.master_state_dict()
assert len(sd) == 8 and all(v.dtype == torch.bfloat16 for v in sd.values())
print("  rotate_to swaps windows; master_state_dict exports full bf16 model")

print("\nALL ROTATION MECHANICS PASSED")
