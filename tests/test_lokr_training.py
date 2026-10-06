"""Trainable LoKR (Phase 1): module math, init, and the save->reload round-trip.

Everything here runs on CPU with toy dims — the point is exactness against the dense
kron reference and compatibility with the loaders the rest of the app already uses,
not speed. GPU behaviour is covered by the smoke training run in Phase 6.
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from fizgig.networks.lora import (LoKRModule, factorization, create_network,  # noqa: E402
                                  create_network_from_weights, detect_lora_format,
                                  ensure_kohya_lora_state_dict)

FAILS = []


def ck(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


torch.manual_seed(0)

# --- 1. factorization ---------------------------------------------------------------------
ck("factorization: exact split at the factor", factorization(6144, 8) == (8, 768))
ck("  factor larger than sqrt clamps to a divisor", factorization(64, 8) == (8, 8))
ck("  non-divisor factor takes the largest divisor below", factorization(24, 5) == (4, 6))
ck("  prime dims degenerate to (1, n), not an error", factorization(13, 8) == (1, 13))
ck("  factor 1", factorization(100, 1) == (1, 100))
for n, f in ((6144, 8), (24576, 16), (30, 4), (7, 3)):
    a, b = factorization(n, f)
    ck(f"  product invariant {n}/{f}", a * b == n and a <= f, (a, b))

# --- 2. LoKRModule math -------------------------------------------------------------------
lin = torch.nn.Linear(24, 16, bias=False)
mod = LoKRModule("lora_unet_toy_lin", lin, multiplier=1.0, lora_dim=4, alpha=1, factor=4)

ck("w1/w2 shapes obey a*c==out, b*d==in",
   mod.a * mod.c == 16 and mod.b * mod.d == 24, (mod.a, mod.b, mod.c, mod.d))
ck("alpha buffer is 1.0 and scale is 1.0",
   float(mod.alpha) == 1.0 and mod.scale == 1.0)
ck("delta is exactly zero at init (w2 zeroed)",
   torch.all(mod.lokr_w2 == 0) and not torch.all(mod.lokr_w1 == 0))

x = torch.randn(3, 24)
base_out = lin(x)
mod.apply_to()
ck("apply_to removed org_module (frozen base stays out of state_dict)",
   not hasattr(mod, "org_module") and
   all("org_module" not in k for k in mod.state_dict().keys()), list(mod.state_dict().keys()))
ck("zero-init forward == base forward exactly", torch.equal(mod.forward(x), base_out))

# Give the factors real values and check against the dense kron reference.
with torch.no_grad():
    mod.lokr_w1.copy_(torch.randn_like(mod.lokr_w1))
    mod.lokr_w2.copy_(torch.randn_like(mod.lokr_w2))
mod.multiplier = 0.7
ref = base_out + 0.7 * (x @ torch.kron(mod.lokr_w1, mod.lokr_w2).T)
got = mod.forward(x)
ck("forward matches dense kron reference",
   torch.allclose(got, ref, atol=1e-5), f"max diff {(got - ref).abs().max():.2e}")

mod.enabled = False
ck("enabled=False returns pure base output", torch.equal(mod.forward(x), base_out))
mod.enabled = True
mod.multiplier = 0.0
ck("multiplier=0 returns pure base output", torch.equal(mod.forward(x), base_out))
mod.multiplier = 1.0

# Gradients: w2 learns immediately; w1 unlocks once w2 is nonzero (same staging as
# LoRA's zeroed lora_up — only one side has grad at the very first step).
lin2 = torch.nn.Linear(24, 16, bias=False)
m2 = LoKRModule("lora_unet_toy_lin2", lin2, 1.0, 4, 1, factor=4)
m2.apply_to()
m2.forward(torch.randn(2, 24)).sum().backward()
ck("step-0 grads: w2 nonzero, w1 zero (w2 is the zeroed factor)",
   m2.lokr_w2.grad is not None and m2.lokr_w2.grad.abs().sum() > 0
   and (m2.lokr_w1.grad is None or torch.all(m2.lokr_w1.grad == 0)))
with torch.no_grad():
    m2.lokr_w2.add_(torch.randn_like(m2.lokr_w2))
m2.zero_grad()
m2.forward(torch.randn(2, 24)).sum().backward()
ck("once w2 is nonzero both factors receive grads",
   m2.lokr_w1.grad.abs().sum() > 0 and m2.lokr_w2.grad.abs().sum() > 0)

# --- 3. network build -> save -> reload round-trip ----------------------------------------
class ToyDiT(torch.nn.Module):
    """Two 'blocks' of Linears with dotted paths, mimicking the DiT walk."""
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        for _ in range(2):
            blk = torch.nn.Module()
            blk.attn = torch.nn.Module()
            blk.attn.qkv = torch.nn.Linear(24, 48, bias=False)
            blk.attn.out = torch.nn.Linear(16, 24, bias=False)
            self.blocks.append(blk)

    def forward(self, x):  # unused; networks patch the Linears directly
        return x


dit = ToyDiT()
net = create_network(None, "lora_unet", 1.0, 4, 1.0, None, [], dit,
                     module_class=LoKRModule, module_kwargs={"factor": 4})
net.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
ck("network built one LoKR module per Linear", len(net.unet_loras) == 4,
   [m.lora_name for m in net.unet_loras])
ck("  all modules are LoKRModule", all(isinstance(m, LoKRModule) for m in net.unet_loras))

# Real (nonzero) weights so the round-trip comparison is meaningful.
with torch.no_grad():
    for m in net.unet_loras:
        m.lokr_w1.copy_(torch.randn_like(m.lokr_w1))
        m.lokr_w2.copy_(torch.randn_like(m.lokr_w2))

with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "toy_lokr.safetensors")
    net.save_weights(p, torch.float32, {"ss_test": "1"})
    sd = load_file(p)

    ck("saved keys are native lokr suffixes",
       any(k.endswith(".lokr_w1") for k in sd) and any(k.endswith(".lokr_w2") for k in sd)
       and any(k.endswith(".alpha") for k in sd), sorted(sd.keys())[:4])
    ck("  no lora_up/lora_down keys",
       not any("lora_up" in k or "lora_down" in k for k in sd))
    ck("  detect_lora_format says lokr", detect_lora_format(sd) == "lokr")
    ck("  ensure_kohya passes native lokr through unchanged",
       ensure_kohya_lora_state_dict(dict(sd)).keys() == sd.keys())

    # Reload path — the exact chain previews and the context LoRA use.
    dit2 = ToyDiT()
    x = torch.randn(3, 24)
    ref_deltas = {}
    for m in net.unet_loras:
        w = torch.kron(m.lokr_w1, m.lokr_w2)
        ref_deltas[m.lora_name] = w

    inf_net = create_network_from_weights(None, 1.0, dict(sd), None, dit2, for_inference=True)
    inf_net.apply_to(text_encoders=None, unet=dit2, apply_text_encoder=False, apply_unet=True)
    missing = inf_net.load_state_dict(dict(sd), strict=False)
    ck("reloaded via create_network_from_weights: 4 inf modules",
       len(inf_net.unet_loras) == 4, [m.lora_name for m in inf_net.unet_loras])
    ok = True
    for m in inf_net.unet_loras:
        w_inf = m._w1() if hasattr(m, "_w1") else None
        kron_inf = torch.kron(m._w1(), m._w2()) * m.scale * m.multiplier
        if not torch.allclose(kron_inf, ref_deltas[m.lora_name], atol=1e-6):
            ok = False
    ck("  reloaded deltas match trained deltas to 1e-6 (scale round-trips)", ok)

# --- 4. comfy-format final save (the driver's FamilyLoRA.save) -----------------------------
# Krea 2 trains through the driver system now. The native trainer's _save_lora this section used to
# drive was deleted with it, and the save moved to fizgig.families.lora.FamilyLoRA.save — a separate
# LoKR implementation from fizgig.networks.lora's, which sections 1-3 cover. The invariant is the same:
# the final file ships LyCORIS-standard keys (diffusion_model.<dotted>.lokr_*), the format every
# ComfyUI LoKR in the wild uses, and the workbench's own loader must render it back exactly. Built on
# the REAL Krea 2 driver, against a toy with Krea 2's block layout, so the target list and the key
# stems come from the shipped description rather than from this file.
import torch.nn as nn  # noqa: E402
from fizgig.families.registry import get as _get_family  # noqa: E402
from fizgig.families.lora import FamilyLoRA, TRAINABLE  # noqa: E402

_K2 = _get_family("krea2").load_driver()


class _K2Attn(nn.Module):
    def __init__(s, d=16):
        super().__init__()
        for n in ("wq", "wk", "wv", "gate", "wo"):
            setattr(s, n, nn.Linear(d, d, bias=False))


class _K2MLP(nn.Module):
    def __init__(s, d=16):
        super().__init__()
        for n in ("gate", "up", "down"):
            setattr(s, n, nn.Linear(d, d, bias=False))


class _K2Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.attn, s.mlp = _K2Attn(), _K2MLP()


class Krea2Toy(nn.Module):
    """Krea 2's block layout (blocks.N.attn.{wq,wk,wv,gate,wo}, blocks.N.mlp.{gate,up,down}) at toy width."""

    def __init__(s, n=2):
        super().__init__()
        s.blocks = nn.ModuleList(_K2Block() for _ in range(n))


torch.manual_seed(0)
k2 = Krea2Toy()
fnet = FamilyLoRA(k2, _K2, device="cpu")
fnet.add_trainable(4, 4, kind="lokr", factor=4)
with torch.no_grad():
    for w in fnet.wrapped.values():
        a = w.adapters[TRAINABLE]
        a.lokr_w1.copy_(torch.randn_like(a.lokr_w1))
        a.lokr_w2.copy_(torch.randn_like(a.lokr_w2))
k2_ref = {f: w.adapters[TRAINABLE].delta().clone() for f, w in fnet.wrapped.items()}
ck("driver LoKR wraps every Krea 2 block Linear", len(fnet.wrapped) == 16, len(fnet.wrapped))

with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "final.safetensors")
    fnet.save(p, dtype=torch.float32)
    csd = load_file(p)

    ck("comfy save: keys are diffusion_model.<dotted>.lokr_*",
       "diffusion_model.blocks.0.attn.wq.lokr_w1" in csd
       and "diffusion_model.blocks.0.attn.wq.alpha" in csd, sorted(csd.keys())[:3])
    ck("  no flattened lora_unet_ keys remain", not any(k.startswith("lora_unet_") for k in csd))
    ck("  detect_lora_format on the comfy file says lokr", detect_lora_format(csd) == "lokr")

    # The full consumer chain: a driver-trained file -> the workbench loader -> the same deltas.
    back = ensure_kohya_lora_state_dict(dict(csd))
    k2b = Krea2Toy()
    inf3 = create_network_from_weights(None, 1.0, back, None, k2b, for_inference=True)
    inf3.apply_to(text_encoders=None, unet=k2b, apply_text_encoder=False, apply_unet=True)
    inf3.load_state_dict(back, strict=False)
    _by_name = {m.lora_name: m for m in inf3.unet_loras}
    ok = len(inf3.unet_loras) == 16
    for full, ref in k2_ref.items():
        m = _by_name.get("lora_unet_" + full.replace(".", "_"))
        if m is None or not torch.allclose(torch.kron(m._w1(), m._w2()) * m.scale * m.multiplier,
                                           ref, atol=1e-6):
            ok = False
    ck("  the workbench loader renders the driver's trained deltas exactly", ok, len(inf3.unet_loras))

    # And back into a fresh adapter set (resume / Load Last Train).
    fnet2 = FamilyLoRA(Krea2Toy(), _K2, device="cpu")
    fnet2.add_trainable(4, 4, kind="lokr", factor=4)
    n = fnet2.load_trainable(p)
    ck("  load_trainable restores every module", n == 16, n)
    ck("  ...with identical deltas",
       all(torch.allclose(fnet2.wrapped[f].adapters[TRAINABLE].delta(), r, atol=1e-6)
           for f, r in k2_ref.items()))

# Standard-LoRA regression: a plain LoRA from the same driver still saves kohya keys.
snet = FamilyLoRA(Krea2Toy(), _K2, device="cpu")
snet.add_trainable(4, 4.0, kind="lora")
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "std.safetensors")
    snet.save(p, dtype=torch.float32)
    ssd = load_file(p)
    ck("standard LoRA from the driver still saves kohya keys",
       any(k.startswith("lora_unet_") and k.endswith(".lora_down.weight") for k in ssd)
       and detect_lora_format(ssd) == "kohya", sorted(ssd)[:2])

# --- 4b. Context LoRA under a trainable LoKR ----------------------------------------------
# The trainer stacks: frozen context (inference net) -> trainable net, both additive forward
# patches. Training LoKR must not change that: output == base + ctx_delta + lokr_delta, and
# grads flow ONLY to the trainable LoKR.
dit_ctx = ToyDiT()
x = torch.randn(2, 24)
base_out = dit_ctx.blocks[0].attn.qkv(x)

# Frozen context: a standard LoRA built from weights (the _apply_context_lora path).
ctx_sd = {
    "lora_unet_blocks_0_attn_qkv.lora_down.weight": torch.randn(2, 24),
    "lora_unet_blocks_0_attn_qkv.lora_up.weight": torch.randn(48, 2),
    "lora_unet_blocks_0_attn_qkv.alpha": torch.tensor(2.0),
}
ctx_net = create_network_from_weights(None, 1.0, dict(ctx_sd), None, dit_ctx, for_inference=True)
ctx_net.apply_to(text_encoders=None, unet=dit_ctx, apply_text_encoder=False, apply_unet=True)
ctx_net.load_state_dict(dict(ctx_sd), strict=False)
ctx_net.requires_grad_(False)
ctx_delta = (ctx_sd["lora_unet_blocks_0_attn_qkv.lora_up.weight"].float()
             @ ctx_sd["lora_unet_blocks_0_attn_qkv.lora_down.weight"].float())

# Trainable LoKR on top, given real weights.
lokr_net = create_network(None, "lora_unet", 1.0, 4, 1.0, None, [], dit_ctx,
                          module_class=LoKRModule, module_kwargs={"factor": 4})
lokr_net.apply_to(text_encoders=None, unet=dit_ctx, apply_text_encoder=False, apply_unet=True)
with torch.no_grad():
    for m in lokr_net.unet_loras:
        m.lokr_w1.copy_(torch.randn_like(m.lokr_w1))
        m.lokr_w2.copy_(torch.randn_like(m.lokr_w2))
m0 = next(m for m in lokr_net.unet_loras if m.lora_name == "lora_unet_blocks_0_attn_qkv")
lokr_delta = torch.kron(m0.lokr_w1, m0.lokr_w2)

got = dit_ctx.blocks[0].attn.qkv.forward(x)
ref = base_out + x @ ctx_delta.T + x @ lokr_delta.T
ck("context LoRA + trainable LoKR stack additively (base + ctx + lokr)",
   torch.allclose(got, ref, atol=1e-4), f"max diff {(got - ref).abs().max():.2e}")
got.sum().backward()
ck("  grads reach the trainable LoKR only",
   m0.lokr_w1.grad is not None and m0.lokr_w1.grad.abs().sum() > 0
   and all(p.grad is None or p.grad.abs().sum() == 0 for p in ctx_net.parameters()))

# --- 5. lossless LoKR bake (Repair Studio / Explorer save path) ---------------------------
# Krea 2's Repair Studio engine was removed with its native trainer (the Krea 2 workbench runs
# through the driver now), and SliderState.default_krea2 went with it. save_repaired_lora still
# ships: the MiniMax H3 Repair Studio bakes through it (minimax/workbench.py). H3 shares the raw
# `lora_unet_blocks_N_` key shape, with block ids h3blk_N, so the same synthetic LoKR file below
# exercises the identical lossless path through the family that still uses it.
from fizgig.repair_studio.bake import save_repaired_lora  # noqa: E402
from fizgig.repair_studio.state import SliderState  # noqa: E402


def _mk_lokr_sd():
    """Two `lora_unet_blocks_N_` LoKR modules (H3 / formerly Krea 2 naming) (out 24 = 4x6, in 16 = 4x4), alpha 1.0."""
    g = torch.Generator().manual_seed(7)
    sd = {}
    for blk in (0, 1):
        sd[f"lora_unet_blocks_{blk}_attn_qkv.lokr_w1"] = torch.randn(4, 4, generator=g)
        sd[f"lora_unet_blocks_{blk}_attn_qkv.lokr_w2"] = torch.randn(6, 4, generator=g)
        sd[f"lora_unet_blocks_{blk}_attn_qkv.alpha"] = torch.tensor(1.0)
    return sd


def _dense(sd, blk):
    return torch.kron(sd[f"lora_unet_blocks_{blk}_attn_qkv.lokr_w1"].float(),
                      sd[f"lora_unet_blocks_{blk}_attn_qkv.lokr_w2"].float())


with tempfile.TemporaryDirectory() as td:
    src = os.path.join(td, "lokr_src.safetensors")
    save_file(_mk_lokr_sd(), src)

    # THE headline regression: a no-op edit keeps LoKR as LoKR, tensors byte-identical.
    st = SliderState.default_h3()
    out1 = os.path.join(td, "noop.safetensors")
    summary = save_repaired_lora(src, st, out1)
    osd = load_file(out1)
    ck("no-op bake: format stays lokr", detect_lora_format(osd) == "lokr")
    ck("  summary reports lycoris out, zero SVD",
       summary["format_out"] == "lycoris" and summary["lycoris_converted"] == 0, summary)
    ck("  tensors byte-identical (alpha included)",
       set(osd) == set(_mk_lokr_sd())
       and all(torch.equal(osd[k], v) for k, v in _mk_lokr_sd().items()))

    # Multiplier bake: dense delta of the baked module == m x original, with sentinel alpha.
    st2 = SliderState.default_h3()
    st2.blocks["h3blk_0"].primary_strength = 0.6
    st2.blocks["h3blk_1"].primary_enabled = False
    out2 = os.path.join(td, "scaled.safetensors")
    summary2 = save_repaired_lora(src, st2, out2)
    osd2 = load_file(out2)
    ck("scaled bake: still lokr, no SVD",
       detect_lora_format(osd2) == "lokr" and summary2["lycoris_converted"] == 0)
    ck("  disabled block dropped",
       not any("blocks_1" in k for k in osd2) and "h3blk_1" in summary2["dropped_blocks"])
    ref = 0.6 * _dense(_mk_lokr_sd(), 0)
    got = _dense(osd2, 0) * 1.0  # sentinel alpha -> scale 1.0 at load
    ck("  baked dense delta == 0.6 x original",
       torch.allclose(got, ref, atol=1e-5), f"max diff {(got - ref).abs().max():.2e}")
    ck("  alpha is the >=1e6 sentinel (scale baked in)",
       float(osd2["lora_unet_blocks_0_attn_qkv.alpha"]) >= 1e6)
    # And the loader agrees: reload the baked file and check the module's effective scale.
    from fizgig.networks.lora import lycoris_scale_from_keys
    mod_keys = {k.split(".", 1)[1]: v for k, v in osd2.items() if k.startswith("lora_unet_blocks_0")}
    ck("  lycoris_scale_from_keys reads the sentinel as 1.0",
       lycoris_scale_from_keys(mod_keys) == 1.0)

    # Standard-LoRA regression through the same path: behaviour unchanged.
    std = {
        "lora_unet_blocks_0_attn_qkv.lora_down.weight": torch.randn(2, 16),
        "lora_unet_blocks_0_attn_qkv.lora_up.weight": torch.randn(24, 2),
        "lora_unet_blocks_0_attn_qkv.alpha": torch.tensor(2.0),
    }
    src_std = os.path.join(td, "std_src.safetensors")
    save_file(std, src_std)
    st3 = SliderState.default_h3()
    st3.blocks["h3blk_0"].primary_strength = 0.5
    out3 = os.path.join(td, "std_out.safetensors")
    s3 = save_repaired_lora(src_std, st3, out3)
    osd3 = load_file(out3)
    ref_std = 0.5 * (std["lora_unet_blocks_0_attn_qkv.lora_up.weight"].float()
                     @ std["lora_unet_blocks_0_attn_qkv.lora_down.weight"].float())
    got_std = (osd3["lora_unet_blocks_0_attn_qkv.lora_up.weight"].float()
               @ osd3["lora_unet_blocks_0_attn_qkv.lora_down.weight"].float())
    ck("standard-LoRA bake unchanged: 0.5 x delta, alpha = rank",
       torch.allclose(got_std, ref_std, atol=1e-5)
       and float(osd3["lora_unet_blocks_0_attn_qkv.alpha"]) == 2.0
       and s3["format_out"] == "standard")

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
