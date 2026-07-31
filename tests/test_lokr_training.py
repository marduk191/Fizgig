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

# --- 4. comfy-format final save (trainer._save_lora) --------------------------------------
# The final artifact ships LyCORIS-standard keys (diffusion_model.<dotted>.lokr_*) — the
# format every ComfyUI LoKR in the wild uses — and must round-trip through our own loader.
from fizgig.krea2.trainer import _save_lora  # noqa: E402

net._network_type = "lokr"
net._lokr_factor = 4
net._dotted_names = {
    f"lora_unet_{name.replace('.', '_')}": name
    for name, m in dit.named_modules() if isinstance(m, torch.nn.Linear)
}

with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "final.safetensors")
    _save_lora(net, p, 4, 1.0, torch.float32, comfy_format=True)
    from safetensors import safe_open
    with safe_open(p, framework="pt") as f:
        meta = f.metadata()
    csd = load_file(p)

    ck("comfy save: keys are diffusion_model.<dotted>.lokr_*",
       "diffusion_model.blocks.0.attn.qkv.lokr_w1" in csd
       and "diffusion_model.blocks.0.attn.qkv.alpha" in csd, sorted(csd.keys())[:3])
    ck("  no flattened lora_unet_ keys remain", not any(k.startswith("lora_unet_") for k in csd))
    ck("  metadata records lokr module + factor",
       meta.get("ss_network_module") == "fizgig.krea2 (lokr, all-Linear)"
       and meta.get("ss_lokr_factor") == "4", meta)
    ck("  detect_lora_format on the comfy file says lokr", detect_lora_format(csd) == "lokr")

    back = ensure_kohya_lora_state_dict(dict(csd))
    native = {k: v for k, v in net.state_dict().items()}
    ck("  ensure_kohya round-trips comfy keys back to native names",
       set(back.keys()) == set(native.keys()),
       sorted(set(back.keys()) ^ set(native.keys()))[:4])
    ck("  ...with identical tensors",
       all(torch.equal(back[k], native[k].float()) or torch.allclose(back[k], native[k].to(back[k].dtype))
           for k in back))

    # And the full consumer chain: comfy file -> inf network -> same deltas as trained.
    dit3 = ToyDiT()
    inf3 = create_network_from_weights(None, 1.0, back, None, dit3, for_inference=True)
    inf3.apply_to(text_encoders=None, unet=dit3, apply_text_encoder=False, apply_unet=True)
    inf3.load_state_dict(back, strict=False)
    ok = all(torch.allclose(torch.kron(m._w1(), m._w2()) * m.scale * m.multiplier,
                            ref_deltas[m.lora_name], atol=1e-6) for m in inf3.unet_loras)
    ck("  comfy file renders the trained deltas exactly", ok and len(inf3.unet_loras) == 4)

# Standard-LoRA regression: comfy_format is a no-op for a normal network.
dit4 = ToyDiT()
lora_net = create_network(None, "lora_unet", 1.0, 4, 4.0, None, [], dit4)
lora_net.apply_to(text_encoders=None, unet=dit4, apply_text_encoder=False, apply_unet=True)
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "std.safetensors")
    _save_lora(lora_net, p, 4, 4.0, torch.float32, comfy_format=True)
    ssd = load_file(p)
    ck("standard LoRA with comfy_format=True still saves kohya keys",
       any(k.startswith("lora_unet_") and k.endswith(".lora_down.weight") for k in ssd)
       and detect_lora_format(ssd) == "kohya")

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
from fizgig.repair_studio.bake import save_repaired_lora  # noqa: E402
from fizgig.repair_studio.state import SliderState  # noqa: E402


def _mk_lokr_sd():
    """Two Krea 2-named LoKR modules (out 24 = 4x6, in 16 = 4x4), alpha 1.0."""
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
    st = SliderState.default_krea2()
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
    st2 = SliderState.default_krea2()
    st2.blocks["block_0"].primary_strength = 0.6
    st2.blocks["block_1"].primary_enabled = False
    out2 = os.path.join(td, "scaled.safetensors")
    summary2 = save_repaired_lora(src, st2, out2)
    osd2 = load_file(out2)
    ck("scaled bake: still lokr, no SVD",
       detect_lora_format(osd2) == "lokr" and summary2["lycoris_converted"] == 0)
    ck("  disabled block dropped",
       not any("blocks_1" in k for k in osd2) and "block_1" in summary2["dropped_blocks"])
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
    st3 = SliderState.default_krea2()
    st3.blocks["block_0"].primary_strength = 0.5
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
