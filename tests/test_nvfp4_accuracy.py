"""NVFP4 feasibility: can we quantise a real weight to fp4 and get the right answer back?

The raw matmul is 0.070 ms at Krea 2's shape (vs 0.121 int8, 0.594 bf16) — but that was random
bits. The question that decides whether NVFP4 is usable is ACCURACY: it is 4-bit like NF4, but
with per-16-element scales instead of NF4's 64-element blocks, so it should land much closer to
int8 than to NF4.

NVFP4 format (Blackwell):
  values  float4_e2m1fn_x2  — e2m1, two packed per byte
  scales  float8_e4m3fn     — one per 16 elements along K, laid out per _scaled_mm's requirement
"""
import sys

sys.path.insert(0, r"W:\Peter\Documents\Development\Fizgig\src")
import torch

DEV = "cuda"
BLOCK = 16

# e2m1 representable magnitudes: sign * {0, .5, 1, 1.5, 2, 3, 4, 6}
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MAX = 6.0


def quantize_nvfp4(w: torch.Tensor):
    """(N,K) bf16 -> (packed fp4 values, fp8 per-16-element scales). Symmetric, no zero point."""
    N, K = w.shape
    assert K % BLOCK == 0, "K must be a multiple of 16"
    wb = w.float().reshape(N, K // BLOCK, BLOCK)
    amax = wb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (amax / _E2M1_MAX).to(torch.float8_e4m3fn)          # fp8 scale per 16 values
    s = scale.to(torch.float32).clamp_min(1e-12)
    q = (wb / s)
    # snap to the nearest representable e2m1 magnitude
    lut = _E2M1.to(q.device)
    idx = (q.abs().unsqueeze(-1) - lut).abs().argmin(dim=-1)
    mag = lut[idx]
    vals = torch.sign(q) * mag
    return vals.reshape(N, K), scale.reshape(N, K // BLOCK), s.reshape(N, K // BLOCK)


def dequantize_nvfp4(vals, s):
    N, K = vals.shape
    return (vals.reshape(N, K // BLOCK, BLOCK) * s.unsqueeze(-1)).reshape(N, K)


def rel(a, b):
    return ((a - b).abs().mean() / b.abs().mean().clamp_min(1e-12)).item()


torch.manual_seed(0)
N = K = 6144
M = 512
w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.02
x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
ref = torch.nn.functional.linear(x, w)

# --- NVFP4 (weight-only, activations bf16 — the fair comparison against NF4) ---
vals, scale8, s = quantize_nvfp4(w)
w_deq = dequantize_nvfp4(vals, s).to(torch.bfloat16)
print(f"  NVFP4 weight-only  rel-err : {rel(torch.nn.functional.linear(x, w_deq), ref):.3e}")

# --- NF4, for comparison ---
try:
    import bitsandbytes.functional as bf
    q4, st = bf.quantize_4bit(w.to(torch.bfloat16), quant_type="nf4")
    w_nf4 = bf.dequantize_4bit(q4, st, quant_type="nf4").to(torch.bfloat16)
    print(f"  NF4   weight-only  rel-err : {rel(torch.nn.functional.linear(x, w_nf4), ref):.3e}")
except Exception as e:
    print("  NF4 compare failed:", str(e)[:80])

# --- INT8 W8A8, for comparison ---
s8 = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
w8 = (w / s8).round().clamp(-127, 127).to(torch.int8)
xs = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8).float() / 127.0
x8 = (x.float() / xs).round().clamp(-127, 127).to(torch.int8)
o8 = (torch._int_mm(x8, w8.t()).to(torch.bfloat16)
      * xs.to(torch.bfloat16) * s8.reshape(1, -1).to(torch.bfloat16))
print(f"  INT8  W8A8         rel-err : {rel(o8, ref):.3e}")

# --- can the real fp4 tensor-core path reproduce it? ---
print()
try:
    # pack two 4-bit values per byte, then view as the fp4 dtype
    def pack(vals_):
        lut = _E2M1.to(vals_.device)
        idx = (vals_.abs().unsqueeze(-1) - lut).abs().argmin(dim=-1).to(torch.uint8)
        idx = idx | (torch.sign(vals_) < 0).to(torch.uint8) * 8      # sign in bit 3
        lo, hi = idx[:, 0::2], idx[:, 1::2]
        return (lo | (hi << 4)).contiguous()

    wp = pack(vals).view(torch.float4_e2m1fn_x2)
    xv, xs4, xsf = quantize_nvfp4(x)
    xp = pack(xv).view(torch.float4_e2m1fn_x2)
    out = torch._scaled_mm(xp, wp.t(),
                           scale_a=xs4.flatten().contiguous(),
                           scale_b=scale8.flatten().contiguous(),
                           out_dtype=torch.bfloat16)
    print(f"  fp4 _scaled_mm ran -> {tuple(out.shape)}; rel-err vs bf16: {rel(out, ref):.3e}")
except Exception as e:
    print(f"  fp4 _scaled_mm path: {str(e)[:180]}")
