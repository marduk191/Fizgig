"""INT8 training path: does it produce the right numbers, and is it actually faster? (needs a CUDA card)

Upstream removed the INT8-backward option and apply_int8_training (28920b29, #221): rounding each gradient row to INT8
let its outliers set the scale and zeroed most of the row (a LoRA's gradients came out at cosine 0.2-0.6 to the exact
ones). INT8 layers now keep the INT8 forward and the exact bf16 backward, set up by families.quant._quantize_one, the
step quantize() takes for each block Linear. So this drives that, both kinds, against a bf16 reference at Krea 2's
real shape:
  1. forward error;
  2. GRADIENT error, the thing that decides whether a LoRA trains properly (now bf16 against the INT8 weight, so it
     carries only the weight's rounding, never the gradient's);
  3. the speed of forward and of forward + backward (printed, not asserted: it depends on the card).

Run: venv/Scripts/python.exe tests/test_int8_train.py
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# src only. The repo root holds Fizgig 8's fizgig.py, which would shadow the fizgig package.
sys.path.insert(0, os.path.join(REPO, "src"))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from fizgig.families import quant  # noqa: E402

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


if not torch.cuda.is_available():
    print("SKIP  no CUDA card")
    sys.exit(0)

DEV = "cuda"
M, K, N = 1024, 6144, 6144
torch.manual_seed(0)


class _Driver:
    """What _quantize_one reads off a family driver."""

    def __init__(self, weight_only):
        self.int8_weight_only = weight_only
        self.int8_fp32_scales = True


def make_linear(w):
    lin = nn.Linear(K, N, bias=False).to(DEV, torch.bfloat16)
    lin.weight.data.copy_(w)
    lin.weight.requires_grad_(False)
    return lin


def rel_err(a, b):
    return ((a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-12)).item()


w0 = (torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16)
x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.5
# realistic, not all-ones: a diffusion gradient carries a few large outlier channels per row. Those are what made an
# INT8-rounded gradient fail (#221): the outlier sets the row's scale and rounds the rest towards zero. Without them
# a rounded gradient still reads cosine 0.9999 and nothing here would notice it coming back.
gy = torch.randn(M, N, device=DEV, dtype=torch.bfloat16)
gy[:, torch.randperm(N, device=DEV)[:16]] *= 400

ref = make_linear(w0)
x_ref = x.clone().requires_grad_(True)
y_ref = ref(x_ref)
y_ref.backward(gy)
g_ref = x_ref.grad.detach().clone()

results = {}
for weight_only in (False, True):
    kind = "weight-only" if weight_only else "weights+activations"
    lin = make_linear(w0)
    quant._quantize_one(lin, _Driver(weight_only), "int8", DEV)
    ck(f"{kind}: the weight is stored as int8 with a per-row scale",
       lin.weight.dtype == torch.int8 and getattr(lin, "_is_int8", False) and lin._int8_wscale.numel() == N,
       (lin.weight.dtype, tuple(lin._int8_wscale.shape)))
    xq = x.clone().requires_grad_(True)
    y = lin(xq)
    y.backward(gy)
    fe, ge = rel_err(y, y_ref), rel_err(xq.grad, g_ref)
    cos = torch.nn.functional.cosine_similarity(xq.grad.float().flatten(), g_ref.float().flatten(), dim=0).item()
    results[kind] = (lin, fe, ge, cos)
    print(f"      {kind:20s} forward rel-err {fe:.2e}   GRADIENT rel-err {ge:.2e}   gradient cosine {cos:.5f}")
    ck(f"{kind}: forward within 2% of bf16", fe < 2e-2, f"{fe:.2e}")
    # an INT8-rounded gradient measured cosine 0.2-0.6 (#221); the bf16 backward only carries the weight's rounding
    ck(f"{kind}: the gradient is the exact bf16 backward (cosine > 0.999, rel-err < 2%)", cos > 0.999 and ge < 2e-2,
       f"cos {cos:.5f}, rel-err {ge:.2e}")
ck("weight-only keeps the activations bf16: its forward is no further from bf16 than the activation-rounded one",
   results["weight-only"][1] <= results["weights+activations"][1] * 1.05,
   (f"{results['weight-only'][1]:.2e}", f"{results['weights+activations'][1]:.2e}"))


def bench(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def fwd_bwd(layer):
    def run():
        xi = x.clone().requires_grad_(True)
        layer(xi).backward(gy)
    return run


print()
print(f"      {'config':<26} {'forward':>10} {'fwd+bwd':>10}")
for name, layer in (("bf16", ref), *((f"int8 ({k})", v[0]) for k, v in results.items())):
    with torch.no_grad():
        f = bench(lambda: layer(x))
    print(f"      {name:<26} {f:>8.3f}ms {bench(fwd_bwd(layer)):>8.3f}ms")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
