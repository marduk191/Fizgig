"""INT8 training path: does it produce the right numbers, and is it actually faster?

Checks against a bf16 reference at Krea 2's real shape:
  1. forward output error
  2. GRADIENT error — the thing that decides whether a LoRA trains properly
  3. speed of forward, and of forward+backward
"""
import sys
import time

sys.path.insert(0, r"W:\Peter\Documents\Development\Fizgig\src")
import torch
import torch.nn as nn

from fizgig.modules.int8_train import apply_int8_training

DEV = "cuda"
M, K, N = 1024, 6144, 6144
torch.manual_seed(0)


def make_linear():
    lin = nn.Linear(K, N, bias=False).to(DEV, torch.bfloat16)
    lin.weight.requires_grad_(False)
    return lin


def rel_err(a, b):
    return ((a - b).abs().mean() / b.abs().mean().clamp_min(1e-12)).item()


ref = make_linear()
w_ref = ref.weight.detach().clone()
x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.5

# bf16 reference: exact forward and gradient
x_ref = x.clone().requires_grad_(True)
gy = torch.randn(M, N, device=DEV, dtype=torch.bfloat16)   # realistic, not all-ones
y_ref = ref(x_ref)
y_ref.backward(gy)
g_ref = x_ref.grad.clone()

print(f"reference: y {tuple(y_ref.shape)} | grad {tuple(g_ref.shape)}\n")

results = {}
for mode in ("bf16", "int8"):
    m = make_linear()
    m.weight.data = w_ref.clone()
    holder = nn.Module()
    holder.blocks = nn.ModuleList([m])           # named "blocks.0" by named_modules()
    apply_int8_training(holder, target_keys=("blocks.",), grad_mode=mode)

    xi = x.clone().requires_grad_(True)
    yi = m(xi)
    yi.backward(gy)
    fe, ge = rel_err(yi.detach(), y_ref.detach()), rel_err(xi.grad, g_ref)
    results[mode] = (fe, ge)
    print(f"grad_mode={mode:5s}  forward rel-err {fe:.2e}   GRADIENT rel-err {ge:.2e}")

print()


def bench(fn, iters=30):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


base = make_linear(); base.weight.data = w_ref.clone()
xb = x.clone().requires_grad_(True)


def fwd_bf16():
    return base(xb)


def fwd_bwd_bf16():
    base.zero_grad(set_to_none=True)
    if xb.grad is not None:
        xb.grad = None
    base(xb).sum().backward()


timings = {"bf16 dense (reference)": (bench(fwd_bf16), bench(fwd_bwd_bf16))}
for mode in ("bf16", "int8"):
    m = make_linear(); m.weight.data = w_ref.clone()
    h = nn.Module(); h.blocks = nn.ModuleList([m])
    apply_int8_training(h, target_keys=("blocks.",), grad_mode=mode)
    xm = x.clone().requires_grad_(True)

    def f(_m=m, _x=xm):
        return _m(_x)

    def fb(_m=m, _x=xm):
        if _x.grad is not None:
            _x.grad = None
        _m(_x).sum().backward()

    timings[f"int8 (grad={mode})"] = (bench(f), bench(fb))

print(f"{'config':<26} {'forward':>10} {'fwd+bwd':>10}")
print("-" * 48)
for k, (a, b) in timings.items():
    print(f"{k:<26} {a:>8.3f}ms {b:>8.3f}ms")

print()
fb_ref = timings["bf16 dense (reference)"][1]
for mode in ("bf16", "int8"):
    sp = fb_ref / timings[f"int8 (grad={mode})"][1]
    print(f"  grad={mode:5s}: {sp:.2f}x on fwd+bwd, gradient rel-err {results[mode][1]:.2e}")
