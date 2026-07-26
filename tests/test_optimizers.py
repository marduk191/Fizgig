"""Every optimizer the dropdown offers must actually construct and step.

The catalog is filtered by what's importable, so this test is a check that the filter is honest:
if `available_optimizers()` lists it, a real parameter must survive a real step. Also covers the
free-form module path and the fallback, since both are user-reachable through the GUI's Optimizer
Type field.
"""
import sys

sys.path.insert(0, r"W:\Peter\Documents\Development\Fizgig\src")
import torch
import torch.nn as nn

from fizgig.training.optimizers import (available_optimizers, create_optimizer,
                                        describe, parse_optimizer_args)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

assert parse_optimizer_args("weight_decay=0.01 betas=0.9,0.99") == {
    "weight_decay": 0.01, "betas": (0.9, 0.99)}
assert parse_optimizer_args("") == {}
print("arg parsing ok")

for name in available_optimizers():
    p = [nn.Parameter(torch.randn(8, 8, device=DEV))]
    before = p[0].detach().clone()
    # Family-appropriate LR: Lion wants ~1/10 of AdamW's.
    lr = 1e-5 if name == "lion8bit" else 1e-4
    opt, label = create_optimizer(name, p, lr)
    p[0].grad = torch.randn_like(p[0])
    opt.step()
    assert not torch.equal(before, p[0].detach()), f"{name} did not move the parameter"
    print(f"  {name:20s} {type(opt).__name__:22s} {describe(name)}")

p = [nn.Parameter(torch.randn(8, 8, device=DEV))]
opt, label = create_optimizer("bitsandbytes.optim.PagedLion8bit", p, 1e-5)
assert type(opt).__name__ == "PagedLion8bit", type(opt).__name__
print(f"module path -> {label}")

opt, label = create_optimizer("does_not_exist", p, 1e-4)
assert isinstance(opt, torch.optim.AdamW) and "fallback" in label
print(f"unknown name -> {label} (run continues rather than dying at minute one)")
