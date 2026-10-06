"""NF4 park/restore must bring BOTH halves of the model back.

Regression for issue #17 — training died at the first step after an auto-recaption with

    RuntimeError: Expected all tensors to be on the same device, but got mat2 is on
    cuda:0, different from other tensors on cpu

The trainer parks the DiT on CPU to make room for Qwen3-VL. An NF4 model is split across
two storage mechanisms, so parking takes two calls:

    dit.to("cpu")                    # ordinary params/buffers
    move_nf4_to_device(dit, "cpu")   # _nf4_packed / _nf4_state — plain attrs .to() can't see

The restore had them as `if nf4: ... elif swap: ... else: dit.to(device)` — mutually
exclusive — so on a 4-bit run only the packed weights returned and every ordinary
parameter stayed on CPU. compute_loss read its device from the first parameter it found,
sent the whole batch to CPU, and the first CUDA-resident tensor it met raised.

The two calls are complementary, never alternatives. That is what this pins down.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
import torch.nn as nn

from fizgig.modules.nf4 import move_nf4_to_device

DEV = "cuda"


class FakeNF4Linear(nn.Module):
    """Mimics the storage split: an ordinary bias .to() moves, plus packed data it cannot see."""

    def __init__(self, n=8):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n))                     # .to() moves this
        self._is_nf4 = True
        self._nf4_packed = torch.zeros(n, n // 2, dtype=torch.uint8)  # .to() cannot see this
        self._nf4_state = None


class FakeDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(8, 8)   # the module that actually stranded in the report
        self.blocks = nn.ModuleList([FakeNF4Linear() for _ in range(3)])
        self._nf4_quantized = True


def fresh():
    dit = FakeDiT().to(DEV)
    move_nf4_to_device(dit, DEV)
    return dit


def park(dit):
    dit.to("cpu")
    move_nf4_to_device(dit, "cpu")


def devices(dit):
    ordinary = {p.device.type for p in dit.parameters()}
    packed = {m._nf4_packed.device.type for m in dit.modules()
              if getattr(m, "_is_nf4", False)}
    return ordinary, packed


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  -> ' + detail) if detail and not cond else ''}")
    return bool(cond)


def main():
    if not torch.cuda.is_available():
        print("no CUDA available — skipped")
        return 0
    ok = True
    print("NF4 park / restore")

    # 1. parking moves both halves
    dit = fresh()
    park(dit)
    o, p = devices(dit)
    ok &= check("park moves ordinary params to CPU", o == {"cpu"}, str(o))
    ok &= check("park moves packed weights to CPU", p == {"cpu"}, str(p))

    # 2. the bug — restoring only the NF4 half strands everything else
    dit = fresh()
    park(dit)
    move_nf4_to_device(dit, DEV)                    # the old, exclusive restore
    o, p = devices(dit)
    ok &= check("packed-only restore returns the packed weights", p == {"cuda"}, str(p))
    ok &= check("packed-only restore STRANDS ordinary params (the bug)", o == {"cpu"},
                f"{o} — if this fails, move_nf4_to_device now moves them too "
                "and the trainer fix can be simplified")

    # 3. the fix — placement first, then the packed weights
    dit = fresh()
    park(dit)
    dit.to(DEV)                                     # the call the NF4 branch used to skip
    move_nf4_to_device(dit, DEV)
    o, p = devices(dit)
    ok &= check("both-halves restore returns ordinary params", o == {"cuda"}, str(o))
    ok &= check("both-halves restore returns packed weights", p == {"cuda"}, str(p))

    # 4. the production restore. The native Krea 2 trainer that carried issue #17's three hand-written
    # park/restore pairs was deleted when every family moved to the driver system. The driver folded
    # the fix into ONE function, fizgig.families.quant.move: dit.to(device) and then the packed
    # weights, unconditionally. So test that function directly, with the same split model as above,
    # rather than grepping a file that no longer exists.
    print("driver restore (fizgig.families.quant.move)")
    from fizgig.families import quant
    dit = fresh()
    park(dit)
    quant.move(dit, DEV)
    o, p = devices(dit)
    ok &= check("quant.move restores ordinary params", o == {"cuda"}, str(o))
    ok &= check("quant.move restores packed weights", p == {"cuda"}, str(p))
    dit = fresh()
    quant.move(dit, "cpu")
    o, p = devices(dit)
    ok &= check("quant.move parks BOTH halves to CPU", o == {"cpu"} and p == {"cpu"}, f"{o} / {p}")

    # 5. static guards over the driver, so the #17 shape cannot come back by another route.
    print("driver park/restore sites")
    fam = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "fizgig", "families")
    srcs = {}
    for name in sorted(os.listdir(fam)):
        if name.endswith(".py"):
            with open(os.path.join(fam, name), encoding="utf-8") as f:
                srcs[name] = f.read()
    n_park = sum(t.count('quant.move(dit, "cpu")') for t in srcs.values())
    n_restore = sum(t.count("quant.move(dit, device)") for t in srcs.values())
    ok &= check("every driver NF4 park has a matching restore", n_park == n_restore and n_park > 0,
                f"{n_park} park(s), {n_restore} restore(s)")
    # The bug was a restore that moved only ONE half. quant.move moves both, so the way to
    # reintroduce it is to call move_nf4_to_device on a whole model directly. Only quant.py may.
    bypass = [n for n, t in srcs.items() if n != "quant.py" and "move_nf4_to_device(dit" in t]
    ok &= check("nothing outside quant.move restores only the packed half", not bypass, str(bypass))
    # Retired: "compute_loss takes an explicit device". That guarded the deleted native trainer's
    # compute_loss, which read its device from the first parameter it found -- a stranded one, in
    # #17. The function was removed with that trainer and the driver has no compute_loss, so there
    # is nothing left to pin. Kept as a note rather than a passing check that tests nothing.

    print()
    print("all passed" if ok else "FAILURES — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
