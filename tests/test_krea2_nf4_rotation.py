"""Krea 2 rotation over an NF4 frozen trunk (GPU-gated; skips clean without CUDA).

The failure modes here are silent. A wrong re-encode still trains and still saves — it
just quietly degrades the frozen context, or worse compounds every cycle so a long run
rots from the inside. Pins:

  * an NF4 Linear is discovered as rotatable (an fp8-only predicate would build an EMPTY
    master and the run would train nothing while looking healthy);
  * activate: weight becomes trainable bf16 taken FROM THE MASTER, the packed 4-bit copy
    is released, and the patched forward is replaced by the plain Linear one;
  * deactivate: master gets the trained value EXACTLY (bf16, no quantizer), the module
    goes back to NF4 storage with its dequant forward, and the orphaned bf16 storage is
    actually freed;
  * forward parity: after deactivate the module computes what a fresh NF4 quantization of
    the trained weight computes — i.e. the re-encode is the real thing, not a stale copy;
  * NO COMPOUNDING: ten activate/deactivate cycles with no training leave the master
    bit-identical, because activate reads the master rather than the residency copy. This
    is why the residency re-encode can round to nearest — H3 needed stochastic rounding
    for its lossy int8 SAVE, which Krea 2 does not have (it saves bf16 from the master).

Run (needs CUDA): venv/Scripts/python.exe tests/test_krea2_nf4_rotation.py
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

if not torch.cuda.is_available():
    print("SKIP  no CUDA device — NF4 needs bitsandbytes on a GPU")
    sys.exit(0)

import torch.nn as nn  # noqa: E402
from fizgig.modules.nf4 import apply_nf4_quantization  # noqa: E402
from fizgig.krea2.rotation import BlockRotator, is_rotatable_linear  # noqa: E402

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


torch.manual_seed(11)
DIM = 256


class Blk(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(DIM, DIM, bias=False)


blocks = nn.ModuleList([Blk() for _ in range(2)])
truth = {f"blocks.{i}.attn.qkv.weight": blocks[i].attn.qkv.weight.data.clone().to(torch.bfloat16)
         for i in range(2)}
blocks.to("cuda")

n = apply_nf4_quantization(blocks, target_keys=("attn",), compute_device=torch.device("cuda"))
ck("NF4 quantization applied to the test blocks", n == 2, n)
lin0 = blocks[0].attn.qkv
ck("an NF4 Linear is seen as rotatable", is_rotatable_linear(lin0))
ck("...and its weight really is emptied by NF4", lin0.weight.numel() == 0)

master = {k: v.clone() for k, v in truth.items()}
rot = BlockRotator(blocks, master, key_prefix="blocks", device="cuda")

# --- activate ---------------------------------------------------------------------------
rot.activate([0])
ck("activate: weight is trainable bf16 of the right shape",
   lin0.weight.requires_grad and lin0.weight.dtype == torch.bfloat16
   and tuple(lin0.weight.shape) == (DIM, DIM))
ck("activate: the value came FROM THE MASTER (bit-exact)",
   torch.equal(lin0.weight.detach().cpu(), master["blocks.0.attn.qkv.weight"]))
ck("activate: the packed 4-bit copy was released",
   getattr(lin0, "_nf4_packed", None) is None)
ck("activate: the patched forward is gone (plain nn.Linear runs)",
   "forward" not in lin0.__dict__)
ck("activate: the untouched block stayed NF4",
   blocks[1].attn.qkv.weight.numel() == 0)

# --- train it a little, then deactivate ---------------------------------------------------
with torch.no_grad():
    lin0.weight.add_(0.05)
trained = lin0.weight.detach().clone()
weight_storage = lin0.weight.detach().untyped_storage()

rot.deactivate([0])
ck("deactivate: master holds the TRAINED value exactly (no quantizer on this path)",
   torch.equal(master["blocks.0.attn.qkv.weight"], trained.cpu().to(torch.bfloat16)))
ck("deactivate: module is back on NF4 storage",
   getattr(lin0, "_nf4_packed", None) is not None and lin0.weight.numel() == 0)
ck("deactivate: the dequant forward is bound again", "forward" in lin0.__dict__)
ck("deactivate: the orphaned bf16 storage was actually freed",
   weight_storage.size() == 0, weight_storage.size())

# --- forward parity: the re-encode is a real quantization of the trained weight ----------
from bitsandbytes.functional import quantize_nf4, dequantize_nf4  # noqa: E402

ref_packed, ref_state = quantize_nf4(trained.contiguous(), compress_statistics=False)
ref = dequantize_nf4(ref_packed, ref_state).float()
got = dequantize_nf4(lin0._nf4_packed, lin0._nf4_state).float()
ck("deactivate: residency dequant matches a fresh quantization of the trained weight",
   torch.equal(got, ref))

x = torch.randn(4, DIM, device="cuda", dtype=torch.bfloat16)
ck("the module's forward runs through the NF4 path and matches that dequant",
   torch.equal(lin0(x), torch.nn.functional.linear(x, got.to(torch.bfloat16), None)))

# --- NO COMPOUNDING over repeated cycles -------------------------------------------------
# This is the property that makes nearest rounding correct for the residency re-encode:
# activate() reads the MASTER, never the quantized copy, so error cannot accumulate.
before = master["blocks.0.attn.qkv.weight"].clone()
for _ in range(10):
    rot.activate([0])
    rot.deactivate([0])
ck("10 activate/deactivate cycles leave the master BIT-IDENTICAL (no drift)",
   torch.equal(master["blocks.0.attn.qkv.weight"], before))

print()
if fails:
    print(f"FAILED: {len(fails)} pin(s):")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL PASS")
