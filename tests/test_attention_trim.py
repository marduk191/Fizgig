"""Sequence trimming must behave exactly as it did when the decision lived inside attention().

The trim decision moved into AttentionParams.__post_init__ because it reads a CUDA tensor on the
CPU, and running it per block meant a device sync in every one of the 28 blocks (56 with gradient
checkpointing) plus a graph break for torch.compile. This checks the behaviour is unchanged:
uniform sequence lengths trim, ragged ones do not, and split-attn never trims.
"""
import sys

sys.path.insert(0, r"W:\Peter\Documents\Development\Fizgig\src")
import torch

from fizgig.krea2.attention import AttentionParams as K2Params, attention as k2_attention
from fizgig.modules.attention import AttentionParams as KleinParams, attention as klein_attention

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, L, H, D, IMG = 2, 16, 8, 64, 32

for name, Params, attention, from_mask in (
        ("krea2", K2Params, k2_attention, "create_attention_params_from_mask"),
        ("klein", KleinParams, klein_attention, "create_from_mask")):
    mk = getattr(Params, from_mask)

    uniform = mk("torch", False, IMG, torch.ones(B, L, device=DEV))
    # Krea 2 rounds the trim length up to a multiple so shape count stays small; Klein trims
    # exactly. L + IMG is 48 here, which rounds to 64 under the x64 default.
    assert uniform.uniform_seqlen >= L + IMG, uniform.uniform_seqlen

    ragged = mk("torch", False, IMG, torch.tensor([[1.] * L, [1.] * 8 + [0.] * 8], device=DEV))
    assert ragged.uniform_seqlen is None, ragged.uniform_seqlen

    split = mk("torch", True, IMG, torch.ones(B, L, device=DEV))
    assert split.uniform_seqlen is None, "split-attn has its own per-sequence path"

    # flash/sageattn take the mask natively and must not be trimmed out from under them.
    assert mk("flash", False, IMG, torch.ones(B, L, device=DEV)).uniform_seqlen is None

    # A full-length mask trims to the full length, so the result must match the no-mask path.
    q = torch.randn(B, L + IMG, H, D, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        trimmed = attention(q, q, q, attn_params=uniform)
        plain = attention(q, q, q, attn_params=Params.create_attention_params("torch", False)
                          if name == "krea2" else Params.create("torch", False))
    assert trimmed.shape == plain.shape, (trimmed.shape, plain.shape)
    assert torch.allclose(trimmed.float(), plain.float(), atol=1e-2)
    print(f"  {name}: uniform trims to {uniform.uniform_seqlen}, ragged/split/flash do not, "
          f"output matches the untrimmed path")


# --- rounding: fewer distinct shapes, identical numbers on the valid tokens -------------------
import fizgig.krea2.attention as K2  # noqa: E402

IMGLEN, TXT = 900, 200
q = torch.randn(1, IMGLEN + TXT, H, D, device=DEV, dtype=torch.bfloat16)
mask = torch.zeros(1, TXT, device=DEV)
mask[:, :174] = 1                                    # 1074 valid tokens of a 1100 window

rounded_p = K2Params.create_attention_params_from_mask("torch", False, IMGLEN, mask)
assert rounded_p.uniform_seqlen == 1088 and not rounded_p.uniform_exact, rounded_p.uniform_seqlen

K2._TRIM_MULTIPLE = 1                                # exact trim, the pre-fix behaviour
exact_p = K2Params.create_attention_params_from_mask("torch", False, IMGLEN, mask)
assert exact_p.uniform_seqlen == 1074 and exact_p.uniform_exact
K2._TRIM_MULTIPLE = 64

with torch.no_grad():
    a = k2_attention(q, q, q, attn_params=rounded_p)
    b = k2_attention(q, q, q, attn_params=exact_p)
n = exact_p.uniform_seqlen
assert torch.equal(a[:, :n], b[:, :n]), "rounding must not change the valid tokens"
print(f"  rounding: 1074 -> 1088 (masked slack), valid tokens bit-identical to the exact trim")

# --- auto backend: cuDNN only once it pays for itself ------------------------------------------
from fizgig.modules import sdpa  # noqa: E402

sdpa._seen_shapes = {1088, 1024, 768}
sdpa._training_cudnn = False
assert sdpa.consider_training_backend(100) is None, "short run must stay on the default backend"
got = sdpa.consider_training_backend(3 * 35 * 2)
assert got == (3, 210), got
assert sdpa._training_cudnn
assert sdpa.consider_training_backend(10**6) is None, "already switched — must not re-fire"
sdpa._seen_shapes, sdpa._training_cudnn = set(), False
print("  auto backend: 3 shapes -> needs 210 steps; declines below, switches above, fires once")
