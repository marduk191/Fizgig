# Fizgig v4.3.2

## MiniMax H3 trains faster — a fused Triton kernel for the int8 base

**@rintic-13**'s fused W8A16 GEMM kernel now powers the frozen int8 base's forward passes
during training — measured **1.14–1.42× faster per matmul** on real checkpoint weights,
which typically lands as a noticeably quicker training step. It's on automatically; nothing
to configure (`FIZGIG_NO_TRITON_W8A16=1` switches it off if ever needed).

Quality was validated the hard way: 40-epoch training runs judged blind in ComfyUI against
the standard path came out at parity. The kernel is actually slightly *more* accurate than
the standard path — it rounds once where the old code rounded twice.

This is @rintic-13's third performance contribution — the int8 block streaming that
headlines v4.0.0, the 16 GB text-encoder streaming in v4.3.0, and now this.
([#89](https://github.com/shootthesound/Fizgig/issues/89))
