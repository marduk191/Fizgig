import os
from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from tqdm import tqdm

from fizgig.utils.safetensors import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from fizgig.utils.device import clean_memory_on_device


def calculate_fp8_maxval(exp_bits=4, mantissa_bits=3, sign_bits=1):
    """
    Calculate the maximum representable value in FP8 format.
    Default is E4M3 format (4-bit exponent, 3-bit mantissa, 1-bit sign). Only supports E4M3 and E5M2 with sign bit.

    Args:
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        sign_bits (int): Number of sign bits (0 or 1)

    Returns:
        float: Maximum value representable in FP8 format
    """
    assert exp_bits + mantissa_bits + sign_bits == 8, "Total bits must be 8"
    if exp_bits == 4 and mantissa_bits == 3 and sign_bits == 1:
        return torch.finfo(torch.float8_e4m3fn).max
    elif exp_bits == 5 and mantissa_bits == 2 and sign_bits == 1:
        return torch.finfo(torch.float8_e5m2).max
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits} with sign_bits={sign_bits}")


def quantize_fp8(tensor, scale, fp8_dtype, max_value, min_value):
    """
    Quantize a tensor to FP8 format using PyTorch's native FP8 dtype support.

    Args:
        tensor (torch.Tensor): Tensor to quantize
        scale (float or torch.Tensor): Scale factor
        fp8_dtype (torch.dtype): Target FP8 dtype (torch.float8_e4m3fn or torch.float8_e5m2)
        max_value (float): Maximum representable value in FP8
        min_value (float): Minimum representable value in FP8

    Returns:
        torch.Tensor: Quantized tensor in FP8 format
    """
    tensor = tensor.to(torch.float32)  # ensure tensor is in float32 for division

    # Create scaled tensor
    tensor = torch.div(tensor, scale).nan_to_num_(0.0)  # handle NaN values, equivalent to nonzero_mask in previous function

    # Clamp tensor to range
    tensor = tensor.clamp_(min=min_value, max=max_value)

    # Convert to FP8 dtype
    tensor = tensor.to(fp8_dtype)

    return tensor


def optimize_state_dict_with_fp8(
    state_dict: dict,
    calc_device: Union[str, torch.device],
    target_layer_keys: Optional[list[str]] = None,
    exclude_layer_keys: Optional[list[str]] = None,
    exp_bits: int = 4,
    mantissa_bits: int = 3,
    move_to_device: bool = False,
    quantization_mode: str = "block",
    block_size: Optional[int] = 64,
):
    """
    Optimize Linear layer weights in a model's state dict to FP8 format. The state dict is modified in-place.
    This function is a static version of load_safetensors_with_fp8_optimization without loading from files.

    Args:
        state_dict (dict): State dict to optimize, replaced in-place
        calc_device (str): Device to quantize tensors on
        target_layer_keys (list, optional): Layer key patterns to target (None for all Linear layers)
        exclude_layer_keys (list, optional): Layer key patterns to exclude
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        move_to_device (bool): Move optimized tensors to the calculating device

    Returns:
        dict: FP8 optimized state dict
    """
    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits}")

    # Calculate FP8 max value
    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value  # this function supports only signed FP8

    # Create optimized state dict
    optimized_count = 0

    # Enumerate tarket keys
    target_state_dict_keys = []
    for key in state_dict.keys():
        # Check if it's a weight key and matches target patterns
        is_target = (target_layer_keys is None or any(pattern in key for pattern in target_layer_keys)) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(pattern in key for pattern in exclude_layer_keys)
        is_target = is_target and not is_excluded

        if is_target and isinstance(state_dict[key], torch.Tensor):
            target_state_dict_keys.append(key)

    # Process each key
    for key in tqdm(target_state_dict_keys):
        value = state_dict[key]

        # Save original device and dtype
        original_device = value.device
        original_dtype = value.dtype

        # Move to calculation device
        if calc_device is not None:
            value = value.to(calc_device)

        quantized_weight, scale_tensor = quantize_weight(key, value, fp8_dtype, max_value, min_value, quantization_mode, block_size)

        # Add to state dict using original key for weight and new key for scale
        fp8_key = key  # Maintain original key
        scale_key = key.replace(".weight", ".scale_weight")

        if not move_to_device:
            quantized_weight = quantized_weight.to(original_device)

        # keep scale shape: [1] or [out,1] or [out, num_blocks, 1]. We can determine the quantization mode from the shape of scale_weight in the patched model.
        scale_tensor = scale_tensor.to(dtype=original_dtype, device=quantized_weight.device)

        state_dict[fp8_key] = quantized_weight
        state_dict[scale_key] = scale_tensor

        optimized_count += 1

        if calc_device is not None:  # optimized_count % 10 == 0 and
            # free memory on calculation device
            clean_memory_on_device(calc_device)

    logger.info(f"Number of optimized Linear layers: {optimized_count}")
    return state_dict


def quantize_weight(
    key: str,
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    max_value: float,
    min_value: float,
    quantization_mode: str = "block",
    block_size: int = 64,
):
    original_shape = tensor.shape

    # Determine quantization mode
    if quantization_mode == "block":
        if tensor.ndim != 2:
            quantization_mode = "tensor"  # fallback to per-tensor
        else:
            out_features, in_features = tensor.shape
            if in_features % block_size != 0:
                quantization_mode = "channel"  # fallback to per-channel
                logger.warning(
                    f"Layer {key} with shape {tensor.shape} is not divisible by block_size {block_size}, fallback to per-channel quantization."
                )
            else:
                num_blocks = in_features // block_size
                tensor = tensor.contiguous().view(out_features, num_blocks, block_size)  # [out, num_blocks, block_size]
    elif quantization_mode == "channel":
        if tensor.ndim != 2:
            quantization_mode = "tensor"  # fallback to per-tensor

    # Calculate scale factor (per-tensor or per-output-channel with percentile or max)
    # value shape is expected to be [out_features, in_features] for Linear weights
    if quantization_mode == "channel" or quantization_mode == "block":
        # row-wise percentile to avoid being dominated by outliers
        # result shape: [out_features, 1] or [out_features, num_blocks, 1]
        scale_dim = 1 if quantization_mode == "channel" else 2
        abs_w = torch.abs(tensor)

        # shape: [out_features, 1] or [out_features, num_blocks, 1]
        row_max = torch.max(abs_w, dim=scale_dim, keepdim=True).values
        scale = row_max / max_value

    else:
        # per-tensor
        tensor_max = torch.max(torch.abs(tensor).view(-1))
        scale = tensor_max / max_value

    # numerical safety
    scale = torch.clamp(scale, min=1e-8)
    scale = scale.to(torch.float32)  # ensure scale is in float32 for division

    # Quantize weight to FP8 (scale can be scalar or [out,1], broadcasting works)
    quantized_weight = quantize_fp8(tensor, scale, fp8_dtype, max_value, min_value)

    # If block-wise, restore original shape
    if quantization_mode == "block":
        quantized_weight = quantized_weight.view(original_shape)  # restore to original shape [out, in]

    return quantized_weight, scale


def load_safetensors_with_fp8_optimization(
    model_files: List[str],
    calc_device: Union[str, torch.device],
    target_layer_keys=None,
    exclude_layer_keys=None,
    exp_bits=4,
    mantissa_bits=3,
    move_to_device=False,
    weight_hook=None,
    quantization_mode: str = "block",
    block_size: Optional[int] = 64,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict:
    """
    Load weight tensors from safetensors files and merge LoRA weights into the state dict with explicit FP8 optimization.

    Args:
        model_files (list[str]): List of model files to load
        calc_device (str or torch.device): Device to quantize tensors on
        target_layer_keys (list, optional): Layer key patterns to target for optimization (None for all Linear layers)
        exclude_layer_keys (list, optional): Layer key patterns to exclude from optimization
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        move_to_device (bool): Move optimized tensors to the calculating device
        weight_hook (callable, optional): Function to apply to each weight tensor before optimization
        quantization_mode (str): Quantization mode, "tensor", "channel", or "block"
        block_size (int, optional): Block size for block-wise quantization (used if quantization_mode is "block")
        disable_numpy_memmap (bool): Disable numpy memmap when loading safetensors
        weight_transform_hooks (WeightTransformHooks, optional): Hooks for weight transformation during loading

    Returns:
        dict: FP8 optimized state dict
    """
    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits}")

    # Calculate FP8 max value
    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value  # this function supports only signed FP8

    # Define function to determine if a key is a target key. target means fp8 optimization, not for weight hook.
    def is_target_key(key):
        # Check if weight key matches target patterns and does not match exclude patterns
        is_target = (target_layer_keys is None or any(pattern in key for pattern in target_layer_keys)) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(pattern in key for pattern in exclude_layer_keys)
        return is_target and not is_excluded

    # Create optimized state dict
    optimized_count = 0

    # Process each file
    state_dict = {}
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
            f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f

            keys = f.keys()
            for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                value = f.get_tensor(key)

                # Save original device
                original_device = value.device  # usually cpu

                if weight_hook is not None:
                    # Apply weight hook if provided
                    value = weight_hook(key, value, keep_on_calc_device=(calc_device is not None))

                if not is_target_key(key):
                    target_device = calc_device if (calc_device is not None and move_to_device) else original_device
                    value = value.to(target_device)
                    state_dict[key] = value
                    continue

                # Move to calculation device
                if calc_device is not None:
                    value = value.to(calc_device)

                original_dtype = value.dtype
                if original_dtype.itemsize == 1:
                    raise ValueError(
                        f"Layer {key} is already in {original_dtype} format. `--fp8_scaled` optimization should not be applied. Please use fp16/bf16/float32 model weights."
                        + f" / レイヤー {key} は既に{original_dtype}形式です。`--fp8_scaled` 最適化は適用できません。FP16/BF16/Float32のモデル重みを使用してください。"
                    )
                quantized_weight, scale_tensor = quantize_weight(
                    key, value, fp8_dtype, max_value, min_value, quantization_mode, block_size
                )

                # Add to state dict using original key for weight and new key for scale
                fp8_key = key  # Maintain original key
                scale_key = key.replace(".weight", ".scale_weight")
                assert fp8_key != scale_key, "FP8 key and scale key must be different"

                if not move_to_device:
                    quantized_weight = quantized_weight.to(original_device)

                # keep scale shape: [1] or [out,1] or [out, num_blocks, 1]. We can determine the quantization mode from the shape of scale_weight in the patched model.
                scale_tensor = scale_tensor.to(dtype=original_dtype, device=quantized_weight.device)

                state_dict[fp8_key] = quantized_weight
                state_dict[scale_key] = scale_tensor

                optimized_count += 1

                if calc_device is not None and optimized_count % 10 == 0:
                    # free memory on calculation device
                    clean_memory_on_device(calc_device)

    logger.info(f"Number of optimized Linear layers: {optimized_count}")
    return state_dict


# region FP8 scaled_mm training path
#
# When training a LoRA on an fp8 base, the frozen-base Linears dominate compute.
# Running them in fp8 via torch._scaled_mm (forward AND backward) is ~1.3-1.5x
# faster than dequantize-to-bf16 on the GEMMs, biggest at low (face-crop)
# resolutions where attention is cheap. Recipe (OneTrainer-style): per-token
# activation quant + POST-matmul scale (sidesteps Blackwell's rowwise-scale
# limitation). The base weight is frozen, so the custom autograd Function only
# returns grad w.r.t. the input (to chain back to earlier LoRA modules).

_FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
_TRAIN_SCALED_MM_SUPPORTED = None      # lazily resolved hardware capability
_ONE_BY_DEVICE = {}                    # cached scalar 1.0 per device
# Diagnostic: logs each fp8 Linear's path + state and flags any divergence
# between a Linear's calls (forward vs checkpoint recompute). Off by default;
# set FIZGIG_FP8_DIAG=1 to enable (very verbose — one line per fp8 Linear).
_FP8_DIAG = os.environ.get("FIZGIG_FP8_DIAG", "0") != "0"
_fp8_diag_last = {}
_scaled_mm_probe_warned = False


def _train_scaled_mm_supported() -> bool:
    global _TRAIN_SCALED_MM_SUPPORTED
    if _TRAIN_SCALED_MM_SUPPORTED is None:
        from fizgig.utils.device import fp8_scaled_mm_supported
        _TRAIN_SCALED_MM_SUPPORTED = fp8_scaled_mm_supported()
    return _TRAIN_SCALED_MM_SUPPORTED


def _one(device):
    o = _ONE_BY_DEVICE.get(device)
    if o is None:
        o = torch.tensor(1.0, device=device, dtype=torch.float32)
        _ONE_BY_DEVICE[device] = o
    return o


def _per_token_quant(t: torch.Tensor):
    """Per-row (per-token) fp8 quant. Returns (fp8 tensor, per-row scale (M,1) fp32)."""
    sc = (t.detach().abs().amax(dim=-1, keepdim=True) / _FP8_E4M3_MAX).clamp_(min=1e-8).to(torch.float32)
    q = (t / sc).clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q, sc


class _FP8ScaledMMLinear(torch.autograd.Function):
    """fp8 matmul for a FROZEN base Linear (weight requires no grad).

    forward:  y = (x_fp8 @ Wᵀ_fp8) · w_scale · x_token_scale
    backward: grad_x = (gradᵀ_fp8 @ W_fp8) · w_scale · grad_token_scale
    Only grad w.r.t. x is returned (weight/scale are frozen → None).
    """
    @staticmethod
    def forward(ctx, x, weight_fp8, scale_weight):
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        xq, xs = _per_token_quant(x2)
        sw = scale_weight.to(torch.float32).reshape(())
        w_fwd = weight_fp8.t()  # (K,N) col-major view — free
        out = torch._scaled_mm(xq, w_fwd, scale_a=_one(x.device), scale_b=sw, out_dtype=torch.bfloat16)
        out = out * xs.to(out.dtype)
        # The weight + scale are FROZEN module attributes that need no gradient, so
        # we stash them on ctx directly instead of save_for_backward. save_for_backward
        # registers tensors into autograd's saved-tensor machinery, which under
        # gradient checkpointing (use_reentrant=False) must line up exactly between
        # the original forward and the recompute — the extra scalar scale tensor
        # shifted that list and triggered a CheckpointError. Stashing on ctx keeps
        # these frozen tensors out of that bookkeeping entirely.
        ctx.weight_fp8 = weight_fp8
        ctx.scale_weight = scale_weight
        ctx.orig_shape = orig_shape
        return out.reshape(*orig_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_out):
        weight_fp8, scale_weight = ctx.weight_fp8, ctx.scale_weight
        go = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        gq, gs = _per_token_quant(go)
        sw = scale_weight.to(torch.float32).reshape(())
        # grad_x = grad_out @ W ; W is (N,K) → need (N,K) col-major (transpose-copy)
        w_bwd = weight_fp8.t().contiguous().t()
        gx = torch._scaled_mm(gq, w_bwd, scale_a=_one(grad_out.device), scale_b=sw, out_dtype=torch.bfloat16)
        gx = gx * gs.to(gx.dtype)
        return gx.reshape(*ctx.orig_shape[:-1], -1), None, None


def _dequant_fp8_weight(weight_fp8, scale, device):
    """fp8 -> bf16 dequant, identical math to the inference dequant path.
    Handles per-tensor (scale.ndim<3) and block-wise (ndim==3) scales, and a
    weight that may be CPU-resident under block swap (moved to `device`)."""
    weight = weight_fp8.to(device)
    scale = scale.to(device)
    out_dtype = scale.dtype
    if scale.ndim < 3:
        return weight.to(out_dtype) * scale
    out_features, num_blocks, _ = scale.shape
    w = weight.to(out_dtype).contiguous().view(out_features, num_blocks, -1)
    w = w * scale
    return w.view(weight.shape)


class _FP8DequantLinear(torch.autograd.Function):
    """Memory-frugal dequant matmul for a FROZEN fp8 base Linear.

    forward:  dequant fp8->bf16 (LOCAL — freed on return), y = x @ Wᵀ
    backward: RECOMPUTE the dequant from fp8, grad_x = grad_out @ W. Only grad_x
              is returned (weight + scale are frozen → None).

    The point: a plain `F.linear(x, dequant)` makes autograd save the bf16
    dequant for the backward grad_x. With gradient checkpointing OFF that keeps
    all ~112 dequantized base weights (~18 GB) alive until backward, stacked on
    the ~9 GB fp8 residency — which OOMs even a 32 GB card. Here the bf16 weight
    never enters autograd's saved-tensor machinery; it's rebuilt from fp8 in the
    backward (cheap, bandwidth-bound) instead. fp8 weight + scale are frozen and
    stashed on ctx (not save_for_backward), the same checkpoint-safe pattern as
    _FP8ScaledMMLinear. Numerically identical to the plain dequant path."""
    @staticmethod
    def forward(ctx, x, weight_fp8, scale_weight):
        w = _dequant_fp8_weight(weight_fp8, scale_weight, x.device)
        out = F.linear(x, w)
        ctx.weight_fp8 = weight_fp8
        ctx.scale_weight = scale_weight
        return out

    @staticmethod
    def backward(ctx, grad_out):
        w = _dequant_fp8_weight(ctx.weight_fp8, ctx.scale_weight, grad_out.device)
        grad_x = grad_out.to(w.dtype) @ w  # F.linear(x,W) -> grad_x = grad_out @ W
        return grad_x.to(grad_out.dtype), None, None


def _try_fp8_scaled_mm_train(self: nn.Linear, x):
    """Return the fp8 scaled_mm result for a training step, or None to fall back.

    Runs whenever a backward through this frozen Linear is needed:
      - GC-on training: self.training is True (train() mode).
      - GC-off training: self.training is False (the trainer uses eval() when
        gradient checkpointing is off) but grad IS enabled.
    NOT inference/preview, which is eval()+no_grad → bit-identical dequant path.

    Why self.training is still the PRIMARY signal (not torch.is_grad_enabled):
    under gradient checkpointing the grad mode flips between the first forward and
    the recompute, so keying on it there would diverge. self.training is True in
    both GC-on passes, so the path is consistent. is_grad_enabled is only
    consulted when self.training is False — i.e. GC-off or inference — and in BOTH
    of those there is no checkpoint, so the grad mode is stable and safe to read.

    CRITICAL — the fp8-vs-dequant decision is PROBED ONCE per module and cached on
    self._fp8_train_decision. A per-call try/except is unsafe under gradient
    checkpointing: torch._scaled_mm can succeed on the first forward but raise on
    the recompute (or vice versa), so the two passes take different code paths ->
    save different tensors -> "recomputed values have different metadata"
    CheckpointError (the crash this fixes). Caching forces both passes to take the
    SAME path. If the probe fails, we cache dequant for that Linear permanently
    and log the real error — so the run completes and we can see why scaled_mm
    failed.

    Stable preconditions: fp8 weight with per-tensor scale, fp8 tensor cores
    (SM 8.9+), 16-aligned matmul dims."""
    def _diag(path, M=None):
        # FIZGIG_FP8_DIAG=1 only. Logs per-fp8-Linear decisions and flags the
        # instant a Linear's path flips between calls (forward vs checkpoint
        # recompute) — the divergence that causes the CheckpointError.
        if _FP8_DIAG and self.weight.dtype == torch.float8_e4m3fn:
            prev = _fp8_diag_last.get(id(self))
            tag = "   <<<<< DIVERGENCE" if (prev is not None and prev != path) else ""
            logger.warning(
                f"[fp8diag] {path:<8} train={self.training} grad={torch.is_grad_enabled()} "
                f"cached={getattr(self, '_fp8_train_decision', None)} M={M} "
                f"wshape={tuple(self.weight.shape)}{tag}")
            _fp8_diag_last[id(self)] = path

    if not (_train_scaled_mm_supported() and (self.training or torch.is_grad_enabled())):
        _diag("DEQUANT")
        return None
    decision = getattr(self, "_fp8_train_decision", None)
    if decision == "dequant":
        _diag("DEQUANT")
        return None
    if self.weight.dtype != torch.float8_e4m3fn:
        return None
    sw = getattr(self, "scale_weight", None)
    if sw is None or sw.numel() != 1:   # per-tensor scale only (not block-wise)
        _diag("DEQUANT")
        return None
    if sw.device != self.weight.device:
        # Block swap moves parameters but not buffers, so a swapped-in Linear can
        # have its weight on GPU with scale_weight still on CPU — _scaled_mm
        # rejects mixed devices. Persist the move so it's one copy, not one per step.
        sw = sw.to(self.weight.device)
        self.scale_weight = sw
    K = x.shape[-1]
    N = self.weight.shape[0]
    M = x.numel() // K if K else 0
    # Only K and N must be 16-aligned (_scaled_mm constrains mat2) — M is free. The old
    # `M % 16` guard sent every ragged-token-count Linear down the dequant path, which is
    # SLOWER than bf16; relaxing it measured 1.89x at M=4173. M==0 stays (empty input).
    if K % 16 or N % 16 or M == 0:
        _diag("DEQUANT", M)
        return None  # alignment is input-dependent — never cached

    if decision == "scaled_mm":
        # Same RuntimeError insurance as the probe below: the cached decision was made at
        # one M, and M varies per call now that it isn't part of the guard.
        try:
            out = _FP8ScaledMMLinear.apply(x, self.weight, sw)
        except RuntimeError:
            self._fp8_train_decision = "dequant"
            _diag("DEQUANT", M)
            return None
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        _diag("SCALED", M)
        return out

    # decision is None: probe once. Catch only RuntimeError (a real _scaled_mm
    # failure) — NOT a bare Exception, which would swallow checkpoint's private
    # _StopRecomputationError control-flow signal and corrupt its state.
    try:
        out = _FP8ScaledMMLinear.apply(x, self.weight, sw)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        self._fp8_train_decision = "scaled_mm"
        _diag("SCALED", M)
        return out
    except RuntimeError as e:
        self._fp8_train_decision = "dequant"
        global _scaled_mm_probe_warned
        if not _scaled_mm_probe_warned:
            _scaled_mm_probe_warned = True
            logger.warning(
                f"fp8 scaled_mm disabled ({type(e).__name__}: {e}); using dequant. "
                "Further Linears that fall back will do so silently (harmless — dequant is bit-identical, just misses the fp8 GEMM speedup).")
        else:
            logger.debug(f"fp8 scaled_mm disabled for this Linear ({type(e).__name__}: {e}); using dequant.")
        _diag("DEQUANT", M)
        return None

# endregion


def fp8_linear_forward_patch(self: nn.Linear, x, use_scaled_mm=False, max_value=None):
    """
    Patched forward method for Linear layers with FP8 weights.

    Args:
        self: Linear layer instance
        x (torch.Tensor): Input tensor
        use_scaled_mm (bool): Use scaled_mm for FP8 Linear layers, requires SM 8.9+ (RTX 40 series)
        max_value (float): Maximum value for FP8 quantization. If None, no quantization is applied for input tensor.

    Returns:
        torch.Tensor: Result of linear transformation
    """
    # Training on fp8 base: run the frozen-base GEMM in fp8 (fwd+bwd) for ~1.3-1.5x
    # over dequant. Inference (no_grad) skips this and uses the bit-identical
    # dequant path below. Returns None (falls through) when not applicable.
    _train_out = _try_fp8_scaled_mm_train(self, x)
    if _train_out is not None:
        return _train_out

    if use_scaled_mm:
        # FP8 tensor-core matmul via torch._scaled_mm (Ada/Hopper/Blackwell,
        # SM 8.9+). Both operands run in fp8; the GEMM is ~2x bf16. Per-tensor
        # scales only (Blackwell rejects rowwise). The activation scale comes
        # from the file's pre-calibrated `input_scale` (registered as
        # self.scale_input) when available, else a cheap dynamic per-tensor
        # amax. On ANY failure we fall through to the dequant path below so a
        # bad shape / odd dim never breaks a run.
        try:
            input_dtype = x.dtype
            target_dtype = self.weight.dtype  # fp8_e4m3fn for patched Linears
            fp8_max = torch.finfo(target_dtype).max

            scale_input = getattr(self, "scale_input", None)
            if scale_input is not None:
                scale_x = scale_input.to(torch.float32).reshape(())
            else:
                scale_x = (x.detach().abs().amax() / fp8_max).clamp(min=1e-8).to(torch.float32)

            # Quantize activations in the input dtype (bf16) — fp8 only keeps 3
            # mantissa bits, so a float32 intermediate buys no precision and just
            # doubles activation memory traffic. inv is a bf16 scalar.
            inv = (1.0 / scale_x).to(x.dtype)
            original_shape = x.shape
            x_fp8 = (x * inv).clamp_(-fp8_max, fp8_max).to(target_dtype)
            x_fp8 = x_fp8.reshape(-1, original_shape[-1]).contiguous()

            weight = self.weight.t()  # (K, N), column-major as _scaled_mm wants
            scale_weight = self.scale_weight.to(torch.float32).reshape(())

            o = torch._scaled_mm(x_fp8, weight, scale_a=scale_x, scale_b=scale_weight, out_dtype=input_dtype)
            if self.bias is not None:
                o = o + self.bias.to(o.dtype)

            return o.reshape(*original_shape[:-1], -1)
        except Exception as e:
            if not getattr(self, "_scaled_mm_warned", False):
                logger.warning(f"_scaled_mm failed ({type(e).__name__}: {e}); falling back to dequant path.")
                self._scaled_mm_warned = True
            # fall through to the dequant path

    # GC-off training only: route the frozen-base dequant through a custom
    # autograd Function so the materialized bf16 weight is NOT retained for
    # backward (it's recomputed from fp8 instead). `not self.training and
    # grad_enabled` uniquely identifies this case: the trainer puts the DiT in
    # eval() when gradient checkpointing is off, so a frozen-base forward that
    # still needs grad = GC-off training. Inference is eval()+no_grad (skipped),
    # and GC-on keeps train() mode (skipped) — both keep their proven paths.
    if (not self.training) and torch.is_grad_enabled():
        out = _FP8DequantLinear.apply(x, self.weight, self.scale_weight)
        if self.bias is not None:
            out = out + self.bias.to(device=out.device, dtype=out.dtype)
        return out

    if True:
        # Dequantize on the input's device (handles block swap: weight on CPU, x on CUDA)
        target_device = x.device
        weight = self.weight.to(target_device)
        scale = self.scale_weight.to(target_device)
        original_dtype = scale.dtype
        if scale.ndim < 3:
            dequantized_weight = weight.to(original_dtype) * scale
        else:
            out_features, num_blocks, _ = scale.shape
            dequantized_weight = weight.to(original_dtype).contiguous().view(out_features, num_blocks, -1)
            dequantized_weight = dequantized_weight * scale
            dequantized_weight = dequantized_weight.view(weight.shape)

        # Perform linear transformation
        if self.bias is not None:
            output = F.linear(x, dequantized_weight, self.bias.to(target_device))
        else:
            output = F.linear(x, dequantized_weight)

        return output


def apply_fp8_monkey_patch(model, optimized_state_dict, use_scaled_mm=False):
    """
    Apply monkey patching to a model using FP8 optimized state dict.

    Args:
        model (nn.Module): Model instance to patch
        optimized_state_dict (dict): FP8 optimized state dict
        use_scaled_mm (bool): Use scaled_mm for FP8 Linear layers, requires SM 8.9+ (RTX 40 series)

    Returns:
        nn.Module: The patched model (same instance, modified in-place)
    """
    # # Calculate FP8 float8_e5m2 max value
    # max_value = calculate_fp8_maxval(5, 2)
    max_value = None  # do not quantize input tensor

    # Find all scale keys to identify FP8-optimized layers
    scale_keys = [k for k in optimized_state_dict.keys() if k.endswith(".scale_weight")]

    # Enumerate patched layers
    patched_module_paths = set()
    scale_shape_info = {}
    for scale_key in scale_keys:
        # Extract module path from scale key (remove .scale_weight)
        module_path = scale_key.rsplit(".scale_weight", 1)[0]
        patched_module_paths.add(module_path)

        # Store scale shape information
        scale_shape_info[module_path] = optimized_state_dict[scale_key].shape

    patched_count = 0

    # Apply monkey patch to each layer with FP8 weights
    for name, module in model.named_modules():
        # Check if this module has a corresponding scale_weight
        has_scale = name in patched_module_paths

        # Apply patch if it's a Linear layer with FP8 scale
        if isinstance(module, nn.Linear) and has_scale:
            # register the scale_weight as a buffer to load the state_dict
            # module.register_buffer("scale_weight", torch.tensor(1.0, dtype=module.weight.dtype))
            scale_shape = scale_shape_info[name]
            module.register_buffer("scale_weight", torch.ones(scale_shape, dtype=module.weight.dtype))

            # For scaled_mm: register the pre-calibrated input (activation) scale
            # as a buffer so load_state_dict populates it. Only present for
            # pre-quantized files that ship `input_scale` per Linear.
            input_scale_key = name + ".scale_input"
            if input_scale_key in optimized_state_dict:
                in_shape = optimized_state_dict[input_scale_key].shape
                module.register_buffer("scale_input", torch.ones(in_shape, dtype=torch.float32))

            # Create a new forward method with the patched version.
            def new_forward(self, x):
                return fp8_linear_forward_patch(self, x, use_scaled_mm, max_value)

            # Bind method to module
            module.forward = new_forward.__get__(module, type(module))

            patched_count += 1

    logger.info(f"Number of monkey-patched Linear layers: {patched_count}")
    return model
