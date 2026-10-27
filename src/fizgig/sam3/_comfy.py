# Derived from ComfyUI (https://github.com/comfyanonymous/ComfyUI), Copyright (C) comfyanonymous and the ComfyUI
# contributors, GNU General Public License v3.0. This file stays under GPL-3.0 and is not covered by Fizgig's Apache-2.0
# licence. See docs/THIRD_PARTY_NOTICES.md (ComfyUI) and LICENSES/GPL-3.0.txt.
"""The few ComfyUI pieces the SAM 3 port uses, in plain PyTorch: layer classes (comfy.ops' disable_weight_init),
attention (comfy.ldm.modules.attention.attention_pytorch), cast_to_input, the FLUX rope helpers (comfy.ldm.flux),
LayerNorm2d (comfy.ldm.cascade.common) and a resize (comfy.utils.common_upscale)."""
import types

import torch
import torch.nn.functional as F
from torch import nn


class _NoInit:
    def reset_parameters(self):          # weights come from the file
        return None


class Linear(_NoInit, nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class Conv2d(_NoInit, nn.Conv2d):
    def forward(self, x):
        return self._conv_forward(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class ConvTranspose2d(_NoInit, nn.ConvTranspose2d):
    def forward(self, x, output_size=None):
        return F.conv_transpose2d(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype),
                                  self.stride, self.padding, self.output_padding, self.groups, self.dilation)


class LayerNorm(_NoInit, nn.LayerNorm):
    def forward(self, x):
        w = None if self.weight is None else self.weight.to(x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.layer_norm(x, self.normalized_shape, w, b, self.eps)


class GroupNorm(_NoInit, nn.GroupNorm):
    def forward(self, x):
        w = None if self.weight is None else self.weight.to(x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.group_norm(x, self.num_groups, w, b, self.eps)


class Embedding(_NoInit, nn.Embedding):
    pass


ops = types.SimpleNamespace(Linear=Linear, Conv2d=Conv2d, ConvTranspose2d=ConvTranspose2d, LayerNorm=LayerNorm,
                            GroupNorm=GroupNorm, Embedding=Embedding)


def cast_to_input(weight, input, non_blocking=False, copy=True):
    return weight.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking)


def optimized_attention(q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


def rope(pos, dim, theta):
    assert dim % 2 == 0
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.to(dtype=torch.float32, device=pos.device)


def apply_rope1(x, freqs_cis):
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, :x_.shape[2]]
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(*x.shape).type_as(x)


def apply_rope(xq, xk, freqs_cis):
    return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)


class EmbedND(nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim, self.theta, self.axes_dim = dim, theta, axes_dim

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


def LayerNorm2d_op(operations):
    class LayerNorm2d(operations.LayerNorm):
        def forward(self, x):
            return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    return LayerNorm2d


def common_upscale(samples, width, height, upscale_method, crop):
    return F.interpolate(samples, size=(height, width), mode=upscale_method)


def intermediate_device():
    return torch.device("cpu")


def is_device_cuda(device):
    return torch.device(device).type == "cuda"
