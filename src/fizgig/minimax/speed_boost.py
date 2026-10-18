"""Speed Boost for H3 clips (the Studio's tick): the first half of the steps on a half-size video, then the video
grown to full size for the rest.

Peter's ComfyUI node "Fizgig H3 Speed Boost", brought into Fizgig's own sampler. The spectral shrink / grow is SPEED
("Spectral Progressive Diffusion for Efficient Image and Video Generation", Xiao, Chao, Yariv and Wetzstein, 2026) as
written in LC Speed Boost (ComfyUI_LC123_nodes, MIT, (c) 2026 lonecatone23; see docs/THIRD_PARTY_NOTICES.md).

Flow: x = (1 - t) * picture + t * noise. Shrinking keeps the low part of the cosine spectrum, so white noise stays
white at the same strength and a picture comes out r times as strong (r = full / small size per side). Growing pads
the spectrum with fresh noise; the picture is then 1/r as strong as the noise, so the state is scaled by
k = r / (1 + (r - 1) * t) and sampling carries on at noise k * t. The audio rides the video's schedule, so when the
video's noise jumps from t to k * t the audio is topped up with fresh noise to the same level.
"""
from __future__ import annotations

import math

import torch


def _dct(n, device):
    """Orthonormal DCT-II matrix: coefficients = D @ signal, signal = D.T @ coefficients."""
    k = torch.arange(n, dtype=torch.float64, device=device).unsqueeze(1)
    i = torch.arange(n, dtype=torch.float64, device=device).unsqueeze(0)
    d = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    d[0] /= math.sqrt(2.0)
    return d.float()


def _spec(x):
    h, w = x.shape[-2:]
    return _dct(h, x.device) @ x.float() @ _dct(w, x.device).T


def _unspec(c):
    h, w = c.shape[-2:]
    return _dct(h, c.device).T @ c @ _dct(w, c.device)


def half_size(lat_h, lat_w):
    """The small latent grid: half per side, even (H3 patches are 2 x 2)."""
    return max(2, 2 * round(lat_h * 0.5 / 2)), max(2, 2 * round(lat_w * 0.5 / 2))


def small_steps(n_eval):
    """How many steps run small: the first half, rounded down (7 -> 3, 8 -> 4); 0 = no boost (fewer than 2)."""
    return n_eval // 2 if n_eval >= 2 else 0


def shrink(x, h, w):
    """Full size -> h x w by the low part of the spectrum (white noise stays white; a picture comes out r times as
    strong - divide by r for the picture itself)."""
    return _unspec(_spec(x)[..., :h, :w].contiguous()).to(x.dtype)


def grow(y, H, W, t, gen):
    """A small state at noise t -> (the full-size state, its noise level k * t)."""
    h, w = y.shape[-2:]
    c = torch.zeros(*y.shape[:-2], H, W, device=y.device, dtype=torch.float32)
    c[..., :h, :w] = _spec(y)
    fresh = torch.randn(c.shape, generator=gen).to(y.device)
    fresh[..., :h, :w] = 0.0
    c += t * fresh
    r = math.sqrt((H * W) / float(h * w))
    k = r / (1.0 + (r - 1.0) * t)
    return (_unspec(c) * k).to(y.dtype), k * t


def grow_audio(a, t, t2, gen):
    """The carried audio state at noise t, rescaled to t2 with fresh noise topping it up to the video's new level."""
    extra = math.sqrt(max(t2 ** 2 - (t * (1.0 - t2) / (1.0 - t)) ** 2, 0.0))
    return a * ((1.0 - t2) / (1.0 - t)) + extra * torch.randn(a.shape, generator=gen).to(a)
