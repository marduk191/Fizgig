"""MiniMax H3 — image sampling for in-training previews.

Renders ONE still per prompt. H3's temporal grid is FRAME_PER_TOKEN = (1, 4, 4, 4, 4), so a
single latent frame decodes to exactly one pixel frame — which is also what the encoder produces
for a still, and what training uses. So a preview is the training forward run backwards: the DiT
packs the same [text | audio | video] sequence, and there is no multi-frame RoPE or cond-row
work to do. The audio rows are DENOISED jointly on their own shift-3 schedule, exactly like the
reference pipeline (training noises silence instead — also what the reference trainer does for
image datasets).

A preview is still a *training-distribution* render, not a byte-faithful ComfyUI one (ComfyUI
generates >= 2 latent frames and denoises the audio stream for real). It answers "is this LoRA
learning?"; final quality judgement belongs in ComfyUI. Same honest caveat Klein's samples carry.

Flow / sign convention (this is the easy thing to get backwards):
  the DiT head returns `video_out = x0 - noise` (the reference NEGATES it to hand a sampler its
  velocity, then uses denoised = x - v*sigma). Integrating our RAW head output from sigma=1 down
  to 0 is therefore:

      x += (sigma_curr - sigma_next) * model_out

  Check: from sigma=1 (x = noise), one step to sigma=0 gives noise + (x0 - noise) = x0.
  Note this is the opposite arrangement to Krea 2's `img += (t_prev - t_curr) * v`, whose t runs
  1 -> 0 rather than being a noise level.
"""

import gc
import logging

import torch

from fizgig.minimax.model import AUDIO_CHANNELS, audio_latents_for_frames

logger = logging.getLogger(__name__)

# Per-voxel linear 24ch-latent -> RGB approximation (ComfyUI's MiniMaxH3Video.latent_rgb_factors).
# A free, instant, very rough preview: no decoder, no VRAM. Good enough to confirm a sampler is
# producing structure; NOT good enough for likeness scoring — that needs the real VAE decoder.
LATENT_RGB_FACTORS = [
    [-0.018555, 0.024344, -0.017536], [0.150164, 0.137244, 0.129221],
    [0.027367, -0.050369, -0.208606], [-0.000793, -0.164622, -0.323161],
    [-0.048556, 0.013970, -0.074286], [0.011740, 0.014172, -0.006906],
    [0.061517, 0.061212, 0.110025], [0.035321, 0.086879, 0.110059],
    [-0.017426, 0.002997, 0.035356], [0.531539, 0.548819, 0.624404],
    [-0.024968, -0.040234, -0.034302], [-0.032549, -0.029096, -0.017221],
    [0.022609, 0.020286, 0.050661], [-0.084001, -0.038131, -0.020805],
    [-0.018830, 0.010412, 0.061120], [0.020777, 0.011196, -0.030994],
    [-0.008390, -0.012201, -0.025687], [-0.013281, -0.002924, 0.006331],
    [0.000260, 0.001833, -0.011038], [0.105471, 0.100482, 0.132106],
    [0.016529, 0.015213, 0.009999], [-0.014015, -0.017438, -0.019134],
    [-0.033787, -0.009984, -0.019725], [0.004224, 0.017284, 0.027196],
]
LATENT_RGB_BIAS = [0.057426, -0.022078, -0.071449]


def sample_schedule(steps: int, shift: float = 12.0, mode: str = "comfy"):
    """Descending sigmas 1 -> 0 on H3's shifted grid: sigma = shift*u / (1 + (shift-1)*u).

    mode="comfy" (DEFAULT) reproduces ComfyUI's `simple` scheduler against
    ModelSamplingDiscreteFlow(shift=12) — the H3 sampling_settings. Comfy builds a 1000-entry
    sigma table from u = (i+1)/1000 and `simple` walks it backwards in even strides, then
    appends 0. That yields `steps + 1` sigmas and `steps` model evaluations, i.e. "20 steps"
    means twenty forwards, as it does in the UI.

    mode="reference" is ai-toolkit's build_sigma_schedule: linspace(1, 0, steps) through the
    same shift, consecutive duplicates collapsed, terminal 0 INSIDE the count — so `steps`
    yields `steps - 1` evaluations. Kept for comparing against their previews.

    Previews default to comfy because ComfyUI is where the results are judged: a preview is
    worth more when it predicts what you will see there than when it matches another trainer.
    The two grids are close but not equal (at 20 steps the last non-zero sigma is 0.3871 for
    comfy against 0.4000 for the reference) and the evaluation counts differ by one.

    shift 12.0 is the video schedule every shipped H3 workflow uses."""
    if steps < 1:
        raise ValueError("sample_schedule needs at least one step")
    if mode == "comfy":
        # comfy's table: sigmas[i] = shift_map((i+1)/1000), walked back in even strides
        table = [(shift * u) / (1.0 + (shift - 1.0) * u)
                 for u in ((i + 1) / 1000.0 for i in range(1000))]
        stride = len(table) / steps
        return [table[-(1 + int(x * stride))] for x in range(steps)] + [0.0]
    if mode != "reference":
        raise ValueError(f"sample_schedule mode must be 'comfy' or 'reference', got {mode!r}")
    if steps < 2:
        raise ValueError("reference mode needs at least 2 sigmas (one model evaluation)")
    out = []
    for i in range(steps):
        u = 1.0 - i / (steps - 1)                       # linspace(1, 0, steps)
        sig = (shift * u) / (1.0 + (shift - 1.0) * u)
        if not out or sig != out[-1]:                   # unique_consecutive
            out.append(sig)
    return out


def latent_to_rgb(latent: torch.Tensor):
    """[1, 24, 1, H, W] latent -> uint8 HWC array, via the linear RGB approximation.

    Upscales nothing: the result is one pixel per latent cell (1/16 scale), so it is a thumbnail
    of a thumbnail. Purely a sanity view until the real decoder lands."""
    import numpy as np
    f = torch.tensor(LATENT_RGB_FACTORS, dtype=torch.float32, device=latent.device)
    b = torch.tensor(LATENT_RGB_BIAS, dtype=torch.float32, device=latent.device)
    z = latent.detach().float()[0, :, 0]                  # [24, H, W]
    rgb = torch.einsum("chw,cr->rhw", z, f) + b[:, None, None]
    rgb = rgb.clamp(0, 1).mul(255).round().byte().cpu().numpy()
    return np.transpose(rgb, (1, 2, 0))                   # HWC


def _res_multistep_coeffs(sig_cur, sig_next, sig_prev):
    """(sigma_fn(h), h*b1, h*b2) for ComfyUI's res_multistep second-order step (eta = 0).

    Ported from comfy/k_diffusion/sampling.py::res_multistep, which implements the multistep
    scheme in arXiv:2308.02157. With eta = 0 the ancestral machinery collapses (sigma_down is
    just the next sigma, sigma_up is 0, no noise is re-injected), leaving:

        t = -log(sigma),  h = t_next - t,  c2 = (t_prev - t) / h
        b1 = phi1(-h) - phi2(-h)/c2 ,  b2 = phi2(-h)/c2
        x <- exp(-h) * x + h * (b1 * denoised + b2 * denoised_prev)

    where phi1(t) = expm1(t)/t and phi2(t) = (phi1(t) - 1)/t. Done in float64: h is small at
    the top of a shift-12 grid and phi2 divides by it twice."""
    import math
    t = -math.log(max(float(sig_cur), 1e-12))
    t_next = -math.log(max(float(sig_next), 1e-12))
    t_prev = -math.log(max(float(sig_prev), 1e-12))
    h = t_next - t
    c2 = (t_prev - t) / h
    phi1 = math.expm1(-h) / -h
    phi2 = (phi1 - 1.0) / -h
    b1 = phi1 - phi2 / c2
    b2 = phi2 / c2
    return math.exp(-h), h * b1, h * b2


@torch.no_grad()
def sample_image(model, text_embeds, *, width=512, height=512, steps=8, cfg_scale=1.0,
                 uncond_embeds=None, seed=0, shift=12.0, device="cuda",
                 dtype=torch.bfloat16, latent_channels=24, spatial=16, log_steps=False,
                 sampler="res_multistep", schedule_mode="comfy",
                 ref_latents=None, text_token_tags=None):
    """Denoise one image and return its LATENT [1, 24, 1, H/16, W/16].

    Decoding is the caller's business (real VAE decoder, or latent_to_rgb for a rough look) so
    this stays testable without a 4.8 GB decoder resident.

    cfg_scale <= 1.0 runs a single forward per step (what every shipped H3 workflow does); above
    that costs a second forward, because the DiT is hard-locked to batch size 1.

    `steps` counts SIGMAS including the terminal 0, matching the reference — so it runs
    steps - 1 model evaluations.

    log_steps prints one line per evaluation (step index and the sigma being denoised from),
    so a preview is visible in the console instead of a silent pause.

    ref_latents / text_token_tags drive r2v reference conditioning: the reference latents ride
    as condition rows in every evaluation, and the tags mark which conditioning rows are the
    `<Picture i>` vision blocks. Both are passed straight through to the DiT, unchanged across
    steps — the references are conditioning, not something being denoised.
    """
    lat_h, lat_w = height // spatial, width // spatial
    # The DiT patchifies 2x2, so the latent grid must be even (compute_loss crops for the same
    # reason on the training side).
    lat_h, lat_w = (lat_h // 2) * 2, (lat_w // 2) * 2
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn(1, latent_channels, 1, lat_h, lat_w, generator=gen, dtype=torch.float32).to(device)
    # Joint audio denoising, mirroring the reference pipeline: the audio rows start as pure
    # noise (sigma_a(1) = 1) and the model's AUDIO head steps them down their own shift-3
    # schedule alongside the video. At low sigma the video tokens then attend to plausible
    # audio content — the regime the base was pretrained in — rather than scaled noise.
    joint_audio = bool(getattr(model, "pack_audio_rows", False))
    audio_rows = None
    if joint_audio:
        n_audio = audio_latents_for_frames(1) * AUDIO_CHANNELS
        audio_rows = torch.randn(n_audio, model.config.audio_latents_dim,
                                 generator=gen, dtype=torch.float32).to(device)

    from fizgig.minimax.model import remap_sigma
    # Built once: identical for every step, so it never lands in the hot loop.
    _ref_kw = {}
    if ref_latents:
        _ref_kw["ref_latents"] = ref_latents
        _ref_kw["seed"] = int(seed)
    if text_token_tags is not None:
        _ref_kw["text_token_tags"] = text_token_tags
    # the uncond prompt has its own length, so it must NOT carry the cond prompt's tags
    _ref_uncond_kw = {k: v for k, v in _ref_kw.items() if k != "text_token_tags"}
    use_cfg = cfg_scale > 1.0 and uncond_embeds is not None
    sigmas = sample_schedule(steps, shift=shift, mode=schedule_mode)
    n_eval = len(sigmas) - 1                            # the terminal 0 is not an evaluation
    prev_denoised = None                                # res_multistep's one-step memory
    for i in range(n_eval):
        s_curr, s_next = sigmas[i], sigmas[i + 1]
        if log_steps:
            print(f"[preview] step {i + 1}/{n_eval}  sigma {s_curr:.4f} -> {s_next:.4f}",
                  flush=True)
        t = torch.tensor([1.0 - s_curr], device=device)     # the DiT is conditioned on cleanness
        if joint_audio:
            out, a_out = model(x.to(dtype), t, text_embeds,
                               audio_rows=audio_rows, return_audio=True, **_ref_kw)
            out = out.float()
            if use_cfg:
                # NOTE the unconditional pass keeps the reference rows: they are conditioning
                # the workflow supplies, not part of the prompt being dropped.
                out_u, _ = model(x.to(dtype), t, uncond_embeds,
                                 audio_rows=audio_rows, return_audio=True, **_ref_uncond_kw)
                out = out_u.float() + cfg_scale * (out - out_u.float())
            # audio Euler step on ITS schedule (shift 3, coupled by the closed-form remap)
            sa_curr = float(remap_sigma(torch.tensor(float(s_curr))))
            sa_next = float(remap_sigma(torch.tensor(float(s_next))))
            audio_rows = audio_rows + (sa_curr - sa_next) * a_out.float()
        else:
            out = model(x.to(dtype), t, text_embeds, **_ref_kw).float()
            if use_cfg:
                out_u = model(x.to(dtype), t, uncond_embeds, **_ref_uncond_kw).float()
                out = out_u + cfg_scale * (out - out_u)
        # `out` is the head's x0 - noise, so the denoised estimate is x + sigma*out (one step
        # from sigma to 0). Euler is x + (sigma - sigma_next)*out, identical to comfy's
        # to_d()/dt form; res_multistep reuses the PREVIOUS denoised for a 2nd-order step.
        denoised = x + s_curr * out
        if sampler == "res_multistep" and prev_denoised is not None and s_next > 0:
            a_x, hb1, hb2 = _res_multistep_coeffs(s_curr, s_next, sigmas[i - 1])
            x = a_x * x + hb1 * denoised + hb2 * prev_denoised
        else:
            x = x + (s_curr - s_next) * out             # Euler (first step, last step, or opt-out)
        prev_denoised = denoised
    return x


def encode_sample_prompts(te_path, prompts, *, device="cuda", quantize=True):
    """Pre-encode preview prompts, then free the encoder.

    Always call this BEFORE the DiT loads: the Qwen3-VL-32B text encoder is ~14 GB NF4 and must
    never be resident alongside the ~17 GB DiT. Returns CPU tensors."""
    from fizgig.minimax.embedder import load_minimax_h3_te
    te = load_minimax_h3_te(te_path, device=device, compute_dtype=torch.bfloat16, quantize=quantize)
    out = [te.encode(p).cpu() for p in prompts]
    del te
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
