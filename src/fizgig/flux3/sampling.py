"""Flux 3 Image Klein sampling and token packing, after the reference (src/flux3/sampling.py + pipeline.py).

Layouts:
* latents as the cache stores them: (128, h, w), the autoencoder's BatchNorm-normalised space, h = H / 16
* tokens as the DiT sees them: (h * w, 128), row-major - the same order the autoencoder's encode returns
* image ids (0, row, col, 0) with rows and cols scaled by 0.5 (IMAGE_ROPE_SCALE); text ids (0, 0, 0, position);
  reference image i gets l = -(1 + i)

Schedule: t from 1 (noise) to 0 (clean), shifted by alpha (6.93 in the reference, for both checkpoints).
Base: Euler on the velocity, CFG on a two-sample pack [unconditional, conditional] (unconditional = the empty prompt).
Distilled: a flow map - each step jumps to the next time, told the target time, no CFG.
"""
import torch

from fizgig.flux3.model import IN_CHANNELS, MODES, Packed

ALPHA = 6.93                  # pipeline.py Pipeline.generate default
IMAGE_ROPE_SCALE = 0.5        # pipeline.py
DOWNSAMPLING_RATIO = 16


def schedule(steps: int, alpha: float = ALPHA) -> torch.Tensor:
    t = torch.linspace(1.0, 0.0, steps + 1)
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def shift(t: float, alpha: float = ALPHA) -> float:
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def latent_ids(h: int, w: int, device="cpu") -> torch.Tensor:
    rows, cols = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    zeros = torch.zeros_like(rows)
    return torch.stack([zeros, rows, cols, zeros], dim=-1).reshape(h * w, 4)


def image_ids(h: int, w: int, device="cpu", ref_index=None) -> torch.Tensor:
    ids = latent_ids(h, w, device).float()
    ids[:, 1:3] *= IMAGE_ROPE_SCALE
    if ref_index is not None:
        ids[:, 3] = -(1 + ref_index)
    return ids


def text_ids(length: int, device="cpu") -> torch.Tensor:
    pos = torch.arange(length, device=device)
    return torch.stack([torch.zeros_like(pos)] * 3 + [pos], dim=-1).float()


def to_tokens(latents: torch.Tensor) -> torch.Tensor:
    """(128, h, w) or (1, 128, h, w) -> (h * w, 128)."""
    z = latents[0] if latents.dim() == 4 else latents
    return z.permute(1, 2, 0).reshape(-1, z.shape[0])


def to_latents(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(h * w, 128) -> (128, h, w)."""
    return tokens.reshape(h, w, -1).permute(2, 0, 1)


def pack(samples) -> Packed:
    """samples[s] = [text, image_cond or None, image], each a (tokens, ids) pair (pipeline.py pack)."""
    data = {m: [] for m in MODES}
    ids = {m: [] for m in MODES}
    mode_indices = {m: [] for m in MODES}
    counts = dict.fromkeys(MODES, 0)
    for text, cond, image in samples:
        for mode, item in (("text", text), ("image_cond", cond), ("image", image)):
            start = counts[mode]
            if item is not None:
                data[mode].append(item[0])
                ids[mode].append(item[1])
                counts[mode] += item[0].shape[0]
            mode_indices[mode].append((start, counts[mode]))
    device = samples[0][0][0].device
    cat = lambda parts, width, dtype: torch.cat(parts) if parts else torch.zeros((0, width), device=device, dtype=dtype)
    data = {m: cat(data[m], IN_CHANNELS[m], torch.bfloat16) for m in MODES}
    ids = {m: cat(ids[m], 4, torch.float32) for m in MODES}
    return Packed(data=data, ids=ids, mode_indices=mode_indices,
                  timesteps={m: torch.zeros(data[m].shape[0], device=device, dtype=torch.bfloat16) for m in MODES})


def text_item(tokens: torch.Tensor, device):
    """Cached text tokens (L, 20480) or (1, L, 20480) -> the (tokens, ids) pair pack() takes."""
    t = tokens[0] if tokens.dim() == 3 else tokens
    return t.to(device, torch.bfloat16), text_ids(t.shape[0], device)


def initial_noise(seed: int, height: int, width: int) -> torch.Tensor:
    """The reference's start for this seed: (h * w, 128) on CPU, bf16 values held in fp32."""
    h, w = height // DOWNSAMPLING_RATIO, width // DOWNSAMPLING_RATIO
    noise = torch.randn((h * w, IN_CHANNELS["image"]), generator=torch.Generator().manual_seed(seed))
    return noise.to(torch.bfloat16).float()


@torch.no_grad()
def sample(dit, txt, height, width, *, steps, seed, cfg=1.0, neg_txt=None, noise=None, on_step=None,
           sigmas=None, alpha=ALPHA, refs=None):
    """-> denoised image tokens (h * w, 128), fp32. txt / neg_txt: cached text tokens. refs: list of
    (128, rh, rw) reference latents (edit), or None."""
    if height % DOWNSAMPLING_RATIO or width % DOWNSAMPLING_RATIO:
        raise ValueError(f"width and height must be multiples of {DOWNSAMPLING_RATIO}")
    device = next(dit.parameters()).device
    h, w = height // DOWNSAMPLING_RATIO, width // DOWNSAMPLING_RATIO
    x = (initial_noise(seed, height, width) if noise is None else noise.float()).to(device)
    ids = image_ids(h, w, device)
    cond = None
    if refs:
        toks = [to_tokens(r).to(device, torch.bfloat16) for r in refs]
        rids = [image_ids(r.shape[-2], r.shape[-1], device, ref_index=i) for i, r in enumerate(refs)]
        cond = (torch.cat(toks), torch.cat(rids))
    sched = torch.tensor(sigmas, dtype=torch.float32) if sigmas is not None else schedule(steps, alpha)
    if sched[-1] != 0.0:
        sched = torch.cat([sched, sched.new_zeros(1)])
    use_cfg = (not dit.flow_map) and neg_txt is not None and cfg != 1.0
    text = [text_item(neg_txt, device), text_item(txt, device)] if use_cfg else [text_item(txt, device)]
    packed = pack([[t, cond, (x, ids)] for t in text])
    n = len(sched) - 1
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        for step in range(n):
            if on_step is not None:
                on_step(step, n)
            t, dt = sched[step].item(), (sched[step + 1] - sched[step]).item()
            packed.timesteps["image"].fill_(t)
            if dit.flow_map:
                target = sched[step + 1].reshape(1).to(device)
                packed.data["image"] = packed.data["image"] + dt * dit(packed, timesteps_ref=target)
            elif use_cfg:
                unconditional, conditional = dit(packed).chunk(2)
                velocity = unconditional + cfg * (conditional - unconditional)
                x = packed.data["image"].chunk(2)[0] + dt * velocity
                packed.data["image"] = torch.cat((x, x))
            else:
                packed.data["image"] = packed.data["image"] + dt * dit(packed)
    return packed.data["image"].chunk(len(text))[0].float()
