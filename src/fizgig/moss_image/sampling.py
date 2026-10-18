"""MOSS-Image sampling, after the reference pipeline (OpenMOSS-Team/MOSS-Image-Base pipeline_moss_image.py:
MOSSImageScheduler + MOSSImagePipeline.__call__).

* latents: (128, h, w) at /16 - the FLUX.2 autoencoder's BatchNorm-normalised space, 2x2-packed (the DiT takes them
  as they are, [B, 128, h, w])
* schedule: sigmas on the shift-5 grid, each snapped to the nearest of the 1000 training sigmas (j / 1000), repeats
  dropped, then 0 appended; the DiT is told 1000 * sigma
* Euler on the velocity (noise - x0) in fp32: x <- x + (sigma_next - sigma) * v; or ComfyUI's res_multistep (second
  order on the denoised estimate x - sigma * v, one model call a step), which the community runs at 20 steps
* CFG 3 against the negative prompt (empty by default), whose branch skips block 22 ("uncond_skip_layers=[22]")
* the seed's noise: fp32 randn on a CPU generator, [1, 128, h, w] (randn_tensor with a CPU generator)
"""
import math

import torch

SHIFT = 5.0                  # scheduler_config.json shift
TRAIN_STEPS = 1000           # num_train_timesteps
TRAINING_SHIFT = 1.0         # training_shift: the snapping grid is unshifted
UNCOND_SKIP = (22,)          # pipeline __call__ uncond_skip_layers default
DOWNSAMPLING_RATIO = 16      # 8x VAE x 2x2 packing


def _grid(count, shift, total=TRAIN_STEPS):
    lower = shift / (total + shift - 1)
    values = torch.linspace(1, lower, count)
    return shift * values / (1 + (shift - 1) * values)


def schedule(steps: int, shift: float = SHIFT) -> torch.Tensor:
    """MOSSImageScheduler.set_timesteps, then FlowMatchEulerDiscreteScheduler's terminal 0: (n + 1,) fp32, n <= steps
    (snapping can merge neighbours near sigma 1)."""
    if steps < 1 or shift <= 0:
        raise ValueError("Steps and shifts must be positive.")
    sigmas = _grid(steps, shift)
    training = _grid(TRAIN_STEPS, TRAINING_SHIFT).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=TRAIN_STEPS - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas.float(), torch.zeros(1)])


def latent_hw(height: int, width: int):
    if height % DOWNSAMPLING_RATIO or width % DOWNSAMPLING_RATIO:
        raise ValueError("height and width must be positive multiples of 16.")
    return height // DOWNSAMPLING_RATIO, width // DOWNSAMPLING_RATIO


def initial_noise(seed: int, height: int, width: int) -> torch.Tensor:
    """The reference's start for this seed: [1, 128, h, w] fp32 on CPU."""
    h, w = latent_hw(height, width)
    gen = torch.Generator("cpu").manual_seed(int(seed))
    return torch.randn((1, 128, h, w), generator=gen, device="cpu", dtype=torch.float32)


def _text(hidden, device):
    """Cached (L, 4096) or (1, L, 4096) -> [1, L, 4096] on the device."""
    h = hidden if hidden.dim() == 3 else hidden[None]
    return h.to(device)


@torch.no_grad()
def sample(dit, text, height, width, *, steps, seed, cfg=3.0, neg_text=None, noise=None, on_step=None, sigmas=None,
           shift=SHIFT, uncond_skip=UNCOND_SKIP, skip_start=0.0, skip_stop=1.0, sampler="euler"):
    """-> denoised latents [1, 128, h, w] fp32. text / neg_text: cached (L, 4096) hidden states. sigmas: an explicit
    schedule (a speed LoRA's), used as given with 0 appended if missing. sampler: "euler" or "res_multistep"."""
    from fizgig.minimax.sampling import _res_multistep_coeffs
    device = next(dit.parameters()).device
    h, w = latent_hw(height, width)
    x = (initial_noise(seed, height, width) if noise is None else noise.float().reshape(1, 128, h, w)).to(device)
    if sigmas is not None:
        sched = torch.tensor(list(sigmas), dtype=torch.float32)
        if float(sched[-1]) != 0.0:
            sched = torch.cat([sched, torch.zeros(1)])
    else:
        sched = schedule(steps, shift)
    pos = _text(text, device)
    use_cfg = cfg != 1.0 and neg_text is not None
    neg = _text(neg_text, device) if use_cfg else None
    n = len(sched) - 1
    prev = None                                      # res_multistep's previous denoised estimate
    for i in range(n):
        if on_step is not None:
            on_step(i, n)
        s, s_next = float(sched[i]), float(sched[i + 1])
        t = torch.tensor([s * TRAIN_STEPS], device=device)
        v = dit(x, t, pos).float()                   # CFG combines fp32 predictions, as the reference
        if use_cfg:
            skip = uncond_skip if (uncond_skip and skip_start <= i / n < skip_stop) else ()
            vu = dit(x, t, neg, skip_layers=skip).float()
            v = vu + cfg * (v - vu)
        denoised = x - s * v
        # second order only where the previous step was at least half as long (in log sigma): MOSS's shift-5 grid
        # ends on one long step (0.23 -> 0.024 at 20 steps), and extrapolating across it overshoots the contrast
        if (sampler == "res_multistep" and prev is not None and s_next > 0
                and math.log(float(sched[i - 1]) / s) >= 0.5 * math.log(s / s_next)):
            a_x, hb1, hb2 = _res_multistep_coeffs(s, s_next, float(sched[i - 1]))
            x = a_x * x + hb1 * denoised + hb2 * prev
        else:
            x = x + (s_next - s) * v                 # Euler (and res_multistep's first and last step)
        prev = denoised
    return x
