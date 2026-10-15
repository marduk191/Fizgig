"""Model EQ LoRA: a described family's own weights as a LoRA - the top `rank` singular directions of every Linear the
family's LoRA wraps (exact SVD of the base weight), saved in the family's LoRA format with Fizgig's marker.

Repair Studio reads the marker (ss_fizgig_model_lora) and maps each block's slider s to (s - 1) x gain of those
directions: 1.0 = the stock model, 0 / 2 = -gain / +gain, an unticked block = stock. So the block sliders become an EQ
of the model itself, and a save bakes the EQ into an ordinary LoRA (no marker) that any loader uses at 1.0.

usage: extract_model_lora.py --family acestep15xl --dit <DiT file> --out <file> [--rank 128] [--gain 0.25]
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MARKER = "ss_fizgig_model_lora"
GAIN_KEY = "ss_fizgig_model_lora_gain"


@torch.no_grad()
def extract(family, dit_path, out_path, rank=128, gain=0.25, device="cpu"):
    from safetensors.torch import save_file
    from fizgig.families.lora import FamilyLoRA
    from fizgig.families.registry import get
    desc = get(family)
    if desc is None:
        raise SystemExit(f"unknown family {family!r}")
    driver = desc.load_driver()
    logger.info(f"loading {os.path.basename(dit_path)} ...")
    dit = driver.load_dit(dit_path, "cpu")
    net = FamilyLoRA(dit, driver, device="cpu")
    sd, kept, total, t0 = {}, 0.0, 0.0, time.time()
    for i, (full, w) in enumerate(sorted(net.wrapped.items())):
        if full not in net.targets:
            continue
        W = w.base.weight.detach().to(device, torch.float32)
        if W.dim() != 2:
            continue
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = min(rank, S.numel())
        root = S[:r].sqrt()
        down = (root[:, None] * Vh[:r]).to("cpu", torch.bfloat16).contiguous()      # (r, in)
        up = (U[:, :r] * root).to("cpu", torch.bfloat16).contiguous()                # (out, r)
        ka, kb, kal = net._keys(full)
        sd[ka], sd[kb], sd[kal] = down, up, torch.tensor(float(r))                   # alpha = rank: scale 1
        kept += float((S[:r] ** 2).sum())
        total += float((S ** 2).sum())
        if i % 32 == 0:
            logger.info(f"  {i + 1}/{len(net.wrapped)} {full} ({time.time() - t0:.0f} s)")
    energy = kept / max(total, 1e-12)
    meta = {MARKER: "1", GAIN_KEY: f"{gain:g}", "ss_fizgig_model_lora_source": os.path.basename(dit_path),
            "ss_fizgig_model_lora_energy": f"{energy:.4f}", "ss_network_dim": str(rank),
            "ss_network_alpha": str(float(rank)), "ss_network_module": "fizgig.model_eq",
            "modelspec.architecture": getattr(desc, "modelspec_arch", "") or desc.key}
    save_file(sd, out_path, metadata=meta)
    logger.info(f"saved {out_path}: {len(sd) // 3} Linears at rank {rank}, {energy:.1%} of the weights' energy, "
                f"gain {gain:g}, {os.path.getsize(out_path) / 2 ** 30:.2f} GB, {time.time() - t0:.0f} s")
    return out_path


def main():
    p = argparse.ArgumentParser(description="A family's own weights as a Model EQ LoRA (Repair Studio)")
    p.add_argument("--family", required=True)
    p.add_argument("--dit", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--gain", type=float, default=0.25, help="slider 0 / 2 = -gain / +gain of the top directions")
    p.add_argument("--device", default="cpu", help="where the SVDs run (cuda is much faster)")
    a = p.parse_args()
    extract(a.family, a.dit, a.out, a.rank, a.gain, a.device)


if __name__ == "__main__":
    main()
