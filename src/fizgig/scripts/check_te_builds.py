"""Check other builds of a family's text encoder against its default one: does the family's own loader take each, and
how close is its conditioning? The numbers a description's text_encoder_builds records (docs/drivers/README.md,
"Text encoder builds").

    python -m fizgig.scripts.check_te_builds --family krea2 --reference <default file> \
        --build <file or repo:path-in-repo> [--build ...] [--download_dir <folder>]

For each build: whether it loads, its SHA-256 (a computer can then tell it holds that build without asking Hugging
Face), the cosine of its conditioning to the reference's on CHECK_PROMPTS (the real tokens only, where the encoder
gives a mask), the worst token's cosine, and the encode time and peak VRAM. A repo:path build is downloaded into
--download_dir first.
"""
import argparse
import hashlib
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

CHECK_PROMPTS = [
    "a woman smiling at a cafe table, a close-up photo, soft window light",
    "a fishing village on a rocky coast at dawn, mist over the water, wide shot",
    "an oil painting of a fox in a snowy forest",
    ("a man in a worn leather jacket leaning against a red brick wall at night, neon signs reflected in the wet "
     "street, shallow depth of field, cinematic colour grade, 35mm film grain, his breath visible in the cold air"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fetch(spec, folder):
    """A local file, or repo:path-in-repo downloaded into `folder`."""
    if os.path.isfile(spec):
        return spec
    repo, _, path = spec.partition(":")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=path, local_dir=folder)


def encode(driver, path, device):
    """The conditioning for CHECK_PROMPTS with one build, the encode time and the peak VRAM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    te = driver.load_text_encoder(path, device)
    t = time.time()
    with torch.no_grad():
        conds = driver.encode_text(te, CHECK_PROMPTS)
    torch.cuda.synchronize()
    secs = time.time() - t
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    driver.unload_text_encoder(te)
    return conds, secs, peak


def _rows(cond):
    """{key: (tokens, features) float32} per conditioning tensor, masked to the real tokens where there's a mask."""
    masks = {k[:-len("attention_mask")]: v.bool() for k, v in cond.items()
             if k.endswith("attention_mask") and torch.is_tensor(v)}
    out = {}
    for k, v in cond.items():
        if not torch.is_tensor(v) or not v.is_floating_point():
            continue
        v = v.float()
        m = next((m for base, m in masks.items() if k.startswith(base) and v.dim() and v.shape[0] == m.shape[0]), None)
        if m is not None:
            v = v[m]
        out[k] = v.reshape(v.shape[0], -1) if v.dim() > 1 else v.reshape(1, -1)
    return out


def compare(ref, got):
    """(mean cosine, worst token's cosine) of a build's conditioning to the reference's, over every prompt."""
    whole, worst = [], 1.0
    for a, b in zip(ref, got):
        ra, rb = _rows(a), _rows(b)
        for k in ra:
            if k not in rb or ra[k].shape != rb[k].shape:
                return None, f"{k}: shape {tuple(rb[k].shape) if k in rb else 'missing'} vs {tuple(ra[k].shape)}"
            whole.append(torch.nn.functional.cosine_similarity(ra[k].flatten(), rb[k].flatten(), dim=0).item())
            worst = min(worst, torch.nn.functional.cosine_similarity(ra[k], rb[k], dim=1).min().item())
    return min(1.0, sum(whole) / len(whole)), min(1.0, worst)       # float32 rounding can put a match a hair over 1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--family", required=True)
    p.add_argument("--reference", required=True, help="the family's default text encoder file")
    p.add_argument("--build", action="append", default=[], help="a file, or repo:path-in-repo to download")
    p.add_argument("--download_dir", default="")
    a = p.parse_args()
    from fizgig.families.registry import get
    desc = get(a.family)
    if desc is None:
        raise SystemExit(f"No family {a.family!r}")
    driver, device = desc.load_driver(), torch.device("cuda")
    folder = a.download_dir or os.getcwd()
    reference = fetch(a.reference, folder)
    ref, secs, peak = encode(driver, reference, device)
    print(f"reference {os.path.basename(reference)}: {secs:.1f} s, peak {peak:.1f} GB, sha256 {sha256(reference)}",
          flush=True)
    for spec in a.build:
        try:
            path = fetch(spec, folder)
            got, secs, peak = encode(driver, path, device)
            cos, worst = compare(ref, got)
            size = os.path.getsize(path) / 1e9
            res = (f"cosine {cos:.5f}, worst token {worst:.4f}" if cos is not None else f"doesn't match: {worst}")
            print(f"build {spec}: loads; {res}; {size:.2f} GB, {secs:.1f} s, peak {peak:.1f} GB, "
                  f"sha256 {sha256(path)}", flush=True)
        except Exception as e:
            print(f"build {spec}: FAILS - {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
