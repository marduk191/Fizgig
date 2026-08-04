"""Cache Qwen3-VL-32B text-encoder states (raw layer-50 output) for MiniMax H3 image training.

    python src/fizgig/scripts/minimax_cache_text.py --dataset_config config.toml --text_encoder path/to/qwen3vl_32b_bf16
"""

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fizgig.dataset.config import (
    BlueprintGenerator,
    ConfigSanitizer,
    generate_dataset_group_by_blueprint,
    load_user_config,
)
from fizgig.scripts.cache_text import prepare_cache_files_and_paths, process_batches, post_process
from fizgig.minimax.embedder import load_minimax_h3_te
from fizgig.minimax.caching import encode_and_save_text
from fizgig.training.metadata import ARCHITECTURE_MINIMAX

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache Qwen3-VL-32B text-encoder states for MiniMax H3 training")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset config .toml file")
    parser.add_argument("--text_encoder", type=str, required=True, help="Path to the bf16 Qwen3-VL-32B safetensors")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Captions per text-encoder forward. The nvfp4-resident encoder "
                             "dequantizes 351 weights per FORWARD, so this is the dial that "
                             "matters: 1 caption/forward costs ~1.2 s, 16 costs barely more. "
                             "Right-padding makes batching exactly equivalent to one-at-a-time "
                             "for this causal stack (tests/diag_batch_encode.py).")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    parser.add_argument("--keep_cache", action="store_true", help="Keep stale cache files")
    parser.add_argument("--no_quantize", action="store_true", help="Load the TE in bf16 (no NF4) — needs ~66 GB VRAM")
    return parser


def main():
    args = setup_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    blueprint_gen = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Loading dataset config from {args.dataset_config}")
    user_config = load_user_config(args.dataset_config)
    blueprint = blueprint_gen.generate(user_config, args, architecture=ARCHITECTURE_MINIMAX)
    datasets = generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets

    all_files, all_paths = prepare_cache_files_and_paths(datasets)

    logger.info(f"Loading Qwen3-VL-32B text encoder from {args.text_encoder}")
    encoder = load_minimax_h3_te(args.text_encoder, device=device, compute_dtype=torch.bfloat16,
                                 quantize=not args.no_quantize)

    process_batches(args, datasets, all_files, all_paths, lambda batch: encode_and_save_text(encoder, batch))

    _uncond = encoder.encode("")[0].detach().cpu().contiguous()      # (L=1, 5120)
    del encoder
    post_process(datasets, all_files, all_paths, args.keep_cache)

    # One UNCONDITIONAL embed per cache dir (empty prompt -> the TE's single-pad fallback), so
    # caption dropout has something to swap in for ~5% of steps.
    #
    # Written AFTER post_process, deliberately. Its filename ends `_minimaxh3_te.safetensors`, so
    # it matches the glob post_process uses to find stale caches — and since no dataset ITEM
    # claims that path, a second caching pass would delete the one the first pass wrote. The GUI
    # re-runs caching on every launch, so the file existed only until the next run and
    # ss_caption_dropout silently read 0.
    from safetensors.torch import save_file as _save_file
    _n = 0
    for ds in datasets:
        _dir = getattr(ds, "cache_directory", None)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
            _save_file({"hidden_states": _uncond,
                        "attention_mask": torch.ones(_uncond.shape[0], dtype=torch.bool)},
                       os.path.join(_dir, f"uncond_{ARCHITECTURE_MINIMAX}_te.safetensors"))
            _n += 1
    logger.info(f"[uncond] cached the empty-prompt embed for caption dropout "
                f"({tuple(_uncond.shape)}) in {_n} cache dir(s)")


if __name__ == "__main__":
    main()
