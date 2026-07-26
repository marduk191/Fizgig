"""Activation-based LoRA extraction.

Runs the model with a source LoRA enabled, captures per-layer activation deltas
at specific timestep ranges, SVD-decomposes to a target rank, and saves as a
new standard safetensors LoRA.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from fizgig.klein.inference import KleinInferencePipeline
from fizgig.klein.position import prc_img, prc_txt

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for a LoRA extraction run."""
    source_lora_path: str
    output_lora_path: str
    target_rank: int = 2
    timestep_range: tuple = (0.0, 1.0)  # (low, high) — full range = no specialization
    include_blocks: list = field(default_factory=lambda: ["all"])
    # "all", "composition", "identity", "style", or "custom" (with custom_blocks)
    custom_blocks: Optional[list] = None
    # When include_blocks=["custom"], this is the list of block names to enable,
    # e.g. ["double_blocks.3", "single_blocks.5", "single_blocks.7"]
    num_samples: int = 16
    prompt: str = "a photo"
    width: int = 1024
    height: int = 1024
    multiplier: float = 1.0
    source_multiplier: float = 1.0  # Multiplier for source LoRA during extraction
    seed: Optional[int] = None


@dataclass
class ExtractionResult:
    """Result of a LoRA extraction run."""
    output_path: str
    num_layers_extracted: int
    target_rank: int
    total_params: int
    source_lora_path: str
    included_blocks: list
    timestep_range: tuple
    elapsed_seconds: float


    # Block-category patterns — mirrors KleinInferencePipeline.BLOCK_PATTERNS
    # so weight-only extraction can filter without loading the pipeline.
_BLOCK_CATEGORY_PATTERNS = {
    "style_composition": [
        (r"double_blocks_\d_", 1.0),
        (r"single_blocks_[01]_", 1.0),
        (r"single_blocks_2_", 0.5),
    ],
    "identity": [
        (r"single_blocks_(1[0-6]|[1-9])_", 1.0),
    ],
    "details": [
        (r"single_blocks_(1[2-9]|2[0-3])_", 1.0),
    ],
}

import re as _re


def _key_matches_categories(key: str, categories: list, custom_blocks=None) -> Optional[float]:
    """Return the multiplier if `key` belongs to any of `categories`, else None.

    For "all" returns 1.0 for every key.  For "custom", matches custom_blocks.
    """
    if "all" in categories:
        return 1.0
    if "custom" in categories and custom_blocks:
        for block_name in custom_blocks:
            pattern = block_name.replace(".", "_") + "_"
            if pattern in key:
                return 1.0
        return None
    for cat in categories:
        for pattern, mult in _BLOCK_CATEGORY_PATTERNS.get(cat, []):
            if _re.search(pattern, key):
                return mult
    return None


class LoRAExtractor:
    """Extract a low-rank LoRA from an existing LoRA using activation-based SVD.

    Usage:
        pipeline = KleinInferencePipeline()
        pipeline.load_models(...)

        extractor = LoRAExtractor(pipeline)
        result = extractor.extract(ExtractionConfig(
            source_lora_path="source.safetensors",
            output_lora_path="extracted.safetensors",
            target_rank=2,
            include_blocks=["style"],  # extract only style blocks
        ))
    """

    def __init__(self, pipeline: KleinInferencePipeline):
        self.pipeline = pipeline

    @staticmethod
    def extract_weight_only(config: ExtractionConfig, progress_callback=None) -> ExtractionResult:
        """Pure weight SVD — no pipeline, no GPU models, no forward passes.

        Loads the source LoRA safetensors directly, computes the dense delta
        (up @ down * scale) per module, SVDs to target_rank, and saves.
        Supports standard LoRA, LoKR, and LoHa source formats.
        """
        from safetensors.torch import load_file, save_file
        from fizgig.networks.lora import ensure_kohya_lora_state_dict

        start_time = time.time()

        sd_raw = load_file(config.source_lora_path)
        sd = ensure_kohya_lora_state_dict(sd_raw)

        # Group keys by module name
        modules: dict = {}  # module_name -> {suffix: tensor}
        for key, tensor in sd.items():
            if "." not in key:
                continue
            mod, _, suffix = key.partition(".")
            modules.setdefault(mod, {})[suffix] = tensor

        # Build source weights with block filtering + multiplier
        source_weights: dict = {}  # module_name -> (W_source, multiplier_applied)
        for mod_name, mod_keys in modules.items():
            mult = _key_matches_categories(mod_name, config.include_blocks, config.custom_blocks)
            if mult is None:
                continue

            up = mod_keys.get("lora_up.weight")
            down = mod_keys.get("lora_down.weight")

            if up is not None and down is not None:
                # Standard LoRA
                rank = int(down.shape[0])
                alpha_t = mod_keys.get("alpha")
                alpha = float(alpha_t.item()) if alpha_t is not None else float(rank)
                scale = alpha / max(rank, 1)
                W = up.float() @ down.float() * scale * mult
            elif mod_keys.get("lokr_w1") is not None or mod_keys.get("lokr_w1_a") is not None:
                # LoKR — scale mirrors LoKRInfModule (incl. Comfy-Realtime sentinel)
                from fizgig.networks.lora import lycoris_scale_from_keys
                if mod_keys.get("lokr_w1_a") is not None:
                    w1 = (mod_keys["lokr_w1_a"].float() @ mod_keys["lokr_w1_b"].float())
                else:
                    w1 = mod_keys["lokr_w1"].float()
                if mod_keys.get("lokr_w2_a") is not None:
                    w2 = (mod_keys["lokr_w2_a"].float() @ mod_keys["lokr_w2_b"].float())
                else:
                    w2 = mod_keys["lokr_w2"].float()
                from fizgig.utils.device import gpu_kron
                W = gpu_kron(w1, w2) * lycoris_scale_from_keys(mod_keys) * mult
            elif mod_keys.get("hada_w1_a") is not None:
                # LoHa — scale mirrors LoHaInfModule
                from fizgig.networks.lora import lycoris_scale_from_keys
                W1 = mod_keys["hada_w1_a"].float() @ mod_keys["hada_w1_b"].float()
                W2 = mod_keys["hada_w2_a"].float() @ mod_keys["hada_w2_b"].float()
                W = (W1 * W2) * lycoris_scale_from_keys(mod_keys) * mult
            else:
                continue

            # Documented CLI flag, previously silent on this path (honoured only by
            # --blocks custom on the activation path). The delta is linear in it.
            if config.source_multiplier != 1.0:
                W = W * config.source_multiplier
            source_weights[mod_name] = W

        logger.info(f"Pure weight SVD (rank {config.target_rank}) on {len(source_weights)} modules")

        # SVD each module
        output_sd: dict = {}
        total_params = 0

        items = list(source_weights.items())
        for idx, (mod_name, W_source) in enumerate(tqdm(items, desc="SVD")):
            if progress_callback:
                progress_callback("svd", idx, len(items))

            if W_source.abs().max().item() < 1e-10:
                continue

            try:
                from fizgig.utils.device import gpu_svd
                U, S, Vt = gpu_svd(W_source)
            except Exception as e:
                logger.warning(f"SVD failed for {mod_name}: {e}")
                continue

            R = min(config.target_rank, S.shape[0])
            sqrt_S = torch.sqrt(S[:R])
            lora_up = U[:, :R] * sqrt_S.unsqueeze(0)
            lora_down = sqrt_S.unsqueeze(1) * Vt[:R, :]

            if not torch.isfinite(lora_up).all() or not torch.isfinite(lora_down).all():
                logger.warning(f"{mod_name}: non-finite values, replacing with zeros")
                lora_up = torch.nan_to_num(lora_up, nan=0.0, posinf=0.0, neginf=0.0)
                lora_down = torch.nan_to_num(lora_down, nan=0.0, posinf=0.0, neginf=0.0)

            output_sd[f"{mod_name}.lora_down.weight"] = lora_down.contiguous().to(torch.float16)
            output_sd[f"{mod_name}.lora_up.weight"] = lora_up.contiguous().to(torch.float16)
            output_sd[f"{mod_name}.alpha"] = torch.tensor(float(R), dtype=torch.float16)
            total_params += lora_up.numel() + lora_down.numel()

        # Save
        os.makedirs(os.path.dirname(os.path.abspath(config.output_lora_path)) or ".", exist_ok=True)
        metadata = {
            "ss_base_model_version": "klein_9b",
            "ss_network_module": "fizgig.networks.lora_klein",
            "ss_network_dim": str(config.target_rank),
            "ss_network_alpha": str(float(config.target_rank)),
            "ss_source_lora": os.path.basename(config.source_lora_path),
            "ss_extraction_blocks": ",".join(config.include_blocks),
            "ss_extraction_timesteps": f"{config.timestep_range[0]:.2f}-{config.timestep_range[1]:.2f}",
            "ss_extraction_samples": "0",
            "ss_extraction_prompt": config.prompt,
        }
        save_file(output_sd, config.output_lora_path, metadata=metadata)

        elapsed = time.time() - start_time
        num_layers = len([k for k in output_sd if k.endswith(".lora_down.weight")])
        logger.info(f"Saved extracted LoRA: {config.output_lora_path}")
        logger.info(f"  {num_layers} layers extracted at rank {config.target_rank}")
        logger.info(f"  Total params: {total_params:,}")
        logger.info(f"  Elapsed: {elapsed:.1f}s")

        return ExtractionResult(
            output_path=config.output_lora_path,
            num_layers_extracted=num_layers,
            target_rank=config.target_rank,
            total_params=total_params,
            source_lora_path=config.source_lora_path,
            included_blocks=config.include_blocks,
            timestep_range=config.timestep_range,
            elapsed_seconds=elapsed,
        )

    def extract(self, config: ExtractionConfig, progress_callback=None) -> ExtractionResult:
        """Run extraction and save output LoRA.

        Args:
            config: ExtractionConfig with all parameters.
            progress_callback: optional fn(stage, current, total) for UI updates.
        """
        start_time = time.time()
        pipeline = self.pipeline

        if not pipeline.is_loaded:
            raise RuntimeError("Pipeline models not loaded.")

        device = pipeline.device

        # Load source LoRA
        pipeline.load_lora_for_profiling(config.source_lora_path, multiplier=config.source_multiplier)

        # Apply block filter
        if config.include_blocks == ["custom"] and config.custom_blocks:
            # Custom per-block selection
            pipeline._lora_network.set_enabled(False)
            for block_name in config.custom_blocks:
                # Convert block_name (e.g. "single_blocks.5") to pattern matching
                # the LoRA key format (e.g. "single_blocks_5_")
                pattern = block_name.replace(".", "_") + "_"
                pipeline._lora_network.set_module_enabled_by_pattern(pattern, True)
        else:
            pipeline.enable_blocks(config.include_blocks)
            # enable_blocks resets every module multiplier to its category preset —
            # re-apply the documented source multiplier ON TOP (it was silently ignored
            # on this path; only --blocks custom honoured it).
            if config.source_multiplier != 1.0:
                for lora in pipeline._lora_network.unet_loras:
                    lora.multiplier *= config.source_multiplier
        num_enabled = sum(1 for lora in pipeline._lora_network.unet_loras if lora.enabled)
        logger.info(f"Enabled {num_enabled} LoRA modules for extraction")

        # Setup extraction capture
        pipeline.setup_extraction_capture()
        pipeline.clear_extraction_data()

        # Encode prompt once
        logger.info(f"Encoding prompt: '{config.prompt}'")
        ctx_vec, _ = pipeline.encode_prompt(config.prompt, " ")
        ctx = ctx_vec.to(device=device, dtype=torch.bfloat16)
        ctx, ctx_ids = prc_txt(ctx)

        # Prepare block swap if active
        if hasattr(pipeline.dit, '_offloader') and pipeline.dit._offloader is not None:
            pipeline.dit._offloader.set_forward_only(True)
            pipeline.dit.prepare_block_swap_before_forward()

        generator = torch.Generator(device=device)
        if config.seed is not None:
            generator.manual_seed(config.seed)

        packed_h, packed_w = config.height // 16, config.width // 16
        t_low, t_high = config.timestep_range

        # Run forward passes and accumulate
        logger.info(f"Running {config.num_samples} forward passes at {config.width}x{config.height}, "
                    f"timestep range {t_low:.2f}-{t_high:.2f}")

        for i in range(config.num_samples):
            if progress_callback:
                progress_callback("sampling", i, config.num_samples)

            # Random timestep in the target range
            t = t_low + (t_high - t_low) * torch.rand(1).item()

            # Generate noisy latent
            clean = torch.randn(
                1, 128, packed_h, packed_w,
                generator=generator if config.seed is not None else None,
                device=device, dtype=torch.bfloat16,
            )
            noise = torch.randn_like(clean)
            noisy_latent = (1.0 - t) * clean + t * noise

            x, x_ids = prc_img(noisy_latent)
            t_vec = torch.full((1,), t, device=device, dtype=torch.bfloat16)
            guidance_vec = torch.full((1,), 1.0, device=device, dtype=torch.bfloat16)

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pipeline.dit(
                    x=x, x_ids=x_ids, timesteps=t_vec,
                    ctx=ctx, ctx_ids=ctx_ids, guidance=guidance_vec,
                )

        # Retrieve captured data
        extraction_data = pipeline.get_extraction_data()
        logger.info(f"Captured activation data for {len(extraction_data)} LoRA modules")

        # Collect source weights per enabled layer, plus activation data by layer.
        # Each LoRA variant computes its dense delta differently — standard LoRA
        # is up@down, LoKR is kron(w1,w2), LoHa is (W1 ⊙ W2). The extractor
        # treats all of them as a single W_source and runs the same SVD pipeline,
        # so LyCORIS primaries can be re-extracted down to a standard low-rank LoRA.
        source_weights = {}  # lora_name -> W_source (dense delta, float32, CPU)
        skipped_variants = []
        for lora in pipeline._lora_network.unet_loras:
            if not lora.enabled:
                continue
            with torch.no_grad():
                cls_name = type(lora).__name__
                if cls_name in ("LoRAInfModule", "LoRAModule"):
                    src_up = lora.lora_up.weight.float()
                    src_down = lora.lora_down.weight.float()
                    W_source = (src_up @ src_down * lora.scale * lora.multiplier).cpu()
                elif cls_name == "LoKRInfModule":
                    w1 = (lora.lokr_w1_a @ lora.lokr_w1_b).float() if lora.w1_decomposed else lora.lokr_w1.float()
                    w2 = (lora.lokr_w2_a @ lora.lokr_w2_b).float() if lora.w2_decomposed else lora.lokr_w2.float()
                    from fizgig.utils.device import gpu_kron
                    W_source = gpu_kron(w1.cpu(), w2.cpu()) * lora.scale * lora.multiplier
                elif cls_name == "LoHaInfModule":
                    W1 = (lora.hada_w1_a.float() @ lora.hada_w1_b.float())
                    W2 = (lora.hada_w2_a.float() @ lora.hada_w2_b.float())
                    W_source = ((W1 * W2) * lora.scale * lora.multiplier).cpu()
                else:
                    skipped_variants.append((lora.lora_name, cls_name))
                    continue
                source_weights[lora.lora_name] = W_source
        if skipped_variants:
            logger.warning(
                f"Extractor skipped {len(skipped_variants)} modules of unrecognized variant: "
                f"{set(v for _, v in skipped_variants)}"
            )

        # Remove capture hooks
        pipeline.remove_extraction_capture()

        # Activation-weighted SVD of SOURCE weights (hybrid approach):
        # - Source weights are deterministic (no sampling noise)
        # - Input-distribution weights (from activation sampling) emphasize
        #   directions that matter at the target timestep range
        # - SVD picks the best rank-R approximation under that weighting
        mode = "activation-weighted" if config.num_samples > 0 else "pure weight"
        logger.info(f"Running {mode} SVD (rank {config.target_rank}) on source weights")
        output_sd = {}
        total_params = 0

        # Iterate over source weights (not extraction_data) since we always have those
        for idx, (lora_name, W_source) in enumerate(tqdm(source_weights.items(), desc="SVD")):
            if progress_callback:
                progress_callback("svd", idx, len(source_weights))

            data = extraction_data.get(lora_name)
            if data is None or data["samples"] == 0:
                # No activation data captured — fall back to plain weight SVD
                c_xx_diag = None
            else:
                c_xx_diag = data["c_xx"].float()  # (in_dim,) per-dim input variance

            if W_source.abs().max().item() < 1e-10:
                continue

            # Build input-weighting: sqrt of per-dim variance from sampled activations
            # If no activation data, use uniform weighting (plain weight SVD).
            if c_xx_diag is not None:
                eps = c_xx_diag.max().item() * 1e-6 + 1e-8
                sqrt_d = torch.sqrt(c_xx_diag.clamp(min=eps))  # (in_dim,)
                # Weighted source: emphasize input dims that matter at this timestep
                W_weighted = W_source * sqrt_d.unsqueeze(0)
            else:
                sqrt_d = None
                W_weighted = W_source

            try:
                from fizgig.utils.device import gpu_svd
                U, S, Vt = gpu_svd(W_weighted)
            except Exception as e:
                logger.warning(f"SVD failed for {lora_name}: {e}")
                continue

            R = min(config.target_rank, S.shape[0])
            U_r = U[:, :R]        # (out_dim, R)
            S_r = S[:R]           # (R,)
            Vt_r = Vt[:R, :]      # (R, in_dim) — in input-weighted space

            sqrt_S = torch.sqrt(S_r)
            lora_up = U_r * sqrt_S.unsqueeze(0)                           # (out_dim, R)
            if sqrt_d is not None:
                # Un-weight: divide down projection by sqrt(d) to reverse the input weighting
                lora_down = (sqrt_S.unsqueeze(1) * Vt_r) / sqrt_d.unsqueeze(0)  # (R, in_dim)
            else:
                lora_down = sqrt_S.unsqueeze(1) * Vt_r

            # Guard against non-finite
            if not torch.isfinite(lora_up).all() or not torch.isfinite(lora_down).all():
                logger.warning(f"{lora_name}: non-finite values, replacing with zeros")
                lora_up = torch.nan_to_num(lora_up, nan=0.0, posinf=0.0, neginf=0.0)
                lora_down = torch.nan_to_num(lora_down, nan=0.0, posinf=0.0, neginf=0.0)

            output_sd[f"{lora_name}.lora_down.weight"] = lora_down.contiguous().to(torch.float16)
            output_sd[f"{lora_name}.lora_up.weight"] = lora_up.contiguous().to(torch.float16)
            output_sd[f"{lora_name}.alpha"] = torch.tensor(float(R), dtype=torch.float16)

            total_params += lora_up.numel() + lora_down.numel()

        # Save
        os.makedirs(os.path.dirname(os.path.abspath(config.output_lora_path)) or ".", exist_ok=True)

        metadata = {
            "ss_base_model_version": "klein_9b",
            "ss_network_module": "fizgig.networks.lora_klein",
            "ss_network_dim": str(config.target_rank),
            "ss_network_alpha": str(float(config.target_rank)),
            "ss_source_lora": os.path.basename(config.source_lora_path),
            "ss_extraction_blocks": ",".join(config.include_blocks),
            "ss_extraction_timesteps": f"{config.timestep_range[0]:.2f}-{config.timestep_range[1]:.2f}",
            "ss_extraction_samples": str(config.num_samples),
            "ss_extraction_prompt": config.prompt,
        }

        from safetensors.torch import save_file
        save_file(output_sd, config.output_lora_path, metadata=metadata)

        elapsed = time.time() - start_time
        num_layers = len([k for k in output_sd if k.endswith(".lora_down.weight")])
        logger.info(f"Saved extracted LoRA: {config.output_lora_path}")
        logger.info(f"  {num_layers} layers extracted at rank {config.target_rank}")
        logger.info(f"  Total params: {total_params:,}")
        logger.info(f"  Elapsed: {elapsed:.1f}s")

        return ExtractionResult(
            output_path=config.output_lora_path,
            num_layers_extracted=num_layers,
            target_rank=config.target_rank,
            total_params=total_params,
            source_lora_path=config.source_lora_path,
            included_blocks=config.include_blocks,
            timestep_range=config.timestep_range,
            elapsed_seconds=elapsed,
        )
