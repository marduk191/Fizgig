"""Flux 3 Image Klein text conditioning: Qwen3-VL-4B-Instruct, eight hidden states stacked per token.

Follows the reference (black-forest-labs/flux-3-image-klein src/flux3/text_encoder.py):
* the prompt as one user turn through the chat template, add_generation_prompt=True
* right padding to a multiple of 80 tokens (max 4096); the pad tokens are kept, the DiT attends them all
* position ids = the mask's cumsum, given as 4 rows (Qwen3-VL's multimodal RoPE)
* hidden_states at layers 4, 8, ..., 32 (no final norm involved), stacked to (L, 8 * 2560 = 20480)

Weights: Comfy-Org's single file (text_encoders/qwen3vl_4b_bf16.safetensors) or official shards, loaded by Krea 2's
Qwen3-VL-4B loader (same model, vendored config); tokenizer + chat template from the copy Fizgig bundles. The LM head
and vision tower are dropped: only the text path encodes.
"""
import logging
import math

import torch

logger = logging.getLogger(__name__)

OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
MAX_LENGTH = 4096
PAD_MULTIPLE = 80


class Flux3TextEncoder:
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16, int8=False):
        from transformers import AutoProcessor

        from fizgig.krea2.embedder import (QWEN3_VL_4B_INSTRUCT_REPO_ID, _load_qwen3_vl_model,
                                           _resolve_tokenizer_source)
        from fizgig.utils.hf_cache import from_pretrained_cache_first

        self.device = torch.device(device)
        model = _load_qwen3_vl_model(model_path, dtype=dtype, device="cpu")
        model.lm_head = None
        model.model.visual = None
        # the loader's model.to(bf16) also casts RoPE's inv_freq; the reference keeps it fp32 (transformers' init).
        # In bf16 the rotary angles drift with position: measured 1.25% off the reference at 160 tokens.
        rotary = model.model.language_model.rotary_emb
        inv_freq, _ = rotary.rope_init_fn(rotary.config, "cpu")
        rotary.register_buffer("inv_freq", inv_freq.float(), persistent=False)
        rotary.original_inv_freq = rotary.inv_freq
        if int8:
            from fizgig.qwen_image21.embedder import _int8_weights
            n = _int8_weights(model, "language_model.layers.", self.device)
            logger.info(f"[text encoder] 8-bit weights: {n} language-model Linears (low-VRAM card); matmuls stay bf16")
        self.model = model.to(self.device).eval().requires_grad_(False)
        source = _resolve_tokenizer_source(model_path, QWEN3_VL_4B_INSTRUCT_REPO_ID)
        self.processor = from_pretrained_cache_first(AutoProcessor, source)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "right"

    def unload(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_one(self, prompt: str) -> torch.Tensor:
        """prompt -> tokens (L, 20480) bf16 on CPU, L a multiple of 80. The text ids (0, 0, 0, position) are rebuilt
        from L by the sampler, so they are not returned."""
        text = self.processor.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True)
        real = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)["input_ids"].shape[1]
        length = min(math.ceil(real / PAD_MULTIPLE) * PAD_MULTIPLE, MAX_LENGTH)
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        mask = inputs["attention_mask"].to(self.device)
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
        # the base model, not the LM wrapper: the same hidden states without vocabulary-wide logits
        out = self.model.model(input_ids=inputs["input_ids"].to(self.device), attention_mask=mask,
                               position_ids=positions[None].expand(4, -1, -1), output_hidden_states=True,
                               use_cache=False)
        stacked = torch.stack([out.hidden_states[k] for k in OUTPUT_LAYERS], dim=1)  # (1, 8, L, 2560)
        return stacked.permute(0, 2, 1, 3).reshape(length, -1).to(torch.bfloat16).cpu()

    def encode(self, prompts):
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        return [self.encode_one(p) for p in prompts]
