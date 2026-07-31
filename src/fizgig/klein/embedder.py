"""Qwen3-8B text embedder for Klein 9B.

Loads the Qwen3-8B language model and extracts hidden states from layers 9, 18, 27
to produce text embeddings for the Klein DiT.
"""

import json
import logging
from typing import Optional, Union

import torch
import torch.nn as nn
from einops import rearrange
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen2Tokenizer
from accelerate import init_empty_weights

from fizgig.utils.safetensors import load_split_weights

logger = logging.getLogger(__name__)

# Klein 9B uses Qwen3-8B with hidden states from these layers
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
MAX_LENGTH = 512


QWEN3_8B_CONFIG_JSON = """{
  "architectures": ["Qwen3ForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "max_position_embeddings": 40960,
  "max_window_layers": 36,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}"""


class Qwen3Embedder(nn.Module):
    """Text embedder using Qwen3-8B for Klein 9B.

    Extracts hidden states from layers 9, 18, 27 and concatenates them
    to produce (B, 512, 12288) text embeddings.
    """

    def __init__(self, tokenizer: Qwen2Tokenizer, model: Qwen3ForCausalLM):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = MAX_LENGTH

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    def to(self, *args, **kwargs):
        return self.model.to(*args, **kwargs)

    def forward(self, txt: list[str]):
        all_input_ids = []
        all_attention_masks = []

        for prompt in txt:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.model.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.model.device)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Stack hidden states from target layers, then apply Qwen3's final
        # RMSNorm. This matches ComfyUI's Klein text-encoder pipeline
        # (comfy/text_encoders/llama.py:final_layer_norm_intermediate=True).
        # Without this norm, raw hidden states have enormous outliers
        # (|values| up to 20k) which destabilizes Klein Distilled 4-step
        # inference — the DiT's conditioning produces garbage predictions.
        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
        out = self.model.model.norm(out)
        return rearrange(out, "b c l d -> b l (c d)")  # (B, 512, 12288)


def load_qwen3(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
    tokenizer_id: str = "Qwen/Qwen3-8B",
) -> tuple[Qwen2Tokenizer, Qwen3ForCausalLM]:
    """Load Qwen3-8B model and tokenizer for Klein 9B text encoding.

    Args:
        ckpt_path: Path to Qwen3-8B safetensors checkpoint.
        dtype: Target dtype (bf16 recommended, fp8 supported with patching).
        device: Target device.
        disable_mmap: Disable memory mapping when loading.
        state_dict: Pre-loaded state dict (optional).
        tokenizer_id: HuggingFace tokenizer ID.

    Returns:
        Tuple of (tokenizer, model).
    """
    config = json.loads(QWEN3_8B_CONFIG_JSON)
    config = Qwen3Config(**config)
    with init_empty_weights():
        qwen3 = Qwen3ForCausalLM._from_config(config)

    if state_dict is not None:
        sd = state_dict
    else:
        logger.info(f"Loading Qwen3-8B state dict from {ckpt_path}")
        sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)

    # Tie embedding weights
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]

    info = qwen3.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded Qwen3-8B: {info}")
    qwen3.to(device)

    if dtype is not None:
        if dtype.itemsize == 1:  # FP8 dtypes
            org_dtype = torch.bfloat16
            logger.info(f"Preparing Qwen3-8B for FP8: {dtype} from {org_dtype}")
            qwen3.to(dtype)
            _prepare_qwen3_fp8(qwen3, org_dtype)
        else:
            logger.info(f"Setting Qwen3-8B to dtype: {dtype}")
            qwen3.to(dtype)

    logger.info(f"Loading tokenizer from {tokenizer_id}")
    from fizgig.utils.hf_cache import from_pretrained_cache_first
    tokenizer = from_pretrained_cache_first(Qwen2Tokenizer, tokenizer_id)
    return tokenizer, qwen3


def _prepare_qwen3_fp8(model: Qwen3ForCausalLM, target_dtype: torch.dtype):
    """Patch Qwen3 for FP8 inference: fix Embedding and RMSNorm dtypes."""
    def rms_norm_forward_hook(module):
        def forward(hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + module.variance_epsilon)
            return (module.weight.to(torch.float32) * hidden_states.to(torch.float32)).to(input_dtype)
        return forward

    for module in model.modules():
        if module.__class__.__name__ == "Embedding":
            module.to(target_dtype)
        if module.__class__.__name__ == "Qwen3RMSNorm":
            module.forward = rms_norm_forward_hook(module)
