"""ACE-Step 1.5's music model (the DiT, its condition encoder and the lyric / timbre encoders), adapted from the official
ACE-Step 1.5 repository (github.com/ace-step/ACE-Step-1.5, acestep/models/base/modeling_acestep_v15_base.py; MIT /
Apache-2.0).

Changes for Fizgig: plain modules instead of transformers' PreTrainedModel, PyTorch SDPA attention with the same
sliding-window mask, no KV cache, gradient checkpointing on the DiT layers. The audio tokenizer / detokenizer (the
cover path's 5 Hz hint) is a separate CoverGuide, read from the same file only when a cover render needs it. Module names
are the original's; AceStep15 roots them at the decoder, so LoRA keys match ComfyUI's ("base_model.model.layers....").
"""
import json
from dataclasses import dataclass, field, replace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding, apply_rotary_pos_emb


@dataclass
class AceConfig:
    """ACE-Step 1.5 XL (acestep-v15-xl-turbo / -xl-sft config.json): a 2560-wide, 32-layer DiT over a 2048-wide
    condition encoder."""
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    encoder_hidden_size: int = 2048
    encoder_intermediate_size: int = 6144
    encoder_num_attention_heads: int = 16
    encoder_num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_position_embeddings: int = 32768
    sliding_window: int = 128
    use_sliding_window: bool = True
    layer_types: list = field(default_factory=lambda: ["sliding_attention", "full_attention"] * 16)
    in_channels: int = 192
    audio_acoustic_hidden_dim: int = 64
    patch_size: int = 2
    text_hidden_dim: int = 1024
    timbre_hidden_dim: int = 64
    num_lyric_encoder_hidden_layers: int = 8
    num_timbre_encoder_hidden_layers: int = 4
    timestep_mu: float = -0.4
    timestep_sigma: float = 1.0
    is_turbo: bool = False

    @classmethod
    def from_json(cls, path):
        d = json.load(open(path))
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def encoder_config(self) -> "AceConfig":
        """The condition encoder's sizes (its own width, MLP and heads)."""
        return replace(self, hidden_size=self.encoder_hidden_size, intermediate_size=self.encoder_intermediate_size,
                       num_attention_heads=self.encoder_num_attention_heads,
                       num_key_value_heads=self.encoder_num_key_value_heads)

    def qwen3(self) -> Qwen3Config:
        """The transformers config the Qwen3 building blocks (RMSNorm, rotary, MLP) are made from."""
        return Qwen3Config(hidden_size=self.hidden_size, intermediate_size=self.intermediate_size,
                           num_attention_heads=self.num_attention_heads, num_key_value_heads=self.num_key_value_heads,
                           head_dim=self.head_dim, rms_norm_eps=self.rms_norm_eps, rope_theta=self.rope_theta,
                           max_position_embeddings=self.max_position_embeddings, hidden_act="silu")


def _sliding_mask(n, window, device, dtype):
    """Additive bidirectional sliding-window mask (1, 1, n, n): |i - j| <= window."""
    idx = torch.arange(n, device=device)
    keep = (idx[:, None] - idx[None, :]).abs() <= window
    return torch.zeros(n, n, device=device, dtype=dtype).masked_fill(~keep, torch.finfo(dtype).min)[None, None]


def _key_mask(mask, dtype):
    """(B, L) 1 = token, 0 = padding -> additive key mask (B, 1, 1, L)."""
    return torch.zeros(mask.shape, device=mask.device, dtype=dtype).masked_fill(mask == 0, torch.finfo(dtype).min)[:, None, None]


class AceStepAttention(nn.Module):
    def __init__(self, c: AceConfig, layer_idx: int, is_cross_attention=False):
        super().__init__()
        self.head_dim = c.head_dim
        self.n_heads, self.n_kv = c.num_attention_heads, c.num_key_value_heads
        self.is_cross_attention = is_cross_attention
        self.q_proj = nn.Linear(c.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, c.hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=c.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=c.rms_norm_eps)

    def forward(self, x, mask=None, position_embeddings=None, encoder_hidden_states=None):
        b, n, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, n, -1, self.head_dim)).transpose(1, 2)
        src = encoder_hidden_states if self.is_cross_attention and encoder_hidden_states is not None else x
        m = src.shape[1]
        k = self.k_norm(self.k_proj(src).view(b, m, -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(src).view(b, m, -1, self.head_dim).transpose(1, 2)
        if src is x and position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        rep = self.n_heads // self.n_kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(b, n, -1))


class AceStepEncoderLayer(nn.Module):
    def __init__(self, c: AceConfig, layer_idx: int, q3):
        super().__init__()
        self.attention_type = c.layer_types[layer_idx]
        self.self_attn = AceStepAttention(c, layer_idx)
        self.input_layernorm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.mlp = Qwen3MLP(q3)

    def forward(self, x, position_embeddings, mask):
        x = x + self.self_attn(self.input_layernorm(x), mask, position_embeddings)
        return x + self.mlp(self.post_attention_layernorm(x))


class AceStepDiTLayer(nn.Module):
    def __init__(self, c: AceConfig, layer_idx: int, q3):
        super().__init__()
        self.attention_type = c.layer_types[layer_idx]
        self.self_attn_norm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.self_attn = AceStepAttention(c, layer_idx)
        self.cross_attn_norm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.cross_attn = AceStepAttention(c, layer_idx, is_cross_attention=True)
        self.mlp_norm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.mlp = Qwen3MLP(q3)
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, c.hidden_size))

    def forward(self, x, position_embeddings, temb, mask, encoder_hidden_states):
        shift, scale, gate, c_shift, c_scale, c_gate = (self.scale_shift_table + temb).chunk(6, dim=1)
        h = (self.self_attn_norm(x) * (1 + scale) + shift).type_as(x)
        x = (x + self.self_attn(h, mask, position_embeddings) * gate).type_as(x)
        x = x + self.cross_attn(self.cross_attn_norm(x).type_as(x), None, None, encoder_hidden_states)
        h = (self.mlp_norm(x) * (1 + c_scale) + c_shift).type_as(x)
        return (x + self.mlp(h) * c_gate).type_as(x)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, dim, scale=1000):
        super().__init__()
        self.in_channels, self.scale = in_channels, scale
        self.linear_1 = nn.Linear(in_channels, dim)
        self.act1 = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim)
        self.act2 = nn.SiLU()
        self.time_proj = nn.Linear(dim, dim * 6)

    def forward(self, t):
        t = t * self.scale
        half = self.in_channels // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        temb = self.linear_2(self.act1(self.linear_1(emb.to(t.dtype))))
        return temb, self.time_proj(self.act2(temb)).unflatten(1, (6, -1))


class _Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class AceStepDiTModel(nn.Module):
    def __init__(self, c: AceConfig):
        super().__init__()
        q3 = c.qwen3()
        self.c = c
        self.rotary_emb = Qwen3RotaryEmbedding(q3)
        self.layers = nn.ModuleList([AceStepDiTLayer(c, i, q3) for i in range(c.num_hidden_layers)])
        self.patch_size = c.patch_size
        self.proj_in = nn.Sequential(_Transpose(), nn.Conv1d(c.in_channels, c.hidden_size, c.patch_size, c.patch_size),
                                     _Transpose())
        self.time_embed = TimestepEmbedding(256, c.hidden_size)
        self.time_embed_r = TimestepEmbedding(256, c.hidden_size)
        self.condition_embedder = nn.Linear(c.encoder_hidden_size, c.hidden_size)
        self.norm_out = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.proj_out = nn.Sequential(_Transpose(), nn.ConvTranspose1d(c.hidden_size, c.audio_acoustic_hidden_dim,
                                                                       c.patch_size, c.patch_size), _Transpose())
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 2, c.hidden_size))
        self.gradient_checkpointing = False

    def forward(self, x, t, t_r, encoder_hidden_states, context_latents):
        """x (B, T, 64) noisy latents, t / t_r (B,), encoder_hidden_states (B, L, 2048), context_latents (B, T, 128)
        -> the velocity (B, T, 64)."""
        temb_t, proj_t = self.time_embed(t)
        temb_r, proj_r = self.time_embed_r(t - t_r)
        temb, proj = temb_t + temb_r, proj_t + proj_r
        h = torch.cat([context_latents, x], dim=-1)
        n0 = h.shape[1]
        if n0 % self.patch_size:
            h = F.pad(h, (0, 0, 0, self.patch_size - n0 % self.patch_size))
        h = self.proj_in(h)
        enc = self.condition_embedder(encoder_hidden_states)
        n = h.shape[1]
        pos = torch.arange(n, device=h.device)[None]
        pe = self.rotary_emb(h, pos)
        sliding = _sliding_mask(n, self.c.sliding_window, h.device, h.dtype) if self.c.use_sliding_window else None
        for layer in self.layers:
            mask = sliding if layer.attention_type == "sliding_attention" else None
            if self.gradient_checkpointing and torch.is_grad_enabled():
                h = checkpoint(layer, h, pe, proj, mask, enc, use_reentrant=False)
            else:
                h = layer(h, pe, proj, mask, enc)
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        h = (self.norm_out(h) * (1 + scale) + shift).type_as(h)
        return self.proj_out(h)[:, :n0]


class _SeqEncoder(nn.Module):
    """The lyric and timbre encoders' shared body: a projection, bidirectional layers (sliding / full, padding keys
    masked) and a final norm."""

    def _run(self, x, mask, c: AceConfig):
        n = x.shape[1]
        pe = self.rotary_emb(x, torch.arange(n, device=x.device)[None])
        full = _key_mask(mask, x.dtype) if mask is not None else None
        sliding = _sliding_mask(n, c.sliding_window, x.device, x.dtype) if c.use_sliding_window else None
        if sliding is not None and full is not None:
            sliding = sliding + full
        for layer in self.layers:
            x = layer(x, pe, sliding if layer.attention_type == "sliding_attention" else full)
        return self.norm(x)


class AceStepLyricEncoder(_SeqEncoder):
    def __init__(self, c: AceConfig):
        super().__init__()
        q3 = c.qwen3()
        self.c = c
        self.embed_tokens = nn.Linear(c.text_hidden_dim, c.hidden_size)
        self.norm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(q3)
        self.layers = nn.ModuleList([AceStepEncoderLayer(c, i, q3) for i in range(c.num_lyric_encoder_hidden_layers)])

    def forward(self, inputs_embeds, attention_mask):
        return self._run(self.embed_tokens(inputs_embeds), attention_mask, self.c)


class AceStepTimbreEncoder(_SeqEncoder):
    def __init__(self, c: AceConfig):
        super().__init__()
        q3 = c.qwen3()
        self.c = c
        self.embed_tokens = nn.Linear(c.timbre_hidden_dim, c.hidden_size)
        self.norm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(q3)
        self.special_token = nn.Parameter(torch.zeros(1, 1, c.hidden_size))
        self.layers = nn.ModuleList([AceStepEncoderLayer(c, i, q3) for i in range(c.num_timbre_encoder_hidden_layers)])

    cls = True              # XL: a learned CLS token leads the frames and carries the summary (ComfyUI leaves it out)

    def forward(self, refer):
        """refer (B, T, 64) reference audio latents, one per batch item -> (B, 1, 2048): the CLS token after the
        layers, as the XL original reads it."""
        x = self.embed_tokens(refer)
        if self.cls:
            x = torch.cat([self.special_token.expand(x.shape[0], 1, -1).to(x.dtype), x], dim=1)
        return self._run(x, None, self.c)[:, :1]


def pack_sequences(h1, h2, m1, m2):
    """Concatenate two masked sequences with every real token first (the original's pack_sequences)."""
    h = torch.cat([h1, h2], dim=1)
    m = torch.cat([m1, m2], dim=1)
    order = m.argsort(dim=1, descending=True, stable=True)
    h = torch.gather(h, 1, order.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
    lengths = m.sum(dim=1)
    return h, (torch.arange(h.shape[1], device=h.device)[None] < lengths[:, None]).to(m.dtype)


class AceStepConditionEncoder(nn.Module):
    def __init__(self, c: AceConfig):
        super().__init__()
        c = c.encoder_config()
        self.text_projector = nn.Linear(c.text_hidden_dim, c.hidden_size, bias=False)
        self.lyric_encoder = AceStepLyricEncoder(c)
        self.timbre_encoder = AceStepTimbreEncoder(c)

    def forward(self, text, text_mask, lyrics, lyric_mask, refer):
        """text (B, Lt, 1024), lyrics (B, Ll, 1024) with their masks, refer (B, T, 64) -> (B, L, 2048), mask."""
        t = self.text_projector(text)
        ly = self.lyric_encoder(lyrics, lyric_mask)
        tb = self.timbre_encoder(refer)
        tb_mask = torch.ones(tb.shape[:2], device=tb.device, dtype=lyric_mask.dtype)
        h, m = pack_sequences(ly, tb, lyric_mask, tb_mask)
        return pack_sequences(h, t, m, text_mask)


class AceStep15(AceStepDiTModel):
    """The checkpoint's DiT ("decoder") as the root module, with its condition encoder ("encoder") and null condition
    embedding beside the layers - so the LoRA's module paths are the decoder's own ("layers.3.self_attn.q_proj"), as
    ComfyUI's ACE-Step 1.5 LoRA keys address them ("base_model.model.layers.3.self_attn.q_proj"). The 5 Hz LM hint
    path ("tokenizer" / "detokenizer") is not built."""

    def __init__(self, c: AceConfig):
        super().__init__(c)
        self.encoder = AceStepConditionEncoder(c)
        self.null_condition_emb = nn.Parameter(torch.zeros(1, 1, c.encoder_hidden_size))

    @staticmethod
    def checkpoint_key(key):
        """A checkpoint key -> this module's name for it, or None for the LM hint path."""
        if key.startswith("decoder."):
            return key[len("decoder."):]
        if key.startswith("encoder.") or key == "null_condition_emb":
            return key
        return None

    def load_checkpoint(self, sd, **kw):
        """Load a checkpoint state dict (ComfyUI's single file); returns (missing, unexpected) of the kept keys."""
        mine = {}
        for k, v in sd.items():
            name = self.checkpoint_key(k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k)
            if name is not None:
                mine[name] = v
        return self.load_state_dict(mine, strict=False, **kw)

    def condition(self, text, text_mask, lyrics, lyric_mask, refer):
        return self.encoder(text, text_mask, lyrics, lyric_mask, refer)

    def forward(self, x, t, encoder_hidden_states, context_latents, t_r=None):
        return super().forward(x, t, t if t_r is None else t_r, encoder_hidden_states, context_latents)


def context_latents(silence_latent, length, dtype=None):
    """Text-to-music context (1, T, 128): the silence latent over the length, then an all-ones mask."""
    src = silence_latent[:, :length]
    if src.shape[1] < length:
        src = torch.cat([src, silence_latent[:, :length - src.shape[1]]], dim=1)
    src = src.to(dtype or silence_latent.dtype)
    return torch.cat([src, torch.ones_like(src)], dim=-1)


# ---- the cover guide: the 5 Hz tokenizer and its detokenizer (the official "tokenizer" / "detokenizer") ----------
FSQ_LEVELS = (8, 8, 8, 5, 5, 5)             # config.json fsq_input_levels, one quantizer
POOL = 5                                    # pool_window_size: 5 latent frames (25 Hz) -> one 5 Hz code
POOLER_LAYERS = 2                           # num_attention_pooler_hidden_layers


class _PatchStack(nn.Module):
    def __init__(self, c: AceConfig):
        super().__init__()
        q3 = c.qwen3()
        self.embed_tokens = nn.Linear(c.hidden_size, c.hidden_size)
        self.norm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(q3)
        self.layers = nn.ModuleList([AceStepEncoderLayer(c, i, q3) for i in range(POOLER_LAYERS)])

    def _run(self, x):
        pe = self.rotary_emb(x, torch.arange(x.shape[1], device=x.device)[None])
        for layer in self.layers:
            x = layer(x, pe, None)          # bidirectional; the sliding window (128) is wider than a patch
        return self.norm(x)


class AttentionPooler(_PatchStack):
    def __init__(self, c: AceConfig):
        super().__init__(c)
        self.special_token = nn.Parameter(torch.zeros(1, 1, c.hidden_size))

    def forward(self, x):
        """(B, T5, 5, D) -> (B, T5, D): a learned token leads each patch and carries its summary."""
        b, t, p, d = x.shape
        x = self.embed_tokens(x)
        x = torch.cat([self.special_token.expand(b, t, 1, -1).to(x.dtype), x], dim=2).reshape(b * t, p + 1, d)
        return self._run(x)[:, 0].reshape(b, t, d)


class ResidualFSQ(nn.Module):
    """vector-quantize-pytorch's ResidualFSQ as ACE-Step builds it (one quantizer, levels 8,8,8,5,5,5, its defaults:
    symmetry-preserving bound with the hard clamp and the soft input clamp 1 + 1 / (L - 1)), inference only."""

    def __init__(self, dim):
        super().__init__()
        self.project_in = nn.Linear(dim, len(FSQ_LEVELS))
        self.project_out = nn.Linear(len(FSQ_LEVELS), dim)

    def forward(self, x):
        dt = x.dtype
        lv = torch.tensor(FSQ_LEVELS, device=x.device, dtype=torch.float32)
        z = F.linear(x.float(), self.project_in.weight.float(), self.project_in.bias.float())
        c = 1 + 1 / (lv - 1)
        z = (z / c).tanh() * c
        q = 2.0 / (lv - 1) * torch.floor((lv - 1) * (z.clamp(-1.0, 1.0) + 1) / 2 + 0.5) - 1.0
        return F.linear(q, self.project_out.weight.float(), self.project_out.bias.float()).to(dt)


class AceStepAudioTokenizer(nn.Module):
    def __init__(self, c: AceConfig):
        super().__init__()
        self.audio_acoustic_proj = nn.Linear(c.audio_acoustic_hidden_dim, c.hidden_size)
        self.attention_pooler = AttentionPooler(c)
        self.quantizer = ResidualFSQ(c.hidden_size)

    def forward(self, x):
        """(B, T, 64), T a multiple of 5 -> the quantized 5 Hz codes (B, T / 5, D)."""
        h = self.audio_acoustic_proj(x)
        return self.quantizer(self.attention_pooler(h.reshape(h.shape[0], -1, POOL, h.shape[-1])))


class AudioTokenDetokenizer(_PatchStack):
    def __init__(self, c: AceConfig):
        super().__init__(c)
        self.special_tokens = nn.Parameter(torch.zeros(1, POOL, c.hidden_size))
        self.proj_out = nn.Linear(c.hidden_size, c.audio_acoustic_hidden_dim)

    def forward(self, q):
        """(B, T5, D) codes -> (B, T5 * 5, 64): each code back out to five 25 Hz frames."""
        b, t, d = q.shape
        x = self.embed_tokens(q)[:, :, None].expand(b, t, POOL, d) + self.special_tokens.to(q.dtype)[:, None]
        x = self._run(x.reshape(b * t, POOL, d))
        return self.proj_out(x).reshape(b, t * POOL, -1)


class CoverGuide(nn.Module):
    """The official cover path's hint: a song's latents squeezed to 5 Hz codes (what is played, not how it sounds)
    and back out to 25 Hz - the context a cover render follows in place of the silence latent."""

    def __init__(self, c: AceConfig = None):
        super().__init__()
        c = (c or AceConfig()).encoder_config()
        self.tokenizer = AceStepAudioTokenizer(c)
        self.detokenizer = AudioTokenDetokenizer(c)

    @torch.no_grad()
    def forward(self, src, silence_latent):
        """src (B, T, 64) -> hints (B, T, 64); padded to whole patches with the silence latent, as the original."""
        n = src.shape[1]
        pad = (-n) % POOL
        if pad:
            src = torch.cat([src, silence_latent[:, :pad].to(src).expand(src.shape[0], -1, -1)], dim=1)
        return self.detokenizer(self.tokenizer(src))[:, :n]

    @classmethod
    def from_checkpoint(cls, path, device, dtype=torch.bfloat16):
        """Read only the tokenizer / detokenizer weights from the model file."""
        from accelerate import init_empty_weights
        from safetensors import safe_open
        with init_empty_weights():
            g = cls()
        sd = {}
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                name = k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k
                if name.startswith(("tokenizer.", "detokenizer.")):
                    sd[name] = f.get_tensor(k).to(dtype)
        missing, unexpected = g.load_state_dict(sd, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"no cover guide in this file: missing {missing[:4]}, unexpected {unexpected[:4]}")
        return g.to(device).eval().requires_grad_(False)
