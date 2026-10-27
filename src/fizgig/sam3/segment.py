# Derived from ComfyUI (https://github.com/comfyanonymous/ComfyUI at 2035799): the checkpoint's key conversion
# (comfy/supported_models.py SAM3, comfy/utils.py transformers_convert), the text encoder's settings
# (comfy/text_encoders/sam3_clip.py) and the detect + refine steps (comfy_extras/nodes_sam3.py SAM3_Detect). Copyright (C)
# comfyanonymous and the ComfyUI contributors, GNU General Public License v3.0. This file stays under GPL-3.0 and is not
# covered by Fizgig's Apache-2.0 licence. See docs/THIRD_PARTY_NOTICES.md (ComfyUI) and LICENSES/GPL-3.0.txt.
"""SAM 3.1 masks from words: a picture and a few short prompts ("person", "hair", "glasses") -> one mask per thing
found. Reads ComfyUI's single-file checkpoint (Comfy-Org/sam3.1, sam3.1_multiplex_fp16.safetensors) as it is."""
import os

import numpy as np
import torch
import torch.nn.functional as F

SIZE = 1008                      # the detector's square input
TEXT_LEN = 32
_TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "clip_tokenizer")
_CLIP = dict(hidden_size=1024, intermediate_size=4096, num_attention_heads=16, num_hidden_layers=24,
             max_position_embeddings=TEXT_LEN, projection_dim=512, vocab_size=49408, hidden_act="quick_gelu",
             layer_norm_eps=1e-5, bos_token_id=49406, eos_token_id=49407, pad_token_id=0)


def _split_state_dict(sd):
    """The file's keys -> (SAM3Model's, the CLIP text model's in transformers' names)."""
    clip = {k: sd.pop(k) for k in list(sd) if "language_backbone" in k and "resizer" not in k}
    for k in list(sd):
        if k.startswith("tracker.model."):
            sd["tracker." + k[len("tracker.model."):]] = sd.pop(k)
    for k in [k for k in sd if ".attn.freqs_cis" in k]:
        sd.pop(k)
    for k in [k for k in sd if k.endswith((".in_proj_weight", ".in_proj_bias"))]:
        t = sd.pop(k)
        base, suffix = k.rsplit(".in_proj_", 1)
        s = ".weight" if suffix == "weight" else ".bias"
        d = t.shape[0] // 3
        sd[base + ".q_proj" + s], sd[base + ".k_proj" + s], sd[base + ".v_proj" + s] = t[:d], t[d:2 * d], t[2 * d:]
    for k in [k for k in sd if "sam_mask_decoder.transformer." in k]:
        new = k.replace(".mlp.lin1.", ".mlp.0.").replace(".mlp.lin2.", ".mlp.2.").replace(".norm_final_attn.", ".norm_final.")
        if new != k:
            sd[new] = sd.pop(k)

    pre = next(p for p in ("detector.backbone.language_backbone.encoder.", "backbone.language_backbone.encoder.")
               if any(k.startswith(p) for k in clip))
    clip = {k[len(pre):]: v for k, v in clip.items() if k.startswith(pre)}
    out = {"text_model.embeddings.position_embedding.weight": clip.pop("positional_embedding"),
           "text_model.embeddings.token_embedding.weight": clip.pop("token_embedding.weight"),
           "text_model.final_layer_norm.weight": clip.pop("ln_final.weight"),
           "text_model.final_layer_norm.bias": clip.pop("ln_final.bias")}
    names = {"ln_1": "layer_norm1", "ln_2": "layer_norm2", "mlp.c_fc": "mlp.fc1", "mlp.c_proj": "mlp.fc2",
             "attn.out_proj": "self_attn.out_proj"}
    for i in range(_CLIP["num_hidden_layers"]):
        src, dst = f"transformer.resblocks.{i}.", f"text_model.encoder.layers.{i}."
        for a, b in names.items():
            for y in ("weight", "bias"):
                out[f"{dst}{b}.{y}"] = clip.pop(f"{src}{a}.{y}")
        for y in ("weight", "bias"):
            w = clip.pop(f"{src}attn.in_proj_{y}")
            d = w.shape[0] // 3
            for j, p in enumerate(("q_proj", "k_proj", "v_proj")):
                out[f"{dst}self_attn.{p}.{y}"] = w[d * j:d * (j + 1)]
    return sd, out


class SAM3Segmenter:
    """Loaded once; masks() for each picture."""

    def __init__(self, path, device="cuda", dtype=torch.float16):
        from safetensors.torch import load_file
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
        from fizgig.sam3._comfy import ops
        from fizgig.sam3.detector import SAM3Model

        self.device, self.dtype = torch.device(device), dtype
        sd = load_file(path)
        multiplex = "detector.backbone.vision_backbone.propagation_convs.0.conv_1x1.weight" in sd
        sd, clip_sd = _split_state_dict(sd)
        # built on the CPU: a few buffers (position tables) are made here, not read from the file
        self.model = SAM3Model(device="cpu", dtype=dtype, operations=ops, image_model="SAM31" if multiplex else "SAM3")
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"SAM 3 file doesn't match: {len(missing)} missing (e.g. {missing[:3]}), "
                               f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})")
        self.model.to(self.device, dtype).eval().requires_grad_(False)
        self.text = CLIPTextModel(CLIPTextConfig(**_CLIP))
        missing, unexpected = self.text.load_state_dict(clip_sd, strict=False)
        if [k for k in missing if "position_ids" not in k] or unexpected:
            raise RuntimeError(f"SAM 3 text encoder doesn't match: missing {missing[:3]}, unexpected {unexpected[:3]}")
        self.text.to(self.device, dtype).eval().requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(_TOKENIZER_DIR)

    @torch.no_grad()
    def encode(self, prompt):
        """A short prompt -> (embeddings (1, 32, 1024), mask (1, 32)): start, the words, end, then zeros."""
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"][:TEXT_LEN - 2]
        ids = [_CLIP["bos_token_id"], *ids, _CLIP["eos_token_id"]]
        mask = [1] * len(ids) + [0] * (TEXT_LEN - len(ids))
        ids = torch.tensor([ids + [0] * (TEXT_LEN - len(ids))], device=self.device)
        mask = torch.tensor([mask], device=self.device)
        emb = self.text(input_ids=ids, attention_mask=mask).last_hidden_state
        return emb.to(self.dtype), mask

    @torch.no_grad()
    def masks(self, image, prompts, threshold=0.5, refine=2, max_each=None):
        """image: uint8 (H, W, 3). prompts: short phrases. -> [{"prompt", "score", "box" (x1, y1, x2, y2) pixels,
        "mask" bool (H, W)}, ...], strongest first per prompt; max_each keeps that many per prompt (None = all)."""
        H, W = image.shape[:2]
        img = torch.from_numpy(np.ascontiguousarray(image[..., :3])).to(self.device, torch.float32) / 255.0
        frame = F.interpolate(img.permute(2, 0, 1)[None], size=(SIZE, SIZE), mode="bilinear").to(self.dtype)
        found = []
        for prompt in prompts:
            emb, mask = self.encode(prompt)
            res = self.model(frame, text_embeddings=emb, text_mask=mask, threshold=threshold, orig_size=(H, W))
            probs = res["scores"][0].sigmoid()
            keep = probs > threshold
            boxes, scores, ms = res["boxes"][0][keep], probs[keep], res["masks"][0][keep]
            order = scores.argsort(descending=True)[:max_each]
            for i in order.tolist():
                m = self._refine(img, ms[i], boxes[i], H, W, refine)
                found.append({"prompt": prompt, "score": float(scores[i]), "box": [float(v) for v in boxes[i]],
                              "mask": m.cpu().numpy()})
        return found

    @torch.no_grad()
    def point_mask(self, image, points, refine=2):
        """image: uint8 (H, W, 3); points: [(x, y, 1 = this / 0 = not this)] in its pixels -> bool (H, W): the object
        at the points, through SAM's decoder (SAM3_Detect's point path)."""
        H, W = image.shape[:2]
        img = torch.from_numpy(np.ascontiguousarray(image[..., :3])).to(self.device, torch.float32) / 255.0
        frame = F.interpolate(img.permute(2, 0, 1)[None], size=(SIZE, SIZE), mode="bilinear").to(self.dtype)
        coords = torch.tensor([[[x / W * SIZE, y / H * SIZE] for x, y, _l in points]], dtype=self.dtype, device=self.device)
        labels = torch.tensor([[int(l) for _x, _y, l in points]], dtype=torch.int32, device=self.device)
        logit = self.model.forward_segment(frame, point_inputs={"point_coords": coords, "point_labels": labels})
        for _ in range(max(0, refine - 1)):
            logit = self.model.forward_segment(frame, mask_inputs=logit)
        return (F.interpolate(logit.float(), size=(H, W), mode="bilinear")[0, 0] > 0).cpu().numpy()

    def _refine(self, img, coarse, box, H, W, iterations):
        """A detector mask through SAM's decoder on a crop around its box (SAM3_Detect's _refine_mask)."""
        def coarse_only():
            return F.interpolate(coarse[None, None].float(), size=(H, W), mode="bilinear")[0, 0] > 0
        if iterations <= 0:
            return coarse_only()
        x1, y1, x2, y2 = box.tolist()
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = max(0, int(x1 - bw * 0.1)), max(0, int(y1 - bh * 0.1))
        cx2, cy2 = min(W, int(x2 + bw * 0.1)), min(H, int(y2 + bh * 0.1))
        if cx2 <= cx1 or cy2 <= cy1:
            return coarse_only()
        crop = img[cy1:cy2, cx1:cx2].permute(2, 0, 1)[None]
        crop = F.interpolate(crop, size=(SIZE, SIZE), mode="bilinear").to(self.dtype)
        mh, mw = coarse.shape[-2:]
        mx1, my1, mx2, my2 = int(cx1 / W * mw), int(cy1 / H * mh), int(cx2 / W * mw), int(cy2 / H * mh)
        if mx2 <= mx1 or my2 <= my1:
            return coarse_only()
        logit = coarse[..., my1:my2, mx1:mx2][None, None]
        for _ in range(iterations):
            logit = self.model.forward_segment(crop, mask_inputs=F.interpolate(logit, size=(SIZE, SIZE), mode="bilinear"))
        out = torch.zeros(H, W, device=img.device, dtype=torch.bool)
        out[cy1:cy2, cx1:cx2] = F.interpolate(logit.float(), size=(cy2 - cy1, cx2 - cx1), mode="bilinear")[0, 0] > 0
        return out | coarse_only()
