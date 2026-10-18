# MOSS-Image

[← Back to the README](../README.md)

MOSS-Image is OpenMOSS's 5B single-stream model for text-to-image, released 16 October 2026 under Apache-2.0. It renders text well (English and Chinese), and takes prompts of up to 4,096 tokens as well as JSON layout prompts that place text and objects.

Fizgig trains it on the **Base** model, the checkpoint OpenMOSS recommend for training. It offers LoRAs, LoKRs, slider LoRAs, full fine-tunes, Multi Concept, DOP and Fizgig LoRA Isolation. Every workbench tool works with it: Repair Studio with Turbo Preview, LoRA Royale with seed and prompt travel, the Profiler and Extract. Saved LoRAs load in ComfyUI's standard LoRA loader. MOSS-Image came to Fizgig through @Murf-Code's pull request.

**No edit LoRAs:** the released model is text-to-image only. Its technical report says it "does not directly condition on reference images or support image editing". The "layout editing" it shows is regenerating from an edited layout prompt.

## Getting set up

Pick **MOSS-Image** on the Training tab, then press **Download models for me** in Preferences. None of the files are gated. From the command line:

```bash
python -m fizgig.scripts.fetch_models --family moss_image
```

| Model | File | Size |
|---|---|---|
| Base DiT (training) | `moss_image_base_bf16.safetensors` from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) | 10.2 GB |
| SFT DiT (previews) | `moss_image_sft_int8_convrot.safetensors` from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) | 5.4 GB |
| Text encoder | [`qwen3.5_9b_bf16.safetensors`](https://huggingface.co/Comfy-Org/Qwen3.5/blob/main/text_encoders/qwen3.5_9b_bf16.safetensors) from Comfy-Org/Qwen3.5 (not in the Comfy-Org/MOSS-Image repo, which only has quantised ones) | 19.3 GB |
| Autoencoder | `flux2-vae_small.safetensors` from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image): FLUX.2's encoder with the small decoder MOSS-Image ships with | 0.25 GB |

- **Train on Base.** MOSS-Image SFT is OpenMOSS's fine-tuned model for everyday use. Fizgig never trains it, but previews on it.
- **Autoencoder:** Flux 3's or Klein 9B's full FLUX.2 autoencoder also works, with the same latents, but the small decoder avoids spots in the pictures.
- **Text encoder alternatives:** Comfy-Org's `qwen3.5_9b_int8_convrot.safetensors` (9.7 GB) also works. So do the official diffusers folders (OpenMOSS-Team/MOSS-Image-Base `transformer/` and `text_encoder/`): give the path of any shard.
- **Optional Turbo LoRA:** Preferences has an empty row for a turbo LoRA (today, lodestones' 8-step DMD LoRA, which is softer on Base). Set it, then raise **Turbo LoRA strength** on the Samples tab and set Steps and CFG for it (8 steps, CFG 1).

## Presets

| Preset | Rank | Learning rate | Epochs | Size |
|---|---|---|---|---|
| **MOSS-Image Fast** | 16 | 1e-4 | 50 | 0.25 MP |
| **MOSS-Image Standard** | 32 | 1e-4 | 50 | 0.5 MP |
| **MOSS-Image Style** | 16 | 5e-5 | 50 | 0.5 MP |
| **MOSS-Image Slider** | 8 | 2e-4 | 20 | 0.5 MP |
| **MOSS-Image Fine-tune** | — | 1e-5 | rotations | 0.5 MP |

- **Fast is the recommended start,** and gives excellent results.
- **Learning rate:** every preset uses a fixed one, with AdamW, EMA 0.98 and the per-image loss watch, saving every epoch. On MOSS-Image's flat loss, Adaptive LR walks itself down to its floor. It's still on the Training tab, off by default.
- **Picking an epoch:** scrub the epochs in LoRA Royale to choose the best one.

## Memory

- **Training:** Auto trains on an INT8 base, or NF4 on smaller cards, and compiles the blocks for about 25% faster steps on longer runs. bf16 is a manual choice.
- **Text encoder:** it always runs at full precision. On cards without room for it (about 15 GB), it streams from system memory one layer at a time, still exact, so captions cache on any card.
- **12 GB cards:** these train with SFT previews and stay under about 10.5 GB.

## Captions

MOSS-Image was trained on long, detailed captions, with short, medium and long variants of each. Its text encoder reads up to 4,096 tokens, so describe the picture fully, including any text in it exactly as written.

Fizgig never trains the text encoder or the model's text adapter, as OpenMOSS advise.

## Previews

- **Samples render on the SFT model** by default, so you see your LoRA where you'll use it, though a Base LoRA carries over only partly. Untick the box on the Samples tab to preview on the Base model being trained. Repair Studio, LoRA Royale and the other workbench tabs render on SFT too.
- **Sampler:** res_multistep for 30 steps at CFG 4 on MOSS-Image's shift-5 schedule, with block 22 left out of the negative branch. That was the best of side-by-side tests against OpenMOSS's own recipe, Euler at 50 steps, CFG 3.
- **Block 22 and the LoRA:** training leaves block 22 out of the LoRA too, so CFG's contrast there stays the base model's.
- **Negative:** the default is `worst quality, bad anatomy, watermark, jpeg artifact`.

## In ComfyUI

| Node | Setting |
|---|---|
| KSampler | sampler `res_multistep`, scheduler `simple`, 30 steps, CFG 4 |
| SkipLayerGuidanceDiTSimple | layer 22, on the negative (whenever CFG is above 1) |
| Negative prompt | `worst quality, bad anatomy, watermark, jpeg artifact` |
| VAE | `flux2-vae_small.safetensors` |
| Text encoder | [`qwen3.5_9b_bf16.safetensors`](https://huggingface.co/Comfy-Org/Qwen3.5/blob/main/text_encoders/qwen3.5_9b_bf16.safetensors), in `models/text_encoders` |
| Your LoRA | Load LoRA |

- **Text encoder:** use the bf16 one. MOSS-Image's developers say the text encoder is more fragile than the model, so avoid heavy quants of it.
- **Seeds:** ComfyUI's schedule isn't snapped to the training grid, so a seed renders slightly differently there than in Fizgig.
