# Fizgig v8.1.0: Flux 3 Image Klein

**Flux 3 Image Klein trains in Fizgig from day one.** Black Forest Labs' new 5.6B model for text-to-image and editing joins with everything the other image models have:

- **LoRAs and LoKRs**, with seven presets: Ultra Fast, Standard, Style, Edit, Edit Strong, Slider and Fine-tune.
- **Edit LoRAs** from pairs of original and edited photos: teach it a grade, a look or a retouch, then apply it to any photo.
- **Slider LoRAs** from photo pairs or three prompts.
- **Full fine-tuning** of the model itself.
- **From an 8 GB card.** Base precision Auto picks bf16, INT8 or 4-bit from your free VRAM; on small cards the model being trained steps off the GPU while previews render, so previews fit too.
- **Every workbench tool:** Studio with Turbo Preview, LoRA Royale with prompt travel, seed travel and edit chains, the Profiler and Extract.

LoRAs load in ComfyUI's standard LoRA loader. The files are gated, like Klein's: accept the licence on [flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy) (Preferences links it), then press **Download models for me** with a free Hugging Face read token. Everything else is in the [Flux 3 guide](https://github.com/shootthesound/Fizgig/blob/master/docs/FLUX3.md). Flux 3 is released under the FLUX Kommunity License: read it before publishing or selling anything made with it.

**For anyone adding a model:** the [driver guide](https://github.com/shootthesound/Fizgig/blob/master/docs/drivers/README.md) has Flux 3 as a reference driver, a way to work out a model's training memory from its file with a few short runs instead of a full sweep (Flux 3 as the worked example), and a checklist that now covers LoRA Royale's seed and prompt travel.

**Also fixed:** the app window no longer opens white on a PC without Microsoft's WebView2: Fizgig opens in your browser instead and says how to install it. Reported by @Robson1970 (#189).
