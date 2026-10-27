# Fizgig v8.9.4: maintenance

- **MOSS-Image trains with ComfyUI's 4-bit text encoder** (`qwen3.5_9b_w4a8_gptq.safetensors`, the file ComfyUI's MOSS-Image workflow uses), so you don't need a second download. It loads about a minute slower than the int8 or bf16 file and uses the same memory once loaded; what it saves is download and disk space (7.3 GB against 19.3 GB) (thanks @driqeks).
- **MOSS-Image text encoders and models that carry activation scales (`.input_scale`) load** instead of stopping at the key check (thanks @gibe9000).
- **MOSS-Image training on Windows no longer crashes at its first preview.** With previews on SFT, the run could stop there with an access violation (Windows error 0xC0000005), seen on a computer with 32 GB of RAM. If you set `FIZGIG_TE_FROM_FILE=1` to get round it, you can remove it (thanks @gibe9000).
- **Prompt × Epoch has a scroll bar above the previews as well as below**, so a long grid scrolls without going to the bottom of the page (thanks @CognitiveDiffusion).
- **Follow Samples tab settings in the MOSS-Image workbench sticks.** The choice wasn't saved, so it went back to SFT.

Everything new in 8.9: [Fizgig v8.9.0 release notes](https://github.com/shootthesound/Fizgig/releases/tag/v8.9.0).

**Updating:** update Fizgig on your computers, and restart your pods.
