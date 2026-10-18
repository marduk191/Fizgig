"""MOSS-Image (OpenMOSS / MOSI, released 16 Oct 2026): a 5B single-stream DiT for text-to-image with strong text
rendering and JSON layout prompts, Qwen3.5-9B text encoder, the FLUX.2 autoencoder. Apache-2.0 weights.

Sources: github.com/OpenMOSS/MOSS-Image (README, USERGUIDE, paper/MOSS_Image_Technical_Report.pdf, commit c1355e6),
the model cards and pipeline_moss_image.py (huggingface.co/OpenMOSS-Team/MOSS-Image-Base, -SFT), Comfy-Org's repack
(huggingface.co/Comfy-Org/MOSS-Image) and ComfyUI's core support (Comfy-Org/ComfyUI PR #17095, merge e087d8f6cb).

"""
from fizgig.families.description import FamilyDescription, LoRAFormat, ModelFile, SamplingSettings, SpeedLoRA

_REPO = "https://github.com/OpenMOSS/MOSS-Image"
_CARD = "https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base"
_COMFY = "Comfy-Org/MOSS-Image"
_GUIDE = f"{_REPO} USERGUIDE.md"
_REPORT = f"{_REPO} paper/MOSS_Image_Technical_Report.pdf"
_PIPE = f"{_CARD} pipeline_moss_image.py"
_TURBO = "lodestones/MOSS-Image-SFT-8-steps-dmd"
_MODULES = ("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down")   # model.py _Block Linears
# USERGUIDE.md Prompt Guidelines: "we recommend the following effective negative tags"
_NEGATIVE = "worst quality, bad anatomy, watermark, jpeg artifact"


def _preset(rank, lr=1e-4, adaptive=None, epochs=30, mp="0.5", slider=False, ft=False):
    # Not yet trained on MOSS-Image (the driver was built without a GPU): Flux 3's preset shape (Krea 2's values,
    # families/flux3.py), at Qwen 2.1's 0.5 MP, with fused AdamW as CHECKLIST.md asks for LoRA presets. EMA 0.98, the
    # per-image loss watch with per-image LR, saved every epoch. adaptive=(min, max) turns Adaptive LR on.
    lo, hi = adaptive or ("1e-4", "4e-4")
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)", "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": adaptive is not None, "ADAPTIVE_LR_MIN": lo, "ADAPTIVE_LR_MAX": hi,
        "OPTIMIZER_TYPE": "adamw", "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": mp, "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_PRECISION": "Auto (fits your free VRAM)", "FAMILY_EMA": "0.98 (recommended)",
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": True, "KREA2_AUTO_RECAPTION": False,
        "KREA2_WARMUP_LOOK": False, "FAMILY_FT": ft, "FAMILY_EDIT": False, "FAMILY_SLIDER": slider,
        "FAMILY_FAST_ID": False, "FAMILY_SLIDER_GUIDANCE": "3",
        **({"FAMILY_FT_ROTATIONS": "10"} if ft else {}),
    }


MOSS_IMAGE = FamilyDescription(
    key="moss_image",
    arch_id="mossimage",
    display_name="MOSS-Image",
    gui_label="MOSS-Image",
    lora_name_suffix="moss",
    aliases=("moss-image", "mossimage"),

    model_files=(
        ModelFile("moss_image_dit", "MOSS-Image Base DiT", True, _COMFY,
                  "diffusion_models/moss_image_base_bf16.safetensors", 10.16,
                  "bf16 Base for training (the model card: Base is for fine-tuning and LoRAs). The SFT file "
                  "(moss_image_sft_bf16) is for use, not training.",
                  role="dit",
                  # verified 17 Oct 2026: tensor for tensor the official transformer/ shards, q / k / v fused into
                  # proj.qkv = [q; k; v]; the loader splits it back
                  hint="MOSS-Image Base - ComfyUI's single file. The official diffusers shards "
                       "(OpenMOSS-Team/MOSS-Image-Base transformer/) and the int8_convrot file load too."),
        ModelFile("moss_image_vae", "FLUX.2 autoencoder (small decoder)", True, _COMFY,
                  "vae/flux2-vae_small.safetensors", 0.25,
                  "The FLUX.2 autoencoder with BFL's small decoder, the decoder MOSS-Image ships with.",
                  role="vae",
                  # MOSS-Image's own checkpoint uses the FLUX.2 small decoder "to help eliminate some strange artifacts
                  # and spots" (a MOSS-Image core developer, r/StableDiffusion AMA, Oct 2026). This file's encoder is
                  # FLUX.2's own, tensor for tensor, so latents are the same as with flux2-vae.safetensors (which
                  # still loads here: same latent space, the full decoder)
                  replaces=("flux2-vae.safetensors", "ae.safetensors"),
                  hint="FLUX.2's encoder with the small decoder MOSS-Image was released with. Flux 3's or Klein's "
                       "full autoencoder also works (same latents), but the small decoder avoids spots in the "
                       "pictures."),
        ModelFile("moss_image_turbo_lora", "Turbo LoRA (optional)", False, _TURBO, "lora.safetensors", 1.19,
                  "Optional speed LoRA for previews; leave empty to preview at full quality.", role="speed_lora",
                  bulk_fetch=False,           # its own Download link only: soft on Base today (Peter, 18 Oct 2026)
                  hint="Optional. A turbo / DMD LoRA for fast previews - lodestones' 8-step DMD distillation of "
                       "MOSS-Image SFT today (work in progress; soft on Base). Leave it empty for full-quality "
                       "previews. With one set, choose its steps and CFG on the Samples tab; the workbench follows."),
        ModelFile("moss_image_text_encoder", "Qwen3.5-9B text encoder", True, "Comfy-Org/Qwen3.5",
                  "text_encoders/qwen3.5_9b_bf16.safetensors", 19.31,
                  "Used for caching only, then unloaded. Loads text-only (about 15 GB), or 8-bit (about 9 GB) on "
                  "smaller cards.",
                  role="text_encoder",
                  # stock Qwen3.5-9B: MOSS-Image's text_encoder/ shards are these tensors (24 of 24 sampled equal
                  # after the reference's bf16 cast); MOSS keeps it frozen (USERGUIDE.md Post-Training)
                  alt_repo=_COMFY, alt_path="text_encoders/qwen3.5_9b_int8_convrot.safetensors", alt_label="int8",
                  alt_note="~9.7 GB int8 ConvRot, the file ComfyUI's MOSS-Image workflow uses; decoded at load"),
        # MOSS-Image SFT (OpenMOSS's aesthetics-tuned checkpoint, same architecture; @Murf-Code, PR #195): Repair
        # Studio and LoRA Royale render on it once set, and training samples do (the Samples tab tick, on by
        # default - a Base LoRA carries over to the SFT only partly). Never trained: Fizgig trains on the Base
        ModelFile("moss_image_sft_dit", "MOSS-Image SFT DiT (previews)", True, _COMFY,
                  "diffusion_models/moss_image_sft_int8_convrot.safetensors", 5.41, role="preview_dit",
                  hint="OpenMOSS's everyday checkpoint: training samples, Repair Studio and LoRA Royale render on "
                       "it (untick it on the Samples tab for samples on the Base), showing the LoRA where most "
                       "people use it. Never trained. ComfyUI's int8 ConvRot file is decoded at load; the bf16 SFT "
                       "file works as well.",
                  download_note="~5.4 GB (int8 ConvRot) - Apache-2.0",
                  alt_repo=_COMFY, alt_path="diffusion_models/moss_image_sft_bf16.safetensors",
                  alt_label="bf16 file", alt_note="~10.2 GB"),
    ),
    text_encoder_label="Qwen3.5-9B",
    vae_label="FLUX.2 autoencoder",
    helper_files=(("OpenMOSS-Team/MOSS-Image-Base", ("tokenizer/*",)),),   # tokenizer.json + chat template

    latent_channels=128,              # transformer/config.json latent_channels: 32 FLUX.2 channels 2x2-packed
    spatial_factor=16,                # report Table 5: effective image stride 16 pixels
    bucket_step=16,                   # pipeline __call__: "height and width must be positive multiples of 16"
    native_megapixels=1.0,            # USERGUIDE.md: 1, 1.5 or 2 MP recommended, 0.25 to 4 MP supported

    n_blocks=42,                      # transformer/config.json num_layers
    block_prefix="layers",
    block_note="42 joint blocks with one shared modulation (so per-block sliders act on attention and MLP), after a "
               "two-block text adapter that refines the Qwen3.5 states before they meet the image.",

    lora=LoRAFormat(
        key_template="lora_unet_layers_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=_MODULES,
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        note="kohya keys over the reference module names. ComfyUI's MOSSImage branch maps lora_unet_layers_N_proj_q "
             "/ _k / _v onto thirds of the fused proj.qkv in Comfy-Org's file; its generic map takes the same keys "
             "on an unfused model (the OpenMOSS ComfyUI plugin). LoKR saves as diffusion_model.layers.N.proj.q, "
             "which both maps also take.",
        source="Comfy-Org/ComfyUI comfy/lora.py model_lora_keys_unet (PR #17095, merge e087d8f6cb)",
    ),

    driver="fizgig.moss_image.driver:MOSSImageDriver",
    modelspec_arch="MOSS-Image",
    implementation=_REPO,
    precisions=("int8", "nf4", "bf16"),
    auto_precisions=("int8", "nf4"),  # AI Toolkit's MOSS-Image default is an 8-bit base; bf16 stays a manual choice
    # MEASURED 17 Oct 2026 on a 5090 (STILLS.md 4's calculation, 4 one-epoch runs + the 2-epoch smoke run). The
    # preview round (1024², 50 steps, CFG 3) sets the peak at every precision, so the figures are the preview's peak
    # after training (optimizer state resident; bf16 smoke run 15.1 GB vs 14.6 before training, +0.5 applied to INT8
    # / NF4). Training alone: bf16 9.2 GB at 0.25 MP with 10 blocks swapped (7.97 resident -> 1.2 GB overhead,
    # ~11.3 GB unswapped), 13.4 at 1 MP; INT8 8.8 and NF4 7.3 at 1 MP. Per swapped block: 0.211 GB bf16 (10 blocks
    # cut the preview peak 2.1 GB), 0.105 INT8.
    train_memory={"bf16": (((0.25, 15.1), (1.0, 15.1)), 0.211), "int8": (((0.25, 10.5), (1.0, 10.5)), 0.105),
                  "nf4": (((0.25, 8.5), (1.0, 8.5)), 0.0)},
    optimizers=("adamw", "adamw8bit"),
    network_types=("lora", "lokr"),
    ema_default="0.98",
    compiles=True,                    # driver.compile_targets: the 42 blocks
    compile_boundary="outside",       # as Qwen 2.1's and Flux 3's: the checkpoint wraps the compiled block
    # measured 17 Oct 2026, 5090, bf16, 1 MP, one bucket: 1.45 -> 1.07 s/step compiled (-26%), the first epoch
    # 13 s slower (compiling) = paid back in ~35 steps; 200 leaves room for runs with several bucket shapes.
    # INT8 / NF4 not measured yet: bf16's figure, and Klein's NF4 one, so Auto compiles the default INT8 base.
    compile_payback_steps={"bf16": 200, "int8": 200, "nf4": 400},
    # int8_attention left off: the model routes attention through int8_attention.attend(), but the speed-up isn't
    # measured yet (ABILITIES.md: keep it where it measures faster)
    finetune=True,                    # driver.ft_spec: every block's six Linears on an NF4 trunk
    ft_learning_rate=1e-5,            # FINETUNE.md's default; the report's pretraining LR was 5e-5 (Muon + AdamW)
    slider_training=True,             # photo pairs (training_loss diff weighting) or prompts (noise_latents / predict)
    # edit_training stays off: the released checkpoints don't condition on reference images (report section 8: "does
    # not directly condition on reference images or support image editing"; pipeline docstring: defaults are "not
    # ... reference-image variants"). The RoPE keeps a reference-index axis for a later release.
    workbench_follows_samples=True,   # CFG 3 needs the Samples tab's negative (empty = the reference's default)
    activation_cache=True,            # Turbo Preview: step-1 replay; generate calls on_step before every step
    repair_size=1024,                 # a 1 MP model (CHECKLIST.md Workbench)

    sampling=(
        SamplingSettings("res_multistep, 30 steps, CFG 4", steps=30, cfg=4.0, sampler="res_multistep",
                         scheduler="simple",
                         options=(("shift", 5.0), ("uncond_skip", (22,)), ("sampler", "res_multistep")),
                         negative_prompt=True,
                         note="ComfyUI's second-order res_multistep on the reference's shift-5 grid. Picked by eye "
                              "from res_multistep 20 / 30 / 50 at CFG 3 and 4 and Euler 30 / 50 on the same seeds.",
                         source="Peter's comparison, 18 Oct 2026 (after r/StableDiffusion's res_multistep reports)"),
        SamplingSettings("Euler, 30 steps", steps=30, cfg=3.0, sampler="euler", scheduler="simple",
                         options=(("shift", 5.0), ("uncond_skip", (22,))), negative_prompt=True,
                         note="AI Toolkit's MOSS-Image sample settings: the reference sampler at 30 steps.",
                         source="ostris/ai-toolkit 7b94dc5, extensions_built_in/diffusion_models/ui.tsx (moss_image)"),
        SamplingSettings("Reference (diffusers pipeline)", steps=50, cfg=3.0, sampler="euler", scheduler="simple",
                         options=(("shift", 5.0), ("uncond_skip", (22,))), negative_prompt=True,
                         note="Flow-matching Euler on the shift-5 grid snapped to the 1000 training sigmas; CFG 3 with "
                              "block 22 skipped in the negative branch; the negative is empty by default.",
                         source=f"{_PIPE} (MOSSImageScheduler, __call__ defaults); {_CARD} model card"),
        SamplingSettings("ComfyUI workflow", steps=50, cfg=3.0, sampler="euler", scheduler="sgm_uniform",
                         options=(("shift", 5.0), ("uncond_skip", (22,))), negative_prompt=True,
                         note="SkipLayerGuidanceDiTSimple layer 22 on the negative, and no ModelSamplingSD3 node: ComfyUI's "
                              "MOSS-Image model brings its own sampling, and adding one renders badly (Peter, 18 Oct 2026). "
                              "ComfyUI's scheduler does not snap to the training grid, so the same seed differs slightly.",
                         source=f"{_REPO} comfyui/moss-image/example_workflows/moss-image-api.json, comfyui/README.md"),
    ),
    speed_loras=(
        SpeedLoRA(
            name="MOSS-Image Turbo LoRA",
            repo=_TURBO,
            file="lora.safetensors",
            pairs_with="MOSS-Image SFT",
            strength=1.0,             # lodestones' card: lora.safetensors at 1.0 (lora_latest at 0.7)
            settings=SamplingSettings("Turbo 8-step", steps=8, cfg=1.0, sampler="euler", scheduler="simple",
                                      options=(("shift", 5.0), ("uncond_skip", ()), ("sampler", "euler")),
                                      note="8 steps, no CFG, the reference's shift-5 grid.",
                                      source=f"https://huggingface.co/{_TURBO} (load in ComfyUI's LoRA loader)"),
            load_unmerged=True,       # an fp32 rank-128 adapter: its delta is below bf16's rounding step
            pref_key="moss_image_turbo_lora",
            community_settings=(("12-16 steps for stronger prompt adherence; strength 0.8 at 16 steps",
                                 "r/StableDiffusion 'MOSS-Image INT8 ComfyUI' thread; dskjal's ComfyUI guide"),),
            caveats=("Distilled from MOSS-Image SFT, and soft on Base (Peter's comparison, 18 Oct 2026).",),
            source=f"https://huggingface.co/{_TURBO}",
        ),
    ),
    preview_speed_lora="MOSS-Image Turbo LoRA",      # used only when its Preferences row is set (empty by default)
    preview_speed_strength=0.0,       # the Samples tab's Turbo LoRA strength starts at 0 (full-quality previews);
    #                                   raise it with the Steps / CFG boxes set for the turbo (1 at 8 steps, CFG 1)
    preview_steps=30,                 # res_multistep 30 at CFG 4: the best of the side-by-side (Peter, 18 Oct)
    preview_cfg=4.0,                  # USERGUIDE.md: 3-6 recommended; 4 won the side-by-side
    preview_negative=_NEGATIVE,
    preview_cfg_note="4 = Fizgig's default (MOSS-Image's guide: 3-6; raise it for short prompts). The negative branch "
                     "skips block 22, as the reference does. 1 = no CFG.",
    samples_text=(("sampler", "res_multistep on MOSS-Image's shift-5 schedule (in ComfyUI: res_multistep, simple "
                              "scheduler), block 22 skipped in the negative branch."),),
    # training samples on the SFT checkpoint (the driver's park_for_preview / load_preview_checkpoint /
    # unpark_after_preview; @Murf-Code, PR #195): off by default - a LoRA trained on the Base carries over only partly
    train_preview_checkpoint=True,
    preview_checkpoint_tick="Use the SFT model for samples (set it in Preferences - a Base LoRA carries over "
                            "only partly)",
    preview_checkpoint_on=True,       # samples on the SFT model by default: where most people use a MOSS LoRA (Peter)
    preview_checkpoint_steps_note="Base samples only - the SFT samples use their own 30 steps, CFG 4",
    preview_checkpoint_sampling=SamplingSettings(
        "MOSS-Image SFT", steps=30, cfg=4.0, sampler="res_multistep", scheduler="simple",
        options=(("shift", 5.0), ("uncond_skip", (22,)), ("sampler", "res_multistep")), negative_prompt=True,
        note="The Base samples' sampler (res_multistep 30, CFG 4, shift 5, the negative branch skipping block 22) on "
             "the SFT checkpoint.",
        source=f"https://huggingface.co/{_COMFY}; OpenMOSS/MOSS-Image-SFT model card"),
    preview_width=1024,
    preview_height=1024,
    presets=(
        # untrained starting values (see _preset)
        # a fixed learning rate: Adaptive LR walks to its floor on MOSS-Image's flat loss (@Murf-Code's runs, PR
        # #195; Peter's runs agree that 1e-4 suits it). Adaptive LR stays on the Training tab, off by default
        ("✨ MOSS-Image Fast (rank 16, 1e-4)", _preset(16, epochs=50, mp="0.25")),
        ("✨ MOSS-Image Standard (rank 32, 1e-4)", _preset(32, epochs=50)),
        ("✨ MOSS-Image Style (rank 16, 5e-5)", _preset(16, lr=5e-5, epochs=50)),
        ("✨ MOSS-Image Slider (rank 8, 2e-4)", _preset(8, lr=2e-4, epochs=20, slider=True)),
        ("✨ MOSS-Image Fine-tune (1e-5)", _preset(16, lr=1e-5, ft=True)),
    ),
    workbench=("repair", "profiler", "extract", "royale"),

    notes=(
        ("Training timesteps follow the released weights' last pretraining interval: sigma = 1 - sigmoid(Z), "
         "Z ~ N(-1.09, 0.95^2), truncated at 0.95 and rescaled onto the 1000-step grid, mixed with 1% uniform; the "
         "DiT is told 1000 * sigma; loss is the unweighted per-token MSE on noise - x0.",
         f"{_REPORT} Appendix A.2, eq. 3-4, Table 6"),
        ("Text conditioning: Qwen3.5-9B's final hidden state through the chat template with an open <think> block, "
         "unpadded, up to 4096 tokens. Keep the text encoder frozen.",
         f"{_PIPE} encode_prompt; Comfy-Org/ComfyUI comfy/text_encoders/moss_image.py; {_GUIDE} Post-Training"),
        ("No edit or reference-image training: the released checkpoints are text-to-image only. 'Layout editing' "
         "means regenerating from an edited layout prompt, not conditioning on a picture.",
         f"{_REPORT} section 6 and 8; {_GUIDE}"),
        ("No speed LoRA or distilled checkpoint yet (DMD distillation is announced as coming), so previews run the "
         "full 50 steps with CFG.", f"{_GUIDE} Few-Step Distillation"),
    ),
)
