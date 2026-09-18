"""MiniMax H3 RefMod — make a reference mod from the dataset and optimise it against the
frozen H3 base (see fizgig.minimax.refmod). The GUI's "MiniMax H3 RefMod" Base Model entry
builds this command; the caches come from the ordinary minimax_cache_latents /
minimax_cache_text steps.

    python -m fizgig.scripts.minimax_refmod --dit <h3> --dataset_config <toml>
        --output_dir out --output_name subject --grid 16 --steps 200
        [--vae <video vae> --text_encoder <qwen3-vl> --sample_prompts prompts.txt]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fizgig.minimax.refmod import run_refmod  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def setup_parser():
    p = argparse.ArgumentParser(description="MiniMax H3 RefMod: encode + optimise a reference mod")
    p.add_argument("--dit", required=True, help="MiniMax H3 DiT (pruned int8 or bf16)")
    p.add_argument("--dataset_config", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_name", required=True)
    p.add_argument("--grid", default="full",
                   help="'full' (default) keeps every reference on the first reference's latent "
                        "canvas — the mode that carries a face; 8 / 16 / 32 average-pool to that "
                        "many latent cells on the long edge (small, stackable, concept-level).")
    p.add_argument("--sigma_min", type=float, default=0.2,
                   help="optimise only at noise levels in [sigma_min, sigma_max] (default 0.2-0.8, "
                        "measured; -1 for both = H3's own shift-12 density)")
    p.add_argument("--sigma_max", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=200,
                   help="optimisation steps against the frozen base (0 = encode only, the "
                        "node extractor's own result)")
    p.add_argument("--ref_subset", type=int, default=1,
                   help="references each optimisation step rides, picked at random from the ones "
                        "with a large face in frame (default 1 — one reference beside the still "
                        "keeps the step small enough to skip recompute on a 32 GB card; 0 = all)")
    p.add_argument("--lr", type=float, default=5e-3, help="latent-space AdamW rate (default 5e-3, flat, no warm-up)")
    p.add_argument("--pull", type=float, default=2.0,
                   help="weight of the L2 pull toward the initial encode (default 2.0, measured)")
    p.add_argument("--max_refs", type=int, default=16, help="references stacked into the mod (default 16, measured)")
    p.add_argument("--clips", choices=["still", "motion"], default="still",
                   help="how a clip enters the mod: its sharpest-face still (default) or every one of its "
                        "latent frames as motion (the node pack's video reference)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_quant", default="nf4",
                   help="nf4 (default: the base at 10.5 GB fits beside the step without recompute on "
                        "32 GB and without streaming on 16/24 GB; the mod's gradient is not the place "
                        "the extra bits show) / int8 / hqq / auto")
    p.add_argument("--blocks_to_swap", default="auto")
    p.add_argument("--vae", default=None, help="H3 video VAE (previews decode with it)")
    p.add_argument("--text_encoder", default=None, help="Qwen3-VL-32B (preview prompts)")
    p.add_argument("--sample_prompts", default=None, help="one prompt per line")
    p.add_argument("--sample_frames", type=int, default=1,
                   help="preview length: 1 = a still, 22 = a ~1 s clip (silent mp4 + the middle "
                        "frame as PNG); the GUI sends 22")
    p.add_argument("--sample_width", type=int, default=768)
    p.add_argument("--sample_height", type=int, default=768)
    p.add_argument("--sample_steps", type=int, default=20)
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--preview_every", type=int, default=0,
                   help="also preview every N steps (0 = raw mod + finished mod only)")
    p.add_argument("--turbo_lora_path", default=None, help="Turbo LoRA for few-step previews")
    p.add_argument("--turbo_lora_strength", type=float, default=1.0)
    p.add_argument("--description", default="", help="stored in the mod as its hint (the loaders can emit it "
                                                        "into the prompt: 'concept_type: description')")
    p.add_argument("--concept_type", default="identity",
                   choices=["identity", "style", "pose_motion", "clothing", "background", "generic"],
                   help="what the mod is (the pack's concept types; default identity)")
    p.add_argument("--token_cap", type=int, default=0,
                   help="thin CLIPS AS MOTION to fit this many tokens the way the node pack's extractor "
                        "thins a clip (near-duplicate frames dropped first, then an even resample); "
                        "photos are never touched. 0 = keep every frame (default). The pack's own "
                        "default is 5120")
    p.add_argument("--audio", choices=["off", "folder", "bundle"], default="off",
                   help="the folder's sound (clip soundtracks and audio files, in file order) through "
                        "the H3 audio VAE — the pack's audio RefMod, a plain encode. 'bundle' = ONE "
                        "file, <name>.safetensors holding the visual mod and the audio mod (the pack's "
                        "version-5 container); 'folder' = two files, the audio one as "
                        "<name>_audio.safetensors. Needs --audio_vae")
    p.add_argument("--audio_max_seconds", type=float, default=30.0,
                   help="how much sound goes in (2 tokens per 1/40 s; 30 s = 2400 tokens)")
    p.add_argument("--audio_concept", default="voice",
                   choices=["voice", "singing", "music_style", "sound_fx", "ambience"],
                   help="the audio mod's concept type, in the pack's terms")
    p.add_argument("--audio_vae", default=None, help="H3 audio VAE (minimax_h3_audio_vae_fp32.safetensors)")
    p.add_argument("--base_model", default="", choices=["", "ref2va", "fl2va"],
                   help="which H3 model the mod is tuned against, written into the file (the GUI passes "
                        "Training Base; blank = read it from the --dit file name)")
    p.add_argument("--holdout_refs", action="store_true",
                   help="hold the reference stills OUT of the optimiser's training set (default: it "
                        "trains on every still, references included)")
    p.add_argument("--ref_cache_dir", action="append", default=None,
                   help="take the references from this cache folder (repeatable) instead of the "
                        "dataset's own — the GUI's Target MP pass; the optimiser's stills stay in "
                        "the dataset caches")
    p.add_argument("--init_from", default=None,
                   help="start from an existing mod file instead of the caches (re-preview it "
                        "with --steps 0, or keep optimising it)")
    return p


def _read_prompts(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        out = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    return out or None


def main():
    from fizgig.utils.device import apply_sim_vram_cap
    apply_sim_vram_cap()
    a = setup_parser().parse_args()
    grid = None if str(a.grid).strip().lower() in ("full", "0", "") else int(str(a.grid).split("x")[0])
    run_refmod(dataset_config=a.dataset_config, output_dir=a.output_dir, output_name=a.output_name,
               dit_path=a.dit, grid=grid, steps=a.steps, lr=a.lr, pull=a.pull, max_refs=a.max_refs,
               seed=a.seed, base_quant=a.base_quant, blocks_to_swap=a.blocks_to_swap,
               vae_path=a.vae, te_path=a.text_encoder, sample_prompts=_read_prompts(a.sample_prompts),
               sample_width=a.sample_width, sample_height=a.sample_height, sample_steps=a.sample_steps,
               sample_seed=a.sample_seed, preview_every=a.preview_every, sample_frames=a.sample_frames,
               turbo_lora_path=a.turbo_lora_path, turbo_lora_strength=a.turbo_lora_strength,
               description=a.description, concept_type=a.concept_type, init_from=a.init_from,
               exclude_refs=a.holdout_refs, token_cap=a.token_cap,
               audio=a.audio, audio_max_seconds=a.audio_max_seconds, audio_concept=a.audio_concept,
               audio_vae_path=a.audio_vae, base_model=a.base_model,
               ref_cache_dirs=a.ref_cache_dir or None, ref_subset=a.ref_subset, clips=a.clips,
               sigma_range=((a.sigma_min, a.sigma_max) if a.sigma_min >= 0 and a.sigma_max > 0 else None))


if __name__ == "__main__":
    main()
