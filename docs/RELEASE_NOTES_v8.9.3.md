# Fizgig v8.9.3: maintenance

- **MOSS-Image workbench: render on SFT, or as your Samples tab does.** While the Samples tab previews on your Base, Studio, LoRA Royale and the Profiler show **Render with:** **Use MOSS-Image SFT** (the default: what most people generate with), or **Follow Samples tab settings**: your Base, with the SFT look LoRA at 1.5 when **Give Base previews the SFT look** is ticked, at the Samples tab's steps and CFG. The look LoRA only changes how renders look: your LoRA and donor still work as before, every slider acts on them alone, and a save never includes the look LoRA. One choice per model, shared by all three tabs.
- **Move a run takes clips and songs with it**, so a MiniMax H3 or ACE-Step run no longer arrives with only its pictures; a training share's folder does the same (thanks @mabseyuk).
- **Captions edited over there come back in the run's own extension** (such as `.caption`), not only `.txt` (thanks @mabseyuk).
- **A thumbnail set to `off` no longer stops a run from moving** (thanks @mabseyuk).

Everything new in 8.9: [Fizgig v8.9.0 release notes](https://github.com/shootthesound/Fizgig/releases/tag/v8.9.0).

**Updating:** update Fizgig on your computers (helpers and moves need the same version on each), and restart your pods.
