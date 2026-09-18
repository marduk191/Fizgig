# Fizgig v6.0.1

RefMods for MiniMax H3, made with the model in the loop, and RefMod Studio to test them. Make them from photos, from clips prepared with Gizmo, and with an audio mod of the folder's sound alongside; tune them against H3 itself; try them side by side with the base before ComfyUI. This release follows 6.0.0 closely, so the whole RefMod story is here in one place.

A RefMod is your reference photos saved as one small file that the [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod) nodes (by **@Luisacaotica**, with a mod library and guide from **@malcolmrey**) load like a LoRA and feed to H3's reference path. No training run, a file in minutes, and the basic kind needs no captions. Fizgig makes them, and adds the one thing no other maker has: it can tune the file against H3 itself. The full guide is [RefMods — how do I…?](https://github.com/shootthesound/Fizgig/blob/master/docs/REFMOD_HOWDOI.md).

## Photos first

A folder of good stills is the main event and makes the best mod. Pick **MiniMax H3 RefMod** in the Base Model selector on the Training tab, choose a preset, press Start. The file lands in the family's Output folder, so point that at ComfyUI's `models/refmods` once and every mod is ready to load.

- **Five presets.** For a person: the **community recipe** (the default, a plain encode of 8 photos at 1 MP, the same file the pack's own extractor makes), **Fizgig recipe lite** (16 photos at 0.5 MP, tuned against the model, about the library's token cost at generation) and the **Fizgig recipe** (the same at 1 MP, for a little more sharpness and smoother skin at a heavier file). For a look rather than a person: **style, community** (the pack's concept recipe, every still pooled to a small grid) and **style, high fidelity** (8 stills at 1 MP, full resolution, for texture, grain and brushwork).
- **Tuned against the model.** With Steps above 0 the H3 model is loaded and frozen and the file itself is tuned so H3 reproduces the person from it. Measured on same-seed clips: on shots like your photos the plain encode and the tuned file are level; on a look the photos never showed, the tuned file is well ahead, its worst frame above the plain encode's best. In both cases skin renders with visibly less noise and fine detail sharper. A few minutes on a 32 GB card; runs on 16 and 24 GB cards without streaming.
- **No captions for a plain encode.** Steps 0 reads the photos alone and skips the text-encoder step. The tuned presets train against your captioned stills.
- **Either H3 model.** Training Base picks the model the mod is made for, Reference or First / Last Frame. A mod does best on the model it was made for, and holds the person about as well on either when it is. Make one per model if you use both.
- **Prompt hints.** A description and concept type on the card go into the file; RefMod Studio's **+ mod hints** and the pack's loader put "concept: description" into the prompt, which the pack's guide says a mod needs to anchor on.
- **Crops keep the face.** References are placed on one canvas and any photo that must be cropped is cropped around its face, never tighter than the canvas needs.
- The file is a standard RefMod: the pack's own layout, nothing special to its nodes.

## Video clips too

Prepare clips with **Gizmo** first, the way you would for H3 training: it cuts them to H3's frame grid at the right size, and its mute option marks the clips whose sound you do not want.

- **Clips as stills or as motion.** A clip in your folder can enter as its sharpest still (the face, the identity choice) or as motion, every latent frame, the pack's video reference for a dance or a camera move.
- **Clip token cap.** Motion clips can be thinned to a token budget the way the pack's extractor thins a clip: near-duplicate frames go first (a held shot is mostly those), then the rest are spread evenly down to what fits. Photos are never touched, whatever the cap, and it is off unless you set it.

## And audio

- **Audio mods.** A third row on the card, off by default. Turn it on and the maker writes a second file, `<name>_audio.safetensors`, alongside the visual mod: the folder's sound, each clip's soundtrack and any audio file in file order, through the H3 audio VAE. It is the pack's audio RefMod, a plain encode, with a kind in the pack's terms (voice, singing, music style, sound effects, ambience) and a length you choose. A clip Gizmo marked mute lends no sound, the same rule as H3 training, so you choose which clips the mod hears. It reads the Audio VAE path regular H3 training uses from Preferences. Load it in the same loader slot as the visual mod with components set to Audio, or on its own. The pack's own tests carried music across but not a speaker's voice, so treat a voice mod as an experiment.

## RefMod Studio

A tab, in the order you work:

- **Setup**: the mods folder, the model (Reference or First / Last Frame, the same choice you make in ComfyUI), Load base once.
- **Mods**: one mod per row with its strength and copies; type in the picker to narrow a long list. Add a second row to see two mods together, a character and a style. Audio mods play in ComfyUI; the Studio lists the visual ones and says how many audio mods sit in the folder.
- **Apply**: every dial is one of the pack's node settings, named in the heading and explained in a sentence — strength (retention), shuffle, fade across the clip, change during the render — with both curves drawn as you set them. The fade directions mean what the current pack says they mean: "concept at start" is full at the start fading to nothing, "concept at end" the rise. Nothing here edits a mod file.
- **Render**: prompt, seed, length, size, steps, Turbo, then Render. With mods appears on the left the moment it is done, No mod on the right on the same seed. **Render sweep** steps one dial through its useful values as a row of 22-frame clips; click a chip to play it. Clips open in the player with sound.
- **Actions**: **ComfyUI settings** copies the exact values for the pack's nodes; **Bake as new RefMod** writes a new file with a row's strength, the master strength, its copies and the frame curve folded in, so it loads plainly at 1.0. Save and Load setup keep the whole tab.

The numbers are the pack's own maths, so what you see in the Studio is what the nodes will do. The IDLE/BUSY light follows the Studio's work like the other studios.

## Also

- The RefMod card's rows now fit at the window's minimum width, and the token readout beside References says what the references cost at generation rather than reading as a cap.
- The docs credit the pack's author correctly: the nodes are by **@Luisacaotica**.
- **Fixed:** a model family you had never set an Output folder for inherited the folder the previous family was using. It now starts at `output_loras` inside Fizgig.
- **Fixed:** tooltips wrap instead of running off the screen.
- The window is a little wider for the RefMod Studio tab.
