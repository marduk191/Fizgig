# Fizgig v6.0.0

RefMods for MiniMax H3 - Train them easily, analyse them with the Refmod Studio Tab, tweak them and use some new training modes for them.

A RefMod is your reference photos saved as one small file that the [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod) nodes (by **@Luisacaotica**, with a mod library and guide from **@malcolmrey**) load like a LoRA and feed to H3's reference path. No training run, a file in minutes, and the basic kind needs no captions. Fizgig now makes them, and adds the one thing no other maker has: it can tune the file against H3 itself. The full guide is [RefMods — how do I…?](https://github.com/shootthesound/Fizgig/blob/master/docs/REFMOD_HOWDOI.md).

## Making a RefMod

Pick **MiniMax H3 RefMod** in the Base Model selector on the Training tab, choose a preset, press Start. The file lands in the family's Output folder, so point that at ComfyUI's `models/refmods` once and every mod is ready to load.

- **Five presets.** For a person: the **community recipe** (the default, a plain encode of 8 photos at 1 MP, the same file the pack's own extractor makes), **Fizgig recipe lite** (16 photos at 0.5 MP, tuned against the model, about the library's token cost at generation) and the **Fizgig recipe** (the same at 1 MP, for a little more sharpness and smoother skin at a heavier file). For a look rather than a person: **style, community** (the pack's concept recipe, every still pooled to a small grid) and **style, high fidelity** (8 stills at 1 MP, full resolution, for texture, grain and brushwork).
- **Tuned against the model.** With Steps above 0 the H3 model is loaded and frozen and the file itself is tuned so H3 reproduces the person from it. Measured on same-seed clips: on shots like your photos the plain encode and the tuned file are level; on a look the photos never showed, the tuned file is well ahead, its worst frame above the plain encode's best. In both cases skin renders with visibly less noise and fine detail sharper. A few minutes on a 32 GB card; runs on 16 and 24 GB cards without streaming.
- **No captions for a plain encode.** Steps 0 reads the photos alone and skips the text-encoder step. The tuned presets train against your captioned stills.
- **Either H3 model.** Training Base picks the model the mod is made for, Reference or First / Last Frame. A mod does best on the model it was made for, and holds the person about as well on either when it is. Make one per model if you use both.
- **Clips as motion.** A clip in your folder can enter as its sharpest still (the face, the identity choice) or as motion, every latent frame, the pack's video reference for a dance or a camera move. Prepare clips with Gizmo first.
- **Prompt hints.** A description and concept type on the card go into the file; RefMod Studio's **+ mod hints** and the pack's loader put "concept: description" into the prompt, which the pack's guide says a mod needs to anchor on.
- **Crops keep the face.** References are placed on one canvas and any photo that must be cropped is cropped around its face, never tighter than the canvas needs.
- The file is a standard RefMod: the pack's own layout, nothing special to its nodes.

## RefMod Studio

A new tab, in the order you work:

- **Setup**: the mods folder, the model (Reference or First / Last Frame, the same choice you make in ComfyUI), Load base once.
- **Mods**: one mod per row with its strength and copies; type in the picker to narrow a long list. Add a second row to see two mods together, a character and a style.
- **Apply**: every dial is one of the pack's node settings, named in the heading and explained in a sentence — strength (retention), shuffle, fade across the clip, change during the render — with both curves drawn as you set them. Nothing here edits a mod file.
- **Render**: prompt, seed, length, size, steps, Turbo, then Render. With mods appears on the left the moment it is done, No mod on the right on the same seed. **Render sweep** steps one dial through its useful values as a row of 22-frame clips; click a chip to play it. Clips open in the player with sound.
- **Actions**: **ComfyUI settings** copies the exact values for the pack's nodes; **Bake as new RefMod** writes a new file with a row's strength, the master strength, its copies and the frame curve folded in, so it loads plainly at 1.0. Save and Load setup keep the whole tab.

The numbers are the pack's own maths, so what you see in the Studio is what the nodes will do. The IDLE/BUSY light follows the Studio's work like the other studios.

## Also

- **Fixed:** a model family you had never set an Output folder for inherited the folder the previous family was using. It now starts at `output_loras` inside Fizgig.
- **Fixed:** tooltips wrap instead of running off the screen.
- The window is a little wider for the new tab.
