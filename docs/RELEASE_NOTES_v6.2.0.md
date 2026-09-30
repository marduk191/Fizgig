# Fizgig v6.2.0

No new features in this one. The last few releases changed a lot about how MiniMax H3 trains — the training adapter, the untrained text token refiner, Automagic setting its own learning rate — and this release is the result of going back and finding out where the good settings actually sit now that all of it is in place. Most of what came back was that Fizgig was still steering people toward the slower, heavier option out of habit. **The defaults have moved, and the two training modes are named and described for what they really do.**

If you train H3, you do not need to change anything. The presets carry it.

## The two training modes are renamed

**Fast is now Default. Ultra quality is now More Blocks.**

The old names sold them on speed, which had people reaching for Ultra expecting a better result. That is not what the difference is. What separates them is how much of the base model they leave alone, so the names say that now, and both carry a description under the dropdown:

- **Default** — high quality, versatile, best at preserving model priors.
- **More Blocks** — less preservation of model priors, high quality, can affect movement ability.

More Blocks is also the honest name: it trains blocks **6-49**, not all fifty. Blocks 0-5 are trained by neither mode, because they deform anatomy and pull the dataset's colour into every render.

## Every preset now ships Default

Including **Style**, which was the last one on the heavier mode and has moved with the rest. Styles measured as training both quicker *and* better on Default, which was the finding that prompted this whole pass.

On a likeness run, the new Default recipe reached **73 by epoch 8**. Earlier H3 runs on this trainer took until epoch 15 and epoch 21 to reach 72 and 74. So it arrives at about the same place, in roughly a third to a half of the epochs, with quicker steps on the way. It then held between 71 and 73 for five more epochs rather than starting to degrade, so there is a wide window to pick a checkpoint from rather than one epoch to catch.

**More Blocks is not a likeness upgrade, Default is king for that.** Where it may earn its place is **training a motion concept specifically**, since it reaches more of the model than Default does. Even there, the regular Default mode may well be more than enough for your movement training — this needs more testing to confirm. Unless you have a reason to reach further, use Default.

Style also moves to **Automagic v3**, so all three presets now let the optimizer set its own rate from 1e-6.

## A style LoRA trained on stills, and the motion is untouched

The clearest thing to come out of this pass is a pair of clips, below: the **first with the style LoRA off**, the **second with it at 1.0**. Same prompt, same seed, same resolution, nothing else changed. The LoRA was trained on **341 stills** of an animated show — no video in the dataset at all. They were captioned on the Captions tab and the run used the **Style preset with nothing changed** — the full settings are below, with the prompt.


https://github.com/user-attachments/assets/b344b90d-a4cd-4ec3-a4bc-c0495b95269b





https://github.com/user-attachments/assets/666fcc76-47f6-4534-853b-e3b922142daf



The look changes completely, from 3D game-cinematic rendering to drawn, cel-shaded animation. The motion does not change at all. Same beats, same timing, same camera moves, same staging: the character turns to camera and raises his hand, drops and turns away, the car goes up, the aftermath settles through the smoke. Shot for shot.

That is the point of training on Default. A style LoRA built entirely from still images had no business knowing how anything moves, and it did not need to — because the blocks it trained left the model's own movement alone.

One thing worth flagging, because it went the other way from what you would fear. The prompt asks for a car speeding through the background of the first shot, and it was the **LoRA'd version that actually moved it** — same seed, same prompt. That is one sample, and it is not a claim that training on stills improves motion. It is simply a pointed illustration that training on stills did not take motion away.

<details>
<summary><b>The whole recipe, and the prompt</b></summary>

**Dataset** — 341 stills of an animated show. No video.

**Captions** — the Captions tab, Qwen with its **Style** captioning preset, and a `zwxem style` trigger word.

**Training** — the **MiniMax H3 Style** preset, nothing changed. The clips use the **epoch 4** checkpoint.

**Prompt**

```
integrated_multimodal_description:

[Shot 1] zwxem style, a medium shot frames a man with dark hair and a serious, intense
expression in a dynamic pose at the centre of the frame, in a dimly lit industrial environment
whose background is blurry and indistinct. He wears a dark, formal-looking jacket with gold trim
over a white shirt. The whole scene is bathed in a pervasive purple light. The camera performs an
arc shot around him with large amplitude at normal speed, keeping the man in the centre of the
frame as the blurred background slides past behind him. His right arm rises, gripping a steampunk
style grenade. He hurls the grenade with a fast overarm throw toward a fast moving car that speeds
through the blurred background, his jacket swinging with the motion.

[Shot 2] At 00:06.000, the camera cuts to a wide shot of the fast moving car in the same
purple-lit industrial environment as the grenade from Shot 1 strikes it and the car explodes, a
fireball bursting outward, orange flame flaring against the purple light, metal panels and dark
debris spinning through the air as the camera shakes strongly.

overall_soundscape: A low industrial hum fills the space under the rising roar of a car engine.
The fabric snaps with the throw, and a metallic clink is followed by a heavy explosion,
shattering glass and debris clattering onto concrete.

non_diegetic_music: N/A
```

</details>

## Voice trains the same blocks as the picture

Audio-only steps were confined to **34-49**, the voice core and its shoulder, because audio gradients beyond it corrupted the visual blocks. Two things that landed since have removed the cause: the training adapter de-distills the base while your LoRA learns, and the text token refiner is no longer trained. The refiner was the leak.

So voice now trains **20-49**, the same blocks as photos and clips. Narrowing it was only costing the voice blocks it could have used.

## Also

- **Off** mode now points at **Blocks to Train in the Other Options section**, which is where it actually lives rather than "below".
- The preset formerly called "Lower LR - slower" is **MiniMax H3 (rank 16, 60 epochs)**, which is what actually distinguishes it now that neither character preset is given a fixed rate.
- The README and the CLI reference describe the modes by what they are for rather than which is faster.

