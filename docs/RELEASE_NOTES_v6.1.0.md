# Fizgig v6.1.0

Two things finish the RefMod story that 6.0.0 and 6.0.1 started: a mod and its sound are now **one file**, and you can **write prompts that name your references** the way ComfyUI's own text encode node does. Separately, MiniMax H3 LoRA runs change: **the learning rate is no longer yours to pick**. There is also a live progress card on the Training tab, contributed by **@mabseyuk**.

RefMods work with the current [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod) pack, by **@Luisacaotica**. The full guide is [RefMods — how do I…?](https://github.com/shootthesound/Fizgig/blob/master/docs/REFMOD_HOWDOI.md).

## One file for a mod and its sound

Audio mods arrived in 6.0.1 as a second file beside the visual one. They no longer have to be.

- **Bundles are the default now.** With Audio support on, the maker writes the visual mod and the audio mod together as the pack's bundle. The pack's loader lists it as two members, so a slot can take All, Visual or Audio without you managing a pair of files.
- **Two files is still there** if you need it, writing the audio mod as `<name>_audio.safetensors` for older readers. Pick it on the card.
- **Fizgig reads bundles wherever it reads mods.** RefMod Studio lists a bundle's visual member as an ordinary row and renders, sweeps and bakes it.
- **The file says which model it was tuned on.** A mod made against the Reference base and one made against First / Last Frame now carry that in their metadata, along with whether it was a plain encode or tuned, so a folder of mods tells you what each one is for.

## Prompts that name your references

RefMod Studio can now show your mods to the text encoder exactly as the pack's H3 RefMod Text Encode node does, so a prompt written here means the same thing in ComfyUI.

- **Tick Numbered references** and each mod gets a label: a single-frame mod is `<Picture n>`, a stacked or motion mod is `<Video n>`, a bundled audio member is `<Audio n>`. The map beside the tick says which number is which.
- **`+ labels` drops them into the prompt**, so you can write "the woman in `<Picture 1>` stands on the left" and have it land.
- **The ComfyUI settings readout gains the Text Encode paragraph**, which is the values to put on that node.
- The token stream and the picture patches are checked against ComfyUI's own tokenizer, so what the Studio sends is what the node sends.

## A learning rate that sets itself

MiniMax H3 LoRA runs now train on **Automagic v3**, **@ostris**'s self-adjusting optimizer, brought in from [AI-Toolkit](https://github.com/ostris/ai-toolkit) under the MIT licence. Both character presets use it and there is nothing to configure: it reads the direction of its own updates and moves the rate itself, up while they hold steady, down while they alternate. In A/B against the flat rate it replaces it reached likeness sooner and finished ahead, which is why it is the default rather than an option.

- **Leave the Learning Rate box at 1e-6.** Under Automagic that box is a starting point, not the rate. An AdamW number like 2e-4 is far too hot as a start, and Fizgig refuses to launch with one rather than letting you find out twenty minutes in.
- **The Style preset stays on `adamw` at a flat 2e-4.** A style set's images all share the look being learned, so the update signs agree for longer and a self-adjusting rate pushes harder than a style wants. Character presets get the new optimizer; style keeps the measured one.
- **Krea 2 can use it too**, from the Optimizer Type row. It is a choice there rather than the default: on Krea 2 it matched the standard recipe rather than beating it, and took longer to get there. Two things are tuned for that family when you do pick it. Each family of layers finds **its own rate** instead of one compromise for all 264 of them, and the **sign window is 16 steps** rather than 8, so a run of ordinary gradient noise no longer reads as overshoot and drags the rate down.
- **While it owns the rate, other rate controls stand down** and say so in the console: the scheduler, Adaptive LR, per-image adaptive LR and the look-outlier warm-up. The adapter ramp and band multipliers are not applied either. Nothing silently fights it.
- Klein is unchanged and keeps its own optimizer list.

## Also

- **A live Progress card on the Training tab**, by **@mabseyuk**: completion, epoch and step, speed, loss, ETA and preview status, read from the trainers' own output. Under the RefMod family it speaks RefMod, reporting making, optimising with the step count, and saved.
- **Start refuses a folder with uncaptioned photos or clips and names the files**, instead of letting the run begin and quietly leaving them out. The caching pass logs anything it skips for the same reason.
- **The H3 preset formerly called "Lower LR - slower" is now "MiniMax H3 (rank 16, 60 epochs)"**, which is what actually distinguishes it now that neither character preset is given a fixed rate.
- **Fixed:** RefMod Studio reused a No-mod clip rendered without sound after Sound was switched on. Spotted by **@mabseyuk**.
- **Fixed:** a bundle whose audio failed to encode could leave you with no file at all. The audio inputs are checked before the model loads, and the visual mod is written either way.
- **Fixed:** bundled audio was presented to the pack's loader in the wrong order, so a bundle's members did not line up with the slots.
- The RefMod guide covers bundles and says where the file is written.
