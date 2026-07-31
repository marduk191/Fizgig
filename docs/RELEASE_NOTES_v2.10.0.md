# Fizgig v2.10.0 — Caption with the model that reads them

## Caption your dataset with the model that reads it

Krea 2's text encoder is **Qwen3-VL-4B** — a full vision-language model. If you have it in
Preferences, the Captions tab can now use it to caption your dataset, and it becomes the default
captioner. It is a substantially better captioner than Florence, and it is the same model that will
read those captions during training.

- **Four task presets, and every one of them is editable** — Training caption, Short caption,
  Detailed description, Exhaustive detail. **Edit instructions…** next to the dropdown opens the
  whole prompt the model is given alongside the image. Rewrite it in plain English and **Save**:
  that preset now uses your wording, every time. Each preset keeps its **own** instruction, so you
  can tune one for products and another for portraits and just switch between them. Edits persist
  between sessions, and **Restore default** puts the shipped text back whenever you want it.
- **Your edited instructions drive auto-recaption too.** If you've rewritten the Training-caption
  preset, mid-run recaptions follow your wording rather than the built-in — so the caption style
  you settle on is the style the run keeps writing. (Attempt two always escalates to Exhaustive
  detail, edited or not.)
- **Your trigger word is respected**, exactly as with Florence.
- **The captioner is released from VRAM** when a job finishes and when you leave the Captions tab.
- Your captioner and task choice are remembered between sessions.

**Klein users: this is for you too.** The captioner is a dataset tool, not a Krea 2 tool. Point
Preferences at the Qwen3-VL encoder and you can caption a Klein dataset with it even if you never
train Krea 2. There's a note in Preferences to that effect.

### Better auto-recaptions

Mid-training auto-recaptions used to hedge — describing a clearly-shown woman as "a person" — and
sometimes opened with filler like "This image shows…". Both are fixed. Recaptions now name the
subject as depicted and start with the description itself.

## Bring your own text encoder

The Qwen3-VL slot takes any Qwen3-VL-4B checkpoint in the ComfyUI layout:

- **`fp8_scaled` — now the recommended download.** It loads and stays fp8-resident: **4.9 GB
  instead of 8.3 GB**, on a model that has to co-exist with the DiT. Previously Fizgig couldn't load
  this file at all and told you the fp8 variant had no working vision path — that turned out to be
  wrong. ComfyUI's `fp8_scaled` quantises only the language layers; the vision tower stays bf16 and
  is entirely intact. It's now properly supported, verified end-to-end including captioning,
  reference images and prompt encoding.
- **bf16** still works exactly as before, unchanged.
- **Community fine-tunes and abliterated builds work too.** Since the same model writes your
  captions, swapping it changes how your dataset gets described — an uncensored or domain-tuned
  build will caption subjects a stock instruct model hedges on. Verified against a third-party
  abliterated fp8 build alongside the official files.

## Fixes

- **Dataset folders containing `[square brackets]` broke training.** A path like
  `D:\shoots\[subject] photos` made the pre-flight report "No caption files found" and refuse to
  start — while the Captions tab read and wrote that same folder perfectly happily. Brackets are
  glob character-class syntax; the pattern was silently matching nothing. Fixed everywhere it
  mattered, including the dataset loader.
- The **LoRA name** now carries a per-family suffix — `_k9b` for Klein, `_krea2` for Krea 2 — and
  switches with the model family instead of leaving a Krea 2 LoRA named `_k9b`. Only the exact
  suffix is touched, so your own naming is left alone, and a paused run is never renamed.
- The **Next-sample override** resolution list now matches the Samples tab, so the higher
  resolutions are available in both places.

## Image Prep

The tab has been reworked for people running their first LoRA — clearer stage ordering, plainer
language about what each step does and when to skip it.

---

**Upgrading:** run `update_fizgig.bat`, or `git pull`.
