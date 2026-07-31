# Fizgig v2.15.0 — One style caption preset, and it's the good one

The Captions tab's three style presets are now **one**, replaced by an instruction tested across
real training runs on both Klein and Krea 2. The style comes through fast.

## How to train a look

1. Set your **trigger word** — this is what the style binds to.
2. Pick the **Style** preset.
3. Caption the folder, then train as normal.

The preset describes the *contents* of each image in detail and never the style itself, so your
trigger word is the only thing every caption has in common. That's what makes it the thing the LoRA
learns.

> `mystyle, a river flows through a desert, palm trees along the banks, antelopes grazing on the
> far side beneath a pale sky`

Nothing to edit, no phrase to fill in.

## What changed

The two `in X style` presets and the previous content-only one are gone. If you had one selected,
the tab falls back to the default Training caption — pick **Style** again.

Captions are longer now, so the preset's token budget went up to match; nothing gets cut off
mid-sentence.

## Also

- The style captioning guidance in `README.md` and `docs/CLI.md` matches the new preset.

---

**Upgrading:** run `update_fizgig.bat`, or `git pull`.
