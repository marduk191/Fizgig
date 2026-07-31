# Fizgig v2.14.0 — Captioning for style LoRAs

Two new presets in the Captions tab, for training a *look* rather than a subject. Both are
editable like the existing four.

## Why style captions work backwards

A subject caption names everything that varies — viewpoint, pose, clothing, lighting — so you can
steer it later. A style caption does the opposite: the look is in every image, so describing it in
different words image by image scatters it instead of concentrating it. Style captions therefore
describe **only what changes**, and never the style.

That includes lighting, which is the one people get wrong. Caption it and the style only fires
under the lighting it was trained on.

## The two modes

**Style — in X style** ends every caption with a phrase you choose:

> a river flows through a desert with palm trees and antelopes grazing, **in mystyle style**

Edit `mystyle` to whatever names your look — a made-up token or a plain phrase like
`in oil painting style`. The trailing `in … style` form is what gives a made-up word a job; on its
own it's just a noise word the model has to decode from repetition.

**Style — content only** leaves the look unnamed and lets your trigger word carry it, if that's
already your workflow.

Use one or the other, not both, or the style gets named twice.

## Both modes

- **Never describe the look** — no medium, brushwork, texture, colour grade or lighting.
- **Name what's actually there** — a mountain range, a teapot, an empty street. The subject presets
  can only say "a woman" or "a man", which is why a landscape captioned with one comes back
  wearing no clothing.
- **30–50 words**, natural language, consistent order.

Tested against a layered paper-cut landscape set, where the style is the kind of thing a captioner
describes without meaning to.

## Also

- The captioning doctrine in `docs/CLI.md` now covers style datasets.

---

**Upgrading:** run `update_fizgig.bat`, or `git pull`.
