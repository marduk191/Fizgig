# Fizgig v2.8.5 — The app explains itself

A polish release aimed squarely at first-time users — plus one installer bug whose fix
matters to everyone who installed fresh recently.

## 🛠️ Fresh installs: lingering console window + blocked updates — fixed

The installer was overwriting the silent launcher with an old console-attached version. Two
symptoms: a console window followed the app around and ended with "Press any key to continue"
after closing Fizgig, and — worse — `update_fizgig.bat` could refuse to update (`git pull`
balked at the modified file). Both fixed; the update script also **heals installs that
already have the old launcher**, so just run `update_fizgig.bat` as normal.

## 🖼️ Image Prep, redesigned for humans

The old Prep Mode pane generated one support question above all others: *"does this touch my
folder?"* — and its Result note could actually answer it wrongly. Reworked from scratch:

- **Three plain-language choices** instead of a dropdown — *Resize + face close-ups*
  (recommended for people), *Resize only*, *Face close-ups only* — each with a one-line
  explanation, including the advice that face close-ups want **high-res originals**, not
  images already shrunk to training size.
- **"Your originals"** is now an explicit choice: keep them safe in an `originals` subfolder
  (the new default) or replace them. The confusing output-folder box is gone — everything
  lands in your training folder, always.
- **A "What will happen" box** states the outcome before you click: live image count, sizes,
  where files go, and an explicit *"nothing is deleted"* when that's true. It even warns when
  your source images are too small for sharp face close-ups.
- **One clear Run button** — "✨ Prepare 34 Images Now" — with the face-detection test
  reframed as what it is: optional, safe, writes nothing.

## ✂️ Captions tab: cull and inspect

- **Remove button on every image card** — moves the image + its caption to a `removed`
  subfolder (never deletes). Made for the post-prep pass: eyeball the face close-ups, remove
  the soft ones before captioning.
- **Resolution shown under every filename** — a too-small face crop announces itself as
  `(180×240)` long before the blur shows in a thumbnail.
- Edit dialog buttons no longer get cut off on portrait images; cards keep their buttons on
  one level and captions use the card's full width. The little-understood Select checkbox
  (and its Caption Selected button) retired — Regenerate (AI) in the Edit dialog covers it.

## 🚦 Start tab: the model warning respects your setup

The "model files not configured" banner is satisfied by **either** family now — Klein's
paths, or Krea 2's training trio (the Turbo checkpoint is optional since previews moved to
the Turbo LoRA). Train only Krea 2 and it leaves you alone. And if you want it gone
regardless: **"Don't show this again"**.

## 🧰 Workbench remembers your setup

Repair Studio and LoRA the Explorer now remember every Setup field across restarts — LoRA
paths, prompt, seed, resolution, reference image, intensity, mutations, structure, all of it.
The app also saves your settings on close, so a last-second edit is never lost.

## 🏷️ Krea 2 drops the "experimental" label

Everywhere — the Training tab's model selector, Preferences, and the family switches in
Repair Studio, Explorer, Profiler, Extract and Royale. It has earned it: native trainer,
full workbench support, the per-image loss watch, and the new preview engine all ship and
are in daily use. Saved presets and settings carry over automatically.
