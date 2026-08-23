# Fizgig v4.3.0 — AMD support arrives

Fizgig now trains on AMD Radeon cards. Plus: 16 GB cards get reference-mode training, the
Repair Studio gets a side-by-side compare view with quality metrics, and the MiniMax presets
got a rethink.

## AMD ROCm support — thanks @scryptio (#53)

Fizgig runs on AMD Radeon with ROCm — RDNA1 through RDNA4, Strix Point / Halo, and Instinct.

- **Windows** is the supported path: install Python 3.12, double-click `install_fizgig_rocm.bat`,
  launch with `run_fizgig_rocm.bat`. The installer detects your GPU and pulls the right wheels.
- **Linux** is available but highly experimental — driver resets and crashes are common on newer
  cards. Use Windows ROCm or NVIDIA Linux for production training.
- The status-bar VRAM readout works on AMD too, and RDNA4 cards get a known ROCm GEMM slowdown
  worked around automatically.
- Full install details are in the README. NVIDIA installs are completely untouched — the AMD
  path is separate files, separate venv steps.

This was a lot of work by @scryptio, tested along the way by @tsubasasora on Linux and sharpened
by @FNGarvin and @taisunyoung in the PR thread. Thank you all.

## 16 GB cards: identity distillation now fits — thanks @rintic-13 (#79)

Reference-mode caching (the teacher for identity distillation) used to peak at ~26 GB — out of
reach below a 32 GB card. It now streams the text encoder layer by layer and peaks at ~12.7 GB,
with output verified bit-for-bit identical to the old path. Nothing to configure: Fizgig
measures your free VRAM and streams only when the resident path wouldn't fit. This closes #74 —
@rintic-13's second major contribution to the 16 GB story, after the int8 streaming in v4.0.

## Repair Studio: side-by-side compare with quality metrics

Click either preview image (or the new **⧉ Compare + Metrics** button) for a full-size
side-by-side of baseline vs tweaked, updating live as you drag sliders. Under it, a metrics
strip that quantifies what your sliders did:

- **Likeness** — set a reference photo of your subject (📷 button) and both renders get an
  ArcFace score against it, with the change shown as you tweak.
- **Patch-grid score** — overbake's earliest visual tell is the model's own patch grid showing
  through; this measures it before your eyes can see it.
- **Face texture** — catches both failure directions: plastic skin and fried detail.
- **Clipped pixels and saturation** — blown highlights and oversaturation.

Also in the Repair Studio this release:

- **Picking a LoRA no longer triggers anything.** Choose a new primary or donor, change the
  prompt, seed and sliders — the Start button arms as **Update** and one click does the swap
  and render.
- **Presets now save sliders only**, per model family — loading one no longer overwrites your
  prompt or LoRA paths.
- The Reference image row is hidden under MiniMax (H3 isn't an edit model).

## MiniMax H3 presets, reshuffled

- **✨ MiniMax H3 Fast is now the default** — it's what loads when you pick the family. It also
  trains 50 epochs now (was 40).
- The rank-16 preset is renamed **✨ MiniMax H3 (Lower LR - slower)**, with a note beside the
  dropdown: more suitable for larger datasets with longer trains.
- The Style preset follows Fast's recipe on the measured style blocks, as before.

## Fizgig speaks Korean — thanks @ssain3d-lgtm

The first community translation of Fizgig:
[a full Korean localization](https://github.com/ssain3d-lgtm/Fizgig-Korean-Translated-Ver) of
the app and Gizmo, around 2,000 UI strings. It's an add-on that doesn't touch a single Fizgig
file — it survives updates, keeps you on official releases, and uninstalls by deleting one
file. Beautifully engineered, and genuinely lovely to see Fizgig reaching people in their own
language. Linked in the README under Getting started.

## Upgrading

Nothing to do. Settings, models, caches and presets are untouched. MiniMax users will see Fast
selected by default on the Training tab — your own saved presets are exactly where you left them.
