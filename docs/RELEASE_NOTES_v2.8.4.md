# Fizgig v2.8.4 — Krea 2 previews without the model shuffle

## ⚡ New preview engine: samples render on the training model

Until now, every Krea 2 in-training preview loaded the ~13 GB Turbo checkpoint and parked the
~14 GB training model to CPU to make room — a lot of memory traffic for one image.

Previews now render **on the training model itself**, using Comfy-Org's official **Turbo
distillation LoRA** (rank 64) switched on just for the render. RAW + this LoRA at strength 1.0
behaves as the Turbo model, so the settings are identical — 8 steps, CFG-free — and your live
LoRA (plus a Context LoRA if you're using one) stays active in every preview, exactly as it
would be deployed. Between previews the LoRA is disabled and costs nothing.

- **Auto-download** — the ~470 MB LoRA fetches itself on update (`update_fizgig.bat`) or at
  first use, and fills in the Preferences path. Nothing to set up.
- **Samples tab choice** — a new *Preview engine* picker (Krea 2 only). *RAW + Turbo LoRA* is
  the default; the classic *Turbo model* mode is still there and still works.
- **Graceful everywhere** — missing LoRA falls back to the Turbo checkpoint; neither present
  just disables previews with a clear message. Training is never blocked.

The Turbo checkpoint is still used by the workbench tools (Repair Studio / Explorer / Royale),
so keep it around if you use those.

## 🚀 The v2.8.3 CPU fix turns out to be a speed fix too

v2.8.3 stopped an all-core CPU burn during training (#18) and we called it "no change to step
times". Measured since on real runs: it **speeds up training as well** — up to **~1.6× the
step rate**, because the spinning threads were competing with the ones that launch GPU work.
Nothing to do; if you're on v2.8.3 or later you already have it.

## ✨ New preset: Krea 2 Ultra Fast

Rank 8 with **Adaptive LR** on an aggressive floor (min 2e-4, max 4e-4) and 20 epochs — fewer
epochs to a usable LoRA, handy for quick results and fast dataset tests. Everything else
matches **Krea 2 Defaults**, which stays the standard pick (and is what loads automatically
when you switch the Training tab to Krea 2).
