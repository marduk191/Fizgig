# Fizgig v4.2.1 — maintenance release

Small, targeted fixes on top of v4.2.0. Nothing changes in how you train.

## Fixed

- **24 GB cards: MiniMax H3 training no longer crashes at step 0** when an in-training
  preview needs to briefly park the base model under int8 block swap. The park was
  corrupting the swapped blocks' staging memory, surfacing as a confusing
  `CUDA error: invalid argument` far from the real cause. Thanks **@dewwwey** for the
  excellent diagnosis and fix (#84).
- **RunPod pods no longer show a false "Update Available" banner** on a freshly built,
  fully up-to-date pod, and the About screen now shows a readable version
  (e.g. `master @ 37c0c2f`) instead of a bare commit hash.
- **Pod boot is sturdier**: if the boot-time update fails, the pod re-clones the app fresh
  (keeping your models) instead of carrying on with old code.

## Changed

- **✨ MiniMax H3 Style preset now trains at 2e-4.** The preset originally shipped at a
  gentler 1e-4, but real style runs found the standard rate is what style actually needs —
  the gentler rate just trains slower for no benefit. Reload the preset to pick it up.

## Upgrading

Nothing to do — settings, models, caches and presets are untouched.
