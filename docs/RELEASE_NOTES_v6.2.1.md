# Fizgig v6.2.1

A maintenance release: three reported problems fixed, one preview trap removed. Nothing changes in how you train.

## Krea 2 full fine-tune: the checkpoint save no longer runs out of RAM

Reported by **@ribawaja** (#143), on a 24 GB card with 96 GB of RAM: the first checkpoint save died with `MemoryError`. The save was holding a second copy of the whole model beside the one it was training, and then building every tensor as bytes before writing the first one — around three copies of a 26 GB model at the peak.

The save now streams, one tensor at a time. Peak memory during a save is the model being trained plus one tensor. The file that comes out is the same as before — same keys, same order, same dtypes — and loads for a continuation exactly as it did. The model copy built at the start of a fine-tune had the same spike and is built the same way now.

One more thing that fell out of it: pointing a fine-tune at an **fp8** checkpoint is refused at the start with a plain message, instead of training for an hour and dying at the first save. Fine-tuning needs the bf16 RAW.

## Problem Images and the Look Consistency Filter work on big datasets

Reported by **@RitschRatsch666** (#140), confirmed in the Look Filter by **@JELSTUDIO**: past roughly 250 images the list went blank while the scrollbar kept moving. Tk stops drawing anything beyond a fixed height, and a thumbnail per row reaches it fast.

Both windows now show **200 rows a page** with Prev / Next. Problem Images also gets a **Show:** dropdown — All, Problems only, or a single verdict — so you can go straight to the stuck images without scrolling at all. The tally at the top still counts every image, and the auto-refresh keeps your page and scroll position.

## The installer refuses a conda Python

Reported by **@mofoni** (#141): installed from inside a conda `base` environment, the app came up with missing fonts and broken symbols. Conda bundles its own Tk, which cannot see the system's fonts, and the installer was building the venv from whatever Python it was run with.

The installer now stops at step 1 if a conda environment is active or the Python is conda's, and prints what to run instead: `conda deactivate`, then the installer with your system `python3`. A venv that was already built inside conda is treated as broken, so the installer offers to recreate it. `FIZGIG_ALLOW_CONDA=1` overrides the check if you know what you are doing. The README says so under the Linux install steps.

## H3 clip previews always carry their sound

Sample length used to offer "56 frames" and "56 frames with sound" side by side. Pick the first and the previews were scrub frames with no clip to play — and a pick made months ago stayed picked. The silent entries are gone: Still, or 22 / 56 / 124 frames with sound. A saved silent choice maps to the sound entry with the same length on the next launch, so nothing to do on your side.

## Pods

The pod guide now says plainly that a RunPod **Global Volume** is not supported (it is object storage, and the pod restarts every few seconds with a fresh password in the log) — use a Volume Disk or a regional Network Volume. Reported by **@aqwarr** (#139).
