# Fizgig v6.3.0

MiniMax H3 LoRAs trained on photos come out sharper, with better likeness: the training adapter now defaults to Circlestone's. Also in this release: two reported problems fixed.

## H3: Circlestone's training adapter is the new default

The training adapter is a frozen LoRA that rides along on every H3 training step, so your LoRA learns your subject rather than undoing the model's distillation. It is off for previews and never saved into your LoRA. **@ribawaja** suggested trying [Circlestone's adapter](https://huggingface.co/circlestone-labs/MiniMax-H3-Image-Training-Adapter) in #142. In A/Bs on the same data it trained sharper LoRAs with better likeness and closer prompt-following than Ostris's, and one file works on both the fl2va and ref2va bases.

The Training tab's tickbox is now a **Training adapter** dropdown:

- **Circlestone — best for photos** (the default)
- **Ostris — best for videos**: on an all-video style dataset it learned the look about three times faster
- **Off**

For a mixed dataset, choose by whether the photos or the videos are the priority. If the dataset is all clips, the Training tab suggests Ostris. It never switches for you.

**What to do:** nothing. The updater downloads Circlestone's file (~620 MB) and fills in its new row in Preferences. If you updated another way (a pod, Linux, or a plain `git pull`), the first H3 run downloads it once before it starts. Ostris's two files stay in Preferences for when you pick Ostris.

## AMD on Windows: the status bar leaves the GPU alone

Reported by **@Linkram** (#145). On Windows with ROCm, the VRAM readout in the status bar started the GPU up inside the GUI and queried it every second, even with nothing running. It now reads only from the Windows performance counter, and shows "VRAM stats unavailable" when there's no reading. NVIDIA cards are unaffected.

## The updater works on Japanese Windows

Reported by **@nuko-masshigura** (#147). `update_fizgig.bat` failed on systems using the Japanese code page (CP932). The batch files were stored with Unix line endings and contained a non-ASCII character. They now always check out with Windows line endings, in git and in the ZIP download, and contain plain ASCII only.
