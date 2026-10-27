# Fizgig v8.9.5: faster training on 40 GB+ cards

- **Training is faster on cards with 40 GB or more**, which is most pods (A100, A40, A6000, L40S, H100, RTX PRO 6000). Auto now turns gradient checkpointing off when the run fits with room to spare: 13–27 % faster per step, measured on Krea 2, Qwen Image 2.1, MOSS-Image, Z-Image and MiniMax H3 (stills and clips). Where it doesn't all fit, the last blocks go without it instead: 2–21 % faster. Previews keep at least 4 GB free either way. Cards of 32 GB and under train as before.
- **MiniMax H3's Auto plan sizes your dataset correctly when other families have trained on the same folder.** It read their caches as its own and planned for a dataset about 4× larger: more block swap than needed, NF4 instead of INT8, and trimmed previews.
- **Fizgig no longer asks your local network every few seconds.** It searches for your other computers only while the Helpers or Pods tab is open, and **Preferences → Network** has two switches: **Find Fizgig on my other computers** (off: it neither searches nor answers) and **Check for new releases** (off: no check at start-up). Pods and computers added by their address work either way (thanks @VRAM-Hoarder).
- **LoRA Isolation's side-by-side pictures use Image Prep's face crops first**, uncropped, then photos framed on the face without zooming. A new tick, **Side-by-side pictures: face crops first** (on), turns it off. The Subjects card's switches sit in one row on a desktop (thanks to Reddit user Ascert).
- **Pictures with a rotation flag (EXIF orientation) are read upright everywhere**: Image Prep, captioning, face finding, training, references and thumbnails. A picture that was cached sideways is re-cached upright on your next run (thanks @mkoryak).
- **The preview-model tick reaches a run that's already going.** Ticking or unticking Qwen's Turbo or Klein's Distilled mid-run changes the next preview, not the next run (thanks @Aymenbayoss).

Everything new in 8.9: [Fizgig v8.9.0 release notes](https://github.com/shootthesound/Fizgig/releases/tag/v8.9.0).

**Updating:** update Fizgig on your computers, and restart your pods.
