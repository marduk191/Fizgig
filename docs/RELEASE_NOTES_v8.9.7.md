# Fizgig v8.9.7: maintenance

- **Masked training: prior preservation outside the mask.** On the Training tab, **Masked training** card → **Outside the mask, keep the model's own picture** (Off, 0.5, 1 or 2). With it on, outside the mask the LoRA learns to draw what the model itself would there, so backgrounds and the model's look stay its own instead of drifting toward your photos'. It still works with **How much outside the mask counts** at 0. Each step takes longer, from a few percent to about half again depending on the model. For LoRAs; sliders and fine-tunes train without it. [Masked training](https://github.com/shootthesound/Fizgig/blob/master/docs/MASKED_TRAINING.md)
- **Each run's log is saved as a file** in `user_data/logs`, named by date, time and LoRA, so a run that failed can be read after a restart. The newest 20 are kept, and a failed run's last line in the log box names its file (thanks @vivi-gomez, #240).
- **A line under ArcFace likeness scores** (Samples gallery, LoRA Royale, Look Consistency Filter) says peak ArcFace is about 72%: that's what two photos of the same person score against each other, so no LoRA gets near 100%.
- **Masked training's folder check and a training share measure a rotated photo upright**, as training does, so a phone photo's mask is no longer reported as the wrong shape (thanks @mabseyuk).
- **Download models for me also fetches Make masks' person finder**, with the face model and the other helpers.
- **The Profiler's Open in Studio button is gone.** Its block suggestions rarely helped on the newer models. Studio still shows a LoRA's profile when it loads it.

Masked training came in 8.9.6: [Fizgig v8.9.6 release notes](https://github.com/shootthesound/Fizgig/releases/tag/v8.9.6).

**Updating:** update Fizgig on your computers, and restart your pods.
