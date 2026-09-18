# Fizgig v5.8.1

A Krea 2 discovery you can use today, and one fix.

## Text fusion presets for Krea 2 in Repair Studio

We've discovered that there is more to be had from Krea 2's four **text fusion** blocks: boosting their trained weights after the fact gives a LoRA a real lift, and if you want to get even more out of your LoRAs these presets are worth trying. Load any Krea 2 LoRA in Repair Studio, open the preset box and pick **✨Text fusion ×2 (experimental)** or **✨Text fusion ×3 (experimental)**: the four text-fusion sliders go to the multiplier, every other block stays where it was.

Measured across several LoRAs, ×3 lifted the detail meter and the likeness meter on every one, with the composition unchanged and the output clean. One caution: the boost rewards a LoRA that still has room to grow. If a LoRA is overtrained, or close to it, increasing these will have a negative effect rather than a positive one, so judge it on the meters and step back to ×2 or Reset All if it goes the wrong way.

To see the meters, click either preview image in Repair Studio. The full-size compare opens with a strip along the bottom that reads each meter as **baseline → tweaked**, so you can watch Detail move as you change a preset. Likeness needs a photo to compare against: press **📷 Reference…** on that strip and pick a clear shot of the person from your training set, and the Likeness meter fills in next to Detail. The strip updates live with every slider or preset change.

**Save Repaired LoRA** bakes the boost in, so the saved file works in ComfyUI and everywhere else at strength 1.0 with nothing extra to load. Reset All puts the sliders back.

This came out of a Reddit thread advising Krea 2 trainers to exclude the text fusion layers. On Fizgig's recipe the opposite holds: those layers carry real signal, and giving them more is a free win.

## Fixed

- **A "Finish one category early" epoch could follow you to the wrong dataset.** Load Settings From Last Train after a mixed voice-and-video run restored the stop epoch onto a photo-only run, where the row is hidden but the flag still applied, so the samples froze from that epoch while the timings looked normal. The setting now only reaches the run when the dataset is genuinely mixed. Thank you [@pauldegroot](https://github.com/pauldegroot) (#136).
