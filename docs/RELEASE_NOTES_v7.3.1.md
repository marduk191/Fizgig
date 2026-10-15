# Fizgig v7.3.1: maintenance

A new version of Fizgig's Qwen Image 2.1 training adapter, more for shaping songs in Repair Studio, and fixes.

## Qwen Image 2.1: training adapter v2

- **Fizgig's training adapter v2** is now the one Fizgig uses for Qwen Image 2.1. Updating downloads it (about 84 MB) and switches Preferences to it if you were on the first version; the first file stays on disk. **Download models for me** and the Download link beside the adapter fetch v2 too. If you chose a different adapter yourself, it's left as it is. It's also on Hugging Face: [ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter](https://huggingface.co/ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter).

## Repair Studio for songs (ACE-Step)

- **Block activity.** While a song plays, each block's name glows with how hard its LoRA pushes at that moment, so you can hear a passage and see which blocks shape it. It follows the playhead whether the song is playing, paused or scrubbed, and costs nothing extra: it's measured during the normal render.
- **Scrub with the playhead.** The playhead has grab handles at the top and bottom of the waveform: drag them to scrub, and a playing song carries on from where you let go. Stop is now Pause while a song plays and Resume once paused.
- **Refine the loop section.** The section has a tab at each end: drag one to move that edge, or drag both out to the ends of the song for the whole song again.

## Fixes

- **Repair Studio, ACE-Step: changing Length re-encodes the prompt.** A song's length is part of its encoded prompt, and a 60-second render after a 30-second one with the same prompt reused the 30-second encoding. Reported by @mabseyuk.
- **LoRA the Explorer:** the four variants are drawn at the same size.
