# Fizgig v8.9.9: masks from words with SAM 3.1

**Make masks of anything by naming it.** On the Image Prep tab's masks card, choose **SAM 3.1: from words** and type what the masks show ("person", "hair", "the dog", "red dress"), and optionally what to leave out ("glasses", "phone"). SAM 3.1 is Meta's Segment Anything model: it finds whatever you name, with much sharper edges than the person finder, around hair especially (thanks @CognitiveDiffusion for asking, #239). [Masked training](https://github.com/shootthesound/Fizgig/blob/master/docs/MASKED_TRAINING.md)

- **Download it from the card**, once: 1.75 GB, into Fizgig's models folder, where it stays through updates.
- **Preview on 2 random pictures** shows masks over two pictures before you make any; press again for two more, with no repeats until every picture has been shown.
- **Only my subject** works with it too, keeping just the person whose face matches the Look Filter's baselines.
- It runs on the GPU, a fraction of a second a picture once it has loaded. Make masks when nothing is training.
- The masks card is laid out in three steps: how the masks are made (Person finder or SAM 3.1), what they show, and options.

**Fixes**

- **LoRA Royale's Render epochs works again.** Since 23 Oct the button did nothing, for every model (thanks @Kartoffel0, #241).
- **Seed travel's Epoch badge tick works again**: clips had always been saved without the badge.

Everything new in 8.9: [Fizgig v8.9.0 release notes](https://github.com/shootthesound/Fizgig/releases/tag/v8.9.0).

**Updating:** update Fizgig on your computers, and restart your pods.
