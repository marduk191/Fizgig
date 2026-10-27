# Fizgig v8.9.6: masked training

**Train on the part of each picture that matters: the person, not the room around them.** A black-and-white mask beside a picture says which part counts. White is trained fully, grey partly, and black only as much as you choose. The picture itself still trains whole, so the LoRA sees your subject in their scene; it just stops copying the scene. In our tests of character LoRAs, masks significantly raised both the best likeness a run reached, by eye and in likeness scores, and how quickly it got there, on every recent model. It seems especially effective on MOSS-Image with its default Clean-end share, Slide. It works for Klein, Krea 2, Qwen Image 2.1, Flux 3, MOSS-Image, Z-Image, SDXL and Anima (photos; not clips or songs). [Masked training](https://github.com/shootthesound/Fizgig/blob/master/docs/MASKED_TRAINING.md) (thanks @Turmoil1112 for asking, #203)

![Make masks, on the Image Prep tab](https://raw.githubusercontent.com/shootthesound/Fizgig/master/assets/masked_training/make_masks.png)

![The Masked training card, on the Training tab](https://raw.githubusercontent.com/shootthesound/Fizgig/master/assets/masked_training/training_card.png)

- **Turn it on** on the Training tab: **Masked training** card → **Train on the parts your masks show**. The line under it says how many pictures have a mask; pictures without one train as normal.
- **Make masks for a whole folder in one click**, on the Image Prep tab: the person, the person without their clothes, just the face and hair, the body and clothes without the face, the body's skin only, just the clothes, everything but the faces, or just the background (for a place or scene LoRA). It runs on the CPU, a fraction of a second a picture. The first time, it downloads Google's person finder (16 MB).
- **Only my subject, when there are several people.** Fizgig finds your subject by their face, with the Look Filter's 3 baselines; everyone else counts as background.
- **Paint your own, or bring them from another trainer.** A mask is `photo-masklabel.png` beside `photo.jpg`, the usual naming, so masks from trainers that use it work as they are. Masks already in a folder are kept unless you tick **Replace masks already there**.
- **Choose how much the background counts**: **How much outside the mask counts**, 0.1 by default, keeps a little of the scene. 0 ignores it completely.
- **Masks go with their pictures**: Remove on the Captions tab, Move Marked in the Look Filter, Image Prep's resize and face crops, Move a run and a training share all take a picture's mask along. **Show mask** on the Captions tab shows a picture's mask.
- **A mask that can't be used is named**, under the tick and in the training log, and its picture trains without one: an all-black mask, or one that isn't its picture's shape.
- **The saved LoRA says it was masked**: its metadata carries `ss_masked_training`, `ss_mask_floor` and how many pictures had a mask.

**Updating:** run update_fizgig on your computers (Make masks needs one new package, ai-edge-litert), and restart your pods.
