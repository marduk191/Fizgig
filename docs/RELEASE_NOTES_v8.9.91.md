# Fizgig v8.9.91: maintenance

- **Edit masks, in Fizgig.** The Image Prep tab's masks card has a fourth step, **Edit masks**: each picture of the training folder with its mask over it, one at a time, to check and fix by hand. [Masked training](https://github.com/shootthesound/Fizgig/blob/master/docs/MASKED_TRAINING.md)
  - **Brush** paints what the LoRA learns, **Eraser** takes it away, with Size, Softness for feathered edges and Strength for grey (learned partly).
  - **SAM click**: click a thing to add it to the mask; right-click or Alt+click takes it away. It uses SAM 3.1, downloaded on the same card.
  - Scroll to zoom, Space+drag to move around, Undo / Redo, and Fill, Clear or Invert the whole mask. Three views: picture and mask, mask only, picture only.
  - A status beside **Save now** says whether there are unsaved changes. Moving to another picture, changing the filter, closing Edit masks or leaving the tab saves too; a mask saved all black is deleted, so that picture trains as normal. Phone photos are shown and saved upright, as training reads them.
- **Only my subject can be ticked as soon as a Look Filter scan finishes**, not only after leaving the tab and coming back (thanks @jordanjay29, #242).

Masks from words with SAM 3.1 came in 8.9.9: [Fizgig v8.9.9 release notes](https://github.com/shootthesound/Fizgig/releases/tag/v8.9.9).

**Updating:** update Fizgig on your computers, and restart your pods.
