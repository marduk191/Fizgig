# Fizgig v8.1.2: edits with extra pictures, all three edit models

**Edit LoRAs can use extra pictures** on Flux 3 Image Klein, Klein 9B and now Qwen Image 2.1: an edit that brings something from other pictures into a photo, such as a person, an object or a place.

- On the Training tab's **Edit pairs** card, pick **The photo + extra pictures** under **What the edit uses**. An **Extra pictures folder** appears as step 2.
- Name each extra picture after the photo it belongs to: `beach_1.jpg`, `beach_2.jpg` … for `beach.jpg`. The number sets the order the model sees them in, so `beach_1` is the second picture. They can be any shape. A photo can have none, or up to 9 on Flux 3 and Qwen, 3 on Klein.
- A line under the folder says how many photos it found extra pictures for. Start checks the names and counts.
- Previews use the test photo's own extra pictures (`test_1.jpg` … for `test.jpg`), or the first original's when no test photo is set. The Samples tab now says that an edit run's previews come from the edit instruction and test photo, not its prompts box.

**If you used 8.1.1's naming:** extra pictures no longer go in the Originals folder. Move them to their own folder and pick **The photo + extra pictures**.

Suggested by @V1sionVerse, with a +1 from @Galvo87 (#187).
