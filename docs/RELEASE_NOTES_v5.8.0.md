# Fizgig v5.8.0

Fine-tuning on MiniMax H3 gets the training aids LoRA runs already had, and a preview fix.

## MiniMax H3 fine-tuning

- **The training adapter now rides on fine-tunes.** [@ostris](https://github.com/ostris)'s de-distillation adapter, on by default for every H3 LoRA run since 5.2, is on by default under Fine-tune too, with the same contract: active for every training step, off for previews, never written into the checkpoint. The file you get is a plain H3 fine-tune. Untick **Training adapter** to run without it.
- **The text token refiner stays frozen unless you ask.** Fine-tunes used to train it on every epoch. LoRA runs already leave it alone, because training it softened output and made previews judder between epochs without helping likeness. The **Train the text token refiner** tick now governs LoRA and fine-tune runs alike, off by default.
- **Ticking Fine-tune sets the learning rate to 3e-5**, the tested rate, instead of 1e-5. Start there, judge the results, and come down to 1e-5 if a run looks too eager. 1e-4 still destroys an H3 fine-tune.
- **EMA hides under Fine-tune.** It was already off in the trainer; the setting no longer pretends otherwise. The guide explains why it is a different job on a fine-tune.
- **The Samples tab explains previews under Fine-tune.** They render whenever a checkpoint saves, on the Training tab's save cadence, so keep sample generation on and set your prompt there as usual.
- The fine-tune guide (`docs/FINETUNE_HOWDOI.md`) has new entries on the adapter, the refiner and EMA.

## Fixed

- **Still-image previews on MiniMax no longer run out of memory at every epoch.** The base steps aside for the decode, as it already did for clips. Thank you [@Hell-Bent-Fox](https://github.com/Hell-Bent-Fox) (#109, #134).
