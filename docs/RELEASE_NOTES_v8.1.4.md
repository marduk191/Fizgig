# Fizgig v8.1.4: maintenance

- **Studio has Width and Height** in place of the square Res box, from 256 to 1536, so previews can match your training shape. They're remembered and saved with a setup. ACE-Step keeps its Length box.
- **A donor LoRA loaded after the model now shows its sliders and buttons straight away.** Before, they only appeared if the donor was there when the model loaded.
- **AMD: PCs with built-in graphics as well as a graphics card now train on the card.** The launcher picks it by itself and shows its choice in the console; a `HIP_VISIBLE_DEVICES` you've set is left alone. Thanks to @johnstephendonahue-ux for tracking this down (#186).
