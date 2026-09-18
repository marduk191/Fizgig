# Fizgig v5.8.2

Maintenance: Krea 2 presets tuned, the LoRA output folder goes per family, and the load-strength box comes to Krea 2 in Repair Studio and to the Explorer.

## Krea 2 presets

- **Ultra Fast (rank 8, adaptive LR) is now the preset you land on** when you first pick Krea 2. Rank 8 is more than enough for a character on a model this size and lands the right result more reliably than a bigger rank.
- **"Krea 2 Defaults" is renamed "Krea 2 Standard"** (rank 32, the model authors' recommended figure). Same recipe, clearer name.
- **Epochs**: Ultra Fast now defaults to 30, Standard and Style to 64.

## LoRA output folder, per model family

The Output Directory on the Training tab is now remembered separately for Klein, Krea 2 and MiniMax H3. Switch family and the field shows that family's folder; come back and yours returns, across restarts. A family you've never set a folder for keeps whatever the field holds, and a fresh install still defaults to `output_loras` inside Fizgig. The "LoRA output" row has gone from Preferences, since one shared entry no longer meant anything; the Extract tab and the Resume picker follow the Training tab's folder.

## Load strength on Krea 2 and in the Explorer

- **Repair Studio on Krea 2** gets the "at strength" boxes beside the primary and donor pickers, as on MiniMax H3. The strength is a preview-time load multiplier: every block slider stays relative to it, the baseline pane renders at it and says so, and Save Repaired LoRA never applies it, so the saved file keeps its original scale and looks like the preview when used at that strength. Every module of a full Krea 2 LoRA takes the strength, including the projections outside the block sliders.
- **LoRA the Explorer** on Krea 2 and MiniMax H3: the Strength box now works the same way, a load strength rather than a value written into every slider. Edit it with a LoRA loaded and the baseline re-renders. The Explorer remembers it across restarts, its state readout names it, and the save dialog reminds you to use the file at that strength. Klein keeps its previous behaviour.
- **Handoffs keep the family.** "Explore this in LoRA the Explorer" from a MiniMax H3 session now lands on the H3 Explorer, not Klein's, and "Refine in Repair Studio" from an H3 Explorer lands on H3 Repair. The load strength travels with the LoRA in both directions.

## Fixed

- **LoRA the Explorer showed the Klein-only Distilled/Base radio** when the app opened with MiniMax H3 remembered as the Explorer's family.
