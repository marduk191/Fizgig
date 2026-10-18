**Fix: Fizgig LoRA Isolation and DOP runs that stopped at the first step with block swap.** On cards that train with Blocks Swap (most cards under 32 GB on Krea 2), a run with **Leave the rest of the model alone** or **Fizgig LoRA Isolation** on the Subjects card stopped on its first step with `Expected all tensors to be on the same device ... cuda:0 and cpu`. They now train normally. Runs with the Subjects card Off, and runs with no block swap, were not affected.

Thanks to u/Zaggeros on Reddit for the report and the log that pinned it down.
