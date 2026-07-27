# Fizgig v2.8.1 — Hotfix

One fix, worth shipping immediately: a v2.8.0 regression that crashed **4-bit Base**
Krea 2 training runs mid-run.

## 🐛 4-bit Base: crash when the model parked on CPU (#17)

On a 4-bit (NF4) run, training died the first time the trainer parked the model on CPU to
make room for something else — in practice at an **auto-recaption epoch boundary**, or at
**step 1 with Sample at Start** enabled — with:

```
RuntimeError: Expected all tensors to be on the same device, but got mat2 is on
cuda:0, different from other tensors on cpu
```

A 4-bit model's weights live in two places: ordinary layers that PyTorch's `.to()` moves,
and 4-bit packed data it can't see. Parking the model moved both; the restore accidentally
treated the two moves as either/or, so on a 4-bit run only the packed data returned to the
GPU and the ordinary layers stayed stranded on CPU. The next training step then computed on
the wrong device, and the crash surfaced at the first GPU-resident tensor it met — the LoRA
layer, which was innocent. (The error's mention of `cuda:0` misdirected toward multi-GPU
setups; a second card plays no part.)

Both broken restore paths now match the one that was always correct, the trainer no longer
infers its compute device from whichever parameter it happens to find first, and a
regression test — verified to fail against the pre-fix code — guards all three park/restore
paths so this shape of bug can't quietly return.

**Not affected:** fp8 and INT8 runs, Klein, and 4-bit runs with auto-recaption and Sample at
Start both off. LoRAs already saved by a crashed run are intact and usable.

Thanks to **@LM-VC-Agent** for the report with the full log — it had everything needed.
