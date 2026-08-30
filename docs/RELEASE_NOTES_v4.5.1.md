# Fizgig v4.5.1

A maintenance release, and every fix in it was found, diagnosed and built by the
community.

## Multi-GPU: the card you pick is the card that trains

**Only affects machines with more than one GPU — if you have a single card, nothing here
changes for you.**

Reported and fixed by **[@rocketsvm](https://github.com/rocketsvm)** (#104). On a
multi-GPU Windows machine where the display GPU isn't the fastest one, Windows and CUDA
can list your cards in **different orders** — so choosing "GPU 0" in Preferences could
hand training to the other card. The status bar then read VRAM from the wrong card too,
which made it look like nothing was happening.

Fizgig now identifies your chosen card by its hardware UUID, which is the same in every
tool regardless of listing order, so the pick sticks — for training, for the workbench
tools, and for the VRAM gauge.

**Multi-GPU users, one thing to do after updating:** open Preferences and pick your GPU
once more. Saved settings still hold the old-style position number, and re-picking
replaces it with the UUID.

## AMD: an updater that can't wipe your ROCm install

Built by **[@scryptio](https://github.com/scryptio)** (#107). `update_fizgig.bat`
installs NVIDIA packages, so running it on an AMD machine replaced the ROCm PyTorch and
bitsandbytes stack and left training broken until a full reinstall.

AMD users now have **`update_fizgig_rocm.bat`**, which updates the same way the ROCm
installer installs: shared dependencies with the NVIDIA-only lines stripped, the
bitsandbytes wheel kept in step with the installer's pin, and the launcher environment
refreshed afterwards. Each updater now checks what it's pointed at and stops with the
name of the right script rather than doing damage, and the in-app About panel tells ROCm
users which one to run.

Also from the same work: on a fresh ROCm update, bitsandbytes no longer reports "could
not detect ROCm GPU architecture" because the ROCm tools weren't yet on PATH.

## System RAM is now in the requirements

Fizgig's requirements listed the GPU, driver, OS, Python and disk — but never system RAM,
which is the quiet limit on MiniMax H3. Its text encoder is a 15.7 GB file that has to sit
in system RAM while captions are cached, and INT8 block streaming stages a similar amount
again during training. **32 GB is comfortable for H3; 24 GB works only with other
applications closed.** Klein 9B and Krea 2 are fine on 16 GB.

Worth knowing if you have hit this: when system RAM runs short, the failure arrives
dressed as **"CUDA error: out of memory"** even though the GPU is nearly empty. Close what
else is running and re-run the caching step before suspecting your graphics card.

## MiniMax H3: a long-sequence limit removed from the INT8 kernel

Contributed by **[@rintic-13](https://github.com/rintic-13)** (#89). The fused INT8
kernel tracked positions in a 32-bit counter, which would have wrapped past roughly
150,000 tokens in a single sequence — long beyond anything today's clips or captions
reach, and it would have corrupted quietly rather than raising an error.

Below that point the new kernel is **bit-for-bit identical** to the old one, verified
across 30 shape and bias combinations, so nothing about current training changes. The
ceiling is simply gone before anything could arrive at it.
