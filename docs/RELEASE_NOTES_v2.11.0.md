# Fizgig v2.11.0 — Never lose a run

## Save state, on your terms

A **state** is your LoRA plus the optimizer — everything needed to pick a run back up exactly where
it left off, not just the weights. Until now Fizgig only wrote one when you pressed Pause, which
meant a crash, a power cut, or a run you set too short left you starting over.

Three new controls sit under Resume Training on the Training tab, for **both Klein and Krea 2**:

- **At each checkpoint** — writes a state every time a LoRA checkpoint is saved, following your
  Save Every N Epochs setting. Crash protection.
- **At end of training** — writes one when the run finishes, so a **completed LoRA can be trained
  further**: raise Max Train Epochs, resume the end state, and it carries straight on with the
  optimizer, learning rate and adaptive-LR history intact.
- **Keep last N** (default 2) — states are big (roughly 470 MB at rank 32), so older ones are
  cleaned up as new ones are written. Only state dirs for the LoRA you're training are touched, and
  the newest is always kept.

Both boxes are on by default. **Pause still saves state whether they're ticked or not** — that
hasn't changed.

## Resuming, explained

The Resume Training field had no explanation at all, which made the whole feature invisible unless
you already knew a state was a *folder* rather than a `.safetensors`. It now tells you what to
browse for, that the number in `myLora-000012-state` is the epoch it finished, and that continuing a
finished LoRA means raising Max Train Epochs first.

If you resume a state that's already at your epoch limit, Fizgig now says so before it starts
instead of appearing to train and doing nothing. (Resuming with nothing left to run is still
allowed — that's exactly how a run paused on its final epoch gets completed.)

## Also

- Adaptive-LR history now travels with **every** saved state, so a resumed run continues the
  watcher instead of restarting it cold.
- Krea 2 refuses to resume a state that doesn't match the network you're training, rather than
  silently training from scratch and overwriting your finished LoRA.

Thanks to **@jowala** for the report ([#19](https://github.com/shootthesound/Fizgig/issues/19)).

---

**Upgrading:** run `update_fizgig.bat`, or `git pull`.
