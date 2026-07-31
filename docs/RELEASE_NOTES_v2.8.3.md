# Fizgig v2.8.3 — CPU back to idle, smarter presets

## 🔥 → 😴 100% CPU on every core while training (#18)

Reported on a 4080 Super: start a Krea 2 run and every CPU core pegs at 100%, even though the
actual training is on the GPU (CUDA ~85%) and step times are normal. The cycles weren't doing
anything.

The cause is a default in the OpenMP runtime PyTorch ships with: after every small CPU
operation, its thread pool — sized to every core on the box — keeps **actively spinning for
200 ms** in case more work arrives. Each training step touches a few small CPU tensors
(batch assembly, noise, timestep bookkeeping), so the pool re-arms that spin constantly and
burns the whole CPU doing literally nothing. Measured while reproducing this: **14.8 cores
busy in what should be idle time — dropping to 0.0** with the fix, at identical speed.

How bad it looks depends on your core count, which is why not everyone saw it: on a 32-core
box it reads as ~12% background CPU, on an 8-core box it's everything you have. It also
explains most of the extra CPU load block-swapping showed on smaller cards.

Fizgig now tells the pool to sleep when there's nothing to do. Applied in the GUI (inherited
by training runs it launches) and in both headless training scripts. Nothing to configure; an
explicitly set `KMP_BLOCKTIME` / `OMP_WAIT_POLICY` in your environment still wins.

## 🎛️ Per-family presets on the Training tab

Switching the model family (Klein ↔ Krea 2) now loads that family's default preset into the
fields, and the tab remembers each family separately for the rest of the session — flip over
to check something and come back, and your tuning is still there on both sides.

The Krea 2 defaults also got a tune-up: 30 epochs, rank 32:32, 0.25 MP, problem-image
detection and per-image adaptive LR on by default, and every memory setting on Auto so the
run shapes itself to your card.

## 🧹 Preset loading polish

Preset values now apply more reliably across every field, including a case where the
Learning Rate box could keep its previous value after loading a preset. Worth a quick glance
at your settings after loading one if you've customised heavily.
