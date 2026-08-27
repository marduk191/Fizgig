# Fizgig v4.5.0 — compiled INT8 at full resolution

## Krea 2: compiled INT8 training now fits at 1024px

Prompted by **@AZanarddy**'s reports: Krea 2's INT8 base is a true W8A8 path (int8
activations × int8 weights on the int8 tensor cores, with exact bf16 gradients), and as
of this release it compiles at full resolution. The blocker was where the gradient
checkpoint sat — compiled inside the graph, its activation stashes blew past 32 GB at
1024px. The checkpoint now moves **outside** the compiled region automatically when
resolution demands it: measured **~18 GB peak and ~27% faster than uncompiled** at
1024px, batch 1 — comfortably inside a 24 GB card. Nothing to configure: Compile Blocks
Auto (or On) places the boundary itself.

Two INT8 fixes from the same thread: an explicit INT8 pick is now honoured whenever
Blocks Swap is 0 (it used to silently fall back to fp8 when swap wasn't on Auto), and
compile can no longer attempt fp8 kernels on RTX 30-series cards — the combination that
crashed before the first step.

## Each run now trains from a frozen copy of its dataset config

Reported by **@jshhmphrs** — and the fix is their own suggestion. Changing the dataset
folder while a run was initialising (the natural way to queue a second run) could
retarget the *running* pipeline: dataset 2 trained under run 1's name. Every launched run
now freezes its dataset config into an immutable per-run snapshot; the live config
belongs to the editor alone. Paused runs resume on the dataset they started with — even
across a restart — and a launch now refuses, with the reason, if the config on disk
doesn't match the Start tab.

## Video-only datasets train again (MiniMax H3)

From an email report: when clips had to be cached slightly smaller than their bucket to
fit the encoder in free VRAM, the stale-resolution guard then rejected every one at
training time — a clips-only dataset crashed with "No training items" right after a
clean cache run, and mixed datasets silently dropped their clips. Capped clips now train
at their cached size, with a console line saying so and how to get full size back.
Stills keep the strict check.

## The VRAM planner stops crying wolf

A plan's "shortfall" used to include its own safety margins, so runs that fit fine (a
field 12 GB int8 run, and the same config under a hard simulated cap) were warned
"does NOT fit". The warning now fires only when the deficit exceeds every margin — the
genuinely-doomed heavy-clip cases still trip it loudly — and near-the-line plans get an
honest "tight fit" note. The 12 GB floor's message also now tells big-RAM machines about
the field-validated explicit-int8 option.

## ROCm: pinned bitsandbytes wheel bumped

**@0xDELUXA**'s Windows ROCm wheel moves to their HIP 7.16 build — a strict superset, so
the default pinned stack is unaffected, and installs on a HIP 7.16 runtime get an exact
DLL match instead of a fallback warning on every launch.
