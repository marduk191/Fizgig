# Fizgig v5.7.1

Four fixes, no new settings.

## Fixed

- **Krea 2 with a Context LoRA crashed at the first training step** when block compile was on, with a long compiler traceback ending in `GuardOnDataDependentSymNode`. Any Krea 2 run that used a Context LoRA since 5.3.0 would have hit this. Fixed; the run starts as it should.
- **The sample gallery showed no previews after you changed the LoRA output folder** between runs in the same session. The gallery now follows whichever folder the current run is writing to, and its Download LoRA links point at the current run's checkpoints.
- **A caption file that was not UTF-8 aborted the whole caching run**, with an error that named no file. One curly apostrophe saved from Notepad was enough. Captions in Windows-1252 or UTF-16 now load, with a warning that names the file so you can re-save it. Thank you **[@marduk191](https://github.com/marduk191)**.
- **The "expandable_segments not supported on this platform" warning on Windows is gone.** Windows never supported that memory option, so it is no longer requested there; Linux is unchanged. Also from **[@marduk191](https://github.com/marduk191)**.
