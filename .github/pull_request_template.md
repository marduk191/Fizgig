**What this does**

<!-- One or two lines. -->

**Issue or discussion it came from**

<!-- A link, e.g. #123. New models start with a post in New models & fine-tuning, and new features with an issue (see CONTRIBUTING.md). Small fixes and docs can skip this. -->

**What you tested, and on what hardware**

<!-- GPU / VRAM, OS, real runs, peaks, before/after numbers. -->

**Checklist**

- [ ] What people get today stays the default; anything new is opt-in
- [ ] No second environment, worker process or change to pinned dependencies
- [ ] A model pull request adds only the model (shared training features go in their own pull request)
- [ ] Any checks added are in `checks/`: CPU only, self-contained, quick (GPU tests stay on your side; describe them above)
- [ ] `python -m pytest checks -q` passes
