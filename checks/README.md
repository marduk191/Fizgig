# Checks

Quick, self-contained checks that run on every pull request (`.github/workflows/checks.yml`) and anyone can run at home. They catch shared code breaking for another model: a family description that no longer validates, sampler maths that drifted, a key mapping that lost a layer.

## Run them

```bash
pip install -r checks/requirements.txt     # pytest, plus CPU torch if you don't have torch already
python -m pytest checks -q
```

## Adding a check

A check belongs here when it is:

- **CPU only.** No GPU, no CUDA.
- **Self-contained.** No model downloads and no model files. Build tiny tensors or toy models in the check itself.
- **Free of personal paths and data.** Nothing from your own folders, datasets or prefs.
- **Quick.** The whole folder should finish in a few minutes on a laptop.
- **Deterministic.** Seed anything random. A check that sometimes fails blocks everyone's pull requests.

Name the file `test_<what>.py` and keep it next to the others. If your feature needs a GPU or real model files to prove, keep that test on your side and describe what you measured in your pull request instead.
