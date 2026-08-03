# Contributors

Fizgig is written by [Peter Neill (shootthesound)](https://github.com/shootthesound).

## FNGarvin

[FNGarvin](https://github.com/FNGarvin) has contributed a string of high-quality features and
fixes, including:

- **The Metadata tab and rich safetensors metadata** ([#34](https://github.com/shootthesound/Fizgig/pull/34)) —
  trigger phrase, auto-embedded thumbnails, and full ModelSpec blocks on every checkpoint;
  the headline feature of v3.0.5.
- **Fully-offline captioning via a local tokenizer folder** ([#38](https://github.com/shootthesound/Fizgig/pull/38)),
  which became the path to zero-internet captioning, plus the Captions-tab path hints.
- **Florence-2 supply-chain hardening** ([#32](https://github.com/shootthesound/Fizgig/pull/32)) —
  pinning `trust_remote_code` revisions, including the subtle cross-repo `auto_map` case.
- **Pod entrypoint robustness** ([#33](https://github.com/shootthesound/Fizgig/pull/33)) — loud
  failures instead of silently running stale code on a bad `FIZGIG_REF`.
- **Optional SSH on pods** ([#30](https://github.com/shootthesound/Fizgig/pull/30)) — off by
  default, wired to RunPod's own `PUBLIC_KEY` convention.
- **`HF_TOKEN` handling in the model downloader** ([#29](https://github.com/shootthesound/Fizgig/pull/29)),
  and the UI truthfulness fix in [#6](https://github.com/shootthesound/Fizgig/pull/6).

**A note on the record:** the PRs above (except #6) show as *closed* rather than *merged* — a
back-end merge script applied them to master under the maintainer's authorship and closed them.
That was wrong, and it erased FNGarvin's contribution record. Retroactive `Co-authored-by`
attribution commits were added for each one (512e6cc…2e9ba2d), and contributor PRs are merged
normally — authorship intact — from here on.
