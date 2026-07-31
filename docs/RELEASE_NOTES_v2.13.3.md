# Fizgig v2.13.3 — RunPod maintenance

## Deploy on RunPod, from the Start tab

The Start tab now has a **⚡ Deploy on RunPod** button next to the tip jar — renting a bigger card is
one click from inside the app.

**Get help on YouTube** is now just **Tutorial**.

## Running on RunPod? Set your template's image tag to `2.13.3`

A pod startup fix ships in the image, so update the tag and redeploy to pick it up.

## Clearer pod instructions

- **Set your own password when you deploy** — add `VNC_PASSWORD` on the deploy screen.
- **First boot takes a few minutes** while the image downloads, with the log line to watch for.
- **Storage** — the default Volume Disk is fine. Stopping a pod keeps everything; terminating it
  doesn't. A Network Volume is an optional upgrade.
- **`FETCH_MODELS`** lists its values: `krea2`, `klein`, `tools`.

## Fixes

- The version line on the pod card is now a readable size and contrast.

---

**Upgrading:** run `update_fizgig.bat`, or `git pull`. On RunPod, set the template's image tag to
`2.13.3`.
