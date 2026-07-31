# Fizgig v2.13.2 — Train on a GPU you don't own

## Fizgig runs on rented hardware

[**⚡ Deploy on RunPod →**](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep)

One click and you get the **whole app** in a browser tab — Training, Repair Studio, LoRA the
Explorer, LoRA Royale, Profiler, Extract and the sample gallery. Not a cut-down web version; the
actual application, running on a card far bigger than the one in your machine, while your own GPU
stays free.

- **Drag-and-drop file transfer** — datasets in, finished LoRAs out, no terminal
- **One-click model downloads** — Krea 2 needs no HuggingFace account
- **Persistent storage** — download the models once; every future session picks up where you left off
- **Stop when finished** — optionally shut the machine down after a run completes, so an overnight
  finish doesn't bill until morning
- **Closing the browser doesn't stop training** — it runs on the machine, not in your tab

A new **RunPod section in Preferences** is the control panel when you're on a rented machine, and
explains the option when you're not.

The deploy link is a referral one — it supports Fizgig's development at no extra cost to you.

**[Full guide →](https://github.com/shootthesound/Fizgig/blob/master/docker/README.md)**

## Readable text

Users reported the Preferences tab as grey-on-grey, and measuring it they were right: the
explanatory text scored **2.54:1** contrast, which fails accessibility guidelines even for large
text. It's now off-white at **8.64:1** and a point larger, across **every tab** — the same problem
was everywhere, Preferences was just the most text-heavy place to notice it.

The **scrollbar** was worse in the same way: the part you drag sat at 1.06:1 against its own track,
effectively invisible. It's now the same blue as the selected tab.

## Fixes

- **Training could fail to start, silently.** Fizgig launched training with a hardcoded path to its
  bundled virtual environment. If yours lives anywhere else — conda, a system install — the process
  never started and the run stopped dead after "starting cache preparation", with nothing in the
  console explaining why.
- **Download models for me** no longer hides the Krea 2 Turbo DiT behind a tickbox. One click now
  fetches everything, so the workbench tools work without a second trip.
- **Folders with `[square brackets]`** in the path no longer break caption discovery.
- **Low disk space** is checked before a run starts, so you find out before four hours in.

---

**Upgrading:** run `update_fizgig.bat`, or `git pull`.
