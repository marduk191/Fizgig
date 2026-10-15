# Fizgig v8.0.1: the new app

Fizgig is now a web app. It opens in its own window on the computer it's installed on like a desktop app, or in any browser: start a run at your desk and watch its previews from your phone, use your desktop's GPU from your laptop, or open a rented one on RunPod in the same window. Every tab is rebuilt for it, and there's plenty that the old app never had.

https://github.com/user-attachments/assets/2db82d0d-e7ed-4f60-a79c-d2ac1f4984ac

The 80-second tour, also [on YouTube](https://youtu.be/F8F-vuiZd88).

> **On RunPod and need the old pod?** The previous pod, with the classic app in a remote desktop, stays available as **[Fizgig Classic v7.x](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep)** for anyone in the middle of a big job. Pods you already have keep running as they are, and the new [Fizgig template](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=bdrdew7hxw&ref=vkb387ep) is the one to deploy from now on.

## Starting Fizgig

- **A welcome on the first start** shows where everything went, with a short tour: the steps are the same, now numbered down the left. Tick **Don't show this again** to keep it closed.
- **On your computer:** **Launch Fizgig.bat** (Windows) or `./launch_fizgig.sh` (Linux) opens Fizgig in its own window. **Launch Fizgig in browser.bat** opens it in your browser instead, **Launch Fizgig AMD.bat** is for AMD cards, and **Launch Gizmo.bat** opens Gizmo.
- **Your settings carry over.** Everything of yours now lives in one **user_data** folder inside the Fizgig folder: settings, presets, the cache and downloaded models. Fizgig moves your existing files there the first time it starts, so moving or backing up an install is one folder. It opens on the model you last trained, with its settings, and runs you had queued are in the new queue, waiting for **Start queue**. Your LoRAs are where they were, and each one Fizgig 7 trained is in **Runs & outputs** as a finished run.
- **The launchers have new names:** `run_fizgig.bat` is now **Launch Fizgig.bat**. If you start Fizgig from a shortcut, point it at the new file.
- **From another device:** set a password in **Preferences → Web UI** and press **Restart Fizgig now**. The card shows the address to open on your phone or another computer on your network, and on your phone the quickest way in is the Start tab's **Open on your phone** QR code: scan it and you're logged in.
- **On RunPod:** the pod is now the app itself. Open the pod's HTTP port 8081 and log in with its password. There's no remote desktop any more, so pods start faster, and the file manager is still on port 8080. Closing the browser tab doesn't stop training.

## Your other computers

- **Fizgig finds your other computers by itself.** Fizgig on your desktop shows up in the laptop's **Pods** tab with its name and GPU, marked "On your network", and **Open** uses it from the laptop as if it were a pod. While it trains, the row says what it's training and how many runs are queued after it.
- **Allow on my network** (Preferences → Web UI, on Windows) lets Fizgig through Windows Firewall on private networks with one administrator's OK. It also clears a block left by a Windows prompt that was cancelled, and the row says when Windows calls your network Public and how to make it Private.
- **Send a training folder across:** in the Fizgig window, the Start tab's **From this computer** card lists your own training folders on the other computer's page, with a sort order and Refresh. **Send** copies one over with its captions and makes it the training folder.
- How to set it up: [Your other computers](docs/WEB_UI.md#your-other-computers-use-ones-gpu-from-another).

## New in the app

- **RefMod Toolkit** (MiniMax H3): try RefMods and ReFLoRAs before ComfyUI, with and without them side by side on the same seed, sweeps of one setting at a time, curve presets, numbered references, the exact ComfyUI node values, Bake and Save as ReFLoRA.
- **Extract from a fine-tune**: the Extract tab turns a full fine-tune into ordinary LoRAs, one per rank you tick, in one pass. It replaces the separate Checkpoint to LoRA app.
- **Metadata**: read and edit any `.safetensors` file's metadata and thumbnail. Saving rewrites only the header, so even a large checkpoint saves in seconds.
- **Runs & outputs**: every run with its loss curve and its files, and **per-image loss charts** for the models with the loss watch. You can fix an image's caption from its chart while the run goes, and the trainer re-encodes it at the next epoch.
- **The queue, in sight:** an **Up next** panel under the run in the sidebar, on every tab, with the next runs in order. A held queue turns amber with the reason and a **Start queue** button. The Run card shows what's up next, and **Add to queue** says where the run landed and what it starts after.
- **Resume at a new size:** change Target Megapixels while a run is paused and **Resume** offers to carry on at the new size. The pictures are cached again first, keeping what still fits. Train most epochs small and the last few large, say.
- **Studio** (was Repair Studio): a No-LoRA pane, crop adjust for H3 frames, epoch and LoRA mixes, Model EQ, metrics, and setups you can save and load. For ACE-Step songs there's one full-width player: Baseline, Tweaked and No LoRA from the same moment, loop sections with handles, scrubbing, and block activity that glows as the song plays.
- **Captions**: each picture's size under it, sort by size to weed out small ones, and remove a picture into a `removed images` folder (it's never deleted).
- **Samples gallery**: likeness scoring against three photos of your subject, and the run visualiser.
- **The file picker** has **This computer**, for every drive (Windows) or mounted disk (Linux).
- **Gizmo** has the new app's look, an **Open Fizgig** button, and opens from the Image Prep tab.
- **Clips play muted** until you unmute them, and **Play together** pauses them together too.

## The classic app

The old desktop app is gone. Everything it did is in the new app, apart from **LoRA the Explorer**: Studio's **I'm feeling lucky** covers the random variations it was used for. **Diff to LoRA** is gone too; Extract's **From a fine-tune** does its job.

## RunPod

- **The new template runs the app directly:** no remote desktop, a smaller image and a faster start. If you deploy from the [Fizgig template](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=bdrdew7hxw&ref=vkb387ep) you'll get it automatically.
- **The Pods tab** opens your RunPod pods from Fizgig on your own computer, in the same window: paste a pod's ID, or add your RunPod API key and your Fizgig pods are listed with Start and Stop. **Switch to this computer** brings you back, and the window reopens on your pod next time.
- **Rent a pod:** with your API key, the Pods tab lists the cards RunPod has free right now, filtered by memory, cloud and price, cheapest per run first, with each one's price and roughly what a run costs on it. **Deploy** opens RunPod's page with that card and the template chosen; you press Deploy there.
- **It logs you in:** in the Fizgig window, a pod whose template sets `FIZGIG_PASSWORD` opens already logged in. The password is read from the pod's settings on RunPod each time and never saved. A password kept as a RunPod secret, or one the pod made up, you type as usual.
- **An older pod keeps working:** update Fizgig on your computer while a long run goes on a pod and the pod still opens, with a note to restart it when the run is done.
- **The file manager** on port 8080 has Fizgig's look and opens from the **File manager** button in the pod's sidebar, with links from the Start tab's upload card and Runs & outputs. It now takes the pod's password exactly as it is, at any length; only a very common one gets 0s added (`abc123` becomes `abc123000000`), and the button's tooltip says when.
- **Watch a pod from your phone:** the Start tab's **Open on your phone** QR code opens the pod on your phone, already logged in for 30 days, so a long run is in your pocket.
- **Keep me logged in:** a login lasts 30 days on that device and survives the pod restarting.
- **Stop this pod when a run finishes** is on the Preferences tab, with a two-minute countdown you can cancel from any page.
- **The password:** set `FIZGIG_PASSWORD` in the template to choose it; otherwise the pod makes one on its first start, prints it in its logs and keeps it. Templates that set `VNC_PASSWORD` keep working.
- **The Classic template stays for now:** the previous pod with the remote desktop on port 6080, for anyone in the middle of a big job. [Fizgig Classic v7.x](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep). Pods you already have keep running as Classic pods; both use the same `/workspace`, so a volume moves across with its models, datasets, LoRAs, settings and past runs.

## Thanks

Thanks to @vishnuteja7054 for pull request #182 that set me on the road to making this work.

## New in 8.0.1

- **Open on your phone:** a new card at the bottom of the Start tab. Press **Show QR code** and scan it with your phone's camera: it opens Fizgig already logged in, and the phone stays logged in for 30 days. On a pod it opens the pod's public address, so you can keep an eye on a long run from anywhere. The code shows for 30 seconds and each one works once. It needs a password set, and the card says how if there isn't one.
- **The tabs fold away on a phone:** a menu button at the top shows the tab you're on; tap it for the tabs, and picking one folds them again.
- **A tidier top bar on a phone:** the name, the run and the buttons fit on one line.
