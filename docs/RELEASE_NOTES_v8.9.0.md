# Fizgig v8.9.0: your other GPUs, at home or in the cloud, speeding up your run

**▶ [Watch the 2-minute video](https://www.youtube.com/watch?v=7Ao0gE2b2qw)**

**Train on two GPUs, and stop pausing for previews.** A second GPU, in this computer or another, can now train alongside yours: on a pod with two RTX 4090s, each epoch took about 40% less time. Another computer can render your previews while your GPU keeps training, or take your run over while you need your PC.

A helper is any other computer running Fizgig: a second PC or laptop, a second GPU in this one, or a RunPod pod. Add one on the new **Helpers** tab, where your network's computers and your running pods are listed for you. **Both computers need this version:** update Fizgig on your computers, and restart your pods.

- **Train on two GPUs** (experimental). Each GPU trains its own copy of the LoRA on its part of the images, and the GPUs merge their work every 3 epochs, so you get one LoRA that has learned from every image. Both start together (the run waits for the helper before its first epoch), and a round the helper doesn't bring back is trained on your GPU, so no image misses an epoch. In testing, its likeness matched a one-GPU run. Two equal GPUs come close to halving each epoch, a helper over the internet saves less, and the share pays most on long runs. On a fast network, such as two GPUs in one computer or your home network, merge every epoch for the closest result to one GPU, at very little time cost. [Training share](https://github.com/shootthesound/Fizgig/blob/master/docs/TRAINING_SHARE.md)
- **Use the second GPU in this computer.** One click (**Use it as a helper**) starts a second Fizgig on that card, ready to train a share, render previews or take a run. It shares your model files, and starts and closes with Fizgig.
- **Previews that never pause your run.** A helper renders them while your GPU keeps training: on a run with previews every 5 epochs, that can be a third of the run's time back. Each enlarged preview says which computer rendered it. [Previews on a helper](https://github.com/shootthesound/Fizgig/blob/master/docs/HELPER_PREVIEWS.md)
- **Need your PC back? Move the run.** **Move to…** on the run card pauses the run here and carries it on on a laptop, another PC or a pod, on the same model files. Bring it back when you want it, captions edited over there included. If the other computer goes away, **Carry on here** takes the run back from your last save. [Move a run](https://github.com/shootthesound/Fizgig/blob/master/docs/MOVE_RUN.md)
- **Your home GPU helping a pod.** Training on a pod? Your GPU at home can render its previews or train a share (**Helpers** > **Help another Fizgig with this GPU**).
- **A pod's checkpoints come home as they're saved.** Or fetch just the epochs you pick and try your own prompts on them in LoRA Royale while the pod keeps training.
- **Never pay for a pod that's finished.** If you tick it, Fizgig deletes the pod once its run has finished and every LoRA on it is home and checked against the pod's copy.
- **A helper that fails or goes away never costs you the run.** The run carries on on your GPU, a helper downloads any model it needs itself, and one that comes back at a new address is followed. Guide: [Helpers](https://github.com/shootthesound/Fizgig/blob/master/docs/HELPERS.md)

**Renting a GPU:**
- **The Pods tab lists only RunPod cards you can rent right now**, cheapest per run first, with one card or two: stock and prices come from hosts whose driver runs Fizgig's image, so a card shown free deploys. On RunPod's page, add a 300 GB **Volume disk** before you deploy ([how](https://github.com/shootthesound/Fizgig/blob/master/docker/README.md#give-it-a-disk-on-runpods-deploy-page)). Deploying through Fizgig's links supports Fizgig at no cost to you.
- **A pod's run keeps going with the window closed**, and its run card says so. Stopping a pod yourself mid-run loses what it trained since its last save, so press **Pause** first.
- **A pod you deleted on RunPod drops off the Pods tab** by itself.
- **Vast.ai is supported too**, with its own [Fizgig template](https://cloud.vast.ai/?ref_id=757773&creator_id=757773&name=Fizgig%20LoRa%20Training%20and%20Fine-tuning): https addresses, and a stop by itself when a run finishes (#131, thanks @MaxNazarov for asking). [Fizgig on Vast.ai](https://github.com/shootthesound/Fizgig/blob/master/docs/VAST.md)

**Also in this release:**
- **Krea 2 trains with little system RAM.** When the model won't fit in free RAM, it's built and quantised one layer at a time: peak RAM about 2 GB instead of 25 GB, with exactly the same weights. A 4090 pod with 31 GB of RAM now trains Krea 2, and a 32 GB PC loads it faster (74 s down to 47 s in testing).
- **Faces for furry, anime and drawn characters** (thanks @Murf-Code, #190): the Look Filter and likeness scoring can find faces with an anime/furry head model and match looks with DINOv2, picked per dataset or automatically. People keep InsightFace and ArcFace by default, and the extra models download only when you choose them.
- **Z-Image Turbo's text encoder fits smaller cards.** With less than 11 GB free it loads in 8 bits: about 4.5 GB instead of 7.7, with the same captions to within rounding.
- **On a pod, memory checks read the pod's own RAM limit**, not the host machine's.
- **Slider runs preview when you asked them to**, following the Samples tab instead of the save interval, and every model's **Slider preset** previews every 5 epochs.
- **Captions in any encoding.** A caption saved as Windows-1252 or UTF-16 by an older tool no longer empties its folder on the Captions tab (#225, thanks @mart-hill).
- **The Start tab says how its two folder cards differ** (#224).
- **On Linux, the app window closes without hanging, and Ctrl+= / Ctrl+- / Ctrl+0 zoom it** (#197, #198, thanks @DmitryAkDev).
- **Prompt boxes no longer spellcheck**, so trigger words aren't underlined red (#222, thanks @CognitiveDiffusion).

**Updating:** update Fizgig on your computers (helpers need the same version), and restart your pods. On Vast.ai, restart the instance.
