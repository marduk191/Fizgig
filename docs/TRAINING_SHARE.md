# Training share (experimental)

[← Back to Helpers](HELPERS.md)

A training share splits each epoch's images between your GPU and a helper's. Each round, both start from the same LoRA, each trains its own images, and the two changes are added together. Whether it pays depends on the link between the two computers and the size of your epochs. The helper's tile measures it and tells you.

## How do I…?

### …train a run on two GPUs?

1. Add the other computer on the **Helpers** tab ([how](HELPERS.md#set-one-up)).
2. Switch **Training share** on, on its tile.
3. Start your run. The helper gets ready (models, then the training folder, then its cache) while your GPU trains, and joins from the next round once it's ready.

### …know whether it's worth it?

Read the helper's tile. From the run's 3rd epoch it gives an estimate, and after each round it says in green "Worth it: about N s saved an epoch", or in amber "Not worth it on this link". If it's not worth it, switch the share off. See [Is it worth it?](#is-it-worth-it).

### …make it cost less time?

Keep **Combine every** on the Training share tile at 3 epochs (the default) or set 5, and save every 3 or 5 epochs rather than every epoch. See [Combine every](#combine-every).

### …keep my pictures on my own computer?

Tick **Send this computer's cache instead** on the Training share tile. Your computer sends its cache rather than the training folder.

### …have my home GPU train a share of a pod's run?

On the home computer, open **Helpers** > **Help another Fizgig with this GPU**, then press **Help it train** beside the pod.

### …try again after it gave up?

Switch **Training share** off and on again on the helper's tile.

---

## How it works

**Start together** (ticked unless you untick it, on the Training share tile): the run's first epoch waits until the helper is ready (the training folder or cache sent, the cache made, the model loaded), so both GPUs train from the first epoch. The Training tab's log says what it's waiting for. Untick it and your GPU starts at once, with the helper joining at the first round once it's ready.

Each GPU trains its own copy of the LoRA on its part of the images, both starting from the same point. At every combine, the two copies' changes are added into one LoRA, and both sides carry on from that. So the run still makes one LoRA that has learned from every image: on a Krea 2 test, combining every 3 epochs, its likeness matched a one-GPU run of the same settings from the 9th epoch on.

The two sides stay in step: every image trains every epoch, on one GPU or the other. The images are split afresh each epoch. Face crops are split on their own: any picture with "face" in its file name (Image Prep's face crops are `FaceCrop_001.png` and so on, but your own names work too) is shared between the two GPUs in the same proportion as the rest, so neither trains all of them. The helper's share follows both GPUs' speeds and the transfer time, so both finish a round together.

The helper uses the run's learning rate (Adaptive LR's, or the schedule's place), and with EMA on, the saved average takes in both halves. A Context LoRA goes with the run. The LoRA still comes out a little different from a one-GPU run with the same settings, because each half trains from the round's start without seeing the other's steps. Compare its samples before relying on it.

## What goes over

- **The model:** the helper trains on the same model file as yours, checked by hash: its own copy, or a download when that's the same file. If it has a different one, the tile says so and nothing is shared. It uses the precision your computer picked (INT8, NF4 or bf16), with block swap planned for its own card.
- **The training folder** (pictures and captions) goes once, and the helper makes the run's cache with the run's own VAE and text encoder (the same files by hash). It can download any build of the text encoder in the model's list, so it uses yours even when that isn't the one Fizgig downloads by default. Both halves then train on exactly the same numbers.
- **Your computer's cache** goes instead when the helper can't get the same text encoder, or when **Send this computer's cache instead** is ticked. For Krea 2 it's trimmed to each caption's real tokens, about an eighth of the size, with the same result. Only the items still in your training folder go.
- **Each round**, the LoRA goes there and only the helper's change comes back, both in bf16.

The tile says which of these it's doing.

## Is it worth it?

Each helper's tile shows its **link**: your upload and download to it in Mb/s. It's measured when you add the helper and again before a share starts (16 MB each way, sent the way the share sends). If the measurement fails or looks wrong, press **Type speeds** and enter your broadband's upload and download as your provider sells them. Typed speeds are kept until you press **Measure**.

From the run's 3rd epoch (the first two run slow while the GPU warms up), every ready helper's tile gives an **estimate** before you switch its share on, from the link, the LoRA's size, both GPUs and the run's speed so far. It counts a fixed cost for each round (handing over and loading, about 20 s) until it has measured that helper's own, so the first estimate errs on the hopeful side. It also gives a rough window for getting the helper ready before the first round. That window is a guess, not a measurement.

After each round, the tile gives the measured verdict: the time saved an epoch, or the transfer time, the epoch size from which it would pay, and whether combining every 5 epochs would.

Measured examples:

- **On your own network**, it pays with two GPUs of similar speed, or with long epochs. A 5090 with a 4090 laptop (about a third of its speed) on a short MOSS-Image run took 29 s an epoch combining every 3, against about 27 s on the 5090 alone, so it didn't pay.
- **A pod with your home GPU helping** pays on big datasets. Each round's transfers take about 40 s for MOSS-Image, set by your home upload speed rather than RunPod's proxy. On a pod 4090 with a home 5090 helping, combining every 3: 144 images an epoch went from 59 s alone to about 39 s, and 36 images an epoch came out slightly slower than alone.
- **Two GPUs in one computer** (a pod with 2× RTX 4090, the second card added with **Use it as a helper**, see [This computer's other GPU](HELPERS.md#this-computers-other-gpu)): MOSS-Image, 144 images, combining every 3. Epochs went from about 15.5 s alone to about 9 s once the second card joined. The transfers took 1 s a round.
- **Your computer leading, a pod helping** (Krea 2 Ultra Fast, 170 pictures, a 5090 at home and a pod 5090, 9 MB/s home upload): the pod helped from the 2nd epoch, and shared epochs took about 50 s against 75–80 s alone, about a third off. Getting the pod ready took under two minutes whichever way the cache went.

## Combine every

On the Training share tile: how many epochs each side trains before the two are added together (every 3 epochs unless you change it; every epoch, or every 2, 3 or 5). Each combine sends the LoRA there and back, so fewer combines cost less time, and the two sides drift a little further apart in between. Combining more often keeps the result closer to a one-GPU run.

**On a fast network, combine every epoch.** Two GPUs in one computer, or a helper on your home network, send the LoRA in a second or two, so every epoch costs very little time and gives the closest result to training on one GPU. Every 3 epochs (the default) suits slower links, such as a pod with your home connection.

A combine always happens before an epoch that previews or saves (so the pictures and files hold both halves), at a pause, and at the run's end. Saving every epoch therefore means combining every epoch. When home helps a pod train, home's own Combine every applies (on its **Help another Fizgig with this GPU** row).

## When it stays off

The share stays off, and the run trains on your GPU alone, for sliders, fine-tunes, **Your subject and other people** (either option), and **Track how each image trains** with its per-image options. The tile and the Training tab say so, and nothing is sent.

## When something goes wrong

The run never stops for the helper, and no image misses an epoch:

- **A slow round:** if the helper's half doesn't come back in time, your GPU trains the helper's images of that round itself, then carries on. The helper drops the round and takes the next one.
- **The helper fails or goes away mid-run** (a crash, a restart, a pod that died): your GPU trains the images of the round it had, the run carries on alone, and Fizgig sets the share up again when the helper is back.
- **It never gets ready** for the first epoch (it fails to start three times, leaves, or you untick Start together): the run starts on your GPU alone.
- **It can't start** (a file the helper can't download, or its worker stops): the tile says why. After three tries it stops trying. Switch Training share off and on to try again.
- **A result from an earlier launch** (a re-run or resume in the same folder) is never added to the new run.

A helper trains a share for one computer at a time. A second computer asking while it's busy, or while it's still making the first one's cache, is told it's taken.
