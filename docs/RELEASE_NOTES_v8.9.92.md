# Fizgig v8.9.92: train on sub-folders

**A big dataset can stay organised in folders.** Tick **Also train on its sub-folders** on the Training tab (off by default) and the pictures in the training folder's sub-folders train too, at any depth: outfits, forms, sources, or the thousands of pictures of a fine-tune (thanks @TensaiKuro, #230, and @ribawaja, #149). [Sub-folders](https://github.com/shootthesound/Fizgig/blob/master/docs/TRAINING.md#sub-folders)

- **Each sub-folder trains as its own set**, with its captions next to its pictures and its own cache, so two folders can hold pictures with the same name. Every picture trains once an epoch, wherever it is.
- **The line under the tick says what it will train** before you press Start: how many folders and pictures, the folders not captioned yet first, and **Show all folders** for the full list. Start refuses a sub-folder that isn't captioned and names it.
- **Left out:** `excluded_by_look`, `originals`, hidden folders, and the output and cache folders when they sit inside.
- **Fine-tunes** train every folder in the tree. If the folder you picked for **Regularisation images** sits inside your training folder, its photos are not trained as your own pictures: they are only used as regularisation images, as they would be from any other folder.
- **Per-image loss, auto-recaption and exclusions keep each picture with its folder.** A caption edited in Per-image loss or rewritten by auto-recaption goes into that picture's own folder, with that folder's trigger word.
- **On a pod:** Upload a folder, Send to this pod and Move a run carry the sub-folders when the tick is on.
- Captions and Image Prep work on one folder at a time: to caption or mask a sub-folder, choose it as the training folder on the Start tab.
- Not with Subjects 2-4, DOP, Fizgig LoRA Isolation, Edit or slider training; the tick is hidden there.

**Continue a run from Runs & outputs.** Each run has a **Continue** button while its newest save is still on disk: it trains on from there with the run's own settings, into the same entry. Pause one project, train another, and come back to the first whenever you like. A run that trained all its epochs asks how many to train to first (a fine-tune: how many rotations), and Continue pressed while another run trains puts it on the queue, where **Edit** changes its settings (more epochs, say) before it starts. [Runs & outputs](https://github.com/shootthesound/Fizgig/blob/master/docs/WEB_UI.md#runs--outputs)

**The queue is always in the sidebar.** The **Queue** box sits under the run on every tab, empty or not, and opens the queue, which now lives at the top of **Runs & outputs**. **Add to queue** stays beside Start on the Training tab. While you edit a queued run, a bar along the bottom of the window, on every tab, says which run it is, with **Save to queue** and **Cancel edit**.

**Per-image loss is easier to find.** Its lilac button is always on the run card and on each run in Runs & outputs, greyed when the run doesn't track it; hover it to see which ticks turn it on. It now also shows when per-image LR, the look warm-up or auto-recaption is ticked, all of which track every image.

**Keep a pod on one version.** A pod runs the newest Fizgig each time it starts. To keep one on a version in the middle of a job, set `FIZGIG_REF` to a release's tag, for example `v8.9.92`. [How](https://github.com/shootthesound/Fizgig/blob/master/docker/README.md#keep-a-pod-on-one-version)

**Auto-recaption: pick what each try uses.** The Training tab's **First try uses** and **Second try uses** pick any Captions tab task each, with your edited instruction (Training caption, then Exhaustive detail, by default). Both defaults are written for a person, so for a style or a concept set both to suit your LoRA. The log names each try's task and shows each new caption in full (thanks @bkrew, #244).

**Fixes**

- **Resume with another Optimizer Type than the run trained with** no longer stops with "loaded state dict has a different number of parameter groups": the LoRA's weights and epoch carry on, with a fresh optimizer, and the log says so. Resume Training's hint now points to Continue, which keeps the run's own settings (thanks @jake661, #246).
- **Pressing Start more than once started the run more than once**, every copy training at the same time into the same files. One press is one run now, and the button reads Starting… until it's up.
- **Starting a run ends editing a queued run**, so Save to queue can't put the new run's settings in the queued run's place.
- **With Subjects 2-4, auto-recaption, exclusions and caption edits in Per-image loss now go to that subject's own folder**, with its own trigger word. Before, they looked in Subject 1's folder.
- **A run moved to the same pod twice:** its per-image view and caption edits looked in the first move's folder, which was gone.
- **Make masks** says to tick **Replace masks already there** when it keeps the masks a folder already has (thanks @crazyscale-dotcom, #245).

Edit masks came in 8.9.91: [Fizgig v8.9.91 release notes](https://github.com/shootthesound/Fizgig/releases/tag/v8.9.91).

**Updating:** update Fizgig on your computers, and restart your pods.
