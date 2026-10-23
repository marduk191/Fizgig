# Previews on a helper

[← Back to Helpers](HELPERS.md)

Without a helper, training stops for every preview round. On a MOSS-Image run with previews every 5 epochs, that's about a third of the run. With a helper rendering them, your GPU never stops training.

## How do I…?

### …switch it on?

1. Add the other computer on the **Helpers** tab ([how](HELPERS.md#set-one-up)).
2. Switch **Previews** on, on its tile.
3. Start your run as usual. The Samples tab and the run card say a helper is rendering the previews.

### …check the helper's pictures match mine?

Press **Check picture** on the helper's tile. The next preview round renders on both computers and the two are compared. See [Check picture](#check-picture).

### …have my home GPU render a pod run's previews?

On the home computer, open **Helpers** > **Help another Fizgig with this GPU**, then press **Help it with previews** beside the pod.

### …tell where a preview was rendered?

Open it in the lightbox: each round is labelled with the computer that rendered it.

### …go back to rendering here?

Switch **Previews** off on the helper's tile. The next round renders on your own GPU.

---

## What goes over

For each preview round, the epoch's LoRA, the round's preview settings and the run's encoded prompts go to the helper. The pictures come back into the run's samples folder. No training pictures are sent, and the helper needs no text encoder.

- **On a fast link**, the LoRA goes as the exact weights that train (about 300 MB for MOSS-Image).
- **On a link under about 160 Mb/s** (a pod through RunPod's proxy, or a home upload), it goes at half the size (bf16). That renders the same picture to within rounding.
- The next round's LoRA goes up while the helper renders the current one.

A helper that lacks the model downloads it first (the SFT look LoRA included). **Get them now** on its tile fetches it ahead of the first round.

## Check picture

**Check picture** renders the next preview round on both computers, compares the two, and keeps both pictures in the run's `helper_check` folder.

| Result | What it means |
|---|---|
| 30 dB and up | Same kind of GPU: identical to within rounding |
| 22–30 dB | Another kind of GPU: small differences that build up over the steps |
| Under 22 dB | The two aren't set up alike (in testing: the SFT look on one side only), and it says so |

## When the helper goes away

If the helper stops answering, the next round renders on your own GPU as usual. Rounds already handed over wait for the helper and go out when it's back, even after the run has finished.

## Not yet

Sliders, edit previews, fine-tunes, and previews on a separate preview checkpoint (Klein's Distilled) render on your own GPU.
