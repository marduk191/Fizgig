# People in group pictures: how to caption

[← Back to the README](../README.md)

On the Training tab's **Subjects** card, **Fizgig LoRA Isolation** trains a LoRA of one or more people that behaves with everyone else: your subjects can stand together in one picture, or next to someone the model already knows, and every face stays right. Without the trigger word you get the model's own people, just as before you trained. It's Fizgig's own techniques combined with DOP (Differential Output Preservation, from @ostris's AI-Toolkit).

**How it works:** [Fizgig LoRA Isolation](LORA_ISOLATION.md) · **Videos:** [see it in action](https://youtu.be/CNh4sN5vN_8) · [how to set up the Subjects card](https://youtu.be/8uW5NdL5b20) (also linked at the top of the card).

It depends on how each person's photos are captioned. Follow the rules below and Fizgig does the rest.

## 1. One person per folder

- Put each person's photos in their own folder.
- On the Subjects card, pick how many people the LoRA is of. Subject 1 is the training folder from the Start tab; subjects 2 to 4 get their own folder boxes.
- Give each person their own trigger word, and set it as that folder's trigger on the Captions tab.

## 2. Start every caption with the trigger, a comma, and what they are

```
zwxem, a woman smiling in a park, close-up, wearing a white top
qrxt, a man standing on a beach, wearing a grey jacket
rexo, a dog lying on a sofa
```

- Fizgig reads `zwxem, a woman` as **`zwxemwoman`** when it trains. Your caption files aren't changed. At the bottom of the Subjects card, each subject shows its word, one of its captions, and what training makes of it.
- The joined word keeps the person apart: it never shares the ordinary word "woman" that every other woman in a picture uses.
- Use a plain one-word class the model already knows: woman, man, girl, boy, dog, cat.
- Describing words are fine: `zwxem, a young blonde woman` still reads as `zwxemwoman`.
- If your captions don't start that way, Fizgig takes the class word your captions mention most ("…the woman is smiling"). The trigger is joined wherever it appears, e.g. "a photo of zwxemwoman in a park".
- If the captions never say what the subject is, tick **Correct a trigger word, or what a subject is** and type it in that subject's **what it is** box. Fizgig tells you when this is needed.

## 3. Choose trigger words that are clearly different from each other

- Use made-up words, not real names or dictionary words: `zwxem`, `qrxt`, `vytk`.
- Start each person's trigger with a different letter, and don't let two triggers share a beginning. `zwxem` and `zwxen` split into almost the same pieces inside the model.
- One trigger per folder: one person per folder.
- Avoid pieces that mean something: `qrxt` starts with `qr` and makes QR codes appear. Acronyms do the same (`jp`, `fr`, `fm`, `vk`, `url`, `ux`), and so do short words (`elf`, `ox`, `alt`, `pal`).
- End the trigger in a consonant. Something like `fyxo` fuses into `fy x ow oman`, which breaks the joined "woman".
- Two women's triggers always share the final `woman`. Keep everything before it different.

Trigger words that work well (checked against Krea 2's text encoder): `dlor`, `gwelt`, `jeskv`, `knurv`, `vorlk`, `thrisk`, `zwxem`. They share no piece or first letter with each other and join cleanly (`dlorwoman`, `gweltman`).

**Check yours before you caption:** **Check a trigger word**, at the bottom of the Training tab's Subjects card, reads a word with the training model's own text encoder, joined to what the subject is, and says whether it joins cleanly or shares a piece or first letter with your other subjects' words. If it doesn't join cleanly, it suggests made-up words that do.

## 4. Describe everything else as usual

Clothes, setting, pose, light, expression: caption them the way you normally would. The trigger stands only for the person, so anything you leave out tends to get learned as part of them.

## 5. Photos with other people in them

- Keep the trigger for your subject only, and describe anyone else plainly: `zwxem, a woman, standing next to a man in a grey coat`.
- If two of your subjects appear in the same photo, use both triggers: `zwxem, a woman, and qrxt, a man, laughing at a table`. Put the photo in either person's folder.
- Never put another subject's trigger in a folder unless that person is actually in the photo.

## What Fizgig adds for you

- It keeps each person out of prompts that don't use their word.
- It adds 16 side-by-side training pictures per person: one of their photos next to one of 32 fixed strangers (about 21 MB), which come with any model's download in Preferences. It uses your Image Prep face crops first, so a folder that mixes face crops with outfit or full-body shots needs no splitting. Other photos are framed on the face, and shots without a face are used only if nothing else is left. Untick **Side-by-side pictures: face crops first** to pick from every photo.

## How long to train

Two or more subjects of the same kind (two women, two men) need about 1.6x the epochs of a mixed pair. In testing, a woman-and-man pair was right at epoch 10, while two women needed epoch 16: both are held to the model's own "a woman" and share the joined `woman` piece, so telling them apart takes longer. Keep the in-between epochs and pick the best one. The Training tab says so when your subjects share a kind.

## Prompting the finished LoRA

- **Use the joined word:** `zwxemwoman`, `qrxtman`. Not "zwxem, a woman". Training previews read your prompts the way training reads your captions, so `zwxem, a woman at a cafe` previews as `zwxemwoman at a cafe`.
- **Name the other people before your subjects:** `three friends at a garden party with zwxemwoman and qrxtman`. That's the order it was trained in, and it holds best that way.
- **Be specific about the others:** `a Japanese man in his 60s with round glasses` keeps its own look better than "a man". Describing your own subjects helps too: see [How many subjects](#how-many-subjects).
- **Without a trigger it's the model's own people:** `a woman and a man at a cafe` draws ordinary people, not your subjects.

Example, two people:

| Folder | Trigger | Captions start | Prompt with |
|---|---|---|---|
| subject 1 (training folder) | `qrxt` | `qrxt, a man, …` | `qrxtman` |
| subject 2 | `zwxem` | `zwxem, a woman, …` | `zwxemwoman` |

Prompt: `an East Asian man in his 50s in the middle, zwxemwoman on the left and qrxtman on the right, a tight close-up photo, cheek to cheek, all looking at the camera`

## How many subjects

- **3 or 4 people** train well together, and 2 or 3 of them can share one picture and stay themselves.
- **With a person the model already knows well in the picture as well,** 2 of your subjects plus that person works almost every time. 3 subjects plus that person is hit and miss.
- **Describe your subjects in the prompt,** for example their hair colour: `zwxemwoman with black hair`. It helps with any number of subjects, matters little with 2 plus a person the model knows, and is essential with 3 plus one. With 3, giving each subject's ethnicity helps too.

## Captioning it yourself

Tick **Use my captions exactly as written** when you'd rather write the joined word yourself and have Fizgig change nothing in your captions: no joining, no guessing what the subject is.

- **Write the joined word into every caption,** in place of the trigger and class: `zwxemwoman smiling in a park, close-up, wearing a white top`. The word is your trigger and the class run together, with no space or comma.
- **Set that joined word as the folder's trigger** on the Captions tab (`zwxemwoman`), so the Subjects card knows which word is the subject.
- **What the subject is** comes from the end of the word (`zwxemwoman` is a woman). If your word doesn't end in a class word (woman, man, girl, boy, dog, cat…), tick **Correct a trigger word, or what a subject is** and type it in that subject's **what it is** box.
- **Everything else in this guide still applies:** different-looking triggers (section 3), everything else described as usual (section 4), other people described plainly (section 5).
- **Fizgig still adds** the side-by-side pictures and keeps each person out of prompts without their word. The Subjects card warns you when a folder's captions don't use the word.
- **Preview prompts** are used as you write them, so use the joined word there too.
