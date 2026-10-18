# Fizgig v8.4.0: Fizgig LoRA Isolation

<a href="https://youtu.be/CNh4sN5vN_8"><img src="https://img.youtube.com/vi/CNh4sN5vN_8/maxresdefault.jpg" alt="Fizgig LoRA Isolation: watch the demo on YouTube" width="720"></a>

**Fizgig LoRA Isolation: train 2 identities into one LoRA and keep everyone else looking like themselves.** A normal LoRA of a person spreads their face onto other people in the picture: strangers in the background, a friend beside them, a crowd. Fizgig LoRA Isolation trains your subjects so they can stand together in one picture, or next to someone the model already knows well, and every face stays its own.

On Krea 2, against a normal LoRA trained on the same pictures with the same settings: 48 of 50 strangers read as the trained person with the normal LoRA, 0 of 50 with Fizgig LoRA Isolation, and likeness of the trained person was the same (ArcFace 0.78 against 0.77). [See it in action](https://youtu.be/CNh4sN5vN_8).

A second person in a LoRA is usually the problem, the two faces blending into each other. Here the second person is part of the fix: learning two identities together is what teaches the LoRA that each face belongs to its own trigger. They don't need to be the same gender: in [the demo](https://youtu.be/CNh4sN5vN_8) the second person was a woman, and every stranger was a man.

**How it works:** DOP at the base, with Fizgig's own layers on top.
- **DOP (Differential Output Preservation),** from @ostris's AI-Toolkit: each picture also trains with its trigger removed, held to what the base model would draw.
- **Multi-subject identity anchoring:** two or more identities trained together, so each binds to its own trigger instead of drifting onto every face.
- **Fused-token class decoupling:** the trigger and the class word merge into a single token (`gwelt, a man` trains as `gweltman`), so identity never binds to the class word everyone else shares.
- **In-frame negative-pair compositing:** each person composited next to strangers, with causal-order captions (the stranger first).
- **Per-sample LR scaling, strided DOP and balanced sampling:** the composites train at 2x the learning rate, DOP runs every second step at 2x weight, and subjects are balanced across each epoch.
- **Class-aware training paths:** people, animals and things, and styles each get their own recipe.

It's set up on the new **Subjects** card on the Training tab, and works on every model except ACE-Step. It has been tested on Krea 2, with more model testing to follow.

**The Subjects card:**
- **Who is this LoRA of?** 1 to 4 subjects, each a folder of its own pictures with its own trigger word in the captions.
- **What each subject is:** a person, an animal or thing, or a style or look. Left on From its captions, Fizgig works it out from words like man and woman.
- **Balance the subjects** repeats the smaller folders so each subject trains about as much as the largest.
- **Your subject and other people:** Off (a normal LoRA), Leave the rest of the model alone (DOP: a prompt without your trigger gives the model's own people), or Fizgig LoRA Isolation. Isolation takes about 1.5x as long to train.
- Under each subject, the card shows exactly what training will use: the word it trains as, one of its captions, and the side-by-side pictures. Your caption files are never changed.

**Getting the best from it:**
- Train two people even if you only want yourself. Learning two faces is part of what keeps strangers clean.
- Two trained people in one picture, plus a person the model already knows, work well. Three or more trained people in one picture come out right less often.
- Prompt with the joined word the card shows, for example `gweltman man at a cafe`, and describe the other people in your prompt.
- The side-by-side pictures use a set of 32 generated strangers with natural skin texture, so they never pull your LoRA towards poor skin and no real person's photo is used. The set downloads with your models (21 MB).

<a href="https://youtu.be/8uW5NdL5b20"><img src="https://img.youtube.com/vi/8uW5NdL5b20/maxresdefault.jpg?v=2" alt="How to set up Fizgig LoRA Isolation: watch on YouTube" width="420"></a>

How to set it up: [video](https://youtu.be/8uW5NdL5b20) (also linked at the top of the Subjects card) · [captioning guide](https://github.com/shootthesound/Fizgig/blob/master/docs/MULTI_PERSON.md).

**On H3** the subjects train from their photos: clips and voice files in a subject's folder are left out, a folder with no photos is refused at the start, and reference distillation is off while Isolation is on. **On ACE-Step** the card offers neither DOP nor Isolation.

**Multi Concept on every model:** 2 to 4 subject folders now work on every model's Standard LoRA and fine-tune, not only H3.

**Smaller changes:**
- Training Parameters: the plain boxes come first, and the on/off tiles (loss watch, per-image LR, look warm-up, auto-recaption, Fast Identity, training adapter) sit together at the bottom.
- Fine-tunes no longer use Adaptive LR on any model. A tick left on from a LoRA run used to reach the trainer; now the Learning Rate box is used as set.
- A picture edited after it was cached is encoded again, rather than training on its old cached version.
