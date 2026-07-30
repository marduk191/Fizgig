# Fizgig v2.12.0 — Get the models without the shopping list

## One button instead of five downloads

Setting Fizgig up used to mean hand-downloading around **32 GB across five files from four
HuggingFace repos**, then pasting five absolute paths into Preferences and hoping you'd picked the
right variant out of the eight similarly-named files in the folder.

Preferences now has a **⬇ Download models for me** button under each model card. It fetches that
family's files, verifies each one, and fills in the paths for you.

- **Krea 2 needs no HuggingFace account.** Those files aren't gated — click and go.
- **Klein needs a free read token**, because Black Forest Labs require every user to accept their
  licence personally. Fizgig gives you the three pages to accept on as **copyable links** with Copy
  and Open buttons, then takes your token. It's held for that download only and never written to disk.
- **The helper models come too** — Florence-2, the face model behind the Look Filter and likeness
  scoring, and the EN→ZH translator (~1.6 GB). They're fetched first, so the Captions tab and Look
  Filter work immediately rather than stalling to download the first time you open them.
- **A real progress window** — current file, percentage, GB of GB, and Cancel. Cancelling is safe:
  interrupted downloads resume where they left off rather than starting over.

**Your own choices are never overwritten.** If a path already points at a file that exists, it's
left alone — including deliberately different ones like the fp8_scaled RAW DiT instead of bf16,
Klein's fp8mixed text encoder, or a community fine-tune. It tells you when it's respecting a file
that isn't the one it would have fetched.

Every download is verified by parsing the safetensors header and checking the size, so a truncated
transfer or an HTML error page can't sit at the destination filename and fail incomprehensibly
three steps later.

There's a CLI too, and it's the same code the GUI runs:

```bash
python -m fizgig.scripts.fetch_models --family krea2   # ~32 GB, no account needed
python -m fizgig.scripts.fetch_models --family klein   # ~34 GB, needs a token
python -m fizgig.scripts.fetch_models --family tools   # Florence-2, face model, translator
python -m fizgig.scripts.fetch_models --all --include-optional
```

## Faster downloads

Adds **hf_xet**, which the HuggingFace repos Fizgig pulls from are backed by. It's a straight
transfer speedup on the files where it matters most — the 26 GB Krea 2 RAW DiT and the 9.5 GB Klein
base. Installed automatically on update; without it, downloads fall back to plain HTTP and still work.

---

**Upgrading:** run `update_fizgig.bat`, or `git pull`.
