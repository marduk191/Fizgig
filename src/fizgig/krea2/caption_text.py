"""Krea 2's Qwen3-VL caption instructions and the Captions tab's task menu: text only, so the web UI can list them
without loading torch (embedder.py, which captions with them, imports them from here)."""

# Ordering comes from the community's structured-caption template (trigger, features, clothing,
# pose, expression, setting, lighting, camera angle) — a fixed order is what makes a whole dataset's
# captions structurally consistent, which is the part a per-image VLM otherwise gets wrong.
#
# What is deliberately NOT taken from that template: its advice to OMIT invariant features ("if your
# character always has blue eyes, don't mention it"). That is SD1.5/SDXL-era guidance and it is
# wrong here. On LLM-conditioned models the omit-to-bake-in trick is dead — unnamed features don't
# bake in, unnamed CONTRADICTIONS fight, and the salient unnamed deviation is exactly the poison the
# per-image loss watch keeps flagging. Naming a thing doesn't stop the model learning it; it binds
# it to a token you can then steer. Hence "name anything prominent", not "omit what's constant".
#
# Two rules earned from real output (29 Jul):
#   SUBJECT_RULE — the 4B model hedged to "a person" in roughly a third of captions. That is a
#   worse token than "a woman": the dataset then teaches the trigger against an inconsistent
#   subject noun, and the noun is the one word every caption shares.
#   NO_PREAMBLE_RULE — it opened with "This image shows…" constantly on the longer tasks. The
#   preamble is pure noise in a training caption and, prepended after the trigger, reads as
#   "<trigger>, this image shows…". Belt and braces: the instruction asks, and
#   _strip_caption_preamble removes it deterministically afterwards.
SUBJECT_RULE = (
    "Name the subject with the specific term that is visually apparent — 'a woman', 'a man', "
    "'a girl', 'a boy' — and not the vague 'a person'. Use 'a person' only when the image "
    "genuinely does not show enough to tell. "
)
NO_PREAMBLE_RULE = (
    "Begin directly with the subject. Never open with 'This image shows', 'The image depicts', "
    "'In this image', 'Here we see', 'The photo shows' or any similar preamble. "
)

CAPTION_INSTRUCTION = (
    "Write one factual training caption for this image as a single sentence, covering these in "
    "order: the subject and what they are doing; the camera viewpoint (e.g. 'viewed from behind', "
    "'side profile', 'close-up') and whether the face is visible; their pose; their clothing; the "
    "setting; the lighting. Use the same order and the same plain phrasing every time. "
    + SUBJECT_RULE + NO_PREAMBLE_RULE +
    "Name anything prominent a viewer would notice, especially anything unusual about the angle, "
    "the framing, or what is hidden or cropped. State only what is visible — no speculation, no "
    "proper names, no style or quality commentary."
)

# Second-attempt instruction: if the standard caption didn't unstick the image, the miss is
# probably something salient the short caption skipped — go exhaustive so every visible element
# that could contradict the conditioning gets named.
DETAILED_CAPTION_INSTRUCTION = (
    "Write a detailed factual training caption for this image, 2-4 sentences. Cover: the subject "
    "and exactly how much of them is visible (state the camera viewpoint and explicitly whether "
    "the face is visible or hidden), their pose and body position, every visible clothing item "
    "with colors, hair style and color, any objects they hold or touch, anything partially "
    "blocking or cropping the subject, the lighting, and the background/setting with its main "
    "objects. " + SUBJECT_RULE + NO_PREAMBLE_RULE +
    "State only what is visible — no speculation, no names, no style commentary."
)


# The system prompt used when ENCODING text for training and inference — NOT a captioning
# instruction. It must stay byte-identical to ComfyUI's Text-Encode-(Krea2) node: changing it
# would silently alter every cached embedding and desync training from ComfyUI. Module-level so
# the GUI can display it read-only without duplicating the string.
ENCODE_SYSTEM_DESCRIPTOR = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
)

SHORT_CAPTION_INSTRUCTION = (
    "Write one short factual caption for this image — a single clause naming the subject, what "
    "they are doing, and the setting. " + SUBJECT_RULE + NO_PREAMBLE_RULE +
    "State only what is visible. No speculation, no names, no style commentary."
)

DETAILED_DESCRIPTION_INSTRUCTION = (
    "Describe this image in 2-3 factual sentences, starting with the subject: their pose and "
    "clothing, the camera viewpoint, the lighting, and the setting. "
    + SUBJECT_RULE + NO_PREAMBLE_RULE +
    "State only what is visible — no speculation, no names, no style commentary."
)

# --- style captioning ------------------------------------------------------------------------
# The identity instructions above are written for a dataset where the subject is a person and the
# lighting, viewpoint and clothing all VARY — you name them so they stay steerable. A style dataset
# inverts that: the look is the constant you are training, and the subject is whatever happens to
# be in front of it. So the caption takes the content and leaves the style unnamed, which is what
# lets the LoRA bind the look to the trigger word the GUI prepends.
#
# This instruction is deliberately SHORT, and that is the finding, not an oversight. Earlier
# versions of this preset stacked four rule fragments — name the subject specifically, no preamble,
# describe it as if it were a real place, never mention medium/technique/grade — and by the obvious
# metric they won easily: on a layered paper-cut set they leaked style words into 1 caption in 9,
# where this one-liner leaks into 7. But the one-liner is what trains better, tested on real runs
# across both Krea 2 and Klein, and it is not close.
#
# The likely reason is worth keeping, because it cuts against the instinct to keep adding rules:
#   - Leaking CONSISTENTLY is not the same failure as leaking half the time. A word in 7 captions
#     of 9 behaves like a style tag; a word in 4 of 9 teaches the model that some of these images
#     have a property the others lack. The rule stack was tuned against the wrong number.
#   - The short instruction produces captions roughly 2.4x richer (~70 words vs ~30). More content
#     named means more of the image accounted for, which leaves the style as the cleaner residual.
#     The rules were buying leak-purity with detail, and detail is what the LoRA works from.
#
# Token budget is 160 rather than the ~90 the length implies: at 90 every caption on the test set
# was cut off mid-word.
STYLE_CAPTION_INSTRUCTION = (
    "Describe the image with zero references to the image style, just the factual contents of "
    "what is depicted."
)

# The task menu the Captions tab offers for this model. Lives here rather than in the GUI so the
# trainer's auto-recaption and the GUI read the same text — the instruction is part of the
# captioning contract, not a piece of UI.
#   key -> (menu label, instruction, suggested max_new_tokens)
# "training" is the default: it is the doctrine-aligned instruction auto-recaption already uses
# (name the viewpoint, say whether the face is visible), which is what makes a caption safe to
# train on rather than merely accurate.
CAPTION_TASKS = {
    "training":   ("Training caption (viewpoint-aware)", CAPTION_INSTRUCTION, 120),
    "short":      ("Short caption", SHORT_CAPTION_INSTRUCTION, 60),
    "detailed":   ("Detailed description", DETAILED_DESCRIPTION_INSTRUCTION, 160),
    "exhaustive": ("Exhaustive detail", DETAILED_CAPTION_INSTRUCTION, 240),
    # Style last: the four above are the identity path, which is the common case and the one
    # auto-recaption uses.
    "style":      ("Style — contents only (trigger word names the style)",
                   STYLE_CAPTION_INSTRUCTION, 160),
}
DEFAULT_CAPTION_TASK = "training"
