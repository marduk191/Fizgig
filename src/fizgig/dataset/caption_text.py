"""Read a caption file whatever encoding it was saved in, without importing torch.

[fork] fizgig.dataset.image_dataset.decode_caption does this for training, but that module imports torch at load,
and the web server deliberately never does. The web server's Captions tab still reads caption files: its listing
(server.Handler._captions) and Find & Replace (web/captions.find_replace) both opened them as strict UTF-8 and
caught only OSError. So ONE Windows-1252 caption, a curly apostrophe saved by Notepad or carried in from another
tool, raised UnicodeDecodeError and took the whole request down, the valid files with it. The Captions tab would
not list the folder at all.

This is the same decoder, stdlib only. tests/test_caption_text.py runs both copies over the same bytes, so neither
can drift from the other unnoticed.
"""
import logging

logger = logging.getLogger(__name__)


def decode_caption(raw: bytes, caption_path: str) -> str:
    """A caption file's bytes as text: UTF-16 if there is a BOM (Notepad's "Unicode" save), then UTF-8 (BOM
    tolerated), then cp1252, then latin-1, which maps every byte and so cannot fail. Anything past UTF-8 is logged
    with the path, so a genuinely mis-encoded caption is visible instead of silently mangled."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            text = raw.decode("utf-16")
            logger.warning("Caption %s is UTF-16 — decoded, but re-save it as UTF-8.", caption_path)
            return text
        except UnicodeDecodeError:
            pass
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    for encoding in ("cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue  # cp1252 leaves a handful of bytes undefined; latin-1 won't.
        logger.warning("Caption %s is not valid UTF-8 — decoded as %s. Re-save it as UTF-8 to "
                       "silence this warning.", caption_path, encoding)
        return text
    logger.warning("Caption %s could not be decoded in any known encoding — undecodable bytes "
                   "replaced.", caption_path)
    return raw.decode("utf-8", errors="replace")


def read_caption(path):
    """The caption file at `path` as text, or None if it cannot be opened (missing, locked)."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    return decode_caption(raw, path)
