"""Pictures as they are meant to be seen. A camera often stores the pixels sideways and writes a rotation flag (EXIF
orientation) that viewers apply; PIL reads the stored pixels as they are. OpenCV already applies the flag, so the
face finders see the picture upright: every PIL read of a training picture goes through here to match."""

ORIENTATION = 0x0112


def orientation(im):
    try:
        return int(im.getexif().get(ORIENTATION, 1) or 1)
    except Exception:
        return 1


def upright(im):
    """The open picture with its rotation flag applied (the flag dropped), or the same picture when it has none."""
    if orientation(im) in (2, 3, 4, 5, 6, 7, 8):
        from PIL import ImageOps
        return ImageOps.exif_transpose(im)
    return im


def open_upright(path):
    from PIL import Image
    return upright(Image.open(path))


def upright_size(im):
    """(width, height) as seen, from the header alone: flags 5-8 turn the picture a quarter."""
    w, h = im.size
    return (h, w) if orientation(im) in (5, 6, 7, 8) else (w, h)
