"""This computer's training folders, for a pod's page inside the Fizgig app window (app_window.Api): the list, a
folder's files, and a file's bytes a piece at a time - so the pod's page can send a local dataset up with its own
resumable upload ("From this computer" on its Start tab). Read-only, and only ever a training folder's own files:
anything outside the training data folder (Preferences > Input Directories > Training images, else training_data/)
is refused.
"""
import base64
import io
import os
import re

MAX_READ = 2 * 2 ** 20                  # bytes per read_local call (the bridge carries them as base64 text)
CAPTIONS = (".txt", ".caption", ".tags", ".json")


def _root():
    from fizgig.web.server import uploads_root
    return os.path.realpath(uploads_root())


def _folder(path):
    """The training folder `path` names (a folder directly inside the training data folder), or ValueError."""
    root, p = _root(), os.path.realpath(str(path or ""))
    if os.path.dirname(p) != root or not os.path.isdir(p):
        raise ValueError("Not one of this computer's training folders.")
    return p


def _wanted(name):
    from fizgig.web.server import MEDIA
    ext = os.path.splitext(name)[1].lower()
    return not name.startswith(".") and (ext in MEDIA or ext in CAPTIONS)


def _thumb(path, side=96):
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.thumbnail((side, side))
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def local_datasets():
    """The training folders, as the local Start tab lists them, each with its size and up to 4 small thumbnails."""
    from fizgig.web.server import list_datasets
    out = []
    for d in list_datasets()["datasets"]:
        try:
            size = sum(e.stat().st_size for e in os.scandir(d["path"]) if e.is_file() and _wanted(e.name))
        except OSError:
            size = 0
        out.append({**{k: d[k] for k in ("name", "path", "pictures", "clips", "songs", "captioned", "time")},
                    "bytes": size, "thumbs": [t for t in (_thumb(p) for p in d.get("thumbs", [])[:4]) if t]})
    return {"root": _root(), "datasets": out}


def local_files(folder):
    """A training folder's own files Fizgig trains on (pictures, clips, songs, their captions): name, size, mtime."""
    p = _folder(folder)
    files = []
    for e in sorted(os.scandir(p), key=lambda e: e.name.lower()):
        if e.is_file() and _wanted(e.name) and not re.search(r"\.part$", e.name):
            st = e.stat()
            files.append({"name": e.name, "size": st.st_size, "mtime": st.st_mtime})
    return files


def read_local(folder, name, offset, n):
    """Up to 2 MB of one of a training folder's files, from `offset`, as base64."""
    p = _folder(folder)
    name = str(name or "")
    if os.path.basename(name) != name or not _wanted(name):
        raise ValueError("Not a training file.")
    path = os.path.join(p, name)
    if not os.path.isfile(path):
        raise ValueError(f"{name} isn't there any more.")
    with open(path, "rb") as f:
        f.seek(max(0, int(offset)))
        return base64.b64encode(f.read(max(0, min(int(n), MAX_READ)))).decode()
