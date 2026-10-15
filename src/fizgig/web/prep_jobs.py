"""The web UI's Image Prep: a prep run (resize / PNG / face close-ups) and the Look Consistency Filter's scan, each
in a background thread on the Fizgig machine, with their state for the page to poll. The work itself is
fizgig/dataset/prep.py - the same code the desktop app's Image Prep tab runs. Face tools run on the CPU.
"""
import base64
import io
import os
import threading
import time

from fizgig.dataset import prep

_lock = threading.Lock()
_job = {"state": "idle", "log": "", "folder": "", "started": 0.0}
_look = {"state": "idle", "folder": "", "baselines": [], "scores": {}, "progress": [0, 0], "error": ""}
_embed_cache = {}             # (path, mtime) -> embedding or None
_detector = None
_embedder = None
_model_lock = threading.Lock()


def faces_available() -> bool:
    try:
        from fizgig.face_utils import is_face_detection_available
        return bool(is_face_detection_available())
    except Exception:
        return False


def _face_detector():
    global _detector
    with _model_lock:
        if _detector is None:
            from fizgig.face_utils import FaceDetector
            _detector = FaceDetector()
    return _detector


def busy() -> bool:
    with _lock:
        return _job["state"] == "running" or _look["state"] == "running"


# ---- a prep run -------------------------------------------------------------------------------------------------
def status(since=0):
    with _lock:
        log = _job["log"]
        return {"state": _job["state"], "folder": _job["folder"], "log": log[since:], "size": len(log)}


def _log(text):
    with _lock:
        _job["log"] = (_job["log"] + text)[-400_000:]


def start(folder, mode, mp, face_label, padding, replace):
    if not folder or not os.path.isdir(folder):
        raise ValueError("Choose the training folder on the Start tab first.")
    if mode not in prep.MODES:
        raise ValueError("Pick what to do.")
    if mode != "Resize Only" and not faces_available():
        raise ValueError("Face detection isn't installed on this machine: run install_fizgig.py to add it.")
    with _lock:
        if _job["state"] == "running":
            raise ValueError("A prep run is already going.")
        if _look["state"] == "running":
            raise ValueError("The Look Filter is scanning. Wait for it to finish.")
        _job.update(state="running", log="", folder=folder, started=time.time())

    def work():
        try:
            det = _face_detector() if mode != "Resize Only" else None
            prep.Prep(_log, det).run(mode, folder, float(mp), face_label, float(padding), bool(replace))
            _log("\nDone.\n")
            end = "done"
        except Exception as e:
            _log(f"\nERROR: prep failed — {type(e).__name__}: {e}\n")
            end = "failed"
        with _lock:
            _job["state"] = end
    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


def test_faces(path, face_label):
    """Test face detection on one photo: the picture with the faces boxed (the chosen one green) as a JPEG data URL,
    and each face found. Writes nothing."""
    from PIL import Image
    from fizgig.face_utils import draw_face_boxes
    if not os.path.isfile(path):
        raise ValueError("That picture isn't there.")
    det = _face_detector()
    faces = det.detect_all(path)
    if not faces:
        return {"faces": [], "image": None}
    job = prep.Prep(lambda _t: None, det)
    sel, _ = job.select_face(faces, prep.face_mode_of(face_label))
    hi = faces.index(sel) if sel in faces else None
    with Image.open(path) as img:
        img = img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img
        shown = draw_face_boxes(img, faces, highlight_index=hi).convert("RGB")
    shown.thumbnail((1400, 1400))
    buf = io.BytesIO()
    shown.save(buf, "JPEG", quality=88)
    return {"image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
            "faces": [{"gender": f.gender, "area": int(f.area), "selected": i == hi} for i, f in enumerate(faces)]}


# ---- the Look Consistency Filter --------------------------------------------------------------------------------
def _embed(path):
    """A face embedding through the (path, mtime) cache: re-scans and baseline swaps cost almost nothing."""
    global _embedder
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    if key not in _embed_cache:
        with _model_lock:
            if key not in _embed_cache:
                if _embedder is None:
                    from fizgig.face_utils import FaceEmbedder
                    _embedder = FaceEmbedder()
                try:
                    _embed_cache[key] = _embedder.embed(path)
                except Exception:
                    _embed_cache[key] = None
    return _embed_cache[key]


def embed_face_box(path):
    """(embedding, bbox) for the largest face in a picture file - the desktop's _repair_embed_pil, on the same lazily
    made embedder as the Look Filter. The unpadded detection gives both; the pad-retry fallback (frame-filling faces)
    gives the embedding only (its box is in padded space). (None, None) without a face."""
    global _embedder
    try:
        import cv2
        import numpy as np
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None, None
        with _model_lock:
            if _embedder is None:
                from fizgig.face_utils import FaceEmbedder
                _embedder = FaceEmbedder()
            _embedder._ensure_loaded()
            faces = _embedder._app.get(bgr)
            if faces:
                f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                x1, y1, x2, y2 = (int(v) for v in f.bbox)
                h, w = bgr.shape[:2]
                return np.asarray(f.normed_embedding, dtype=np.float32), (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
            faces = _embedder._detect_with_pad_retry(bgr)
            if faces:
                f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                return np.asarray(f.normed_embedding, dtype=np.float32), None
    except Exception:
        pass
    return None, None


def look_status():
    with _lock:
        scores = dict(_look["scores"])
        base = list(_look["baselines"])
        state, err, prog, folder = _look["state"], _look["error"], list(_look["progress"]), _look["folder"]
    med, cut = prep.cutoff(scores) if scores else (None, None)
    rows = []
    for p, s in scores.items():
        label, colour, blurb = prep.verdict(s)
        rows.append({"path": p, "name": os.path.basename(p), "score": s, "verdict": label, "colour": colour,
                     "blurb": blurb, "baseline": p in base, "suggest": bool(cut is not None and s is not None
                                                                          and s < cut and p not in base)})
    rows.sort(key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else 0.0))   # worst first
    return {"state": state, "error": err, "progress": prog, "folder": folder, "baselines": base, "rows": rows,
            "median": med, "cutoff": cut}


def look_scan(folder, baselines):
    if not folder or not os.path.isdir(folder):
        raise ValueError("Choose the training folder on the Start tab first.")
    if not faces_available():
        raise ValueError("Face detection isn't installed on this machine: run install_fizgig.py to add it.")
    base = [os.path.normpath(b) for b in baselines or []]
    if len(base) != 3 or not all(os.path.isfile(b) for b in base):
        raise ValueError("Pick exactly 3 baseline pictures.")
    files = prep.look_files(folder)
    if not files:
        raise ValueError("No pictures in the training folder.")
    with _lock:
        if _look["state"] == "running":
            raise ValueError("A scan is already going.")
        if _job["state"] == "running":
            raise ValueError("Image prep is running. Wait for it to finish.")
        _look.update(state="running", folder=folder, baselines=base, scores={}, progress=[0, len(files)], error="")

    def progress(i, total):
        with _lock:
            _look["progress"] = [i, total]

    def work():
        try:
            scores = prep.score(files, base, _embed, progress)
            try:
                prep.save_scores(folder, base, scores)     # for training's "Warm up look outliers"
            except OSError:
                pass
            with _lock:
                _look.update(state="done", scores=scores)
        except Exception as e:
            with _lock:
                _look.update(state="failed", error=str(e))
    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


def look_move(folder, paths, caption_ext=".txt"):
    """Move the marked pictures (and their captions) to excluded_by_look/; they leave the scan's list."""
    with _lock:
        if _look["state"] == "running":
            raise ValueError("A scan is going. Wait for it to finish.")
        known = set(_look["scores"])
    paths = [os.path.normpath(p) for p in paths or [] if os.path.normpath(p) in known]   # a marked baseline goes too
    moved, errors = prep.move_out(folder, paths, (".txt", caption_ext or ".txt"))
    with _lock:
        for p in moved:
            _look["scores"].pop(p, None)
    return {"moved": len(moved), "errors": errors, "dest": os.path.join(folder, "excluded_by_look"),
            "left": len(_look["scores"])}
