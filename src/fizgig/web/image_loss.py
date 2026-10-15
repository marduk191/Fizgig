"""Per-image loss charts for the Runs page: each image's RESIDUAL by epoch - its loss minus the mean loss at the same
noise level (timestep bucket) over the whole log - from a run's loss_log/per_image_loss.jsonl, with the loss watch's
latest verdicts (problem_images.json). The residual is the loss watch's own measure (training/loss_logger.py
epoch_boundary: 40 buckets, means over every record), so a chart reads as the watch reads it: above zero = harder than
the dataset at that noise level, falling = being learned. A family whose description leaves loss_watch off never
writes the log, so it never has charts."""

import json
import os

LOG = os.path.join("loss_log", "per_image_loss.jsonl")
REPORT = os.path.join("loss_log", "problem_images.json")
N_BUCKETS = 40              # = training/loss_logger._N_BUCKETS
_IMAGES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".gif")

_cache = {}                 # log path -> ((mtime, size), result)


def log_path(output_dir):
    p = os.path.join(output_dir or "", LOG)
    return p if output_dir and os.path.isfile(p) else ""


def residuals(output_dir):
    """{"epochs": [..], "images": [{key, series: [[epoch, residual], ...], mean, recent, n}], "report": {..}} - a
    batch-mean key ("a|b") is no single image's and is left out, as the watch's report leaves it out."""
    p = log_path(output_dir)
    if not p:
        return None
    stamp = (os.path.getmtime(p), os.path.getsize(p))
    hit = _cache.get(p)
    if hit and hit[0] == stamp:
        return hit[1]
    recs, bsum, bcnt = [], [0.0] * N_BUCKETS, [0] * N_BUCKETS
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                t, loss = float(r["t"]), float(r["loss"])
            except (ValueError, KeyError, TypeError):
                continue                             # a line cut short by a crash / a write in progress
            b = min(int(min(max(t, 0.0), 0.999999) * N_BUCKETS), N_BUCKETS - 1)
            bsum[b] += loss
            bcnt[b] += 1
            recs.append((str(r.get("key", "")), int(r.get("epoch", 0)), b, loss))
    mean = [bsum[i] / bcnt[i] if bcnt[i] else 0.0 for i in range(N_BUCKETS)]
    per = {}
    for key, ep, b, loss in recs:
        if "|" in key:
            continue
        per.setdefault(key, {}).setdefault(ep, []).append(loss - mean[b])
    epochs = sorted({ep for _k, ep, _b, _l in recs})
    images = []
    for key, eps in per.items():
        series = [[e, round(sum(v) / len(v), 6)] for e, v in sorted(eps.items())]
        vals = [s[1] for s in series]
        recent = vals[-5:]                           # the watch's window (PerImageLossWatch window=5)
        images.append({"key": key, "series": series, "mean": round(sum(vals) / len(vals), 6),
                       "recent": round(sum(recent) / len(recent), 6), "n": sum(len(v) for v in eps.values())})
    report = {}
    try:
        with open(os.path.join(output_dir, REPORT), encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, ValueError):
        pass
    verdicts = report.get("images") or {}
    for im in images:
        v = verdicts.get(im["key"]) or {}
        im["verdict"], im["multiplier"] = v.get("verdict"), v.get("multiplier")
    out = {"epochs": epochs, "images": images,
           "report": {k: report.get(k) for k in ("epoch", "plateaued", "best_epoch_estimate", "improving_count",
                                                 "apply_lr", "batched")}}
    _cache[p] = (stamp, out)
    return out


def caption_state(output_dir):
    """(keys waiting in the trainer's caption queue, {key: [{epoch, caption, attempt, auto}]} already re-encoded)."""
    d = os.path.join(output_dir or "", "loss_log")
    queued = set()
    for name in ("caption_updates.json", "caption_updates.json.processing"):
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                queued.update(json.load(f).keys())
        except (OSError, ValueError, AttributeError):
            pass
    try:
        with open(os.path.join(d, "caption_updates_applied.json"), encoding="utf-8") as f:
            applied = {k: (v if isinstance(v, list) else [v]) for k, v in json.load(f).items()}
    except (OSError, ValueError, AttributeError):
        applied = {}
    return queued, applied


def read_caption(folder, key, ext):
    p = os.path.join(folder or "", os.path.basename(key) + ext)
    try:
        with open(p, encoding="utf-8-sig", errors="replace") as f:     # a BOM never rides into the embedding
            return f.read().strip(), True
    except OSError:
        return "", os.path.isdir(folder or "")


def save_caption(run, key, caption, ext):
    """An image's caption fixed from the per-image loss view (the desktop's Problem Images editor): the caption file
    rewritten, and - while the run is going or paused - the edit queued for the trainer, which re-encodes it at the
    next epoch boundary and restarts that image's history (fizgig/families/loss_watch.py)."""
    caption = str(caption or "").strip()
    if not caption:
        raise ValueError("The caption is empty - not saved.")
    key = os.path.basename(str(key))
    folder = run.get("image_folder") or ""
    wrote, queued = False, False
    if os.path.isdir(folder):
        p = os.path.join(folder, key + ext)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(caption)
        os.replace(tmp, p)
        wrote = True
    if run.get("result") in ("running", "paused") and run.get("output_dir"):
        d = os.path.join(run["output_dir"], "loss_log")
        os.makedirs(d, exist_ok=True)
        q = os.path.join(d, "caption_updates.json")
        updates = {}
        if os.path.exists(q):
            try:
                with open(q, encoding="utf-8") as f:
                    updates = json.load(f)
            except (OSError, ValueError):
                # other edits are waiting in a file we can't read: writing just this one would lose them
                raise ValueError("The trainer's caption queue couldn't be read - not overwriting it. Try again in a "
                                 "moment.")
            if not isinstance(updates, dict):
                updates = {}
        updates[key] = caption
        with open(q + ".tmp", "w", encoding="utf-8") as f:
            json.dump(updates, f, indent=2)
        os.replace(q + ".tmp", q)                    # atomic: the trainer never reads half a file
        queued = True
    if not wrote and not queued:
        raise ValueError("The run's training folder isn't there any more - nothing saved.")
    return {"wrote": wrote, "queued": queued}


def image_paths(folder, keys):
    """{key: picture path} - a key is the picture's name without its extension, in the run's training folder."""
    out = {}
    try:
        names = os.listdir(folder or "")
    except OSError:
        return out
    want = set(keys)
    for n in names:
        stem, ext = os.path.splitext(n)
        if ext.lower() in _IMAGES and stem in want and stem not in out:
            out[stem] = os.path.join(folder, n)
    return out


def owner_runs(runs):
    """{run id: True} for the run each log belongs to: a fresh run rotates the log away, so a folder's log is the
    newest run there that started before the log was last written."""
    by_dir = {}
    for r in runs:
        p = log_path(r.get("output_dir"))
        if p:
            by_dir.setdefault(os.path.normcase(os.path.abspath(p)), (p, []))[1].append(r)
    owners = {}
    for p, rs in by_dir.values():
        written = os.path.getmtime(p)
        fits = [r for r in rs if float(r.get("started_at") or 0) <= written + 60]
        if fits:
            owners[max(fits, key=lambda r: float(r.get("started_at") or 0))["id"]] = True
    return owners
