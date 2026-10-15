"""A LoRA's run (the epochs saved beside it) and the moving-average mix over them - plain file and number work, no
torch, so the web UI's server can list a run without loading a model. The workbench mixes with these (Repair Studio's
"Epochs of this run" and the donor's epoch blend)."""

import os
import re

_EPOCH_RE = re.compile(r"^(?P<stem>.+)-(?P<n>\d{6})\.safetensors$", re.I)


def run_epochs(path):
    """[(epoch, file)] of the run `path` belongs to, oldest first: the <name>-NNNNNN.safetensors files beside it. A
    final file without a number (the run's last save) counts as the epoch after the highest numbered one."""
    folder, base = os.path.split(path)
    m = _EPOCH_RE.match(base)
    stem = m.group("stem") if m else os.path.splitext(base)[0]
    found = {}
    for f in os.listdir(folder or "."):
        mm = _EPOCH_RE.match(f)
        if mm and mm.group("stem") == stem:
            found[int(mm.group("n"))] = os.path.join(folder, f)
    if not m:
        found[(max(found) + 1) if found else 1] = path
    return sorted(found.items())


def ema_weights(epochs, up_to, history, floor=0.005):
    """{epoch: weight} summing to 1: an EMA over the epochs up to `up_to`, as if it had run through training once per
    epoch with decay `history` (0 = that epoch alone). The newest gets 1 - history, each older one `history` times
    the next, and the oldest the remainder (the EMA's starting value)."""
    eps = [e for e in sorted(epochs) if e <= up_to]
    if not eps:
        return {}
    h = min(max(float(history), 0.0), 0.999)
    w = {}
    for k, e in enumerate(reversed(eps)):
        w[e] = (1 - h) * h ** k
    w[eps[0]] = h ** (len(eps) - 1)
    # an epoch under `floor` is left out and the rest rescaled: a long history would otherwise load dozens of files
    # for a few percent between them
    kept = {e: v for e, v in w.items() if v >= floor}
    total = sum(kept.values())
    return {e: v / total for e, v in kept.items()}
