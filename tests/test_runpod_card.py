"""RunPod auto-stop and the pod storage line (Fizgig 8 web server) — headless, no GPU.

The auto-stop is the dangerous one and gets the most coverage here. It shuts down a machine somebody
is paying for, so every way a run can end is asserted individually. The two that look alike are the
trap: a PAUSE exits 0 just like a completed run, and a user STOP arrives as a non-zero code that, on
its own, is indistinguishable from a crash.

Fizgig 8 replaced the desktop app's RunPod card with the web server. There is no separate
"should we stop?" predicate any more: the decision is the ORDER of the branches at the end of
server._run_stages (stopped returns, failed returns, paused skips, and only a run that finished with
nothing queued reaches pod_stop.maybe_start). So this drives _run_stages itself, with fake stage
processes, and records whether maybe_start was reached.

Retired with the desktop app, with no Fizgig 8 successor: the RunPod Preferences card's widgets and
version line, and the pre-run "low disk" confirmation dialog (the web app has none; the pod's storage
line below carries the warning instead).

Run: venv/Scripts/python.exe tests/test_runpod_card.py
"""
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# src only. The repo root also holds Fizgig 8's fizgig.py, which would shadow the fizgig package.
sys.path.insert(0, os.path.join(REPO, "src"))

from fizgig.web import server as S  # noqa: E402
from fizgig.web import pod_stop  # noqa: E402

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


# --- 1. detection ------------------------------------------------------------------------------
for k in ("FIZGIG_POD", "RUNPOD_POD_ID"):
    os.environ.pop(k, None)
ck("off a pod: not detected", pod_stop.on_pod() is False)
os.environ["FIZGIG_POD"] = "1"
ck("the pod image's own marker: detected", pod_stop.on_pod() is True)
os.environ.pop("FIZGIG_POD")
os.environ["RUNPOD_POD_ID"] = "abc123"
ck("RunPod's injected pod id: detected", pod_stop.on_pod() is True)
os.environ.pop("RUNPOD_POD_ID")

# --- 2. THE AUTO-STOP DECISION, driven through _run_stages --------------------------------------
reached = []
S.append_log = lambda text: None
S.history.end = lambda *a, **k: None
S.webqueue.on_run_ended = lambda *a, **k: None
_queued = []
S.webqueue.status = lambda: {"items": list(_queued)}
pod_stop.maybe_start = lambda prefs, log: reached.append("maybe_start")


class _Stage:
    def __init__(self, name="Training"):
        self.name, self.cmd = name, ["fake"]


class _FakeProc:
    """Stands in for a stage's subprocess: one line of output, then the given exit code. on_read runs while
    the output is read, which is when the server can be told to stop."""

    def __init__(self, code, on_read=None):
        self.code, self.on_read = code, on_read

    @property
    def stdout(self):
        if self.on_read:
            self.on_read()
        return iter(["working\n"])

    def wait(self):
        return self.code

    def poll(self):
        return self.code


def _user_stops():
    with S._lock:
        S._run["state"] = "stopped"


CASES = [
    # (label,                                       code, pause, checkpoint, queued, on_read,     should_stop)
    ("a completed run",                                0, False, True,  False, None,        True),
    ("a PAUSE with its checkpoint (also exits 0!)",    0, True,  True,  False, None,        False),
    ("a user Stop (taskkill on Windows)",              1, False, True,  False, _user_stops, False),
    ("a user Stop (SIGTERM on POSIX)",               -15, False, True,  False, _user_stops, False),
    ("a crash",                                        1, False, True,  False, None,        False),
    ("an OOM kill",                                   -9, False, True,  False, None,        False),
    ("a completed run with more runs queued",          0, False, True,  True,  None,        False),
]
for label, code, pause, checkpoint, queued, on_read, expect in CASES:
    reached.clear()
    _queued[:] = ["next run"] if queued else []
    out_dir = tempfile.mkdtemp(prefix="fizgig_autostop_")
    S._record_pause = lambda d, _c=checkpoint: _c
    S._popen = lambda cmd, _p=_FakeProc(code, on_read): _p
    with S._lock:
        S._run.update(state="running", pause=pause, family="krea2", values={"MAX_TRAIN_EPOCHS": 1})
    try:
        S._run_stages([_Stage()], out_dir)
    except Exception as e:
        ck(f"auto-stop: {label}", False, f"raised {type(e).__name__}: {e}")
        continue
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    ck(f"auto-stop: {label} -> {'stops' if expect else 'does NOT stop'}", bool(reached) == expect,
       f"reached={bool(reached)} state={S._run['state']}")

# A pause that left nothing to resume from is treated as a finished run by design (the server says so in its
# log), so it does reach the auto-stop. Pinned so that a change to this is deliberate, not accidental.
reached.clear()
_queued[:] = []
S._record_pause = lambda d: False
S._popen = lambda cmd: _FakeProc(0)
with S._lock:
    S._run.update(state="running", pause=True, family="krea2", values={"MAX_TRAIN_EPOCHS": 1})
_d = tempfile.mkdtemp(prefix="fizgig_autostop_")
S._run_stages([_Stage()], _d)
shutil.rmtree(_d, ignore_errors=True)
ck("a pause with no checkpoint is treated as finished -> reaches the auto-stop", bool(reached))

# --- 3. maybe_start's own gates (the real one, the network call stubbed out) -------------------
import importlib  # noqa: E402
pod_stop = importlib.reload(pod_stop)          # undo the stub above
_net = []
pod_stop.urllib.request.urlopen = lambda *a, **k: _net.append(a) or (_ for _ in ()).throw(AssertionError("network"))
pod_stop.COUNTDOWN = 3600                      # never actually reaches _stop during this test

# Upstream (b11296ab, Vast.ai): the pod image counts as a pod on any provider (FIZGIG_POD), but only a RunPod pod
# (RunPod's API, the renter's key) or a Vast.ai instance (its own CONTAINER_API_KEY) can stop itself. A pod on any
# other provider never starts a countdown that would end in nothing.
_POD_ENV = ("FIZGIG_POD", "RUNPOD_POD_ID", "CONTAINER_API_KEY", "CONTAINER_ID")
for k in _POD_ENV:
    os.environ.pop(k, None)
pod_stop.maybe_start({"runpod_stop_when_done": "1"}, lambda t: None)
ck("toggle on but NOT on a pod -> no countdown", pod_stop.status()["counting"] is False)
os.environ["FIZGIG_POD"] = "1"
pod_stop.maybe_start({"runpod_stop_when_done": "1"}, lambda t: None)
ck("the pod image on another provider (no RunPod id, not Vast) -> no countdown", pod_stop.status()["counting"] is False)
for label, extra in (("RunPod", {"RUNPOD_POD_ID": "abc123"}),
                     ("Vast.ai", {"CONTAINER_API_KEY": "k", "CONTAINER_ID": "42"})):
    os.environ.update(extra)
    pod_stop.maybe_start({"runpod_stop_when_done": "0"}, lambda t: None)
    ck(f"{label}: the toggle off -> no countdown", pod_stop.status()["counting"] is False)
    pod_stop.maybe_start({"runpod_stop_when_done": "1"}, lambda t: None)
    st = pod_stop.status()
    ck(f"{label}: the toggle on -> the countdown starts", st["counting"] is True and st["left"] > 0, st)
    ck(f"  {label}: Keep the pod running cancels it", pod_stop.cancel(lambda t: None) is True
       and pod_stop.status()["counting"] is False)
    ck(f"  {label}: and a second cancel is a no-op", pod_stop.cancel(lambda t: None) is False)
    for k in extra:
        os.environ.pop(k, None)

# --- 3b. the API key: never the pod-scoped one RunPod injects -----------------------------------
os.environ["RUNPOD_API_KEY"] = "pod-scoped"
os.environ.pop("RUNPOD_STOP_API_KEY", None)
ck("RunPod's injected RUNPOD_API_KEY is never used (it cannot stop pods)", pod_stop.api_key({}) == "")
os.environ["RUNPOD_STOP_API_KEY"] = "env-key"
ck("  RUNPOD_STOP_API_KEY is the fallback", pod_stop.api_key({}) == "env-key")
ck("  the key saved in Preferences wins", pod_stop.api_key({"runpod_api_key": "pref-key"}) == "pref-key")
for k in ("RUNPOD_API_KEY", "RUNPOD_STOP_API_KEY"):
    os.environ.pop(k, None)

_logged = []
pod_stop._stop({}, _logged.append)
ck("no API key -> it says so, and never calls RunPod",
   not _net and pod_stop.status()["note"] and "no RunPod API key" in pod_stop.status()["note"], _logged[-1:])
os.environ.pop("FIZGIG_POD", None)
ck("the toggle defaults to OFF", S.PREF_DEFAULTS.get("runpod_stop_when_done") == "0",
   S.PREF_DEFAULTS.get("runpod_stop_when_done"))

# --- 4. the storage line copes with what real hosts report -------------------------------------
# A network volume reports the host's whole backing pool, not the volume quota: a real 100 GB
# volume read as "431035 GB free of 1430281 GB". Printing that verbatim looks broken.
_real_usage = shutil.disk_usage
for total_gb, free_gb, expect, warn in ((1430281, 431035, "network volume", False),
                                        (25, 23, "container disk", True),
                                        (1000, 840, "GB free of", False),
                                        (1000, 30, "GB free of", True)):
    class _U:
        total = int(total_gb * 1024 ** 3)
        free = int(free_gb * 1024 ** 3)
        used = total - free
    S.shutil.disk_usage = lambda p, _u=_U: _u
    text, warned = S.pod_storage()
    ck(f"storage: {total_gb} GB total, {free_gb} free -> says '{expect}', warn={warn}",
       expect in text and warned is warn, (text[:70], warned))
    if total_gb > 10000:
        ck("  and does NOT print the absurd figure", "1430281" not in text)
S.shutil.disk_usage = lambda p: (_ for _ in ()).throw(OSError("no such device"))
ck("a failed disk probe reports 'unavailable' and does not warn", S.pod_storage() == ["Storage: unavailable", False])
S.shutil.disk_usage = _real_usage

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
