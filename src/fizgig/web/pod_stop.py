"""Stop a rented pod once a run has finished on its own (the desktop's Preferences > RunPod switch): a two-minute
countdown anyone on the page can cancel, then RunPod's API stops - never terminates - the pod, so the files under
/workspace are still there when it starts again. Runs in the server, so it works with no page open: the point is the
unattended finish."""

import json
import os
import shutil
import threading
import time
import urllib.request

COUNTDOWN = 120                # seconds
_lock = threading.Lock()
_s = {"deadline": 0.0, "gen": 0, "note": ""}


def on_pod() -> bool:
    """The Docker pod image (its own marker), or RunPod's id."""
    return os.environ.get("FIZGIG_POD", "0") not in ("0", "") or bool(os.environ.get("RUNPOD_POD_ID"))


def file_manager():
    """The pod's file manager (filebrowser on port 8080, started by the pod image's entrypoint) -> {url, login}, or
    None off a pod or without it. Its login is admin with the pod's password: "exact", "padded" (filebrowser
    refused it as too common, so the entrypoint added 0s up to 12 characters) or "none" (refused both ways). The
    entrypoint writes which; an older image wrote nothing and padded any password under 12 characters."""
    if not on_pod() or not shutil.which("filebrowser"):
        return None
    pid = pod_id()
    if not pid:
        return None
    try:
        with open("/tmp/fizgig_filebrowser_login", encoding="utf-8") as f:
            login = f.read().strip()
    except OSError:
        login = "padded" if len(os.environ.get("VNC_PASSWORD", "")) < 12 else "exact"
    return {"url": f"https://{pid}-8080.proxy.runpod.net/", "login": login if login in ("exact", "padded", "none") else "exact"}


def pod_id() -> str:
    for k in ("RUNPOD_POD_ID", "HOSTNAME"):
        v = (os.environ.get(k) or "").strip()
        if v:
            return v
    return ""


def api_key(prefs) -> str:
    """The key saved in Preferences, else RUNPOD_STOP_API_KEY. Never RunPod's injected RUNPOD_API_KEY: that one is
    pod-scoped and can't stop pods."""
    return str(prefs.get("runpod_api_key") or "").strip() or (os.environ.get("RUNPOD_STOP_API_KEY") or "").strip()


def status():
    with _lock:
        left = max(0, int(round(_s["deadline"] - time.time()))) if _s["deadline"] else 0
        return {"counting": bool(_s["deadline"]), "left": left, "note": _s["note"]}


def cancel(log):
    with _lock:
        was = bool(_s["deadline"])
        _s.update(deadline=0.0, gen=_s["gen"] + 1, note="")
    if was:
        log("[pod] auto-stop cancelled - pod left running.\n")
    return was


def maybe_start(prefs, log):
    """After a run that finished on its own with nothing queued: the countdown, when the switch is on."""
    if not on_pod() or str(prefs.get("runpod_stop_when_done") or "0").strip() != "1":
        return
    with _lock:
        _s["gen"] += 1
        gen = _s["gen"]
        _s.update(deadline=time.time() + COUNTDOWN, note="")
    log(f"[pod] training finished - this pod stops in {COUNTDOWN} s to stop it billing (Keep the pod running cancels).\n")

    def wait():
        while True:
            with _lock:
                if _s["gen"] != gen:
                    return
                left = _s["deadline"] - time.time()
            if left <= 0:
                break
            time.sleep(min(1.0, left))
        with _lock:
            if _s["gen"] != gen:
                return
            _s["deadline"] = 0.0
        _stop(prefs, log)
    threading.Thread(target=wait, daemon=True).start()


def _stop(prefs, log):
    key, pid = api_key(prefs), pod_id()
    if not key:
        msg = ("Auto-stop is on but no RunPod API key is set, so the pod is still running. Add one in Preferences > "
               "RunPod (RunPod > Settings > API Keys).")
        log("[pod] " + msg + "\n")
        with _lock:
            _s["note"] = msg
        return
    log(f"[pod] stopping pod {pid or '(unknown id)'}…\n")
    try:
        body = json.dumps({"query": "mutation($id: String!) { podStop(input: {podId: $id}) { id desiredStatus } }",
                           "variables": {"id": pid}}).encode()
        req = urllib.request.Request(f"https://api.runpod.io/graphql?api_key={key}", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            payload = json.loads(resp.read().decode())
        if payload.get("errors"):
            msg = f"RunPod refused the stop: {payload['errors'][0].get('message', 'unknown error')}. Stop it from the " \
                  "dashboard to stop billing."
        else:
            msg = "Stop requested - this pod is shutting down."
    except Exception as e:
        msg = f"Auto-stop failed ({type(e).__name__}: {e}). Stop it from the dashboard to stop billing."
    log("[pod] " + msg + "\n")
    with _lock:
        _s["note"] = msg
