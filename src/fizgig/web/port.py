"""The web UI's port (Preferences `web_port`), settled by whichever of the desktop app and the web UI starts first.

8081 at first, the port the pod image exposes. When the saved port is taken by another program, the nearest free one
above it is used and saved, so every later launch gets that same port. A port answered by Fizgig's own web UI counts
as Fizgig's: it is already serving there. A pod always uses 8081, the one port its proxy forwards.
"""
import os
import socket
import urllib.request

PORT_DEFAULT = 8081
PORT_WALK = 100                         # how far above a taken port to look for a free one


def on_pod() -> bool:
    return bool(os.environ.get("RUNPOD_POD_ID"))


def saved_port(prefs) -> int:
    try:
        return int(prefs.get("web_port") or PORT_DEFAULT)
    except (TypeError, ValueError):
        return PORT_DEFAULT


def listening(port) -> bool:
    """Something already answers on this port locally (on Windows a bind on 127.0.0.1 can succeed beside another
    program's 0.0.0.0 listener and shadow it, so a free bind alone does not prove the port is free)."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def is_fizgig(port) -> bool:
    """Fizgig's web UI answers on this port (its page carries the fizgig-token meta tag)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as r:
            return b'name="fizgig-token"' in r.read(4096)
    except Exception:
        return False


def lan_ip() -> str:
    """This machine's address on the local network, or "" (a UDP connect sends nothing; it only picks the route)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        return "" if ip.startswith("127.") else ip
    except OSError:
        return ""


def free_from(start):
    """The first port from `start` up that nothing answers on, or None."""
    for port in range(start, min(65536, start + PORT_WALK)):
        if not listening(port):
            return port
    return None


def settle(prefs) -> int | None:
    """For the desktop app at start-up: the port the web UI will get, checked now so Preferences shows it before
    the web UI is ever opened. Returns a new port to save, or None when the saved one stands (free, Fizgig's own,
    or a pod)."""
    if on_pod():
        return None
    want = saved_port(prefs)
    if not listening(want) or is_fizgig(want):
        return None if str(prefs.get("web_port") or "") == str(want) else want
    return free_from(want + 1)
