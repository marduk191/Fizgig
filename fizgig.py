"""Fizgig: the web UI in its own app window (the default, --app), or served for a web browser (--web; --open
opens the browser on it). Options for --web: python fizgig.py --web --help."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

# Set before anything imports torch, and inherited by every caching and training run started from here:
# - CUDA's allocator grows and shrinks segments instead of fragmenting under the workbench's load / preview / swap
#   churn (not on Windows, whose allocator rejects the option). An existing value wins; FIZGIG_NO_EXPANDABLE=1 opts
#   out.
# - Intel OpenMP's thread pool sleeps instead of spinning 200 ms after every parallel region: training's small
#   per-step CPU work kept every core at 100% doing nothing (#18).
if (sys.platform != "win32" and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        and os.environ.get("FIZGIG_NO_EXPANDABLE") != "1"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# [fork] Windows gets CUDA's stream-ordered allocator instead, since it rejects expandable_segments outright
# and would otherwise run every caching, training and workbench subprocess on the default allocator,
# which fragments under exactly that churn. Measured on a fragmentation repro at ~5% free headroom:
# num_alloc_retries 9 -> 0, worst-case step 84 ms -> 3.8 ms. Every subprocess's env is built as
# dict(os.environ, ...) and passed through gpu_env, so they all inherit this. Neither this process nor the
# server imports torch, but the backend must be fixed before torch loads, so the guard stays: on torch 2.10,
# setting it after import kills the first CUDA allocation on an internal assert. An existing value and
# FIZGIG_NO_EXPANDABLE=1 both still win.
if ("torch" not in sys.modules and sys.platform == "win32" and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        and os.environ.get("FIZGIG_NO_EXPANDABLE") != "1"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

# [fork] "Launch Fizgig.bat" starts this on pythonw.exe, which has no console, so a failed start went nowhere:
# no window, no message, nothing to report. Fizgig 7's launch.pyw caught exactly that (e7e8fe02) and Fizgig 8
# deleted it with the classic app; Gizmo's launcher (src/gizmo/gizmo.pyw) still does it, and this is its shape.
# Only under pythonw: python.exe has a console, which shows the traceback as ever.
LAUNCH_LOG = os.path.join(HERE, "user_data", "launch_error.log")
_WINDOWLESS = os.path.basename(sys.executable).lower() == "pythonw.exe"


def _report(message, detail=""):
    """The failure path under pythonw: launch_error.log and a native message box. Must not raise."""
    logged = False
    try:
        os.makedirs(os.path.dirname(LAUNCH_LOG), exist_ok=True)
        with open(LAUNCH_LOG, "w", encoding="utf-8") as f:
            f.write(f"Fizgig could not start\n\n{message}\n\n{detail}\n"
                    f"\npython: {sys.executable}\nversion: {sys.version}\n")
        logged = True
    except Exception:
        pass
    body = message + (f"\n\nDetails saved to:\n{LAUNCH_LOG}" if logged else "")
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, body, "Fizgig could not start", 0x10010)   # error icon, foreground
    except Exception:
        if getattr(sys, "stderr", None):        # None under pythonw
            sys.stderr.write(f"Fizgig could not start\n{body}\n{detail}\n")


def _guarded(fn):
    """fn(), reporting a failed start when windowless. A SystemExit carrying a message (app_window's "the web UI
    did not start") is one too; a numeric exit and Ctrl+C are deliberate and pass through."""
    if not _WINDOWLESS:
        return fn()
    import traceback
    try:
        return fn()
    except KeyboardInterrupt:
        raise
    except SystemExit as e:
        if isinstance(e.code, str):
            _report(e.code)
            raise SystemExit(1)
        raise
    except BaseException as e:
        _report(f"Fizgig hit an error while starting:\n\n{type(e).__name__}: {e}\n\n"
                "If this keeps happening, please open a GitHub issue and attach the log below.",
                traceback.format_exc())
        raise SystemExit(1)


def _migrate():
    from fizgig import user_data
    user_data.migrate()                   # Fizgig 7's loose files into user_data (once; nothing to do after)
    if _WINDOWLESS:
        try:
            os.remove(LAUNCH_LOG)         # a stale report from an earlier failed start
        except OSError:
            pass


_guarded(_migrate)


def main():
    if "--web" in sys.argv[1:]:
        from fizgig.web.server import main as web
        return web()
    from fizgig.web.app_window import main as app
    return app()


if __name__ == "__main__":
    sys.exit(_guarded(main))
