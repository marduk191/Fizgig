"""Fizgig launcher — double-click to run without a console window.

The .pyw extension tells Windows to use pythonw.exe, which hides the
console.  Console output is captured by the GUI's built-in log — click
the status indicator in the top-right corner to view it.

A hidden console hides FAILURES too: if the GUI dies on import, pythonw
discards the traceback and the app simply never appears — no window, no
error, no clue.  (A user hit exactly this with a Python installed without
Tkinter.)  So every startup failure is reported through two channels that
cannot themselves depend on what broke: a native Windows message box via
ctypes — deliberately not a Tkinter dialog, since missing Tkinter is the
classic cause — and launch_error.log beside this file.
"""
import os
import sys
import runpy
import subprocess
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHONW = os.path.join(HERE, "venv", "Scripts", "pythonw.exe")
LOG = os.path.join(HERE, "launch_error.log")


def _report(title, message, detail=""):
    """The failure path. Must not use Tkinter, and must not raise."""
    logged = False
    try:
        with open(LOG, "w", encoding="utf-8") as f:
            f.write(f"{title}\n\n{message}\n\n{detail}\n"
                    f"\npython: {sys.executable}\nversion: {sys.version}\n")
        logged = True
    except Exception:
        pass
    body = message + (f"\n\nDetails saved to:\n{LOG}" if logged else "")
    try:
        import ctypes
        # 0x10 = error icon, 0x10000 = bring the box to the foreground
        ctypes.windll.user32.MessageBoxW(0, body, title, 0x10010)
    except Exception:
        # Not Windows, or no usable GUI session. Guarded because stderr is
        # None under pythonw, and the reporter must never raise.
        if getattr(sys, "stderr", None):
            sys.stderr.write(f"{title}\n{body}\n{detail}\n")
    sys.exit(1)


# If the venv exists and we're not already running from it, re-launch
if os.path.exists(VENV_PYTHONW) and os.path.normcase(sys.executable) != os.path.normcase(VENV_PYTHONW):
    try:
        subprocess.Popen([VENV_PYTHONW, os.path.abspath(__file__)])
    except Exception as exc:
        _report("Fizgig could not start",
                "The bundled Python failed to launch:\n"
                f"{VENV_PYTHONW}\n\n"
                "Re-run install_fizgig.bat to repair the environment.",
                f"{type(exc).__name__}: {exc}")
    sys.exit(0)

# Checked HERE, before the GUI is touched, so the report names the actual
# problem instead of surfacing as a traceback from line 1 of a 20,000-line
# file.  Tkinter ships with Python itself — it is NOT pip-installable (the
# `tk` package on PyPI is unrelated), so requirements.txt cannot supply it;
# the fix is a Python installed with it.
try:
    import tkinter  # noqa: F401
except Exception as exc:
    _report(
        "Fizgig — Python is missing Tkinter",
        "This Python was installed without Tkinter, which Fizgig needs for "
        "its interface.\n\n"
        "Reinstall Python from python.org and tick “tcl/tk and IDLE” "
        "during setup, then run install_fizgig.bat again.",
        f"{type(exc).__name__}: {exc}")

try:
    if os.path.isfile(LOG):
        os.remove(LOG)          # stale report from an earlier failed start
except Exception:
    pass

sys.path.insert(0, HERE)
try:
    runpy.run_path(os.path.join(HERE, "lora_trainer_gui.py"), run_name="__main__")
except (SystemExit, KeyboardInterrupt):
    raise                       # deliberate exits are not errors
except BaseException as exc:
    _report("Fizgig could not start",
            f"Fizgig hit an error while starting:\n\n{type(exc).__name__}: {exc}\n\n"
            "If this keeps happening, please open a GitHub issue and attach "
            "the log below.",
            traceback.format_exc())
