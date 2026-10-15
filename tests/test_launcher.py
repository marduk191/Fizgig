"""The launchers' every path, actually executed — not read.

Fizgig and Gizmo both start on pythonw (Launch Fizgig.bat, Launch Fizgig AMD.bat, Launch Gizmo.bat), where a
startup failure has no console: the app just never appears (a user with a Tkinter-less Python reported exactly that
against Fizgig 7). Fizgig 7's launch.pyw reported through a native message box + launch_error.log. Fizgig 8 deleted
launch.pyw with the classic app and starts fizgig.py directly, so the fork re-placed that reporter in fizgig.py
([fork] block there); Gizmo's launcher, src/gizmo/gizmo.pyw, still carries upstream's own. This suite runs both in
sandboxed copies with stub apps, so neither real app opens and the real tree (its user_data/) is never touched.

Two testing tricks worth knowing:
 * A real MessageBoxW BLOCKS until clicked, so failure paths run with a recorder ctypes shadowing the stdlib via
   PYTHONPATH. Any run that pops a real dialog therefore hangs and fails by timeout — which doubles as proof the
   happy path never shows one.
 * Missing Tkinter is simulated the same way: a poisoned tkinter.py that raises on import, shadowing the real one.
   (Only Gizmo needs Tkinter now; Fizgig 8 is a web UI.)

Run: venv/Scripts/python.exe tests/test_launcher.py
"""
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, "venv", "Scripts", "python.exe")
PYW = os.path.join(REPO, "venv", "Scripts", "pythonw.exe")

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


BASE = tempfile.mkdtemp(prefix="fizgig_launch_sim_")

# --- the stand-ins ------------------------------------------------------------------------------
FAKECT = os.path.join(BASE, "fakectypes")
os.makedirs(FAKECT)
MSGBOX = os.path.join(FAKECT, "msgbox.json")
with open(os.path.join(FAKECT, "ctypes.py"), "w", encoding="utf-8") as f:
    f.write(textwrap.dedent(f"""
        import json
        class _User32:
            def MessageBoxW(self, hwnd, text, title, flags):
                with open({MSGBOX!r}, "w", encoding="utf-8") as fh:
                    json.dump({{"title": title, "text": text, "flags": flags}}, fh)
                return 1
        class _WinDLL:
            user32 = _User32()
        windll = _WinDLL()
        """))

POISON = os.path.join(BASE, "poison_tk")
os.makedirs(POISON)
with open(os.path.join(POISON, "tkinter.py"), "w", encoding="utf-8") as f:
    f.write("raise ImportError('simulated: Python built without Tkinter')\n")

BROKENCT = os.path.join(BASE, "broken_ct")
os.makedirs(BROKENCT)
with open(os.path.join(BROKENCT, "ctypes.py"), "w", encoding="utf-8") as f:
    f.write("raise ImportError('simulated: no ctypes on this platform')\n")

# Linux is the other shape: `import ctypes` SUCCEEDS but there is no windll attribute at all.
LINUXCT = os.path.join(BASE, "linux_ct")
os.makedirs(LINUXCT)
with open(os.path.join(LINUXCT, "ctypes.py"), "w", encoding="utf-8") as f:
    f.write("# imports fine; ctypes.windll does not exist off Windows\n")

# The shadowing trick only works if the interpreter has not already imported the real ctypes during startup.
for exe in (PY, PYW):
    r = subprocess.run([exe, "-c", "import sys; open(sys.argv[1], 'w').write(str('ctypes' in sys.modules))",
                        os.path.join(BASE, "pre.txt")], capture_output=True, text=True)
    pre = open(os.path.join(BASE, "pre.txt")).read() if os.path.isfile(os.path.join(BASE, "pre.txt")) else "?"
    ck(f"{os.path.basename(exe)} does not preload ctypes (shadowing is sound)", pre == "False", pre)


def box():
    if os.path.isfile(MSGBOX):
        with open(MSGBOX, encoding="utf-8") as f:
            return json.load(f)
    return None


def text_of(path):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return ""


def run(cmd, *, cwd, extra_path=(), reset=(), pre_log=None, log=None, timeout=40):
    for p in (MSGBOX, *reset):
        if os.path.isfile(p):
            os.remove(p)
    if log is not None:
        if os.path.isfile(log):
            os.remove(log)
        if pre_log is not None:
            os.makedirs(os.path.dirname(log), exist_ok=True)
            with open(log, "w", encoding="utf-8") as f:
                f.write(pre_log)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTORCH_CUDA_ALLOC_CONF")}
    env["PYTHONPATH"] = os.pathsep.join(extra_path)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None  # a real dialog appeared, or the launcher hung — both are failures


# ================================================================================================
# A. fizgig.py — the app window's launcher (Fizgig 8), with the fork's re-placed reporter
# ================================================================================================
FZ = os.path.join(BASE, "fizgig_app")
os.makedirs(os.path.join(FZ, "src", "fizgig", "web"))
shutil.copy(os.path.join(REPO, "fizgig.py"), os.path.join(FZ, "fizgig.py"))
for p in ("src/fizgig/__init__.py", "src/fizgig/web/__init__.py"):
    open(os.path.join(FZ, p), "w").close()
FZ_LOG = os.path.join(FZ, "user_data", "launch_error.log")
FZ_MARK = os.path.join(FZ, "started.txt")
FZ_WEB = os.path.join(FZ, "web_started.txt")
with open(os.path.join(FZ, "src", "fizgig", "web", "server.py"), "w", encoding="utf-8") as f:
    f.write(f"def main():\n    open({FZ_WEB!r}, 'w').write('web')\n")


def run_fizgig(app_src, *, migrate_src="pass", exe=PYW, args=(), **kw):
    """fizgig.py in the sandbox, with app_window.main's body = app_src and user_data.migrate's = migrate_src."""
    with open(os.path.join(FZ, "src", "fizgig", "web", "app_window.py"), "w", encoding="utf-8") as f:
        f.write("import sys\ndef main():\n" + textwrap.indent(textwrap.dedent(app_src), "    ") + "\n")
    with open(os.path.join(FZ, "src", "fizgig", "user_data.py"), "w", encoding="utf-8") as f:
        f.write("def migrate(log=print):\n" + textwrap.indent(textwrap.dedent(migrate_src), "    ") + "\n")
    return run([exe, os.path.join(FZ, "fizgig.py"), *args], cwd=FZ, reset=(FZ_MARK, FZ_WEB), log=FZ_LOG, **kw)


FZ_HAPPY = f"open({FZ_MARK!r}, 'w').write('started')"

# --- A1. the happy path is untouched ------------------------------------------------------------
r = run_fizgig(FZ_HAPPY, extra_path=[FAKECT])
ck("fizgig.py happy path under pythonw: exit 0", r is not None and r.returncode == 0,
   None if r is None else (r.returncode, r.stderr[-200:]))
ck("  the app window ran", os.path.isfile(FZ_MARK))
ck("  no error log, no message box", not os.path.isfile(FZ_LOG) and box() is None)

r = run_fizgig(FZ_HAPPY, extra_path=[FAKECT], pre_log="old failure")
ck("a stale launch_error.log is removed on a clean start",
   r is not None and r.returncode == 0 and not os.path.isfile(FZ_LOG))

r = run_fizgig("open(r'%s', 'w').write('x')" % FZ_MARK, extra_path=[FAKECT], args=("--web",))
ck("--web still routes to the web server, not the window",
   r is not None and r.returncode == 0 and os.path.isfile(FZ_WEB) and not os.path.isfile(FZ_MARK))

r = run_fizgig(FZ_HAPPY, extra_path=[FAKECT], exe=PY)
ck("happy path under python.exe", r is not None and r.returncode == 0 and os.path.isfile(FZ_MARK))

# --- A2. the window's own "did not start" (a SystemExit carrying a message) ----------------------
_MSG = "Fizgig's web UI did not start. Run `python fizgig.py --web` to see why."
r = run_fizgig(f"sys.exit({_MSG!r})", extra_path=[FAKECT])
b = box()
ck("web UI did not start, under pythonw: exit 1", r is not None and r.returncode == 1,
   None if r is None else r.returncode)
ck("  message box appeared, titled for Fizgig", b is not None and b["title"] == "Fizgig could not start",
   b and b["title"])
ck("  box carries app_window's message", b is not None and _MSG in b["text"])
ck("  box points at the log", b is not None and "launch_error.log" in b["text"])
ck("  log holds the message", _MSG in text_of(FZ_LOG))

# --- A3. any other startup crash: in the window, or in the migration before it -------------------
r = run_fizgig("raise RuntimeError('boom at startup')", extra_path=[FAKECT])
b = box()
ck("app window crash under pythonw: exit 1 + box with the exception",
   r is not None and r.returncode == 1 and b is not None and "RuntimeError: boom at startup" in b["text"])
ck("  log holds the full traceback", "Traceback" in text_of(FZ_LOG) and "boom at startup" in text_of(FZ_LOG))

r = run_fizgig(FZ_HAPPY, migrate_src="raise PermissionError('user_data is read-only')", extra_path=[FAKECT])
b = box()
ck("a crash in the user_data migration (before main) is reported too",
   r is not None and r.returncode == 1 and b is not None and "user_data is read-only" in b["text"]
   and not os.path.isfile(FZ_MARK), None if r is None else (r.returncode, b))

r = run_fizgig("import nonexistent_module_xyz", extra_path=[FAKECT])
ck("an import error inside the window (a missing dependency) is reported",
   r is not None and r.returncode == 1 and box() is not None and "ModuleNotFoundError" in box()["text"])

# --- A4. deliberate exits are not errors ---------------------------------------------------------
r = run_fizgig("sys.exit(7)", extra_path=[FAKECT])
ck("numeric SystemExit passes through untouched (code 7, no box, no log)",
   r is not None and r.returncode == 7 and box() is None and not os.path.isfile(FZ_LOG),
   None if r is None else r.returncode)

r = run_fizgig("raise KeyboardInterrupt", extra_path=[FAKECT])
ck("KeyboardInterrupt is not dressed up as a crash (no box, no log)",
   r is not None and r.returncode != 0 and box() is None and not os.path.isfile(FZ_LOG))

# --- A5. python.exe has a console: no box there --------------------------------------------------
r = run_fizgig("raise RuntimeError('boom in a console')", extra_path=[FAKECT], exe=PY)
ck("under python.exe a crash shows its traceback on the console, no box, no log",
   r is not None and r.returncode == 1 and "boom in a console" in r.stderr and box() is None
   and not os.path.isfile(FZ_LOG), None if r is None else (r.returncode, r.stderr[-150:]))

# --- A6. the reporter itself must never die ------------------------------------------------------
if os.path.isfile(FZ_LOG):
    os.remove(FZ_LOG)  # left by a failing check above; the sections below must still run
os.makedirs(FZ_LOG)  # a directory at the log path makes open(..., 'w') fail
try:
    r = run_fizgig("raise RuntimeError('boom')", extra_path=[FAKECT])
    b = box()
    ck("unwritable log: box still shown, clean exit 1", r is not None and r.returncode == 1 and b is not None)
    ck("  box does not promise a log that does not exist", b is not None and "Details saved" not in b["text"])
finally:
    os.rmdir(FZ_LOG)

for label, shadow in (("no ctypes at all", BROKENCT), ("Linux shape (ctypes imports, no windll)", LINUXCT)):
    r = run_fizgig("raise RuntimeError('boom')", extra_path=[shadow])
    ck(f"{label}: falls back to stderr + log, exit 1",
       r is not None and r.returncode == 1 and "could not start" in r.stderr and "boom" in text_of(FZ_LOG),
       None if r is None else r.stderr[-150:])

# ================================================================================================
# B. src/gizmo/gizmo.pyw — Gizmo's launcher (upstream's own reporter)
# ================================================================================================
GZ = os.path.join(BASE, "gizmo_app")
GZ_SRC = os.path.join(GZ, "src", "gizmo")
os.makedirs(GZ_SRC)
shutil.copy(os.path.join(REPO, "src", "gizmo", "gizmo.pyw"), os.path.join(GZ_SRC, "gizmo.pyw"))
GZ_LAUNCH = os.path.join(GZ_SRC, "gizmo.pyw")
GZ_LOG = os.path.join(GZ, "user_data", "gizmo_error.log")
GZ_MARK = os.path.join(GZ, "started.txt")


def run_gizmo(app_src, *, exe=PY, **kw):
    with open(os.path.join(GZ_SRC, "gizmo.py"), "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(app_src))
    return run([exe, GZ_LAUNCH], cwd=GZ, reset=(GZ_MARK,), log=GZ_LOG, **kw)


GZ_HAPPY = f"with open({GZ_MARK!r}, 'w') as f: f.write('started')\n"

# --- B1. the happy path ---------------------------------------------------------------------------
r = run_gizmo(GZ_HAPPY, extra_path=[FAKECT])
ck("gizmo happy path: exit 0, app ran, no log, no box",
   r is not None and r.returncode == 0 and os.path.isfile(GZ_MARK) and not os.path.isfile(GZ_LOG)
   and box() is None, None if r is None else (r.returncode, r.stderr[-200:]))
r = run_gizmo(GZ_HAPPY, extra_path=[FAKECT], pre_log="old failure")
ck("  a stale gizmo_error.log is removed on a clean start",
   r is not None and r.returncode == 0 and not os.path.isfile(GZ_LOG))
r = run_gizmo(GZ_HAPPY, extra_path=[FAKECT], exe=PYW)
ck("  happy path under pythonw (the real deployment)",
   r is not None and r.returncode == 0 and os.path.isfile(GZ_MARK))

# --- B2. missing Tkinter: the reported case -------------------------------------------------------
for exe in (PY, PYW):
    r = run_gizmo(GZ_HAPPY, extra_path=[FAKECT, POISON], exe=exe)
    b = box()
    n = os.path.basename(exe)
    ck(f"missing tkinter ({n}): exit 1, app never started",
       r is not None and r.returncode == 1 and not os.path.isfile(GZ_MARK), None if r is None else r.returncode)
    ck(f"  ({n}) box names Tkinter and how to fix it",
       b is not None and "Tkinter" in b["title"] and "tcl/tk and IDLE" in b["text"], b and b["title"])
    ck(f"  ({n}) box points at the log, log holds the cause",
       b is not None and "gizmo_error.log" in b["text"] and "simulated" in text_of(GZ_LOG))

# --- B3. any other startup crash; deliberate exits -----------------------------------------------
r = run_gizmo("raise RuntimeError('boom at startup')", extra_path=[FAKECT])
b = box()
ck("gizmo crash: exit 1, box carries the exception, log the traceback",
   r is not None and r.returncode == 1 and b is not None and "RuntimeError: boom at startup" in b["text"]
   and "Traceback" in text_of(GZ_LOG))
r = run_gizmo("import sys; sys.exit(7)", extra_path=[FAKECT])
ck("  SystemExit passes through untouched (code 7, no box, no log)",
   r is not None and r.returncode == 7 and box() is None and not os.path.isfile(GZ_LOG))
r = run_gizmo("raise KeyboardInterrupt", extra_path=[FAKECT])
ck("  KeyboardInterrupt is not dressed up as a crash",
   r is not None and r.returncode != 0 and box() is None and not os.path.isfile(GZ_LOG))

# --- B4. the reporter itself must never die -------------------------------------------------------
if os.path.isfile(GZ_LOG):
    os.remove(GZ_LOG)  # left by a failing check above; the sections below must still run
os.makedirs(GZ_LOG)
try:
    r = run_gizmo("raise RuntimeError('boom')", extra_path=[FAKECT])
    b = box()
    ck("gizmo, unwritable log: box still shown, exit 1, no promise of a log",
       r is not None and r.returncode == 1 and b is not None and "Details saved" not in b["text"])
finally:
    os.rmdir(GZ_LOG)
for label, shadow in (("no ctypes at all", BROKENCT), ("Linux shape", LINUXCT)):
    r = run_gizmo("raise RuntimeError('boom')", extra_path=[shadow])
    ck(f"gizmo, {label}: stderr + log, exit 1",
       r is not None and r.returncode == 1 and "could not start" in r.stderr and "boom" in text_of(GZ_LOG),
       None if r is None else r.stderr[-150:])

# --- B5. the venv re-launch hop -------------------------------------------------------------------
# gizmo.pyw re-launches itself under <Fizgig>/venv/Scripts/pythonw.exe when it isn't already on it. The child is
# a fresh interpreter, but PYTHONPATH (the fake ctypes) is inherited, so if it ever fails its box lands in MSGBOX
# instead of on the desktop of whoever runs the suite.
r = subprocess.run([PY, "-m", "venv", "--without-pip", os.path.join(GZ, "venv")],
                   capture_output=True, text=True, timeout=120)
if r.returncode == 0:
    with open(os.path.join(GZ_SRC, "gizmo.py"), "w", encoding="utf-8") as f:
        f.write(f"import sys\nwith open({GZ_MARK!r}, 'w') as fh:\n    fh.write(sys.executable)\n")
    r = run([PY, GZ_LAUNCH], cwd=GZ, extra_path=[FAKECT], reset=(GZ_MARK,), log=GZ_LOG)
    ck("gizmo: outer python hands off to the venv and exits 0", r is not None and r.returncode == 0,
       None if r is None else r.stderr[-200:])
    end = time.time() + 25
    while time.time() < end and not os.path.isfile(GZ_MARK):
        time.sleep(0.3)
    time.sleep(1)
    ck("  the venv child actually started Gizmo, on the venv's pythonw",
       os.path.normcase(text_of(GZ_MARK)) == os.path.normcase(os.path.join(GZ, "venv", "Scripts", "pythonw.exe")),
       text_of(GZ_MARK))
    ck("  and the child finished without raising an error report",
       box() is None and not os.path.isfile(GZ_LOG), (box() or {}).get("text", "")[:200])
else:
    ck("venv creation for the re-launch test", False, r.stderr[-200:])

# ================================================================================================
# C. static guards
# ================================================================================================
for rel in ("fizgig.py", os.path.join("src", "gizmo", "gizmo.pyw")):
    try:
        py_compile.compile(os.path.join(REPO, rel), doraise=True)
        ck(f"{rel} compiles", True)
    except py_compile.PyCompileError as e:
        ck(f"{rel} compiles", False, e)
    src = open(os.path.join(REPO, rel), encoding="utf-8").read()
    ck(f"  {rel}: stderr fallback is guarded (stderr is None under pythonw)", 'getattr(sys, "stderr", None)' in src)
    ck(f"  {rel}: the reporter never imports tkinter",
       "tkinter" not in src.split("def _report")[1].split("\n\n\n")[0])

# The reporter in fizgig.py fires only under pythonw, so the launchers must still start it there.
for bat in ("Launch Fizgig.bat", "Launch Fizgig AMD.bat"):
    src = open(os.path.join(REPO, bat), encoding="utf-8", errors="replace").read()
    ck(f"{bat} starts fizgig.py on pythonw (where the reporter is live)",
       "pythonw.exe" in src and "fizgig.py" in src)

shutil.rmtree(BASE, ignore_errors=True)
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
