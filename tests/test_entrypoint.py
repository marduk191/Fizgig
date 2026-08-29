"""The pod entrypoint's boot path — specifically the branches that only run in production.

Why this file exists: the generated-password branch shipped broken and crash-looped every pod
deployed from the public template. `tr -dc ... </dev/urandom | head -c 12` returns 141 under
`set -euo pipefail` — head exits at 12 bytes, tr takes SIGPIPE — so the script died before its
first log line. It survived every manual test because those all had VNC_PASSWORD set, and it
became the DEFAULT path the moment the template shipped without one.

The lesson generalises: a boot path that only runs when a variable is ABSENT is exactly the path
no one exercises by hand. So this runs it, rather than reading it.

Needs bash (Git Bash on Windows is fine). Run:
  venv/Scripts/python.exe tests/test_entrypoint.py
"""
import os
import re
import shutil
import subprocess
import sys

REPO = r"W:/Peter/Documents/Development/Fizgig"
ENTRY = os.path.join(REPO, "docker", "entrypoint.sh")

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


BASH = shutil.which("bash") or r"C:\Program Files\Git\bin\bash.exe"
if not os.path.isfile(BASH):
    print("SKIP  bash not found — cannot exercise the entrypoint")
    sys.exit(0)


def run(script, env=None):
    e = dict(os.environ)
    e.pop("VNC_PASSWORD", None)
    e.update(env or {})
    return subprocess.run([BASH, "-c", script], capture_output=True, text=True, env=e)


src = open(ENTRY, encoding="utf-8").read()

# --- the script is at least syntactically sound -------------------------------------------------
r = subprocess.run([BASH, "-n", ENTRY], capture_output=True, text=True)
ck("entrypoint.sh parses", r.returncode == 0, r.stderr.strip()[:200])

ck("still runs under set -euo pipefail", "set -euo pipefail" in src)

# --- the password branch, actually executed ------------------------------------------------------
# Lifted verbatim from the file so this cannot drift into testing a copy that no longer matches.
m = re.search(r'^if \[ -z "\$\{VNC_PASSWORD:-\}" \]; then\n(.*?)^fi$', src, re.S | re.M)
ck("found the VNC_PASSWORD branch", m is not None)
if m:
    block = 'set -euo pipefail\nlog() { :; }\n' + m.group(0) + '\necho "${#VNC_PASSWORD}"'
    # Repeated because the old bug was deterministic but a length bug would not be: filtering
    # random bytes to alnum yields a variable number of usable characters.
    lens, bad = set(), 0
    for _ in range(25):
        r = run(block)
        if r.returncode != 0:
            bad += 1
        else:
            lens.add(r.stdout.strip())
    ck("generating a password never exits non-zero", bad == 0, f"{bad}/25 failed")
    ck("  and is always 12 characters", lens == {"12"}, lens)

    r = run(block, {"VNC_PASSWORD": "chosen-by-user"})
    ck("a password set in the template is left alone",
       r.returncode == 0 and r.stdout.strip() == "14", (r.returncode, r.stdout.strip()))

# --- the whole class of bug, not just the one line -----------------------------------------------
# Under pipefail, any consumer that exits before its producer finishes kills the script. Producers
# reading an endless source (/dev/urandom, yes, tail -f) are the fatal ones; a finite echo is not.
risky = []
for i, line in enumerate(src.splitlines(), 1):
    if line.lstrip().startswith("#"):
        continue
    if re.search(r"\|\s*(head|grep -m|sed -n '[^']*q')", line) and "/dev/urandom" in line:
        risky.append((i, line.strip()))
ck("no pipeline truncates an endless producer", not risky, risky)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
