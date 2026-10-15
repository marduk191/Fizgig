"""LoRA Choose… pickers start in a folder that actually exists — headless, no GPU, no browser.

On RunPod the input_lora_dir pref was never seeded, so the desktop's Repair Studio / Explorer / Extract / Profiler /
Context LoRA pickers opened in the process cwd — the git clone, which holds no .safetensors and made the picker look
broken. The desktop fixed it with a fallback order: input_lora_dir when it is a real folder, else the Output
Directory, else "".

Fizgig 8's web UI keeps that order in app.js (loraStart: Preferences' LoRA folder, else the output folder) and asks
the server to list it (/api/browse -> Handler._browse). But loraStart picked the pref whenever it was SET, and
_browse, finding it gone, fell back to BASE_DIR: a deleted or renamed Preferences folder put every LoRA picker back in
the git clone, the original bug. The fork sends the order as candidates and _browse opens the first that exists.
Pinned here:
  1. _browse: candidates in order, first existing folder wins; a stale first one falls through; a file path opens its
     folder; a relative one is inside Fizgig; nothing valid lands on BASE_DIR without crashing; a single path and
     the drives listing still work;
  2. app.js, executed in node: loraStart / refStart / the form's lora start give that order, and browse() sends every
     candidate, in order, as its own path=;
  3. every LoRA picker seeds from loraStart().

Run: venv/Scripts/python.exe tests/test_lora_picker_initialdir.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# src only. The repo root holds Fizgig 8's fizgig.py, which would shadow the fizgig package.
sys.path.insert(0, os.path.join(REPO, "src"))
_UD = os.path.join(REPO, "user_data")
_had_ud = os.path.isdir(_UD)

from fizgig.web import drives  # noqa: E402
from fizgig.web import server as S  # noqa: E402

if not _had_ud and os.path.isdir(_UD) and not os.listdir(_UD):
    os.rmdir(_UD)         # importing the server makes user_data/ (module-level user_file); never leave it behind

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


class _Handler:
    """Stands in for the HTTP handler: _browse ends in self.json(payload); this hands the payload back."""

    def json(self, payload):
        return payload

    def _places(self):
        return []


def opened(targets):
    return os.path.normcase(os.path.normpath(S.Handler._browse(_Handler(), targets)["path"]))


def norm(p):
    return os.path.normcase(os.path.normpath(p))


# --- 1. the server opens the first candidate that exists -----------------------------------------
T = tempfile.mkdtemp(prefix="fizgig_picker_")
lora_dir = os.path.join(T, "loras")
out_dir = os.path.join(T, "outputs")
os.makedirs(lora_dir)
os.makedirs(out_dir)
open(os.path.join(lora_dir, "a.safetensors"), "wb").close()
stale = os.path.join(T, "deleted", "gone")              # its parent does not exist either: a whole folder removed

ck("the pref wins when it is a real folder", opened([lora_dir, out_dir]) == norm(lora_dir))
ck("a stale pref falls through to the output folder (the regression)", opened([stale, out_dir]) == norm(out_dir),
   opened([stale, out_dir]))
renamed = os.path.join(T, "loras_old")                  # renamed away: its parent is still there
ck("a renamed pref folder falls through to the output folder, not to its parent",
   opened([renamed, out_dir]) == norm(out_dir), opened([renamed, out_dir]))
ck("  the stale pref is skipped even when the output folder is a file path's folder",
   opened([stale, os.path.join(lora_dir, "a.safetensors")]) == norm(lora_dir))
ck("nothing valid anywhere: Fizgig's folder, not a crash", opened([stale, os.path.join(stale, "x")]) == norm(S.BASE_DIR))
ck("no candidates (an empty pref, an empty field): Fizgig's folder", opened([""]) == norm(S.BASE_DIR)
   and opened([]) == norm(S.BASE_DIR))
ck("a single path still works (the picker's Up, a folder clicked)", opened(lora_dir) == norm(lora_dir))
ck("a file path opens its folder", opened([os.path.join(lora_dir, "a.safetensors")]) == norm(lora_dir))
ck("a relative pref is inside Fizgig (as pref_path resolves it), not the server's cwd",
   opened(["src", out_dir]) == norm(os.path.join(S.BASE_DIR, "src")))
_d = S.Handler._browse(_Handler(), [drives.DRIVES])
ck("the drives listing still answers", _d.get("path") == drives.DRIVES and _d.get("label") == "This computer")
_l = S.Handler._browse(_Handler(), [stale, lora_dir])
ck("  and the listing is of the folder opened", [f["name"] for f in _l["files"]] == ["a.safetensors"], _l["files"])

_src = open(os.path.join(REPO, "src", "fizgig", "web", "server.py"), encoding="utf-8").read()
ck("/api/browse hands _browse every path= (not just the first)",
   'self._browse(qs.get("path") or [""])' in _src)
ck("  parse_qs keeps repeated path= in order",
   urllib.parse.parse_qs("path=a&path=b%2Fc", keep_blank_values=True)["path"] == ["a", "b/c"])

# --- 2. app.js, executed --------------------------------------------------------------------------
JS = open(os.path.join(REPO, "src", "fizgig", "web", "static", "app.js"), encoding="utf-8").read()
m_lora = re.search(r"^const loraStart = .*?;$", JS, re.M)
m_ref = re.search(r"^const refStart = .*?;$", JS, re.M)
m_form = re.search(r'const startAt = r\.start === "lora" \? (.*?) :\s*$', JS, re.M)
m_q = re.search(r'await api\(("/api/browse\?" \+ .*?)\);', JS)
ck("app.js still has loraStart, refStart, the form's lora start and browse's query",
   all((m_lora, m_ref, m_form, m_q)))
NODE = shutil.which("node")
if not NODE:
    ck("node is on PATH (the app.js checks execute it)", False)
elif all((m_lora, m_ref, m_form, m_q)):
    prog = f"""
    let S;
    {m_lora.group(0)}
    {m_ref.group(0)}
    const formStart = () => {m_form.group(1)};
    const query = (path) => {m_q.group(1)};
    const out = {{}};
    S = {{set: {{input_lora_dir: "P", input_ref_dir: "R"}}, outputDefault: "O", imageFolder: "I", values: {{LORA_OUTPUT_DIR: "V"}}}};
    out.lora = loraStart(); out.ref = refStart(); out.form = formStart();
    out.picked = ("" || loraStart());
    S = {{set: {{}}, outputDefault: "O", imageFolder: "I", values: {{}}}};
    out.lora_nopref = loraStart(); out.ref_nopref = refStart(); out.form_nopref = formStart();
    out.q_list = query(["C:/a b/loras", "D:/out&put"]); out.q_one = query("C:/x"); out.q_none = query([]);
    console.log(JSON.stringify(out));
    """
    r = subprocess.run([NODE, "-e", prog], capture_output=True, text=True, timeout=60)
    try:
        o = json.loads(r.stdout)
    except ValueError:
        o = {}
    ck("the app.js snippets run in node", bool(o), (r.stdout + r.stderr)[-300:])
    if o:
        ck("loraStart: Preferences' LoRA folder, then the output folder", o["lora"] == ["P", "O"], o["lora"])
        ck("  no pref: the output folder alone", o["lora_nopref"] == ["O"], o["lora_nopref"])
        ck("refStart: Preferences' reference folder, then the training folder", o["ref"] == ["R", "I"], o["ref"])
        ck("  no pref: the training folder alone", o["ref_nopref"] == ["I"], o["ref_nopref"])
        ck("the form's LoRA rows: pref, this model's output folder, the default output folder",
           o["form"] == ["P", "V", "O"] and o["form_nopref"] == ["O"], (o["form"], o["form_nopref"]))
        ck("an empty field falls to the candidates (`value || loraStart()`)", o["picked"] == ["P", "O"])
        q = urllib.parse.parse_qs(o["q_list"].split("?", 1)[1], keep_blank_values=True)
        ck("browse() sends every candidate, in order, each its own path=",
           q.get("path") == ["C:/a b/loras", "D:/out&put"], o["q_list"])
        ck("  a single path is still one path=",
           urllib.parse.parse_qs(o["q_one"].split("?", 1)[1]).get("path") == ["C:/x"], o["q_one"])
        ck("  no candidates is a plain listing request", o["q_none"] == "/api/browse?", o["q_none"])

# --- 3. every LoRA picker uses it -----------------------------------------------------------------
for field in ("rySingle", "pfLora", "exSource", "mdPath", "stPrimary", "rmLora"):
    ck(f"the {field} picker seeds from loraStart()",
       re.search(r'pick\(\$\("' + field + r'"\)\.value \|\| loraStart\(\)', JS) is not None)

shutil.rmtree(T, ignore_errors=True)
print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
