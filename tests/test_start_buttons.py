"""The sidebar's link buttons (Fizgig 8 web UI): right labels, right colours, working links — headless, no browser.

The desktop's Start tab had a button row (Tutorial / Buy me a coffee / Deploy on RunPod / About) and this walked the
Tk tree INVOKING each one, because a button with a broken command builds fine and only fails when clicked. In
Fizgig 8 those links live in the web UI's sidebar (index.html <aside>) as plain <a href target=_blank> links, plus
an About button that app.js wires up. A dead link here is a wrong or drifted href, so that is what is pinned:
  1. Deploy on RunPod: exactly one, RunPod's violet, its href the same deploy link pods.py builds (Fizgig's
     template, the referral tag) so the two cannot drift; hidden on a pod; every per-GPU deploy link carries it too;
  2. Buy me a coffee: the real page, in a new tab;
  3. the sidebar's order, and the About button is wired;
  4. the app window sends these links to the browser (pywebview would otherwise open them inside itself).

Retired with the desktop: the per-tab YouTube Tutorial buttons. Fizgig 8 has none; check 0 pins that no
"Get help on YouTube" leftover came back with them.

Run: venv/Scripts/python.exe tests/test_start_buttons.py
"""
import os
import re
import sys
import urllib.parse
from html.parser import HTMLParser

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# src only. The repo root holds Fizgig 8's fizgig.py, which would shadow the fizgig package.
sys.path.insert(0, os.path.join(REPO, "src"))
_UD = os.path.join(REPO, "user_data")
_had_ud = os.path.isdir(_UD)
from fizgig.web import pods  # noqa: E402
if not _had_ud and os.path.isdir(_UD) and not os.listdir(_UD):
    os.rmdir(_UD)

STATIC = os.path.join(REPO, "src", "fizgig", "web", "static")
HTML = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
JS = open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()
CSS = open(os.path.join(STATIC, "style.css"), encoding="utf-8").read()
APPWIN = open(os.path.join(REPO, "src", "fizgig", "web", "app_window.py"), encoding="utf-8").read()

sys.stdout.reconfigure(errors="replace")   # the sidebar holds glyphs (the tabs toggle) a cp1252 console lacks
fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


class _Side(HTMLParser):
    """The <a> and <button> elements directly inside the first <aside> (the sidebar), in order, with their text."""

    def __init__(self):
        super().__init__()
        self.depth, self.done, self.items, self.cur = 0, False, [], None

    def handle_starttag(self, tag, attrs):
        if self.done:
            return
        if tag == "aside":
            self.depth += 1
            return
        if self.depth and tag in ("a", "button") and self.cur is None:
            self.cur = {"tag": tag, **dict(attrs), "text": ""}

    def handle_endtag(self, tag):
        if self.done:
            return
        if tag == "aside" and self.depth:
            self.depth -= 1
            self.done = not self.depth
        elif self.cur is not None and tag == self.cur["tag"]:
            self.cur["text"] = self.cur["text"].strip()
            self.items.append(self.cur)
            self.cur = None

    def handle_data(self, data):
        if self.cur is not None:
            self.cur["text"] += data


p = _Side()
p.feed(HTML)
side = p.items
by_text = {i["text"]: i for i in side}

# --- 0. the retired tutorial button left nothing behind --------------------------------------------
ck("no 'Get help on YouTube' left anywhere", "Get help on YouTube" not in HTML + JS)

# --- 1. Deploy on RunPod -----------------------------------------------------------------------------
ck("exactly one Deploy on RunPod in the UI", (HTML + JS).count("Deploy on RunPod") == 1,
   (HTML + JS).count("Deploy on RunPod"))
dep = by_text.get("Deploy on RunPod")
ck("  it is in the sidebar", dep is not None, f"{len(side)} sidebar links and buttons")
if dep:
    ck("  its href is pods.deploy_url(), so the two can't drift", dep.get("href") == pods.deploy_url(),
       (dep.get("href"), pods.deploy_url()))
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(dep.get("href", "")).query)
    ck("  carries the referral tag and Fizgig's template",
       q.get("ref") == [pods.REF] and q.get("template") == [pods.TEMPLATE], q)
    ck("  opens in a new tab, without handing the page an opener",
       dep.get("target") == "_blank" and "noopener" in (dep.get("rel") or ""))
    ck("  hidden on a pod (nothing to deploy from one)",
       re.search(r'\$\("' + dep.get("id", "?") + r'"\)\.style\.display = S\.onPod \? "none" : ""', JS) is not None)
# the last .runpod-btn rule that sets a background (later rules win; the phone layout's only re-spaces it)
_rp = [m.group(1) for m in re.finditer(r"\.runpod-btn\s*\{([^}]*)\}", CSS) if "background" in m.group(1)]
# The stylesheet names the colour as a token now (background: var(--runpod)); resolve it, last :root definition wins.
_tok = {k: v.upper() for blk in re.findall(r":root\s*\{([^}]*)\}", CSS)
        for k, v in re.findall(r"--([a-z0-9-]+)\s*:\s*(#[0-9A-Fa-f]{6})", blk)}
_bg = re.search(r"background:\s*(#[0-9A-Fa-f]{6}|var\(--([a-z0-9-]+)\))", _rp[-1]) if _rp else None
_col = (_tok.get(_bg.group(2)) if _bg and _bg.group(2) else (_bg.group(1).upper() if _bg else None))
ck("  RunPod's violet, not a Fizgig accent", _col == "#7C3AED", (_bg.group(1) if _bg else None, _col))
for gpu in ("RTX 5090", "RTX 4090", "H100 80GB HBM3"):
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(pods.deploy_url(gpu)).query)
    ck(f"the Pods tab's deploy link for {gpu} carries the referral and that GPU",
       q.get("ref") == [pods.REF] and q.get("gpu") == [gpu] and q.get("template") == [pods.TEMPLATE], q)

# --- 2. Buy me a coffee ----------------------------------------------------------------------------
cof = by_text.get("Buy me a coffee")
ck("Buy me a coffee is in the sidebar", cof is not None)
if cof:
    u = urllib.parse.urlsplit(cof.get("href", ""))
    ck("  opens buymeacoffee.com/lorasandlenses over https",
       u.scheme == "https" and u.netloc == "buymeacoffee.com" and u.path == "/lorasandlenses", cof.get("href"))
    ck("  in a new tab, noopener", cof.get("target") == "_blank" and "noopener" in (cof.get("rel") or ""))

# --- 3. order, and About is wired --------------------------------------------------------------------
_order = [i["text"] for i in side if i["text"] in ("Deploy on RunPod", "File manager", "About", "Buy me a coffee")]
ck("sidebar order is Deploy on RunPod / File manager / About / Buy me a coffee",
   _order == ["Deploy on RunPod", "File manager", "About", "Buy me a coffee"], _order)
ab = by_text.get("About")
ck("the About button is wired in app.js",
   ab is not None and f'$("{ab.get("id")}").addEventListener("click"' in JS, ab and ab.get("id"))

# --- 4. the app window opens them in the browser ------------------------------------------------------
ck("the app window sends external links to the browser",
   'webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True' in APPWIN)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
