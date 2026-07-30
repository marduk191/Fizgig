#!/usr/bin/env bash
# Bring up a virtual screen, put Fizgig on it, and serve it to a browser over noVNC.
#
# Order matters: the screen has to exist before Tkinter starts, and the VNC server has to be
# running before anything tries to draw on it.
set -euo pipefail

log() { echo "[fizgig] $*"; }

REPO_URL="${FIZGIG_REPO:-https://github.com/shootthesound/Fizgig.git}"
REPO_REF="${FIZGIG_REF:-master}"
APP_DIR="/workspace/Fizgig"
# Only the STARTING size now — KasmVNC resizes the desktop to match the browser window, so this
# stops being the compromise it used to be between "fits a laptop" and "less in-app scrolling".
SCREEN_W="${SCREEN_W:-1600}"
SCREEN_H="${SCREEN_H:-1400}"

mkdir -p /workspace/.cache /workspace/.insightface

# ---------------------------------------------------------------- VNC password
# An unprotected VNC on a public URL is a shell on someone else's GPU. If no password is set we
# generate one and print it, rather than leaving it open — a random password in the logs is
# recoverable; an open port is not fixable after the fact.
if [ -z "${VNC_PASSWORD:-}" ]; then
  VNC_PASSWORD="$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 12)"
  log "No VNC_PASSWORD set — generated one for this pod: ${VNC_PASSWORD}"
  log "Set VNC_PASSWORD in the template to choose your own."
fi
mkdir -p /root/.vnc
# KasmVNC authenticates the WEB session against its own user db, not a VNC-protocol password.
# -w gives write (input) permission; without it you get a read-only desktop you cannot click.
echo -e "${VNC_PASSWORD}
${VNC_PASSWORD}
" | kasmvncpasswd -u fizgig -w -o /root/.kasmpasswd >/dev/null 2>&1

# ---------------------------------------------------------------- Fizgig source
# Pulled rather than baked, so a new release needs no image rebuild. A pod restart is an update.
if [ -d "$APP_DIR/.git" ]; then
  log "Updating Fizgig in $APP_DIR"
  git -C "$APP_DIR" fetch --depth 1 origin "$REPO_REF" && git -C "$APP_DIR" reset --hard FETCH_HEAD
else
  log "Cloning Fizgig ($REPO_REF)"
  git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$APP_DIR"
fi

# Top up dependencies in case the checkout is newer than the image. Almost always a no-op, and
# a few seconds when it isn't — far cheaper than making users re-pull 10 GB for a new pin.
log "Checking dependencies"
uv pip install --link-mode=copy --index-strategy unsafe-best-match \
   -r "$APP_DIR/requirements.txt" 2>&1 | tail -2 || log "dependency top-up skipped"

# ---------------------------------------------------------------- persistent paths
# Point the portable output dirs at the volume. Model paths are left alone: fetch_models writes
# those itself, and overwriting them here would stomp a path the user set by hand.
python3 - <<'PY'
import json, os
p = "/workspace/Fizgig/prefs.json"
prefs = {}
if os.path.exists(p):
    try:
        prefs = json.load(open(p, encoding="utf-8"))
    except Exception:
        prefs = {}
for key, path in (("lora_output_dir", "/workspace/output_loras"),
                  ("profiles_dir",    "/workspace/profiles"),
                  ("cache_dir",       "/workspace/cache")):
    prefs.setdefault(key, path)
    os.makedirs(prefs[key], exist_ok=True)
os.makedirs("/workspace/models", exist_ok=True)
json.dump(prefs, open(p, "w", encoding="utf-8"), indent=2)
print("[fizgig] output dirs on /workspace")
PY

# ---------------------------------------------------------------- optional model prefetch
# Off by default. Downloading tens of GB unasked spends the user's money and may fetch the
# family they didn't want — the Preferences button does this on demand, with a progress bar.
if [ -n "${FETCH_MODELS:-}" ]; then
  log "FETCH_MODELS=${FETCH_MODELS} — downloading before launch"
  IFS=',' read -ra FAMS <<< "$FETCH_MODELS"
  ARGS=()
  for f in "${FAMS[@]}"; do ARGS+=(--family "$(echo "$f" | xargs)"); done
  ( cd "$APP_DIR" && PYTHONPATH="$APP_DIR/src" python3 -m fizgig.scripts.fetch_models \
      "${ARGS[@]}" ${FETCH_MODELS_EXTRA:-} ) || log "model fetch incomplete — use the Preferences button"
fi

# ---------------------------------------------------------------- the screen
# A restarted container keeps /tmp, so the previous run's X lock survives and the X server refuses
# to start with "Server is already active for display 1" — even though nothing is running. It then
# has no display, exits, and the container dies with no obvious cause. Stopping and starting a pod
# is routine, so this has to be cleared every boot, not just on first run.
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1
rm -rf /tmp/.X11-unix/X1-lock /root/.vnc/*.pid

# Session bus, started before anything that might open a link. Firefox passes a URL to an
# already-running instance over this bus; without it the second link you click in a session
# starts a second firefox, hits the profile lock, and shows "Close Firefox" instead of the page.
# Deliberately non-fatal: set -e is on, and a bus that fails to start must not take the whole
# container down with it.
if [ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]; then
    if eval "$(dbus-launch --sh-syntax 2>/dev/null)" 2>/dev/null; then
        export DBUS_SESSION_BUS_ADDRESS
        log "session bus up"
    else
        log "WARN: no session bus — links may only open once per session"
    fi
fi

log "Starting KasmVNC display (${SCREEN_W}x${SCREEN_H}, resizes to your browser window)"
# One process instead of Xvfb + x11vnc + websockify: KasmVNC is an X server WITH a web server,
# encoding in WebP rather than the LAN-shaped defaults noVNC fell back to.
# SecurityTypes None + basic auth ON is the pairing that works, and the two halves are not
# independent. KasmVNC defaults SecurityTypes to VncAuth, which wants a VNC-PROTOCOL password
# (PasswordFile) -- a different credential from the web login. Set only the web one and the
# browser authenticates fine, loads the client, and THEN the session underneath is rejected with
# "No password configured for VNC Auth". Turning the VNC layer off is correct here because
# DisableBasicAuth=0 means the HTTP gate is still doing the authenticating; it is stated
# explicitly so nobody removes it later and leaves the pod wide open.
# -httpd is NOT optional: the built-in default is /usr/local/share/kasmvnc/www but the Debian
# package installs to /usr/share/kasmvnc/www, so without it the server authenticates you fine and
# then closes the connection with an empty reply, having nothing to serve.
# KasmPasswordFile is the WEB login (what kasmvncpasswd writes); PasswordFile/-rfbauth is the
# VNC-protocol password, which nothing here uses since the client is the browser.
# DynamicQuality 4-9 lets it drop quality while you drag and restore it when you stop, which is
# where the responsiveness actually comes from over a long link.
Xvnc :1     -geometry "${SCREEN_W}x${SCREEN_H}"     -depth 24     -websocketPort 6080     -interface 0.0.0.0     -KasmPasswordFile /root/.kasmpasswd     -httpd /usr/share/kasmvnc/www     -SecurityTypes None     -DisableBasicAuth=0     -sslOnly=0     -AlwaysShared     -FrameRate=60     -DynamicQualityMin=4     -DynamicQualityMax=9     >/var/log/kasmvnc.log 2>&1 &

for _ in $(seq 1 60); do xdpyinfo -display :1 >/dev/null 2>&1 && break; sleep 0.25; done
if ! xdpyinfo -display :1 >/dev/null 2>&1; then
    log "ERROR: display never came up. Last lines of /var/log/kasmvnc.log:"
    tail -15 /var/log/kasmvnc.log || true
    exit 1
fi

# Tkinter draws Toplevels without decoration and mishandles focus if nothing is managing the
# screen — Fizgig leans on dialogs (Repair Studio pop-out, the token prompt, Problem Images),
# so a window manager is required, not cosmetic.
openbox &

log "Serving Fizgig on :6080"

# Drag-and-drop file transfer on its own port: datasets in, trained LoRAs out, no terminal.
# Shares VNC_PASSWORD rather than inventing a second credential — it exposes the same /workspace
# the desktop already does, so a separate password would be security theatre, and one fewer
# thing to lose. Non-fatal: file transfer failing must not stop training.
if command -v filebrowser >/dev/null 2>&1; then
    FB_DB=/workspace/.filebrowser.db
    # filebrowser rejects passwords under 12 characters. A shorter VNC_PASSWORD therefore creates
    # NO user at all, and the login just 403s — so pad rather than fail, and print what the
    # padded password actually is. (Found the hard way: the first version wrapped these commands
    # in `|| true`, so the log said "File manager on :8080" while the account did not exist.)
    FB_PW="$VNC_PASSWORD"
    while [ ${#FB_PW} -lt 12 ]; do FB_PW="${FB_PW}0"; done

    filebrowser config init --database "$FB_DB" >/dev/null 2>&1 || true
    filebrowser config set --database "$FB_DB" \
        --address 0.0.0.0 --port 8080 --root /workspace >/dev/null 2>&1 || true
    # Errors are logged, not swallowed: a file manager nobody can log into is worse than one
    # that says why.
    if ! _fb_out=$(filebrowser users add admin "$FB_PW" --perm.admin --database "$FB_DB" 2>&1); then
        if ! _fb_out=$(filebrowser users update admin --password "$FB_PW" \
                          --database "$FB_DB" 2>&1); then
            log "WARN: file manager login not set up: $(echo "$_fb_out" | tail -1)"
        fi
    fi
    filebrowser --database "$FB_DB" >/dev/null 2>&1 &
    if [ "$FB_PW" = "$VNC_PASSWORD" ]; then
        log "File manager on :8080 — user 'admin', same password as VNC"
    else
        log "File manager on :8080 — user 'admin', password '${FB_PW}'"
        log "  (padded: filebrowser needs 12+ characters and VNC_PASSWORD is shorter)"
    fi
fi

cat <<EOF
[fizgig] ------------------------------------------------------------
[fizgig]  Ready. Open the pod's HTTP port 6080.
[fizgig]  Your browser will ask for a username and password:
[fizgig]      username: fizgig
[fizgig]      password: ${VNC_PASSWORD}
[fizgig]
[fizgig]  Models: Preferences -> "Download models for me"
[fizgig]          Krea 2 needs no HuggingFace account; Klein needs a token.
[fizgig] ------------------------------------------------------------
EOF

# Maximise Fizgig once it appears. Its geometry is tuned for a Windows desktop with Segoe UI;
# under DejaVu the tab strip is wider, so "Preferences" truncates at the designed width. Filling
# the screen fixes that and stops a browser-sized viewport being mostly empty desktop. Runs in
# the background because the GUI has to exist first, and the launch below never returns.
(
  for _ in $(seq 1 90); do
    if wmctrl -l 2>/dev/null | grep -q "Fizgig"; then
      wmctrl -r "Fizgig" -b add,maximized_vert,maximized_horz 2>/dev/null && \
        log "window maximised"
      break
    fi
    sleep 1
  done
) &

cd "$APP_DIR"
export PYTHONPATH="$APP_DIR/src"
# exec so Fizgig is PID 1's successor: a container stop signals the app directly, and the
# container's lifetime is the app's lifetime rather than outliving a crashed GUI.
exec python3 lora_trainer_gui.py
