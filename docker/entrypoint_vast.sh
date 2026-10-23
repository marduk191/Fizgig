#!/usr/bin/env bash
# Fizgig on Vast.ai: HTTPS addresses for the web UI and the file manager (Cloudflare quick tunnels), then the pod
# image's own startup. Vast serves ports as plain http on a public address; the tunnels carry the password and the
# login over https. A tunnel's address is new each time the instance starts. Non-fatal: without a tunnel the plain
# http address still works.
set -uo pipefail

log() { echo "[fizgig] $*"; }
TUNNELS=/tmp/fizgig_tunnels.json         # read by the web UI (fizgig/web/pod_stop.py tunnels())
echo "{}" > "$TUNNELS"

record() {   # name url -> into $TUNNELS (both tunnels write it, so one at a time)
  ( flock 9
    python3 - "$TUNNELS" "$1" "$2" <<'PY'
import json, sys
p, k, v = sys.argv[1:4]
try:
    d = json.load(open(p))
except Exception:
    d = {}
d[k] = v
json.dump(d, open(p, "w"))
PY
  ) 9>"$TUNNELS.lock"
}

tunnel() {   # port name: a tunnel, its address recorded once Cloudflare gives one; tried again if it ends without one
  local port="$1" name="$2" logf="/tmp/cloudflared_$1.log" url="" pid
  for attempt in 1 2 3; do
    cloudflared tunnel --no-autoupdate --url "http://localhost:${port}" > "$logf" 2>&1 &
    pid=$!
    for _ in $(seq 1 300); do          # a slow Cloudflare answer (far from its edge) can take minutes
      url=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$logf" | head -n 1)
      [ -n "$url" ] && break
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if [ -n "$url" ]; then
      record "$name" "$url"
      log "HTTPS ${name}: ${url}"
      return 0
    fi
    kill "$pid" 2>/dev/null
    log "no HTTPS address for ${name} yet (try ${attempt} of 3)"
    sleep 5
  done
  log "WARN: no HTTPS address for ${name}: use the plain http one"
}

if command -v cloudflared >/dev/null 2>&1; then
  ( sleep 5
    tunnel 8081 web & tunnel 8080 files & wait
    plain=""
    [ -n "${PUBLIC_IPADDR:-}" ] && [ -n "${VAST_TCP_PORT_8081:-}" ] && plain="http://${PUBLIC_IPADDR}:${VAST_TCP_PORT_8081}"
    web=$(python3 -c "import json;print(json.load(open('$TUNNELS')).get('web',''))" 2>/dev/null)
    echo "[fizgig] ------------------------------------------------------------"
    [ -n "$web" ] && echo "[fizgig]  Open Fizgig (https): ${web}"
    [ -n "$plain" ] && echo "[fizgig]  Or (plain http):    ${plain}"
    echo "[fizgig]  The https address is new each time the instance starts."
    echo "[fizgig] ------------------------------------------------------------" ) &
fi

exec /usr/local/bin/entrypoint.sh
