#!/bin/bash
# Fizgig's web UI in your web browser: this terminal runs it (Ctrl+C stops it) and the browser opens on it once
# it is up. Already running? The browser just opens on it.
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
    echo "Fizgig's Python environment is missing - run the installer first."
    exit 1
fi
source venv/bin/activate
exec python fizgig.py --web --open "$@"
