#!/bin/bash
# Fizgig in its own app window. In your web browser instead: ./launch_fizgig_browser.sh
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
    echo "Fizgig's Python environment is missing - run the installer first."
    exit 1
fi
source venv/bin/activate
python fizgig.py "$@" &
disown
