#!/bin/bash
# Gizmo - cut training clips Fizgig will accept (the Linux twin of Launch Gizmo.bat)
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
    echo "Fizgig's Python environment is missing - run the installer first."
    exit 1
fi
source venv/bin/activate
python src/gizmo/gizmo.py "$@" &
disown
