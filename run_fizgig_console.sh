#!/bin/bash
# Foreground launcher with a visible console.
#
# launch_fizgig.sh backgrounds and disowns fizgig.py, detaching it from the terminal. That
# hides stdout, where the server and the caching, training and Studio subprocesses log, so a
# run that dies early leaves nothing to read.
#
# Use this one when something is failing and you need the traceback.
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
    echo "Fizgig's Python environment is missing - run the installer first."
    exit 1
fi
source venv/bin/activate
python fizgig.py "$@"
