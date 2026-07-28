#!/bin/bash
# Foreground launcher with a visible console.
#
# run_fizgig.sh backgrounds and disowns the GUI, which detaches it from the terminal.
# That hides stdout, and the caching and training subprocesses log there -- so a run
# that dies early leaves nothing to read.
#
# Use this one when something is failing and you need the traceback.
cd "$(dirname "$0")"
source venv/bin/activate
python lora_trainer_gui.py
