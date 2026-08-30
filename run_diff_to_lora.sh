#!/bin/bash
# Checkpoint to LoRA - Linux/pod launcher (the .bat twin). Windows users double-click
# run_diff_to_lora.bat instead.
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
    echo "venv not found - run the installer first."
    exit 1
fi
source venv/bin/activate
python diff_to_lora_gui.py
