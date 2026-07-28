"""CLI entry point for Klein 9B LoRA training.

Usage:
    accelerate launch src/fizgig/scripts/train.py --dit path/to/dit --vae path/to/ae ...

This is the script the GUI calls via subprocess.
"""

import sys
import os

# Ensure fizgig package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# CUDA allocator policy, set before torch is imported below — the backend is fixed at CUDA init.
# Training churns the allocator: large tensors allocated and freed every step, and on a rotating
# fine-tune whole windows swap between bf16 and fp8 each epoch. The default allocator carves from
# fixed-size segments, which fragments under that pattern and worsens as a run goes on.
# The GUI already sets this and the training subprocess inherits it; this covers headless runs.
# Respects an existing value, and FIZGIG_NO_EXPANDABLE=1 opts out for A/B testing.
#
# Windows can't use expandable_segments — the allocator rejects it outright and silently falls
# back to the default allocator, so the fix never applied there. It gets CUDA's stream-ordered
# allocator (cudaMallocAsync) instead, which solves the same problem a different way: the driver
# manages one growable pool rather than PyTorch carving fixed segments, so a freed block of the
# wrong shape doesn't strand memory. Measured on a fragmentation repro at ~5% headroom:
# num_alloc_retries 9 -> 0, worst-case step 84ms -> 3.8ms. See lora_trainer_gui.py for the full
# note and the rejected alternatives.
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF") and os.environ.get("FIZGIG_NO_EXPANDABLE") != "1":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "backend:cudaMallocAsync" if sys.platform == "win32" else "expandable_segments:True"
    )

from fizgig.training.trainer import KleinTrainer, setup_parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    trainer = KleinTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
