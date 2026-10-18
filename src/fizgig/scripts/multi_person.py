"""Multi-person mode's first stage: each subject's side-by-side composites (fizgig/families/multi_person.py), built
before the cache stages so they cache and train as one more dataset block per subject.

    multi_person.py --ext .txt --subject FOLDER FUSED_TRIGGER OUT_DIR [--subject ...]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fizgig.families import multi_person  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", nargs=3, action="append", required=True, metavar=("FOLDER", "FUSED", "OUT"))
    p.add_argument("--ext", default=".txt")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    multi_person.build([{"folder": f, "fused": t, "out": o, "index": i} for i, (f, t, o) in enumerate(a.subject)],
                       ext=a.ext, seed=a.seed, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
