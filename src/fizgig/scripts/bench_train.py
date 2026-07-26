"""Measure training throughput, VRAM and CPU load for a given Fizgig configuration.

Runs a real training subprocess and samples the machine while it works, so the numbers are
what a user would actually experience — model load, quantization, block swap and all.

    python src/fizgig/scripts/bench_train.py --label fp8-noswap \
        --dataset_config my.toml --dit /models/Krea-2-raw.safetensors \
        --steps 40 -- --blocks_to_swap 0

Everything after a bare `--` is passed straight through to krea2_train.py, so any
configuration can be benchmarked without this script knowing about it.

Results append to bench_results.json as one row per run; --table prints the accumulated
rows as a comparison. Report speed as SECONDS PER STEP so bigger is always worse.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_RESULTS = os.path.join(REPO, "bench_results.json")

# tqdm writes "1.23s/it" or "1.23it/s"; take the LAST match so we get the settled rate rather
# than the wildly optimistic first few steps.
_RATE_RE = re.compile(r"([0-9.]+)\s*(s/it|it/s)")
_STEP_RE = re.compile(r"steps:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)")


def _nvidia_smi_mb() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5)
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return None


class Sampler(threading.Thread):
    """Polls CPU and VRAM while the training subprocess runs.

    CPU is normalised to a percentage of the WHOLE machine (psutil reports per-core sums, so a
    fully-loaded 16-thread box reads 1600% without this) — that makes it comparable to what
    people quote from Task Manager, which is how the reports we're chasing were phrased.
    """

    def __init__(self, proc, interval: float = 0.5):
        super().__init__(daemon=True)
        self.proc = proc
        self.interval = interval
        self.cpu_samples: list[float] = []
        self.vram_samples: list[int] = []
        self.stop_flag = threading.Event()
        self.error: str | None = None

    def run(self):
        p = None
        ncpu = 1
        try:
            import psutil
            p = psutil.Process(self.proc.pid)
            ncpu = psutil.cpu_count() or 1
            p.cpu_percent(None)          # prime; first call always returns 0.0
        except Exception as e:
            self.error = f"psutil unavailable: {e}"
        # Children are primed lazily and kept, so each keeps its own cpu_percent baseline —
        # re-creating them every tick would make every sample read 0.0.
        kids = {}
        while not self.stop_flag.is_set():
            time.sleep(self.interval)
            mb = _nvidia_smi_mb()
            if mb is not None:
                self.vram_samples.append(mb)
            if p is None:
                continue
            total = None
            try:
                total = p.cpu_percent(None)
            except Exception as e:
                # The process exiting between our tick and its wait() is normal, not a failure.
                if not self.cpu_samples:
                    self.error = f"parent sample failed: {e}"
                break
            if total is None:
                continue
            try:
                for c in p.children(recursive=True):
                    if c.pid not in kids:
                        kids[c.pid] = c
                        try:
                            c.cpu_percent(None)   # prime, contributes 0 this tick
                        except Exception:
                            pass
                        continue
                    try:
                        total += kids[c.pid].cpu_percent(None)
                    except Exception:
                        kids.pop(c.pid, None)
            except Exception:
                pass    # children vanishing mid-walk must not lose the parent's sample
            self.cpu_samples.append(total / ncpu)


def run_once(args, passthrough: list[str]) -> dict:
    train_script = os.path.join(HERE, "krea2_train.py")
    out_dir = args.output_dir or os.path.join(REPO, "_bench_out")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [sys.executable, train_script,
           "--dataset_config", args.dataset_config,
           "--dit", args.dit,
           "--output_dir", out_dir,
           "--output_name", f"bench_{args.label}",
           "--max_train_epochs", str(args.epochs),
           "--save_every_n_epochs", "0",
           "--learning_rate", "1e-4",
           "--seed", "42"] + passthrough

    print(f"[bench] {args.label}: {' '.join(cmd[1:])}\n", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1)
    sampler = Sampler(proc)
    sampler.start()

    rates: list[tuple[float, str]] = []
    first_step_at = None
    steps_seen = 0
    tail: list[str] = []
    for line in proc.stdout:
        tail.append(line.rstrip()[:200])
        if len(tail) > 40:
            tail.pop(0)
        for chunk in line.replace("\r", "\n").split("\n"):
            m = _STEP_RE.search(chunk)
            if m:
                n = int(m.group(1))
                if n > steps_seen:
                    steps_seen = n
                    if first_step_at is None:
                        first_step_at = time.time() - t0
            r = _RATE_RE.search(chunk)
            if r:
                rates.append((float(r.group(1)), r.group(2)))
        if args.verbose:
            print(line, end="")

    rc = proc.wait()
    sampler.stop_flag.set()
    sampler.join(timeout=3)
    elapsed = time.time() - t0

    # Convert every rate to s/it, then take the median of the SECOND HALF: the early steps
    # include cache warm-up and allocator growth, which flatter or punish a config unfairly.
    s_per_it = [v if unit == "s/it" else (1.0 / v if v else 0.0) for v, unit in rates]
    settled = s_per_it[len(s_per_it) // 2:] or s_per_it
    row = {
        "label": args.label,
        "ok": rc == 0,
        "returncode": rc,
        "s_per_it": round(statistics.median(settled), 4) if settled else None,
        "steps_measured": steps_seen,
        "load_seconds": round(first_step_at, 1) if first_step_at else None,
        "total_seconds": round(elapsed, 1),
        "peak_vram_mb": max(sampler.vram_samples) if sampler.vram_samples else None,
        "mean_cpu_pct": round(statistics.mean(sampler.cpu_samples), 1) if sampler.cpu_samples else None,
        "peak_cpu_pct": round(max(sampler.cpu_samples), 1) if sampler.cpu_samples else None,
        "args": passthrough,
        "epochs": args.epochs,
    }
    if sampler.error:
        # Surface it: a silently-empty CPU column would quietly invalidate the whole point.
        row["sampler_error"] = sampler.error
        print(f"[bench] WARNING: CPU sampling failed — {sampler.error}")
    elif not sampler.cpu_samples:
        row["sampler_error"] = "no CPU samples collected"
        print("[bench] WARNING: no CPU samples collected")
    if rc != 0:
        row["tail"] = tail[-12:]
        print("\n[bench] FAILED — last output:")
        for t in tail[-12:]:
            print("   ", t)
    return row


def print_table(rows: list[dict]):
    if not rows:
        print("no results yet")
        return
    hdr = f"{'label':<22} {'s/it':>7} {'peak VRAM':>11} {'CPU mean':>9} {'CPU peak':>9} {'load':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        vram = f"{r['peak_vram_mb'] / 1024:.1f} GB" if r.get("peak_vram_mb") else "-"
        print(f"{r['label']:<22} {r.get('s_per_it') or '-':>7} {vram:>11} "
              f"{str(r.get('mean_cpu_pct') or '-') + '%':>9} "
              f"{str(r.get('peak_cpu_pct') or '-') + '%':>9} "
              f"{str(r.get('load_seconds') or '-') + 's':>7}"
              + ("" if r.get("ok") else "   [FAILED]"))


def main():
    p = argparse.ArgumentParser(
        description="Benchmark a Fizgig training configuration (speed / VRAM / CPU).",
        epilog="Pass trainer flags after a bare --, e.g.  ... --label nf4 -- --quantize_4bit")
    p.add_argument("--label", help="short name for this configuration")
    p.add_argument("--dataset_config", help="dataset .toml (caches must already exist)")
    p.add_argument("--dit", help="Krea 2 RAW DiT")
    p.add_argument("--epochs", type=int, default=2, help="epochs to run (default 2)")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--results", default=DEFAULT_RESULTS)
    p.add_argument("--verbose", action="store_true", help="stream the trainer output")
    p.add_argument("--table", action="store_true", help="print accumulated results and exit")
    argv = sys.argv[1:]
    passthrough: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    args = p.parse_args(argv)

    rows = []
    if os.path.exists(args.results):
        try:
            with open(args.results, encoding="utf-8") as f:
                rows = json.load(f)
        except Exception:
            rows = []

    if args.table:
        print_table(rows)
        return

    missing = [n for n in ("label", "dataset_config", "dit") if not getattr(args, n)]
    if missing:
        p.error("missing required: " + ", ".join("--" + m for m in missing))

    row = run_once(args, passthrough)
    rows.append(row)
    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print()
    print_table([row])
    print(f"\n[bench] appended to {args.results}")


if __name__ == "__main__":
    main()
