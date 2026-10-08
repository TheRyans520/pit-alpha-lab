"""Reproducible synthetic ledger benchmark; each measurement uses a fresh process."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def worker(days: int, instruments: int) -> dict:
    import resource
    import numpy as np
    import pandas as pd
    from pitalpha.backtest import build_ranked_portfolio

    rng = np.random.default_rng(17)
    frame = pd.DataFrame({
        "datetime": np.repeat(pd.bdate_range("2019-01-01", periods=days), instruments),
        "instrument": np.tile([f"S{i:05d}" for i in range(instruments)], days),
        "score": rng.normal(size=days * instruments),
        "realized_return_1d": rng.normal(0, .01, size=days * instruments),
    })
    started = time.perf_counter()
    daily, holdings = build_ranked_portfolio(frame, top_k=min(30, instruments), cost_bps=[0, 5, 10, 20])
    elapsed = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"seconds": elapsed, "rows_per_second": len(frame) / elapsed,
            "process_peak_rss_mib": rss / (1024 ** 2 if sys.platform == "darwin" else 1024),
            "input_frame_mib": int(frame.memory_usage(deep=True).sum()) / 1024 ** 2,
            "output_sha256": hashlib.sha256(
                daily.to_csv(index=False).encode() + holdings.to_csv(index=False).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1260)
    parser.add_argument("--instruments", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.days, args.instruments, args.repeats) <= 0:
        parser.error("sizes and repetitions must be positive")
    if args.worker:
        print(json.dumps(worker(args.days, args.instruments)))
        return
    from pitalpha.artifacts.manifest import source_identity
    from importlib.metadata import version
    from datetime import datetime, timezone

    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--days", str(args.days),
               "--instruments", str(args.instruments)]
    runs = [json.loads(subprocess.check_output(command, env=env, text=True)) for _ in range(args.repeats)]
    if len({row["output_sha256"] for row in runs}) != 1:
        raise RuntimeError("benchmark outputs differ between repetitions")
    payload = {
        "schema_version": 1, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
        "logical_cpus": os.cpu_count(), "packages": {name: version(name) for name in ("numpy", "pandas")},
        "source": source_identity(Path(__file__).resolve().parents[1]),
        "fixture": {"days": args.days, "instruments": args.instruments, "rows": args.days * args.instruments,
                    "seed": 17, "top_k": min(30, args.instruments), "cost_bps": [0, 5, 10, 20]},
        "measurement": "ledger only; imports/fixture construction excluded from time; RSS is whole child process high-water mark",
        "threads": {name: env[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "median_seconds": statistics.median(row["seconds"] for row in runs),
        "median_rows_per_second": statistics.median(row["rows_per_second"] for row in runs),
        "maximum_process_peak_rss_mib": max(row["process_peak_rss_mib"] for row in runs), "runs": runs,
    }
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
