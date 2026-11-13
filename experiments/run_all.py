"""Reproduce every table and figure in the report.

    python experiments/run_all.py            # 5,000 replications (about 1-2 minutes)
    python experiments/run_all.py --R 1000   # quicker pass
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e01_linear_algebra, e02_baseline, e03_violations, e04_robust_inference, e05_diagnostics  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=5000, help="Monte Carlo replications per cell")
    R = ap.parse_args().R
    t0 = time.time()
    for mod in (e01_linear_algebra, e02_baseline, e03_violations, e04_robust_inference, e05_diagnostics):
        mod.main(R)
    print(f"done in {time.time() - t0:.0f}s")
