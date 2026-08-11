r"""
Regression check for the optional tower-SS <-> drivetrain feedback.

Runs the two-turbine sim three times on a short horizon and verifies:

  * feedback OFF        vs  feedback ON with c=0   -> byte-identical
    (the c=0 branch must reproduce the forward-only model exactly),
  * feedback OFF        vs  feedback ON with c>0   -> differs on WT1
    (the two-way coupling actually changes the solution).

Usage
-----
  .\.venv\Scripts\python.exe casestudies\dyn_sim\_verify_ss_feedback_c0.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIM = PROJECT_ROOT / "casestudies" / "dyn_sim" / "test_WT_LEOGO_2WT_sim.py"
CSV_DIR = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "csv_files"
LOG = CSV_DIR / "_verify_log.txt"


def _log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")

COMMON = [
    "--osc-load-mw", "2", "--osc-freq-hz", "0.234",
    "--f-ss-hz", "0.234", "--f-ss-hz-wt2", "0.25",
    "--droop-wt1", "0", "--droop-wt2", "0",
    "--wind-wt1", "10", "--wind-wt2", "10",
    "--warmup-s", "5", "--t-end", "25", "--dt", "0.01",
    "--headroom", "0.05",
]
COLS = [
    "ss_accel_mps2_wt1", "ss_accel_mps2_wt2",
    "omega_e_pu_wt1", "P_uic_bus_sys_pu_wt2",
]


def _run(extra, out):
    cmd = [sys.executable, str(SIM), *COMMON, *extra, "--out", str(out)]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True,
                   stdout=subprocess.DEVNULL)


def _maxdiff(a, b):
    return {c: float(np.max(np.abs(a[c].to_numpy() - b[c].to_numpy()))) for c in COLS}


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    LOG.unlink(missing_ok=True)
    f_off = CSV_DIR / "_verify_ff_off.csv"
    f_c0 = CSV_DIR / "_verify_ff_c0.csv"
    f_c05 = CSV_DIR / "_verify_ff_c05.csv"

    _log("Running 3 short 2WT sims (feedback off / c=0 / c=0.05)...")
    _run([], f_off)
    _log("  [1/3] feedback OFF done")
    _run(["--ss-feedback", "--ss-feedback-c", "0",
          "--ss-feedback-target", "wt1"], f_c0)
    _log("  [2/3] feedback c=0 done")
    _run(["--ss-feedback", "--ss-feedback-c", "0.05",
          "--ss-feedback-target", "wt1"], f_c05)
    _log("  [3/3] feedback c=0.05 done")

    off = pd.read_csv(f_off)
    c0 = pd.read_csv(f_c0)
    c05 = pd.read_csv(f_c05)

    d_c0 = _maxdiff(off, c0)
    d_c05 = _maxdiff(off, c05)

    _log("\n  max|off - (feedback c=0)|  (should be ~0):")
    for k, v in d_c0.items():
        _log(f"    {k:24s}: {v:.3e}")
    _log("\n  max|off - (feedback c=0.05)|  (WT1 should be > 0):")
    for k, v in d_c05.items():
        _log(f"    {k:24s}: {v:.3e}")

    c0_ok = max(d_c0.values()) < 1e-12
    c05_active = d_c05["ss_accel_mps2_wt1"] > 1e-9
    _log(f"\n  c=0 reproduces forward-only exactly : "
         f"{'PASS' if c0_ok else 'FAIL'}")
    _log(f"  c>0 activates the two-way coupling   : "
         f"{'PASS' if c05_active else 'FAIL'}")

    for f in (f_off, f_c0, f_c05):
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
