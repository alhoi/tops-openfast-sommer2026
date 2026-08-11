r"""
Side-to-side (SS) tower-resonance check for the five support flavours.

A sustained slug-flow process load pulsates the LEOGO grid at the tower SS mode
(0.233 Hz). Each support law modulates the WT generator torque in response to
the grid-frequency oscillation and can pump the lightly damped tower SS mode.
This runs the five cases and measures the settled tower-top SS acceleration
(fmu_YawBrTAyp) at the drive frequency, to show which laws excite the resonance
and that the support-path notch (0.233 Hz) suppresses it.

Cases (same gains as the tuned GT-trip study):
  no support               droop 0,   inertia 0
  droop                    droop 2e7, inertia 0
  droop + notch            droop 2e7, inertia 0,   notch 0.233 Hz Q8
  droop + inertia          droop 2e7, inertia 8e7, LPF 2 Hz
  droop + inertia + notch  droop 2e7, inertia 8e7, LPF 2 Hz, notch 0.233 Hz Q8

Needs ElastoDyn SS-only (TwSSDOF1=True, TwFADOF1=False) and fast_debug.fmu
(exposes fmu_YawBrTAyp). ~25 min per case at t_end 400.

Usage:
  .\.venv\Scripts\python.exe casestudies\dyn_sim\ss_resonance_cases.py --dry-run
  .\.venv\Scripts\python.exe casestudies\dyn_sim\ss_resonance_cases.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\ss_resonance_cases.py --plot-only
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIVER = PROJECT_ROOT / "casestudies" / "dyn_sim" / "test_WT_LEOGO_FMU_sim.py"
OUT_DIR = PROJECT_ROOT / "results" / "em_interaction_sweep" / "ss_resonance"

F_SS_HZ = 0.233
T_END = 400.0
SS_COL = "fmu_YawBrTAyp"

# (tag, label, colour, droop [Nm/Hz], inertia [Nm.s/Hz], lpf [Hz], notch_hz, notch_q)
CASES = [
    ("none",          "no support",              "#7f7f7f", 0.0, 0.0, 0.5, 0.0,   2.0),
    ("droop",         "droop",                   "#1f77b4", 2e7, 0.0, 0.5, 0.0,   2.0),
    ("droop_notch",   "droop + notch",           "#ff7f0e", 2e7, 0.0, 0.5, 0.233, 8.0),
    ("di",            "droop + inertia",         "#2ca02c", 2e7, 8e7, 2.0, 0.0,   2.0),
    ("di_notch",      "droop + inertia + notch", "#9467bd", 2e7, 8e7, 2.0, 0.233, 8.0),
]


def run_case(case, *, force: bool) -> Path:
    tag, _lab, _c, droop, inertia, lpf, notch_hz, notch_q = case
    out = OUT_DIR / f"ss_{tag}.csv"
    if out.exists() and not force:
        print(f"  [skip] {tag} (exists)")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(DRIVER),
        "--t-end", f"{T_END}", "--fmu", "debug",
        "--wt-pref-mw", "12.88", "--zmq-grid", "--fix-leogo-xqt",
        "--droop-nm-per-hz", f"{droop:.0f}",
        "--inertia-nm-s-per-hz", f"{inertia:.0f}",
        "--freq-lpf-hz", f"{lpf}", "--freq-lpf-order", "2",
        "--support-notch-hz", f"{notch_hz}", "--support-notch-q", f"{notch_q}",
        "--support-start", "5", "--support-max-nm", "15000000",
        "--load-step-mw", "3.0", "--event-time", "10",
        "--event-duration", "400", "--load-ramp-on-s", "3",
        "--load-sine-mean", "0.0", "--load-sine-amplitude", "1.0",
        "--load-sine-freq-hz", f"{F_SS_HZ}",
        "--zmq-log", str((OUT_DIR / f"ss_{tag}_zmq.csv").relative_to(PROJECT_ROOT)),
        "--out", str(out.relative_to(PROJECT_ROOT)),
    ]
    print(f"  [run ] {tag} (droop={droop:.0e}, inertia={inertia:.0e}, "
          f"lpf={lpf} Hz, notch={notch_hz} Hz Q{notch_q}) ...", flush=True)
    log = OUT_DIR / f"ss_{tag}.log"
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=fh,
                              stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not out.exists():
        print(f"    FAILED (see {log})")
    return out


def lockin(t, sig, f, t_lo, t_hi) -> float:
    m = (t >= t_lo) & (t <= t_hi)
    tt, s = t[m], np.asarray(sig[m], float)
    if tt.size < 8:
        return float("nan")
    s = s - s.mean()
    w = 2.0 * np.pi * f
    c = np.trapezoid(s * np.cos(w * tt), tt)
    q = np.trapezoid(s * np.sin(w * tt), tt)
    span = tt[-1] - tt[0]
    return 2.0 * np.hypot(c, q) / span if span > 0 else float("nan")


def analyze_and_plot() -> None:
    amps, labels, colours = [], [], []
    print(f"\n{'case':>26}  settled SS accel @ {F_SS_HZ} Hz [m/s^2]")
    for tag, lab, colour, *_ in CASES:
        fp = OUT_DIR / f"ss_{tag}.csv"
        if not fp.exists():
            print(f"{lab:>26}  (missing)")
            continue
        df = pd.read_csv(fp)
        if SS_COL not in df:
            print(f"{lab:>26}  (no {SS_COL})")
            continue
        t = df["t"].to_numpy()
        a = lockin(t, df[SS_COL].to_numpy(), F_SS_HZ, 0.6 * t.max(), t.max())
        amps.append(a)
        labels.append(lab)
        colours.append(colour)
        print(f"{lab:>26}  {a:.4f}")

    if not amps:
        print("No result CSVs; run without --plot-only first.")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = ax.bar(range(len(amps)), amps, color=colours, edgecolor="k", lw=0.6)
    for i, a in enumerate(amps):
        ax.text(i, a, f"{a:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Settled tower SS acceleration\nat 0.233 Hz  [m/s$^2$]")
    ax.set_title("Tower side-to-side resonance under a 0.233 Hz slug load\n"
                 "(OpenFAST, Region 3, SS-only)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "ss_resonance_cases.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these case tags (e.g. --only di di_notch)")
    args = ap.parse_args()

    if args.plot_only:
        analyze_and_plot()
        return

    selected = CASES
    if args.only:
        valid = {c[0] for c in CASES}
        bad = [t for t in args.only if t not in valid]
        if bad:
            print(f"Unknown case tag(s): {bad}. Valid: {sorted(valid)}")
            return
        selected = [c for c in CASES if c[0] in set(args.only)]

    pending = [c for c in selected
               if args.force or not (OUT_DIR / f"ss_{c[0]}.csv").exists()]
    est = len(pending) * T_END * 4.0 / 60.0
    print(f"SS-resonance cases: {len(CASES)} total, {len(pending)} to run "
          f"(~{est:.0f} min at t_end {T_END:.0f}s, fast_debug.fmu)")
    for c in CASES:
        mark = "RUN " if c in pending else "skip"
        print(f"  [{mark}] {c[0]:<12} droop={c[3]:.0e} inertia={c[4]:.0e} "
              f"notch={c[6]} Hz")
    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    for c in pending:
        run_case(c, force=args.force)
    analyze_and_plot()


if __name__ == "__main__":
    main()
