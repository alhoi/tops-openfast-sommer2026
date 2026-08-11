r"""
Drivetrain-torsion vs support LPF: does the closed frequency-support loop
excite the ~3.49 Hz drivetrain torsional mode, and does the support LPF protect
the drivetrain?

Sweeps a sustained platform-load sine near the drivetrain torsional frequency
(3.49 Hz) for two support configurations (same gains as the 5e6 tower study,
droop 2e7 + inertia 5e6, no notch):

  lpf05    - freq-LPF 0.5 Hz  (realistic): 3.49 Hz is blocked -> drivetrain protected
  lpfwide  - freq-LPF 10  Hz  (wide open): 3.49 Hz passes     -> torsion excited

Measures the settled high-speed-shaft-torque (fmu_HSShftTq) oscillation at the
drive frequency. Uses fast.fmu (exposes HSShftTq, faster). Resumable.

Usage:
  python casestudies/dyn_sim/drivetrain_torsion_lpf.py            # run + plot
  python casestudies/dyn_sim/drivetrain_torsion_lpf.py --plot-only
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
OUT_DIR = PROJECT_ROOT / "results" / "em_interaction_sweep" / "drivetrain_torsion_lpf"
OUT_PNG = OUT_DIR / "drivetrain_torsion_lpf.png"

TQ_COL = "fmu_HSShftTq"          # high-speed shaft torque (kN.m)
TORS_HZ = 3.49
T_END = 100.0
DROOP = 2.0e7
INERTIA = 5.0e6                  # matches the 5e6 tower study

# Fine points around the free-decay-identified torsional mode (~3.05-3.10 Hz).
# The wide-range points (3.0..4.0) already exist and are still globbed by the plot.
FREQS = [3.05, 3.10]
# (suffix, label, freq-LPF Hz, colour)
CONFIGS = [
    ("lpf05",   "support, LPF 0.5 Hz (realistic)", 0.5,  "#1f77b4"),
    ("lpfwide", "support, LPF wide-open (10 Hz)",  10.0, "#d62728"),
]


def _tag(f: float) -> str:
    return f"{f:.3f}".replace(".", "p")


def run_point(f: float, suffix: str, lpf: float, *, force: bool) -> Path:
    tag = _tag(f)
    out = OUT_DIR / f"tors_f{tag}_{suffix}.csv"
    if out.exists() and not force:
        print(f"  [skip] {out.name}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(DRIVER),
        "--t-end", f"{T_END}", "--fmu", "fast",
        "--wt-pref-mw", "12.88", "--zmq-grid", "--fix-leogo-xqt",
        "--droop-nm-per-hz", f"{DROOP:.0f}",
        "--inertia-nm-s-per-hz", f"{INERTIA:.0f}",
        "--freq-lpf-hz", f"{lpf}", "--freq-lpf-order", "2",
        "--support-notch-hz", "0.0", "--support-notch-q", "2.0",
        "--support-start", "5", "--support-max-nm", "15000000",
        "--load-step-mw", "3.0", "--event-time", "10",
        "--event-duration", f"{T_END - 10.0}", "--load-ramp-on-s", "3",
        "--load-sine-mean", "0.0", "--load-sine-amplitude", "1.0",
        "--load-sine-freq-hz", f"{f}",
        "--zmq-log", str((OUT_DIR / f"tors_f{tag}_{suffix}_zmq.csv").relative_to(PROJECT_ROOT)),
        "--out", str(out.relative_to(PROJECT_ROOT)),
    ]
    print(f"  [run ] tors f={f:.3f} Hz {suffix} (LPF {lpf} Hz) ...", flush=True)
    log = OUT_DIR / f"tors_f{tag}_{suffix}.log"
    with open(log, "w", encoding="utf-8") as fh:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT)
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
    return 2.0 * np.hypot(c, q) / span


def curve(suffix: str):
    pts = []
    for fp in OUT_DIR.glob(f"tors_f*_{suffix}.csv"):
        m = re.search(r"_f([0-9]+p[0-9]+)_", fp.name)
        if not m:
            continue
        f = float(m.group(1).replace("p", "."))
        df = pd.read_csv(fp)
        if TQ_COL not in df or "t" not in df:
            continue
        t = df["t"].to_numpy()
        a = lockin(t, df[TQ_COL].to_numpy(), f, 0.6 * t.max(), t.max())
        pts.append((f, a))
    return sorted(pts)


def plot() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for suffix, label, _lpf, colour in CONFIGS:
        pts = curve(suffix)
        if not pts:
            print(f"  {label:34s}: (no data)")
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                color=colour, lw=1.8, label=label)
        pk = max(pts, key=lambda p: p[1])
        print(f"  {label:34s}: peak {pk[1]:.3f} kN.m @ {pk[0]:.3f} Hz")
    ax.axvline(TORS_HZ, color="k", ls=":", lw=0.8)
    ax.text(TORS_HZ, ax.get_ylim()[1], f"  drivetrain torsion {TORS_HZ} Hz",
            fontsize=8, va="top", ha="left")
    ax.set_xlabel("Process-load frequency [Hz]")
    ax.set_ylabel("Settled shaft-torque oscillation  HSShftTq  [kN$\\cdot$m]")
    ax.set_title("The support LPF protects the drivetrain\n"
                 "(shaft-torque response near the torsional mode)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"\nSaved {OUT_PNG}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.plot_only:
        pending = [(f, suf, lpf) for (suf, _l, lpf, _c) in CONFIGS for f in FREQS]
        print(f"Drivetrain-torsion LPF sweep: {len(pending)} points "
              f"(fast.fmu, t_end {T_END:.0f}).")
        for f, suf, lpf in pending:
            run_point(f, suf, lpf, force=args.force)
    plot()


if __name__ == "__main__":
    main()
