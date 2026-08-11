r"""
Combined 5e6 figure: SS notch resonance, sweep (left) + time series (right).

Both panels use the SAME inertia gain (5e6) and the same three cases/colours,
so they are directly comparable:

  LEFT  - Frequency sweep of settled tower-top SS acceleration vs process-load
          frequency (full_matrix/ss/ss_f*_{off,on,onN}.csv, inertia 5e6).
          Same content as ss_notch_resonance_5e6_old.png.

  RIGHT - Time series under a sustained platform load at 0.233 Hz, inertia 5e6
          (ss_resonance/ss_none, ss_di5e6, ss_di5e6_notch).

Reads existing CSVs only (no simulation). New filename (does not overwrite).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEEP = PROJECT_ROOT / "results" / "em_interaction_sweep"
SS_SWEEP_DIR = SWEEP / "full_matrix" / "ss"
SS_TIME_DIR = SWEEP / "ss_resonance"
OUT = SWEEP / "full_matrix" / "ss_notch_combined_5e6.png"

SS_COL = "fmu_YawBrTAyp"
SS_HZ = 0.233
SS_WIN = (300.0, 350.0)

GREY, GREEN, PURPLE = "#7f7f7f", "#2ca02c", "#9467bd"

# LEFT sweep suffixes (5e6): off / on / onN
SWEEP_SERIES = [
    ("off", "no support",              GREY,   "--", "o"),
    ("on",  "droop + inertia",         GREEN,  "-",  "o"),
    ("onN", "droop + inertia + notch", PURPLE, "-",  "s"),
]
# RIGHT time-series stems (5e6)
TIME_SERIES = [
    ("ss_none",        "no support",              GREY,   "--"),
    ("ss_di5e6",       "droop + inertia",         GREEN,  "-"),
    ("ss_di5e6_notch", "droop + inertia + notch", PURPLE, "-"),
]


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


def curve(sup: str):
    pts = []
    for fp in SS_SWEEP_DIR.glob(f"ss_f*_{sup}.csv"):
        m = re.search(r"_f([0-9]+p[0-9]+)_", fp.name)
        if not m:
            continue
        f = float(m.group(1).replace("p", "."))
        df = pd.read_csv(fp)
        if SS_COL not in df or "t" not in df:
            continue
        t = df["t"].to_numpy()
        a = lockin(t, df[SS_COL].to_numpy(), f, 0.6 * t.max(), t.max())
        pts.append((f, a))
    return sorted(pts)


def main() -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    # LEFT: frequency sweep (5e6)
    print("Sweep peaks (5e6):")
    for sup, label, colour, ls, mk in SWEEP_SERIES:
        pts = curve(sup)
        if not pts:
            print(f"  {label:26s}: (no data)")
            continue
        lw = 1.3 if ls == "--" else 1.8
        axL.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk,
                 color=colour, lw=lw, ls=ls, label=label)
        pk = max(pts, key=lambda p: p[1])
        print(f"  {label:26s}: peak {pk[1]:.4f} @ {pk[0]:.3f} Hz")
    axL.axvline(SS_HZ, color="k", ls=":", lw=0.8)
    axL.set_xlabel("Process-load frequency [Hz]")
    axL.set_ylabel("Settled tower-top SS acceleration [m/s$^2$]")
    axL.set_title("Frequency sweep (inertia 5e6)")
    axL.grid(True, alpha=0.3)
    axL.legend(fontsize=8, loc="upper right")

    # RIGHT: time series at 0.233 Hz (5e6)
    print("\nSettled SS amplitude at 0.233 Hz (5e6):")
    for stem, label, colour, ls in TIME_SERIES:
        fp = SS_TIME_DIR / f"{stem}.csv"
        if not fp.exists():
            print(f"  {label:26s}: (missing {fp.name})")
            continue
        df = pd.read_csv(fp)
        t = df["t"].to_numpy()
        m = (t >= SS_WIN[0]) & (t <= SS_WIN[1])
        y = df[SS_COL].to_numpy()
        lw = 1.3 if ls == "--" else 1.6
        axR.plot(t[m], y[m], color=colour, lw=lw, ls=ls, label=label)
        amp = np.sqrt(2.0) * np.std(y[m])
        print(f"  {label:26s}: {amp:.4f}")
    axR.set_xlim(*SS_WIN)
    axR.set_xlabel("Time [s]")
    axR.set_ylabel("Tower-top SS acceleration [m/s$^2$]")
    axR.set_title(f"Sustained platform load at {SS_HZ} Hz (inertia 5e6)")
    axR.grid(True, alpha=0.3)
    axR.legend(fontsize=8, loc="upper right")

    fig.suptitle("Tower SS resonance at inertia 5e6: frequency sweep (left) "
                 "and time series (right)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
