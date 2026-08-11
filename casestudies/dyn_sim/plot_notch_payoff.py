r"""
"The notch is (almost) free" payoff figure.

Two scenarios, side by side, with consistent colours (grey / green / purple):

  LEFT  - Grid-frequency event (gas-turbine trip, freq_support_3way data).
          f_grid_hz vs time. droop+inertia (green) and droop+inertia+notch
          (purple) nearly overlap and sit far above "no support" (grey):
          the notch keeps the grid-frequency benefit (nadir / RoCoF).

  RIGHT - Sustained platform load at the tower SS frequency (0.233 Hz,
          ss_resonance data). Tower-top SS acceleration vs time. Here
          droop+inertia (green) rings up a large resonance while
          droop+inertia+notch (purple) collapses back onto the "no support"
          floor (grey): the notch removes the tower resonance.

So green is good on the grid (left) but bad for the tower (right); purple is
good in BOTH -> the notch buys the tower protection at essentially no grid cost.

Reads existing CSVs only (no simulation).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEEP = PROJECT_ROOT / "results" / "em_interaction_sweep"
GT_DIR = SWEEP / "freq_support_3way"
SS_DIR = SWEEP / "ss_resonance"
OUT = SWEEP / "notch_payoff.png"

GREY, GREEN, PURPLE = "#7f7f7f", "#2ca02c", "#9467bd"

# (stem, label, colour, linestyle)
GT_CASES = [
    ("deload_none",                "no support",              GREY,   "--"),
    ("deload_droop_inertia_cap",   "droop + inertia",         GREEN,  "-"),
    ("deload_droop_inertia_capq8", "droop + inertia + notch", PURPLE, "-"),
]
SS_CASES = [
    ("ss_none",     "no support",              GREY,   "--"),
    ("ss_di",       "droop + inertia",         GREEN,  "-"),
    ("ss_di_notch", "droop + inertia + notch", PURPLE, "-"),
]

EVENT_TIME = 20.0
SS_HZ = 0.233
SS_WIN = (300.0, 350.0)   # settled window for the resonance panel


def _load(d: Path, stem: str) -> pd.DataFrame | None:
    fp = d / f"{stem}.csv"
    if not fp.exists():
        print(f"  (missing {fp.name})")
        return None
    return pd.read_csv(fp)


def _nadir(df: pd.DataFrame) -> float:
    t = df["t"].to_numpy()
    f = df["f_grid_hz"].to_numpy()
    pre = f[(t >= EVENT_TIME - 5) & (t < EVENT_TIME)].mean()
    return float(np.min(f[t >= EVENT_TIME]) - pre) * 1000.0


def _ss_amp(df: pd.DataFrame) -> float:
    t = df["t"].to_numpy()
    y = df["fmu_YawBrTAyp"].to_numpy()
    m = (t >= SS_WIN[0]) & (t <= SS_WIN[1])
    return float(np.sqrt(2.0) * np.std(y[m]))


def main() -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # LEFT: grid-frequency event
    print("Grid event (nadir, mHz):")
    for stem, label, colour, ls in GT_CASES:
        df = _load(GT_DIR, stem)
        if df is None:
            continue
        t = df["t"].to_numpy()
        lw = 1.4 if ls == "--" else 1.8
        axL.plot(t, df["f_grid_hz"], color=colour, lw=lw, ls=ls, label=label)
        print(f"  {label:26s}: {_nadir(df):+7.1f}")
    axL.axvline(EVENT_TIME, color="k", ls=":", lw=0.8)
    axL.set_xlim(0, 60)
    axL.set_xlabel("Time [s]")
    axL.set_ylabel("Grid frequency  $f_{grid}$  [Hz]")
    axL.set_title("Grid-frequency event (gas-turbine trip)\n"
                  "notch keeps the support benefit")
    axL.grid(True, alpha=0.3)
    axL.legend(fontsize=8, loc="upper right")

    # RIGHT: sustained tower-frequency load
    print("\nResonance (settled SS accel amplitude, m/s^2):")
    for stem, label, colour, ls in SS_CASES:
        df = _load(SS_DIR, stem)
        if df is None:
            continue
        t = df["t"].to_numpy()
        m = (t >= SS_WIN[0]) & (t <= SS_WIN[1])
        lw = 1.4 if ls == "--" else 1.6
        axR.plot(t[m], df["fmu_YawBrTAyp"].to_numpy()[m], color=colour,
                 lw=lw, ls=ls, label=label)
        print(f"  {label:26s}: {_ss_amp(df):.4f}")
    axR.set_xlim(*SS_WIN)
    axR.set_xlabel("Time [s]")
    axR.set_ylabel("Tower-top SS acceleration  [m/s$^2$]")
    axR.set_title(f"Sustained platform load at {SS_HZ} Hz\n"
                  "notch removes the tower resonance")
    axR.grid(True, alpha=0.3)
    axR.legend(fontsize=8, loc="upper right")

    fig.suptitle("The notch is (almost) free: same grid-frequency support, "
                 "no tower side-to-side resonance", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
