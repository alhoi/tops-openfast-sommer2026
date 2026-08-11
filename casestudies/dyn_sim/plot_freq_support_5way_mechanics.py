r"""
Mechanical response of the OpenFAST turbine during the de-loaded gas-turbine-trip
frequency-support runs, for the five capped-burst cases shown in
freq_support_5way_cap.png (no support, droop, droop + notch, droop + inertia,
droop + inertia + notch), with the inertia burst capped (--max-over-nm -3e6).

Plots the internal FMU mechanics (rotor speed, blade pitch, generator torque and
high-speed-shaft torque) so the mechanical cost of each support flavour is
visible: the de-loaded rotor speed, the deceleration/de-pitch that releases the
reserve at the event, and the shaft-torque swing.

Usage:
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_freq_support_5way_mechanics.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "results" / "em_interaction_sweep" / "freq_support_3way"
EVENT_TIME = 20.0
RATED_RPM = 7.56
T_LO, T_HI = 0.0, 60.0
SETTLE = 0.5   # drop the FMU priming glitch in the first samples (t < SETTLE)

# Same cases and colours as freq_support_5way_cap.png (inertia burst capped).
CASES = [
    ("deload_none",                "no support",               "#7f7f7f", "--"),
    ("deload_droop",               "droop",                    "#1f77b4", "-"),
    ("deload_droop_q8",            "droop + notch",            "#ff7f0e", "-"),
    ("deload_droop_inertia_cap",   "droop + inertia",          "#2ca02c", "-"),
    ("deload_droop_inertia_capq8", "droop + inertia + notch",  "#9467bd", "-"),
]

# (column, axis label, optional reference line)
PANELS = [
    ("fmu_RotSpeed", "Rotor speed\n[rpm]", RATED_RPM),
    ("fmu_BldPitch1", "Blade pitch\n[deg]", None),
    ("fmu_GenTq", "Generator torque\n[kN$\\cdot$m]", None),
    ("fmu_HSShftTq", "Shaft torque\n[kN$\\cdot$m]", None),
]


def main() -> None:
    frames = {}
    for tag, *_ in CASES:
        fp = DATA / f"{tag}.csv"
        if fp.exists():
            frames[tag] = pd.read_csv(fp)
        else:
            print(f"  (missing {fp.name})")
    if not frames:
        print(f"No CSVs in {DATA}; run freq_support_3way.py first.")
        return

    panels = [p for p in PANELS
              if any(p[0] in df.columns for df in frames.values())]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9.5, 2.2 * len(panels)),
                             sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (col, ylabel, ref) in zip(axes, panels):
        for tag, lab, colour, ls in CASES:
            df = frames.get(tag)
            if df is None or col not in df.columns:
                continue
            m = (df["t"] >= max(T_LO, SETTLE)) & (df["t"] <= T_HI)
            lw = 1.2 if ls == "--" else 1.4
            ax.plot(df["t"][m], df[col][m], color=colour, lw=lw, ls=ls,
                    label=lab)
        if ref is not None:
            ax.axhline(ref, color="k", ls="--", lw=0.8)
            ax.text(T_HI, ref, " rated", va="bottom", ha="right",
                    fontsize=8, color="k")
        ax.axvline(EVENT_TIME, color="k", ls=":", lw=0.8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].set_xlim(T_LO, T_HI)
    axes[0].legend(loc="lower right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Turbine mechanics during a de-loaded gas-turbine-trip "
                 "frequency-support event, capped burst\n(OpenFAST, Region 3)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = DATA / "freq_support_5way_mechanics.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
