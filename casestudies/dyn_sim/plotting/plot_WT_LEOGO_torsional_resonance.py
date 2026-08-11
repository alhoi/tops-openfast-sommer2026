r"""
Plots for step 3 -- electrically driven drivetrain torsional resonance.

Reads the CSV files produced by test_WT_LEOGO_torsional_resonance_sim.py and
generates thesis-ready figures:

  1. WT_LEOGO_torsional_resonance_curve.png
        peak-to-peak shaft torque vs electrical forcing frequency (resonance
        curve), with the torsional eigenfrequency marked.
  2. WT_LEOGO_torsional_timeseries_on_off.png
        shaft-torque time series, on-resonance (3.49 Hz) vs off-resonance
        (1.0 Hz), sharing the same forcing amplitude.
  3. WT_LEOGO_torsional_buildup.png
        two-panel: electrical forcing (terminal power) and the shaft-torque
        response at 3.49 Hz, showing onset and resonant build-up to saturation.

Headless (matplotlib Agg). Run:
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plotting\plot_WT_LEOGO_torsional_resonance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from thesis_plot_style import (  # noqa: E402
    COLOR_BASELINE,
    COLOR_COUPLED,
    COLOR_REF,
    COLOR_WIND,
    THESIS_FIGSIZE,
    apply_thesis_td_style,
    style_time_axis,
)

PROJECT_ROOT = THIS_DIR.parents[2]
CSV_DIR = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "csv_files"
PLOT_DIR = THIS_DIR / "plots" / "torsional_resonance"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

F_EIGEN_HZ = 3.4909      # torsional eigenfrequency (modal analysis)
ONSET_S = 5.0            # forcing switched on at this time in the sweep runs
NM_TO_MNM = 1e-6


def _freq_tag(freq_hz: float) -> str:
    """Match the filename convention of the sim script (e.g. 3.49 -> 3p49)."""
    return f"{freq_hz:.2f}Hz".replace(".", "p", 1)


def load_case(freq_hz: float) -> pd.DataFrame:
    path = CSV_DIR / f"WT1_LEOGO_torsional_forcing_{_freq_tag(freq_hz)}.csv"
    return pd.read_csv(path)


def plot_resonance_curve() -> None:
    summary = pd.read_csv(CSV_DIR / "WT1_LEOGO_torsional_sweep_summary.csv")
    summary = summary.sort_values("forcing_freq_hz")
    f = summary["forcing_freq_hz"].to_numpy()
    pp = summary["T_shaft_pp_Nm"].to_numpy() * NM_TO_MNM

    i_peak = int(np.argmax(pp))
    f_peak, pp_peak = f[i_peak], pp[i_peak]

    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE)
    ax.plot(f, pp, marker="o", color=COLOR_COUPLED, markersize=4.5,
            markerfacecolor="white", markeredgecolor=COLOR_COUPLED,
            markeredgewidth=1.1, label="Simulated response")
    ax.axvline(F_EIGEN_HZ, color=COLOR_REF, linestyle="--", linewidth=1.0,
               label=f"Torsional eigenfrequency ({F_EIGEN_HZ:.2f} Hz)")
    ax.plot([f_peak], [pp_peak], marker="*", color=COLOR_BASELINE,
            markersize=13, zorder=5)
    ax.annotate(f"peak {pp_peak:.2f} MNm\nat {f_peak:.2f} Hz",
                xy=(f_peak, pp_peak),
                xytext=(f_peak + 0.35, pp_peak * 0.82),
                fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))

    style_time_axis(ax, xlabel="Electrical forcing frequency (Hz)")
    ax.set_ylabel("Peak-to-peak shaft torque (MNm)")
    ax.set_title("Electrically driven drivetrain torsional resonance")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    out = PLOT_DIR / "WT_LEOGO_torsional_resonance_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def plot_timeseries_on_off() -> None:
    on = load_case(3.49)
    off = load_case(1.0)

    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE)
    ax.plot(off["t"], off["T_shaft_Nm"] * NM_TO_MNM,
            color=COLOR_BASELINE, linewidth=0.9,
            label="Off-resonance (1.0 Hz)")
    ax.plot(on["t"], on["T_shaft_Nm"] * NM_TO_MNM,
            color=COLOR_COUPLED, linewidth=0.9,
            label="On-resonance (3.49 Hz)")
    ax.axvline(ONSET_S, color="0.6", linestyle=":", linewidth=0.9)
    ax.text(ONSET_S + 0.3, ax.get_ylim()[1] * 0.9, "forcing on",
            fontsize=7.5, color="0.4")

    style_time_axis(ax)
    ax.set_ylabel("Shaft torque (MNm)")
    ax.set_title("Shaft-torque response: on- vs off-resonance electrical forcing")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, ncol=2)

    out = PLOT_DIR / "WT_LEOGO_torsional_timeseries_on_off.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def plot_buildup() -> None:
    on = load_case(3.49)
    t = on["t"].to_numpy()

    fig, (ax_f, ax_t) = plt.subplots(
        2, 1, sharex=True, figsize=(THESIS_FIGSIZE[0], THESIS_FIGSIZE[1] * 1.35)
    )

    # Top: electrical forcing seen at the turbine (terminal power deviation).
    p_wt = on["P_e_wt_pu"].to_numpy()
    ax_f.plot(t, p_wt, color=COLOR_WIND, linewidth=0.85)
    style_time_axis(ax_f, xlabel="")
    ax_f.set_ylabel("WT electrical\npower (pu)")
    ax_f.set_title("Electrical forcing at 3.49 Hz and resonant shaft-torque build-up")

    # Bottom: shaft-torque response.
    ax_t.plot(t, on["T_shaft_Nm"].to_numpy() * NM_TO_MNM,
              color=COLOR_COUPLED, linewidth=0.85)
    style_time_axis(ax_t)
    ax_t.set_ylabel("Shaft torque (MNm)")

    for ax in (ax_f, ax_t):
        ax.axvline(ONSET_S, color="0.6", linestyle=":", linewidth=0.9)

    out = PLOT_DIR / "WT_LEOGO_torsional_buildup.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    apply_thesis_td_style()
    plot_resonance_curve()
    plot_timeseries_on_off()
    plot_buildup()
    print(f"\nAll figures written to {PLOT_DIR}")


if __name__ == "__main__":
    main()
