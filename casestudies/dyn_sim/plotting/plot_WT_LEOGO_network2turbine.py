r"""
Plots for the network -> turbine excitation study (Low-Emission angle).

Demonstrates that an electrical disturbance injected on the LEOGO side
(the synchronous-generator bus, ``--load-bus main``) reaches and excites the
wind-turbine drivetrain torsional mode at 3.49 Hz, just like a disturbance
applied directly at the WT terminal -- only attenuated by the intervening
network impedance.

Reads:
  * network-side sweep   : net2turb_main_<tag>Hz.csv  (this study)
  * direct WT-side sweep  : WT1_LEOGO_torsional_sweep_summary.csv (step 3)

Generates:
  1. WT_LEOGO_network2turbine_curve.png
        network-side resonance curve (peak-to-peak shaft torque vs forcing
        frequency), eigenfrequency marked, peak annotated.
  2. WT_LEOGO_network2turbine_compare.png
        direct WT-side vs network-side resonance curves on twin y-axes --
        same resonant shape, different amplitude (network attenuation).
  3. WT_LEOGO_network2turbine_normalized.png
        both curves normalized to their own peak: proves it is the SAME mode.
  4. WT_LEOGO_network2turbine_buildup.png
        network-side forcing (WT terminal power) and the resonant shaft-torque
        build-up at 3.49 Hz.

Headless (matplotlib Agg). Run:
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plotting\plot_WT_LEOGO_network2turbine.py
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

F_EIGEN_HZ = 3.4909       # torsional eigenfrequency (modal analysis)
ONSET_MAIN_S = 10.0       # forcing onset in the net2turb sweep runs
NM_TO_MNM = 1e-6

# network-side sweep (LEOGO genset bus injection)
MAIN_FREQS = [1.0, 2.0, 2.5, 3.0, 3.25, 3.49, 3.75, 4.5]


def _freq_tag(freq_hz: float) -> str:
    """Match the sim-script filename convention (e.g. 3.49 -> 3p49)."""
    return f"{freq_hz:.2f}Hz".replace(".", "p", 1)


def load_main_case(freq_hz: float) -> pd.DataFrame:
    path = CSV_DIR / f"net2turb_main_{_freq_tag(freq_hz)}.csv"
    return pd.read_csv(path)


def _post_onset_pp(df: pd.DataFrame, onset_s: float) -> float:
    m = df["t"].to_numpy() >= onset_s
    return float(np.ptp(df["T_shaft_Nm"].to_numpy()[m]))


def build_main_summary() -> pd.DataFrame:
    rows = []
    for f in MAIN_FREQS:
        df = load_main_case(f)
        rows.append({"forcing_freq_hz": f,
                     "T_shaft_pp_Nm": _post_onset_pp(df, ONSET_MAIN_S)})
    return pd.DataFrame(rows).sort_values("forcing_freq_hz")


def load_wt_summary() -> pd.DataFrame:
    df = pd.read_csv(CSV_DIR / "WT1_LEOGO_torsional_sweep_summary.csv")
    return df.sort_values("forcing_freq_hz")


def plot_network_curve(main: pd.DataFrame) -> None:
    f = main["forcing_freq_hz"].to_numpy()
    pp = main["T_shaft_pp_Nm"].to_numpy() * NM_TO_MNM
    i_peak = int(np.argmax(pp))
    f_peak, pp_peak = f[i_peak], pp[i_peak]

    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE)
    ax.plot(f, pp, marker="o", color=COLOR_COUPLED, markersize=4.5,
            markerfacecolor="white", markeredgecolor=COLOR_COUPLED,
            markeredgewidth=1.1, label="Response (LEOGO-bus forcing)")
    ax.axvline(F_EIGEN_HZ, color=COLOR_REF, linestyle="--", linewidth=1.0,
               label=f"Torsional eigenfrequency ({F_EIGEN_HZ:.2f} Hz)")
    ax.plot([f_peak], [pp_peak], marker="*", color=COLOR_BASELINE,
            markersize=13, zorder=5)
    ax.annotate(f"peak {pp_peak:.2f} MNm\nat {f_peak:.2f} Hz",
                xy=(f_peak, pp_peak),
                xytext=(f_peak + 0.35, pp_peak * 0.80),
                fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))

    style_time_axis(ax, xlabel="Electrical forcing frequency (Hz)")
    ax.set_ylabel("Peak-to-peak shaft torque (MNm)")
    ax.set_title("Network-side excitation of the WT drivetrain torsional mode")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    out = PLOT_DIR / "WT_LEOGO_network2turbine_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def plot_compare(main: pd.DataFrame, wt: pd.DataFrame) -> None:
    fm = main["forcing_freq_hz"].to_numpy()
    ppm = main["T_shaft_pp_Nm"].to_numpy() * NM_TO_MNM
    fw = wt["forcing_freq_hz"].to_numpy()
    ppw = wt["T_shaft_pp_Nm"].to_numpy() * NM_TO_MNM

    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE)
    l1, = ax.plot(fw, ppw, marker="s", color=COLOR_BASELINE, markersize=4.2,
                  markerfacecolor="white", markeredgecolor=COLOR_BASELINE,
                  markeredgewidth=1.1,
                  label="Direct WT-terminal forcing (left axis)")
    ax.set_ylabel("Shaft torque p2p, WT-terminal (MNm)", color=COLOR_BASELINE)
    ax.tick_params(axis="y", labelcolor=COLOR_BASELINE)
    ax.set_ylim(bottom=0.0)

    ax2 = ax.twinx()
    l2, = ax2.plot(fm, ppm, marker="o", color=COLOR_COUPLED, markersize=4.2,
                   markerfacecolor="white", markeredgecolor=COLOR_COUPLED,
                   markeredgewidth=1.1,
                   label="LEOGO-bus forcing (right axis)")
    ax2.set_ylabel("Shaft torque p2p, LEOGO-bus (MNm)", color=COLOR_COUPLED)
    ax2.tick_params(axis="y", labelcolor=COLOR_COUPLED)
    ax2.set_ylim(bottom=0.0)

    ax.axvline(F_EIGEN_HZ, color=COLOR_REF, linestyle="--", linewidth=1.0,
               zorder=0)
    style_time_axis(ax, xlabel="Electrical forcing frequency (Hz)")
    ax.set_title("Same torsional resonance, excited directly vs through the grid")
    ax.legend(handles=[l1, l2], loc="upper left", frameon=True, framealpha=0.9)

    out = PLOT_DIR / "WT_LEOGO_network2turbine_compare.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def plot_normalized(main: pd.DataFrame, wt: pd.DataFrame) -> None:
    fm = main["forcing_freq_hz"].to_numpy()
    ppm = main["T_shaft_pp_Nm"].to_numpy()
    fw = wt["forcing_freq_hz"].to_numpy()
    ppw = wt["T_shaft_pp_Nm"].to_numpy()

    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE)
    ax.plot(fw, ppw / ppw.max(), marker="s", color=COLOR_BASELINE,
            markersize=4.2, markerfacecolor="white",
            markeredgecolor=COLOR_BASELINE, markeredgewidth=1.1,
            label="Direct WT-terminal forcing")
    ax.plot(fm, ppm / ppm.max(), marker="o", color=COLOR_COUPLED,
            markersize=4.2, markerfacecolor="white",
            markeredgecolor=COLOR_COUPLED, markeredgewidth=1.1,
            label="LEOGO-bus forcing")
    ax.axvline(F_EIGEN_HZ, color=COLOR_REF, linestyle="--", linewidth=1.0,
               label=f"Torsional eigenfrequency ({F_EIGEN_HZ:.2f} Hz)")

    style_time_axis(ax, xlabel="Electrical forcing frequency (Hz)")
    ax.set_ylabel("Normalized shaft-torque response (-)")
    ax.set_title("Normalized resonance shape: identical mode from either bus")
    ax.set_ylim(0.0, 1.08)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    out = PLOT_DIR / "WT_LEOGO_network2turbine_normalized.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def plot_buildup() -> None:
    df = load_main_case(3.49)
    t = df["t"].to_numpy()
    p_e = df["P_e_wt_pu"].to_numpy()
    tau = df["T_shaft_Nm"].to_numpy() * NM_TO_MNM

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(THESIS_FIGSIZE[0], 4.6),
                                   sharex=True)
    ax1.plot(t, p_e, color=COLOR_WIND, linewidth=0.8)
    ax1.axvline(ONSET_MAIN_S, color="0.6", linestyle=":", linewidth=0.9)
    ax1.set_ylabel("WT terminal power (pu)")
    ax1.set_title("Network-side forcing at 3.49 Hz and the resonant shaft-torque build-up")
    style_time_axis(ax1, xlabel="")

    ax2.plot(t, tau, color=COLOR_COUPLED, linewidth=0.8)
    ax2.axvline(ONSET_MAIN_S, color="0.6", linestyle=":", linewidth=0.9)
    ax2.text(ONSET_MAIN_S + 0.3, ax2.get_ylim()[1] * 0.9, "forcing on",
             fontsize=7.5, color="0.4")
    ax2.set_ylabel("Shaft torque (MNm)")
    style_time_axis(ax2)

    out = PLOT_DIR / "WT_LEOGO_network2turbine_buildup.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    apply_thesis_td_style()
    main_summary = build_main_summary()
    wt_summary = load_wt_summary()

    plot_network_curve(main_summary)
    plot_compare(main_summary, wt_summary)
    plot_normalized(main_summary, wt_summary)
    plot_buildup()

    # quick attenuation report at the resonant frequency
    m349 = float(main_summary.loc[np.isclose(main_summary["forcing_freq_hz"], 3.49),
                                  "T_shaft_pp_Nm"].iloc[0])
    w349 = float(wt_summary.loc[np.isclose(wt_summary["forcing_freq_hz"], 3.49),
                                "T_shaft_pp_Nm"].iloc[0])
    print(f"\n@3.49 Hz: WT-bus {w349 * NM_TO_MNM:.3f} MNm, "
          f"LEOGO-bus {m349 * NM_TO_MNM:.3f} MNm, "
          f"network attenuation x{w349 / m349:.1f}")


if __name__ == "__main__":
    main()
