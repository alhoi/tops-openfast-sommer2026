r"""
Plots for step 4 (network -> turbine, realistic events): a LEOGO process-load
event exciting the wind-turbine drivetrain torsional mode.

Reads the CSVs produced by test_WT_LEOGO_process_load_excitation_sim.py and
generates thesis-ready figures:

  1. WT_LEOGO_process_pulsation_response.png
        two-panel: the small process-load pulsation at the PCC (top) and the
        resonant shaft-torque build-up it drives (bottom).
  2. WT_LEOGO_process_pulsation_propagation.png
        the disturbance travelling PCC -> turbine: Main Bus A voltage, WT
        terminal power and shaft torque on a common time axis.
  3. WT_LEOGO_process_step_ringdown.png
        two-panel: the full shaft-torque response to a discrete load-switching
        event (top) and the high-pass-isolated 3.49 Hz torsional ring-down with
        a fitted decay envelope (bottom), confirming the modal damping.
  4. WT_LEOGO_process_pulsation_vs_step.png
        two-panel comparison of the two excitation mechanisms on a common
        time-since-onset axis: the sustained forced response of the periodic
        pulsation (top) versus the decaying ring-down of the discrete
        load-switching event (bottom), both isolated to the torsional band.

Headless (matplotlib Agg). Run:
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plotting\plot_WT_LEOGO_process_load_excitation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

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

F_EIGEN_HZ = 3.4909
ONSET_S = 10.0
NM_TO_MNM = 1e-6
NM_TO_KNM = 1e-3

PULSATION_CSV = CSV_DIR / "WT1_LEOGO_process_pulsation_3p49Hz_0p50MW.csv"
STEP_CSV = CSV_DIR / "WT1_LEOGO_process_step_9p70MW.csv"


# Same-amplitude (0.5 MW) pulsations at different frequencies, used to show
# frequency selectivity: only the disturbance at the 3.49 Hz eigenfrequency
# rings the drivetrain torsional mode.
SELECTIVITY_CSVS = [
    (2.00, CSV_DIR / "WT1_LEOGO_process_pulsation_2p00Hz_0p50MW.csv"),
    (3.00, CSV_DIR / "WT1_LEOGO_process_pulsation_3p00Hz_0p50MW.csv"),
    (F_EIGEN_HZ, PULSATION_CSV),
]


def _highpass(y: np.ndarray, dt: float, f_cut: float = 1.5) -> np.ndarray:
    b, a = butter(2, f_cut / (0.5 / dt), btype="highpass")
    return filtfilt(b, a, y)


def plot_pulsation_response() -> None:
    df = pd.read_csv(PULSATION_CSV)
    t = df["t"].to_numpy()
    load = df["load_mw"].to_numpy()
    tau = df["T_shaft_Nm"].to_numpy() * NM_TO_KNM
    tau0 = np.mean(tau[t < ONSET_S])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(THESIS_FIGSIZE[0], 4.6),
                                   sharex=True)
    ax1.plot(t, load, color=COLOR_WIND, linewidth=0.9)
    ax1.axvline(ONSET_S, color="0.6", linestyle=":", linewidth=0.9)
    ax1.set_ylabel("Process-load\npulsation (MW)")
    ax1.set_title("A 0.5 MW process-load pulsation at 3.49 Hz drives the drivetrain torsional mode")
    style_time_axis(ax1, xlabel="")

    ax2.plot(t, tau - tau0, color=COLOR_COUPLED, linewidth=0.8)
    ax2.axvline(ONSET_S, color="0.6", linestyle=":", linewidth=0.9)
    ax2.text(ONSET_S + 0.3, ax2.get_ylim()[1] * 0.88, "pulsation on",
             fontsize=7.5, color="0.4")
    ax2.set_ylabel("Shaft-torque\ndeviation (kNm)")
    style_time_axis(ax2)

    out = PLOT_DIR / "WT_LEOGO_process_pulsation_response.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def plot_pulsation_propagation() -> None:
    df = pd.read_csv(PULSATION_CSV)
    t = df["t"].to_numpy()
    # zoom to a few cycles after steady state to show the phase chain
    m = (t >= 30.0) & (t <= 31.5)
    tt = t[m]
    v_main = df["V_mainbus_pu"].to_numpy()[m]
    p_wt = df["P_e_wt_pu"].to_numpy()[m]
    tau = df["T_shaft_Nm"].to_numpy()[m] * NM_TO_MNM

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(THESIS_FIGSIZE[0], 5.2),
                                        sharex=True)
    ax1.plot(tt, v_main, color=COLOR_WIND, linewidth=1.0)
    ax1.set_ylabel("Main Bus A\nvoltage (pu)")
    ax1.set_title("Disturbance propagation PCC $\\rightarrow$ turbine (steady-state cycles at 3.49 Hz)")
    style_time_axis(ax1, xlabel="")

    ax2.plot(tt, p_wt, color=COLOR_BASELINE, linewidth=1.0)
    ax2.set_ylabel("WT terminal\npower (pu)")
    style_time_axis(ax2, xlabel="")

    ax3.plot(tt, tau, color=COLOR_COUPLED, linewidth=1.0)
    ax3.set_ylabel("Shaft\ntorque (MNm)")
    style_time_axis(ax3)

    out = PLOT_DIR / "WT_LEOGO_process_pulsation_propagation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def plot_step_ringdown() -> None:
    df = pd.read_csv(STEP_CSV)
    t = df["t"].to_numpy()
    tau = df["T_shaft_Nm"].to_numpy() * NM_TO_KNM
    tau0 = np.mean(tau[t < ONSET_S])
    dt = float(np.median(np.diff(t)))

    post = t >= ONSET_S
    tp = t[post]
    ring = _highpass(tau[post] - tau0, dt, f_cut=1.5)
    edge = int(0.5 / dt)
    tr, rr = tp[edge:-edge], ring[edge:-edge]

    # exponential envelope fit on decaying peaks (for annotation)
    a = np.abs(rr)
    idx = np.where((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:]))[0] + 1
    zeta_pct = float("nan")
    env_t = env_a = None
    if len(idx) >= 3:
        tpk, apk = tr[idx], a[idx]
        kmax = int(np.argmax(apk))
        tpk, apk = tpk[kmax:], apk[kmax:]
        keep = apk > 0.05 * apk[0]
        tpk, apk = tpk[keep], apk[keep]
        if len(tpk) >= 3:
            slope, inter = np.polyfit(tpk - tpk[0], np.log(apk), 1)
            sigma = -slope
            wn = 2.0 * np.pi * F_EIGEN_HZ
            zeta_pct = 100.0 * sigma / np.sqrt(wn**2 + sigma**2)
            env_t = tpk
            env_a = np.exp(inter) * np.exp(slope * (tpk - tpk[0]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(THESIS_FIGSIZE[0], 4.8),
                                   sharex=True)
    ax1.plot(t, tau - tau0, color=COLOR_COUPLED, linewidth=0.8)
    ax1.axvline(ONSET_S, color="0.6", linestyle=":", linewidth=0.9)
    ax1.text(ONSET_S + 0.2, ax1.get_ylim()[1] * 0.85,
             "9.7 MW load\nswitching event", fontsize=7.5, color="0.4")
    ax1.set_ylabel("Shaft-torque\ndeviation (kNm)")
    ax1.set_title("A discrete LEOGO load-switching event rings the WT torsional mode")
    style_time_axis(ax1, xlabel="")

    ax2.plot(tr, rr, color=COLOR_BASELINE, linewidth=0.8,
             label="Isolated torsional ring (>1.5 Hz)")
    if env_t is not None:
        ax2.plot(env_t, env_a, color=COLOR_REF, linestyle="--", linewidth=1.1,
                 label=f"Fitted decay ($\\zeta$ = {zeta_pct:.2f} %)")
        ax2.plot(env_t, -env_a, color=COLOR_REF, linestyle="--", linewidth=1.1)
    ax2.set_ylabel("Torsional ring\n(kNm)")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=7.5)
    style_time_axis(ax2)

    out = PLOT_DIR / "WT_LEOGO_process_step_ringdown.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}  (ring-down zeta = {zeta_pct:.2f} %)")


def _ringdown_zeta_env(tr, rr):
    """Return (zeta_pct, env_t, env_a) from a log-decrement fit, or NaNs."""
    a = np.abs(rr)
    idx = np.where((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:]))[0] + 1
    if len(idx) < 3:
        return float("nan"), None, None
    tpk, apk = tr[idx], a[idx]
    kmax = int(np.argmax(apk))
    tpk, apk = tpk[kmax:], apk[kmax:]
    keep = apk > 0.05 * apk[0]
    tpk, apk = tpk[keep], apk[keep]
    if len(tpk) < 3:
        return float("nan"), None, None
    slope, inter = np.polyfit(tpk - tpk[0], np.log(apk), 1)
    sigma = -slope
    wn = 2.0 * np.pi * F_EIGEN_HZ
    zeta_pct = 100.0 * sigma / np.sqrt(wn**2 + sigma**2)
    env_a = np.exp(inter) * np.exp(slope * (tpk - tpk[0]))
    return zeta_pct, tpk, env_a


def plot_pulsation_vs_step() -> None:
    """Side-by-side of the two excitation mechanisms on a time-since-onset axis:
    a sustained forced response (pulsation) versus a decaying ring (step)."""
    dfp = pd.read_csv(PULSATION_CSV)
    tp_all = dfp["t"].to_numpy()
    taup = dfp["T_shaft_Nm"].to_numpy() * NM_TO_KNM
    taup0 = np.mean(taup[tp_all < ONSET_S])
    dtp = float(np.median(np.diff(tp_all)))
    mp = tp_all >= ONSET_S
    tp = tp_all[mp] - ONSET_S
    ringp = _highpass(taup[mp] - taup0, dtp, f_cut=1.5)

    dfs = pd.read_csv(STEP_CSV)
    ts_all = dfs["t"].to_numpy()
    taus = dfs["T_shaft_Nm"].to_numpy() * NM_TO_KNM
    taus0 = np.mean(taus[ts_all < ONSET_S])
    dts = float(np.median(np.diff(ts_all)))
    ms = ts_all >= ONSET_S
    ts = ts_all[ms] - ONSET_S
    rings = _highpass(taus[ms] - taus0, dts, f_cut=1.5)

    # trim filter edges so both start cleanly
    ep, es = int(0.5 / dtp), int(0.5 / dts)
    tp, ringp = tp[ep:-ep], ringp[ep:-ep]
    ts, rings = ts[es:-es], rings[es:-es]

    zeta_pct, env_t, env_a = _ringdown_zeta_env(ts, rings)
    pul_amp = float(np.percentile(
        np.abs(ringp[tp > tp[-1] - 0.4 * (tp[-1] - tp[0])]), 95))

    tmax = min(tp[-1], ts[-1])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(THESIS_FIGSIZE[0], 4.8),
                                   sharex=True)

    ax1.plot(tp, ringp, color=COLOR_COUPLED, linewidth=0.8,
             label="Torsional response")
    ax1.axhline(pul_amp, color=COLOR_REF, linestyle="--", linewidth=1.0,
                label=f"Sustained $\\pm${pul_amp:.0f} kNm")
    ax1.axhline(-pul_amp, color=COLOR_REF, linestyle="--", linewidth=1.0)
    ax1.set_ylabel("Shaft-torque\ndeviation (kNm)")
    ax1.set_title("Periodic pulsation (0.5 MW @ 3.49 Hz): sustained forced response")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=7.5)
    style_time_axis(ax1, xlabel="")

    ax2.plot(ts, rings, color=COLOR_BASELINE, linewidth=0.8,
             label="Torsional ring")
    if env_t is not None:
        ax2.plot(env_t, env_a, color=COLOR_REF, linestyle="--", linewidth=1.1,
                 label=f"Decay ($\\zeta$ = {zeta_pct:.2f} %)")
        ax2.plot(env_t, -env_a, color=COLOR_REF, linestyle="--", linewidth=1.1)
    ax2.set_ylabel("Shaft-torque\ndeviation (kNm)")
    ax2.set_title("Discrete 9.7 MW switching event: transient ring-down")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=7.5)
    style_time_axis(ax2, xlabel="Time since event (s)")
    ax2.set_xlim(0.0, tmax)

    out = PLOT_DIR / "WT_LEOGO_process_pulsation_vs_step.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}  (pulsation +/-{pul_amp:.0f} kNm sustained, "
          f"step ring-down zeta = {zeta_pct:.2f} %)")


def _steady_amp_knm(t: np.ndarray, y: np.ndarray) -> float:
    """95th-percentile |y| over the last 40 % of the post-onset window."""
    tail = t > (t[-1] - 0.4 * (t[-1] - ONSET_S))
    return float(np.percentile(np.abs(y[tail]), 95))


def plot_frequency_selectivity() -> None:
    """Same 0.5 MW pulsation at 2.0 / 3.0 / 3.49 Hz on a shared torque axis:
    only the disturbance at the eigenfrequency rings the mode. Visual proof that
    the grid does not couple broadband into the drivetrain -- the response is
    sharply frequency-selective."""
    fig, axes = plt.subplots(len(SELECTIVITY_CSVS), 1, sharex=True, sharey=True,
                             figsize=(THESIS_FIGSIZE[0], 5.4))
    amps = []
    for ax, (f_hz, csv) in zip(axes, SELECTIVITY_CSVS):
        df = pd.read_csv(csv)
        t = df["t"].to_numpy()
        tau = df["T_shaft_Nm"].to_numpy() * NM_TO_KNM
        tau0 = np.mean(tau[t < ONSET_S])
        y = tau - tau0
        amp = _steady_amp_knm(t, y)
        amps.append((f_hz, amp))
        on_res = abs(f_hz - F_EIGEN_HZ) < 0.05
        color = COLOR_COUPLED if on_res else COLOR_BASELINE
        ax.plot(t, y, color=color, linewidth=0.8)
        ax.axvline(ONSET_S, color="0.6", linestyle=":", linewidth=0.9)
        tag = "ON resonance" if on_res else "off resonance"
        ax.text(0.995, 0.90,
                f"{f_hz:.2f} Hz  ({tag}):  steady $\\pm${amp:.0f} kNm",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color,
                          alpha=0.85, lw=0.8))
        ax.set_ylabel("$\\Delta T_{\\mathrm{sh}}$\n(kNm)")
        style_time_axis(ax, xlabel="" if ax is not axes[-1] else None)
    axes[0].set_title("Same 0.5 MW PCC pulsation at three frequencies: "
                      "only 3.49 Hz rings the mode")

    out = PLOT_DIR / "WT_LEOGO_process_frequency_selectivity.png"
    fig.savefig(out)
    plt.close(fig)
    ratio = amps[-1][1] / max(amps[0][1], 1e-9)
    print(f"saved {out}  (3.49 Hz +/-{amps[-1][1]:.0f} kNm vs "
          f"2.0 Hz +/-{amps[0][1]:.0f} kNm -> x{ratio:.1f})")


def main() -> None:
    apply_thesis_td_style()
    plot_pulsation_response()
    plot_pulsation_propagation()
    plot_step_ringdown()
    plot_pulsation_vs_step()
    plot_frequency_selectivity()


if __name__ == "__main__":
    main()
