r"""
Poster figure for the GENUINE OpenFAST drivetrain-torsion resonance case.

Unlike the co-simulation wrapper shaft (a numerical/sampled-data artifact),
this figure uses the high-fidelity OpenFAST drivetrain torsion DOF (ElastoDyn
DrTrDOF = True, softened DTTorSpr so the mode sits at ~3.10 Hz with ~5% damping).
The mode is excited entirely inside OpenFAST through the ROSCO open-loop
generator-torque channel (OL_Mode = 1) -- the only torque path that reaches the
turbine structure -- and its response is read from the OpenFAST high-speed-shaft
torque output HSShftTq. Driving on the mode (3.10 Hz) versus off the mode
(2.00 Hz) shows a clean, bounded, frequency-selective forced resonance.

Reads two LEOGO + OpenFAST FMU runs:
  * on-resonance  : ROSCO open-loop torque ripple at 3.10 Hz
  * off-resonance : same ripple amplitude at 2.00 Hz (control)

and draws a 4-panel poster figure:

  (top)      the shared applied generator-torque ripple [kNm]
  (large)    HSShftTq deviation vs time: on-resonance build-up vs off-resonance
  (bottom L) post-onset HSShftTq spectrum, on vs off, 3.10 Hz mode marked
  (bottom R) zoom on the settled on-resonance shaft-torque ripple

Usage
-----
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_fmu_genuine_torsion.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_fmu_genuine_torsion.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = PROJECT_ROOT / "results" / "em_interaction_sweep"

F_TORSION_HZ = 3.10        # OpenFAST drivetrain torsion eigenfrequency
T0_KNM = 11023.871         # Region-2 (8 m/s) operating-point generator torque

C_ON = "#b9770e"           # on-resonance (amber)
C_OFF = "#8a8f98"          # off-resonance control (grey)
C_LOAD = "#2c3e50"         # applied torque ripple (dark)

CHANNEL = "fmu_HSShftTq"   # OpenFAST high-speed-shaft torque [kNm]


def _shaft_dev_knm(df: pd.DataFrame, onset: float) -> np.ndarray:
    """HSShftTq as a deviation [kNm] from its pre-onset steady value."""
    t = df["t"].to_numpy()
    tq = df[CHANNEL].to_numpy()
    base = float(np.mean(tq[t < onset])) if np.any(t < onset) else float(tq[0])
    return tq - base


def _applied_ripple_knm(t: np.ndarray, freq: float, amp: float,
                        onset: float) -> np.ndarray:
    """Reconstruct the ROSCO open-loop generator-torque ripple [kNm]."""
    return np.where(t >= onset,
                    T0_KNM * amp * np.sin(2.0 * np.pi * freq * (t - onset)),
                    0.0)


def _post_onset(t: np.ndarray, y: np.ndarray, onset: float):
    m = t >= onset
    return t[m], y[m]


def _spectrum(t: np.ndarray, y: np.ndarray):
    dt = float(np.median(np.diff(t)))
    win = np.hanning(len(y))
    spec = np.abs(np.fft.rfft((y - np.mean(y)) * win)) * (2.0 / np.sum(win))
    freqs = np.fft.rfftfreq(len(y), dt)
    return freqs, spec


def _sine_fit_amp(t: np.ndarray, y: np.ndarray, freq: float) -> float:
    """Least-squares amplitude of y at freq (a*sin + b*cos + c)."""
    w = 2.0 * np.pi * freq
    A = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b, _c = coef
    return float(np.hypot(a, b))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the genuine OpenFAST drivetrain-torsion resonance case.")
    parser.add_argument("--on-csv", type=str,
                        default=str(SWEEP_DIR / "WT1_LEOGO_FMU_torsion_3p10Hz.csv"))
    parser.add_argument("--off-csv", type=str,
                        default=str(SWEEP_DIR / "WT1_LEOGO_FMU_torsion_2p00Hz.csv"))
    parser.add_argument("--on-freq-hz", type=float, default=F_TORSION_HZ)
    parser.add_argument("--off-freq-hz", type=float, default=2.00)
    parser.add_argument("--amp", type=float, default=0.10,
                        help="Fractional torque-modulation amplitude.")
    parser.add_argument("--onset", type=float, default=10.0,
                        help="Time [s] at which the modulation switches on.")
    parser.add_argument("--out", type=str,
                        default=str(SWEEP_DIR / "fmu_genuine_torsion.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    df_on = pd.read_csv(args.on_csv)
    df_off = pd.read_csv(args.off_csv)
    onset = args.onset

    t_on = df_on["t"].to_numpy()
    t_off = df_off["t"].to_numpy()
    sh_on = _shaft_dev_knm(df_on, onset)
    sh_off = _shaft_dev_knm(df_off, onset)

    tp_on, shp_on = _post_onset(t_on, sh_on, onset)
    tp_off, shp_off = _post_onset(t_off, sh_off, onset)

    f_on, s_on = _spectrum(tp_on, shp_on)
    f_off, s_off = _spectrum(tp_off, shp_off)

    # Sine-fit amplitude at each run's own forcing frequency over the settled
    # tail (last 40% of the post-onset window) -> clean, self-oscillation-free
    # amplitude for the genuine OpenFAST mode.
    def _tail_fit(tp, shp, freq):
        i0 = int(0.6 * len(tp))
        return _sine_fit_amp(tp[i0:], shp[i0:] - np.mean(shp[i0:]), freq)

    amp_on = _tail_fit(tp_on, shp_on, args.on_freq_hz)
    amp_off = _tail_fit(tp_off, shp_off, args.off_freq_hz)

    ripple_knm = args.amp * T0_KNM                     # applied +/- torque [kNm]
    ampl_factor = amp_on / max(ripple_knm, 1e-9)       # resonant amplification
    resonance_ratio = amp_on / max(amp_off, 1e-12)

    load_on = _applied_ripple_knm(t_on, args.on_freq_hz, args.amp, onset)

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    fig = plt.figure(figsize=(13.5, 8.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.55, 1.25, 1.0],
                  hspace=0.42, wspace=0.22)

    # --- (top) shared applied generator-torque ripple --------------------------
    ax_load = fig.add_subplot(gs[0, :])
    ax_load.plot(t_on, load_on, color=C_LOAD, lw=0.9)
    ax_load.set_ylabel("Pålagt gen.moment-\nrippel [kNm]")
    ax_load.set_title(
        f"ROSCO open-loop generatormoment-rippel:  "
        f"±{ripple_knm:.0f} kNm (±{args.amp*100:.0f}%),  "
        f"på-resonans {args.on_freq_hz:.2f} Hz",
        fontsize=11)
    ax_load.set_xlim(0, t_on[-1])
    ax_load.axvline(onset, color="k", ls=":", lw=0.8, alpha=0.6)

    # --- (large) shaft-torque deviation: on vs off -----------------------------
    ax_sh = fig.add_subplot(gs[1, :])
    ax_sh.plot(t_off, sh_off, color=C_OFF, lw=0.9,
               label=f"av-resonans {args.off_freq_hz:.2f} Hz  (amp {amp_off:.0f} kNm)")
    ax_sh.plot(t_on, sh_on, color=C_ON, lw=1.0,
               label=f"på-resonans {args.on_freq_hz:.2f} Hz  (amp {amp_on:.0f} kNm)")
    ax_sh.set_ylabel("HSShftTq-avvik\n[kNm]")
    ax_sh.set_xlabel("Tid [s]")
    ax_sh.set_xlim(0, t_on[-1])
    ax_sh.axvline(onset, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_sh.legend(loc="upper left", framealpha=0.9)
    ax_sh.set_title(
        f"Genuin OpenFAST drivverk-torsjon: {resonance_ratio:.0f}× større "
        f"akselmoment på moden enn av-resonans", fontsize=11)

    # --- (bottom left) shaft-torque spectra ------------------------------------
    ax_sp = fig.add_subplot(gs[2, 0])
    ax_sp.plot(f_off, s_off, color=C_OFF, lw=1.2,
               label=f"av-resonans {args.off_freq_hz:.2f} Hz")
    ax_sp.plot(f_on, s_on, color=C_ON, lw=1.4,
               label=f"på-resonans {args.on_freq_hz:.2f} Hz")
    ax_sp.axvline(args.on_freq_hz, color=C_ON, ls="--", lw=1.0, alpha=0.7)
    ax_sp.annotate(f"torsjonsmode\n{args.on_freq_hz:.2f} Hz",
                   xy=(args.on_freq_hz, s_on.max()),
                   xytext=(args.on_freq_hz - 1.7,
                           0.78 * max(s_on.max(), s_off.max())),
                   fontsize=9, color=C_ON)
    ax_sp.set_xlim(0, 6)
    ax_sp.set_xlabel("Frekvens [Hz]")
    ax_sp.set_ylabel("HSShftTq-\nspekter [kNm]")
    ax_sp.set_title("Etter-pådrag spekter (akselmoment)", fontsize=11)
    ax_sp.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # --- (bottom right) settled on-resonance oscillation -----------------------
    ax_zoom = fig.add_subplot(gs[2, 1])
    t_win = t_on[-1] - 5.0
    win = t_on >= t_win
    ax_zoom.plot(t_on[win], sh_on[win], color=C_ON, lw=1.3,
                 label=f"akselmoment  (±{amp_on:.0f} kNm)")
    ax_zoom.set_xlim(t_win, t_on[-1])
    ax_zoom.set_xlabel("Tid [s]")
    ax_zoom.set_ylabel("HSShftTq-avvik\n[kNm]")
    ax_zoom.set_title(f"Innsvingt {args.on_freq_hz:.2f} Hz akselmoment-rippel",
                      fontsize=11)
    ax_zoom.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Elektrisk moment-pådrag -> genuin OpenFAST drivverk-torsjonsresonans  "
        f"(forsterkning {ampl_factor:.1f}× av pålagt rippel)",
        fontsize=13, y=0.985)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Lagret figur: {out}")
    print(f"  HSShftTq på-res amp = {amp_on:.1f} kNm,  av-res amp = {amp_off:.1f} kNm")
    print(f"  resonansforhold = {resonance_ratio:.1f}x,  "
          f"forsterkning = {ampl_factor:.1f}x av ±{ripple_knm:.0f} kNm rippel")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
