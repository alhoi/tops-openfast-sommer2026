r"""
Poster figure for the GENUINE OpenFAST tower side-to-side (SS) resonance case.

Companion to the drivetrain-torsion figure (plot_fmu_genuine_torsion.py). Both
excite a lightly damped high-fidelity OpenFAST structural mode entirely through
the ROSCO open-loop generator-torque channel (OL_Mode = 1) -- the only torque
path that reaches the turbine structure. Here the target is the 1st tower
side-to-side bending mode (ElastoDyn TwSSDOF1 = True) at ~0.234 Hz, read from
the tower-top side-to-side acceleration YawBrTAyp (only exposed by fast_debug.fmu).

Because YawBrTAyp carries broadband ambient motion (rotor harmonics, aeroelastic
content) on top of the forced response, the resonant amplitude is quantified by
a least-squares sine fit at the forcing frequency -- NOT peak-to-peak, which is
dominated by the broadband floor off resonance.

Reads two LEOGO + OpenFAST FMU runs from the SS frequency sweep (sweep_resonance):
  * on-resonance  : ROSCO open-loop torque ripple at 0.234 Hz  (ss_0p234.csv)
  * off-resonance : same ripple amplitude at 0.100 Hz (control) (ss_0p100.csv)

and draws a 4-panel poster figure:

  (top)      the shared applied generator-torque ripple [kNm]
  (large)    tower SS acceleration vs time: on-resonance build-up vs off-res.
  (bottom L) post-onset SS acceleration spectrum, on vs off, 0.234 Hz marked
  (bottom R) SS vs fore-aft (FA) acceleration on resonance -> the torque
             pulsation drives SS but barely touches thrust-driven FA

Usage
-----
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_fmu_genuine_ss.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_fmu_genuine_ss.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "sweep"
OUT_DIR = PROJECT_ROOT / "results" / "em_interaction_sweep"

F_SS_HZ = 0.234           # tower side-to-side eigenfrequency
T0_KNM = 11023.871        # Region-2 (8 m/s) operating-point generator torque

C_ON = "#0b6e4f"          # on-resonance (green)
C_OFF = "#8a8f98"         # off-resonance control (grey)
C_FA = "#c0392b"          # fore-aft (red)
C_LOAD = "#2c3e50"        # applied torque ripple (dark)

SS_CH = "fmu_YawBrTAyp"   # tower-top side-to-side acceleration [m/s^2]
FA_CH = "fmu_YawBrTAxp"   # tower-top fore-aft acceleration [m/s^2]


def _post_onset(df: pd.DataFrame, col: str, onset: float, hi=None):
    t = df["t"].to_numpy()
    m = t >= onset
    if hi is not None:
        m = m & (t <= hi)
    return t[m], df[col].to_numpy()[m]


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


def _applied_ripple_knm(t: np.ndarray, freq: float, amp: float, onset: float,
                        ramp_s: float = 0.0, stop=None) -> np.ndarray:
    # Reconstruct the smoothstep envelope actually written to the ROSCO OL table:
    # ramp 0 -> 1 over ramp_s from onset, held at 1, and (if stop is given) ramp
    # 1 -> 0 over ramp_s ending at stop, then held at 0 (return to normal drift).
    if ramp_s > 0.0:
        u_up = np.clip((t - onset) / ramp_s, 0.0, 1.0)
        env = u_up * u_up * (3.0 - 2.0 * u_up)
        if stop is not None:
            u_dn = np.clip((stop - t) / ramp_s, 0.0, 1.0)
            env = env * (u_dn * u_dn * (3.0 - 2.0 * u_dn))
    else:
        env = np.where(t >= onset, 1.0, 0.0)
        if stop is not None:
            env = np.where(t >= stop, 0.0, env)
    return T0_KNM * amp * env * np.sin(2.0 * np.pi * freq * (t - onset))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the genuine OpenFAST tower side-to-side resonance case.")
    parser.add_argument("--on-csv", type=str,
                        default=str(SWEEP_DIR / "ss_0p234.csv"))
    parser.add_argument("--off-csv", type=str,
                        default=str(SWEEP_DIR / "ss_0p100.csv"))
    parser.add_argument("--on-freq-hz", type=float, default=F_SS_HZ)
    parser.add_argument("--off-freq-hz", type=float, default=0.100)
    parser.add_argument("--amp", type=float, default=0.10,
                        help="Fractional torque-modulation amplitude.")
    parser.add_argument("--onset", type=float, default=10.0)
    parser.add_argument("--ramp-s", type=float, default=15.0,
                        help="Smoothstep ramp duration [s] of the OL table "
                             "(used to reconstruct the applied-ripple trace).")
    parser.add_argument("--stop", type=float, default=None,
                        help="Time [s] at which the excitation was switched off "
                             "(return to normal drift). Enables the ring-down "
                             "view and restricts the settled-amplitude fit and "
                             "spectra to the forced window [onset, stop].")
    parser.add_argument("--out", type=str,
                        default=str(OUT_DIR / "fmu_genuine_ss.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    df_on = pd.read_csv(args.on_csv)
    df_off = pd.read_csv(args.off_csv)
    onset = args.onset

    t_on, ss_on = _post_onset(df_on, SS_CH, onset)
    t_off, ss_off = _post_onset(df_off, SS_CH, onset)

    # Spectra over the forced window only (exclude the post-stop ring-down).
    ts_on, ys_on = _post_onset(df_on, SS_CH, onset, hi=args.stop)
    ts_off, ys_off = _post_onset(df_off, SS_CH, onset, hi=args.stop)
    f_on, s_on = _spectrum(ts_on, ys_on)
    f_off, s_off = _spectrum(ts_off, ys_off)

    # Sine-fit settled amplitude at each run's forcing frequency. With --stop the
    # settled window is [stop-40, stop] (just before the excitation is removed);
    # otherwise the last 40% of the record. Robust to the broadband ambient floor.
    def _settled_fit(t, y, freq):
        if args.stop is not None:
            m = (t >= args.stop - 40.0) & (t <= args.stop)
        else:
            m = np.arange(len(t)) >= int(0.6 * len(t))
        tt, yy = t[m], y[m]
        return _sine_fit_amp(tt, yy - np.mean(yy), freq)

    amp_ss_on = _settled_fit(t_on, ss_on, args.on_freq_hz)
    amp_ss_off = _settled_fit(t_off, ss_off, args.off_freq_hz)

    resonance_ratio = amp_ss_on / max(amp_ss_off, 1e-12)
    ripple_knm = args.amp * T0_KNM

    t_full = df_on["t"].to_numpy()
    load_on = _applied_ripple_knm(t_full, args.on_freq_hz, args.amp, onset,
                                  ramp_s=args.ramp_s, stop=args.stop)

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    fig = plt.figure(figsize=(13.5, 8.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.55, 1.25, 1.0],
                  hspace=0.42, wspace=0.22)

    # --- (top) shared applied generator-torque ripple --------------------------
    ax_load = fig.add_subplot(gs[0, :])
    ax_load.plot(t_full, load_on, color=C_LOAD, lw=0.7)
    ax_load.set_ylabel("Pålagt gen.moment-\nrippel [kNm]")
    ax_load.set_title(
        f"ROSCO open-loop generatormoment-rippel:  "
        f"±{ripple_knm:.0f} kNm (±{args.amp*100:.0f}%),  "
        f"på-resonans {args.on_freq_hz:.3f} Hz",
        fontsize=11)
    ax_load.set_xlim(0, t_full[-1])
    ax_load.axvline(onset, color="k", ls=":", lw=0.8, alpha=0.6)
    if args.stop is not None:
        ax_load.axvline(args.stop, color="k", ls=":", lw=0.8, alpha=0.6)

    # --- (large) SS acceleration: on vs off ------------------------------------
    ax_ss = fig.add_subplot(gs[1, :])
    ax_ss.plot(t_off, ss_off, color=C_OFF, lw=0.8,
               label=f"av-resonans {args.off_freq_hz:.3f} Hz  "
                     f"(amp {amp_ss_off:.3g} m/s\u00b2)")
    ax_ss.plot(t_on, ss_on, color=C_ON, lw=1.0,
               label=f"på-resonans {args.on_freq_hz:.3f} Hz  "
                     f"(amp {amp_ss_on:.3g} m/s\u00b2)")
    ax_ss.set_ylabel("Tårn SS-akselerasjon\nYawBrTAyp [m/s\u00b2]")
    ax_ss.set_xlabel("Tid [s]")
    ax_ss.set_xlim(0, t_full[-1])
    ax_ss.axvline(onset, color="k", ls=":", lw=0.8, alpha=0.6)
    if args.stop is not None:
        ax_ss.axvline(args.stop, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_ss.legend(loc="upper left", framealpha=0.9)
    ax_ss.set_title(
        f"Genuin OpenFAST tårn side-til-side: {resonance_ratio:.0f}× større "
        f"respons på moden enn av-resonans", fontsize=11)

    # --- (bottom left) SS spectra ----------------------------------------------
    ax_sp = fig.add_subplot(gs[2, 0])
    ax_sp.plot(f_off, s_off, color=C_OFF, lw=1.2,
               label=f"av-resonans {args.off_freq_hz:.3f} Hz")
    ax_sp.plot(f_on, s_on, color=C_ON, lw=1.4,
               label=f"på-resonans {args.on_freq_hz:.3f} Hz")
    ax_sp.axvline(args.on_freq_hz, color=C_ON, ls="--", lw=1.0, alpha=0.7)
    ax_sp.annotate(f"tårn SS-mode\n{args.on_freq_hz:.3f} Hz",
                   xy=(args.on_freq_hz, s_on.max()),
                   xytext=(args.on_freq_hz + 0.06,
                           0.72 * max(s_on.max(), s_off.max())),
                   fontsize=9, color=C_ON)
    ax_sp.set_xlim(0, 1.0)
    ax_sp.set_xlabel("Frekvens [Hz]")
    ax_sp.set_ylabel("SS-akselerasjon-\nspekter [m/s²]")
    ax_sp.set_title("Etter-pådrag spekter (SS-akselerasjon)", fontsize=11)
    ax_sp.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # --- (bottom right) settled oscillation, or ring-down after --stop ---------
    ax_zoom = fig.add_subplot(gs[2, 1])
    if args.stop is not None:
        t_win = args.stop - 15.0
        zoom_title = (f"Retur til normal drift ved t={args.stop:.0f} s "
                      f"(utsvingning)")
    else:
        t_win = t_full[-1] - 25.0
        zoom_title = f"Innsvingt {args.on_freq_hz:.3f} Hz SS-oscillasjon"
    win = t_on >= t_win
    ax_zoom.plot(t_on[win], ss_on[win], color=C_ON, lw=1.1,
                 label=f"SS-akselerasjon  (amp {amp_ss_on:.3g} m/s²)")
    ax_zoom.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    if args.stop is not None:
        ax_zoom.axvline(args.stop, color="k", ls=":", lw=0.9, alpha=0.7)
    ax_zoom.set_xlim(t_win, t_full[-1])
    ax_zoom.set_xlabel("Tid [s]")
    ax_zoom.set_ylabel("Tårn SS-akselerasjon\n[m/s²]")
    ax_zoom.set_title(zoom_title, fontsize=11)
    ax_zoom.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Elektrisk moment-pådrag -> genuin OpenFAST tårn side-til-side-resonans  "
        f"(resonansforhold {resonance_ratio:.0f}× vs av-resonans)",
        fontsize=13, y=0.985)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Lagret figur: {out}")
    print(f"  SS på-res amp = {amp_ss_on:.3g} m/s²,  "
          f"av-res amp = {amp_ss_off:.3g} m/s²")
    print(f"  resonansforhold = {resonance_ratio:.1f}x")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
