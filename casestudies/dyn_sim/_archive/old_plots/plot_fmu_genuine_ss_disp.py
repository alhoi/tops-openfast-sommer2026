r"""
Tower-top side-to-side DISPLACEMENT figure (derived from YawBrTAyp).

The OpenFAST FMU (openfast_fmu/modelDescription.xml) exposes only the tower-top
side-to-side *acceleration* YawBrTAyp -- there is no displacement output channel.
This script therefore recovers the tower-top side-to-side *displacement* by
double-integrating the logged acceleration in the frequency domain (X = -A/w^2)
with a high-pass floor that suppresses the 1/w^2 integration drift, and compares
the on-resonance (0.234 Hz) and off-resonance (0.100 Hz) runs in BOTH
acceleration and displacement.

Point of the figure
-------------------
A modest acceleration at a low frequency corresponds to a large displacement,
because a = -w^2 * x. At 0.234 Hz an amplitude of ~0.3 m/s^2 is only ~0.03 g but
maps to ~15 cm of tower-top sway. The figure makes that explicit, and separates
the on-resonance forced build-up from the off-resonance control (whose visible
motion is the onset-transient ring at the tower mode, not the 0.100 Hz forcing).

Usage
-----
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_fmu_genuine_ss_disp.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_fmu_genuine_ss_disp.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.integrate import cumulative_trapezoid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "sweep"
OUT_DIR = PROJECT_ROOT / "results" / "em_interaction_sweep"

F_SS_HZ = 0.234
SS_CH = "fmu_YawBrTAyp"    # tower-top side-to-side acceleration [m/s^2]

C_ON = "#0b6e4f"          # on-resonance (green)
C_OFF = "#8a8f98"         # off-resonance control (grey)


def _post_onset(df: pd.DataFrame, col: str, onset: float):
    t = df["t"].to_numpy()
    m = t >= onset
    return t[m], df[col].to_numpy()[m]


def _sine_fit_amp(t: np.ndarray, y: np.ndarray, freq: float) -> float:
    """Least-squares amplitude of y at freq (a*sin + b*cos + c)."""
    w = 2.0 * np.pi * freq
    A = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b, _c = coef
    return float(np.hypot(a, b))


def _tail_fit(t, y, freq, frac=0.6):
    i0 = int(frac * len(t))
    return _sine_fit_amp(t[i0:], y[i0:] - np.mean(y[i0:]), freq)


def accel_to_disp(t: np.ndarray, a: np.ndarray, f_hp: float) -> np.ndarray:
    """Double-integrate acceleration to displacement (standard pipeline).

    Integrate acceleration -> velocity -> displacement with a zero-phase
    Butterworth high-pass applied after each integration to remove the low-
    frequency drift that integration otherwise accumulates. The high-pass floor
    f_hp sits below both forcing frequencies (0.100 and 0.234 Hz), so the
    physical content is preserved while sub-f_hp drift is suppressed.
    """
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    sos = butter(4, f_hp, btype="highpass", fs=fs, output="sos")
    a = a - np.mean(a)
    v = cumulative_trapezoid(a, dx=dt, initial=0.0)
    v = sosfiltfilt(sos, v)
    x = cumulative_trapezoid(v, dx=dt, initial=0.0)
    x = sosfiltfilt(sos, x)
    return x


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tower-top side-to-side displacement (derived) figure.")
    parser.add_argument("--on-csv", type=str,
                        default=str(SWEEP_DIR / "ss_0p234.csv"))
    parser.add_argument("--off-csv", type=str,
                        default=str(SWEEP_DIR / "ss_0p100.csv"))
    parser.add_argument("--on-freq-hz", type=float, default=F_SS_HZ)
    parser.add_argument("--off-freq-hz", type=float, default=0.100)
    parser.add_argument("--onset", type=float, default=10.0)
    parser.add_argument("--f-hp", type=float, default=0.05,
                        help="High-pass floor [Hz] for the double integration.")
    parser.add_argument("--out", type=str,
                        default=str(OUT_DIR / "fmu_genuine_ss_disp.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    df_on = pd.read_csv(args.on_csv)
    df_off = pd.read_csv(args.off_csv)
    onset = args.onset

    t_on, acc_on = _post_onset(df_on, SS_CH, onset)
    t_off, acc_off = _post_onset(df_off, SS_CH, onset)

    dsp_on = accel_to_disp(t_on, acc_on, args.f_hp)
    dsp_off = accel_to_disp(t_off, acc_off, args.f_hp)

    # Acceleration amplitudes over the settled tail (sine fit at forcing freq).
    amp_acc_on = _tail_fit(t_on, acc_on, args.on_freq_hz)
    amp_acc_off = _tail_fit(t_off, acc_off, args.off_freq_hz)
    # off-resonance content AT THE TOWER MODE (the onset-transient ring)
    amp_acc_off_ring = _tail_fit(t_off, acc_off, args.on_freq_hz)

    # Displacement amplitudes from the exact narrowband relation x = a / w^2
    # (robust; the integrated trace below is only used for the waveform shape,
    # where the sub-cutoff filtering slightly biases a raw amplitude read-out).
    def _disp_amp(acc_amp, freq):
        return acc_amp / (2.0 * np.pi * freq) ** 2

    amp_dsp_on = _disp_amp(amp_acc_on, args.on_freq_hz)
    amp_dsp_off = _disp_amp(amp_acc_off, args.off_freq_hz)
    amp_dsp_off_ring = _disp_amp(amp_acc_off_ring, args.on_freq_hz)

    ratio_acc = amp_acc_on / max(amp_acc_off, 1e-12)
    ratio_dsp = amp_dsp_on / max(amp_dsp_off, 1e-12)

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    fig, (ax_a, ax_d) = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True)

    # --- acceleration (as logged) ---------------------------------------------
    ax_a.plot(t_off, acc_off, color=C_OFF, lw=0.8,
              label=f"av-resonans {args.off_freq_hz:.3f} Hz "
                    f"(tvunget @0.100 Hz = {amp_acc_off:.2e} m/s²; "
                    f"synlig svai = mode-ring @0.234 Hz = {amp_acc_off_ring:.2e} m/s²)")
    ax_a.plot(t_on, acc_on, color=C_ON, lw=1.0,
              label=f"på-resonans {args.on_freq_hz:.3f} Hz "
                    f"(amp {amp_acc_on:.2e} m/s²)")
    ax_a.set_ylabel("SS-akselerasjon\nYawBrTAyp [m/s²]")
    ax_a.axvline(onset, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_a.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax_a.set_title(
        f"Akselerasjon: {ratio_acc:.0f}× større på moden enn av-resonans "
        f"(sinustilpasset ved pådragsfrekvensen)", fontsize=11)

    # --- displacement (derived by double integration) -------------------------
    ax_d.plot(t_off, dsp_off * 100.0, color=C_OFF, lw=0.8,
              label=f"av-resonans {args.off_freq_hz:.3f} Hz "
                    f"(tvunget {amp_dsp_off*100:.2f} cm; "
                    f"mode-ring {amp_dsp_off_ring*100:.2f} cm)")
    ax_d.plot(t_on, dsp_on * 100.0, color=C_ON, lw=1.0,
              label=f"på-resonans {args.on_freq_hz:.3f} Hz "
                    f"(amp {amp_dsp_on*100:.1f} cm)")
    ax_d.set_ylabel("SS-forskyvning\n(utledet) [cm]")
    ax_d.set_xlabel("Tid [s]")
    ax_d.axvline(onset, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_d.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax_d.set_title(
        f"Forskyvning (utledet ved dobbeltintegrasjon): "
        f"{ratio_dsp:.1f}× større på moden – samme svai i cm er stor "
        f"selv om m/s² er liten", fontsize=11)

    fig.suptitle(
        "Genuin OpenFAST tårn side-til-side: akselerasjon vs. forskyvning\n"
        "lav frekvens gjør at liten m/s² svarer til stor utsvingning",
        fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Lagret figur: {out}")
    print(f"  ACC  paa-res @{args.on_freq_hz:.3f}Hz = {amp_acc_on:.3e} m/s2")
    print(f"  ACC  av-res  @{args.off_freq_hz:.3f}Hz = {amp_acc_off:.3e} m/s2 "
          f"(tvunget),  @{args.on_freq_hz:.3f}Hz = {amp_acc_off_ring:.3e} m/s2 "
          f"(mode-ring)")
    print(f"  ACC  resonansforhold = {ratio_acc:.1f}x")
    print(f"  DISP paa-res @{args.on_freq_hz:.3f}Hz = {amp_dsp_on*100:.2f} cm")
    print(f"  DISP av-res  @{args.off_freq_hz:.3f}Hz = {amp_dsp_off*100:.3f} cm "
          f"(tvunget),  @{args.on_freq_hz:.3f}Hz = {amp_dsp_off_ring*100:.3f} cm "
          f"(mode-ring)")
    print(f"  DISP resonansforhold = {ratio_dsp:.1f}x")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
