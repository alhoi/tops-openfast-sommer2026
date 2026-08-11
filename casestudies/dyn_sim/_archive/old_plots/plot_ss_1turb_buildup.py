r"""
Single-turbine genuine OpenFAST tower side-to-side (SS) resonance build-up.

Reads ONE LEOGO + OpenFAST FMU run in which the ROSCO open-loop generator-torque
channel (OL_Mode = 1) applies a sinusoidal torque ripple at the 1st tower
side-to-side eigenfrequency (~0.234 Hz). This is the only torque path that
reaches the turbine structure in this build (ROSCO VSContrl = 5 blocks every
electrical / external-command channel), so it is the correct way to probe the
genuine high-fidelity SS mode.

The tower-top side-to-side acceleration YawBrTAyp (only exposed by
fast_debug.fmu) carries broadband ambient content, so the resonant amplitude is
quantified by a least-squares sine fit at the forcing frequency -- NOT
peak-to-peak.

Draws a 3-panel figure:
  (top)    applied ROSCO open-loop generator-torque ripple [kNm]
  (middle) SS acceleration vs time with the resonant build-up envelope
  (bottom) post-onset SS acceleration spectrum, forcing frequency marked

Usage
-----
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_ss_1turb_buildup.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_ss_1turb_buildup.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = PROJECT_ROOT / "results" / "sweep"
OUT_DIR = PROJECT_ROOT / "results" / "em_interaction_sweep"

F_SS_HZ = 0.234           # tower side-to-side eigenfrequency
T0_KNM = 11023.871        # Region-2 (8 m/s) operating-point generator torque

C_ON = "#0b6e4f"          # SS response (green)
C_ENV = "#c0392b"         # build-up envelope (red)
C_LOAD = "#2c3e50"        # applied torque ripple (dark)

SS_CH = "fmu_YawBrTAyp"   # tower-top side-to-side acceleration [m/s^2]


def _sine_fit_amp(t: np.ndarray, y: np.ndarray, freq: float) -> float:
    """Least-squares amplitude of y at freq (a*sin + b*cos + c)."""
    w = 2.0 * np.pi * freq
    A = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b, _c = coef
    return float(np.hypot(a, b))


def _spectrum(t: np.ndarray, y: np.ndarray):
    dt = float(np.median(np.diff(t)))
    win = np.hanning(len(y))
    spec = np.abs(np.fft.rfft((y - np.mean(y)) * win)) * (2.0 / np.sum(win))
    freqs = np.fft.rfftfreq(len(y), dt)
    return freqs, spec


def _applied_ripple_knm(t, freq, amp, onset, ramp):
    """Reconstruct the applied torque ripple (smoothstep onset envelope)."""
    u = np.clip((t - onset) / max(ramp, 1e-9), 0.0, 1.0)
    env = u * u * (3.0 - 2.0 * u)
    return np.where(t >= onset,
                    T0_KNM * amp * env * np.sin(2.0 * np.pi * freq * (t - onset)),
                    0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot single-turbine OpenFAST tower SS resonance build-up.")
    parser.add_argument("--csv", type=str,
                        default=str(SWEEP_DIR / "ss_1turb_0p234.csv"))
    parser.add_argument("--freq-hz", type=float, default=F_SS_HZ)
    parser.add_argument("--amp", type=float, default=0.10,
                        help="Fractional torque-modulation amplitude.")
    parser.add_argument("--onset", type=float, default=10.0)
    parser.add_argument("--ramp-s", type=float, default=15.0)
    parser.add_argument("--out", type=str,
                        default=str(OUT_DIR / "fmu_ss_1turb_buildup.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    t = df["t"].to_numpy()
    ss = df[SS_CH].to_numpy()
    t_end = float(t[-1])
    onset = args.onset

    # Build-up envelope: sine-fit amplitude at the forcing frequency in a sliding
    # window (robust to the broadband ambient floor).
    win_s = 12.0
    centers, env = [], []
    for c in np.arange(onset + win_s, t_end - win_s / 2, win_s / 2):
        m = (t >= c - win_s / 2) & (t <= c + win_s / 2)
        if m.sum() < 20:
            continue
        centers.append(c)
        env.append(_sine_fit_amp(t[m], ss[m], args.freq_hz))
    centers = np.asarray(centers)
    env = np.asarray(env)

    # Settled amplitude: sine fit over the last 40 % of the record.
    i0 = int(0.6 * len(t))
    amp_ss = _sine_fit_amp(t[i0:], ss[i0:] - np.mean(ss[i0:]), args.freq_hz)

    # Spectrum of the post-onset response.
    m_post = t >= onset + args.ramp_s
    f, s = _spectrum(t[m_post], ss[m_post])
    band = (f >= 0.05) & (f <= 1.0)
    f_peak = f[band][np.argmax(s[band])]

    print(f"Record: {t_end:.1f} s, {len(t)} samples, dt={np.median(np.diff(t)):.4f}")
    print(f"Forcing frequency        : {args.freq_hz:.3f} Hz")
    print(f"Spectral peak (0.05-1 Hz): {f_peak:.4f} Hz")
    print(f"Settled SS amplitude     : {amp_ss:.4f} m/s^2 (sine-fit, last 40%)")
    if len(env):
        print(f"Envelope start / end     : {env[0]:.4f} -> {env[-1]:.4f} m/s^2")
        # displacement amplitude x = a / omega^2
        w = 2.0 * np.pi * args.freq_hz
        print(f"Settled SS displacement  : {100 * amp_ss / w**2:.2f} cm "
              f"(a/omega^2)")

    ripple_knm = args.amp * T0_KNM
    load = _applied_ripple_knm(t, args.freq_hz, args.amp, onset, args.ramp_s)

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    fig = plt.figure(figsize=(12.5, 8.5))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[0.55, 1.3, 0.9],
                  hspace=0.42)

    # --- (top) applied generator-torque ripple ---------------------------------
    ax_load = fig.add_subplot(gs[0])
    ax_load.plot(t, load, color=C_LOAD, lw=0.7)
    ax_load.set_ylabel("Pålagt gen.moment-\nrippel [kNm]")
    ax_load.set_xlim(0, t_end)
    ax_load.axvline(onset, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_load.set_title(
        f"ROSCO open-loop generatormoment-rippel:  ±{ripple_knm:.0f} kNm "
        f"(±{args.amp*100:.0f}%) ved {args.freq_hz:.3f} Hz "
        f"(mykstart {args.ramp_s:.0f} s)", fontsize=11)

    # --- (middle) SS acceleration build-up -------------------------------------
    ax_ss = fig.add_subplot(gs[1])
    ax_ss.plot(t, ss, color=C_ON, lw=0.9, label="Tårn SS-akselerasjon YawBrTAyp")
    if len(env):
        ax_ss.plot(centers, env, color=C_ENV, lw=2.0, marker="o", ms=3,
                   label="Resonans-envelope (sine-fit ved 0.234 Hz)")
        ax_ss.plot(centers, -env, color=C_ENV, lw=2.0, alpha=0.6)
    ax_ss.axvline(onset, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_ss.set_ylabel("Tårn SS-akselerasjon\nYawBrTAyp [m/s²]")
    ax_ss.set_xlabel("Tid [s]")
    ax_ss.set_xlim(0, t_end)
    ax_ss.legend(loc="upper left", framealpha=0.9)
    ax_ss.set_title(
        f"Genuin OpenFAST tårn side-til-side-resonans: oppbygging mot "
        f"innsvingt amplitude {amp_ss:.3f} m/s²", fontsize=11)

    # --- (bottom) spectrum -----------------------------------------------------
    ax_sp = fig.add_subplot(gs[2])
    ax_sp.plot(f, s, color=C_ON, lw=1.4)
    ax_sp.axvline(args.freq_hz, color=C_ENV, ls="--", lw=1.0, alpha=0.8)
    ax_sp.annotate(f"tårn SS-mode\n{f_peak:.3f} Hz",
                   xy=(f_peak, s[band].max()),
                   xytext=(f_peak + 0.06, 0.72 * s[band].max()),
                   fontsize=9, color=C_ENV)
    ax_sp.set_xlim(0, 1.0)
    ax_sp.set_xlabel("Frekvens [Hz]")
    ax_sp.set_ylabel("SS-akselerasjon-\nspekter [m/s²]")
    ax_sp.set_title("Etter-pådrag spekter (SS-akselerasjon)", fontsize=11)

    fig.suptitle(
        "Elektrisk moment-pådrag → genuin OpenFAST tårn side-til-side-resonans "
        "(1 turbin, LEOGO)", fontsize=13, y=0.98)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
