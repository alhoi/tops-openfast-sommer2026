"""Analyze a resonant operational scenario: a cyclic process load at f_ss.

On the islanded LEOGO microgrid a large cyclic mechanical load (e.g. a
reciprocating compressor or a beam / sucker-rod pump running near 14 strokes/min
= 0.234 Hz) shows up as a sustained periodic active-power oscillation at the main
bus. Because that frequency coincides with the tower side-to-side (SS) natural
frequency, it drives the SS mode resonantly through grid -> Pe -> Te -> SS and the
SS response BUILDS UP over many cycles, while the thrust-driven fore-aft (FA)
mode stays quiet. This quantifies the build-up and the SS/FA selectivity.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lockin(t, sig, f, t_lo, t_hi):
    """Amplitude of sig at frequency f over [t_lo, t_hi] (quadrature demod)."""
    m = (t >= t_lo) & (t <= t_hi)
    tt = t[m]
    s = sig[m].astype(float)
    s = s - s.mean()
    w = 2.0 * np.pi * f
    c = np.trapezoid(s * np.cos(w * tt), tt)
    q = np.trapezoid(s * np.sin(w * tt), tt)
    span = tt[-1] - tt[0]
    return 2.0 * np.hypot(c, q) / span


def envelope(sig):
    """Analytic-signal amplitude envelope via Hilbert transform."""
    from scipy.signal import hilbert
    s = sig.astype(float) - float(np.mean(sig))
    return np.abs(hilbert(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/tower_test/resonance_0234.csv")
    ap.add_argument("--f", type=float, default=0.234)
    ap.add_argument("--mod-start", type=float, default=20.0)
    ap.add_argument("--t-skip", type=float, default=120.0,
                    help="Settled-window start for the lock-in amplitudes.")
    ap.add_argument("--out", default="results/tower_test/resonance_0234.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    t = d["t"].to_numpy()
    ss = d["ss_accel_mps2"].to_numpy()
    fa = d["fa_accel_mps2"].to_numpy()
    gmod = d["grid_mod_mw"].to_numpy()
    gfreq = d["grid_freq_hz"].to_numpy()

    ss_amp = lockin(t, ss, args.f, args.t_skip, t[-1])
    fa_amp = lockin(t, fa, args.f, args.t_skip, t[-1])
    gf_amp = lockin(t, gfreq, args.f, args.t_skip, t[-1])

    print("=" * 60)
    print("RESONANT SCENARIO: cyclic process load at f_ss")
    print("=" * 60)
    print(f"CSV                     : {args.csv}")
    print(f"Excitation frequency    : {args.f:.4f} Hz")
    print(f"Grid-freq oscillation   : {1000*gf_amp:.3f} mHz amplitude at {args.f:.3f} Hz")
    print("-" * 60)
    print(f"SS accel lock-in @f_ss  : {ss_amp:.4f} m/s^2")
    print(f"FA accel lock-in @f_ss  : {fa_amp:.4f} m/s^2")
    print(f"SS / FA amplitude ratio : {ss_amp/max(fa_amp,1e-12):.1f} x")
    print("=" * 60)

    # SS build-up envelope after the modulation switches on
    m_on = t >= args.mod_start
    env = envelope(ss[m_on])
    t_on = t[m_on]

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(t, gmod, color="tab:purple")
    ax[0].axvline(args.mod_start, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[0].set_ylabel("Cyclic load [MW]")
    ax[0].set_title(f"Cyclic process load at {args.f:.3f} Hz "
                    f"(= tower SS mode): resonant build-up")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, gfreq, color="tab:blue")
    ax[1].axvline(args.mod_start, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[1].set_ylabel("Grid freq [Hz]")
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(t, ss, color="tab:red", label="SS accel", lw=0.8)
    ax[2].plot(t_on, env, color="darkred", ls="--", lw=1.3, label="SS envelope")
    ax[2].plot(t, fa, color="tab:green", label="FA accel", alpha=0.8, lw=0.8)
    ax[2].axvline(args.mod_start, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[2].set_ylabel("Tower accel [m/s$^2$]")
    ax[2].set_xlabel("Time [s]")
    ax[2].legend(loc="upper left", fontsize=8)
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Figure written to: {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
