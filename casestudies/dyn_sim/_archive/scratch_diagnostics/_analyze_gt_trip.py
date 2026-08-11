"""Analyze a GT-trip (generation-loss) LEOGO run.

Reads a CSV produced by test_WT_LEOGO_tower_sim.py and shows how a realistic
grid-frequency transient rings the side-to-side (SS) tower mode while the
fore-aft (FA) mode stays quiet. Also checks the spectral overlap between the
grid-frequency transient and the tower band near f_ss.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def band_energy(sig, dt, f_lo, f_hi):
    """Fraction of spectral energy of sig within [f_lo, f_hi]."""
    sig = np.asarray(sig, dtype=float)
    sig = sig - np.mean(sig)
    n = sig.size
    freqs = np.fft.rfftfreq(n, d=dt)
    amp = np.abs(np.fft.rfft(sig)) ** 2
    total = np.sum(amp)
    if total <= 0:
        return 0.0, freqs, amp
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.sum(amp[mask]) / total), freqs, amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/tower_test/gt_trip.csv")
    ap.add_argument("--f-ss", type=float, default=0.234)
    ap.add_argument("--event-time", type=float, default=30.0)
    ap.add_argument("--out", default="results/tower_test/gt_trip.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    t = df["t"].to_numpy()
    dt = float(np.median(np.diff(t)))

    f_grid = df["grid_freq_hz"].to_numpy()
    ss = df["ss_accel_mps2"].to_numpy()
    fa = df["fa_accel_mps2"].to_numpy()

    # Post-event window (skip initial transient before the trip)
    post = t >= args.event_time
    tp = t[post]

    ss_pk = float(np.max(np.abs(ss[post])))
    fa_pk = float(np.max(np.abs(fa[post])))
    ss_rms = rms(ss[post])
    fa_rms = rms(fa[post])

    f_nadir = float(np.min(f_grid))
    f_nadir_t = float(t[np.argmin(f_grid)])
    f_settle = float(np.mean(f_grid[t >= t[-1] - 10.0]))

    # Spectral overlap of the grid-frequency transient with the tower band
    band = (args.f_ss - 0.03, args.f_ss + 0.03)
    frac_grid, fr, amp_grid = band_energy(f_grid[post], dt, *band)
    frac_ss, _, amp_ss = band_energy(ss[post], dt, *band)

    print("=" * 60)
    print("GT-TRIP ANALYSIS")
    print("=" * 60)
    print(f"CSV                     : {args.csv}")
    print(f"Grid freq nadir         : {f_nadir:.4f} Hz @ t={f_nadir_t:.1f} s")
    print(f"Grid freq settle (last10s): {f_settle:.4f} Hz")
    print(f"Grid freq drop          : {50.0 - f_nadir:.4f} Hz")
    print("-" * 60)
    print(f"SS accel peak (post)    : {ss_pk:.4f} m/s^2   RMS: {ss_rms:.4f}")
    print(f"FA accel peak (post)    : {fa_pk:.4f} m/s^2   RMS: {fa_rms:.4f}")
    print(f"SS/FA peak ratio        : {ss_pk / max(fa_pk, 1e-12):.1f} x")
    print(f"SS/FA RMS  ratio        : {ss_rms / max(fa_rms, 1e-12):.1f} x")
    print("-" * 60)
    print(f"Energy in [{band[0]:.3f},{band[1]:.3f}] Hz band:")
    print(f"  grid_freq transient   : {100*frac_grid:.1f} %")
    print(f"  ss_accel              : {100*frac_ss:.1f} %")
    print("=" * 60)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

    ax[0].plot(t, f_grid, color="tab:blue")
    ax[0].axvline(args.event_time, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[0].set_ylabel("Grid freq [Hz]")
    ax[0].set_title("LEOGO GT-trip: grid frequency and tower response")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, ss, color="tab:red", label="SS accel")
    ax[1].plot(t, fa, color="tab:green", label="FA accel", alpha=0.8)
    ax[1].axvline(args.event_time, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[1].set_ylabel("Tower accel [m/s$^2$]")
    ax[1].set_xlabel("Time [s]")
    ax[1].legend(loc="upper right")
    ax[1].grid(True, alpha=0.3)

    # Spectra (post-event)
    ax[2].semilogy(fr, amp_grid / np.max(amp_grid), color="tab:blue",
                   label="grid freq (norm)")
    ax[2].semilogy(fr, amp_ss / np.max(amp_ss), color="tab:red",
                   label="SS accel (norm)", alpha=0.8)
    ax[2].axvspan(band[0], band[1], color="orange", alpha=0.2,
                  label=f"tower band {args.f_ss:.3f} Hz")
    ax[2].set_xlim(0, 2.0)
    ax[2].set_ylim(1e-5, 2)
    ax[2].set_xlabel("Frequency [Hz]")
    ax[2].set_ylabel("Power (norm)")
    ax[2].legend(loc="upper right", fontsize=8)
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
