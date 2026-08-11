"""Quick validation analysis of a genuine-OpenFAST torsion probe CSV.

Reads a co-simulation CSV and reports, for the drivetrain-torsion channel
HSShftTq: the mean, the single-sided FFT peak (frequency + amplitude) in a
band, and the least-squares sine-fit amplitude at a given forcing frequency,
over an early and a late settled window. Used only to confirm that the
OpenFAST-native DrTrDOF mode responds cleanly and stays bounded.

Usage:
    python casestudies/dyn_sim/_val_analyze.py <csv> --freq 3.1 \
        --channel fmu_HSShftTq
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def fft_peak(t, y, f_lo, f_hi):
    """Single-sided FFT peak (freq, amplitude) of y(t) in [f_lo, f_hi]."""
    y = y - np.mean(y)
    n = len(y)
    dt = float(np.median(np.diff(t)))
    freqs = np.fft.rfftfreq(n, d=dt)
    amp = np.abs(np.fft.rfft(y)) * 2.0 / n
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return float("nan"), float("nan")
    i = np.argmax(amp[band])
    return float(freqs[band][i]), float(amp[band][i])


def sine_fit_amp(t, y, freq_hz):
    """Least-squares amplitude of y at freq_hz (a*sin + b*cos + c)."""
    w = 2.0 * np.pi * freq_hz
    A = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b, _c = coef
    return float(np.hypot(a, b))


def window(d, t_lo, t_hi):
    m = (d["t"].values >= t_lo) & (d["t"].values <= t_hi)
    return d["t"].values[m], d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=str)
    p.add_argument("--freq", type=float, required=True)
    p.add_argument("--channel", type=str, default="fmu_HSShftTq")
    p.add_argument("--band", type=float, nargs=2, default=[2.0, 5.0])
    p.add_argument("--early", type=float, nargs=2, default=[6.0, 12.0])
    p.add_argument("--late", type=float, nargs=2, default=[22.0, 30.0])
    args = p.parse_args()

    d = pd.read_csv(args.csv)
    t = d["t"].values
    y = d[args.channel].values
    print(f"CSV: {args.csv}")
    print(f"rows={len(d)}  t=[{t.min():.2f},{t.max():.2f}]  "
          f"channel={args.channel}")
    print(f"global: mean={np.mean(y):.4e}  std={np.std(y):.4e}  "
          f"p2p={np.ptp(y):.4e}")

    for label, (lo, hi) in (("early", args.early), ("late", args.late)):
        m = (t >= lo) & (t <= hi)
        if not np.any(m):
            print(f"  {label} window [{lo},{hi}] empty")
            continue
        tt, yy = t[m], y[m]
        fpk, apk = fft_peak(tt, yy, args.band[0], args.band[1])
        afit = sine_fit_amp(tt, yy - np.mean(yy), args.freq)
        print(f"  {label} [{lo:.0f}-{hi:.0f}s]: FFT peak {fpk:.3f} Hz "
              f"amp={apk:.4e}   sine-fit@{args.freq:.2f}Hz={afit:.4e}  "
              f"mean={np.mean(yy):.4e}")


if __name__ == "__main__":
    main()
