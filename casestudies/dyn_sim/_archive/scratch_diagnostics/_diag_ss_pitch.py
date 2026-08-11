"""Quantify the pitch response alongside the tower side-to-side mode.

Reads a 2WT results CSV and reports, for both turbines, the side-to-side
acceleration and the blade pitch angle in a pre-event and a during-event
window: peak-to-peak, standard deviation and a single-bin lock-in (amplitude)
at the side-to-side modal frequency. The point is to see whether the same grid
event that resonates the side-to-side tower mode also drives the pitch actuator
(expected on the Region-3, pitch-controlled turbine, and not on the Region-2
turbine which sits at fixed fine pitch).

Usage:
    python _diag_ss_pitch.py <csv> [event_time] [f_ss]
"""
import sys

import numpy as np
import pandas as pd


def lockin(t, y, f):
    """Single-bin Fourier projection: return the amplitude at frequency f."""
    y = y - np.mean(y)
    n = len(t)
    c = (2.0 / n) * np.sum(y * np.exp(-1j * 2.0 * np.pi * f * t))
    return float(np.abs(c))


def window_stats(df, col, f, t0, t1):
    m = (df['t'] >= t0) & (df['t'] <= t1)
    t = df.loc[m, 't'].to_numpy()
    y = df.loc[m, col].to_numpy()
    return {
        'mean': float(np.mean(y)),
        'ptp': float(np.ptp(y)),
        'std': float(np.std(y)),
        'lockin_amp': lockin(t, y, f),
    }


def main():
    csv = sys.argv[1]
    event = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    f_ss = float(sys.argv[3]) if len(sys.argv) > 3 else 0.234
    df = pd.read_csv(csv)

    channels = [
        ('ss_accel_mps2_wt1', 'SS accel WT1 [m/s2]'),
        ('ss_accel_mps2_wt2', 'SS accel WT2 [m/s2]'),
        ('pitch_deg_wt1', 'pitch WT1 [deg]'),
        ('pitch_deg_wt2', 'pitch WT2 [deg]'),
    ]

    pre = (2.0, event - 1.0)
    dur = (event + 3.0, min(float(df['t'].max()), event + 60.0))

    print(f"csv: {csv}")
    print(f"event_time = {event:.1f} s,  f_ss = {f_ss:.3f} Hz")
    print(f"pre-window    = {pre[0]:.1f}..{pre[1]:.1f} s")
    print(f"during-window = {dur[0]:.1f}..{dur[1]:.1f} s\n")

    hdr = (f"{'channel':22s} {'window':6s} {'mean':>10s} {'ptp':>10s} "
           f"{'std':>10s} {'lockin@f_ss':>12s}")
    print(hdr)
    print('-' * len(hdr))
    for col, label in channels:
        if col not in df.columns:
            continue
        a = window_stats(df, col, f_ss, *pre)
        b = window_stats(df, col, f_ss, *dur)
        print(f"{label:22s} {'pre':6s} {a['mean']:10.4f} {a['ptp']:10.3e} "
              f"{a['std']:10.3e} {a['lockin_amp']:12.3e}")
        print(f"{label:22s} {'during':6s} {b['mean']:10.4f} {b['ptp']:10.3e} "
              f"{b['std']:10.3e} {b['lockin_amp']:12.3e}")
        ratio = (b['lockin_amp'] / a['lockin_amp']
                 if a['lockin_amp'] > 0 else float('nan'))
        print(f"{label:22s} {'lock-in during/pre':>36s} = {ratio:.1f}\n")


if __name__ == '__main__':
    main()
