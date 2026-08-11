"""Quantify the tower SS/FA modal amplitude before vs after the grid event.

Reads a 2WT results CSV and reports, for the WT1 side-to-side and fore-aft
acceleration channels, the peak-to-peak and a single-bin lock-in (amplitude +
phase) at the modal frequency in a pre-event and a post-event window. This is
used to explain why the tower oscillation can look SMALLER after the event: the
step response superimposes on the pre-existing init-transient ring-down, so the
net amplitude depends on the phase at the event time (constructive vs
destructive interference).

Usage:
    python _diag_tower_amplitude.py <csv> [event_time]
"""
import sys

import numpy as np
import pandas as pd


def lockin(t, y, f):
    """Single-bin Fourier projection: return (amplitude, phase_rad)."""
    y = y - np.mean(y)
    n = len(t)
    c = (2.0 / n) * np.sum(y * np.exp(-1j * 2.0 * np.pi * f * t))
    return float(np.abs(c)), float(np.angle(c))


def window_stats(df, col, f, t0, t1):
    m = (df['t'] >= t0) & (df['t'] <= t1)
    t = df.loc[m, 't'].to_numpy()
    y = df.loc[m, col].to_numpy()
    amp, phase = lockin(t, y, f)
    return {
        'ptp': float(np.ptp(y)),
        'std': float(np.std(y)),
        'lockin_amp': amp,
        'phase_deg': np.degrees(phase),
    }


def main():
    csv = sys.argv[1]
    event = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    df = pd.read_csv(csv)

    channels = [
        ('ss_accel_mps2_wt1', 0.234, 'SS WT1'),
        ('fa_accel_mps2_wt1', 0.235, 'FA WT1'),
        ('ss_accel_mps2_wt2', 0.234, 'SS WT2'),
        ('fa_accel_mps2_wt2', 0.235, 'FA WT2'),
    ]

    pre = (2.0, event - 1.0)
    post = (event + 3.0, min(float(df['t'].max()), event + 25.0))

    print(f"csv: {csv}")
    print(f"event_time = {event:.1f} s")
    print(f"pre-window  = {pre[0]:.1f}..{pre[1]:.1f} s")
    print(f"post-window = {post[0]:.1f}..{post[1]:.1f} s\n")

    hdr = f"{'channel':10s} {'window':5s} {'ptp':>10s} {'std':>10s} {'lockin':>10s} {'phase[deg]':>11s}"
    print(hdr)
    print('-' * len(hdr))
    for col, f, label in channels:
        if col not in df.columns:
            continue
        a = window_stats(df, col, f, *pre)
        b = window_stats(df, col, f, *post)
        print(f"{label:10s} {'pre':5s} {a['ptp']:10.3e} {a['std']:10.3e} "
              f"{a['lockin_amp']:10.3e} {a['phase_deg']:11.1f}")
        print(f"{label:10s} {'post':5s} {b['ptp']:10.3e} {b['std']:10.3e} "
              f"{b['lockin_amp']:10.3e} {b['phase_deg']:11.1f}")
        ratio = b['lockin_amp'] / a['lockin_amp'] if a['lockin_amp'] > 0 else float('nan')
        print(f"{label:10s} {'post/pre lockin ratio':>28s} = {ratio:.2f}\n")


if __name__ == '__main__':
    main()
