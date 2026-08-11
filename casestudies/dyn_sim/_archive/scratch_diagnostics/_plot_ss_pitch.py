"""Focused side-to-side + pitch figure for a 2WT resonance run.

Reads a 2WT results CSV and draws a compact three-panel figure that isolates
the side-to-side story: the grid-frequency drive, the side-to-side tower
acceleration on both turbines, and the blade pitch response on both turbines.
Pitch is shown as a deviation from its pre-event mean (in millidegrees) so the
small controller response at the side-to-side frequency is visible; the
absolute operating pitch is written in the legend.

Usage:
    python _plot_ss_pitch.py --csv <csv> --event-time 10 [--out out.png] [--show]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

C1 = '#1f5fbf'   # WT1 (Region 3, 12 m/s, pitch-controlled)
C2 = '#c1272d'   # WT2 (Region 2, 9 m/s, fixed fine pitch)


def _pre_mean(df, col, event_time):
    m = (df['t'] >= 2.0) & (df['t'] <= event_time - 1.0)
    return float(df.loc[m, col].mean())


def make_figure(df, event_time, out_path, show, title):
    t = df['t'].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(10.0, 9.0), sharex=True)

    # Panel 0: grid (centre-of-inertia) frequency = the electrical drive
    ax = axes[0]
    ax.plot(t, df['grid_freq_hz'], color='0.25', lw=1.3)
    ax.set_ylabel('Nettfrekvens\n[Hz]')
    ax.set_title(title, fontsize=11)

    # Panel 1: side-to-side tower acceleration on both turbines
    ax = axes[1]
    ax.plot(t, df['ss_accel_mps2_wt1'], color=C1, lw=1.0, label='WT1 (12 m/s)')
    ax.plot(t, df['ss_accel_mps2_wt2'], color=C2, lw=1.0, label='WT2 (9 m/s)')
    ax.set_ylabel('Tårn side-side\n$a_{ss}$ [m/s²]')
    ax.legend(loc='upper left', fontsize=8, ncol=2)

    # Panel 2: pitch deviation from the pre-event operating point [millideg]
    ax = axes[2]
    p1_0 = _pre_mean(df, 'pitch_deg_wt1', event_time)
    p2_0 = _pre_mean(df, 'pitch_deg_wt2', event_time)
    ax.plot(t, 1e3 * (df['pitch_deg_wt1'] - p1_0), color=C1, lw=1.0,
            label=f'WT1 (drift fra {p1_0:.2f}°)')
    ax.plot(t, 1e3 * (df['pitch_deg_wt2'] - p2_0), color=C2, lw=1.0,
            label=f'WT2 (fast {p2_0:.2f}°, ingen regulering)')
    ax.set_ylabel('Pitch-avvik\nΔβ [millideg]')
    ax.set_xlabel('Tid [s]')
    ax.legend(loc='upper left', fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.axvline(event_time, color='k', ls='--', lw=0.9, alpha=0.7)
    axes[0].annotate('syklisk last starter', xy=(event_time, 0.5),
                     xycoords=('data', 'axes fraction'),
                     xytext=(6, 0), textcoords='offset points',
                     fontsize=8, color='k', va='center')

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f'wrote {out_path}')
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--event-time', type=float, default=10.0)
    ap.add_argument('--title', default='Side-til-side-resonans: nettdrivkraft, '
                    'tårnrespons og pitch')
    ap.add_argument('--out', default=None)
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    make_figure(df, args.event_time, args.out, args.show, args.title)


if __name__ == '__main__':
    main()
