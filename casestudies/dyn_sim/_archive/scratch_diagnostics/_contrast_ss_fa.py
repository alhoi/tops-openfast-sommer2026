"""
Contrast experiment: grid-side power modulation at the tower modal frequency.

Loads a run produced by ``test_WT_LEOGO_tower_sim.py`` (with both tower modes
enabled and a sinusoidal grid-bus load modulation, ``--grid-mod-amp``) and
quantifies how selectively each tower mode responds to the SAME electrical
disturbance:

- Side-to-side (SS) is driven DIRECTLY by the generator torque Te, so a grid
  power modulation at f_ss builds up a large tower-top SS acceleration.
- Fore-aft (FA) is driven only INDIRECTLY (grid -> omega_m -> lambda -> C_T),
  through the huge rotor inertia, so the SAME modulation barely excites it.

Both modes are calibrated against the OpenFAST FMU (SS vs YawBrTAyp, FA vs
YawBrTAxp), so the amplitude contrast is physically meaningful, not an artefact
of arbitrary gains.

Usage:
    python _contrast_ss_fa.py [--csv results/tower_test/contrast_ssfa.csv]
                              [--f 0.235] [--t-skip 120] [--show]
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def lockin(t, x, f):
    """Single-bin (lock-in) amplitude of x(t) at frequency f [Hz]."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    w = 2.0 * np.pi * f
    T = t[-1] - t[0]
    c = np.trapezoid(x * np.cos(w * t), t)
    s = np.trapezoid(x * np.sin(w * t), t)
    return 2.0 / T * np.hypot(c, s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str,
                        default='results/tower_test/contrast_ssfa.csv',
                        help='Result CSV (abs or relative to project root).')
    parser.add_argument('--f', type=float, default=0.235,
                        help='Tower modal frequency to evaluate the lock-in at [Hz].')
    parser.add_argument('--t-skip', type=float, default=120.0,
                        help='Ignore t < t_skip so only the settled response is used.')
    parser.add_argument('--out', type=str,
                        default='results/tower_test/contrast_ss_fa.png',
                        help='Figure output path (abs or relative to project root).')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    if not csv_path.exists():
        sys.exit(f'CSV not found: {csv_path}')

    d = pd.read_csv(csv_path)
    settled = d['t'] >= args.t_skip

    a_ss = lockin(d['t'][settled], d['ss_accel_mps2'][settled], args.f)
    a_fa = lockin(d['t'][settled], d['fa_accel_mps2'][settled], args.f)
    rms_ss = float(d['ss_accel_mps2'][settled].std())
    rms_fa = float(d['fa_accel_mps2'][settled].std())
    ratio = a_ss / a_fa if a_fa > 0 else np.inf

    print(f'Settled window: t >= {args.t_skip:.0f} s   (f = {args.f:.3f} Hz)')
    print(f'  SS acceleration lock-in : {a_ss:.4f} m/s^2   (rms {rms_ss:.4f})')
    print(f'  FA acceleration lock-in : {a_fa:.4f} m/s^2   (rms {rms_fa:.4f})')
    print(f'  SS / FA amplitude ratio : {ratio:.1f} x')

    # ------------------------------------------------------------------
    # Figure: disturbance (top) and the two tower-top accelerations (bottom).
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(8.5, 6.0), sharex=True,
        gridspec_kw={'height_ratios': [1.0, 2.0]},
    )

    ax0.plot(d['t'], d['grid_mod_mw'], color='tab:gray', lw=1.2)
    ax0.set_ylabel('Grid load\nmodulation [MW]')
    ax0.grid(True, alpha=0.3)
    ax0.set_title(
        f'Grid power modulation at {args.f:.3f} Hz: '
        f'SS responds {ratio:.0f}x more than FA'
    )

    ax1.plot(d['t'], d['ss_accel_mps2'], color='tab:blue', lw=1.1,
             label=f'Side-to-side (direct via $T_e$), lock-in {a_ss:.3f} m/s$^2$')
    ax1.plot(d['t'], d['fa_accel_mps2'], color='tab:red', lw=1.1,
             label=f'Fore-aft (indirect via $\\lambda\\!\\to\\!C_T$), lock-in {a_fa:.3f} m/s$^2$')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Tower-top acceleration [m/s$^2$]')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', framealpha=0.9)

    fig.tight_layout()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f'Figure written to: {out_path}')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
