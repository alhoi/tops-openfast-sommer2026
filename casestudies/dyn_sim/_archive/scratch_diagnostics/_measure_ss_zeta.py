"""Measure the genuine OpenFAST tower side-to-side damping from a ring-down.

The tower side-to-side mode (~0.234 Hz) is very lightly damped, so a single
generator-torque pulse rings it down slowly. This script isolates that mode in
the ``fmu_YawBrTAyp`` channel of an existing co-simulation CSV (constant wind,
no grid event -> free decay after the pulse) and estimates the damping ratio two
independent ways:

  1. Log decrement on the band-passed peak sequence (reuses _decay_torsion).
  2. Exponential fit of the Hilbert amplitude envelope.

Both should agree; the value is used to recalibrate the reduced-model zeta_ss.

Usage
-----
    python casestudies/dyn_sim/_measure_ss_zeta.py
    python casestudies/dyn_sim/_measure_ss_zeta.py --csv <file> --t-lo 15 --t-hi 70
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from casestudies.dyn_sim._decay_torsion import _bandpass_fft, _log_decrement_zeta


def hilbert_envelope_zeta(t, sig, f0):
    """Damping ratio from an exponential fit of the Hilbert amplitude envelope."""
    from scipy.signal import hilbert
    env = np.abs(hilbert(sig))
    # Fit ln(env) ~ ln(A0) - sigma*t over the part above 10 % of the peak, so the
    # noise floor does not flatten the slope.
    keep = env > 0.1 * env.max()
    tt, ee = t[keep], env[keep]
    A = np.column_stack([tt, np.ones_like(tt)])
    slope, _ = np.linalg.lstsq(A, np.log(ee), rcond=None)[0]
    sigma = -slope
    omega = 2.0 * np.pi * f0
    return float(sigma / np.sqrt(sigma**2 + omega**2)), sigma


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=str(
        PROJECT_ROOT / "results" / "em_interaction_sweep"
        / "WT1_LEOGO_FMU_hero_pulse.csv"))
    p.add_argument("--channel", default="fmu_YawBrTAyp")
    p.add_argument("--f0", type=float, default=0.234,
                   help="Side-to-side modal frequency [Hz].")
    p.add_argument("--t-lo", type=float, default=15.0,
                   help="Start of the free-decay window (after the pulse) [s].")
    p.add_argument("--t-hi", type=float, default=70.0)
    args = p.parse_args()

    d = pd.read_csv(args.csv)
    t = d["t"].to_numpy()
    y = d[args.channel].to_numpy()
    m = (t >= args.t_lo) & (t <= args.t_hi)
    tt, yy = t[m], y[m]
    yy = yy - np.mean(yy)
    dt = float(np.median(np.diff(tt)))

    band = _bandpass_fft(yy, dt, 0.6 * args.f0, 1.6 * args.f0)

    zeta_ld = _log_decrement_zeta(tt, band, args.f0)
    zeta_he, sigma_he = hilbert_envelope_zeta(tt, band, args.f0)

    # Peak-to-peak of the band-passed ring for a sanity check of the envelope.
    print(f"CSV: {args.csv}")
    print(f"channel={args.channel}  window=[{args.t_lo},{args.t_hi}]s  "
          f"f0={args.f0} Hz  dt={dt:.4f}s")
    print(f"  band-pass p2p = {np.ptp(band):.4e}")
    print(f"  log-decrement   zeta = {100*zeta_ld:.3f} %")
    print(f"  Hilbert envelope zeta = {100*zeta_he:.3f} %  "
          f"(sigma={sigma_he:.4f} 1/s, tau={1.0/sigma_he:.1f} s)")
    zbar = 0.5 * (zeta_ld + zeta_he)
    print(f"  --> mean zeta ~ {100*zbar:.3f} %  (Q ~ {1.0/(2*zbar):.0f})")


if __name__ == "__main__":
    main()
