r"""
Write a ROSCO open-loop generator-torque table with a short Hann-shaped torque
PULSE on top of the Region-2 operating point.

Unlike a single-frequency sine (which rings only one mode), a short pulse is
broadband and simultaneously excites both lightly damped OpenFAST structural
modes reached through the generator-torque channel:
  * drivetrain torsion  (~3.10 Hz, HSShftTq)
  * tower side-to-side  (~0.234 Hz, YawBrTAyp)

This is the genuine-OpenFAST analogue of the reduced-model "grid load step
rings both modes" used in the hero figure -- here the kick enters through the
electromagnetic-torque port, the only path that reaches the turbine structure.

The table is  GenTq(t) = T0 * (1 + amp * hann(t)),  where hann(t) is a raised
sine-squared bump of width W centred at the onset, and zero elsewhere, so the
net angular impulse is small (little permanent rotor drift).

Usage
-----
  .\.venv\Scripts\python.exe casestudies\dyn_sim\_write_ol_pulse.py ^
      --onset 10 --width 0.5 --amp 0.25 --t-max 70
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from casestudies.dyn_sim.make_ol_gentq import write_ol_file

T0_NM = 11023871.0  # Region-2 (8 m/s) operating-point generator torque [Nm]
OL_FILE = "test1002/ControlData/OLInput_ROSCO.dat"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onset", type=float, default=10.0,
                   help="Pulse start time [s].")
    p.add_argument("--width", type=float, default=0.5,
                   help="Pulse width [s] (Hann bump).")
    p.add_argument("--amp", type=float, default=0.25,
                   help="Fractional peak torque amplitude (0.25 = 25 pct).")
    p.add_argument("--mean", type=float, default=T0_NM,
                   help="Region-2 operating torque [Nm].")
    p.add_argument("--t-max", type=float, default=70.0)
    p.add_argument("--dt", type=float, default=0.01,
                   help="Breakpoint grid step [s] (matches FMU comm step).")
    p.add_argument("--out", type=str, default=OL_FILE)
    args = p.parse_args()

    t = np.arange(0.0, args.t_max + args.dt, args.dt)
    bump = np.zeros_like(t)
    m = (t >= args.onset) & (t <= args.onset + args.width)
    bump[m] = np.sin(np.pi * (t[m] - args.onset) / args.width) ** 2
    gentq = args.mean * (1.0 + args.amp * bump)

    out_path = Path(args.out)
    write_ol_file(t, gentq, out_path)

    print(f"Wrote pulse OL table: onset={args.onset}s, width={args.width}s, "
          f"amp={args.amp} -> {out_path.resolve()}")
    print(f"  peak GenTq = {gentq.max():.4e} Nm "
          f"(+{args.amp*100:.0f}% of {args.mean:.4e} Nm)")


if __name__ == "__main__":
    main()
