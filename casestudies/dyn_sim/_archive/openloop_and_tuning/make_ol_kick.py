"""Generate a ROSCO open-loop generator-torque table with a brief rectangular
torque pulse ("kick") on top of the constant Region-2 operating torque.

Used for modal frequency identification: a short broadband torque pulse excites
the active structural mode(s), which then ring at their natural frequency in the
tower-top acceleration. The generator torque returns exactly to the mean, so
there is no net operating-point drift. Reuses write_ol_file from make_ol_gentq
for the exact ROSCO open-loop file format.
"""

from pathlib import Path
import argparse
import numpy as np

from make_ol_gentq import write_ol_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mean", type=float, required=True,
                   help="Mean generator torque [Nm] (Region-2 operating point).")
    p.add_argument("--pulse-amp", type=float, default=0.10,
                   help="Fractional pulse height (0.10 = +10 pct of mean).")
    p.add_argument("--start", type=float, default=10.0,
                   help="Pulse start time [s].")
    p.add_argument("--dur", type=float, default=0.5,
                   help="Pulse duration [s]. Short -> broadband kick.")
    p.add_argument("--t-max", type=float, default=100.0,
                   help="Last time breakpoint [s].")
    p.add_argument("--dt", type=float, default=0.05,
                   help="Time step of the breakpoint grid [s].")
    p.add_argument("--out", type=str,
                   default="test1002/ControlData/OLInput_ROSCO.dat")
    args = p.parse_args()

    t = np.arange(0.0, args.t_max + args.dt, args.dt)
    # Rectangular pulse: +pulse_amp of the mean for [start, start+dur), else 0.
    # Sharp edges give broadband content that excites the mode(s) regardless of
    # their (unknown) natural frequency; net-zero drift because it returns to mean.
    pulse = np.where((t >= args.start) & (t < args.start + args.dur),
                     args.pulse_amp, 0.0)
    gentq = args.mean * (1.0 + pulse)

    out_path = Path(args.out)
    write_ol_file(t, gentq, out_path)

    print(f"Wrote {len(t)} rows to {out_path}")
    print(f"  mean={args.mean:.4e} Nm, pulse=+{args.pulse_amp * 100:.0f}% "
          f"from {args.start} to {args.start + args.dur} s "
          f"(dur {args.dur} s), range=[{gentq.min():.4e}, {gentq.max():.4e}] Nm")


if __name__ == "__main__":
    main()
