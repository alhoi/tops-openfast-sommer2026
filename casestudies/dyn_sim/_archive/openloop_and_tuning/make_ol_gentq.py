"""
Generate a ROSCO open-loop input file (OLInput_ROSCO.dat) that prescribes the
generator torque as a function of time.

The file format matches ROSCO's own OpenLoopControl.write_input() exactly:
  line 1:  "!Time<tab><tab>GenTq"
  line 2:  "!sec.<tab><tab>(Nm)"
  data  :  each value formatted "{:<10.8f}\t"

Used with these ROSCO.IEA15MW.IN settings:
  OL_Mode       = 1   (open loop vs. time)
  Ind_Breakpoint = 1  (time in column 1)
  Ind_GenTq     = 2   (generator torque in column 2)

The torque is  GenTq(t) = mean * (1 + amp * sin(2*pi*f*t))  for t >= start,
and the constant mean before start, so it is a zero-mean modulation around the
Region-2 operating torque (no net rotor drift).
"""

from pathlib import Path
import argparse
import numpy as np


def write_ol_file(t, gentq, out_path):
    """Write a time / generator-torque table in ROSCO open-loop format.

    Parameters
    ----------
    t : array_like
        Time breakpoints [s], strictly increasing.
    gentq : array_like
        Generator torque [Nm] at each breakpoint.
    out_path : str or Path
        Destination file (parent directories are created).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["Time", "GenTq"]
    units = ["sec.", "(Nm)"]
    with open(out_path, "w") as f:
        f.write("!" + "\t\t".join(headers) + "\n")
        f.write("!" + "\t\t".join(units) + "\n")
        for ti, qi in zip(t, gentq):
            f.write("{:<10.8f}\t{:<10.8f}\t\n".format(ti, qi))
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mean", type=float, required=True,
                   help="Mean generator torque [Nm] (Region-2 operating point).")
    p.add_argument("--amp", type=float, default=0.10,
                   help="Fractional sinusoidal amplitude (0.10 = 10 pct).")
    p.add_argument("--freq-hz", type=float, default=0.234,
                   help="Modulation frequency [Hz].")
    p.add_argument("--start", type=float, default=0.0,
                   help="Time [s] at which the modulation switches on.")
    p.add_argument("--ramp-s", type=float, default=0.0,
                   help="Duration [s] over which the sine amplitude is smoothly "
                        "ramped 0 -> full (smoothstep). 0 = abrupt onset. A ramp "
                        "avoids kicking the lightly damped tower mode with a "
                        "broadband onset transient.")
    p.add_argument("--stop", type=float, default=None,
                   help="Time [s] at which the modulation is switched off. The "
                        "sine amplitude is smoothly ramped 1 -> 0 over ramp-s "
                        "ending at stop, then held at zero so the torque returns "
                        "to the constant Region-2 value (normal operation). "
                        "None = modulation stays on until t-max.")
    p.add_argument("--t-max", type=float, default=300.0,
                   help="Last time breakpoint [s].")
    p.add_argument("--dt", type=float, default=0.05,
                   help="Time step of the breakpoint grid [s].")
    p.add_argument("--out", type=str,
                   default="test1002/ControlData/OLInput_ROSCO.dat")
    args = p.parse_args()

    t = np.arange(0.0, args.t_max + args.dt, args.dt)
    # Smoothstep amplitude envelope: 0 before start, ramps 0 -> 1 over ramp_s,
    # held at 1, then (if --stop is given) ramps 1 -> 0 over ramp_s ending at
    # stop and held at 0 afterwards (return to normal operation). The smooth
    # transitions remove the abrupt torque steps that would otherwise excite the
    # lightly damped tower mode with a broadband kick.
    if args.ramp_s > 0.0:
        u_up = np.clip((t - args.start) / args.ramp_s, 0.0, 1.0)
        env = u_up * u_up * (3.0 - 2.0 * u_up)
        if args.stop is not None:
            u_dn = np.clip((args.stop - t) / args.ramp_s, 0.0, 1.0)
            env = env * (u_dn * u_dn * (3.0 - 2.0 * u_dn))
    else:
        env = np.where(t >= args.start, 1.0, 0.0)
        if args.stop is not None:
            env = np.where(t >= args.stop, 0.0, env)
    factor = 1.0 + args.amp * env * np.sin(
        2.0 * np.pi * args.freq_hz * (t - args.start))
    factor = np.where(t >= args.start, factor, 1.0)
    gentq = args.mean * factor

    out_path = Path(args.out)
    write_ol_file(t, gentq, out_path)

    print(f"Wrote {len(t)} rows to {out_path}")
    print(f"  mean={args.mean:.4e} Nm, amp={args.amp:.3f}, "
          f"f={args.freq_hz} Hz, start={args.start} s, ramp={args.ramp_s} s, "
          f"stop={args.stop}, "
          f"range=[{gentq.min():.4e}, {gentq.max():.4e}] Nm")


if __name__ == "__main__":
    main()
