"""
Build a ROSCO open-loop generator-torque file from a recorded grid-frequency
trace, using a proportional (droop) frequency-support law.

This is the "pass 2" input generator for the droop demo:

  pass 1:  run test_WT_LEOGO_FMU_sim.py with a real grid event (a load step),
           ROSCO open-loop OFF, and log f_grid_hz(t) to the results CSV.
  here:    read that CSV, convert the frequency deviation into a droop torque
           command, and write OLInput_ROSCO.dat.
  pass 2:  set OL_Mode=1 and re-run so the OpenFAST structure feels the
           grid-derived torque, then inspect the side-to-side response.

Droop law (open-loop / feed-forward approximation):

    df(t)  = f_grid(t) - f_ref                      [Hz]
    dP(t)  = -(1/R) * (df/f_n) * P0                  [W]      (P0 = mean power)
    T(t)   = (P0 + dP(t)) / omega_gen                [Nm]
           ~ T0 * (1 - (1/R) * df/f_n)

with droop R (per-unit, e.g. 0.05 = 5 %). A frequency dip (df < 0) raises the
commanded torque, i.e. the turbine supports the grid. This is a one-way replay:
the recorded f_grid did NOT include this droop feedback, so the demo shows the
excitation path (grid -> torque -> structure), not the closed-loop support.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from make_ol_gentq import write_ol_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str,
                   default="casestudies/dyn_sim/results/WT1_LEOGO_FMU_results.csv",
                   help="Pass-1 results CSV containing t and f_grid_hz.")
    p.add_argument("--mean-torque", type=float, required=True,
                   help="Baseline generator torque T0 [Nm] (Region-2 point).")
    p.add_argument("--droop", type=float, default=0.05,
                   help="Per-unit droop R (0.05 = 5 pct).")
    p.add_argument("--f-n", type=float, default=50.0,
                   help="Nominal grid frequency [Hz].")
    p.add_argument("--f-ref", type=float, default=None,
                   help="Reference frequency [Hz]. Default: pre-event mean.")
    p.add_argument("--ref-window", type=float, default=5.0,
                   help="Seconds at the start used to set f_ref (pre-event).")
    p.add_argument("--gain", type=float, default=1.0,
                   help="Extra scalar on the droop response (1.0 = physical).")
    p.add_argument("--t-max", type=float, default=None,
                   help="Truncate the OL table at this time [s].")
    p.add_argument("--dt", type=float, default=0.05,
                   help="Resample step of the OL breakpoint grid [s].")
    p.add_argument("--out", type=str,
                   default="test1002/ControlData/OLInput_ROSCO.dat")
    args = p.parse_args()

    d = pd.read_csv(args.csv)
    tcol = "t" if "t" in d.columns else d.columns[0]
    if "f_grid_hz" not in d.columns:
        raise SystemExit(
            f"'f_grid_hz' not found in {args.csv}. Columns: {list(d.columns)}"
        )

    t_raw = d[tcol].to_numpy()
    f_raw = d["f_grid_hz"].to_numpy()

    # Resample onto a uniform grid for the OL table.
    t_max = args.t_max if args.t_max is not None else float(t_raw[-1])
    t = np.arange(0.0, t_max + args.dt, args.dt)
    f = np.interp(t, t_raw, f_raw)

    # Reference frequency: explicit, or the average over the first ref-window.
    if args.f_ref is not None:
        f_ref = args.f_ref
    else:
        pre = t < args.ref_window
        f_ref = float(f[pre].mean()) if pre.any() else float(f[0])

    df = f - f_ref
    # dP/P0 = -(1/R) * df/f_n  ->  T = T0 * (1 + gain * dP/P0)
    dp_over_p = -(1.0 / args.droop) * (df / args.f_n)
    gentq = args.mean_torque * (1.0 + args.gain * dp_over_p)

    write_ol_file(t, gentq, args.out)

    print(f"Wrote {len(t)} rows to {args.out}")
    print(f"  f_ref={f_ref:.4f} Hz, droop R={args.droop}, gain={args.gain}")
    print(f"  df range   = [{df.min():+.4f}, {df.max():+.4f}] Hz")
    print(f"  GenTq range= [{gentq.min():.4e}, {gentq.max():.4e}] Nm "
          f"(T0={args.mean_torque:.4e})")
    print(f"  GenTq p2p  = {gentq.max() - gentq.min():.4e} Nm "
          f"({100*(gentq.max()-gentq.min())/args.mean_torque:.2f}% of T0)")


if __name__ == "__main__":
    main()
