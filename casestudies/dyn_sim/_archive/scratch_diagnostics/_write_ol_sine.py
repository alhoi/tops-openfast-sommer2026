"""Write a sinusoidal ROSCO open-loop generator-torque table.

Small helper so the open-loop table can be (re)written from the command line
without embedding quoted Python in PowerShell (which trips the shell). Wraps
write_sine_ol_table() from sweep_resonance.py.

Usage:
    python casestudies/dyn_sim/_write_ol_sine.py --freq 3.1 --amp 0.10 \
        --t-start 5 --t-max 30
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from casestudies.dyn_sim.sweep_resonance import write_sine_ol_table, OL_FILE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freq", type=float, required=True,
                   help="Modulation frequency [Hz].")
    p.add_argument("--amp", type=float, default=0.10,
                   help="Fractional torque-modulation amplitude.")
    p.add_argument("--t-start", type=float, default=5.0,
                   help="Time [s] at which the modulation switches on.")
    p.add_argument("--t-max", type=float, default=30.0,
                   help="End time [s] of the table.")
    args = p.parse_args()

    write_sine_ol_table(args.freq, args.amp, args.t_start, args.t_max)
    print(f"Wrote OL table: f={args.freq} Hz, amp={args.amp}, "
          f"t_start={args.t_start}, t_max={args.t_max} -> {OL_FILE}")


if __name__ == "__main__":
    main()
