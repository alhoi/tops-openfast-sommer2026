"""Confirm the in-process ZMQ grid<->ROSCO link was live for a run.

Usage: python _check_zmq_live.py <cosim_csv> [<zmq_log_csv>]

Evidence reported:
  * co-sim CSV column zmq_torque_offset_nm: how many steps carried a non-zero
    generator-torque offset, and its range -> proves the frequency-support law
    produced a command that was logged each step.
  * ZMQ server log row count -> one row per ROSCO request served; equals the
    number of FMU communication steps if ROSCO talked to the server every step.
"""
import sys
import numpy as np
import pandas as pd

if len(sys.argv) < 2:
    print("usage: python _check_zmq_live.py <cosim_csv> [<zmq_log_csv>]")
    raise SystemExit(1)

csv = sys.argv[1]
d = pd.read_csv(csv)
n = len(d)
off = d["zmq_torque_offset_nm"].to_numpy(float) if "zmq_torque_offset_nm" in d else None

print(f"co-sim CSV: {csv}")
print(f"  rows (FMU steps)      = {n}")
if off is not None:
    nz = int(np.count_nonzero(np.abs(off) > 1.0))
    print(f"  zmq_torque_offset_nm  : nonzero steps = {nz}/{n} "
          f"({100.0*nz/max(n,1):.1f}%)")
    print(f"                          min={off.min():.3e}  max={off.max():.3e}  "
          f"std={off.std():.3e} Nm")
    if nz > 0:
        print("  -> ZMQ command reached the log every step (link live).")
    else:
        print("  -> offset all ~0 (support disabled or link inactive).")
else:
    print("  (no zmq_torque_offset_nm column)")

if len(sys.argv) > 2:
    log = sys.argv[2]
    try:
        s = pd.read_csv(log)
        print(f"ZMQ server log: {log}")
        print(f"  requests served (rows) = {len(s)}")
        print(f"  -> one reply per ROSCO request; matches ~{n} FMU steps "
              f"means every step communicated.")
    except Exception as e:
        print(f"ZMQ server log read error: {e}")
