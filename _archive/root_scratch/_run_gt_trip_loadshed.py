"""
Run the LEOGO GT-trip + under-frequency load-shedding scenario, with the wind
turbine's frequency-support droop OFF and ON, and build the comparison figure.

This reproduces the reference dynamic event from the LEOGO paper (Svendsen
et al. 2023, IET Energy Syst. Integr.): a gas-turbine trip followed ~200 ms
later by shedding ~9 MW of water-injection pumps. It is the paper's own
canonical disturbance, so it is the most defensible large-signal event.

Modelling choices (documented honestly):
  * Generation loss is represented as an equivalent permanent load step at the
    main gas-turbine busbar (net power balance), the same method as the earlier
    GT-trip runs. All three gensets keep responding, so the dip is a little
    optimistic versus a genuine single-machine trip.
  * Load shedding is a permanent 9 MW load reduction at the same busbar,
    switched in 200 ms after the trip with a fast breaker-like ramp.
  * Frequency support ON/OFF share the same headroom (de-loaded reserve), so
    the difference isolates the droop *response*.

Outputs go to results/em_interaction_sweep/gt_trip_loadshed/.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

DRIVER = "casestudies/dyn_sim/test_WT_LEOGO_tower_sim.py"
ANALYZER = "casestudies/dyn_sim/_analyze_gt_trip_loadshed.py"

OUT_DIR = ROOT / "results" / "em_interaction_sweep" / "gt_trip_loadshed"

# LEOGO-faithful event parameters.
#   * Lost generation: one 15.8 MW genset (each GEN in LEOGO_ps.py is 15.804 MW).
#   * Load shed: 9 MW = 3 x water-injection pumps (WIN, 3 MW each), 200 ms later.
#   * Matched headroom 0.05 pu (1 MW on the 20 MVA UIC base) in both runs.
EVENT_TIME = 10.0
SHED_DELAY = 0.2
T_END = 60.0

COMMON = [
    "--load-step-mw", "15.8",     # lost generation (one 15.8 MW GT)
    "--event-time", str(EVENT_TIME),
    "--event-duration", str(T_END),   # >= (t_end - event_time) -> permanent
    "--load-ramp-on-s", "0.02",   # fast breaker action
    "--load-ramp-off-s", "0",
    "--load-shed-mw", "9.0",      # 3 x WIN water-injection pumps
    "--load-shed-delay-s", str(SHED_DELAY),
    "--t-end", str(T_END),
    "--dt", "0.005",
    "--headroom", "0.05",         # matched de-loaded reserve in both runs
    # Recalibrated tower side-to-side mode (matches the genuine OpenFAST
    # lightly-damped SS response used in the other reduced-model figures).
    "--zeta-ss", "0.0034",
    "--g-ss", "0.01825",
]

OFF_CSV = OUT_DIR / "gt_trip_loadshed_off.csv"
ON_CSV = OUT_DIR / "gt_trip_loadshed_on.csv"
FIG = OUT_DIR / "gt_trip_loadshed.png"


def run(cmd: list[str]) -> None:
    """Run a subprocess and raise if it fails."""
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # 1. Frequency support OFF
    print("=== GT-trip + load-shed: frequency support OFF ===")
    run([PY, "-u", DRIVER, *COMMON, "--out", str(OFF_CSV)])

    # 2. Frequency support ON (same headroom, droop enabled)
    print("\n=== GT-trip + load-shed: frequency support ON ===")
    run([PY, "-u", DRIVER, *COMMON, "--wt-droop", "--k-droop", "0.75",
         "--out", str(ON_CSV)])

    # 3. Comparison figure + text summary
    print("\n=== Building comparison figure ===")
    run([PY, "-u", ANALYZER,
         "--off-csv", str(OFF_CSV),
         "--on-csv", str(ON_CSV),
         "--event-time", str(EVENT_TIME),
         "--shed-delay", str(SHED_DELAY),
         "--out", str(FIG)])

    print(f"\nDone in {time.perf_counter() - t0:.1f} s")
    print(f"CSVs : {OFF_CSV.name}, {ON_CSV.name}")
    print(f"Figure: {FIG}")


if __name__ == "__main__":
    main()
