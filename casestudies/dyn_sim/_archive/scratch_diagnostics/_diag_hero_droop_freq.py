"""Diagnostic: does the WT frequency-support droop lift the grid frequency?

Runs the hero grid event (10 MW permanent load step) with droop OFF and ON,
then compares the LEOGO COI grid frequency (nadir + settled) and the WT
electrical power. Used to explain why proportional droop reduces but does not
remove the steady-state frequency offset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from casestudies.dyn_sim.plot_em_interaction_hero import run_event_series

ONSET = 10.0
T_END = 70.0
DT = 0.01
WIND = 10.0
LOAD_MW = 10.0
S_N_WT_MVA = 20.0
HEADROOM_PU = 0.05


def summarize(res: dict[str, np.ndarray]) -> dict[str, float]:
    t = res["t"]
    f = res["grid_freq_hz"]
    p = res["p_e_pu"]
    pre = t < ONSET
    settle = t >= (T_END - 5.0)
    base_f = float(np.mean(f[pre]))
    return {
        "base_f": base_f,
        "nadir_f": float(np.min(f[t >= ONSET])),
        "settled_f": float(np.mean(f[settle])),
        "base_p": float(np.mean(p[pre])),
        "peak_p": float(np.max(p[t >= ONSET])),
        "settled_p": float(np.mean(p[settle])),
    }


def main() -> None:
    print("Running droop OFF ...")
    off = summarize(run_event_series(wind_mps=WIND, onset=ONSET, t_end=T_END,
                                     dt=DT, load_step_mw=LOAD_MW, droop_enable=0))
    print("Running droop ON ...")
    on = summarize(run_event_series(wind_mps=WIND, onset=ONSET, t_end=T_END,
                                    dt=DT, load_step_mw=LOAD_MW, droop_enable=1))

    def mhz(x: float) -> float:
        return 1000.0 * x

    print("\n=== GRID FREQUENCY (LEOGO COI) ===")
    print(f"  baseline (pre-onset):   off {off['base_f']:.4f} Hz | on {on['base_f']:.4f} Hz")
    print(f"  nadir:                  off {off['nadir_f']:.4f} Hz | on {on['nadir_f']:.4f} Hz")
    print(f"  settled (last 5 s):     off {off['settled_f']:.4f} Hz | on {on['settled_f']:.4f} Hz")
    print(f"  nadir DIP vs base:      off {mhz(off['base_f']-off['nadir_f']):.1f} mHz | "
          f"on {mhz(on['base_f']-on['nadir_f']):.1f} mHz")
    print(f"  settled OFFSET vs base: off {mhz(off['base_f']-off['settled_f']):.1f} mHz | "
          f"on {mhz(on['base_f']-on['settled_f']):.1f} mHz")
    print(f"  droop lift (on-off):    nadir {mhz(on['nadir_f']-off['nadir_f']):+.1f} mHz | "
          f"settled {mhz(on['settled_f']-off['settled_f']):+.1f} mHz")

    print("\n=== WT ELECTRICAL POWER (pu on 20 MVA WT base) ===")
    print(f"  baseline:  off {off['base_p']:.4f} | on {on['base_p']:.4f} pu")
    print(f"  peak:      off {off['peak_p']:.4f} | on {on['peak_p']:.4f} pu")
    print(f"  settled:   off {off['settled_p']:.4f} | on {on['settled_p']:.4f} pu")
    dp_settled = on["settled_p"] - off["settled_p"]
    dp_peak = on["peak_p"] - off["peak_p"]
    print(f"  droop extra power (on-off): settled {dp_settled:+.4f} pu "
          f"({dp_settled*S_N_WT_MVA:+.2f} MW) | peak {dp_peak*S_N_WT_MVA:+.2f} MW")
    print(f"  headroom cap: {HEADROOM_PU:.3f} pu = {HEADROOM_PU*S_N_WT_MVA:.2f} MW "
          f"(max the droop can add above the de-loaded base)")


if __name__ == "__main__":
    main()
