r"""
Robustness check of the very lightly damped ~5.2 Hz LEOGO mode.

The coupled WT+LEOGO eigenanalysis (participation_WT_LEOGO_torsional.py) reports
a pair of near-degenerate modes at ~5.204 Hz with zeta = 0.26 %, dominated by
the synchronous generators' d-axis EMF states (e_d_t, e_d_st).  Because those
states are governed by the q-axis machine data, and the LEOGO data set has a
NON-PHYSICAL q-axis ordering (X_q_t = 0.01 < X_q_st = 0.159, i.e. the transient
reactance is smaller than the subtransient one), this script asks: is the
0.26 % damping a genuine physical mode, or an artefact of that data?

For each generator-data variant we rebuild the coupled model, linearise it, and
report the mode with the largest participation on the generator e_d states
(the "5.2 Hz mode"): its frequency, damping ratio and real part.

Headless.  Run:
    .\.venv\Scripts\python.exe casestudies\modal_analysis\verify_5p2Hz_mode.py
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops_openfast.dyn_models as ext_lib

from casestudies.dyn_sim.test_WT_LEOGO_sim import build_model

# Column indices in the GEN row (see LEOGO_ps.py header).
COL = {
    "H": 6, "D": 7, "X_d": 8, "X_q": 9, "X_d_t": 10, "X_q_t": 11,
    "X_d_st": 12, "X_q_st": 13, "T_d0_t": 14, "T_q0_t": 15,
    "T_d0_st": 16, "T_q0_st": 17,
}

# States that define the ~5.2 Hz mode (generator d-axis EMFs).
ED_KEYS = ("e_d_t", "e_d_st")


def state_label(desc) -> str:
    if isinstance(desc, (tuple, list, np.ndarray)):
        return " ".join(str(d) for d in desc)
    return str(desc)


def build_model_with_gen_overrides(overrides: dict[str, float]):
    """Build the coupled model, overriding named GEN columns on ALL generators.

    `overrides` maps a column name in COL to a new value (applied to every
    synchronous generator row).
    """
    model = build_model()
    gen_block = model["generators"]["GEN"]
    header, rows = gen_block[0], gen_block[1:]
    new_rows = []
    for row in rows:
        row = list(row)
        for name, value in overrides.items():
            row[COL[name]] = value
        new_rows.append(row)
    model["generators"]["GEN"] = [header] + new_rows
    return model


def find_ed_mode(ps, ps_lin):
    """Return (f_hz, zeta, real, top3_str) for the oscillatory mode with the
    largest generator e_d participation."""
    eigs = ps_lin.eigs
    pfs = ps_lin.lev.T * ps_lin.rev
    pfs_abs = np.abs(pfs) / np.max(np.abs(pfs), axis=0)
    state_desc = ps.state_desc

    ed_idx = [
        i for i, d in enumerate(state_desc)
        if any(k in state_label(d) for k in ED_KEYS)
    ]

    best = None
    for i, lam in enumerate(eigs):
        if lam.imag <= 1e-6:
            continue
        ed_part = float(np.sum(pfs_abs[ed_idx, i]))
        if best is None or ed_part > best[0]:
            f_hz = lam.imag / (2.0 * np.pi)
            zeta = -lam.real / abs(lam)
            top3 = np.argsort(pfs_abs[:, i])[-3:][::-1]
            top3_str = ", ".join(
                f"{state_label(state_desc[j])}({pfs_abs[j, i]:.2f})"
                for j in top3
            )
            best = (ed_part, f_hz, zeta, lam.real, top3_str)
    if best is None:
        return None
    _, f_hz, zeta, re, top3_str = best
    return f_hz, zeta, re, top3_str


def run_variant(label: str, overrides: dict[str, float]) -> None:
    model = build_model_with_gen_overrides(overrides)
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)
    ps.power_flow()
    ps.init_dyn_sim()
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    res = find_ed_mode(ps, ps_lin)
    if res is None:
        print(f"{label:<34}  (no oscillatory e_d mode found)")
        return
    f_hz, zeta, re, top3_str = res
    tau = (-1.0 / re) if re < 0 else float("inf")
    print(f"{label:<34}  f={f_hz:7.3f} Hz  zeta={100*zeta:6.2f} %  "
          f"real={re:8.4f}  tau={tau:6.2f} s")
    print(f"{'':<34}  top: {top3_str}")


def main() -> None:
    print("Baseline LEOGO q-axis data: X_q=2.1, X_q_t=0.01, X_q_st=0.159 "
          "(X_q_t < X_q_st is non-physical)\n")
    print("=== ~5.2 Hz generator d-axis mode under generator-data variants ===")

    run_variant("baseline (as-is)", {})
    # Fix the non-physical q-axis ordering: make X_q_t > X_q_st.
    run_variant("X_q_t 0.01 -> 0.40 (physical)", {"X_q_t": 0.40})
    run_variant("X_q_t 0.01 -> 0.30", {"X_q_t": 0.30})
    run_variant("X_q_t 0.01 -> 0.20", {"X_q_t": 0.20})
    # More generator (mechanical) damping.
    run_variant("D 5 -> 10", {"D": 10.0})
    run_variant("D 5 -> 20", {"D": 20.0})
    # Longer q-axis subtransient open-circuit time constant.
    run_variant("T_q0_st 0.013 -> 0.05", {"T_q0_st": 0.05})
    run_variant("T_q0_st 0.013 -> 0.10", {"T_q0_st": 0.10})
    # Combined physical fix.
    run_variant("X_q_t=0.40 + T_q0_st=0.05", {"X_q_t": 0.40, "T_q0_st": 0.05})


if __name__ == "__main__":
    main()
