r"""
Headless eigen-/participation analysis of the simplified LEOGO wind-turbine
model (the exact model built by build_model() in test_WT_LEOGO_sim.py, i.e.
the same model used by the electrical-forcing diagnostics script).

Goal: locate the drivetrain torsional mode, report its frequency (Hz) and
damping ratio, and confirm it is generator-dominated (participation on
omega_e / theta_e) so it is observable/controllable from the electrical side.

No GUI: prints a text table. Run:
    .\.venv\Scripts\python.exe casestudies\modal_analysis\participation_WT_LEOGO_torsional.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops_openfast.dyn_models as ext_lib

from casestudies.dyn_sim.test_WT_LEOGO_sim import build_model


# States that define the WT drivetrain torsional mode.
DRIVETRAIN_STATE_KEYS = ("omega_m", "omega_e", "theta_m", "theta_e")


def state_label(desc) -> str:
    """Return a flat string for a state descriptor (tuple or str)."""
    if isinstance(desc, (tuple, list, np.ndarray)):
        return " ".join(str(d) for d in desc)
    return str(desc)


def drivetrain_participation(pfs_abs, state_desc, mode_idx) -> float:
    """Max participation of any drivetrain state in the given mode."""
    vals = [
        pfs_abs[i, mode_idx]
        for i, desc in enumerate(state_desc)
        if any(k in state_label(desc) for k in DRIVETRAIN_STATE_KEYS)
    ]
    return float(max(vals)) if vals else 0.0


def main() -> None:
    model = build_model()
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)
    ps.power_flow()
    ps.init_dyn_sim()

    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    eigs = ps_lin.eigs
    pfs = ps_lin.lev.T * ps_lin.rev
    pfs_abs = np.abs(pfs) / np.max(np.abs(pfs), axis=0)
    state_desc = ps.state_desc

    print("\n=== All states in the linearized LEOGO WT model ===")
    for i, desc in enumerate(state_desc):
        print(f"  [{i:3d}] {state_label(desc)}")

    # Build per-mode summary (unique modes: keep only positive imaginary part
    # of each complex-conjugate pair, plus real modes).
    print("\n=== Oscillatory modes (freq > 0), sorted by frequency ===")
    print(f"{'idx':>4}  {'f [Hz]':>9}  {'zeta':>8}  {'real':>10}  "
          f"{'drivetrain PF':>13}  top-3 participating states")

    rows = []
    for i, lam in enumerate(eigs):
        if lam.imag <= 1e-9:
            continue  # skip real modes and negative-imag conjugate
        f_hz = lam.imag / (2.0 * np.pi)
        zeta = -lam.real / abs(lam)
        dt_pf = drivetrain_participation(pfs_abs, state_desc, i)
        top3 = np.argsort(pfs_abs[:, i])[-3:][::-1]
        top3_str = ", ".join(
            f"{state_label(state_desc[j])}({pfs_abs[j, i]:.2f})" for j in top3
        )
        rows.append((f_hz, zeta, lam.real, dt_pf, top3_str))

    for f_hz, zeta, re, dt_pf, top3_str in sorted(rows, key=lambda r: r[0]):
        print(f"{'':>4}  {f_hz:9.4f}  {zeta:8.4f}  {re:10.4f}  "
              f"{dt_pf:13.3f}  {top3_str}")

    # Identify the torsional mode: the oscillatory mode with the highest
    # drivetrain participation.
    if rows:
        tors = max(rows, key=lambda r: r[3])
        f_hz, zeta, re, dt_pf, top3_str = tors
        print("\n=== Likely drivetrain TORSIONAL mode ===")
        print(f"  frequency      : {f_hz:.4f} Hz")
        print(f"  damping ratio  : {zeta:.4f}  ({100*zeta:.2f} %)")
        print(f"  real part      : {re:.4f}  (time const {(-1.0/re) if re < 0 else float('inf'):.2f} s)")
        print(f"  drivetrain PF  : {dt_pf:.3f}")
        print(f"  top states     : {top3_str}")
        if zeta > 0.3:
            print("\n  NOTE: damping ratio is high -> torsional resonance will be weak.")
            print("        Consider reducing shaft damping D in build_model() to")
            print("        obtain a lightly damped, resonant mode.")


if __name__ == "__main__":
    main()
