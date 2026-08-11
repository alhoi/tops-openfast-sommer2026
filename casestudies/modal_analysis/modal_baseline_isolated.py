r"""
Isolated modal analysis of the BASELINE (simplified, analytic) wind-turbine
model.

The turbine + UIC converter are connected to a stiff INFINITE BUS instead of
the LEOGO grid (model: casestudies/ps_data/test_WT.py, generator 'IB' with
H = 1e5). Isolating the turbine this way removes the LEOGO gas-turbine and
network modes, so the eigenvalue spectrum shows the turbine's OWN
electromechanical modes --- most importantly the drivetrain torsional mode and
the converter/pitch modes.

Method: the baseline model is a set of differential-algebraic equations, so it
is linearised about its operating point and the system matrix is
eigen-decomposed. Each complex-conjugate eigenvalue pair gives a mode's natural
frequency f = Im(lambda)/2pi and damping ratio zeta = -Re(lambda)/|lambda|;
participation factors identify the states that dominate each mode.

NOTE on the tower modes: in the baseline model the tower side-to-side (~0.234 Hz)
and fore-aft (~0.235 Hz) modes are PRESCRIBED second-order oscillators calibrated
against OpenFAST (in windturbine_tower.py). They are not emergent from this
drivetrain/electrical linearisation; the drivetrain torsional mode below IS
emergent and is the mode that couples the electrical side to the structure.

Headless: prints a table and saves an s-plane figure.
    .\.venv\Scripts\python.exe casestudies\modal_analysis\modal_baseline_isolated.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops_openfast.dyn_models as ext_lib
import casestudies.ps_data.test_WT as model_data

# States that identify each mode family (matched as substrings of the descriptor).
DRIVETRAIN_KEYS = ("omega_m", "omega_e", "theta_m", "theta_e")
CONVERTER_KEYS = ("vi_x", "vi_y", "x_filter")


def state_label(desc) -> str:
    if isinstance(desc, (tuple, list, np.ndarray)):
        return " ".join(str(d) for d in desc)
    return str(desc)


def max_participation(pfs_abs, state_desc, mode_idx, keys) -> float:
    vals = [
        pfs_abs[i, mode_idx]
        for i, desc in enumerate(state_desc)
        if any(k in state_label(desc) for k in keys)
    ]
    return float(max(vals)) if vals else 0.0


def main() -> None:
    model = model_data.load()

    # The isolated test_WT data predates the droop parameters now required by the
    # WindTurbine model; append them (droop disabled) so the model constructs.
    _wt = model["windturbine"]["WindTurbine"]
    for _name, _val in (("f_nom_hz", 50.0), ("droop_enable", 0),
                        ("K_droop_pu_per_hz", 0.0), ("headroom_pu", 0.0)):
        if _name not in _wt[0]:
            _wt[0].append(_name)
            for _row in _wt[1:]:
                _row.append(_val)

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)

    # Set the UIC active-power reference from the turbine's MPT operating point
    # so the linearisation is taken about a physically consistent equilibrium.
    wt = ps.windturbine["WindTurbine"]
    uic = ps.vsc["UIC_sig"]
    wind = wt.wind_speed_init()
    uic.par["p_ref"][:] = wt.P_ref_from_wind(wind, uic.par["S_n"])
    uic.par["q_ref"][:] = 0.0

    ps.power_flow()
    ps.init_dyn_sim()

    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    eigs = ps_lin.eigs
    pfs = ps_lin.lev.T * ps_lin.rev
    pfs_abs = np.abs(pfs) / np.max(np.abs(pfs), axis=0)
    state_desc = ps.state_desc

    print("\n=== Isolated baseline WT (infinite bus) - oscillatory modes ===")
    print(f"  wind = {float(np.ravel(wind)[0]):.2f} m/s,  "
          f"{len(state_desc)} states\n")
    print(f"{'f [Hz]':>9}  {'zeta':>8}  {'tau [s]':>9}  {'family':>11}  "
          f"top-3 participating states")

    rows = []
    for i, lam in enumerate(eigs):
        if lam.imag <= 1e-9:
            continue  # keep one of each conjugate pair; skip real modes
        f_hz = lam.imag / (2.0 * np.pi)
        zeta = -lam.real / abs(lam)
        tau = (-1.0 / lam.real) if lam.real < 0 else float("inf")
        dt_pf = max_participation(pfs_abs, state_desc, i, DRIVETRAIN_KEYS)
        cv_pf = max_participation(pfs_abs, state_desc, i, CONVERTER_KEYS)
        family = "drivetrain" if dt_pf >= max(cv_pf, 0.5) else (
            "converter" if cv_pf >= 0.5 else "other")
        top3 = np.argsort(pfs_abs[:, i])[-3:][::-1]
        top3_str = ", ".join(
            f"{state_label(state_desc[j])}({pfs_abs[j, i]:.2f})" for j in top3)
        rows.append((f_hz, zeta, tau, family, dt_pf, lam, top3_str))

    for f_hz, zeta, tau, family, dt_pf, lam, top3_str in sorted(rows, key=lambda r: r[0]):
        print(f"{f_hz:9.4f}  {zeta:8.4f}  {tau:9.2f}  {family:>11}  {top3_str}")

    # Highlight the drivetrain torsional mode (highest drivetrain participation).
    tors = max(rows, key=lambda r: r[4]) if rows else None
    if tors is not None:
        f_hz, zeta, tau, family, dt_pf, lam, top3_str = tors
        print("\n=== Drivetrain torsional mode ===")
        print(f"  frequency     : {f_hz:.4f} Hz")
        print(f"  damping ratio : {zeta:.4f}  ({100 * zeta:.2f} %)")
        print(f"  time constant : {tau:.2f} s")
        print(f"  top states    : {top3_str}")

    print("\n  (Tower side-to-side ~0.234 Hz and fore-aft ~0.235 Hz are prescribed,"
          "\n   OpenFAST-calibrated modes and are not part of this linearisation.)")

    # --- s-plane figure -------------------------------------------------------
    out = PROJECT_ROOT / "results" / "modal" / "baseline_isolated_eigs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.scatter(eigs.real, eigs.imag / (2 * np.pi), s=28, c="#1f77b4", alpha=0.8)
    if tors is not None:
        ax.scatter([tors[5].real], [tors[5].imag / (2 * np.pi)], s=90,
                   facecolors="none", edgecolors="#d62728", linewidths=1.6,
                   label=f"drivetrain torsion ({tors[0]:.2f} Hz, {100*tors[1]:.1f}%)")
        ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("Real part [1/s]")
    ax.set_ylabel("Frequency  Im($\\lambda$)/2$\\pi$ [Hz]")
    ax.set_title("Isolated baseline WT (infinite bus) - eigenvalue spectrum")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved s-plane figure: {out}")


if __name__ == "__main__":
    main()
