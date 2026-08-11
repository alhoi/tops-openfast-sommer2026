r"""
Diagnostic: is the ~3.8 Hz wrapper-drivetrain oscillation in the LEOGO +
OpenFAST-FMU co-simulation a genuine self-excited limit cycle, and is it
sensitive to the Python-side integration step?

Two questions:

  1. SUSTAIN vs DECAY.  Force the shaft at 3.49 Hz only during a short window
     [t_force_on, t_force_off], then switch the electrical pulsation OFF and
     watch the shaft torque.  If the 3.5-4.0 Hz content rings DOWN to zero the
     mode is stable (the overlap in fmu_pulsation_torsion.png would just be a
     forced response).  If it SUSTAINS at roughly constant amplitude the mode
     is self-excited (a limit cycle) -- which is what makes the on- and
     off-resonance traces overlap.

  2. dt SENSITIVITY.  The OpenFAST FMU communication step is locked to the
     OpenFAST internal DT = 0.01 s, so it cannot be reduced.  What we CAN
     refine is the Python-side RK4 step of the wrapper/network ODE, holding the
     FMU rotor speed omega_m constant (zero-order hold) across each 0.01 s FMU
     window.  We run n_sub = 1 (RK4 at 0.01 s, as in the production sim) and
     n_sub = 4 (RK4 at 0.0025 s).  If the sustained amplitude barely changes,
     the limit cycle is NOT a plain ODE-integration error -- it comes from the
     0.01 s co-simulation coupling delay acting on the artificially soft,
     lightly damped shaft (K = K_original/100).

Usage
-----
  .\.venv\Scripts\python.exe casestudies\dyn_sim\_diagnose_fmu_torsion_dt.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tops.dynamic as dps
import tops.solvers as dps_sol
import tops_openfast.dyn_models as ext_lib

from casestudies.dyn_sim.test_WT_LEOGO_FMU_sim import build_model, scalar

FMU_DT = 0.01
F_FORCE_HZ = 3.49
FORCE_AMP_MW = 0.5
T_FORCE_ON = 1.0
T_FORCE_OFF = 6.0
T_END = 30.0
BAND = (3.0, 4.5)          # Hz, the wrapper-torsion band


def _force_fast_fmu(model) -> None:
    """Point the FMU block at the fast (release) FMU to avoid the debug stall."""
    fast_fmu = PROJECT_ROOT / "fast.fmu"
    if not fast_fmu.is_file():
        print(f"WARNING: {fast_fmu} not found; using build_model default FMU.")
        return
    fmu_row = model["FMUtoUICdrivetrain"]["FMUtoUICdrivetrain"]
    header, values = fmu_row[0], fmu_row[1]
    values[header.index("FMU_path")] = str(fast_fmu)
    values[header.index("fmu_filename")] = ""


def run_case(n_sub: int) -> pd.DataFrame:
    """Run one co-simulation with n_sub RK4 sub-steps per 0.01 s FMU window."""
    model = build_model()
    _force_fast_fmu(model)
    s_base_mva = float(model["base_mva"])

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)
    fmu_model = ps.FMUtoUICdrivetrain["FMUtoUICdrivetrain"]
    uic_model = ps.vsc["UIC_sig"]
    gen_model = ps.gen["GEN"]

    wt_s_n = float(fmu_model.par["S_n"][0])
    uic_s_n = float(uic_model.par["S_n"][0])
    uic_model.par["p_ref"][:] = 0.443 * wt_s_n / uic_s_n
    uic_model.par["q_ref"][:] = 0.0

    ps.power_flow()
    ps.init_dyn_sim()
    x0 = ps.x0.copy()

    K_pu = scalar(fmu_model.par["K"])
    D_pu = scalar(fmu_model.par["D"])
    torque_base_nm = scalar(fmu_model._T_base_Nm)

    main_bus_idx = int(gen_model.bus_idx_red["terminal"][0])
    y_load_unit = FORCE_AMP_MW / s_base_mva

    def load_scale(t: float) -> float:
        if T_FORCE_ON <= t < T_FORCE_OFF:
            return np.sin(2.0 * np.pi * F_FORCE_HZ * (t - T_FORCE_ON))
        return 0.0

    def set_load(t: float) -> None:
        ps.y_bus_red_mod[(main_bus_idx, main_bus_idx)] = load_scale(t) * y_load_unit

    def f_ode(t, x):
        set_load(t)
        v = ps.solve_algebraic(t, x)
        return ps.state_derivatives(t, x, v)

    dt_sub = FMU_DT / n_sub
    solver = dps_sol.SimpleRK4(f_ode, 0.0, x0, T_END, max_step=dt_sub)

    def shaft_nm(x, v) -> float:
        X = fmu_model.local_view(x)
        omega_e = scalar(X["omega_e"])
        theta_s = scalar(X["theta_s"])
        omega_m = (omega_e if fmu_model._omega_m_pu_meas is None
                   else float(fmu_model._omega_m_pu_meas))
        return float((K_pu * theta_s + D_pu * (omega_m - omega_e)) * torque_base_nm)

    rows: list[dict[str, float]] = []
    set_load(0.0)
    v0 = ps.solve_algebraic(0.0, x0)
    rows.append({"t": 0.0, "load_scale": load_scale(0.0), "T_shaft_Nm": shaft_nm(x0, v0)})

    t_wall = time.perf_counter()
    win = 0
    while solver.t < T_END - 1e-9:
        # Advance one 0.01 s FMU window with n_sub RK4 sub-steps (omega_m held).
        for _ in range(n_sub):
            solver.step()
        x = solver.x
        t = solver.t
        if not np.all(np.isfinite(x)):
            print(f"  [n_sub={n_sub}] non-finite state at t={t:.3f}; stopping.")
            break
        v = ps.solve_algebraic(t, x)
        fmu_model._te_mod_factor = 1.0
        fmu_model._epc_mod_factor = 1.0
        fmu_model.step_fmu(x, v, t, FMU_DT)
        rows.append({"t": float(t), "load_scale": load_scale(t),
                     "T_shaft_Nm": shaft_nm(x, v)})
        win += 1
        if win % 300 == 0:
            wall = time.perf_counter() - t_wall
            print(f"  [n_sub={n_sub}] t={t:5.1f}s  wall={wall:5.1f}s  "
                  f"{win / wall:4.0f} win/s", flush=True)

    if hasattr(fmu_model, "terminate_fmu"):
        try:
            fmu_model.terminate_fmu()
        except Exception as exc:  # noqa: BLE001
            print(f"  terminate_fmu failed: {exc}")

    return pd.DataFrame(rows)


def band_amp(df: pd.DataFrame, t_lo: float, t_hi: float) -> tuple[float, float]:
    """Single-sided FFT peak amplitude [kNm] and its frequency in BAND."""
    t = df["t"].to_numpy()
    y = df["T_shaft_Nm"].to_numpy() / 1.0e3
    m = (t >= t_lo) & (t <= t_hi)
    ts, ys = t[m], y[m] - y[m].mean()
    w = np.hanning(len(ys))
    sp = np.abs(np.fft.rfft(ys * w)) * 2.0 / w.sum()
    fr = np.fft.rfftfreq(len(ys), float(np.median(np.diff(ts))))
    b = (fr >= BAND[0]) & (fr <= BAND[1])
    if not b.any():
        return 0.0, float("nan")
    i = np.argmax(sp[b])
    return float(sp[b][i]), float(fr[b][i])


def main() -> None:
    print("=" * 68)
    print("FMU wrapper-torsion diagnostic: sustain-vs-decay + dt sensitivity")
    print(f"  force {F_FORCE_HZ:.2f} Hz @ {FORCE_AMP_MW:.2f} MW during "
          f"[{T_FORCE_ON:.0f}, {T_FORCE_OFF:.0f}] s, then OFF; t_end={T_END:.0f} s")
    print("=" * 68)

    results = {}
    for n_sub in (1, 4):
        print(f"\n--- n_sub = {n_sub}  (RK4 step {FMU_DT / n_sub * 1e3:.1f} ms, "
              f"FMU step {FMU_DT * 1e3:.0f} ms) ---")
        t0 = time.perf_counter()
        df = run_case(n_sub)
        early = band_amp(df, T_FORCE_OFF + 1.0, T_FORCE_OFF + 6.0)   # [7, 12]
        late = band_amp(df, T_END - 8.0, T_END)                      # [22, 30]
        results[n_sub] = (early, late)
        print(f"  wall = {time.perf_counter() - t0:.1f} s")
        print(f"  early [7-12 s] band peak  = {early[0]:8.1f} kNm @ {early[1]:.3f} Hz")
        print(f"  late  [22-30 s] band peak = {late[0]:8.1f} kNm @ {late[1]:.3f} Hz")
        print(f"  late/early amplitude ratio = {late[0] / max(early[0], 1e-9):.3f}")

    print("\n" + "=" * 68)
    print("SUMMARY")
    for n_sub, (early, late) in results.items():
        print(f"  n_sub={n_sub}: after forcing OFF, band peak {early[0]:.0f} -> "
              f"{late[0]:.0f} kNm  (late/early {late[0] / max(early[0], 1e-9):.2f})")
    e1 = results[1][1][0]
    e4 = results[4][1][0]
    print(f"  dt refinement (late peak): n_sub=1 {e1:.0f} kNm vs n_sub=4 {e4:.0f} kNm  "
          f"(ratio {e4 / max(e1, 1e-9):.2f})")
    print("  Interpretation:")
    print("   - late/early ~1 or growing => self-excited limit cycle (sustains w/o forcing)")
    print("   - late/early << 1          => stable, only a decaying forced transient")
    print("   - n_sub=4 ~ n_sub=1        => not an ODE-step error; co-sim coupling / soft shaft")
    print("=" * 68)


if __name__ == "__main__":
    main()
