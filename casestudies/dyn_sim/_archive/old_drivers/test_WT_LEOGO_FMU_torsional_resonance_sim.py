r"""
Electrically driven drivetrain torsional resonance in the OpenFAST FMU model.

This is the FMU (high-fidelity) counterpart of
``test_WT_LEOGO_torsional_resonance_sim.py``.

The 3.49 Hz torsional mode does NOT live inside OpenFAST (its internal
drivetrain DOF, DrTrDOF, is disabled -- the real IEA-15MW shaft torsion sits
near 30 Hz, outside OpenFAST's range). Instead it lives in the co-simulation
wrapper ``FMUtoUICdrivetrain``: a generator mass (state ``omega_e``, inertia
H_e) on a soft shaft spring (K = K_original/100), with the rotor-side speed
``omega_m`` prescribed by the OpenFAST rotor (RotSpeed). Its natural frequency
is

    f_n = (1 / 2 pi) * sqrt(K_pu / (2 H_e)) ~ 3.49 Hz,

essentially identical to the simplified 2-mass model because H_m >> H_e.

The wrapper generator mass feels the grid electrical torque directly,
T_e = P_e / (eta * omega_e_filt), entirely on the TOPS side. It therefore
does NOT pass through ROSCO, so an oscillating shunt load at 3.49 Hz drives it
to resonance -- unlike the tower side-to-side mode, which is blocked by
ROSCO (VSContrl=5).

The script logs the wrapper shaft torque plus the OpenFAST tower-top
side-to-side acceleration (YawBrTAyp), so we can show the torsion resonates
while the tower does not respond to the electrical forcing.

On resonance:
  .\.venv\Scripts\python.exe casestudies\dyn_sim\test_WT_LEOGO_FMU_torsional_resonance_sim.py --forcing-freq-hz 3.49 --t-end 40
Off resonance:
  .\.venv\Scripts\python.exe casestudies\dyn_sim\test_WT_LEOGO_FMU_torsional_resonance_sim.py --forcing-freq-hz 1.0 --t-end 40
"""

from __future__ import annotations

import argparse
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

# Reuse the LEOGO + OpenFAST-FMU model assembly (this import also chdirs to
# the project root, which the FMU needs for its relative paths).
from casestudies.dyn_sim.test_WT_LEOGO_FMU_sim import build_model, scalar


def print_wrapper_drivetrain_constants(fmu_model) -> float:
    """Print the wrapper drivetrain pu constants; return the analytical f_n."""
    H_e = scalar(fmu_model.H_e)
    H_m = scalar(fmu_model.H_m)
    K_pu = scalar(fmu_model.par["K"])          # already pu (converted in __init__)
    D_pu = scalar(fmu_model.par["D"])          # already pu
    w_base = scalar(fmu_model._omega_base_rad_s)

    # Rotor speed is prescribed by the FMU, so the wrapper is a single
    # generator mass on the shaft spring: omega_n = sqrt(K_pu / (2 H_e)).
    w_n = np.sqrt(K_pu / (2.0 * H_e))
    zeta = D_pu / (2.0 * np.sqrt(2.0 * H_e * K_pu))
    f_n = w_n / (2.0 * np.pi)
    print("Wrapper drivetrain constants (from the running FMU model):")
    print(f"  omega_base = {w_base:.6f} rad/s")
    print(f"  H_m = {H_m:.4f} s,  H_e = {H_e:.5f} s")
    print(f"  K_pu = {K_pu:.4f},  D_pu = {D_pu:.4f}")
    print(f"  torsional (rotor speed prescribed): f_n = {f_n:.4f} Hz,  zeta = {100*zeta:.2f} %")
    print("  (simplified-model eigenvalue: 3.4909 Hz, 4.37 %)")
    return f_n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Electrically driven torsional resonance in the OpenFAST FMU model."
    )
    parser.add_argument("--forcing-freq-hz", type=float, default=3.49,
                        help="Electrical forcing frequency (Hz). Torsional mode = 3.49 Hz.")
    parser.add_argument("--t-end", type=float, default=40.0)
    parser.add_argument("--onset", type=float, default=10.0,
                        help="Time at which the electrical forcing switches on (s).")
    parser.add_argument("--load-amp-mw", type=float, default=2.0,
                        help="Peak amplitude of the shunt-load disturbance (MW).")
    parser.add_argument("--load-bus", choices=["wt", "main"], default="wt",
                        help="Inject at the WT terminal ('wt') or the main gen bus ('main').")
    parser.add_argument("--fmu", choices=["fast", "debug"], default="fast",
                        help="Which OpenFAST FMU to use. 'fast' (release build) is far "
                             "faster and avoids the per-step ROSCO .dbg writes that stall "
                             "on OneDrive-synced folders. 'debug' also exposes YawBrTAyp.")
    parser.add_argument("--shaft-damping-scale", type=float, default=1.0,
                        help="Scale factor on the wrapper shaft damping D_pu. The bare "
                             "co-simulation (scale 1.0) is net negative-damped at the "
                             "torsion mode -- the 0.01 s zero-order-hold coupling delay on "
                             "the artificially soft shaft (K=K_orig/100) makes the mode "
                             "self-excite. A scale of ~2.5 restores a positive net "
                             "(effective) damping so the mode behaves as a bounded FORCED "
                             "resonance, matching the reduced-order model.")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    t_wall_start = time.perf_counter()

    model = build_model()
    s_base_mva = float(model["base_mva"])

    # Optionally force the fast (release) FMU to avoid the debug FMU's per-step
    # debug-file writes, which stall non-deterministically inside OneDrive.
    if args.fmu == "fast":
        fast_fmu = PROJECT_ROOT / "fast.fmu"
        if fast_fmu.is_file():
            fmu_row = model["FMUtoUICdrivetrain"]["FMUtoUICdrivetrain"]
            header, values = fmu_row[0], fmu_row[1]
            values[header.index("FMU_path")] = str(fast_fmu)
            values[header.index("fmu_filename")] = ""
        else:
            print(f"WARNING: {fast_fmu} not found; falling back to build_model default FMU.")

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)
    fmu_model = ps.FMUtoUICdrivetrain["FMUtoUICdrivetrain"]
    uic_model = ps.vsc["UIC_sig"]
    gen_model = ps.gen["GEN"]

    # Optionally raise the wrapper shaft damping to counteract the co-simulation
    # coupling's negative-damping contribution, turning the self-excited torsion
    # mode into a bounded forced resonance (see --shaft-damping-scale help).
    if abs(args.shaft_damping_scale - 1.0) > 1e-12:
        fmu_model.par["D"][:] = fmu_model.par["D"] * args.shaft_damping_scale
        print(f"Wrapper shaft damping scaled by {args.shaft_damping_scale:.2f} "
              f"(D_pu now {scalar(fmu_model.par['D']):.4f}).")

    # Power-flow guess for the UIC reference (~6.6 MW at 8 m/s, Region 2).
    wt_s_n = float(fmu_model.par["S_n"][0])
    uic_s_n = float(uic_model.par["S_n"][0])
    uic_model.par["p_ref"][:] = 0.443 * wt_s_n / uic_s_n
    uic_model.par["q_ref"][:] = 0.0

    ps.power_flow()
    ps.init_dyn_sim()
    x0 = ps.x0.copy()

    # The OpenFAST FMU requires a fixed communication step = fmu_dt.
    dt = float(fmu_model._fmu_dt)

    print("\nInitialised LEOGO + OpenFAST FMU turbine (torsional forcing)")
    print(f"FMU communication step: {dt:.4f} s")
    f_n = print_wrapper_drivetrain_constants(fmu_model)
    print(f"Forcing frequency: {args.forcing_freq_hz:.3f} Hz "
          f"({'ON' if abs(args.forcing_freq_hz - f_n) < 0.15 else 'OFF'}-resonance)\n")

    # Bus to inject the electrical disturbance at.
    wt_bus_idx = int(uic_model.bus_idx_red["terminal"][0])
    main_bus_idx = int(gen_model.bus_idx_red["terminal"][0])
    inject_bus_idx = wt_bus_idx if args.load_bus == "wt" else main_bus_idx
    y_load_unit = args.load_amp_mw / s_base_mva

    # Wrapper drivetrain constants for the shaft-torque diagnostic.
    K_pu = scalar(fmu_model.par["K"])
    D_pu = scalar(fmu_model.par["D"])
    torque_base_nm = scalar(fmu_model._T_base_Nm)
    eta = float(fmu_model._efficiency)

    def load_scale(t: float) -> float:
        """Sinusoidal disturbance, switched on at args.onset."""
        if t < args.onset:
            return 0.0
        phase = 2.0 * np.pi * args.forcing_freq_hz * (t - args.onset)
        return np.sin(phase)

    def set_load(t: float) -> None:
        ps.y_bus_red_mod[(inject_bus_idx, inject_bus_idx)] = (
            load_scale(t) * y_load_unit
        )

    # The FMU is stepped once per macro-step (after the network step); its
    # cached outputs are read inside state_derivatives().
    def f_ode(t, x):
        set_load(t)
        v = ps.solve_algebraic(t, x)
        return ps.state_derivatives(t, x, v)

    solver = dps_sol.SimpleRK4(f_ode, 0.0, x0, args.t_end, max_step=dt)

    rows: list[dict[str, float]] = []

    def collect_row(t: float, x: np.ndarray, v: np.ndarray) -> None:
        X = fmu_model.local_view(x)
        omega_e = scalar(X["omega_e"])
        theta_s = scalar(X["theta_s"])
        omega_m = (omega_e if fmu_model._omega_m_pu_meas is None
                   else float(fmu_model._omega_m_pu_meas))
        omega_s = omega_m - omega_e
        t_shaft_pu = K_pu * theta_s + D_pu * omega_s

        p_e_wt_pu = fmu_model._pe_local_pu(x, v)
        t_e_pu = fmu_model._te_pu(X, p_e_wt_pu)

        v_terminal = uic_model.v_t(x, v)[0]
        s_uic = uic_model.s_e(x, v)[0]

        row = {
            "t": float(t),
            "forcing_freq_hz": args.forcing_freq_hz,
            "load_scale": load_scale(t),
            "omega_e_pu": omega_e,
            "omega_m_pu": omega_m,
            "omega_s_pu": omega_s,
            "theta_s_state": theta_s,
            "T_shaft_pu": float(t_shaft_pu),
            "T_shaft_Nm": float(t_shaft_pu * torque_base_nm),
            "T_e_pu": float(t_e_pu),
            "P_e_wt_pu": float(p_e_wt_pu),
            "P_uic_bus_pu": float(s_uic.real),
            "Q_uic_bus_pu": float(s_uic.imag),
            "V_wt_terminal_pu": float(abs(v_terminal)),
        }

        # OpenFAST outputs: tower-top side-to-side (YawBrTAyp) + fore-aft
        # (YawBrTAxp) accelerations, rotor/generator speed, generator torque.
        if hasattr(fmu_model, "get_all_fmu_outputs"):
            fmu_out = fmu_model.get_all_fmu_outputs()
            for key in ("RotSpeed", "GenSpeed", "GenTq",
                        "YawBrTAxp", "YawBrTAyp", "BldPitch1"):
                if key in fmu_out:
                    row[f"fmu_{key}"] = float(fmu_out[key])

        rows.append(row)

    # t = 0 sample.
    set_load(0.0)
    v0 = ps.solve_algebraic(0.0, x0)
    collect_row(0.0, x0, v0)

    aborted_at = None
    step_count = 0
    try:
        while solver.t < args.t_end:
            solver.step()
            x = solver.x
            t = solver.t

            # Guard: if the resonant response has diverged, the state contains
            # NaN/Inf or absurdly large values. Feeding either into the OpenFAST
            # FMU makes the DLL hang, so stop cleanly here and keep the data
            # collected up to this point.
            Xg = fmu_model.local_view(x)
            omega_e_g = scalar(Xg["omega_e"])
            theta_s_g = scalar(Xg["theta_s"])
            if (not np.all(np.isfinite(x))
                    or abs(omega_e_g - 1.0) > 0.5
                    or abs(theta_s_g) > 5.0):
                aborted_at = t
                print(f"\nDiverged state at t={t:.3f} s "
                      f"(omega_e={omega_e_g:.4f} pu, theta_s={theta_s_g:.4f}) -- "
                      f"stopping before the FMU call. Partial data kept.")
                break

            v = ps.solve_algebraic(t, x)

            # Leave the diagnostic torque/power modulations off (default 1.0).
            fmu_model._te_mod_factor = 1.0
            fmu_model._epc_mod_factor = 1.0

            # Advance the OpenFAST FMU one communication step, then cache outputs.
            fmu_model.step_fmu(x, v, t, dt)

            collect_row(t, x, v)

            step_count += 1
            if step_count % 100 == 0:
                wall = time.perf_counter() - t_wall_start
                rate = step_count / wall if wall > 0 else 0.0
                print(f"t={t:6.2f}s / {args.t_end:.0f}s   "
                      f"wall={wall:6.1f}s   {rate:5.1f} steps/s", flush=True)

        print("done integrating.")
    finally:
        if hasattr(fmu_model, "terminate_fmu"):
            try:
                fmu_model.terminate_fmu()
            except Exception as exc:  # noqa: BLE001
                print(f"terminate_fmu failed: {exc}")

    df = pd.DataFrame(rows)

    output_dir = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "csv_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.forcing_freq_hz:.2f}Hz".replace(".", "p", 1)
    out = args.out or str(output_dir / f"WT1_LEOGO_FMU_torsional_forcing_{tag}.csv")
    df.to_csv(out, index=False)
    print(f"\nsaved {out}  ({len(df)} rows)")

    if aborted_at is not None:
        print(f"NOTE: run aborted at t={aborted_at:.3f} s (diverged). "
              f"Reduce --load-amp-mw for a bounded steady-state response.")

    # Resonance diagnostic on the shaft torque.
    tt = df["t"].to_numpy()
    y = df["T_shaft_Nm"].to_numpy()
    post = tt >= args.onset
    yp = y[post] - np.mean(y[post]) if post.any() else np.array([])
    t_shaft_pp = float(np.ptp(yp)) if yp.size else 0.0

    ss = df.get("fmu_YawBrTAyp")
    ss_pp = float(np.ptp(ss.to_numpy()[post])) if ss is not None and post.any() else float("nan")

    print(f"T_shaft peak-to-peak (post-onset): {t_shaft_pp:.3e} Nm")
    if ss is not None:
        print(f"Tower side-to-side accel peak-to-peak: {ss_pp:.3e} m/s^2 "
              f"(should stay small -- ROSCO blocks the tower path)")
    print(f"Simulation wall time: {time.perf_counter() - t_wall_start:.1f} s")



if __name__ == "__main__":
    main()
