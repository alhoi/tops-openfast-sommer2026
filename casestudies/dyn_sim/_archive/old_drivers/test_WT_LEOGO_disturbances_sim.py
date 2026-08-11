"""
LEOGO wind-turbine frequency support under different platform disturbances.

This extends the load-step study (test_WT_LEOGO_droop_comparison_diagnostics.py)
by letting the LEOGO platform itself experience a realistic disturbance while
the load step is (optionally) applied. Every scenario is run twice: once with
the internal WT droop disabled and once with it enabled, so the value of the
droop response can be compared for each disturbance type.

Available disturbances (--disturbance):

    none            Only the configurable load step is applied (baseline,
                    identical to the existing load-step study).

    short_circuit   Temporary three-phase-to-ground bus fault, modelled as a
                    large shunt admittance added to the reduced Y-bus at a
                    chosen bus, applied for --sc-duration seconds and then
                    cleared. Tests fault ride-through / transient recovery.
                    Realistic offshore cause: subsea cable or switchgear fault.

    line_outage     Permanent trip of a selected line/cable via the TOPS line
                    event. Removes a feeder, so the load beyond it is rejected
                    (over-frequency) or a supply path is lost. Realistic cause:
                    protection trip of an inter-array cable or a load feeder.

    gen_trip        Loss of one gas turbine, modelled as a prime-mover
                    flame-out: the tripped unit's governor output limits are
                    driven to zero so its mechanical power collapses over the
                    governor time constant. The machine stays electrically
                    connected (still provides inertia and synchronising
                    torque), while the two surviving gas turbines' governors
                    respond normally. This is the classic generation-deficit
                    event and is the strongest test of WT frequency support.

All disturbances occur at --disturbance-time. The load step timing is
controlled separately by --event-time / --event-duration, so the two can be
combined or the load step disabled with --load-step-mw 0.

Result CSV files (results/csv_files):
    WT1_LEOGO_<disturbance>_noDroop.csv
    WT1_LEOGO_<disturbance>_droop.csv
    WT1_LEOGO_<disturbance>_metrics.csv
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

from casestudies.dyn_sim.test_WT_LEOGO_sim import build_model
from casestudies.dyn_sim.test_WT_LEOGO_droop_comparison_diagnostics import (
    configure_wt_case,
    coi_speed_pu,
    generator_speed_field,
    scalar,
)


# ---------------------------------------------------------------------
# One scenario (one droop setting) with an optional LEOGO disturbance
# ---------------------------------------------------------------------

def run_case(
    *,
    case_name: str,
    droop_enabled: bool,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, float | str | bool]]:
    """Run one droop setting for the requested disturbance scenario."""
    model = build_model()
    f_nom = float(model["f"])
    s_base_mva = float(model["base_mva"])

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)

    wt_model = ps.windturbine["WindTurbine"]
    uic_model = ps.vsc["UIC_sig"]
    gen_model = ps.gen["GEN"]

    configure_wt_case(
        wt_model,
        f_nom=f_nom,
        droop_enabled=droop_enabled,
        droop_gain_pu_per_hz=args.droop_gain_pu_per_hz,
        headroom_pu=args.headroom_pu,
    )

    # Power-flow initialization using the same de-loading / droop command
    # logic as the dynamic model at nominal frequency.
    wind_speed_initial = scalar(wt_model.wind_speed_init())
    p_ref_initial_uic_pu = scalar(
        wt_model.P_ref_from_wind(wind_speed_initial, uic_model.par["S_n"])
    )
    uic_model.par["p_ref"][:] = p_ref_initial_uic_pu
    uic_model.par["q_ref"][:] = 0.0

    ps.power_flow()
    ps.init_dyn_sim()
    x0 = ps.x0.copy()

    # COI frequency measurement.
    speed_field = generator_speed_field(gen_model, x0)
    speed_coi_0 = coi_speed_pu(gen_model, x0, speed_field)

    # Bus indices in the reduced system.
    main_bus_idx = gen_model.bus_idx_red["terminal"][0]        # Main Bus A
    wt_bus_idx = uic_model.bus_idx_red["terminal"][0]          # Busbar WTG1 LV
    sc_bus_idx = wt_bus_idx if args.sc_location == "wt_terminal" else main_bus_idx

    # Temporary shunt-load event admittance (constant-Z step on Main Bus A).
    y_load_step = (
        args.load_step_mw / s_base_mva
        - 1j * args.load_step_mvar / s_base_mva
    )

    # Generator names / order (for the gas-turbine trip).
    gen_names = [str(n) for n in np.asarray(gen_model.par["name"]).ravel()]

    def system_frequency_hz(x: np.ndarray) -> float:
        speed_coi = coi_speed_pu(gen_model, x, speed_field)
        return f_nom + f_nom * (speed_coi - speed_coi_0)

    # -----------------------------------------------------------------
    # Continuous inputs (evaluated at every RK4 sub-step)
    #   - load step (assignment-based, safe to set every call)
    #   - self-clearing short circuit (assignment-based)
    #   - measured COI frequency fed to the WT droop controller
    # -----------------------------------------------------------------
    def smoothstep(u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5

    def load_event_scale(t: float) -> float:
        t_on = args.event_time
        t_off = args.event_time + args.event_duration
        if t < t_on:
            return 0.0
        if t < t_on + args.load_ramp_on_s:
            return smoothstep((t - t_on) / args.load_ramp_on_s)
        if t < t_off:
            return 1.0
        if t < t_off + args.load_ramp_off_s:
            return 1.0 - smoothstep((t - t_off) / args.load_ramp_off_s)
        return 0.0

    def short_circuit_active(t: float) -> bool:
        if args.disturbance != "short_circuit":
            return False
        return args.disturbance_time <= t < (
            args.disturbance_time + args.sc_duration
        )

    def apply_external_inputs(t: float, x: np.ndarray) -> None:
        wt_model._sim_time = t

        load_scale = load_event_scale(t)
        sc_admittance = args.sc_admittance_pu if short_circuit_active(t) else 0.0

        # Load step (and short circuit, if it happens to be on the same bus).
        main_entry = load_scale * y_load_step
        if sc_bus_idx == main_bus_idx:
            main_entry = main_entry + sc_admittance
        ps.y_bus_red_mod[(main_bus_idx, main_bus_idx)] = main_entry

        if sc_bus_idx != main_bus_idx:
            ps.y_bus_red_mod[(sc_bus_idx, sc_bus_idx)] = sc_admittance

        # Droop-controller measurement path.
        wt_model.set_grid_frequency_hz(system_frequency_hz(x))

    def f_ode(t: float, x: np.ndarray) -> np.ndarray:
        apply_external_inputs(t, x)
        v = ps.solve_algebraic(t, x)
        return ps.state_derivatives(t, x, v)

    # -----------------------------------------------------------------
    # Discrete one-shot events (applied once, between solver steps)
    # -----------------------------------------------------------------
    event_state = {"line_outage_done": False, "gen_trip_done": False}

    def apply_discrete_events(t: float) -> None:
        if t < args.disturbance_time:
            return

        if (
            args.disturbance == "line_outage"
            and not event_state["line_outage_done"]
        ):
            ps.lines["Line"].event(ps, args.lo_line, "disconnect")
            event_state["line_outage_done"] = True

        if (
            args.disturbance == "gen_trip"
            and not event_state["gen_trip_done"]
        ):
            gov = ps.gov["TGOV1"]
            gen_idx = gen_names.index(args.gen_trip_name)
            # Prime-mover flame-out: drive this unit's governor output
            # limits to zero so its mechanical power collapses. The other
            # units keep their governor limits and respond normally.
            gov.time_constant_lim.par["V_min"][gen_idx] = 0.0
            gov.time_constant_lim.par["V_max"][gen_idx] = 0.0
            event_state["gen_trip_done"] = True

    def disturbance_active(t: float) -> bool:
        if args.disturbance == "none":
            return False
        if args.disturbance == "short_circuit":
            return short_circuit_active(t)
        # line_outage / gen_trip are permanent once triggered.
        return t >= args.disturbance_time

    # -----------------------------------------------------------------
    # Result logging
    # -----------------------------------------------------------------
    rows: list[dict[str, float | bool | str]] = []

    def store_row(t: float, x: np.ndarray) -> None:
        apply_external_inputs(t, x)
        v = ps.solve_algebraic(t, x)

        sys_s_n = float(wt_model.sys_par["s_n"])
        wt_s_n = float(wt_model.par["S_n"][0])
        uic_s_n = float(uic_model.par["S_n"][0])

        wt_ref = wt_model.P_ref_components(x, v)
        wt_states = wt_model.local_view(x)

        # Drivetrain torsional response (the electro-mechanical signal): the
        # shaft twist, speed difference and shaft torque through which a LEOGO
        # grid event reaches the turbine drivetrain. K and D are already in WT
        # pu; convert the shaft torque to Nm with T_base = S_n / omega_m_rated.
        k_shaft_pu = float(np.asarray(wt_model.par["K"]).ravel()[0])
        d_shaft_pu = float(np.asarray(wt_model.par["D"]).ravel()[0])
        theta_s = scalar(wt_states["theta_m"]) - scalar(wt_states["theta_e"])
        omega_s = scalar(wt_states["omega_m"]) - scalar(wt_states["omega_e"])
        T_shaft_pu = k_shaft_pu * theta_s + d_shaft_pu * omega_s
        omega_m_rated_rad_s = float(
            np.asarray(wt_model.par["omega_m_rated"]).ravel()[0]
        )
        T_base_Nm = wt_s_n * 1e6 / omega_m_rated_rad_s
        T_shaft_Nm = T_shaft_pu * T_base_Nm

        v_terminal = uic_model.v_t(x, v)[0]
        s_uic = uic_model.s_e(x, v)[0]
        i_uic = uic_model.i_a(x, v)[0]

        # Per-generator electrical output and speed.
        p_gen_local = np.asarray(gen_model.p_e(x, v), dtype=float)
        q_gen_local = np.asarray(gen_model.q_e(x, v), dtype=float)
        gen_s_n = np.asarray(gen_model.par["S_n"], dtype=float)
        gen_states = gen_model.local_view(x)
        gen_speed = np.asarray(gen_states[speed_field], dtype=float)

        p_sync_total = float(np.sum(p_gen_local * gen_s_n / sys_s_n))
        q_sync_total = float(np.sum(q_gen_local * gen_s_n / sys_s_n))

        row: dict[str, float | bool | str] = {
            "case": case_name,
            "droop_enabled": bool(droop_enabled),
            "droop_gain_pu_per_hz": float(
                args.droop_gain_pu_per_hz if droop_enabled else 0.0
            ),
            "headroom_uic_pu": float(args.headroom_pu),
            "disturbance": args.disturbance,

            "t": float(t),
            "disturbance_active": bool(disturbance_active(t)),
            "load_step_active": bool(load_event_scale(t) > 0.0),
            "load_step_mw": float(args.load_step_mw * load_event_scale(t)),

            "f_grid_hz": float(system_frequency_hz(x)),
            "frequency_deviation_hz": float(system_frequency_hz(x) - f_nom),

            # Wind turbine mechanical / control state.
            "omega_m_pu": scalar(wt_states["omega_m"]),
            "omega_e_pu": scalar(wt_states["omega_e"]),
            # Drivetrain torsion (electro-mechanical response to the grid event).
            "theta_s_rad": float(theta_s),
            "omega_s_pu": float(omega_s),
            "T_shaft_pu": float(T_shaft_pu),
            "T_shaft_Nm": float(T_shaft_Nm),
            "pitch_deg": float(np.degrees(scalar(wt_states["pitch_angle"]))),
            "wind_speed_mps": scalar(wt_model.wind_speed(x, v)),

            # Wind turbine power (system base unless stated).
            "P_aero_sys_pu": float(
                scalar(wt_model.P_aero(x, v)) * wt_s_n / sys_s_n
            ),
            "P_base_uic_pu": float(wt_ref["p_base_uic_pu"]),
            "P_available_uic_pu": float(wt_ref["p_available_uic_pu"]),
            "P_droop_delta_uic_pu": float(wt_ref["p_droop_delta_uic_pu"]),
            "P_ref_uic_pu": float(wt_ref["p_ref_uic_pu"]),
            "P_ref_sys_pu": float(
                wt_ref["p_ref_uic_pu"] * uic_s_n / sys_s_n
            ),
            "P_uic_bus_sys_pu": float(s_uic.real * uic_s_n / sys_s_n),
            "Q_uic_bus_sys_pu": float(s_uic.imag * uic_s_n / sys_s_n),
            "I_uic_pu": float(abs(i_uic)),

            # Voltages.
            "V_WTG1_LV_pu": float(abs(v_terminal)),
            "V_main_bus_pu": float(abs(v[main_bus_idx])),

            # Synchronous generators (gas turbines).
            "P_sync_total_sys_pu": p_sync_total,
            "Q_sync_total_sys_pu": q_sync_total,
        }

        for i, name in enumerate(gen_names):
            tag = f"gen{i + 1}"
            row[f"{tag}_name"] = name
            row[f"{tag}_speed_pu"] = float(gen_speed[i])
            row[f"{tag}_freq_hz"] = float(f_nom * (1.0 + gen_speed[i]))
            row[f"{tag}_P_mw"] = float(p_gen_local[i] * gen_s_n[i])
            row[f"{tag}_Q_mvar"] = float(q_gen_local[i] * gen_s_n[i])

        rows.append(row)

    # -----------------------------------------------------------------
    # Time integration
    # -----------------------------------------------------------------
    solver = dps_sol.SimpleRK4(f_ode, 0.0, x0, args.t_end, max_step=args.dt)

    store_row(0.0, x0)
    while solver.t < args.t_end:
        solver.step()
        apply_discrete_events(solver.t)
        store_row(solver.t, solver.x)

    result = pd.DataFrame(rows)

    metadata: dict[str, float | str | bool] = {
        "case": case_name,
        "disturbance": args.disturbance,
        "droop_enabled": droop_enabled,
        "droop_gain_pu_per_hz": (
            float(args.droop_gain_pu_per_hz) if droop_enabled else 0.0
        ),
        "headroom_uic_pu": float(args.headroom_pu),
        "load_step_mw": float(args.load_step_mw),
        "initial_wind_speed_mps": float(wind_speed_initial),
        "frequency_nadir_hz": float(result["f_grid_hz"].min()),
        "frequency_peak_hz": float(result["f_grid_hz"].max()),
        "maximum_droop_power_uic_pu": float(
            result["P_droop_delta_uic_pu"].max()
        ),
        "min_V_main_bus_pu": float(result["V_main_bus_pu"].min()),
        "min_V_WTG1_LV_pu": float(result["V_WTG1_LV_pu"].min()),
    }
    return result, metadata


# ---------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run LEOGO WT droop comparison with a selectable platform "
            "disturbance (short circuit, line outage or gas-turbine trip) "
            "on top of the configurable load step."
        )
    )

    # Simulation timing.
    parser.add_argument("--t-end", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.005)

    # Load step (same mechanism as the existing study).
    parser.add_argument("--event-time", type=float, default=5.0)
    parser.add_argument("--event-duration", type=float, default=25.0)
    parser.add_argument("--load-ramp-on-s", type=float, default=0.0)
    parser.add_argument("--load-ramp-off-s", type=float, default=0.0)
    parser.add_argument("--load-step-mw", type=float, default=5.0)
    parser.add_argument("--load-step-mvar", type=float, default=0.0)

    # Disturbance selection.
    parser.add_argument(
        "--disturbance",
        choices=["none", "short_circuit", "line_outage", "gen_trip"],
        default="gen_trip",
        help="LEOGO-side disturbance to apply in addition to the load step.",
    )
    parser.add_argument(
        "--disturbance-time",
        type=float,
        default=5.0,
        help="Time at which the selected disturbance occurs.",
    )

    # Short-circuit options.
    parser.add_argument(
        "--sc-location",
        choices=["main_bus", "wt_terminal"],
        default="main_bus",
        help="Bus for the temporary short circuit.",
    )
    parser.add_argument(
        "--sc-duration",
        type=float,
        default=0.15,
        help="Short-circuit duration before clearing, in seconds.",
    )
    parser.add_argument(
        "--sc-admittance-pu",
        type=float,
        default=1.0e6,
        help="Fault shunt admittance in system pu (large => solid fault).",
    )

    # Line-outage options.
    parser.add_argument(
        "--lo-line",
        type=str,
        default="Line GEX_01",
        help="Name of the line/cable to trip permanently.",
    )

    # Gas-turbine trip options.
    parser.add_argument(
        "--gen-trip-name",
        type=str,
        default="Synchronous Generator 3",
        help="Name of the gas turbine to flame-out (governor to zero).",
    )

    # WT droop / reserve settings (shared by both cases).
    parser.add_argument("--headroom-pu", type=float, default=0.05)
    parser.add_argument("--droop-gain-pu-per-hz", type=float, default=0.50)

    args = parser.parse_args()

    if args.t_end <= 0.0 or args.dt <= 0.0:
        raise SystemExit("--t-end and --dt must be positive.")

    output_dir = (
        PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "csv_files"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = args.disturbance
    cases = [
        (
            "no_droop",
            False,
            output_dir / f"WT1_LEOGO_{tag}_noDroop.csv",
        ),
        (
            "droop",
            True,
            output_dir / f"WT1_LEOGO_{tag}_droop.csv",
        ),
    ]

    metrics: list[dict[str, float | str | bool]] = []
    all_start = time.perf_counter()

    for case_name, droop_enabled, output_file in cases:
        print(
            f"\nRunning case '{case_name}' "
            f"(droop={'on' if droop_enabled else 'off'}, "
            f"disturbance={args.disturbance})..."
        )
        case_start = time.perf_counter()
        result, metadata = run_case(
            case_name=case_name,
            droop_enabled=droop_enabled,
            args=args,
        )
        result.to_csv(output_file, index=False)
        metrics.append(metadata)
        print(
            f"  Saved: {output_file.name} "
            f"(nadir {metadata['frequency_nadir_hz']:.4f} Hz, "
            f"peak {metadata['frequency_peak_hz']:.4f} Hz, "
            f"{time.perf_counter() - case_start:.1f} s)"
        )

    metrics_file = output_dir / f"WT1_LEOGO_{tag}_metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_file, index=False)

    nadir_no = metrics[0]["frequency_nadir_hz"]
    nadir_dr = metrics[1]["frequency_nadir_hz"]
    print(f"\nSaved: {metrics_file}")
    print(
        f"Frequency nadir: no-droop {nadir_no:.4f} Hz, "
        f"droop {nadir_dr:.4f} Hz "
        f"(improvement {nadir_dr - nadir_no:+.4f} Hz)"
    )
    print(f"Total runtime: {time.perf_counter() - all_start:.2f} s")


if __name__ == "__main__":
    main()
