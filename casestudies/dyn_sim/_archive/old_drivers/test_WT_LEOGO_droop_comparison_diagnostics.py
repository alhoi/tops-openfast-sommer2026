"""
Run matched LEOGO wind-turbine frequency-response cases with and without
internal WT droop control.

Both cases use the same:
  - extended electromechanical diagnostics in the output CSV files
  - wind-turbine / UIC / LEOGO model
  - load step
  - MPT-based de-loading reserve (headroom)
  - initial operating point

The only intended difference is droop_enable / K_droop_pu_per_hz.

Important:
  * With headroom > 0, the no-droop case is de-loaded but does not respond
    to frequency. The droop case starts from the same output and can use
    that reserve during under-frequency.
  * The script overwrites the canonical result files used by the existing
    comparison plot script:
        WT1_LEOGO_frequency_results.csv
        WT1_LEOGO_frequency_droop_results.csv
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


def scalar(value) -> float:
    """Return the first scalar value from a NumPy-like object."""
    return float(np.asarray(value).reshape(-1)[0])


def generator_speed_field(gen_model, x: np.ndarray) -> str:
    """Find the synchronous-generator speed-state field."""
    states = gen_model.local_view(x)

    for field in ("speed", "omega", "omega_m"):
        if field in states.dtype.names:
            return field

    raise KeyError(
        "Could not find a generator speed state. "
        f"Available fields: {states.dtype.names}"
    )


def coi_speed_pu(gen_model, x: np.ndarray, speed_field: str) -> float:
    """Inertia-weighted centre-of-inertia speed in pu."""
    states = gen_model.local_view(x)
    speed = np.asarray(states[speed_field], dtype=float)
    h = np.asarray(gen_model.par["H"], dtype=float)
    s_n = np.asarray(gen_model.par["S_n"], dtype=float)

    return float(np.average(speed, weights=h * s_n))


def configure_wt_case(
    wt_model,
    *,
    f_nom: float,
    droop_enabled: bool,
    droop_gain_pu_per_hz: float,
    headroom_pu: float,
) -> None:
    """
    Configure a matched de-loaded MPT case.

    The normal reference is:
        P_base = P_available - headroom

    The droop case then adds:
        Delta P = K_droop * (f_nom - f_grid)
    before clipping against available WT power.
    """
    wt_model.par["f_nom_hz"][:] = f_nom
    wt_model.par["headroom_pu"][:] = headroom_pu

    wt_model.par["droop_enable"][:] = int(droop_enabled)
    wt_model.par["K_droop_pu_per_hz"][:] = (
        droop_gain_pu_per_hz if droop_enabled else 0.0
    )
    wt_model.set_grid_frequency_hz(f_nom)


def run_case(
    *,
    case_name: str,
    droop_enabled: bool,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, float | str | bool]]:
    """Run one frequency-response case and return its complete result table."""
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

    # Power-flow initialization.
    # P_ref_from_wind uses exactly the same de-loading / droop command
    # logic as the dynamic model at nominal frequency.
    wind_speed_initial = scalar(wt_model.wind_speed_init())
    p_ref_initial_uic_pu = scalar(
        wt_model.P_ref_from_wind(
            wind_speed_initial,
            uic_model.par["S_n"],
        )
    )

    uic_model.par["p_ref"][:] = p_ref_initial_uic_pu
    uic_model.par["q_ref"][:] = 0.0

    ps.power_flow()
    ps.init_dyn_sim()
    x0 = ps.x0.copy()

    
    # COI frequency measurement and the temporary shunt-load event.
    speed_field = generator_speed_field(gen_model, x0)
    speed_coi_0 = coi_speed_pu(gen_model, x0, speed_field)
    main_bus_idx = gen_model.bus_idx_red["terminal"][0]

    y_load_step = (
        args.load_step_mw / s_base_mva
        - 1j * args.load_step_mvar / s_base_mva
    )

    def system_frequency_hz(x: np.ndarray) -> float:
        speed_coi = coi_speed_pu(gen_model, x, speed_field)
        return f_nom + f_nom * (speed_coi - speed_coi_0)

    """
    def load_event_is_active(t: float) -> bool:
        return args.event_time <= t < (args.event_time + args.event_duration)"""
    
    def smoothstep(u: float) -> float:
        """Smooth transition from 0 to 1."""
        u = float(np.clip(u, 0.0, 1.0))
        return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


    def load_event_scale(t: float) -> float:
        """Return the fraction of the extra load currently applied."""
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

    def load_event_sine_scale(t: float) -> float:
        """
        Sinusoidal load profile.

        The extra load is now a sinus function of time:

            scale(t) = envelope(t)
                       * [mean + amplitude * sin(2*pi*f*(t - t_on) + phase)]

        where envelope(t) is the existing on/off (optionally ramped) window
        from load_event_scale(). This keeps the event start/stop and the
        smooth ramps intact while making the applied load oscillate.

        The applied load in MW is args.load_step_mw * scale(t), so with the
        defaults (mean = 0, amplitude = 1) the load is a pure sinus that
        swings between -load_step_mw and +load_step_mw. Use a non-zero mean
        to oscillate around a positive load level.
        """
        envelope = load_event_scale(t)
        if envelope <= 0.0:
            return 0.0

        phase_rad = (
            2.0 * np.pi * args.load_sine_freq_hz * (t - args.event_time)
            + np.radians(args.load_sine_phase_deg)
        )
        sine = args.load_sine_mean + args.load_sine_amplitude * np.sin(
            phase_rad
        )
        return float(envelope * sine)

    def apply_external_inputs(t: float, x: np.ndarray) -> None:
        """Update load event and measured grid frequency before algebraic solve."""
        wt_model._sim_time = t

        """ps.y_bus_red_mod[(main_bus_idx, main_bus_idx)] = (
            y_load_step if load_event_is_active(t) else 0.0
        )"""
        load_scale = load_event_sine_scale(t)

        ps.y_bus_red_mod[(main_bus_idx, main_bus_idx)] = (
            load_scale * y_load_step
        )

        # This is the droop-controller measurement path:
        # COI frequency -> WT model -> UIC active-power reference.
        wt_model.set_grid_frequency_hz(system_frequency_hz(x))

    def f_ode(t: float, x: np.ndarray) -> np.ndarray:
        apply_external_inputs(t, x)
        v = ps.solve_algebraic(t, x)
        return ps.state_derivatives(t, x, v)

    solver = dps_sol.SimpleRK4(
        f_ode,
        0.0,
        x0,
        args.t_end,
        max_step=args.dt,
    )

    rows: list[dict[str, float | bool | str]] = []

    def store_row(t: float, x: np.ndarray) -> None:
        apply_external_inputs(t, x)
        v = ps.solve_algebraic(t, x)

        wt_ref = wt_model.P_ref_components(x, v)
        p_ref_cmd = float(wt_ref["p_ref_uic_pu"])
        p_base = float(wt_ref["p_base_uic_pu"])
        p_available = float(wt_ref["p_available_uic_pu"])
        delta_p = float(wt_ref["p_droop_delta_uic_pu"])

        sys_s_n = float(wt_model.sys_par["s_n"])
        wt_s_n = float(wt_model.par["S_n"][0])
        uic_s_n = float(uic_model.par["S_n"][0])

        wt_states = wt_model.local_view(x)
        uic_states = uic_model.local_view(x)

        v_terminal = uic_model.v_t(x, v)[0]
        s_uic = uic_model.s_e(x, v)[0]
        i_uic = uic_model.i_a(x, v)[0]

        vi = uic_states["vi_x"][0] + 1j * uic_states["vi_y"][0]

        s_ref_internal = (
            uic_model.p_ref(x, v)[0]
            + 1j * uic_model.q_ref(x, v)[0]
        )
        xf = float(uic_model.par["xf"][0])
        s_bus_ref = s_ref_internal - 1j * xf * abs(i_uic) ** 2

        omega_rated_rad_s = float(
            np.asarray(wt_model.par["omega_m_rated"]).ravel()[0]
        )
        omega_e_pu = scalar(wt_states["omega_e"])
        t_mpt_wt_pu = float(
            wt_model._mpt_torque_mech_pu(
                omega_e_pu * omega_rated_rad_s
            )
        )

        p_gen_local = np.asarray(gen_model.p_e(x, v), dtype=float)
        q_gen_local = np.asarray(gen_model.q_e(x, v), dtype=float)
        gen_s_n = np.asarray(gen_model.par["S_n"], dtype=float)

        p_sync_total = float(np.sum(p_gen_local * gen_s_n / sys_s_n))
        q_sync_total = float(np.sum(q_gen_local * gen_s_n / sys_s_n))

        
        # Diagnostics for electromechanical interaction analysis.
        # These calculations reproduce the quantities used by the WT
        # drivetrain and pitch-controller equations without changing
        # the simulated states or controller outputs.
        
        omega_m_pu = scalar(wt_states["omega_m"])
        omega_e_pu = scalar(wt_states["omega_e"])

        # The present model uses the filtered generator speed for both
        # electromagnetic torque and pitch control whenever speed_lpf_type
        # is non-zero. Log the exact speed signal seen by the controller.
        speed_lpf_type = int(
            np.asarray(wt_model._speed_lpf_type).ravel()[0]
        )
        omega_e_filt_state_pu = scalar(wt_states["omega_e_filt"])
        omega_e_control_pu = (
            omega_e_pu if speed_lpf_type == 0 else omega_e_filt_state_pu
        )

        # Powers and torques on the WT local base. These follow the same
        # expressions as WindTurbine.state_derivatives().
        p_aero_wt_pu = scalar(wt_model.P_aero(x, v))
        p_e_wt_pu = scalar(wt_model.P_e(x, v)) * uic_s_n / wt_s_n
        eta = float(np.asarray(wt_model.par["efficiency"]).ravel()[0])

        t_a_wt_pu = p_aero_wt_pu / max(omega_m_pu, 1e-6)
        t_e_wt_pu = p_e_wt_pu / max(omega_e_control_pu * eta, 1e-6)

        theta_s = (
            scalar(wt_states["theta_m"]) - scalar(wt_states["theta_e"])
        )
        omega_s_pu = omega_m_pu - omega_e_pu
        t_shaft_wt_pu = float(
            np.asarray(wt_model.par["K"]).ravel()[0] * theta_s
            + np.asarray(wt_model.par["D"]).ravel()[0] * omega_s_pu
        )

        torque_base_nm = wt_s_n * 1e6 / omega_rated_rad_s

        # Pitch-controller input, reference and actuator-rate command.
        # This duplicates the active Region-3 pitch law used by the current
        # model, while preserving the Region-2 behaviour for completeness.
        pitch_rad = scalar(wt_states["pitch_angle"])
        omega_ref_pu = 1.0
        omega_speed_error_pu = omega_e_control_pu - omega_ref_pu

        min_pitch_rad = float(
            np.asarray(wt_model.par["min_pitch"]).ravel()[0]
        )
        max_pitch_rad = float(
            np.asarray(wt_model.par["max_pitch"]).ravel()[0]
        )
        max_pitch_rate_rad_s = float(
            np.asarray(wt_model.par["max_pitch_rate"]).ravel()[0]
        )
        t_pitch_s = float(np.asarray(wt_model.par["T_pitch"]).ravel()[0])
        wind_rated_mps = float(
            np.asarray(wt_model.par["wind_rated"]).ravel()[0]
        )
        wind_now_mps = scalar(wt_model.wind_speed(x, v))

        if wind_now_mps < wind_rated_mps:
            pitch_reference_rad = min_pitch_rad
        else:
            pitch_reference_unclamped_rad = (
                float(np.asarray(wt_model.par["Ki_pitch"]).ravel()[0])
                * scalar(wt_states["pitch_PI_integral_state"])
                + float(np.asarray(wt_model.par["Kp_pitch"]).ravel()[0])
                * omega_speed_error_pu
            )
            pitch_reference_rad = float(
                np.clip(
                    pitch_reference_unclamped_rad,
                    min_pitch_rad,
                    max_pitch_rad,
                )
            )

        pitch_rate_rad_s = float(
            np.clip(
                (pitch_reference_rad - pitch_rad) / t_pitch_s,
                -max_pitch_rate_rad_s,
                max_pitch_rate_rad_s,
            )
        )
        # NEW LINE
        load_scale = load_event_sine_scale(t)
        load_envelope = load_event_scale(t)

        rows.append(
            {
                # Case metadata repeated in each row so every CSV is self-describing.
                "case": case_name,
                "droop_enabled": bool(droop_enabled),
                "droop_gain_pu_per_hz": float(
                    args.droop_gain_pu_per_hz if droop_enabled else 0.0
                ),
                "headroom_uic_pu": float(args.headroom_pu),

                "t": float(t),
                """"load_step_active": bool(load_event_is_active(t)),
                "load_step_mw": float(
                    args.load_step_mw if load_event_is_active(t) else 0.0
                ),"""
                "load_step_active": bool(load_envelope > 0.0),
                "load_step_mw": float(args.load_step_mw * load_scale),

                "f_grid_hz": float(system_frequency_hz(x)),
                "frequency_deviation_hz": float(
                    system_frequency_hz(x) - f_nom
                ),

                "omega_base_rpm": float(
                    omega_rated_rad_s * 60.0 / (2.0 * np.pi)
                ),
                "wind_speed_mps": wind_now_mps,

                # Drivetrain states and torque balance (WT local base).
                "omega_m_pu": omega_m_pu,
                "omega_e_pu": omega_e_pu,
                "omega_e_filt_pu": omega_e_filt_state_pu,
                "omega_e_control_pu": omega_e_control_pu,
                "omega_ref_pu": omega_ref_pu,
                "omega_speed_error_pu": omega_speed_error_pu,
                "omega_s_pu": omega_s_pu,
                "theta_s_state": theta_s,
                "T_a_wt_pu": t_a_wt_pu,
                "T_e_wt_pu": t_e_wt_pu,
                "T_shaft_wt_pu": t_shaft_wt_pu,
                "T_a_Nm": t_a_wt_pu * torque_base_nm,
                "T_e_Nm": t_e_wt_pu * torque_base_nm,
                "T_shaft_Nm": t_shaft_wt_pu * torque_base_nm,
                "T_mpt_wt_pu": t_mpt_wt_pu,

                # Pitch-controller states and output.
                "pitch_rad": pitch_rad,
                "pitch_deg": float(np.degrees(pitch_rad)),
                "pitch_reference_rad": pitch_reference_rad,
                "pitch_reference_deg": float(
                    np.degrees(pitch_reference_rad)
                ),
                "pitch_rate_rad_s": pitch_rate_rad_s,
                "pitch_rate_deg_s": float(np.degrees(pitch_rate_rad_s)),
                "pitch_PI_integral_state": scalar(
                    wt_states["pitch_PI_integral_state"]
                ),

                # Aerodynamic, electrical and UIC power.
                "P_aero_wt_pu": p_aero_wt_pu,
                "P_aero_sys_pu": float(
                    p_aero_wt_pu * wt_s_n / sys_s_n
                ),
                "P_e_wt_pu": p_e_wt_pu,
                "P_e_sys_pu": float(
                    p_e_wt_pu * wt_s_n / sys_s_n
                ),
                "P_base_uic_pu": p_base,
                "P_available_uic_pu": p_available,
                "P_droop_delta_uic_pu": delta_p,
                "P_ref_uic_pu": p_ref_cmd,
                "P_ref_sys_pu": float(p_ref_cmd * uic_s_n / sys_s_n),
                "P_ref_instant_sys_pu": float(
                    p_ref_cmd * uic_s_n / sys_s_n
                ),
                "P_mpt_available_sys_pu": float(
                    p_available * uic_s_n / sys_s_n
                ),

                "V_WTG1_LV_pu": float(abs(v_terminal)),
                "v_bus_pu": float(abs(v_terminal)),
                "vi_mag_pu": float(abs(vi)),
                "P_uic_bus_sys_pu": float(
                    s_uic.real * uic_s_n / sys_s_n
                ),
                "Q_uic_bus_sys_pu": float(
                    s_uic.imag * uic_s_n / sys_s_n
                ),
                "P_uic_bus_actual_sys_pu": float(
                    s_uic.real * uic_s_n / sys_s_n
                ),
                "Q_uic_bus_actual_sys_pu": float(
                    s_uic.imag * uic_s_n / sys_s_n
                ),
                "P_uic_bus_ref_sys_pu": float(
                    s_bus_ref.real * uic_s_n / sys_s_n
                ),
                "Q_uic_bus_ref_sys_pu": float(
                    s_bus_ref.imag * uic_s_n / sys_s_n
                ),
                "I_uic_pu": float(abs(i_uic)),
                "i_a_mag_pu_uic": float(abs(i_uic)),
                "i_a_angle_deg": float(np.degrees(np.angle(i_uic))),

                "P_sync_generators_total_sys_pu": p_sync_total,
                "Q_sync_generators_total_sys_pu": q_sync_total,
            }
        )

    store_row(0.0, x0)

    while solver.t < args.t_end:
        solver.step()
        store_row(solver.t, solver.x)

    result = pd.DataFrame(rows)

    metadata: dict[str, float | str | bool] = {
        "case": case_name,
        "droop_enabled": droop_enabled,
        "droop_gain_pu_per_hz": (
            float(args.droop_gain_pu_per_hz) if droop_enabled else 0.0
        ),
        "headroom_uic_pu": float(args.headroom_pu),
        "initial_wind_speed_mps": float(wind_speed_initial),
        "initial_p_ref_uic_pu": float(p_ref_initial_uic_pu),
        "frequency_state_field": speed_field,
        "frequency_nadir_hz": float(result["f_grid_hz"].min()),
        "maximum_droop_power_uic_pu": float(
            result["P_droop_delta_uic_pu"].max()
        ),
    }
    return result, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched de-loaded MPT cases without and with WT droop control."
        )
    )
    parser.add_argument("--t-end", type=float, default=50.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--event-time", type=float, default=5.0)
    parser.add_argument("--event-duration", type=float, default=50)
    # 2 NEW PARSERS ADDED
    parser.add_argument(
        "--load-ramp-on-s",
        type=float,
        default=0.0,
        help="Duration of the smooth load increase. Zero gives an ideal step.",
    )

    parser.add_argument(
        "--load-ramp-off-s",
        type=float,
        default=0.0,
        help="Duration of the smooth load removal. Zero gives an ideal step.",
    )
    parser.add_argument("--load-step-mw", type=float, default=2.5)
    parser.add_argument("--load-step-mvar", type=float, default=0.0)
    # Sinusoidal load profile. The applied load in MW is
    # load_step_mw * envelope(t) * [mean + amplitude * sin(2*pi*f*t + phase)].
    parser.add_argument(
        "--load-sine-freq-hz",
        type=float,
        default=1.2,
        help="Frequency of the sinusoidal load, in Hz.",
    )
    parser.add_argument(
        "--load-sine-amplitude",
        type=float,
        default=0.5,
        help=(
            "Amplitude of the sinusoid as a fraction of --load-step-mw. "
            "1.0 gives a full-amplitude sinus."
        ),
    )
    parser.add_argument(
        "--load-sine-mean",
        type=float,
        default=0.5,
        help=(
            "Mean/offset of the sinusoid as a fraction of --load-step-mw. "
            "0.0 oscillates around zero; 1.0 oscillates around load_step_mw."
        ),
    )
    parser.add_argument(
        "--load-sine-phase-deg",
        type=float,
        default=0.0,
        help="Phase offset of the sinusoidal load, in degrees.",
    )
    parser.add_argument(
        "--headroom-pu",
        type=float,
        default=0.05,
        help=(
            "WT reserve on the 20 MVA UIC base. Both cases use the same "
            "value. 0.05 pu = 1 MW."
        ),
    )
    parser.add_argument(
        "--droop-gain-pu-per-hz",
        type=float,
        default=0.50,
        help="Droop gain on UIC base, in pu/Hz, used only in the droop case.",
    )
    args = parser.parse_args()

    if args.t_end <= 0.0:
        raise ValueError("--t-end must be positive.")
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive.")
    if args.event_duration < 0.0:
        raise ValueError("--event-duration must be non-negative.")
    if args.headroom_pu < 0.0:
        raise ValueError("--headroom-pu must be non-negative.")
    if args.droop_gain_pu_per_hz < 0.0:
        raise ValueError("--droop-gain-pu-per-hz must be non-negative.")

    if args.headroom_pu == 0.0:
        print(
            "WARNING: headroom is zero. The droop case has no upward "
            "active-power reserve at the operating point."
        )

    output_dir = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "csv_files"
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        (
            "no_droop_de_loaded",
            False,
            output_dir / "WT1_LEOGO_frequency_Sine_freq_NOdroop.csv",
        ),
        (
            "droop_de_loaded",
            True,
            output_dir / "WT1_LEOGO_frequency_Sine_freq_droop.csv",
        ),
    ]

    metrics: list[dict[str, float | str | bool]] = []
    all_start = time.perf_counter()

    for case_name, droop_enabled, output_file in cases:
        print(f"\nRunning {case_name} ...")
        case_start = time.perf_counter()

        result, metadata = run_case(
            case_name=case_name,
            droop_enabled=droop_enabled,
            args=args,
        )

        result.to_csv(output_file, index=False)
        metadata["runtime_s"] = float(time.perf_counter() - case_start)
        metrics.append(metadata)

        print(f"Saved: {output_file}")
        print(
            f"  Nadir: {metadata['frequency_nadir_hz']:.6f} Hz | "
            f"max ΔP droop: {metadata['maximum_droop_power_uic_pu']:.6f} pu"
        )

    metrics_file = output_dir / "WT1_LEOGO_frequency_comparison_metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_file, index=False)

    print(f"\nSaved: {metrics_file}")
    print(f"Total runtime: {time.perf_counter() - all_start:.2f} s")
    print(
        "\nBoth result CSVs are written with the existing canonical filenames, "
        "so compare_droopvsNOdroop_plot.py can be run directly."
    )


if __name__ == "__main__":
    main()
