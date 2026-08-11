r"""
Step 3 -- Electrically driven torsional resonance in the simplified LEOGO
wind-turbine model.

The drivetrain torsional mode identified by the modal analysis
(participation_WT_LEOGO_torsional.py) sits at f_t = 3.49 Hz with a light
damping ratio zeta_t = 4.37 %, and is generator-dominated. Because the
electromagnetic torque enters the generator-mass equation of motion directly
through T_e = P_e / (omega_e_filt * eta), an oscillating electrical power at
f_t should drive that torsional mode to resonance.

This script applies a sinusoidal shunt load (an electrical disturbance) at a
chosen bus, which makes the wind-turbine terminal power -- and hence T_e --
oscillate. It logs the drivetrain response (T_shaft, omega_s, theta_s, ...),
writes a CSV, and prints a resonance diagnostic (envelope build-up + FFT peak).

Single run (on resonance):
  .\.venv\Scripts\python.exe casestudies\dyn_sim\test_WT_LEOGO_torsional_resonance_sim.py --forcing-freq-hz 3.49 --t-end 60

Frequency sweep (find the peak empirically):
  .\.venv\Scripts\python.exe casestudies\dyn_sim\test_WT_LEOGO_torsional_resonance_sim.py --sweep 1.0 2.5 3.0 3.25 3.49 3.75 4.0
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
    return float(np.asarray(value).reshape(-1)[0])


def print_drivetrain_constants(wt_model) -> float:
    """Print the drivetrain per-unit constants and return the analytical f_n."""
    H_m = scalar(wt_model.H_m)
    H_e = scalar(wt_model.H_e)
    K_pu = scalar(wt_model.par["K"])          # already pu (converted in __init__)
    D_pu = scalar(wt_model.par["D"])          # already pu
    w_base = scalar(wt_model.par["omega_m_rated"])
    s = 1.0 / (2.0 * H_m) + 1.0 / (2.0 * H_e)
    w_n = np.sqrt(K_pu * s)
    zeta = D_pu * s / (2.0 * w_n)
    f_n = w_n / (2.0 * np.pi)
    print("Drivetrain constants (from the running model):")
    print(f"  omega_m_rated (w_base) = {w_base:.6f}")
    print(f"  H_m = {H_m:.4f} s,  H_e = {H_e:.5f} s")
    print(f"  K_pu = {K_pu:.4f},  D_pu = {D_pu:.4f}")
    print(f"  analytical torsional: f_n = {f_n:.4f} Hz,  zeta = {100*zeta:.2f} %")
    print("  (compare with eigenvalue analysis: 3.4909 Hz, 4.37 %)")
    return f_n


def run_case(*, forcing_freq_hz: float, args: argparse.Namespace):
    """Run one forced-response case and return the result DataFrame."""
    model = build_model()
    f_nom = float(model["f"])
    s_base_mva = float(model["base_mva"])

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)
    wt_model = ps.windturbine["WindTurbine"]
    uic_model = ps.vsc["UIC_sig"]
    gen_model = ps.gen["GEN"]

    # No droop, no de-loading reserve: pure MPT operating point so the only
    # dynamic forcing is the electrical disturbance we inject.
    wt_model.par["droop_enable"][:] = 0
    wt_model.par["K_droop_pu_per_hz"][:] = 0.0
    wt_model.par["headroom_pu"][:] = 0.0
    wt_model.par["f_nom_hz"][:] = f_nom
    wt_model.set_grid_frequency_hz(f_nom)

    # Initialise the UIC power reference from the wind operating point.
    wind_speed_initial = scalar(wt_model.wind_speed_init())
    p_ref_initial = scalar(
        wt_model.P_ref_from_wind(wind_speed_initial, uic_model.par["S_n"])
    )
    uic_model.par["p_ref"][:] = p_ref_initial
    uic_model.par["q_ref"][:] = 0.0

    ps.power_flow()
    ps.init_dyn_sim()
    x0 = ps.x0.copy()

    # Bus to inject the disturbance at.
    wt_bus_idx = int(uic_model.bus_idx_red["terminal"][0])
    main_bus_idx = int(gen_model.bus_idx_red["terminal"][0])
    inject_bus_idx = wt_bus_idx if args.load_bus == "wt" else main_bus_idx

    # Shunt admittance for a unit (1 pu MW) load; scaled by the sinusoid.
    y_load_unit = args.load_amp_mw / s_base_mva

    omega_rated_rad_s = scalar(wt_model.par["omega_m_rated"])
    torque_base_nm = scalar(wt_model.par["S_n"]) * 1e6 / omega_rated_rad_s
    K_pu = scalar(wt_model.par["K"])
    D_pu = scalar(wt_model.par["D"])
    uic_s_n = scalar(uic_model.par["S_n"])
    wt_s_n = scalar(wt_model.par["S_n"])
    sys_s_n = float(wt_model.sys_par["s_n"])

    def load_scale(t: float) -> float:
        """Sinusoidal disturbance, switched on at args.onset."""
        if t < args.onset:
            return 0.0
        phase = 2.0 * np.pi * forcing_freq_hz * (t - args.onset)
        return args.load_mean + args.load_amp_frac * np.sin(phase)

    def apply_external_inputs(t: float, x: np.ndarray) -> None:
        wt_model._sim_time = t
        ps.y_bus_red_mod[(inject_bus_idx, inject_bus_idx)] = (
            load_scale(t) * y_load_unit
        )
        # Droop is disabled, so the measured grid frequency is held at nominal.
        wt_model.set_grid_frequency_hz(f_nom)

    def f_ode(t: float, x: np.ndarray) -> np.ndarray:
        apply_external_inputs(t, x)
        v = ps.solve_algebraic(t, x)
        return ps.state_derivatives(t, x, v)

    solver = dps_sol.SimpleRK4(f_ode, 0.0, x0, args.t_end, max_step=args.dt)

    rows: list[dict[str, float]] = []

    def store_row(t: float, x: np.ndarray) -> None:
        apply_external_inputs(t, x)
        v = ps.solve_algebraic(t, x)
        wt_states = wt_model.local_view(x)

        omega_m = scalar(wt_states["omega_m"])
        omega_e = scalar(wt_states["omega_e"])
        theta_s = scalar(wt_states["theta_m"]) - scalar(wt_states["theta_e"])
        omega_s = omega_m - omega_e
        t_shaft_pu = K_pu * theta_s + D_pu * omega_s

        eta = scalar(wt_model.par["efficiency"])
        omega_e_filt = scalar(wt_states["omega_e_filt"])
        p_e_wt_pu = scalar(wt_model.P_e(x, v)) * uic_s_n / wt_s_n
        t_e_pu = p_e_wt_pu / max(omega_e_filt * eta, 1e-6)

        v_terminal = uic_model.v_t(x, v)[0]
        s_uic = uic_model.s_e(x, v)[0]

        rows.append({
            "t": float(t),
            "forcing_freq_hz": forcing_freq_hz,
            "load_scale": load_scale(t),
            "omega_m_pu": omega_m,
            "omega_e_pu": omega_e,
            "omega_s_pu": omega_s,
            "theta_s_state": theta_s,
            "T_shaft_pu": t_shaft_pu,
            "T_shaft_Nm": t_shaft_pu * torque_base_nm,
            "T_e_pu": t_e_pu,
            "P_e_wt_pu": p_e_wt_pu,
            "P_uic_sys_pu": float(s_uic.real * uic_s_n / sys_s_n),
            "Q_uic_sys_pu": float(s_uic.imag * uic_s_n / sys_s_n),
            "V_wt_terminal_pu": float(abs(v_terminal)),
        })

    store_row(0.0, x0)
    while solver.t < args.t_end:
        solver.step()
        store_row(solver.t, solver.x)

    return pd.DataFrame(rows)


def analyse_resonance(df: pd.DataFrame, onset: float) -> dict[str, float]:
    """Envelope build-up and FFT peak of T_shaft after the forcing onset."""
    t = df["t"].to_numpy()
    y = df["T_shaft_Nm"].to_numpy()
    post = t >= onset
    tp = t[post]
    yp = y[post] - np.mean(y[post])

    # Envelope in five equal windows (peak-to-peak / 2).
    n = len(tp)
    seg = max(n // 5, 1)
    env = []
    for k in range(5):
        s = yp[k * seg:(k + 1) * seg] if k < 4 else yp[4 * seg:]
        env.append(0.5 * np.ptp(s) if len(s) else 0.0)

    # FFT peak.
    dt = np.median(np.diff(tp))
    win = np.hanning(len(yp))
    spec = np.abs(np.fft.rfft(yp * win))
    freqs = np.fft.rfftfreq(len(yp), dt)
    pk = int(np.argmax(spec[1:]) + 1)
    f_peak = float(freqs[pk])

    return {
        "env_start_Nm": float(env[0]),
        "env_end_Nm": float(env[-1]),
        "env_ratio": float(env[-1] / env[0]) if env[0] > 0 else float("inf"),
        "fft_peak_hz": f_peak,
        "T_shaft_pp_Nm": float(np.ptp(yp)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Electrically driven drivetrain torsional resonance test."
    )
    parser.add_argument("--forcing-freq-hz", type=float, default=3.49,
                        help="Electrical forcing frequency (Hz). Torsional mode = 3.49 Hz.")
    parser.add_argument("--sweep", type=float, nargs="+", default=None,
                        help="Run several forcing frequencies and tabulate the response.")
    parser.add_argument("--t-end", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--onset", type=float, default=10.0,
                        help="Time at which the electrical forcing switches on (s).")
    parser.add_argument("--load-amp-mw", type=float, default=2.0,
                        help="Peak amplitude of the shunt-load disturbance (MW at 1.0 scale).")
    parser.add_argument("--load-amp-frac", type=float, default=1.0,
                        help="Sinusoid amplitude as a fraction of --load-amp-mw.")
    parser.add_argument("--load-mean", type=float, default=0.0,
                        help="Sinusoid mean/offset as a fraction of --load-amp-mw.")
    parser.add_argument("--load-bus", choices=["wt", "main"], default="wt",
                        help="Inject at the WT terminal ('wt') or the main gen bus ('main').")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "csv_files"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print drivetrain constants once (also verifies f_n vs the eigenvalue).
    probe = build_model()
    ps_probe = dps.PowerSystemModel(model=probe, user_mdl_lib=ext_lib)
    ps_probe.power_flow()
    ps_probe.init_dyn_sim()
    print_drivetrain_constants(ps_probe.windturbine["WindTurbine"])
    print()

    freqs = args.sweep if args.sweep else [args.forcing_freq_hz]
    summary: list[dict[str, float]] = []
    t0 = time.perf_counter()

    for f in freqs:
        print(f"Running forcing at {f:.3f} Hz ...")
        df = run_case(forcing_freq_hz=f, args=args)
        metrics = analyse_resonance(df, args.onset)
        metrics = {"forcing_freq_hz": f, **metrics}
        summary.append(metrics)

        out = args.out or str(
            output_dir / f"WT1_LEOGO_torsional_forcing_{f:.2f}Hz.csv".replace(".", "p", 1)
        )
        df.to_csv(out, index=False)
        print(f"  saved {out}")
        print(f"  T_shaft envelope: {metrics['env_start_Nm']:.3e} -> "
              f"{metrics['env_end_Nm']:.3e} Nm  (x{metrics['env_ratio']:.2f}),  "
              f"FFT peak {metrics['fft_peak_hz']:.3f} Hz,  "
              f"p2p {metrics['T_shaft_pp_Nm']:.3e} Nm")

    if len(summary) > 1:
        sdf = pd.DataFrame(summary)
        sfile = output_dir / "WT1_LEOGO_torsional_sweep_summary.csv"
        sdf.to_csv(sfile, index=False)
        print("\nSweep summary (peak-to-peak T_shaft vs forcing frequency):")
        for r in summary:
            bar = "#" * int(40 * r["T_shaft_pp_Nm"] / max(x["T_shaft_pp_Nm"] for x in summary))
            print(f"  {r['forcing_freq_hz']:6.3f} Hz | {r['T_shaft_pp_Nm']:.3e} Nm | {bar}")
        print(f"saved {sfile}")

    print(f"\nTotal runtime: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
