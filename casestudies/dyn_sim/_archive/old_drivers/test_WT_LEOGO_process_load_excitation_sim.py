r"""
Step 4 (network -> turbine, realistic events) -- a LEOGO process-load event
excites the wind-turbine drivetrain torsional mode.

Motivation (Low-Emission angle)
-------------------------------
On an oil & gas platform such as LEOGO the process machinery (gas-export
compressors GEX, pumps, drilling loads) draws tens of MW and is a natural
source of periodic and transient power fluctuations.  All of these loads and
the wind park share the same 11 kV main switchboard ("Main Bus A"), which is
therefore the point of common coupling (PCC) between the platform process
loads and the wind turbine.

Step 3 showed that an oscillating electrical power at 3.49 Hz drives the
drivetrain torsional mode to resonance.  This script replaces the idealised
swept-sine probe with two *physically motivated events* injected at the PCC:

  * scenario "pulsation" -- a steady process load with a small periodic
        power pulsation (e.g. a reciprocating / cyclically loaded compressor
        train) at (or near) the 3.49 Hz torsional frequency.  Demonstrates a
        realistic, modest ripple still driving a meaningful shaft torque, and
        reports the transfer gain (kNm of shaft torque per MW of pulsation).

  * scenario "step" -- a discrete load switching event (a large process load
        block energised at the PCC).  The broadband step excites the lightly
        damped torsional mode, which then rings at 3.49 Hz and decays at the
        modal damping ratio -- proof that *any* disturbance with content near
        the eigenfrequency rings the drivetrain, not just a tuned sinusoid.

The disturbance is applied at Main Bus A (the PCC) as a shunt-load admittance,
exactly as in step 3 but interpreted as an aggregate process-load event.

Examples
--------
Realistic pulsation (0.5 MW ripple at 3.49 Hz on the main switchboard):
  .\.venv\Scripts\python.exe casestudies\dyn_sim\test_WT_LEOGO_process_load_excitation_sim.py --scenario pulsation --amp-mw 0.5 --forcing-freq-hz 3.49 --t-end 40

Discrete load-switching event (5 MW block energised at t=10 s):
  .\.venv\Scripts\python.exe casestudies\dyn_sim\test_WT_LEOGO_process_load_excitation_sim.py --scenario step --amp-mw 5.0 --t-end 30
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
from casestudies.dyn_sim.sweep_em_interaction import (
    generator_speed_field,
    coi_speed_pu,
)


def scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def run_case(*, args: argparse.Namespace):
    """Run one process-load-event case and return the result DataFrame."""
    model = build_model()
    f_nom = float(model["f"])
    s_base_mva = float(model["base_mva"])

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)
    wt_model = ps.windturbine["WindTurbine"]
    uic_model = ps.vsc["UIC_sig"]
    gen_model = ps.gen["GEN"]

    # Frequency-support droop. Off (default): pure MPT operating point so the
    # only dynamic forcing is the process-load event injected at the PCC. On
    # (--wt-droop): the turbine is de-loaded by headroom and raises/lowers its
    # power reference with the measured grid frequency, so the comparison
    # against the off run isolates the droop *response*.
    if args.wt_droop:
        wt_model.par["droop_enable"][:] = 1
        wt_model.par["K_droop_pu_per_hz"][:] = 0.75
        wt_model.par["headroom_pu"][:] = 0.05
    else:
        wt_model.par["droop_enable"][:] = 0
        wt_model.par["K_droop_pu_per_hz"][:] = 0.0
        wt_model.par["headroom_pu"][:] = 0.0
    wt_model.par["f_nom_hz"][:] = f_nom
    wt_model.set_grid_frequency_hz(f_nom)

    wind_speed_initial = scalar(wt_model.wind_speed_init())
    p_ref_initial = scalar(
        wt_model.P_ref_from_wind(wind_speed_initial, uic_model.par["S_n"])
    )
    uic_model.par["p_ref"][:] = p_ref_initial
    uic_model.par["q_ref"][:] = 0.0

    ps.power_flow()
    ps.init_dyn_sim()
    x0 = ps.x0.copy()

    # Centre-of-inertia grid frequency feeding the WT frequency-support droop.
    # With droop off this returns f_nom exactly, so the off run is unchanged.
    speed_field = generator_speed_field(gen_model, x0)
    coi0 = coi_speed_pu(gen_model, x0, speed_field)

    def grid_frequency_hz(x: np.ndarray) -> float:
        if not args.wt_droop:
            return f_nom
        coi = coi_speed_pu(gen_model, x, speed_field)
        return f_nom * (1.0 + (coi - coi0))

    # Inject at Main Bus A -- the point of common coupling between the LEOGO
    # process loads and the wind park.
    main_bus_idx = int(gen_model.bus_idx_red["terminal"][0])
    wt_bus_idx = int(uic_model.bus_idx_red["terminal"][0])

    # Shunt admittance for a 1 MW load; scaled by the event profile.
    y_load_unit = args.amp_mw / s_base_mva

    omega_rated_rad_s = scalar(wt_model.par["omega_m_rated"])
    torque_base_nm = scalar(wt_model.par["S_n"]) * 1e6 / omega_rated_rad_s
    K_pu = scalar(wt_model.par["K"])
    D_pu = scalar(wt_model.par["D"])
    uic_s_n = scalar(uic_model.par["S_n"])
    wt_s_n = scalar(wt_model.par["S_n"])
    sys_s_n = float(wt_model.sys_par["s_n"])

    onset = args.onset
    ramp = max(args.ramp, 1e-6)
    f_hz = args.forcing_freq_hz

    def load_scale(t: float) -> float:
        """Event profile (per unit of args.amp_mw)."""
        if t < onset:
            return 0.0
        if args.scenario == "pulsation":
            # Zero-mean AC component of a cyclically loaded process machine.
            return np.sin(2.0 * np.pi * f_hz * (t - onset))
        # scenario == "step": a load block energised at onset, smoothly ramped
        # over 'ramp' seconds so the event is a realistic breaker/soft-start
        # rather than a numerical impulse.
        return float(min((t - onset) / ramp, 1.0))

    def apply_external_inputs(t: float, x: np.ndarray) -> None:
        wt_model._sim_time = t
        ps.y_bus_red_mod[(main_bus_idx, main_bus_idx)] = load_scale(t) * y_load_unit
        wt_model.set_grid_frequency_hz(grid_frequency_hz(x))

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

        v_terminal = abs(uic_model.v_t(x, v)[0])
        v_mainbus = float(abs(v[main_bus_idx]))
        gen_speed_mean = float(np.mean(gen_model.speed(x, v)))

        rows.append({
            "t": float(t),
            "scenario": args.scenario,
            "load_scale": load_scale(t),
            "load_mw": load_scale(t) * args.amp_mw,
            "omega_m_pu": omega_m,
            "omega_e_pu": omega_e,
            "omega_s_pu": omega_s,
            "theta_s_state": theta_s,
            "T_shaft_pu": t_shaft_pu,
            "T_shaft_Nm": t_shaft_pu * torque_base_nm,
            "T_e_pu": t_e_pu,
            "P_e_wt_pu": p_e_wt_pu,
            "V_wt_terminal_pu": float(v_terminal),
            "V_mainbus_pu": v_mainbus,
            "grid_freq_hz": grid_frequency_hz(x),
            "gen_speed_mean_pu": gen_speed_mean,
        })

    store_row(0.0, x0)
    while solver.t < args.t_end:
        solver.step()
        store_row(solver.t, solver.x)

    return pd.DataFrame(rows)


def _highpass(y: np.ndarray, dt: float, f_cut: float) -> np.ndarray:
    """Zero-phase high-pass to isolate the torsional ring from the slow swing."""
    from scipy.signal import butter, filtfilt

    b, a = butter(2, f_cut / (0.5 / dt), btype="highpass")
    return filtfilt(b, a, y)


def analyse(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, float]:
    """Post-onset FFT peak, shaft-torque swing, and (for 'step') ring-down zeta."""
    t = df["t"].to_numpy()
    y = df["T_shaft_Nm"].to_numpy()
    post = t >= args.onset
    tp, yp = t[post], y[post] - np.mean(y[post])
    dt = float(np.median(np.diff(tp)))

    out = {
        "scenario": args.scenario,
        "amp_mw": args.amp_mw,
        "T_shaft_pp_Nm": float(np.ptp(yp)),
    }

    if args.scenario == "pulsation":
        win = np.hanning(len(yp))
        spec = np.abs(np.fft.rfft(yp * win))
        freqs = np.fft.rfftfreq(len(yp), dt)
        out["fft_peak_hz"] = float(freqs[int(np.argmax(spec[1:]) + 1)])
        # steady swing amplitude (last 40 % of the record) and transfer gain
        tail = yp[int(0.6 * len(yp)):]
        amp = 0.5 * float(np.ptp(tail))
        out["T_shaft_amp_Nm"] = amp
        out["gain_kNm_per_MW"] = 1e-3 * amp / max(args.amp_mw, 1e-9)
    else:
        # The raw step response is dominated by the slow electromechanical
        # settling of the LEOGO gensets/governors; high-pass to isolate the
        # 3.49 Hz torsional ring, then measure its frequency and decay.
        ring = _highpass(yp, dt, f_cut=1.5)
        # skip the filter edge transient at both ends
        edge = int(0.5 / dt)
        tr, rr = tp[edge:-edge], ring[edge:-edge]
        win = np.hanning(len(rr))
        spec = np.abs(np.fft.rfft(rr * win))
        freqs = np.fft.rfftfreq(len(rr), dt)
        f_peak = float(freqs[int(np.argmax(spec[1:]) + 1)])
        out["fft_peak_hz"] = f_peak
        out["T_shaft_ring_pp_Nm"] = float(np.ptp(rr))
        out["zeta_ringdown_pct"] = _ringdown_zeta(tr, rr, f_peak)

    return out


def _ringdown_zeta(t: np.ndarray, y: np.ndarray, f_hz: float) -> float:
    """Estimate zeta from the decay of successive oscillation peaks."""
    # find local maxima of the (absolute) envelope
    a = np.abs(y)
    idx = np.where((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:]))[0] + 1
    if len(idx) < 3:
        return float("nan")
    tp, ap = t[idx], a[idx]
    # keep the decaying part after the initial peak
    kmax = int(np.argmax(ap))
    tp, ap = tp[kmax:], ap[kmax:]
    mask = ap > 0.05 * ap[0]
    tp, ap = tp[mask], ap[mask]
    if len(tp) < 3:
        return float("nan")
    # ln(envelope) = ln A0 - sigma * t ; sigma = zeta * 2 pi f_n / sqrt(1-z^2) ~ zeta*wn
    slope = np.polyfit(tp - tp[0], np.log(ap), 1)[0]
    sigma = -slope
    wn = 2.0 * np.pi * f_hz
    zeta = sigma / np.sqrt(wn**2 + sigma**2)
    return float(100.0 * zeta)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LEOGO process-load event exciting the WT torsional mode."
    )
    parser.add_argument("--scenario", choices=["pulsation", "step"],
                        default="pulsation")
    parser.add_argument("--forcing-freq-hz", type=float, default=3.49,
                        help="Pulsation frequency (Hz), scenario 'pulsation'.")
    parser.add_argument("--amp-mw", type=float, default=0.5,
                        help="Pulsation amplitude or step size (MW at the PCC).")
    parser.add_argument("--ramp", type=float, default=0.05,
                        help="Rise time of the load step (s), scenario 'step'.")
    parser.add_argument("--t-end", type=float, default=40.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--onset", type=float, default=10.0,
                        help="Time at which the process-load event starts (s).")
    parser.add_argument("--wt-droop", action="store_true",
                        help="Enable the WT frequency-support droop (feeds the "
                             "measured COI grid frequency; de-loads by headroom).")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "csv_files"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"Scenario '{args.scenario}' at the PCC (Main Bus A), "
          f"amp = {args.amp_mw:.3f} MW"
          + (f", f = {args.forcing_freq_hz:.3f} Hz" if args.scenario == "pulsation"
             else f", ramp = {args.ramp:.3f} s") + " ...")

    df = run_case(args=args)
    m = analyse(df, args)

    if args.out:
        out = args.out
    else:
        if args.scenario == "pulsation":
            tag = f"{args.forcing_freq_hz:.2f}Hz_{args.amp_mw:.2f}MW".replace(".", "p")
        else:
            tag = f"{args.amp_mw:.2f}MW".replace(".", "p")
        out = str(output_dir / f"WT1_LEOGO_process_{args.scenario}_{tag}.csv")
    df.to_csv(out, index=False)

    print(f"  saved {out}")
    print(f"  T_shaft p2p = {m['T_shaft_pp_Nm']:.3e} Nm,  FFT peak = {m['fft_peak_hz']:.3f} Hz")
    if args.scenario == "pulsation":
        print(f"  steady shaft-torque amplitude = {m['T_shaft_amp_Nm']:.3e} Nm")
        print(f"  transfer gain = {m['gain_kNm_per_MW']:.1f} kNm shaft torque per MW pulsation")
    else:
        print(f"  isolated torsional ring p2p = {m['T_shaft_ring_pp_Nm']:.3e} Nm")
        print(f"  ring-down damping ratio = {m['zeta_ringdown_pct']:.2f} % "
              f"(modal analysis: 4.37 %)")
    print(f"\nTotal runtime: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
