r"""
Step 4b (network -> turbine, realistic events) -- a LEOGO low-frequency
process-load event (slug flow / severe slugging) excites the wind-turbine
tower side-to-side (SS) mode.

Motivation (Low-Emission / O&G angle)
-------------------------------------
The torsional counterpart (test_WT_LEOGO_process_load_excitation_sim.py) showed
a fast reciprocating-compressor pulsation near 3.49 Hz ringing the drivetrain.
The tower structural modes sit ~15x lower in frequency (SS ~0.234 Hz, ~4.3 s
period), so they need a genuinely SLOW process disturbance.

On an oil & gas platform the classic slow, large-amplitude power pulsation is
**severe slugging / slug flow** in the risers and inlet separators: liquid
accumulates and is periodically expelled as a gas/liquid slug, cycling the
electric-submersible-pump / booster-compressor loads (and hence the platform
electrical load) on a multi-second period. When that slug period lands near the
turbine's tower side-to-side eigenfrequency, the power pulsation -- shared over
the common 11 kV switchboard (Main Bus A, the PCC) -- drives the tower SS mode
to resonance through the converter electrical-torque channel.

This is the tower-mode analogue of the compressor-pulsation script: a modest
periodic process-load pulsation at (or near) 0.234 Hz injected at the PCC as a
shunt-load admittance, with the resulting tower SS acceleration measured and
the transfer gain reported. Because the disturbance is an electrical-power
(torque) pulsation, it excites the torque-driven SIDE-TO-SIDE mode but leaves
the thrust-driven FORE-AFT mode essentially untouched -- the same SS-vs-FA
selectivity seen for imposed generator-torque forcing.

Examples
--------
Slug pulsation on resonance (2 MW ripple at 0.234 Hz on the main switchboard):
  .\.venv\Scripts\python.exe casestudies\dyn_sim\test_WT_LEOGO_slugflow_ss_sim.py --amp-mw 2.0 --forcing-freq-hz 0.234 --t-end 120

Off-resonance control (same ripple at 0.12 Hz -- little SS response):
  .\.venv\Scripts\python.exe casestudies\dyn_sim\test_WT_LEOGO_slugflow_ss_sim.py --amp-mw 2.0 --forcing-freq-hz 0.12 --t-end 120
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

from casestudies.dyn_sim.test_WT_LEOGO_tower_sim import build_model, scalar
from casestudies.dyn_sim.sweep_em_interaction import (
    generator_speed_field,
    coi_speed_pu,
)


def run_case(*, args: argparse.Namespace) -> pd.DataFrame:
    """Run one slug-flow process-load case and return the result DataFrame."""
    model = build_model(ss_enable=1, fa_enable=1,
                        droop_enable=1 if args.wt_droop else 0,
                        k_droop_pu_per_hz=args.droop_gain_pu_per_hz,
                        headroom_pu=args.headroom_pu)
    f_nom = float(model["f"])
    s_base_mva = float(model["base_mva"])

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)
    wt_model = ps.windturbine_tower["WindTurbineTower"]
    uic_model = ps.vsc["UIC_sig"]
    gen_model = ps.gen["GEN"]

    # Constant wind operating point (override the hard-coded 11 m/s) so the only
    # dynamic forcing is the process-load pulsation injected at the PCC.
    wt_model.wind_speed = lambda x, v, u=args.wind: u
    wt_model.wind_speed_init = lambda u=args.wind: u

    wt_model.par["f_nom_hz"][:] = f_nom
    wt_model.set_grid_frequency_hz(f_nom)

    p_ref_initial = scalar(
        wt_model.P_ref_from_wind(
            scalar(wt_model.wind_speed_init()), uic_model.par["S_n"]
        )
    )
    uic_model.par["p_ref"][:] = p_ref_initial
    uic_model.par["q_ref"][:] = 0.0

    ps.power_flow()
    ps.init_dyn_sim()
    x0 = ps.x0.copy()

    # Centre-of-inertia grid frequency from the LEOGO synchronous machines.
    # Always computed from the COI so BOTH the droop-on and droop-off runs log
    # the true main-bus frequency swing; the WT droop only *acts* on it when
    # droop_enable=1 (droop-off then leaves the operating point unchanged).
    speed_field = generator_speed_field(gen_model, x0)
    coi0 = coi_speed_pu(gen_model, x0, speed_field)

    def grid_frequency_hz(x: np.ndarray) -> float:
        coi = coi_speed_pu(gen_model, x, speed_field)
        return f_nom * (1.0 + (coi - coi0))

    # Inject at Main Bus A -- the point of common coupling between the LEOGO
    # process loads and the wind park.
    main_bus_idx = int(gen_model.bus_idx_red["terminal"][0])
    inv_s_base = 1.0 / s_base_mva

    K_pu = scalar(wt_model.par["K"])
    D_pu = scalar(wt_model.par["D"])

    onset = args.onset
    f_hz = args.forcing_freq_hz
    hold_start = args.hold_start
    hold_mw = args.hold_mw

    def load_mw_of_t(t: float) -> float:
        """Slug-flow process load [MW] at the PCC: a zero-mean sine pulsation
        that optionally switches to a constant hold (a slug settling to a
        steady offset) at hold_start."""
        if t < onset:
            return 0.0
        if hold_start is not None and t >= hold_start:
            return hold_mw
        return args.amp_mw * float(np.sin(2.0 * np.pi * f_hz * (t - onset)))

    def apply_external_inputs(t: float, x: np.ndarray) -> None:
        wt_model._sim_time = t
        ps.y_bus_red_mod[(main_bus_idx, main_bus_idx)] = load_mw_of_t(t) * inv_s_base
        wt_model.set_grid_frequency_hz(grid_frequency_hz(x))

    def f_ode(t: float, x: np.ndarray) -> np.ndarray:
        apply_external_inputs(t, x)
        v = ps.solve_algebraic(t, x)
        return ps.state_derivatives(t, x, v)

    solver = dps_sol.SimpleRK4(f_ode, 0.0, x0, args.t_end, max_step=args.dt)

    rows: list[dict[str, float]] = []
    uic_s_n = float(np.asarray(uic_model.par["S_n"]).ravel()[0])

    def store_row(t: float, x: np.ndarray) -> None:
        apply_external_inputs(t, x)
        v = ps.solve_algebraic(t, x)
        X = wt_model.local_view(x)
        theta_s = scalar(X["theta_m"]) - scalar(X["theta_e"])
        omega_s = scalar(X["omega_m"]) - scalar(X["omega_e"])
        # WT/UIC terminal electrical quantities at Busbar WTG1 LV (UIC base).
        v_term = uic_model.v_t(x, v)[0]
        s_term = uic_model.s_e(x, v)[0]
        i_term = uic_model.i_a(x, v)[0]
        rows.append({
            "t": float(t),
            "load_scale": load_mw_of_t(t) / max(args.amp_mw, 1e-9),
            "load_mw": load_mw_of_t(t),
            "grid_freq_hz": grid_frequency_hz(x),
            "ss_accel_mps2": scalar(wt_model.ss_acceleration(x, v)),
            "fa_accel_mps2": scalar(wt_model.fa_acceleration(x, v)),
            "q_ss": scalar(X["q_ss"]),
            "q_fa": scalar(X["q_fa"]),
            "T_shaft_pu": K_pu * theta_s + D_pu * omega_s,
            "P_e_pu": scalar(wt_model.P_e(x, v)),
            "V_mainbus_pu": float(abs(v[main_bus_idx])),
            # UIC terminal (WT converter) quantities; UIC base = uic_s_n MVA.
            "V_term_pu": float(abs(v_term)),
            "P_term_pu": float(s_term.real),
            "Q_term_pu": float(s_term.imag),
            "P_term_mw": float(s_term.real) * uic_s_n,
            "I_term_pu": float(abs(i_term)),
        })

    store_row(0.0, x0)
    while solver.t < args.t_end:
        solver.step()
        store_row(solver.t, solver.x)

    return pd.DataFrame(rows)


def _fft_peak_hz(t: np.ndarray, y: np.ndarray) -> float:
    dt = float(np.median(np.diff(t)))
    win = np.hanning(len(y))
    spec = np.abs(np.fft.rfft((y - np.mean(y)) * win))
    freqs = np.fft.rfftfreq(len(y), dt)
    return float(freqs[int(np.argmax(spec[1:]) + 1)])


def _steady_amp(y: np.ndarray) -> float:
    """Half peak-to-peak over the settled tail (last 40 % of the record)."""
    tail = y[int(0.6 * len(y)):]
    return 0.5 * float(np.ptp(tail))


def analyse(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, float]:
    """Post-onset FFT peak and steady SS/FA amplitudes + transfer gain."""
    t = df["t"].to_numpy()
    upper = args.hold_start if args.hold_start is not None else float(t[-1]) + 1.0
    post = (t >= args.onset) & (t < upper)
    tp = t[post]
    ss = df["ss_accel_mps2"].to_numpy()[post]
    fa = df["fa_accel_mps2"].to_numpy()[post]
    q_ss = df["q_ss"].to_numpy()[post]

    amp_ss = _steady_amp(ss)
    amp_fa = _steady_amp(fa)
    return {
        "amp_mw": args.amp_mw,
        "forcing_freq_hz": args.forcing_freq_hz,
        "ss_fft_peak_hz": _fft_peak_hz(tp, ss),
        "ss_accel_amp_mps2": amp_ss,
        "fa_accel_amp_mps2": amp_fa,
        "ss_gain_mps2_per_mw": amp_ss / max(args.amp_mw, 1e-9),
        "ss_over_fa": amp_ss / max(amp_fa, 1e-12),
        "q_ss_amp": _steady_amp(q_ss),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LEOGO slug-flow process load exciting the WT tower SS mode."
    )
    parser.add_argument("--forcing-freq-hz", type=float, default=0.234,
                        help="Slug pulsation frequency (Hz). Tower SS ~0.234 Hz.")
    parser.add_argument("--amp-mw", type=float, default=2.0,
                        help="Slug pulsation amplitude (MW at the PCC).")
    parser.add_argument("--wind", type=float, default=10.0,
                        help="Fixed wind speed [m/s] (10 = Region 2).")
    parser.add_argument("--t-end", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--droop-gain-pu-per-hz", type=float, default=0.75,
                        help="WT frequency-support droop gain K_droop [pu/Hz]. "
                             "Larger = more aggressive frequency support (and "
                             "more electrical-torque ripple into the tower).")
    parser.add_argument("--headroom-pu", type=float, default=0.05,
                        help="De-loaded up-reserve [pu, UIC base]. Caps the "
                             "droop up-response; raise it so an aggressive gain "
                             "does not saturate on frequency dips.")
    parser.add_argument("--hold-start", type=float, default=None,
                        help="If set, the slug pulsation switches to a constant "
                             "hold of --hold-mw at this time [s] (a slug "
                             "settling to a steady offset).")
    parser.add_argument("--hold-mw", type=float, default=1.0,
                        help="Constant load [MW] held after --hold-start.")
    parser.add_argument("--onset", type=float, default=10.0,
                        help="Time at which the slug pulsation starts (s).")
    parser.add_argument("--wt-droop", action="store_true",
                        help="Enable the WT frequency-support droop (feeds the "
                             "measured COI grid frequency; de-loads by headroom).")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "csv_files"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"Slug-flow pulsation at the PCC (Main Bus A), "
          f"amp = {args.amp_mw:.3f} MW, f = {args.forcing_freq_hz:.3f} Hz, "
          f"wind = {args.wind:.1f} m/s ...")

    df = run_case(args=args)
    m = analyse(df, args)

    if args.out:
        out = args.out
    else:
        tag = f"{args.forcing_freq_hz:.3f}Hz_{args.amp_mw:.2f}MW".replace(".", "p")
        out = str(output_dir / f"WT1_LEOGO_slugflow_ss_{tag}.csv")
    df.to_csv(out, index=False)

    print(f"  saved {out}")
    print(f"  SS FFT peak            = {m['ss_fft_peak_hz']:.3f} Hz")
    print(f"  SS accel amplitude     = {m['ss_accel_amp_mps2']:.3e} m/s^2")
    print(f"  SS transfer gain       = {m['ss_gain_mps2_per_mw']:.3e} (m/s^2)/MW")
    print(f"  FA accel amplitude     = {m['fa_accel_amp_mps2']:.3e} m/s^2 "
          f"(SS/FA = {m['ss_over_fa']:.1f}x -> side-to-side selectivity)")
    print(f"\nTotal runtime: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
