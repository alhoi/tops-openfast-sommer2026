r"""
Forklaringsfigur for den elektromekaniske interaksjonen i den reduserte
LEOGO-vindturbinen.

I motsetning til oppsummeringsfiguren fra sweep_em_interaction.py (som kollapser
hele spekteret ned til ETT tall per kjoering) viser denne figuren fysikken
direkte for EN representativ case:

  * venstre panel (tidsserie): et lastesteg paa hovednettbussen sparker
    drivverket; akselmomentet ringer ned ved torsjonsmodets egenfrekvens.
  * hoyre panel (spekter): FFT av akselmomentet etter steget -- toppen ved
    ~3.49 Hz er torsjonsmodet. Droop paa loefter denne toppen (sterkere
    E->M-kobling), droop av gir en lavere topp.

Kjoeres for droop av vs droop paa, saa de to kurvene kan sammenlignes.

Eksempel
--------
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_em_interaction_timeseries_fft.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_em_interaction_timeseries_fft.py --wind 10 --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tops.dynamic as dps
import tops.solvers as dps_sol
import tops_openfast.dyn_models as ext_lib

from casestudies.dyn_sim.test_WT_LEOGO_sim import build_model
from casestudies.dyn_sim.sweep_em_interaction import (
    scalar,
    generator_speed_field,
    coi_speed_pu,
)


# ---------------------------------------------------------------------
# One case -> full shaft-torque time series
# ---------------------------------------------------------------------

def run_case_series(
    *,
    wind_mps: float,
    perfect_tracking: int,
    t_filter: float,
    droop_enabled: bool,
    onset: float,
    t_end: float,
    dt: float,
    load_step_mw: float,
    headroom_pu: float,
    droop_gain_pu_per_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one scenario and return (t, shaft_torque_pu) over the whole sim."""
    model = build_model()
    f_nom = float(model["f"])
    s_base_mva = float(model["base_mva"])

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)
    wt_model = ps.windturbine["WindTurbine"]
    uic_model = ps.vsc["UIC_sig"]
    gen_model = ps.gen["GEN"]

    # Constant wind operating point (override the hard-coded 11 m/s).
    wt_model.wind_speed = lambda x, v, u=wind_mps: u
    wt_model.wind_speed_init = lambda u=wind_mps: u

    uic_model.par["perfect_tracking"][:] = int(perfect_tracking)
    uic_model.par["T_filter"][:] = float(t_filter)

    wt_model.par["f_nom_hz"][:] = f_nom
    wt_model.par["headroom_pu"][:] = headroom_pu
    wt_model.par["droop_enable"][:] = int(droop_enabled)
    wt_model.par["K_droop_pu_per_hz"][:] = (
        droop_gain_pu_per_hz if droop_enabled else 0.0
    )
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

    speed_field = generator_speed_field(gen_model, x0)
    coi0 = coi_speed_pu(gen_model, x0, speed_field)

    main_bus_idx = int(gen_model.bus_idx_red["terminal"][0])
    y_load_step = load_step_mw / s_base_mva

    K_pu = scalar(wt_model.par["K"])
    D_pu = scalar(wt_model.par["D"])

    def grid_frequency_hz(x: np.ndarray) -> float:
        coi = coi_speed_pu(gen_model, x, speed_field)
        return f_nom * (1.0 + (coi - coi0))

    def apply_external_inputs(t: float, x: np.ndarray) -> None:
        wt_model._sim_time = t
        ps.y_bus_red_mod[(main_bus_idx, main_bus_idx)] = (
            y_load_step if t >= onset else 0.0
        )
        wt_model.set_grid_frequency_hz(grid_frequency_hz(x))

    def f_ode(t: float, x: np.ndarray) -> np.ndarray:
        apply_external_inputs(t, x)
        v = ps.solve_algebraic(t, x)
        return ps.state_derivatives(t, x, v)

    solver = dps_sol.SimpleRK4(f_ode, 0.0, x0, t_end, max_step=dt)

    t_series: list[float] = []
    t_shaft_series: list[float] = []

    def record(t: float, x: np.ndarray) -> None:
        X = wt_model.local_view(x)
        theta_s = scalar(X["theta_m"]) - scalar(X["theta_e"])
        omega_s = scalar(X["omega_m"]) - scalar(X["omega_e"])
        t_series.append(t)
        t_shaft_series.append(K_pu * theta_s + D_pu * omega_s)

    apply_external_inputs(0.0, x0)
    record(0.0, x0)
    while solver.t < t_end:
        solver.step()
        record(solver.t, solver.x)

    return np.asarray(t_series), np.asarray(t_shaft_series)


def post_onset_spectrum(
    t: np.ndarray, sig: np.ndarray, onset: float, f_hi: float = 8.0
) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum of the mean-removed post-onset signal."""
    mask = t >= onset
    tt = t[mask]
    s = np.asarray(sig, dtype=float)[mask]
    s = s - np.mean(s)
    dt = float(np.mean(np.diff(tt)))
    n = s.size
    freqs = np.fft.rfftfreq(n, dt)
    amp = np.abs(np.fft.rfft(s)) * 2.0 / n
    keep = freqs <= f_hi
    return freqs[keep], amp[keep]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind", type=float, default=12.0,
                        help="Wind speed [m/s] (12 = Region 3, clean torsional peak).")
    parser.add_argument("--perfect-tracking", type=int, default=0,
                        help="UIC perfect_tracking (0 = grid-forming, strongest coupling).")
    parser.add_argument("--t-filter", type=float, default=0.01,
                        help="UIC T_filter [s].")
    parser.add_argument("--t-end", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--onset", type=float, default=5.0,
                        help="Load-step time [s].")
    parser.add_argument("--load-step-mw", type=float, default=2.5)
    parser.add_argument("--headroom-pu", type=float, default=0.05)
    parser.add_argument("--droop-gain-pu-per-hz", type=float, default=0.50)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    common = dict(
        wind_mps=args.wind,
        perfect_tracking=args.perfect_tracking,
        t_filter=args.t_filter,
        onset=args.onset,
        t_end=args.t_end,
        dt=args.dt,
        load_step_mw=args.load_step_mw,
        headroom_pu=args.headroom_pu,
        droop_gain_pu_per_hz=args.droop_gain_pu_per_hz,
    )

    print(f"Representative case: U={args.wind:g} m/s, "
          f"perfect_tracking={args.perfect_tracking}, T_filter={args.t_filter:g}s, "
          f"load step {args.load_step_mw:g} MW at t={args.onset:g}s\n")

    print("Running droop OFF ...", flush=True)
    t_off, sh_off = run_case_series(droop_enabled=False, **common)
    print("Running droop ON  ...", flush=True)
    t_on, sh_on = run_case_series(droop_enabled=True, **common)

    f_off, a_off = post_onset_spectrum(t_off, sh_off, args.onset)
    f_on, a_on = post_onset_spectrum(t_on, sh_on, args.onset)

    def peak_in_band(freqs, amp, lo=2.5, hi=4.5):
        band = (freqs >= lo) & (freqs <= hi)
        i = int(np.argmax(amp[band]))
        return float(freqs[band][i]), float(amp[band][i])

    fpk_off, apk_off = peak_in_band(f_off, a_off)
    fpk_on, apk_on = peak_in_band(f_on, a_on)
    print(f"\nTorsional peak (droop OFF): {apk_off:.3e} pu at {fpk_off:.2f} Hz")
    print(f"Torsional peak (droop ON ): {apk_on:.3e} pu at {fpk_on:.2f} Hz")
    print(f"Droop raises the peak by x{apk_on / apk_off:.2f}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping figure.")
        return

    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # --- (C) time series -------------------------------------------------
    ax_t.plot(t_off, sh_off, color="tab:orange", lw=1.2, label="droop off")
    ax_t.plot(t_on, sh_on, color="tab:blue", lw=1.2, label="droop on")
    ax_t.axvline(args.onset, color="grey", ls="--", lw=1.0, label="load step")
    ax_t.set_xlim(args.onset - 1.0, min(args.onset + 15.0, args.t_end))
    ax_t.set_xlabel("Time [s]")
    ax_t.set_ylabel("Shaft torque [pu]")
    ax_t.set_title("Drivetrain response to a grid load step")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="upper right")

    # --- (A) spectrum ----------------------------------------------------
    ax_f.plot(f_off, a_off, color="tab:orange", lw=1.4, label="droop off")
    ax_f.plot(f_on, a_on, color="tab:blue", lw=1.4, label="droop on")
    ax_f.axvline(fpk_on, color="grey", ls="--", lw=1.0,
                 label=f"torsional mode ~{fpk_on:.2f} Hz")
    ax_f.set_xlim(0.0, 6.0)
    ax_f.set_xlabel("Frequency [Hz]")
    ax_f.set_ylabel("Shaft-torque amplitude [pu]")
    ax_f.set_title("Spectrum of the shaft torque after the step")
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(loc="upper right")

    fig.suptitle(
        f"Electromechanical interaction  "
        f"(U={args.wind:g} m/s, grid-forming, droop raises torsional peak "
        f"x{apk_on / apk_off:.2f})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_dir = (
        PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "em_interaction_sweep"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "em_interaction_timeseries_fft.png"
    fig.savefig(png_path, dpi=150)
    print(f"\nPlot written to: {png_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
