r"""
"Foerste time i morgen"-eksperimentet: slaar UIC perfect_tracking av/paa den
elektromekaniske koblingen alene?

Rammeverket (interaksjon krever tre ting samtidig):
  (i)   aapen E->M-vei   -- her styrt av UIC perfect_tracking,
  (ii)  en mekanisk modus i rekkevidde -- aksel-torsjon ~3.49 Hz OG taarn-SS ~0.234 Hz,
  (iii) nok koblingsgain / svakt nok nett.

Dette skriptet holder (ii) og (iii) faste (samme vind, samme lastesteg, samme
nett) og vrir KUN paa (i): perfect_tracking = 0 (grid-forming, nettet naar
generatormomentet) vs 1 (grid-following, converteren holder ~konstant effekt).
Begge kjoeringene faar det samme brede lastesteget paa hovednettbussen, og BAADE
akselmomentet (torsjon) og taarn-topp-akselerasjonen (side-til-side) logges.

Resultatet er EN ren poster-figur (2x2):
  * rad 1: aksel-torsjon  -- tidsserie (venstre) + spekter (hoyre),
  * rad 2: taarn side-til-side -- tidsserie (venstre) + spekter (hoyre),
med grid-forming (bla) vs grid-following (oransje) lagt oppaa hverandre. For hver
modus skrives topp-amplituden for begge tilfeller + forsterkningsforholdet ut, saa
man ser om perfect_tracking ALENE slaar modusene av/paa.

Eksempel
--------
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_em_interaction_pt_toggle.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_em_interaction_pt_toggle.py --wind 8 --show
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

from casestudies.dyn_sim.test_WT_LEOGO_tower_sim import build_model, scalar
from casestudies.dyn_sim.sweep_em_interaction import (
    generator_speed_field,
    coi_speed_pu,
)


# ---------------------------------------------------------------------
# One case -> full shaft-torque and tower-SS time series
# ---------------------------------------------------------------------

def run_case_series(
    *,
    wind_mps: float,
    perfect_tracking: int,
    t_filter: float,
    onset: float,
    t_end: float,
    dt: float,
    load_step_mw: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one scenario; return (t, shaft_torque_pu, ss_accel_mps2)."""
    # WindTurbineTower with both structural modes on, outer droop OFF so the
    # comparison isolates the inner (perfect_tracking) coupling.
    model = build_model(
        ss_enable=1,
        fa_enable=1,
        droop_enable=0,
    )
    f_nom = float(model["f"])
    s_base_mva = float(model["base_mva"])

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)
    wt_model = ps.windturbine_tower["WindTurbineTower"]
    uic_model = ps.vsc["UIC_sig"]
    gen_model = ps.gen["GEN"]

    # Constant wind operating point (override the hard-coded 11 m/s).
    wt_model.wind_speed = lambda x, v, u=wind_mps: u
    wt_model.wind_speed_init = lambda u=wind_mps: u

    uic_model.par["perfect_tracking"][:] = int(perfect_tracking)
    uic_model.par["T_filter"][:] = float(t_filter)

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
    shaft_series: list[float] = []
    ss_series: list[float] = []

    def record(t: float, x: np.ndarray) -> None:
        apply_external_inputs(t, x)
        v = ps.solve_algebraic(t, x)
        X = wt_model.local_view(x)
        theta_s = scalar(X["theta_m"]) - scalar(X["theta_e"])
        omega_s = scalar(X["omega_m"]) - scalar(X["omega_e"])
        t_series.append(t)
        shaft_series.append(K_pu * theta_s + D_pu * omega_s)
        ss_series.append(scalar(wt_model.ss_acceleration(x, v)))

    record(0.0, x0)
    while solver.t < t_end:
        solver.step()
        record(solver.t, solver.x)

    return (
        np.asarray(t_series),
        np.asarray(shaft_series),
        np.asarray(ss_series),
    )


def post_onset_spectrum(
    t: np.ndarray, sig: np.ndarray, onset: float, f_hi: float
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


def peak_in_band(
    freqs: np.ndarray, amp: np.ndarray, lo: float, hi: float
) -> tuple[float, float]:
    band = (freqs >= lo) & (freqs <= hi)
    i = int(np.argmax(amp[band]))
    return float(freqs[band][i]), float(amp[band][i])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind", type=float, default=10.0,
                        help="Fixed wind speed [m/s] (10 = Region 2, clean torsional peak).")
    parser.add_argument("--t-filter", type=float, default=0.01,
                        help="UIC T_filter [s] (used in both runs).")
    parser.add_argument("--t-end", type=float, default=70.0,
                        help="Long enough for the slow (~0.234 Hz) tower ring-down.")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--onset", type=float, default=10.0,
                        help="Load-step time [s].")
    parser.add_argument("--load-step-mw", type=float, default=5.0,
                        help="Broadband load step at the main grid bus.")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    common = dict(
        wind_mps=args.wind,
        t_filter=args.t_filter,
        onset=args.onset,
        t_end=args.t_end,
        dt=args.dt,
        load_step_mw=args.load_step_mw,
    )

    print(
        f"First-hour experiment: U={args.wind:g} m/s, "
        f"load step {args.load_step_mw:g} MW at t={args.onset:g}s, "
        f"T_filter={args.t_filter:g}s\n"
    )

    print("Running perfect_tracking = 0 (grid-forming)  ...", flush=True)
    t_gf, sh_gf, ss_gf = run_case_series(perfect_tracking=0, **common)
    print("Running perfect_tracking = 1 (grid-following) ...", flush=True)
    t_gl, sh_gl, ss_gl = run_case_series(perfect_tracking=1, **common)

    # Shaft torsion: 2.5-4.5 Hz avoids the 5.28 Hz LEOGO-gen artifact and the
    # ~1.5 Hz converter mode. Tower SS: 0.15-0.35 Hz around 0.234 Hz.
    fsh_gf, ash_gf = post_onset_spectrum(t_gf, sh_gf, args.onset, f_hi=8.0)
    fsh_gl, ash_gl = post_onset_spectrum(t_gl, sh_gl, args.onset, f_hi=8.0)
    fss_gf, ass_gf = post_onset_spectrum(t_gf, ss_gf, args.onset, f_hi=1.5)
    fss_gl, ass_gl = post_onset_spectrum(t_gl, ss_gl, args.onset, f_hi=1.5)

    fsh_pk_gf, ash_pk_gf = peak_in_band(fsh_gf, ash_gf, 2.5, 4.5)
    fsh_pk_gl, ash_pk_gl = peak_in_band(fsh_gl, ash_gl, 2.5, 4.5)
    fss_pk_gf, ass_pk_gf = peak_in_band(fss_gf, ass_gf, 0.15, 0.35)
    fss_pk_gl, ass_pk_gl = peak_in_band(fss_gl, ass_gl, 0.15, 0.35)

    shaft_ratio = ash_pk_gf / ash_pk_gl if ash_pk_gl > 0 else float("inf")
    ss_ratio = ass_pk_gf / ass_pk_gl if ass_pk_gl > 0 else float("inf")

    print("\n=== Shaft torsion (~3.49 Hz) ===")
    print(f"  grid-forming  (pt=0): {ash_pk_gf:.3e} pu  at {fsh_pk_gf:.2f} Hz")
    print(f"  grid-following(pt=1): {ash_pk_gl:.3e} pu  at {fsh_pk_gl:.2f} Hz")
    print(f"  grid-forming rings the shaft x{shaft_ratio:.2f} harder")

    print("\n=== Tower side-to-side (~0.234 Hz) ===")
    print(f"  grid-forming  (pt=0): {ass_pk_gf:.3e} m/s^2 at {fss_pk_gf:.3f} Hz")
    print(f"  grid-following(pt=1): {ass_pk_gl:.3e} m/s^2 at {fss_pk_gl:.3f} Hz")
    print(f"  grid-forming rings the tower x{ss_ratio:.2f} harder")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("\nmatplotlib not available; skipping figure.")
        return

    c_gf = "tab:blue"     # grid-forming (open path)
    c_gl = "tab:orange"   # grid-following (throttled path)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    (ax_sh_t, ax_sh_f), (ax_ss_t, ax_ss_f) = axes

    # --- Row 1: shaft torsion -------------------------------------------
    ax_sh_t.plot(t_gf, sh_gf, color=c_gf, lw=1.1, label="grid-forming (pt=0)")
    ax_sh_t.plot(t_gl, sh_gl, color=c_gl, lw=1.1, label="grid-following (pt=1)")
    ax_sh_t.axvline(args.onset, color="grey", ls="--", lw=1.0, label="load step")
    ax_sh_t.set_xlim(args.onset - 1.0, min(args.onset + 8.0, args.t_end))
    ax_sh_t.set_xlabel("Time [s]")
    ax_sh_t.set_ylabel("Shaft torque [pu]")
    ax_sh_t.set_title("Drivetrain torsion: response to the grid load step")
    ax_sh_t.grid(True, alpha=0.3)
    ax_sh_t.legend(loc="upper right", fontsize=8)

    ax_sh_f.plot(fsh_gf, ash_gf, color=c_gf, lw=1.4, label="grid-forming (pt=0)")
    ax_sh_f.plot(fsh_gl, ash_gl, color=c_gl, lw=1.4, label="grid-following (pt=1)")
    ax_sh_f.axvline(fsh_pk_gf, color="grey", ls="--", lw=1.0,
                    label=f"torsional mode ~{fsh_pk_gf:.2f} Hz")
    ax_sh_f.set_xlim(0.0, 6.0)
    ax_sh_f.set_xlabel("Frequency [Hz]")
    ax_sh_f.set_ylabel("Shaft-torque amplitude [pu]")
    ax_sh_f.set_title(f"Torsion spectrum (pt=0 x{shaft_ratio:.2f} vs pt=1)")
    ax_sh_f.grid(True, alpha=0.3)
    ax_sh_f.legend(loc="upper right", fontsize=8)

    # --- Row 2: tower side-to-side --------------------------------------
    ax_ss_t.plot(t_gf, ss_gf, color=c_gf, lw=1.1, label="grid-forming (pt=0)")
    ax_ss_t.plot(t_gl, ss_gl, color=c_gl, lw=1.1, label="grid-following (pt=1)")
    ax_ss_t.axvline(args.onset, color="grey", ls="--", lw=1.0, label="load step")
    ax_ss_t.set_xlim(args.onset - 2.0, args.t_end)
    ax_ss_t.set_xlabel("Time [s]")
    ax_ss_t.set_ylabel("Tower SS accel [m/s$^2$]")
    ax_ss_t.set_title("Tower side-to-side: response to the grid load step")
    ax_ss_t.grid(True, alpha=0.3)
    ax_ss_t.legend(loc="upper right", fontsize=8)

    ax_ss_f.plot(fss_gf, ass_gf, color=c_gf, lw=1.4, label="grid-forming (pt=0)")
    ax_ss_f.plot(fss_gl, ass_gl, color=c_gl, lw=1.4, label="grid-following (pt=1)")
    ax_ss_f.axvline(fss_pk_gf, color="grey", ls="--", lw=1.0,
                    label=f"SS mode ~{fss_pk_gf:.3f} Hz")
    ax_ss_f.set_xlim(0.0, 1.0)
    ax_ss_f.set_xlabel("Frequency [Hz]")
    ax_ss_f.set_ylabel("Tower SS amplitude [m/s$^2$]")
    ax_ss_f.set_title(f"Tower SS spectrum (pt=0 x{ss_ratio:.2f} vs pt=1)")
    ax_ss_f.grid(True, alpha=0.3)
    ax_ss_f.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"perfect_tracking OFF (grid-forming) vs ON (grid-following)  --  "
        f"U={args.wind:g} m/s: grid-forming rings shaft x{shaft_ratio:.2f}, "
        f"tower x{ss_ratio:.2f} harder"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_dir = (
        PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "em_interaction_sweep"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "em_interaction_pt_toggle.png"
    fig.savefig(png_path, dpi=150)
    print(f"\nPlot written to: {png_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
