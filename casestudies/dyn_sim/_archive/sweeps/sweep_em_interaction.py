r"""
Sweep: NAAR oppstaar elektromekaniske interaksjoner i den reduserte
LEOGO-vindturbinen?

Bakgrunn
--------
I den reduserte modellen er generatormomentet koblet til det elektriske
klemmepunktet gjennom

    T_e = P_e / (omega_e_filt * eta),

der P_e er den faktiske UIC-effekten. En nettforstyrrelse endrer dermed P_e og
forplanter seg inn i drivverket -- E->M-veien er ALLTID aapen her. Dette er
kontrasten til OpenFAST/ROSCO, der momentet settes som K*omega^2 (funksjon av
turtall alene) og generatoren er en ren moment-aktuator uten elektrisk port, saa
nettet ikke naar drivverket.

Dette skriptet kartlegger hvor STERKT en standard nettforstyrrelse (et lastesteg
paa hovednettbussen) eksiterer drivverkets torsjonsmodus (~3.49 Hz), som en
funksjon av:

  * driftspunkt        -- vindhastighet (Region 2 momentstyrt vs Region 3 pitch),
  * converter-kontroll -- UIC perfect_tracking av/paa og T_filter,
  * frekvensstoette     -- ytre droop av/paa.

For hvert tilfelle logges drivverksresponsen, og en resonans-/eksitasjonsmetrikk
regnes ut (FFT-topp i torsjonsbaandet + RMS/peak-to-peak av akselmomentet).

Eksempel
--------
  .\.venv\Scripts\python.exe casestudies\dyn_sim\sweep_em_interaction.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\sweep_em_interaction.py --winds 8 10 12 --show
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import product
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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def scalar(value) -> float:
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


def torsional_metrics(
    t: np.ndarray,
    t_shaft: np.ndarray,
    onset: float,
    f_lo: float = 1.5,
    f_hi: float = 6.0,
) -> dict[str, float]:
    """
    Characterise the drivetrain oscillation after the disturbance.

    Returns the dominant frequency and single-sided amplitude in the torsional
    band [f_lo, f_hi], plus RMS and peak-to-peak of the (mean-removed) shaft
    torque over the post-onset window.
    """
    mask = t >= onset
    tt = t[mask]
    sig = np.asarray(t_shaft, dtype=float)[mask]
    if sig.size < 8:
        return {"fft_peak_hz": 0.0, "fft_peak_amp": 0.0, "shaft_rms": 0.0, "shaft_pp": 0.0}

    sig = sig - np.mean(sig)
    dt = float(np.mean(np.diff(tt)))
    n = sig.size
    freqs = np.fft.rfftfreq(n, dt)
    amp = np.abs(np.fft.rfft(sig)) * 2.0 / n

    band = (freqs >= f_lo) & (freqs <= f_hi)
    if np.any(band):
        i = int(np.argmax(amp[band]))
        f_peak = float(freqs[band][i])
        a_peak = float(amp[band][i])
    else:
        f_peak = 0.0
        a_peak = 0.0

    return {
        "fft_peak_hz": f_peak,
        "fft_peak_amp": a_peak,
        "shaft_rms": float(np.std(sig)),
        "shaft_pp": float(np.max(sig) - np.min(sig)),
    }


# ---------------------------------------------------------------------
# One sweep case
# ---------------------------------------------------------------------

def run_case(
    *,
    wind_mps: float,
    perfect_tracking: int,
    t_filter: float,
    droop_enabled: bool,
    args: argparse.Namespace,
) -> dict[str, float | str | int]:
    """Run one operating scenario and return its interaction metrics."""
    model = build_model()
    f_nom = float(model["f"])
    s_base_mva = float(model["base_mva"])

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)
    wt_model = ps.windturbine["WindTurbine"]
    uic_model = ps.vsc["UIC_sig"]
    gen_model = ps.gen["GEN"]

    # --- Operating point: constant wind (override the hard-coded 11 m/s) -----
    wt_model.wind_speed = lambda x, v, u=wind_mps: u
    wt_model.wind_speed_init = lambda u=wind_mps: u

    # --- Converter control (UIC) --------------------------------------------
    uic_model.par["perfect_tracking"][:] = int(perfect_tracking)
    uic_model.par["T_filter"][:] = float(t_filter)

    # --- Frequency-support droop (WT) ---------------------------------------
    wt_model.par["f_nom_hz"][:] = f_nom
    wt_model.par["headroom_pu"][:] = args.headroom_pu
    wt_model.par["droop_enable"][:] = int(droop_enabled)
    wt_model.par["K_droop_pu_per_hz"][:] = (
        args.droop_gain_pu_per_hz if droop_enabled else 0.0
    )
    wt_model.set_grid_frequency_hz(f_nom)

    # Initialise UIC power reference from the wind operating point.
    p_ref_initial = scalar(
        wt_model.P_ref_from_wind(scalar(wt_model.wind_speed_init()), uic_model.par["S_n"])
    )
    uic_model.par["p_ref"][:] = p_ref_initial
    uic_model.par["q_ref"][:] = 0.0

    ps.power_flow()
    ps.init_dyn_sim()
    x0 = ps.x0.copy()

    # Centre-of-inertia frequency feedback.
    speed_field = generator_speed_field(gen_model, x0)
    coi0 = coi_speed_pu(gen_model, x0, speed_field)

    # Load step at the main (gas-turbine) bus -> network frequency event.
    main_bus_idx = int(gen_model.bus_idx_red["terminal"][0])
    y_load_step = args.load_step_mw / s_base_mva

    K_pu = scalar(wt_model.par["K"])
    D_pu = scalar(wt_model.par["D"])
    wind_rated = scalar(wt_model.par["wind_rated"])

    def grid_frequency_hz(x: np.ndarray) -> float:
        coi = coi_speed_pu(gen_model, x, speed_field)
        return f_nom * (1.0 + (coi - coi0))

    def apply_external_inputs(t: float, x: np.ndarray) -> None:
        wt_model._sim_time = t
        ps.y_bus_red_mod[(main_bus_idx, main_bus_idx)] = (
            y_load_step if t >= args.onset else 0.0
        )
        wt_model.set_grid_frequency_hz(grid_frequency_hz(x))

    def f_ode(t: float, x: np.ndarray) -> np.ndarray:
        apply_external_inputs(t, x)
        v = ps.solve_algebraic(t, x)
        return ps.state_derivatives(t, x, v)

    solver = dps_sol.SimpleRK4(f_ode, 0.0, x0, args.t_end, max_step=args.dt)

    t_series: list[float] = []
    t_shaft_series: list[float] = []
    omega_s_series: list[float] = []

    def record(t: float, x: np.ndarray) -> None:
        X = wt_model.local_view(x)
        theta_s = scalar(X["theta_m"]) - scalar(X["theta_e"])
        omega_s = scalar(X["omega_m"]) - scalar(X["omega_e"])
        t_series.append(t)
        t_shaft_series.append(K_pu * theta_s + D_pu * omega_s)
        omega_s_series.append(omega_s)

    apply_external_inputs(0.0, x0)
    record(0.0, x0)
    while solver.t < args.t_end:
        solver.step()
        record(solver.t, solver.x)

    t_arr = np.asarray(t_series)
    m = torsional_metrics(t_arr, np.asarray(t_shaft_series), args.onset)

    return {
        "wind_mps": wind_mps,
        "region": "R2 (torque)" if wind_mps < wind_rated else "R3 (pitch)",
        "perfect_tracking": int(perfect_tracking),
        "T_filter_s": float(t_filter),
        "droop": int(droop_enabled),
        "fft_peak_hz": m["fft_peak_hz"],
        "fft_peak_amp_pu": m["fft_peak_amp"],
        "shaft_rms_pu": m["shaft_rms"],
        "shaft_pp_pu": m["shaft_pp"],
        "omega_s_rms_pu": float(np.std(np.asarray(omega_s_series))),
        "p_ref_init_uic_pu": p_ref_initial,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-end", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--onset", type=float, default=5.0,
                        help="Time of the load step [s].")
    parser.add_argument("--load-step-mw", type=float, default=2.5,
                        help="Abrupt load step at the main bus [MW].")
    parser.add_argument("--headroom-pu", type=float, default=0.05,
                        help="WT reserve on the 20 MVA UIC base (0.05 pu = 1 MW).")
    parser.add_argument("--droop-gain-pu-per-hz", type=float, default=0.50,
                        help="Droop gain on UIC base [pu/Hz], used only when droop is on.")

    parser.add_argument("--winds", type=float, nargs="+",
                        default=[8.0, 10.0, 12.0],
                        help="Wind speeds to sweep [m/s]. wind_rated = 10.6.")
    parser.add_argument("--perfect-tracking", type=int, nargs="+",
                        default=[0, 1],
                        help="UIC perfect_tracking settings to sweep.")
    parser.add_argument("--t-filters", type=float, nargs="+",
                        default=[0.01, 0.1],
                        help="UIC T_filter values to sweep (only bite when pt=1).")
    parser.add_argument("--droop", type=int, nargs="+",
                        default=[0, 1],
                        help="Droop settings to sweep (0/1).")
    parser.add_argument("--show", action="store_true",
                        help="Show the summary plot in addition to saving it.")

    args = parser.parse_args()

    # Build the case list, de-duplicating T_filter when perfect_tracking = 0
    # (T_filter is inert there, so a single value suffices).
    cases: list[tuple[float, int, float, bool]] = []
    for wind, pt, droop in product(args.winds, args.perfect_tracking, args.droop):
        t_filters = args.t_filters if pt == 1 else args.t_filters[:1]
        for tf in t_filters:
            cases.append((wind, pt, tf, bool(droop)))

    print(f"Running {len(cases)} cases "
          f"(t_end={args.t_end}s, dt={args.dt}s, load step={args.load_step_mw} MW)\n")

    rows: list[dict[str, float | str | int]] = []
    all_start = time.perf_counter()
    for k, (wind, pt, tf, droop) in enumerate(cases, start=1):
        label = (f"U={wind:g} m/s  pt={pt}  T_f={tf:g}s  droop={int(droop)}")
        print(f"[{k:2d}/{len(cases)}] {label} ...", end="", flush=True)
        c_start = time.perf_counter()
        row = run_case(
            wind_mps=wind,
            perfect_tracking=pt,
            t_filter=tf,
            droop_enabled=droop,
            args=args,
        )
        rows.append(row)
        print(f"  f_peak={row['fft_peak_hz']:.2f} Hz  "
              f"amp={row['fft_peak_amp_pu']:.3e} pu  "
              f"({time.perf_counter() - c_start:.1f}s)")

    results = pd.DataFrame(rows)

    output_dir = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "em_interaction_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "em_interaction_sweep.csv"
    results.to_csv(csv_path, index=False)

    print("\n=== Interaction metric (shaft-torque FFT peak amplitude, pu) ===")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(results[[
            "wind_mps", "region", "perfect_tracking", "T_filter_s", "droop",
            "fft_peak_hz", "fft_peak_amp_pu", "shaft_rms_pu",
        ]].to_string(index=False))
    print(f"\nResults written to: {csv_path}")
    print(f"Total wall time: {time.perf_counter() - all_start:.1f} s")

    # Summary plot: interaction strength vs wind, grouped by perfect_tracking,
    # for the fastest T_filter and droop off (a clean baseline slice).
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    slice_df = results[(results["droop"] == 0)]
    tf_min = min(args.t_filters)
    slice_df = slice_df[(slice_df["perfect_tracking"] == 0) |
                        (slice_df["T_filter_s"] == tf_min)]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for pt, grp in slice_df.groupby("perfect_tracking"):
        grp = grp.sort_values("wind_mps")
        ax.plot(grp["wind_mps"], grp["fft_peak_amp_pu"], marker="o",
                label=f"perfect_tracking = {pt}")
    ax.axvline(10.6, color="grey", ls="--", lw=1.0, label="wind_rated (R2/R3)")
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Shaft-torque FFT peak amplitude [pu]")
    ax.set_title("Electromechanical interaction vs operating point (droop off)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = output_dir / "em_interaction_sweep.png"
    fig.savefig(png_path, dpi=150)
    print(f"Plot written to: {png_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
