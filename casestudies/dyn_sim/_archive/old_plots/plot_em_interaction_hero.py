r"""
Hero-figur: den elektromekaniske kausalkjeden i EN LEOGO-nett-hendelse.

Hele poenget med prosjektet paa ett bilde: en ren *elektrisk* forstyrrelse paa
LEOGO-nettet (et GT-trip / brått lastesteg paa hovednettbussen) forplanter seg
gjennom UIC-omformerens elektriske port og inn i vindturbinens *mekanikk*:

    f_nett  ->  P_e (WT elektrisk)  ->  T_e = P_e/(w_e*eta)  ->  aksel-torsjon
                                                              ->  taarn side-til-side

I den reduserte modellen er E->M-veien ALLTID aapen (generatoren er en
momentaktuator uten elektrisk klemme i OpenFAST/ROSCO, saa der er den blokkert).
Derfor ringer BAADE drivverket (~3.49 Hz) og taarnet (~0.234 Hz) fra en ren
nett-hendelse -- det er den elektromekaniske interaksjonen vi studerer.

Figuren (venstre kolonne = kausalkjeden i tid, hoyre kolonne = spektrene som
beviser at ringingen skjer paa egenfrekvensene):

    [ f_nett        ] [ aksel-spekter  (topp ~3.49 Hz) ]
    [ P_e           ] [ aksel-spekter                  ]
    [ aksel-torsjon ] [ taarn-spekter (topp ~0.234 Hz) ]
    [ taarn SS-akse.] [ taarn-spekter                  ]

I tillegg regnes en SLITASJE-metrikk (Damage Equivalent Load, DEL) via
rainflow-telling (ASTM E1049) + Woehler-eksponent:
  * drivverk : DEL paa akselmomentet (m ~ 8, staal),
  * taarn    : DEL paa modal-forskyvningen q_ss (~ boyemoment, m = 4, sveiset staal).
DEL er en RELATIV last (redusert modell = en modal frihetsgrad m/ kalibrert gain),
ikke absolutt levetid -- men den lar frekvensstoette-trade-offen uttrykkes som ett
tall: "droep paa oeker taarnets DEL med X %".

Eksempler
---------
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_em_interaction_hero.py --show
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_em_interaction_hero.py --compare-droop
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
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


# =====================================================================
# Fatigue: rainflow counting (ASTM E1049) + damage-equivalent load
# =====================================================================

def _reversals(series):
    """Yield turning points (reversals) of a series, incl. first and last."""
    series = list(series)
    n = len(series)
    if n < 2:
        if n == 1:
            yield series[0]
        return
    x_last = series[0]
    x = series[1]
    d_last = x - x_last
    yield x_last
    for i in range(2, n):
        x_next = series[i]
        d_next = x_next - x
        if d_last * d_next < 0.0:
            yield x
        if d_next != 0.0:
            d_last = d_next
        x = x_next
    yield series[-1]


def rainflow_cycles(series):
    """Return list of (range, count) using 3-point ASTM E1049 rainflow counting.

    Counts are 0.5 (half cycle) or 1.0 (full cycle).
    """
    points: deque = deque()
    out: list[tuple[float, float]] = []
    for x in _reversals(series):
        points.append(x)
        while len(points) >= 3:
            x1, x2, x3 = points[-3], points[-2], points[-1]
            rng_new = abs(x3 - x2)
            rng_prev = abs(x2 - x1)
            if rng_new < rng_prev:
                break
            if len(points) == 3:
                out.append((rng_prev, 0.5))
                points.popleft()
            else:
                out.append((rng_prev, 1.0))
                last = points.pop()
                points.pop()
                points.pop()
                points.append(last)
    while len(points) > 1:
        x1 = points.popleft()
        out.append((abs(points[0] - x1), 0.5))
    return out


def damage_equivalent_load(series, m):
    """Damage-equivalent constant range for a load signal (Woehler exponent m).

    DEL_eq = ( sum(n_i * S_i^m) / sum(n_i) )^(1/m).
    Duration-independent (damage-weighted RMS range), so it is directly
    comparable between scenarios over similar windows.
    """
    cycles = rainflow_cycles(series)
    num = sum(c * (r ** m) for r, c in cycles)
    den = sum(c for _, c in cycles)
    if den <= 0.0:
        return 0.0
    return (num / den) ** (1.0 / m)


# =====================================================================
# One LEOGO event -> full causal-chain time series
# =====================================================================

def run_event_series(
    *,
    wind_mps: float,
    onset: float,
    t_end: float,
    dt: float,
    load_step_mw: float,
    droop_enable: int,
) -> dict[str, np.ndarray]:
    """Run one grid event; return the causal-chain signals over time.

    Keys: t, grid_freq_hz, p_e_pu, t_e_pu, shaft_torque_pu, ss_accel_mps2, q_ss.
    """
    model = build_model(
        ss_enable=1,
        fa_enable=1,
        droop_enable=int(droop_enable),
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

    rows: dict[str, list[float]] = {
        "t": [],
        "grid_freq_hz": [],
        "p_e_pu": [],
        "t_e_pu": [],
        "shaft_torque_pu": [],
        "ss_accel_mps2": [],
        "q_ss": [],
    }

    def record(t: float, x: np.ndarray) -> None:
        apply_external_inputs(t, x)
        v = ps.solve_algebraic(t, x)
        X = wt_model.local_view(x)
        theta_s = scalar(X["theta_m"]) - scalar(X["theta_e"])
        omega_s = scalar(X["omega_m"]) - scalar(X["omega_e"])
        rows["t"].append(t)
        rows["grid_freq_hz"].append(grid_frequency_hz(x))
        rows["p_e_pu"].append(scalar(wt_model.P_e(x, v)))
        rows["t_e_pu"].append(scalar(wt_model._electromagnetic_torque(x, v)))
        rows["shaft_torque_pu"].append(K_pu * theta_s + D_pu * omega_s)
        rows["ss_accel_mps2"].append(scalar(wt_model.ss_acceleration(x, v)))
        rows["q_ss"].append(scalar(X["q_ss"]))

    record(0.0, x0)
    while solver.t < t_end:
        solver.step()
        record(solver.t, solver.x)

    return {k: np.asarray(v) for k, v in rows.items()}


def post_onset_spectrum(
    t: np.ndarray, sig: np.ndarray, onset: float, f_hi: float,
    detrend_window_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum of the post-onset signal.

    With ``detrend_window_s`` the slow settling transient is removed by
    subtracting a centred moving average of that width (a simple high-pass),
    which isolates a fast oscillatory ring sitting on top of a big step; else
    only the mean is removed.
    """
    mask = t >= onset
    tt = t[mask]
    s = np.asarray(sig, dtype=float)[mask]
    dt = float(np.mean(np.diff(tt)))
    if detrend_window_s:
        w = max(1, int(round(detrend_window_s / dt)))
        trend = np.convolve(s, np.ones(w) / w, mode="same")
        s = s - trend
    else:
        s = s - np.mean(s)
    n = s.size
    freqs = np.fft.rfftfreq(n, dt)
    amp = np.abs(np.fft.rfft(s)) * 2.0 / n
    keep = freqs <= f_hi
    return freqs[keep], amp[keep]


def peak_in_band(
    freqs: np.ndarray, amp: np.ndarray, lo: float, hi: float
) -> tuple[float, float]:
    band = (freqs >= lo) & (freqs <= hi)
    if not np.any(band):
        return float("nan"), float("nan")
    i = int(np.argmax(amp[band]))
    return float(freqs[band][i]), float(amp[band][i])


def event_fatigue(res: dict[str, np.ndarray], onset: float,
                  m_shaft: float, m_tower: float) -> dict[str, float]:
    """DEL of the post-onset drivetrain and tower loads."""
    mask = res["t"] >= onset
    shaft = res["shaft_torque_pu"][mask]
    q_ss = res["q_ss"][mask]
    return {
        "shaft_del_pu": damage_equivalent_load(shaft, m_shaft),
        "tower_del": damage_equivalent_load(q_ss, m_tower),
        "shaft_ptp_pu": float(np.ptp(shaft)),
        "tower_ptp": float(np.ptp(q_ss)),
    }


# =====================================================================
# Figure
# =====================================================================

def make_hero_figure(res, onset, fat, args):
    import matplotlib.pyplot as plt

    t = res["t"]
    tmin = max(0.0, onset - 2.0)
    xlim = (tmin, float(t[-1]))

    # Shaft torque steps permanently (big slow settle) -> detrend so the fast
    # torsional ring is not buried under near-DC content. Tower SS oscillates
    # about zero, so plain mean removal is enough there.
    f_shaft, a_shaft = post_onset_spectrum(
        t, res["shaft_torque_pu"], onset, f_hi=6.0, detrend_window_s=2.0)
    f_tower, a_tower = post_onset_spectrum(
        t, res["ss_accel_mps2"], onset, f_hi=1.0)
    pk_shaft = peak_in_band(f_shaft, a_shaft, 2.5, 4.5)
    pk_tower = peak_in_band(f_tower, a_tower, 0.15, 0.35)

    mosaic = [
        ["gridf", "shaft_sp"],
        ["pe", "shaft_sp"],
        ["shaft", "tower_sp"],
        ["tower", "tower_sp"],
    ]
    fig, ax = plt.subplot_mosaic(
        mosaic, figsize=(13, 9),
        gridspec_kw={"width_ratios": [2.0, 1.0], "hspace": 0.35, "wspace": 0.25},
    )

    chain = ("#1f4e79", "#2e75b6", "#c55a11", "#548235")

    # --- left column: the causal chain in time -----------------------
    ax["gridf"].plot(t, res["grid_freq_hz"], color=chain[0])
    ax["gridf"].set_ylabel("f_nett\n[Hz]")
    ax["gridf"].set_title("Elektrisk forstyrrelse (LEOGO hovednettbuss)",
                          fontsize=10, loc="left")

    ax["pe"].plot(t, res["p_e_pu"], color=chain[1])
    ax["pe"].set_ylabel("P_e\n[pu]")
    ax["pe"].set_title("Vindturbinens elektriske port (UIC)  |  T_e = P_e/(w_e*eta)",
                       fontsize=10, loc="left")

    ax["shaft"].plot(t, res["shaft_torque_pu"], color=chain[2])
    ax["shaft"].set_ylabel("Akselmoment\n[pu]")
    ax["shaft"].set_title("Drivverk (torsjon)", fontsize=10, loc="left")

    ax["tower"].plot(t, res["ss_accel_mps2"], color=chain[3])
    ax["tower"].set_ylabel("Tårn SS\n[m/s², rel.]")
    ax["tower"].set_xlabel("Tid [s]")
    ax["tower"].set_title("Tårn side-til-side", fontsize=10, loc="left")

    for key in ("gridf", "pe", "shaft", "tower"):
        ax[key].axvline(onset, color="0.4", ls="--", lw=1.0)
        ax[key].set_xlim(*xlim)
        ax[key].grid(True, alpha=0.3)
    for key in ("gridf", "pe", "shaft"):
        ax[key].tick_params(labelbottom=False)

    # --- right column: spectra prove the eigenfrequencies ------------
    # Show only the 1-6 Hz band for the shaft so the torsional peak is not
    # dwarfed by the residual low-frequency step content.
    sh_band = f_shaft >= 1.0
    ax["shaft_sp"].plot(f_shaft[sh_band], a_shaft[sh_band], color=chain[2])
    ax["shaft_sp"].axvline(pk_shaft[0], color="0.4", ls=":", lw=1.0)
    ax["shaft_sp"].set_title(
        f"Akselmoment-spekter  (topp {pk_shaft[0]:.2f} Hz)",
        fontsize=10, loc="left")
    ax["shaft_sp"].set_ylabel("Amplitude [pu]")
    ax["shaft_sp"].set_xlim(1.0, 6)

    ax["tower_sp"].plot(f_tower, a_tower, color=chain[3])
    ax["tower_sp"].axvline(pk_tower[0], color="0.4", ls=":", lw=1.0)
    ax["tower_sp"].set_title(
        f"Tårn-spekter  (topp {pk_tower[0]:.3f} Hz)",
        fontsize=10, loc="left")
    ax["tower_sp"].set_ylabel("Amplitude [m/s², rel.]")
    ax["tower_sp"].set_xlabel("Frekvens [Hz]")
    ax["tower_sp"].set_xlim(0, 1)

    for key in ("shaft_sp", "tower_sp"):
        ax[key].grid(True, alpha=0.3)

    droep = "på" if args.wt_droop else "av"
    fig.suptitle(
        "En LEOGO-nett-hendelse -> elektromekanisk respons i vindturbinen\n"
        f"(redusert modell, relativ amplitude — vind {args.wind:.0f} m/s, frekvensstøtte {droep})",
        fontsize=12, fontweight="bold",
    )

    txt = (f"Slitasje (DEL, post-hendelse):\n"
           f"  drivverk = {fat['shaft_del_pu']:.3e} pu  (m={args.m_shaft:g})\n"
           f"  tårn     = {fat['tower_del']:.3e}     (m={args.m_tower:g})")
    fig.text(0.5, 0.005, txt, ha="center", va="bottom", fontsize=9,
             family="monospace",
             bbox=dict(boxstyle="round", fc="0.95", ec="0.7"))

    fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.11,
                        hspace=0.35, wspace=0.25)
    return fig


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind", type=float, default=10.0,
                        help="Fast vindhastighet [m/s] (10 = Region 2, ren torsjonstopp).")
    parser.add_argument("--t-end", type=float, default=70.0,
                        help="Lang nok for treg (~0.234 Hz) taarn-utringing.")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--onset", type=float, default=10.0,
                        help="Tidspunkt for nett-hendelsen [s].")
    parser.add_argument("--load-step-mw", type=float, default=10.0,
                        help="Braatt lastesteg paa hovednettbussen (GT-trip-lignende).")
    parser.add_argument("--wt-droop", action="store_true",
                        help="Slaa paa WT frekvensstoette-droep for hero-kjoeringen.")
    parser.add_argument("--m-shaft", type=float, default=8.0,
                        help="Woehler-eksponent drivverk (staal).")
    parser.add_argument("--m-tower", type=float, default=4.0,
                        help="Woehler-eksponent taarn (sveiset staal).")
    parser.add_argument("--compare-droop", action="store_true",
                        help="Kjoer OGSAA med droep paa og skriv ut DEL-trade-offen.")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out", type=str,
                        default="results/em_interaction_sweep/em_interaction_hero.png")
    args = parser.parse_args()

    res = run_event_series(
        wind_mps=args.wind,
        onset=args.onset,
        t_end=args.t_end,
        dt=args.dt,
        load_step_mw=args.load_step_mw,
        droop_enable=1 if args.wt_droop else 0,
    )
    fat = event_fatigue(res, args.onset, args.m_shaft, args.m_tower)

    print("=" * 64)
    print(f"LEOGO-hendelse: {args.load_step_mw:.1f} MW steg @ t={args.onset:.0f}s, "
          f"vind {args.wind:.0f} m/s, droep {'paa' if args.wt_droop else 'av'}")
    print(f"  drivverk DEL = {fat['shaft_del_pu']:.4e} pu   "
          f"(peak-to-peak {fat['shaft_ptp_pu']:.4e} pu, m={args.m_shaft:g})")
    print(f"  taarn    DEL = {fat['tower_del']:.4e}      "
          f"(peak-to-peak {fat['tower_ptp']:.4e}, m={args.m_tower:g})")

    if args.compare_droop:
        res_on = run_event_series(
            wind_mps=args.wind,
            onset=args.onset,
            t_end=args.t_end,
            dt=args.dt,
            load_step_mw=args.load_step_mw,
            droop_enable=1,
        )
        fat_on = event_fatigue(res_on, args.onset, args.m_shaft, args.m_tower)
        print("-" * 64)
        print("Frekvensstoette-trade-off (droep av -> paa):")

        def pct(a, b):
            return 100.0 * (b - a) / a if a else float("nan")

        print(f"  taarn    DEL: {fat['tower_del']:.4e} -> {fat_on['tower_del']:.4e}"
              f"  ({pct(fat['tower_del'], fat_on['tower_del']):+.1f} %)")
        print(f"  drivverk DEL: {fat['shaft_del_pu']:.4e} -> {fat_on['shaft_del_pu']:.4e}"
              f"  ({pct(fat['shaft_del_pu'], fat_on['shaft_del_pu']):+.1f} %)")
    print("=" * 64)

    fig = make_hero_figure(res, args.onset, fat, args)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Lagret figur: {out_path}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
