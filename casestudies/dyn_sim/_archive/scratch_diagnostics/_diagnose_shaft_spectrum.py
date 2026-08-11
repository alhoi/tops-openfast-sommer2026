r"""
Diagnose av akselmoment-spekteret: hva er toppene, og endrer frekvensstoette dem?

Kjoerer den samme LEOGO-hendelsen som hero-figuren (permanent 10 MW steg) med
frekvensstoette AV og PAA, og legger akselmoment-spektrene oppaa hverandre paa
log-y saa BAADE den lavfrekvente nett-toppen (~1.17 Hz) og drivverk-torsjonen
(3.49 Hz) er synlige samtidig. De tre relevante egenmodene fra lineariseringen
markeres:

  1.168 Hz  LEOGO-gensettenes eksitasjonssystem (AVR), zeta~23% -> lager topp
  1.712 Hz  UIC intern spenning <-> gensett EMF, zeta~96%     -> ingen topp
  3.490 Hz  drivverk-torsjon, zeta~4.4%                        -> mekanisk resonans

Slik ser vi svart paa hvitt hva 1.2 Hz-toppen er, og om droep endrer den.

Kjoer:
  .\.venv\Scripts\python.exe casestudies\dyn_sim\_diagnose_shaft_spectrum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from casestudies.dyn_sim.plot_em_interaction_hero import (
    run_event_series,
    post_onset_spectrum,
    peak_in_band,
)

# Eigenmodes from casestudies/modal_analysis/interaction_WT_LEOGO.py.
MODES = [
    (1.168, "LEOGO AVR (zeta~23%)", "#c0392b"),
    (1.712, "UIC<->genset (zeta~96%)", "#7f8c8d"),
    (3.490, "drivverk-torsjon (zeta~4.4%)", "#2e7d32"),
]

WIND = 10.0
ONSET = 10.0
T_END = 70.0
DT = 0.01
LOAD_MW = 10.0


def main() -> None:
    import matplotlib.pyplot as plt

    cases = {}
    for droop in (0, 1):
        res = run_event_series(
            wind_mps=WIND, onset=ONSET, t_end=T_END, dt=DT,
            load_step_mw=LOAD_MW, droop_enable=droop,
        )
        # Mean-removed only (NO moving-average detrend) so the low-frequency
        # grid-mode content is preserved and visible next to the torsion peak.
        f, a = post_onset_spectrum(
            res["t"], res["shaft_torque_pu"], ONSET, f_hi=6.0)
        cases[droop] = (f, a)

    print("=" * 60)
    print("Akselmoment-spekter: topp-amplitude i hvert baand")
    print(f"{'baand [Hz]':>16} {'droop AV':>12} {'droop PAA':>12}")
    for lo, hi, name in [(1.05, 1.30, "AVR ~1.17"),
                         (3.30, 3.70, "torsjon 3.49")]:
        off = peak_in_band(*cases[0], lo, hi)
        on = peak_in_band(*cases[1], lo, hi)
        print(f"{name:>16} {off[1]:12.3e} {on[1]:12.3e}"
              f"   (f_off={off[0]:.2f}, f_on={on[0]:.2f})")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(cases[0][0], cases[0][1], color="#1f4e79",
                label="frekvensstoette AV")
    ax.semilogy(cases[1][0], cases[1][1], color="#c55a11",
                label="frekvensstoette PAA")
    for f_m, name, c in MODES:
        ax.axvline(f_m, color=c, ls="--", lw=1.2)
        ax.text(f_m, ax.get_ylim()[1], f" {f_m:.2f} Hz\n {name}",
                rotation=90, va="top", ha="left", fontsize=8, color=c)
    ax.set_xlim(0.3, 6.0)
    ax.set_xlabel("Frekvens [Hz]")
    ax.set_ylabel("Akselmoment-amplitude [pu]")
    ax.set_title("Akselmoment-spekter etter LEOGO-steg: hva er toppene?")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = PROJECT_ROOT / "results/em_interaction_sweep/shaft_spectrum_diag.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Lagret figur: {out}")


if __name__ == "__main__":
    main()
