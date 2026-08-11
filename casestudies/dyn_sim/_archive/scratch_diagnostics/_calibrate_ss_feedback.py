r"""
Calibration groundwork for the reciprocal tower-SS <-> drivetrain coupling.

The reduced tower model can now run with an optional two-way coupling between
the generator-rotor speed and the tower side-to-side (SS) modal acceleration,
implemented as a symmetric 2x2 mass matrix

    [ J_e  -c ] [ d(omega_e) ]   [ T_shaft - Te                    ]
    [ -c    1 ] [   dd(q_ss)  ] = [ g_ss*Te - 2 zeta w qd - w^2 q   ]

with J_e = 2*H_e (pu) and a single coupling coefficient c (ss_feedback_c).

This script does the analytical groundwork BEFORE we switch the feedback on:

  1. Computes J_e from the reduced-model parameters and the positive-definite
     stability bound c_max = sqrt(J_e).
  2. Shows the leading-order effect of c on the SS eigenfrequency/damping
     (mass-loading by the rotor inertia) and confirms c=0 leaves the mode
     exactly at f_ss = 0.234 Hz, zeta_ss = 0.0034.
  3. Reports, for a chosen demonstration c, the base f_ss to program so the
     COUPLED resonance still lands on 0.234 Hz.

Leading-order derivation (rotor locally free, Te ~ 0): eliminating d(omega_e)
from the 2x2 system gives, for the SS mode,

    q'' + k*(2 zeta w) q' + k*w^2 q = 0,   k = J_e / (J_e - c^2),

so the coupled mode has w_eff = w*sqrt(k) and zeta_eff = zeta*sqrt(k). The full
electrical/torsional coupling shifts this slightly; the exact c is pinned later
against the OpenFAST FMU. Here we only size c and verify the c=0 limit.

Usage
-----
  .\.venv\Scripts\python.exe casestudies\dyn_sim\_calibrate_ss_feedback.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\_calibrate_ss_feedback.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "results" / "em_interaction_sweep"

# Reduced-model parameters (mirror _wt_row in test_WT_LEOGO_2WT_sim.py).
J_E_KGM2 = 1836784.0            # generator/HSS-referred inertia [kg m^2]
S_N_MVA = 15.0                 # turbine rating [MVA]
OMEGA_M_RATED_RPM = 7.559987120819503   # rated LSS speed [rpm]
F_SS_HZ = 0.234                # target tower SS natural frequency [Hz]
ZETA_SS = 0.0034               # target SS damping ratio [-]

C_DEMO = 0.05                  # demonstration coupling (pending FMU calibration)


def he_and_je():
    """Return (H_e [s], J_e [pu]) from the reduced-model parameters."""
    w_base = OMEGA_M_RATED_RPM * 2.0 * np.pi / 60.0        # rad/s
    h_e = 0.5 * J_E_KGM2 * w_base**2 / (S_N_MVA * 1e6)     # s
    return h_e, 2.0 * h_e


def mass_loading(c: np.ndarray, je: float) -> np.ndarray:
    """Leading-order mass-loading factor k = J_e/(J_e - c^2)."""
    return je / (je - c**2)


def main() -> None:
    p = argparse.ArgumentParser(description="Size the SS<->drivetrain coupling.")
    p.add_argument("--c-demo", type=float, default=C_DEMO,
                   help="Demonstration coupling coefficient to report.")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    h_e, je = he_and_je()
    c_max = float(np.sqrt(je))

    print("Reciprocal tower-SS <-> drivetrain coupling -- calibration groundwork")
    print(f"  H_e                 = {h_e:.6f} s")
    print(f"  J_e = 2*H_e         = {je:.6f} pu")
    print(f"  stability bound c<  = sqrt(J_e) = {c_max:.6f}")
    print(f"  target f_ss / zeta  = {F_SS_HZ:.4f} Hz / {ZETA_SS:.4f}")

    # c=0 limit (must reproduce the uncoupled mode exactly)
    k0 = mass_loading(np.array([0.0]), je)[0]
    print("\n  c = 0 limit (must be unchanged):")
    print(f"    k = {k0:.6f}  ->  f_coupled = {F_SS_HZ * np.sqrt(k0):.6f} Hz, "
          f"zeta = {ZETA_SS * np.sqrt(k0):.6f}  (identical) [OK]")

    # Table across c
    print("\n  Leading-order shift vs c (base f_ss unchanged):")
    print("     c       c/c_max     k        f_coupled[Hz]   df[%]    zeta_eff")
    c_tab = np.array([0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25])
    c_tab = c_tab[c_tab < c_max]
    for c in c_tab:
        k = mass_loading(np.array([c]), je)[0]
        f_c = F_SS_HZ * np.sqrt(k)
        z_c = ZETA_SS * np.sqrt(k)
        print(f"   {c:5.3f}    {c / c_max:6.3f}   {k:7.4f}   {f_c:10.6f}    "
              f"{100 * (f_c / F_SS_HZ - 1):5.2f}   {z_c:.6f}")

    # Demonstration c: base f_ss to program so the COUPLED mode stays on target
    c = float(args.c_demo)
    if c >= c_max:
        raise SystemExit(f"c-demo={c} exceeds stability bound {c_max:.4f}")
    k = mass_loading(np.array([c]), je)[0]
    f_base_retuned = F_SS_HZ * np.sqrt(1.0 - c**2 / je)
    print(f"\n  Demonstration c = {c:g}  (c/c_max = {c / c_max:.3f}):")
    print(f"    if base f_ss left at {F_SS_HZ:.4f} -> coupled f = "
          f"{F_SS_HZ * np.sqrt(k):.5f} Hz ({100 * (np.sqrt(k) - 1):.2f}% high)")
    print(f"    to keep coupled f = {F_SS_HZ:.4f} Hz, program base f_ss = "
          f"{f_base_retuned:.5f} Hz")

    # Figure: coupled frequency and damping vs c, with bound + demo marker
    cc = np.linspace(0.0, 0.98 * c_max, 400)
    kk = mass_loading(cc, je)
    f_cc = F_SS_HZ * np.sqrt(kk)
    z_cc = ZETA_SS * np.sqrt(kk)

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(cc, f_cc, color="#1f5fb0", lw=1.6)
    ax1.axhline(F_SS_HZ, color="#888888", lw=0.9, ls=":")
    ax1.axvline(c_max, color="#c0392b", lw=1.0, ls="--",
                label=f"stabilitetsgrense c$_{{max}}$={c_max:.3f}")
    ax1.axvline(c, color="#2c3e50", lw=1.0, ls="-.",
                label=f"demo c={c:g}")
    ax1.set_xlabel("koblingskoeffisient c")
    ax1.set_ylabel("koblet SS-frekvens  [Hz]")
    ax1.set_title("a) SS-egenfrekvens vs c  (c=0 → 0,234 Hz uendret)",
                  fontsize=10.5)
    ax1.legend(loc="upper left", fontsize=9)

    ax2.plot(cc, z_cc, color="#1f5fb0", lw=1.6)
    ax2.axhline(ZETA_SS, color="#888888", lw=0.9, ls=":")
    ax2.axvline(c_max, color="#c0392b", lw=1.0, ls="--")
    ax2.axvline(c, color="#2c3e50", lw=1.0, ls="-.")
    ax2.set_xlabel("koblingskoeffisient c")
    ax2.set_ylabel("effektiv SS-demping ζ$_{eff}$")
    ax2.set_title("b) SS-demping vs c", fontsize=10.5)

    fig.suptitle(
        "Kalibrering av toveis tårn-SS ↔ drivverk-kobling: leddende orden "
        f"(J$_e$={je:.4f} pu, c$_{{max}}$={c_max:.3f})", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "ss_feedback_calibration.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\n  Lagret figur: {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
