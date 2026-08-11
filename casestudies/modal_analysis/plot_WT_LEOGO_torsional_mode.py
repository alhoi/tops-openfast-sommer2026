r"""
Visualize the drivetrain torsional mode (~3.49 Hz) of the simplified LEOGO
wind-turbine model (build_model() from test_WT_LEOGO_sim.py).

Produces four thesis-styled figures (saved as PNG, no GUI):
  1. Full s-plane eigenvalue spectrum with the torsional mode highlighted.
  2. Zoom of the s-plane around the torsional eigenvalue.
  3. Polar mode-shape: rotor mass (omega_m) vs generator mass (omega_e),
     showing the two inertias swinging in anti-phase (torsional signature).
  4. Participation-factor bar chart for the torsional mode.

Run:
    .\.venv\Scripts\python.exe casestudies\modal_analysis\plot_WT_LEOGO_torsional_mode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops_openfast.dyn_models as ext_lib
import tops_openfast.plotting as dps_plt

from casestudies.dyn_sim.test_WT_LEOGO_sim import build_model

DRIVETRAIN_STATE_KEYS = ("omega_m", "omega_e", "theta_m", "theta_e")
OUT_DIR = PROJECT_ROOT / "casestudies" / "modal_analysis" / "plots"


def state_label(desc) -> str:
    if isinstance(desc, (tuple, list, np.ndarray)):
        return " ".join(str(d) for d in desc)
    return str(desc)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)
    ps.power_flow()
    ps.init_dyn_sim()

    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    eigs = ps_lin.eigs
    rev = ps_lin.rev
    pfs = ps_lin.lev.T * ps_lin.rev
    pfs_abs = np.abs(pfs) / np.max(np.abs(pfs), axis=0)
    state_desc = ps.state_desc

    # Locate the torsional mode: oscillatory mode with highest drivetrain
    # participation (matches participation_WT_LEOGO_torsional.py).
    dt_idx = [
        i for i, d in enumerate(state_desc)
        if any(k in state_label(d) for k in DRIVETRAIN_STATE_KEYS)
    ]
    best_i, best_pf = None, -1.0
    for i, lam in enumerate(eigs):
        if lam.imag <= 1e-9:
            continue
        pf = float(np.max(pfs_abs[dt_idx, i]))
        if pf > best_pf:
            best_pf, best_i = pf, i

    lam = eigs[best_i]
    f_hz = lam.imag / (2.0 * np.pi)
    zeta = -lam.real / abs(lam)
    print(f"Torsional mode: idx={best_i}  f={f_hz:.4f} Hz  zeta={100*zeta:.2f}%  "
          f"lambda={lam.real:.4f}{lam.imag:+.4f}j")

    # --- Figure 1: full s-plane, torsional mode highlighted -----------------
    fig1, ax1 = dps_plt.plot_eigs_thesis(eigs, annotate=False, print_ready=True)
    for s in (+1, -1):
        ax1.scatter(
            lam.real, s * abs(lam.imag),
            s=340, facecolors="none", edgecolors="#c0392b",
            linewidths=2.2, zorder=5,
        )
    ax1.annotate(
        f"Torsional drivetrain mode\n{f_hz:.2f} Hz,  \u03b6 = {100*zeta:.2f}%",
        xy=(lam.real, abs(lam.imag)),
        xytext=(lam.real + 0.30 * (abs(eigs.real.min())), abs(lam.imag) + 0.14 * np.max(np.abs(eigs.imag))),
        fontsize=11, color="#c0392b", weight="bold",
        arrowprops=dict(arrowstyle="->", color="#c0392b", linewidth=1.6),
        zorder=6,
    )
    ax1.set_title("LEOGO simplified WT model \u2014 eigenvalue spectrum", fontsize=13)
    f1 = OUT_DIR / "WT_LEOGO_eigs_full.png"
    fig1.savefig(f1, bbox_inches="tight", dpi=300)
    plt.close(fig1)

    # --- Figure 2: zoom around the torsional eigenvalue ---------------------
    fig2, ax2 = plt.subplots(figsize=(6.5, 5.2), facecolor="white")
    ax2.scatter(eigs.real, eigs.imag, s=90, c="#95a5a6",
                edgecolors="#5d6d7e", linewidths=1.0, zorder=3)
    for s in (+1, -1):
        ax2.scatter(lam.real, s * abs(lam.imag), s=200, c="#c0392b",
                    edgecolors="k", linewidths=1.2, zorder=5)
    ax2.axvline(0, color="0.4", lw=0.75, ls="--")
    ax2.axhline(0, color="0.4", lw=0.75, ls="--")
    ax2.grid(True, color="0.85", lw=0.55)
    ax2.set_axisbelow(True)
    w = abs(lam.imag)
    ax2.set_xlim(min(lam.real * 2.2, -1.0), 0.5)
    ax2.set_ylim(-1.6 * w, 1.6 * w)
    ax2.set_xlabel("Real part (1/s)", fontsize=12)
    ax2.set_ylabel("Imaginary part (rad/s)", fontsize=12)
    ax2.set_title(f"Zoom on torsional mode  ({f_hz:.2f} Hz, \u03b6={100*zeta:.2f}%)",
                  fontsize=12)
    ax2.annotate(f"{f_hz:.2f} Hz", xy=(lam.real, w),
                 xytext=(lam.real + 0.15, w * 1.05), fontsize=11,
                 color="#c0392b", weight="bold")
    fig2.tight_layout()
    f2 = OUT_DIR / "WT_LEOGO_eigs_zoom_torsional.png"
    fig2.savefig(f2, bbox_inches="tight", dpi=300)
    plt.close(fig2)

    # --- Figure 3: polar mode shape (rotor vs generator inertia) ------------
    omega_m_idx = ps.windturbine["WindTurbine"].state_idx_global["omega_m"][0]
    omega_e_idx = ps.windturbine["WindTurbine"].state_idx_global["omega_e"][0]
    c_m = rev[omega_m_idx, best_i]
    c_e = rev[omega_e_idx, best_i]
    # Normalize so the dominant component has unit length and zero phase.
    ref = c_e if abs(c_e) >= abs(c_m) else c_m
    c_m_n = c_m / ref
    c_e_n = c_e / ref

    fig3 = plt.figure(figsize=(6.4, 6.4), facecolor="white")
    ax3 = plt.subplot(111, projection="polar")
    ax3.set_rlim(0, 1.15)
    ax3.grid(color=[0.85, 0.85, 0.85])
    # Draw both phasors at UNIT length (true angles) so the 180 deg anti-phase
    # is clearly visible; the true magnitudes are given in the legend.
    ang_e = np.angle(c_e_n)
    ang_m = np.angle(c_m_n)
    ax3.annotate("", xy=(ang_e, 1.0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#2166ac", linewidth=3.0))
    ax3.annotate("", xy=(ang_m, 1.0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#c0392b", linewidth=3.0))
    ax3.plot([], [], color="#2166ac", lw=3,
             label=fr"generator $\omega_e$   (|rev| = {abs(c_e):.3f})")
    ax3.plot([], [], color="#c0392b", lw=3,
             label=fr"rotor $\omega_m$   (|rev| = {abs(c_m):.4f})")
    ax3.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=10,
               frameon=False, ncol=1)
    ax3.set_title(
        f"Torsional mode shape ({f_hz:.2f} Hz) \u2014 arrows drawn unit-length\n"
        f"anti-phase (180\u00b0); true ratio "
        fr"$\omega_m/\omega_e$ = {abs(c_m)/abs(c_e):.4f} $\approx J_e/J_m$ "
        "(generator-dominated)",
        fontsize=11, pad=18,
    )
    f3 = OUT_DIR / "WT_LEOGO_modeshape_torsional.png"
    fig3.savefig(f3, bbox_inches="tight", dpi=300)
    plt.close(fig3)

    # --- Figure 4: participation-factor bar chart ---------------------------
    pf_col = pfs_abs[:, best_i]
    order = np.argsort(pf_col)[::-1]
    top = [j for j in order if pf_col[j] > 0.02][:12]
    labels = [state_label(state_desc[j]) for j in top]
    values = [pf_col[j] for j in top]
    is_dt = [any(k in labels[n] for k in DRIVETRAIN_STATE_KEYS)
             for n in range(len(top))]
    colors = ["#c0392b" if d else "#7f8c8d" for d in is_dt]

    fig4, ax4 = plt.subplots(figsize=(8.0, 5.0), facecolor="white")
    ypos = np.arange(len(top))[::-1]
    ax4.barh(ypos, values, color=colors, edgecolor="k", linewidth=0.6)
    ax4.set_yticks(ypos)
    ax4.set_yticklabels(labels, fontsize=9)
    ax4.set_xlabel("Normalized participation factor", fontsize=12)
    ax4.set_xlim(0, 1.05)
    ax4.grid(True, axis="x", color="0.85", lw=0.55)
    ax4.set_axisbelow(True)
    ax4.set_title(
        f"Participation factors \u2014 torsional mode ({f_hz:.2f} Hz)\n"
        "red = drivetrain states (generator-dominated)",
        fontsize=12,
    )
    fig4.tight_layout()
    f4 = OUT_DIR / "WT_LEOGO_participation_torsional.png"
    fig4.savefig(f4, bbox_inches="tight", dpi=300)
    plt.close(fig4)

    print("Saved:")
    for f in (f1, f2, f3, f4):
        print(f"  {f}")


if __name__ == "__main__":
    main()
