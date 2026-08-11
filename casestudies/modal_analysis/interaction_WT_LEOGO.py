r"""
Wind-turbine-centric small-signal interaction analysis on the coupled
WT + LEOGO model.

Linearises the coupled model built by build_model() in test_WT_LEOGO_sim.py
(the full LEOGO offshore network with the WT + UIC converter embedded at
"Busbar WTG1 LV"), and asks a WT-centric question: which oscillatory modes
involve the wind turbine, and through which path does the rest of the system
couple into them?

For every mode the (sum-normalised) participation is split into three physical
groups:

    LEOGO : platform synchronous gensets + AVR/PSS/GOV controls
    WT    : wind-turbine drivetrain, pitch and speed-filter states
    UIC   : the wind-turbine grid-side voltage-source converter (WT1_LEOGO)

Modes are ranked by their WT-side participation (WT + UIC), so the
turbine-relevant modes come first. Each WT-relevant mode is then categorised:

    WT-internal : drivetrain and converter both participate (WT <-> UIC
                  electromechanical coupling inside the turbine)
    grid-coupled: the LEOGO network also participates in a turbine mode
                  (the platform grid reaches into the WT dynamics)
    WT-isolated : the turbine mode is essentially decoupled from the rest

Headless. Run:
    .\.venv\Scripts\python.exe casestudies\modal_analysis\interaction_WT_LEOGO.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops_openfast.dyn_models as ext_lib

from casestudies.dyn_sim.test_WT_LEOGO_sim import build_model

PLOT_DIR = Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = PLOT_DIR / "WT_LEOGO_interaction_modes.csv"

# A mode is "WT-relevant" when the turbine (WT + UIC) owns at least this much
# of the participation.
WT_SIDE_FLOOR = 0.10
# Minimum participation for a group to count as "meaningfully present" when
# classifying internal / grid coupling of a WT-relevant mode.
COUPLING_FLOOR = 0.10

GROUP_COLORS = {"LEOGO": "#a67c52", "WT": "#5f7d6a", "UIC": "#4a6a85"}


def state_label(desc) -> str:
    """Flat string for a state descriptor (tuple or str)."""
    if isinstance(desc, (tuple, list, np.ndarray)):
        return " ".join(str(d) for d in desc)
    return str(desc)


def classify_state(label: str) -> str:
    """Assign a state to one of the three physical groups."""
    if "WT1_LEOGO" in label:      # grid-side converter component name
        return "UIC"
    if label.startswith("WT1 "):  # WT drivetrain / pitch / speed filters
        return "WT"
    return "LEOGO"                 # gensets + AVR/PSS/GOV controls


def main() -> None:
    model = build_model()
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)
    ps.power_flow()
    ps.init_dyn_sim()

    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    eigs = ps_lin.eigs
    # Participation factors p_ki = |L_ik * R_ki|.
    pfs = np.abs(ps_lin.lev.T * ps_lin.rev)
    state_desc = ps.state_desc
    groups = np.array([classify_state(state_label(d)) for d in state_desc])

    n_by_group = {g: int(np.sum(groups == g)) for g in GROUP_COLORS}
    print("State counts by group:", n_by_group, f"(total {len(state_desc)})")

    rows = []
    for i, lam in enumerate(eigs):
        if lam.imag <= 1e-9:
            continue  # keep only the positive-imag member of each pair
        col = pfs[:, i]
        total = col.sum()
        if total <= 0:
            continue
        frac = {g: float(col[groups == g].sum() / total) for g in GROUP_COLORS}
        wt_side = frac["WT"] + frac["UIC"]

        # WT-centric category.
        if wt_side < WT_SIDE_FLOOR:
            category = "LEOGO-only"
        elif frac["LEOGO"] >= COUPLING_FLOOR:
            category = "grid-coupled"
        elif min(frac["WT"], frac["UIC"]) >= COUPLING_FLOOR:
            category = "WT-internal"
        else:
            category = "WT-isolated"

        col_n = col / col.max()
        top3 = np.argsort(col)[-3:][::-1]
        top3_str = ", ".join(
            f"{state_label(state_desc[j])}({col_n[j]:.2f})" for j in top3
        )
        rows.append({
            "f_hz": lam.imag / (2.0 * np.pi),
            "zeta": -lam.real / abs(lam),
            "real": lam.real,
            "frac_LEOGO": frac["LEOGO"],
            "frac_WT": frac["WT"],
            "frac_UIC": frac["UIC"],
            "wt_side": wt_side,
            "category": category,
            "top_states": top3_str,
        })

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)

    # ---- WT-relevant modes first, ranked by turbine participation ----
    wt_modes = df[df["wt_side"] >= WT_SIDE_FLOOR].sort_values(
        "wt_side", ascending=False)
    print("\n=== Wind-turbine-relevant modes "
          "(ranked by WT-side participation) ===")
    print(f"{'f[Hz]':>8} {'zeta%':>7} {'WTside':>7} "
          f"{'LEOGO':>6} {'WT':>6} {'UIC':>6}  {'category':>12}  top states")
    for _, r in wt_modes.iterrows():
        print(f"{r['f_hz']:8.3f} {100*r['zeta']:7.2f} {r['wt_side']:7.2f} "
              f"{r['frac_LEOGO']:6.2f} {r['frac_WT']:6.2f} {r['frac_UIC']:6.2f} "
              f"  {r['category']:>12}  {r['top_states']}")

    grid_coupled = wt_modes[wt_modes["category"] == "grid-coupled"]
    wt_internal = wt_modes[wt_modes["category"] == "WT-internal"]
    print(f"\nGrid-coupled WT modes (LEOGO reaches into the turbine): "
          f"{len(grid_coupled)}")
    for _, r in grid_coupled.iterrows():
        print(f"  {r['f_hz']:.3f} Hz  zeta={100*r['zeta']:.2f}%  "
              f"(LEOGO {r['frac_LEOGO']:.2f} / WT {r['frac_WT']:.2f} / "
              f"UIC {r['frac_UIC']:.2f})")
    print(f"WT-internal modes (drivetrain <-> converter): {len(wt_internal)}")
    for _, r in wt_internal.iterrows():
        print(f"  {r['f_hz']:.3f} Hz  zeta={100*r['zeta']:.2f}%  "
              f"(WT {r['frac_WT']:.2f} / UIC {r['frac_UIC']:.2f})")
    if len(grid_coupled) == 0 and len(wt_internal) == 0:
        print("  -> Every turbine mode is WT-isolated: at this operating point "
              "the LEOGO grid does not measurably couple into the WT dynamics.")

    # Least-damped WT-relevant modes -- resonance risks inside the turbine.
    print("\nLeast-damped WT-relevant modes:")
    for _, r in wt_modes.sort_values("zeta").head(4).iterrows():
        print(f"  {r['f_hz']:7.3f} Hz  zeta={100*r['zeta']:5.2f}%  "
              f"category={r['category']}")

    _plot_splane(df.sort_values("f_hz").reset_index(drop=True))
    print(f"\nsaved {CSV_OUT}")


def _plot_splane(df: pd.DataFrame) -> None:
    """s-plane scatter: WT-relevant modes highlighted, grid-coupled ones ringed."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    leogo_only = df[df["category"] == "LEOGO-only"]
    ax.scatter(leogo_only["real"], 2 * np.pi * leogo_only["f_hz"], s=34,
               c="0.75", edgecolors="0.5", linewidths=0.5,
               label="LEOGO-only", zorder=2)
    for cat, c in (("WT-isolated", "#5f7d6a"), ("WT-internal", "#4a6a85"),
                   ("grid-coupled", "#a67c52")):
        sub = df[df["category"] == cat]
        ax.scatter(sub["real"], 2 * np.pi * sub["f_hz"], s=52, c=c,
                   edgecolors="0.2", linewidths=0.6, label=cat, zorder=3)
    grid = df[df["category"] == "grid-coupled"]
    ax.scatter(grid["real"], 2 * np.pi * grid["f_hz"], s=160,
               facecolors="none", edgecolors="crimson", linewidths=1.4,
               zorder=4)
    ax.axvline(0.0, color="0.4", linewidth=0.8)
    ax.set_xlabel(r"Real part $\sigma$ [1/s]")
    ax.set_ylabel(r"Imag part $\omega$ [rad/s]")
    ax.set_title("Coupled WT+LEOGO eigenvalues (wind-turbine view)")
    ax.grid(True, color="0.9", linewidth=0.5)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8)
    out = PLOT_DIR / "WT_LEOGO_interaction_splane.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
