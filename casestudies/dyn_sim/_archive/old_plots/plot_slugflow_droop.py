r"""
Droop ON/OFF comparison for the LEOGO slug-flow -> tower side-to-side (SS)
resonance case (companion to test_WT_LEOGO_slugflow_ss_sim.py).

Reads two runs of the SAME slug-flow process-load pulsation on resonance
(0.234 Hz, +/-2 MW at the PCC): one with the WT frequency-support droop OFF and
one with it ON. Because the headroom de-loads the turbine in BOTH runs, the two
share the same operating point and the difference isolates the droop *response*.

The figure answers three questions:
  1. Does the droop reduce the grid-frequency swing? (frequency support)
  2. How do the WT terminal voltage / power / current change?
  3. Does the droop DAMP or AMPLIFY the tower SS electromechanical oscillation
     when the disturbance is periodic?

Usage
-----
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_slugflow_droop.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_slugflow_droop.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "csv_files"

F_SS_HZ = 0.234           # tower side-to-side eigenfrequency
ONSET_S = 10.0            # slug pulsation onset used in the runs

C_OFF = "#1f5fb0"         # droop OFF (blue)
C_ON = "#c0392b"          # droop ON (red)
LS_ON = (0, (5, 2))       # droop ON dashed
C_LOAD = "#2c3e50"        # slug load (dark)


def _post_onset_mask(t: np.ndarray) -> np.ndarray:
    return t >= ONSET_S


def _steady_amp(y: np.ndarray) -> float:
    """Half peak-to-peak over the settled tail (last 40 % of the record)."""
    tail = y[int(0.6 * len(y)):]
    return 0.5 * float(np.ptp(tail))


def _steady_mean(y: np.ndarray) -> float:
    tail = y[int(0.6 * len(y)):]
    return float(np.mean(tail))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Droop ON/OFF comparison for the slug-flow tower SS case.")
    parser.add_argument("--nodroop-csv", type=str,
                        default=str(CSV_DIR / "slugflow_ss_0p234_2MW_nodroop.csv"))
    parser.add_argument("--droop-csv", type=str,
                        default=str(CSV_DIR / "slugflow_ss_0p234_2MW_droop.csv"))
    parser.add_argument("--out", type=str,
                        default=str(PROJECT_ROOT / "results" / "em_interaction_sweep"
                                    / "slugflow_ss_droop.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    off = pd.read_csv(args.nodroop_csv)
    on = pd.read_csv(args.droop_csv)

    t = off["t"].to_numpy()
    t_on = on["t"].to_numpy()
    t_end = float(t[-1])
    m = _post_onset_mask(t)

    f_nom = float(off["grid_freq_hz"].iloc[0])
    amp_mw = float(np.max(np.abs(off["load_mw"].to_numpy())))

    # --- frequency deviation [mHz] ---------------------------------------------
    df_off = (off["grid_freq_hz"].to_numpy() - f_nom) * 1e3
    df_on = (on["grid_freq_hz"].to_numpy() - f_nom) * 1e3
    amp_df_off = _steady_amp(df_off[m])
    amp_df_on = _steady_amp(df_on[m])
    df_reduction = 100.0 * (1.0 - amp_df_on / max(amp_df_off, 1e-12))

    # --- tower SS acceleration -------------------------------------------------
    ss_off = off["ss_accel_mps2"].to_numpy()
    ss_on = on["ss_accel_mps2"].to_numpy()
    amp_ss_off = _steady_amp(ss_off[m])
    amp_ss_on = _steady_amp(ss_on[m])
    ss_ratio = amp_ss_on / max(amp_ss_off, 1e-12)
    ss_verdict = "forsterker" if ss_ratio > 1.0 else "demper"

    # --- terminal quantities ---------------------------------------------------
    p_off = off["P_term_mw"].to_numpy()
    p_on = on["P_term_mw"].to_numpy()
    v_off = off["V_term_pu"].to_numpy()
    v_on = on["V_term_pu"].to_numpy()
    i_off = off["I_term_pu"].to_numpy()
    i_on = on["I_term_pu"].to_numpy()

    amp_p_off, amp_p_on = _steady_amp(p_off[m]), _steady_amp(p_on[m])
    amp_v_off, amp_v_on = _steady_amp(v_off[m]), _steady_amp(v_on[m])
    amp_i_off, amp_i_on = _steady_amp(i_off[m]), _steady_amp(i_on[m])

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    fig = plt.figure(figsize=(13.5, 11.0))
    gs = GridSpec(4, 3, figure=fig,
                  height_ratios=[0.55, 1.0, 1.15, 1.0],
                  hspace=0.48, wspace=0.28)

    # --- (row 0) shared slug-flow power pulsation ------------------------------
    ax_load = fig.add_subplot(gs[0, :])
    ax_load.plot(t, off["load_mw"], color=C_LOAD, lw=1.1)
    ax_load.set_ylabel("Slug-last\n[MW]")
    ax_load.set_title(
        f"LEOGO slug-flow effektpulsering på PCC (Main Bus A):  "
        f"{amp_mw:.1f} MW,  på-resonans {F_SS_HZ:.3f} Hz", fontsize=11)
    ax_load.set_xlim(0, t_end)
    ax_load.axvline(ONSET_S, color="k", ls=":", lw=0.8, alpha=0.6)

    # --- (row 1) grid-frequency deviation: droop support -----------------------
    ax_f = fig.add_subplot(gs[1, :])
    ax_f.plot(t, df_off, color=C_OFF, lw=1.1,
              label=f"droop AV   (svingamp {amp_df_off:.2f} mHz)")
    ax_f.plot(t_on, df_on, color=C_ON, lw=1.4, ls=LS_ON,
              label=f"droop PÅ   (svingamp {amp_df_on:.2f} mHz)")
    ax_f.set_ylabel("Nettfrekvens-\navvik [mHz]")
    ax_f.set_xlim(0, t_end)
    ax_f.axhline(0.0, color="k", lw=0.6, alpha=0.5)
    ax_f.axvline(ONSET_S, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_f.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax_f.set_title(
        f"Frekvensstøtte: droop reduserer nettfrekvens-svinget med "
        f"{df_reduction:.0f} %", fontsize=11)

    # --- (row 2) tower SS acceleration: damp or amplify ------------------------
    ax_ss = fig.add_subplot(gs[2, :])
    ax_ss.plot(t, ss_off, color=C_OFF, lw=1.1,
               label=f"droop AV   (amp {amp_ss_off:.2e} m/s²)")
    ax_ss.plot(t_on, ss_on, color=C_ON, lw=1.4, ls=LS_ON,
               label=f"droop PÅ   (amp {amp_ss_on:.2e} m/s²)")
    ax_ss.set_ylabel("Tårn side-til-side\nakselerasjon [m/s²]")
    ax_ss.set_xlim(0, t_end)
    ax_ss.axvline(ONSET_S, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_ss.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax_ss.set_title(
        f"Elektromekanisk kobling: droop {ss_verdict} SS-oscillasjonen  "
        f"({ss_ratio:.2f}× av droop-av-amplituden)", fontsize=11)

    # --- (row 3) terminal quantities: P, V, I ----------------------------------
    ax_p = fig.add_subplot(gs[3, 0])
    ax_p.plot(t, p_off, color=C_OFF, lw=1.0)
    ax_p.plot(t_on, p_on, color=C_ON, lw=1.3, ls=LS_ON)
    ax_p.set_xlim(0, t_end)
    ax_p.set_xlabel("Tid [s]")
    ax_p.set_ylabel("Terminal aktiv-\neffekt P [MW]")
    ax_p.set_title(f"P-svai:  AV {amp_p_off:.2f} → PÅ {amp_p_on:.2f} MW",
                   fontsize=10)

    ax_v = fig.add_subplot(gs[3, 1])
    ax_v.plot(t, v_off, color=C_OFF, lw=1.0, label="droop AV")
    ax_v.plot(t_on, v_on, color=C_ON, lw=1.3, ls=LS_ON, label="droop PÅ")
    ax_v.set_xlim(0, t_end)
    ax_v.set_xlabel("Tid [s]")
    ax_v.set_ylabel("Terminal-\nspenning V [pu]")
    ax_v.set_title(f"V-svai:  AV {amp_v_off:.1e} → PÅ {amp_v_on:.1e} pu",
                   fontsize=10)
    ax_v.legend(loc="upper right", fontsize=8, framealpha=0.9)

    ax_i = fig.add_subplot(gs[3, 2])
    ax_i.plot(t, i_off, color=C_OFF, lw=1.0)
    ax_i.plot(t_on, i_on, color=C_ON, lw=1.3, ls=LS_ON)
    ax_i.set_xlim(0, t_end)
    ax_i.set_xlabel("Tid [s]")
    ax_i.set_ylabel("Terminal-\nstrøm I [pu]")
    ax_i.set_title(f"I-svai:  AV {amp_i_off:.1e} → PÅ {amp_i_on:.1e} pu",
                   fontsize=10)

    fig.suptitle(
        "Slug-flow -> tårn SS-resonans:  frekvensstøtte-droop PÅ vs AV",
        fontsize=13, y=0.995)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Lagret figur: {out}")
    print(f"  f_nom                   = {f_nom:.4f} Hz")
    print(f"  Df svingamp  AV / PA    = {amp_df_off:.3f} / {amp_df_on:.3f} mHz "
          f"(reduksjon {df_reduction:.1f} %)")
    print(f"  SS accel amp AV / PA    = {amp_ss_off:.3e} / {amp_ss_on:.3e} m/s^2 "
          f"(forhold {ss_ratio:.3f}x -> droop {ss_verdict})")
    print(f"  P term amp   AV / PA    = {amp_p_off:.3f} / {amp_p_on:.3f} MW")
    print(f"  V term amp   AV / PA    = {amp_v_off:.3e} / {amp_v_on:.3e} pu")
    print(f"  I term amp   AV / PA    = {amp_i_off:.3e} / {amp_i_on:.3e} pu")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
