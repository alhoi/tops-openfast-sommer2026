r"""
Droop-gain aggressiveness sweep for the LEOGO slug-flow -> tower side-to-side
(SS) resonance case (companion to test_WT_LEOGO_slugflow_ss_sim.py).

Overlays a droop-OFF baseline against several increasingly aggressive
frequency-support droop gains K_droop [pu/Hz] (same slug-flow forcing on
resonance, same de-loaded headroom so the operating point is shared). The
figure shows the fundamental trade-off:

  * a more aggressive droop shrinks the grid-frequency swing further
    (better frequency support), but
  * it does so by modulating the WT power harder at the disturbance frequency,
    which AMPLIFIES the tower SS electromechanical oscillation.

Panels: slug load; grid-frequency deviation overlay; SS acceleration overlay;
and a trade-off summary (frequency-swing reduction % and SS amplification x
versus K_droop).

Usage
-----
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_slugflow_droop_sweep.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_slugflow_droop_sweep.py --show
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
BASE_START_S = 40.0       # start of the settled pre-step baseline window
HOLD_START_S = 150.0      # slug switches to a constant +hold_mw step here
HOLD_WIN_S = 20.0         # tail window used for the steady hold-phase metrics

C_AV = "#333333"          # droop OFF (black)
C_LOAD = "#2c3e50"        # slug load (dark)
LS_ON = (0, (5, 2))       # droop ON dashed

# (gain [pu/Hz], csv filename, colour) for the sweep, low -> high gain.
GAINS = [
    (0.75, "slugflow_ss_sweep_k0p75.csv", "#1f5fb0"),   # blue
    (3.0,  "slugflow_ss_sweep_k3p0.csv",  "#8e44ad"),   # purple
    (6.0,  "slugflow_ss_sweep_k6p0.csv",  "#c0392b"),   # red
]


def _post_onset_mask(t: np.ndarray) -> np.ndarray:
    return t >= ONSET_S


def _steady_amp(y: np.ndarray) -> float:
    """Half peak-to-peak over the settled tail (last 40 % of the record)."""
    tail = y[int(0.6 * len(y)):]
    return 0.5 * float(np.ptp(tail))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Droop-gain aggressiveness sweep for the slug-flow SS case.")
    parser.add_argument("--av-csv", type=str,
                        default=str(CSV_DIR / "slugflow_ss_sweep_av.csv"))
    parser.add_argument("--out", type=str,
                        default=str(PROJECT_ROOT / "results" / "em_interaction_sweep"
                                    / "slugflow_ss_droop_sweep.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    av = pd.read_csv(args.av_csv)
    t = av["t"].to_numpy()
    t_end = float(t[-1])
    # Periodic phase [onset, hold_start): the slug pulsates and drives the SS
    # mode.  Constant-hold phase [hold_start, t_end]: the slug settles to a
    # steady +step and the frequency droop shows its classic support benefit.
    # Frequency is referenced per run to the settled pre-step operating point
    # (mean over the zero-mean-load window [BASE_START_S, hold_start)) so the
    # start-up/de-load offset is removed and the slug-induced deviation is
    # isolated: the periodic phase then straddles 0 and the +hold step shows a
    # physical (negative) frequency dip that droop lifts back toward nominal.
    mask_base = (t >= BASE_START_S) & (t < HOLD_START_S)
    mask_per = (t >= ONSET_S) & (t < HOLD_START_S)
    mask_hold = t >= (t_end - HOLD_WIN_S)

    amp_mw = float(np.max(np.abs(av["load_mw"].to_numpy())))

    freq_av = av["grid_freq_hz"].to_numpy()
    base_av = float(np.mean(freq_av[mask_base]))
    df_av = (freq_av - base_av) * 1e3
    ss_av = av["ss_accel_mps2"].to_numpy()
    amp_df_av = _steady_amp(df_av[mask_per])
    amp_ss_av = _steady_amp(ss_av[mask_per])
    dip_av = float(np.mean(df_av[mask_hold]))     # hold-phase step [mHz] (< 0)

    # Load every gain run present on disk.
    runs = []  # (gain, colour, t, df_hz, ss, amp_df, amp_ss, dip)
    for gain, fname, colour in GAINS:
        path = CSV_DIR / fname
        if not path.exists():
            print(f"  (skipping missing {fname})")
            continue
        d = pd.read_csv(path)
        freq = d["grid_freq_hz"].to_numpy()
        base = float(np.mean(freq[mask_base]))
        df_hz = (freq - base) * 1e3
        ss = d["ss_accel_mps2"].to_numpy()
        runs.append((gain, colour, d["t"].to_numpy(), df_hz, ss,
                     _steady_amp(df_hz[mask_per]), _steady_amp(ss[mask_per]),
                     float(np.mean(df_hz[mask_hold]))))

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    fig = plt.figure(figsize=(13.5, 11.0))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[0.5, 1.1, 1.15, 1.1],
                  hspace=0.5)

    # --- (row 0) slug load -----------------------------------------------------
    ax_load = fig.add_subplot(gs[0])
    ax_load.plot(t, av["load_mw"], color=C_LOAD, lw=1.1)
    ax_load.set_ylabel("Slug-last\n[MW]")
    ax_load.set_title(
        f"LEOGO slug-flow effektpulsering på PCC:  {amp_mw:.1f} MW,  "
        f"på-resonans {F_SS_HZ:.3f} Hz", fontsize=11)
    ax_load.set_xlim(0, t_end)
    ax_load.axvline(ONSET_S, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_load.axvline(HOLD_START_S, color="k", ls="--", lw=0.9, alpha=0.6)
    ax_load.text((HOLD_START_S + t_end) * 0.5, 0.62, "konstant +hold",
                 transform=ax_load.get_xaxis_transform(), ha="center",
                 va="center", fontsize=8, color="#555555")

    # --- (row 1) grid-frequency deviation overlay ------------------------------
    ax_f = fig.add_subplot(gs[1])
    ax_f.plot(t, df_av, color=C_AV, lw=1.2,
              label=f"droop AV  (sving {amp_df_av:.1f} | dypp {dip_av:.1f} mHz)")
    for gain, colour, tg, df_hz, _ss, amp_df, _amp_ss, dip in runs:
        ax_f.plot(tg, df_hz, color=colour, lw=1.3, ls=LS_ON,
                  label=f"K={gain:g} pu/Hz  (sving {amp_df:.1f} | dypp {dip:.1f} mHz)")
    ax_f.set_ylabel("Nettfrekvens-avvik\nfra driftspunkt [mHz]")
    ax_f.set_xlim(0, t_end)
    ax_f.axhline(0.0, color="k", lw=0.6, alpha=0.5)
    ax_f.axvline(ONSET_S, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_f.axvline(HOLD_START_S, color="k", ls="--", lw=0.9, alpha=0.6)
    ax_f.legend(loc="lower left", framealpha=0.9, fontsize=8, ncol=2)
    ax_f.set_title("Frekvensstøtte: periodisk fase (0–150 s) og konstant hold (150–200 s)",
                   fontsize=11)

    # --- (row 2) SS acceleration overlay ---------------------------------------
    ax_ss = fig.add_subplot(gs[2])
    ax_ss.plot(t, ss_av, color=C_AV, lw=1.2,
               label=f"droop AV  ({amp_ss_av:.2e} m/s²)")
    for gain, colour, tg, _df, ss, _amp_df, amp_ss, _dip in runs:
        ax_ss.plot(tg, ss, color=colour, lw=1.3, ls=LS_ON,
                   label=f"K={gain:g} pu/Hz  ({amp_ss:.2e} m/s²)")
    ax_ss.set_ylabel("Tårn side-til-side\nakselerasjon [m/s²]")
    ax_ss.set_xlim(0, t_end)
    ax_ss.axvline(ONSET_S, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_ss.axvline(HOLD_START_S, color="k", ls="--", lw=0.9, alpha=0.6)
    ax_ss.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=2)
    ax_ss.set_title("Elektromekanisk kobling: samme aggressive droop forsterker "
                    "SS-oscillasjonen", fontsize=11)

    # --- (row 3) trade-off summary --------------------------------------------
    ax_t = fig.add_subplot(gs[3])
    ks = np.array([0.0] + [g for g, *_ in runs])
    swing_red = np.array([0.0] + [
        100.0 * (1.0 - r[5] / max(amp_df_av, 1e-12)) for r in runs])
    dip_red = np.array([0.0] + [
        100.0 * (1.0 - abs(r[7]) / max(abs(dip_av), 1e-12)) for r in runs])
    amplifs = np.array([1.0] + [
        r[6] / max(amp_ss_av, 1e-12) for r in runs])

    C_RED = "#1f5fb0"
    C_DIP = "#16a085"
    C_AMP = "#c0392b"
    ax_t.plot(ks, swing_red, "o-", color=C_RED, lw=1.6, ms=7,
              label="periodisk svingreduksjon")
    ax_t.plot(ks, dip_red, "^-", color=C_DIP, lw=1.6, ms=7,
              label="konstant-fase dyppreduksjon")
    ax_t.set_xlabel("Droop-gain  K$_{droop}$  [pu/Hz]")
    ax_t.set_ylabel("Frekvens-\nreduksjon [%]", color=C_RED)
    ax_t.tick_params(axis="y", labelcolor=C_RED)
    ax_t.set_ylim(bottom=0)
    ax_t.legend(loc="lower right", fontsize=8, framealpha=0.9)

    ax_t2 = ax_t.twinx()
    ax_t2.plot(ks, amplifs, "s--", color=C_AMP, lw=1.6, ms=7,
               label="SS-forsterkning")
    ax_t2.set_ylabel("SS-amplitude\nforsterkning [×]", color=C_AMP)
    ax_t2.tick_params(axis="y", labelcolor=C_AMP)
    ax_t2.axhline(1.0, color=C_AMP, ls=":", lw=0.8, alpha=0.6)
    ax_t.set_title("Avveining: mer frekvensstøtte  ↔  mer tårnbelastning",
                   fontsize=11)
    ax_t.grid(True, alpha=0.3)

    fig.suptitle(
        "Slug-flow -> tårn SS-resonans:  tuning av frekvensstøtte-droopens "
        "aggressivitet",
        fontsize=13, y=0.995)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Lagret figur: {out}")
    print(f"  droop AV:  sving {amp_df_av:.2f} mHz,  SS {amp_ss_av:.3e} m/s^2,  "
          f"hold-dypp {dip_av:.2f} mHz")
    for gain, _c, _tg, _df, _ss, amp_df, amp_ss, dip in runs:
        red = 100.0 * (1.0 - amp_df / max(amp_df_av, 1e-12))
        amp = amp_ss / max(amp_ss_av, 1e-12)
        dred = 100.0 * (1.0 - abs(dip) / max(abs(dip_av), 1e-12))
        print(f"  K={gain:>4g} pu/Hz:  sving {amp_df:5.2f} mHz (-{red:4.1f}%),  "
              f"SS {amp_ss:.3e} m/s^2 ({amp:.2f}x),  hold-dypp {dip:6.2f} mHz (-{dred:4.1f}%)")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
