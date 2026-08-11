r"""
Poster figure for the slug-flow / severe-slugging tower side-to-side (SS)
resonance case (companion to test_WT_LEOGO_slugflow_ss_sim.py).

Reads two runs -- an on-resonance run (slug period ~ tower SS eigenperiod,
0.234 Hz) and an off-resonance control (0.12 Hz) -- and draws a single, clear
poster panel:

  (top)     the shared LEOGO slug-flow power pulsation at the PCC [MW]
  (large)   tower SS acceleration vs time: on-resonance build-up vs the
            almost-flat off-resonance control
  (bottom L) post-onset SS acceleration spectrum, on vs off, with the
            0.234 Hz tower SS eigenfrequency marked
  (bottom R) SS vs fore-aft (FA) acceleration on resonance -> the power/torque
            pulsation drives SS but barely touches thrust-driven FA

Usage
-----
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_slugflow_ss.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_slugflow_ss.py --show
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

C_ON = "#0b6e4f"          # on-resonance (green)
C_OFF = "#8a8f98"         # off-resonance control (grey)
C_FA = "#c0392b"          # fore-aft (red)
C_LOAD = "#2c3e50"        # slug load (dark)


def _post_onset(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    t = df["t"].to_numpy()
    m = t >= ONSET_S
    return t[m], df[col].to_numpy()[m]


def _spectrum(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dt = float(np.median(np.diff(t)))
    win = np.hanning(len(y))
    spec = np.abs(np.fft.rfft((y - np.mean(y)) * win)) * (2.0 / np.sum(win))
    freqs = np.fft.rfftfreq(len(y), dt)
    return freqs, spec


def _steady_amp(y: np.ndarray) -> float:
    tail = y[int(0.6 * len(y)):]
    return 0.5 * float(np.ptp(tail))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the slug-flow tower SS resonance case.")
    parser.add_argument("--on-csv", type=str,
                        default=str(CSV_DIR / "WT1_LEOGO_slugflow_ss_0p234Hz_2p00MW.csv"))
    parser.add_argument("--off-csv", type=str,
                        default=str(CSV_DIR / "WT1_LEOGO_slugflow_ss_0p120Hz_2p00MW.csv"))
    parser.add_argument("--out", type=str,
                        default=str(PROJECT_ROOT / "results" / "em_interaction_sweep" / "slugflow_ss_tower.png"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--fsupport", action="store_true",
                        help="Marker figuren som 'frekvensstoette paa' i suptittelen.")
    args = parser.parse_args()

    df_on = pd.read_csv(args.on_csv)
    df_off = pd.read_csv(args.off_csv)

    # Post-onset signals for the spectra + amplitudes.
    t_on, ss_on = _post_onset(df_on, "ss_accel_mps2")
    t_off, ss_off = _post_onset(df_off, "ss_accel_mps2")
    _, fa_on = _post_onset(df_on, "fa_accel_mps2")

    f_on, s_on = _spectrum(t_on, ss_on)
    f_off, s_off = _spectrum(t_off, ss_off)

    amp_ss_on = _steady_amp(ss_on)
    amp_ss_off = _steady_amp(ss_off)
    amp_fa_on = _steady_amp(fa_on)
    amp_mw = float(np.max(np.abs(df_on["load_mw"].to_numpy())))

    gain_on = amp_ss_on / max(amp_mw, 1e-9)
    gain_off = amp_ss_off / max(amp_mw, 1e-9)
    resonance_ratio = amp_ss_on / max(amp_ss_off, 1e-12)
    ss_fa_ratio = amp_ss_on / max(amp_fa_on, 1e-12)

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    fig = plt.figure(figsize=(13.5, 8.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.55, 1.25, 1.0],
                  hspace=0.42, wspace=0.22)

    # --- (top) shared slug-flow power pulsation --------------------------------
    ax_load = fig.add_subplot(gs[0, :])
    ax_load.plot(df_on["t"], df_on["load_mw"], color=C_LOAD, lw=1.1)
    ax_load.set_ylabel("Slug-last\n[MW]")
    ax_load.set_title(
        f"LEOGO slug-flow effektpulsering på PCC (Main Bus A):  "
        f"{amp_mw:.1f} MW amplitude,  på-resonans {F_SS_HZ:.3f} Hz",
        fontsize=11)
    ax_load.set_xlim(0, df_on["t"].iloc[-1])
    ax_load.axvline(ONSET_S, color="k", ls=":", lw=0.8, alpha=0.6)

    # --- (large) SS acceleration: on vs off ------------------------------------
    ax_ss = fig.add_subplot(gs[1, :])
    ax_ss.plot(df_off["t"], df_off["ss_accel_mps2"], color=C_OFF, lw=1.0,
               label=f"av-resonans 0.120 Hz  (amp {amp_ss_off:.2e} m/s²)")
    ax_ss.plot(df_on["t"], df_on["ss_accel_mps2"], color=C_ON, lw=1.2,
               label=f"på-resonans {F_SS_HZ:.3f} Hz  (amp {amp_ss_on:.2e} m/s²)")
    ax_ss.set_ylabel("Tårn side-til-side\nakselerasjon [m/s²]")
    ax_ss.set_xlabel("Tid [s]")
    ax_ss.set_xlim(0, df_on["t"].iloc[-1])
    ax_ss.axvline(ONSET_S, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_ss.legend(loc="upper left", framealpha=0.9)
    ax_ss.set_title(
        f"Resonant oppbygging: {resonance_ratio:.0f}× større SS-respons på "
        f"tårnmoden enn av-resonans", fontsize=11)

    # --- (bottom left) SS spectra ----------------------------------------------
    ax_sp = fig.add_subplot(gs[2, 0])
    ax_sp.plot(f_off, s_off, color=C_OFF, lw=1.2, label="av-resonans 0.120 Hz")
    ax_sp.plot(f_on, s_on, color=C_ON, lw=1.4, label="på-resonans 0.234 Hz")
    ax_sp.axvline(F_SS_HZ, color=C_ON, ls="--", lw=1.0, alpha=0.7)
    ax_sp.annotate(f"tårnmode\n{F_SS_HZ:.3f} Hz", xy=(F_SS_HZ, ax_sp.get_ylim()[1]),
                   xytext=(F_SS_HZ + 0.03, 0.82 * max(s_on.max(), s_off.max())),
                   fontsize=9, color=C_ON)
    ax_sp.set_xlim(0, 0.6)
    ax_sp.set_xlabel("Frekvens [Hz]")
    ax_sp.set_ylabel("SS-akselerasjon\nspekter [m/s²]")
    ax_sp.set_title("Etter-pulsering spekter (SS)", fontsize=11)
    ax_sp.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # --- (bottom right) SS vs FA on resonance (selectivity) --------------------
    ax_sel = fig.add_subplot(gs[2, 1])
    t_win = df_on["t"].iloc[-1] - 40.0
    win = df_on["t"] >= t_win
    ax_sel.plot(df_on["t"][win], df_on["ss_accel_mps2"][win], color=C_ON, lw=1.2,
                label=f"SS  (amp {amp_ss_on:.2e})")
    ax_sel.plot(df_on["t"][win], df_on["fa_accel_mps2"][win], color=C_FA, lw=1.2,
                label=f"FA  (amp {amp_fa_on:.2e})")
    ax_sel.set_xlim(t_win, df_on["t"].iloc[-1])
    ax_sel.set_xlabel("Tid [s]")
    ax_sel.set_ylabel("Akselerasjon [m/s²]")
    ax_sel.set_title(f"Modeselektivitet på resonans:  SS/FA = {ss_fa_ratio:.0f}×",
                     fontsize=11)
    ax_sel.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Slug-flow prosesslast -> tårn side-til-side resonans  "
        f"(transfergain {gain_on:.2e} vs {gain_off:.2e} (m/s²)/MW)"
        + ("   —   frekvensstøtte på" if args.fsupport else ""),
        fontsize=13, y=0.985)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Lagret figur: {out}")
    print(f"  SS paa-res amp = {amp_ss_on:.3e} m/s^2,  av-res amp = {amp_ss_off:.3e} m/s^2")
    print(f"  resonansforhold = {resonance_ratio:.1f}x,  SS/FA = {ss_fa_ratio:.1f}x")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
