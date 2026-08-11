r"""
Poster figure for the reciprocating-compressor drivetrain-torsion resonance
case (companion to test_WT_LEOGO_process_load_excitation_sim.py).

Reads two runs -- an on-resonance run (compressor pulsation at the 3.49 Hz
drivetrain torsion mode) and an off-resonance control (2.00 Hz) -- and draws a
single, clear poster panel:

  (top)      the shared LEOGO compressor power pulsation at the PCC [MW]
  (large)    drivetrain shaft-torque deviation vs time: on-resonance build-up
             vs the small off-resonance response
  (bottom L) post-onset shaft-torque spectrum, on vs off, with the 3.49 Hz
             torsion eigenfrequency marked
  (bottom R) zoom on the settled on-resonance oscillation -> the +-230 kNm
             shaft-torque ripple locked to the compressor pulsation

Usage
-----
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_process_pulsation.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_process_pulsation.py --show
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

F_TORSION_HZ = 3.49       # drivetrain torsion eigenfrequency

C_ON = "#b9770e"          # on-resonance (amber)
C_OFF = "#8a8f98"         # off-resonance control (grey)
C_LOAD = "#2c3e50"        # compressor load (dark)


def _onset(df: pd.DataFrame) -> float:
    nz = df.index[df["load_mw"].to_numpy() != 0.0]
    return float(df["t"].iloc[nz[0]]) if len(nz) else 0.0


def _shaft_dev_knm(df: pd.DataFrame, onset: float) -> np.ndarray:
    """Shaft torque as a deviation [kNm] from its pre-onset steady value."""
    t = df["t"].to_numpy()
    tq = df["T_shaft_Nm"].to_numpy()
    base = float(np.mean(tq[t < onset])) if np.any(t < onset) else float(tq[0])
    return (tq - base) / 1.0e3


def _post_onset(t: np.ndarray, y: np.ndarray, onset: float) -> tuple[np.ndarray, np.ndarray]:
    m = t >= onset
    return t[m], y[m]


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
    parser = argparse.ArgumentParser(description="Plot the compressor-pulsation drivetrain torsion case.")
    parser.add_argument("--on-csv", type=str,
                        default=str(CSV_DIR / "WT1_LEOGO_process_pulsation_3p49Hz_0p50MW.csv"))
    parser.add_argument("--off-csv", type=str,
                        default=str(CSV_DIR / "WT1_LEOGO_process_pulsation_2p00Hz_0p50MW.csv"))
    parser.add_argument("--off-freq-hz", type=float, default=2.00)
    parser.add_argument("--out", type=str,
                        default=str(PROJECT_ROOT / "results" / "em_interaction_sweep" / "process_pulsation_torsion.png"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--fsupport", action="store_true",
                        help="Marker figuren som 'frekvensstoette paa' i suptittelen.")
    args = parser.parse_args()

    df_on = pd.read_csv(args.on_csv)
    df_off = pd.read_csv(args.off_csv)

    onset_on = _onset(df_on)
    onset_off = _onset(df_off)

    t_on = df_on["t"].to_numpy()
    t_off = df_off["t"].to_numpy()
    sh_on = _shaft_dev_knm(df_on, onset_on)
    sh_off = _shaft_dev_knm(df_off, onset_off)

    tp_on, shp_on = _post_onset(t_on, sh_on, onset_on)
    tp_off, shp_off = _post_onset(t_off, sh_off, onset_off)

    f_on, s_on = _spectrum(tp_on, shp_on)
    f_off, s_off = _spectrum(tp_off, shp_off)

    amp_on = _steady_amp(shp_on)
    amp_off = _steady_amp(shp_off)
    amp_mw = float(np.max(np.abs(df_on["load_mw"].to_numpy())))

    gain_on = amp_on / max(amp_mw, 1e-9)
    gain_off = amp_off / max(amp_mw, 1e-9)
    resonance_ratio = amp_on / max(amp_off, 1e-12)

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    fig = plt.figure(figsize=(13.5, 8.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.55, 1.25, 1.0],
                  hspace=0.42, wspace=0.22)

    # --- (top) shared compressor power pulsation -------------------------------
    ax_load = fig.add_subplot(gs[0, :])
    ax_load.plot(df_on["t"], df_on["load_mw"], color=C_LOAD, lw=0.9)
    ax_load.set_ylabel("Kompressor-\nlast [MW]")
    ax_load.set_title(
        f"LEOGO kompressor-effektpulsering paa PCC (Main Bus A):  "
        f"{amp_mw:.2f} MW amplitude,  paa-resonans {F_TORSION_HZ:.2f} Hz",
        fontsize=11)
    ax_load.set_xlim(0, df_on["t"].iloc[-1])
    ax_load.axvline(onset_on, color="k", ls=":", lw=0.8, alpha=0.6)

    # --- (large) shaft-torque deviation: on vs off -----------------------------
    ax_sh = fig.add_subplot(gs[1, :])
    ax_sh.plot(t_off, sh_off, color=C_OFF, lw=0.9,
               label=f"av-resonans {args.off_freq_hz:.2f} Hz  (amp {amp_off:.0f} kNm)")
    ax_sh.plot(t_on, sh_on, color=C_ON, lw=1.0,
               label=f"paa-resonans {F_TORSION_HZ:.2f} Hz  (amp {amp_on:.0f} kNm)")
    ax_sh.set_ylabel("Akselmoment-avvik\n[kNm]")
    ax_sh.set_xlabel("Tid [s]")
    ax_sh.set_xlim(0, df_on["t"].iloc[-1])
    ax_sh.axvline(onset_on, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_sh.legend(loc="upper left", framealpha=0.9)
    ax_sh.set_title(
        f"Resonant oppbygging: {resonance_ratio:.0f}x stoerre akselmoment paa "
        f"torsjonsmoden enn av-resonans", fontsize=11)

    # --- (bottom left) shaft-torque spectra ------------------------------------
    ax_sp = fig.add_subplot(gs[2, 0])
    ax_sp.plot(f_off, s_off, color=C_OFF, lw=1.2, label=f"av-resonans {args.off_freq_hz:.2f} Hz")
    ax_sp.plot(f_on, s_on, color=C_ON, lw=1.4, label=f"paa-resonans {F_TORSION_HZ:.2f} Hz")
    ax_sp.axvline(F_TORSION_HZ, color=C_ON, ls="--", lw=1.0, alpha=0.7)
    ax_sp.annotate(f"torsjonsmode\n{F_TORSION_HZ:.2f} Hz",
                   xy=(F_TORSION_HZ, s_on.max()),
                   xytext=(F_TORSION_HZ - 1.7, 0.78 * max(s_on.max(), s_off.max())),
                   fontsize=9, color=C_ON)
    ax_sp.set_xlim(0, 6)
    ax_sp.set_xlabel("Frekvens [Hz]")
    ax_sp.set_ylabel("Akselmoment-\nspekter [kNm]")
    ax_sp.set_title("Etter-pulsering spekter (akselmoment)", fontsize=11)
    ax_sp.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # --- (bottom right) settled on-resonance oscillation -----------------------
    ax_zoom = fig.add_subplot(gs[2, 1])
    t_win = df_on["t"].iloc[-1] - 5.0
    win = df_on["t"] >= t_win
    ax_zoom.plot(df_on["t"][win], sh_on[win.to_numpy()], color=C_ON, lw=1.3,
                 label=f"akselmoment  (+/-{amp_on:.0f} kNm)")
    ax_zoom.set_xlim(t_win, df_on["t"].iloc[-1])
    ax_zoom.set_xlabel("Tid [s]")
    ax_zoom.set_ylabel("Akselmoment-avvik\n[kNm]")
    ax_zoom.set_title(f"Innsvingt {F_TORSION_HZ:.2f} Hz akselmoment-rippel",
                      fontsize=11)
    ax_zoom.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Kompressor-pulsering -> drivverk torsjonsresonans  "
        f"(transfergain {gain_on:.0f} vs {gain_off:.0f} kNm/MW)"
        + ("   —   frekvensstøtte på" if args.fsupport else ""),
        fontsize=13, y=0.985)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Lagret figur: {out}")
    print(f"  akselmoment paa-res amp = {amp_on:.1f} kNm,  av-res amp = {amp_off:.1f} kNm")
    print(f"  resonansforhold = {resonance_ratio:.1f}x,  transfergain paa-res = {gain_on:.1f} kNm/MW")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
