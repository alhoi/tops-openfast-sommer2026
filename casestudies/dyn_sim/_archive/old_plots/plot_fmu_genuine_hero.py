r"""
FMU hero figure: one generator-torque kick -> BOTH genuine OpenFAST resonances.

Companion to the reduced-model hero (plot_em_interaction_hero.py), in the SAME
4x2 layout, but built from a genuine high-fidelity OpenFAST co-simulation.

In the OpenFAST/ROSCO FMU the electrical->mechanical path is blocked (VSContrl=5:
the generator is a torque actuator, so a pure LEOGO grid event never reaches the
turbine structure). The only channel that does reach the structure is the
generator electromagnetic torque, imposed here through ROSCO open-loop control
(OL_Mode = 1). A short Hann torque PULSE is broadband, so a single kick
simultaneously rings the two lightly damped OpenFAST modes reachable through
that torque port:

    electromagnetic-torque pulse
        -> P_e perturbation on the LEOGO co-sim grid
        -> drivetrain torsion   (HSShftTq, ~3.10 Hz)
        -> tower side-to-side   (YawBrTAyp, ~0.234 Hz)

Layout (identical to the reduced-model hero):

    [ gen-torque pulse ] [ HSShftTq spectrum (peak ~3.10 Hz) ]
    [ P_e (co-sim)     ] [ HSShftTq spectrum                 ]
    [ drivetrain tors. ] [ tower SS spectrum (peak ~0.234 Hz)]
    [ tower SS accel.  ] [ tower SS spectrum                 ]

Reads the pulse co-sim CSV written by test_WT_LEOGO_FMU_sim.py --fmu debug
(records fmu_HSShftTq, fmu_YawBrTAyp, P_e_sys_pu, f_grid_hz).

Usage
-----
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_fmu_genuine_hero.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_fmu_genuine_hero.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "em_interaction_sweep"

F_TORSION_HZ = 3.10
F_SS_HZ = 0.234

SHAFT_CH = "fmu_HSShftTq"   # high-speed shaft torque [kNm]
SS_CH = "fmu_YawBrTAyp"     # tower-top side-to-side acceleration [m/s^2]
TQ_CH = "fmu_GenTq"         # applied generator torque [kNm]
PE_CH = "P_e_sys_pu"        # WT electrical power (system base) [pu]

PE_FCUT_HZ = 1.0            # low-pass cutoff for the P_e panel [Hz]


def lowpass(t, sig, f_cut):
    """Zero-phase Butterworth low-pass.

    Suppresses the high-frequency (~4.9 Hz) self-excited co-simulation
    oscillation so the P_e panel shows the genuine sub-Hz electromagnetic
    coupling of the torque event to the LEOGO grid.
    """
    from scipy.signal import butter, filtfilt
    y = np.asarray(sig, dtype=float)
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt
    b, a = butter(4, f_cut / (0.5 * fs), btype="low")
    return filtfilt(b, a, y)


def post_onset_spectrum(t, sig, onset, f_hi):
    mask = t >= onset
    tt = t[mask]
    s = np.asarray(sig, dtype=float)[mask]
    s = s - np.mean(s)
    dt = float(np.mean(np.diff(tt)))
    win = np.hanning(len(s))
    amp = np.abs(np.fft.rfft(s * win)) * (2.0 / np.sum(win))
    freqs = np.fft.rfftfreq(len(s), dt)
    keep = freqs <= f_hi
    return freqs[keep], amp[keep]


def peak_in_band(freqs, amp, lo, hi):
    band = (freqs >= lo) & (freqs <= hi)
    if not np.any(band):
        return float("nan"), float("nan")
    i = int(np.argmax(amp[band]))
    return float(freqs[band][i]), float(amp[band][i])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FMU hero: one torque pulse -> both genuine OpenFAST modes.")
    parser.add_argument("--csv", type=str,
                        default=str(OUT_DIR / "WT1_LEOGO_FMU_hero_pulse.csv"))
    parser.add_argument("--onset", type=float, default=10.0)
    parser.add_argument("--out", type=str,
                        default=str(OUT_DIR / "fmu_genuine_hero.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    t = df["t"].to_numpy()
    onset = args.onset
    xlim = (max(0.0, onset - 2.0), float(t[-1]))

    # Shaft torque as deviation from the pre-onset mean [kNm].
    pre = t < onset
    shaft0 = float(np.mean(df[SHAFT_CH].to_numpy()[pre]))
    shaft_dev = df[SHAFT_CH].to_numpy() - shaft0
    gentq = df[TQ_CH].to_numpy()
    p_e = df[PE_CH].to_numpy()
    p_e_lp = lowpass(t, p_e, PE_FCUT_HZ)
    ss = df[SS_CH].to_numpy()

    f_shaft, a_shaft = post_onset_spectrum(t, df[SHAFT_CH].to_numpy(),
                                           onset, f_hi=6.0)
    f_tower, a_tower = post_onset_spectrum(t, ss, onset, f_hi=1.0)
    pk_shaft = peak_in_band(f_shaft, a_shaft, 2.5, 4.5)
    pk_tower = peak_in_band(f_tower, a_tower, 0.15, 0.35)

    chain = ("#1f4e79", "#2e75b6", "#c55a11", "#548235")

    mosaic = [
        ["tq", "shaft_sp"],
        ["pe", "shaft_sp"],
        ["shaft", "tower_sp"],
        ["tower", "tower_sp"],
    ]
    fig, ax = plt.subplot_mosaic(
        mosaic, figsize=(13, 9),
        gridspec_kw={"width_ratios": [2.0, 1.0], "hspace": 0.35, "wspace": 0.25},
    )

    # --- left column: the cascade in time --------------------------------------
    ax["tq"].plot(t, gentq, color=chain[0])
    ax["tq"].set_ylabel("Gen.moment\n[kNm]")
    ax["tq"].set_title("Elektromagnetisk moment-puls (ROSCO open-loop)",
                       fontsize=10, loc="left")

    ax["pe"].plot(t, p_e, color="0.75", lw=0.8, alpha=0.6,
                  label="rå co-sim")
    ax["pe"].plot(t, p_e_lp, color=chain[1], lw=1.5,
                  label=f"lavpass < {PE_FCUT_HZ:.0f} Hz")
    ax["pe"].set_ylabel("P_e\n[pu]")
    ax["pe"].set_title(
        "Vindturbinens elektriske port (UIC) i LEOGO-cosim  "
        "(lavpass; høyfrekvent co-sim-numerikk dempet)",
        fontsize=10, loc="left")
    ax["pe"].legend(fontsize=7, loc="upper right", framealpha=0.6)

    ax["shaft"].plot(t, shaft_dev, color=chain[2])
    ax["shaft"].set_ylabel("Akselmoment-\navvik [kNm]")
    ax["shaft"].set_title("Drivverk (torsjon) -- genuin OpenFAST DrTrDOF",
                          fontsize=10, loc="left")

    ax["tower"].plot(t, ss, color=chain[3])
    ax["tower"].set_ylabel("Tårn SS\n[m/s²]")
    ax["tower"].set_xlabel("Tid [s]")
    ax["tower"].set_title("Tårn side-til-side -- genuin OpenFAST TwSSDOF1",
                          fontsize=10, loc="left")

    for key in ("tq", "pe", "shaft", "tower"):
        ax[key].axvline(onset, color="0.4", ls="--", lw=1.0)
        ax[key].set_xlim(*xlim)
        ax[key].grid(True, alpha=0.3)
    for key in ("tq", "pe", "shaft"):
        ax[key].tick_params(labelbottom=False)

    # --- right column: spectra prove both eigenfrequencies ---------------------
    sh_band = f_shaft >= 1.0
    ax["shaft_sp"].plot(f_shaft[sh_band], a_shaft[sh_band], color=chain[2])
    ax["shaft_sp"].axvline(pk_shaft[0], color="0.4", ls=":", lw=1.0)
    ax["shaft_sp"].set_title(
        f"Akselmoment-spekter  (topp {pk_shaft[0]:.2f} Hz)",
        fontsize=10, loc="left")
    ax["shaft_sp"].set_ylabel("Amplitude [kNm]")
    ax["shaft_sp"].set_xlim(1.0, 6.0)

    ax["tower_sp"].plot(f_tower, a_tower, color=chain[3])
    ax["tower_sp"].axvline(pk_tower[0], color="0.4", ls=":", lw=1.0)
    ax["tower_sp"].set_title(
        f"Tårn-spekter  (topp {pk_tower[0]:.3f} Hz)",
        fontsize=10, loc="left")
    ax["tower_sp"].set_ylabel("Amplitude [m/s²]")
    ax["tower_sp"].set_xlabel("Frekvens [Hz]")
    ax["tower_sp"].set_xlim(0.0, 1.0)

    for key in ("shaft_sp", "tower_sp"):
        ax[key].grid(True, alpha=0.3)

    fig.suptitle(
        "Én generatormoment-puls -> genuin OpenFAST drivverk-torsjon "
        f"({pk_shaft[0]:.2f} Hz) + tårn side-til-side ({pk_tower[0]:.3f} Hz)",
        fontsize=13, fontweight="bold",
    )

    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.08,
                        hspace=0.35, wspace=0.25)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Lagret figur: {out}")
    print(f"  drivverk-topp = {pk_shaft[0]:.3f} Hz ({pk_shaft[1]:.1f} kNm), "
          f"tårn-topp = {pk_tower[0]:.4f} Hz ({pk_tower[1]:.4f} m/s²)")
    print(f"  akselmoment-avvik p2p = {np.ptp(shaft_dev):.1f} kNm, "
          f"tårn SS p2p = {np.ptp(ss):.4f} m/s²")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
