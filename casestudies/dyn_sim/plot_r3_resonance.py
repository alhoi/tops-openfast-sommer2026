"""Region-3 slug-flow -> tower side-to-side (SS) resonance figure.

FMU-adapted counterpart of plot_slugflow_ss.py (which reads the reduced
analytic tower model). Here both traces come from the full OpenFAST FMU +
LEOGO co-simulation, with the grid-forming UIC and the droop-based
frequency support ON:

    slug-flow process-load pulsation on the LEOGO PCC (Main Bus A)
        -> grid frequency oscillation at f_load
        -> droop frequency support modulates the WT generator torque
        -> excites the lightly damped tower SS mode (0.233 Hz).

Two runs share every parameter except the load pulsation frequency:
  * on-resonance  : f_load = F_SS_HZ (0.233 Hz, the tower SS mode)
  * off-resonance : f_load = --f-off (0.120 Hz, away from the mode)

Because the fore-aft DOF is disabled (SS-only study), the reference
figure's SS/FA selectivity panel is replaced by the coupling signal that
drives the tower: the droop generator-torque modulation.

Signals (FMU CSV columns):
  * tower SS acceleration : fmu_YawBrTAyp   [m/s^2]
  * slug load             : load_step_scale x --load-amp-mw  [MW]
  * droop torque offset   : zmq_torque_offset_nm            [Nm]
  * grid frequency        : f_grid_hz                        [Hz]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

F_SS_HZ = 0.233  # tower side-to-side mode [Hz]

C_ON = "#0b6e4f"    # green  - on resonance
C_OFF = "#8a8f98"   # grey   - off resonance
C_LOAD = "#2c3e50"  # dark   - slug load
C_TRQ = "#b8860b"   # gold   - droop torque modulation
C_F = "#c0392b"     # red    - grid frequency


def _post_onset(df: pd.DataFrame, col: str, onset: float):
    """Return (t, signal) for t >= onset with the onset-time mean removed."""
    m = df["t"].to_numpy() >= onset
    t = df["t"].to_numpy()[m]
    y = df[col].to_numpy()[m]
    return t, y - np.mean(y)


def _late_window(df: pd.DataFrame, col: str, onset: float, frac: float = 0.4):
    """Return (t, signal) over the last `frac` of the post-onset span.

    The tower SS mode is very lightly damped (tau ~ 170 s), so the FMU
    initialisation transient at 0.233 Hz decays slowly and contaminates the
    early record. Restricting the spectrum/amplitude estimate to the tail lets
    the forced (slug-driven) response dominate over that residual ring-down.
    """
    t = df["t"].to_numpy()
    t0 = t[-1] - frac * (t[-1] - onset)
    m = t >= t0
    tt = t[m]
    yy = df[col].to_numpy()[m]
    return tt, yy - np.mean(yy)


def _spectrum(t: np.ndarray, y: np.ndarray):
    """One-sided amplitude spectrum of a uniformly sampled signal (Hann)."""
    if t.size < 8:
        return np.array([0.0]), np.array([0.0])
    dt = float(np.median(np.diff(t)))
    w = np.hanning(y.size)
    yw = (y - y.mean()) * w
    spec = np.abs(np.fft.rfft(yw)) / np.sum(w) * 2.0
    freq = np.fft.rfftfreq(y.size, dt)
    return freq, spec


def _steady_amp(y: np.ndarray, frac: float = 0.4) -> float:
    """Half peak-to-peak of the last `frac` of the signal (settled amplitude)."""
    n = max(4, int(y.size * frac))
    tail = y[-n:]
    return 0.5 * float(np.ptp(tail))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--on-csv", default=r"results\sweep\r3res3_on.csv")
    p.add_argument("--off-csv", default=r"results\sweep\r3res3_off.csv")
    p.add_argument("--onset", type=float, default=40.0,
                   help="Time [s] at which the slug pulsation ramps in.")
    p.add_argument("--load-amp-mw", type=float, default=2.0,
                   help="Slug-load pulsation amplitude [MW] (load_step_scale x this).")
    p.add_argument("--f-off", type=float, default=0.120,
                   help="Off-resonance load frequency [Hz] (for labels).")
    p.add_argument("--support-label",
                   default="droop frequency support on (no virtual inertia)",
                   help="Description of the active frequency support (suptitle).")
    p.add_argument("--xstart", type=float, default=0.0,
                   help="Left x-axis limit [s] for the load + SS time panels "
                        "(trim the FMU init transient before the onset).")
    p.add_argument("--out", default=r"results\em_interaction_sweep\r3_droop_ss_resonance.png")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    df_on = pd.read_csv(args.on_csv)
    df_off = pd.read_csv(args.off_csv)

    ss_col = "fmu_YawBrTAyp"
    amp = args.load_amp_mw
    for d in (df_on, df_off):
        d["load_mw"] = d["load_step_scale"] * amp
        d["trq_knm"] = d["zmq_torque_offset_nm"] / 1.0e3
        d["df_mhz"] = (d["f_grid_hz"] - 50.0) * 1.0e3

    # Spectra + settled amplitudes over the tail window, where the forced slug
    # response dominates over the slowly decaying init-transient tower ring.
    t_on, ss_on = _late_window(df_on, ss_col, args.onset)
    t_off, ss_off = _late_window(df_off, ss_col, args.onset)

    f_on, s_on = _spectrum(t_on, ss_on)
    f_off, s_off = _spectrum(t_off, ss_off)

    amp_ss_on = _steady_amp(ss_on)
    amp_ss_off = _steady_amp(ss_off)
    amp_mw = float(np.max(np.abs(df_on["load_mw"].to_numpy())))

    _, trq_on = _late_window(df_on, "trq_knm", args.onset)
    _, trq_off = _late_window(df_off, "trq_knm", args.onset)
    amp_trq_on = _steady_amp(trq_on)
    amp_trq_off = _steady_amp(trq_off)

    _, df_on_w = _late_window(df_on, "df_mhz", args.onset)
    _, df_off_w = _late_window(df_off, "df_mhz", args.onset)
    amp_df_on = _steady_amp(df_on_w)
    amp_df_off = _steady_amp(df_off_w)

    gain_on = amp_ss_on / max(amp_mw, 1e-9)
    gain_off = amp_ss_off / max(amp_mw, 1e-9)
    resonance_ratio = amp_ss_on / max(amp_ss_off, 1e-12)

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    fig = plt.figure(figsize=(13.5, 8.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.55, 1.25, 1.0],
                  hspace=0.42, wspace=0.22)

    t_end = float(df_on["t"].iloc[-1])

    # --- (top) shared slug-flow power pulsation --------------------------------
    ax_load = fig.add_subplot(gs[0, :])
    ax_load.plot(df_on["t"], df_on["load_mw"], color=C_LOAD, lw=1.1)
    ax_load.set_ylabel("Slug load\n[MW]")
    ax_load.set_title(
        f"Slug load on the grid:  ±{amp_mw:.1f} MW at {F_SS_HZ:.3f} Hz",
        fontsize=11)
    ax_load.set_xlim(args.xstart, t_end)
    ax_load.axvline(args.onset, color="k", ls=":", lw=0.8, alpha=0.6)

    # --- (large) SS acceleration: on vs off ------------------------------------
    ax_ss = fig.add_subplot(gs[1, :])
    ax_ss.plot(df_off["t"], df_off[ss_col], color=C_OFF, lw=1.0,
               label=f"off resonance ({args.f_off:.3f} Hz)")
    ax_ss.plot(df_on["t"], df_on[ss_col], color=C_ON, lw=1.2,
               label=f"on resonance ({F_SS_HZ:.3f} Hz)")
    ax_ss.set_ylabel("Tower side-to-side\nacceleration [m/s$^2$]")
    ax_ss.set_xlabel("Time [s]")
    ax_ss.set_xlim(args.xstart, t_end)
    ax_ss.axvline(args.onset, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_ss.annotate("slug load on", xy=(args.onset, ax_ss.get_ylim()[1]),
                   xytext=(args.onset + 4, 0.82 * ax_ss.get_ylim()[1]),
                   fontsize=9, color="k")
    ax_ss.legend(loc="upper left", framealpha=0.9)
    ax_ss.set_title(
        f"SS resonance: {resonance_ratio:.0f}× larger at the tower mode", fontsize=11)

    # --- (bottom left) SS spectra ----------------------------------------------
    ax_sp = fig.add_subplot(gs[2, 0])
    ax_sp.plot(f_off, s_off, color=C_OFF, lw=1.2, label="off resonance")
    ax_sp.plot(f_on, s_on, color=C_ON, lw=1.4, label="on resonance")
    ax_sp.axvline(F_SS_HZ, color=C_ON, ls="--", lw=1.0, alpha=0.7)
    smax = max(s_on.max(), s_off.max())
    ax_sp.annotate(f"tower mode\n{F_SS_HZ:.3f} Hz", xy=(F_SS_HZ, smax),
                   xytext=(F_SS_HZ + 0.03, 0.82 * smax), fontsize=9, color=C_ON)
    ax_sp.set_xlim(0, 0.6)
    ax_sp.set_xlabel("Frequency [Hz]")
    ax_sp.set_ylabel("SS spectrum\n[m/s$^2$]")
    ax_sp.set_title("Spectrum (SS)", fontsize=11)
    ax_sp.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # --- (bottom right) grid frequency: the coupling variable ------------------
    ax_f = fig.add_subplot(gs[2, 1])
    t_win = t_end - 60.0
    win_on = df_on["t"] >= t_win
    win_off = df_off["t"] >= t_win
    ax_f.plot(df_off["t"][win_off], df_off["df_mhz"][win_off], color=C_OFF, lw=1.0,
              label="off resonance")
    ax_f.plot(df_on["t"][win_on], df_on["df_mhz"][win_on], color=C_F, lw=1.3,
              label="on resonance")
    ax_f.set_xlim(t_win, t_end)
    ax_f.set_xlabel("Time [s]")
    ax_f.set_ylabel("Grid frequency deviation\n$\\Delta f$ [mHz]")
    ax_f.set_title("Grid frequency pulsates", fontsize=11)
    ax_f.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Slug load drives tower resonance (Region 3)",
        fontsize=13, y=0.995)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {out}")
    print(f"  SS on-res amp   = {amp_ss_on:.3e} m/s2")
    print(f"  SS off-res amp  = {amp_ss_off:.3e} m/s2")
    print(f"  resonance ratio = {resonance_ratio:.2f}x")
    print(f"  droop torque amp: on-res {amp_trq_on:.0f} kNm, off-res {amp_trq_off:.0f} kNm")
    print(f"  grid freq amp:    on-res {amp_df_on:.1f} mHz, off-res {amp_df_off:.1f} mHz")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
