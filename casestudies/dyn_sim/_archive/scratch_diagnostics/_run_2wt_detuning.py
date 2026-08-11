r"""
Two-turbine tower side-to-side (SS) de-tuning demonstration on LEOGO.

Two IEA-15MW turbines sit at two different physical locations on the LEOGO
oil-rig grid (Busbar WTG1 LV and WTG2 LV). They are identical EXCEPT for their
tower SS natural frequency, reflecting that real monopiles in different soil
profiles do not share exactly the same first tower mode:

    WT1: f_ss = 0.234 Hz
    WT2: f_ss = 0.250 Hz   (~6.8 % higher)

A slug-flow process-load pulsation (+/-2 MW) is injected at the rig's main bus
(the point of common coupling felt by both turbines). Because the SS mode is
extremely lightly damped (zeta ~ 0.0034 => Q ~ 147, half-power bandwidth ~0.7 %),
a 6.8 % frequency separation is far wider than the resonance peak, so a single
process frequency cannot resonate both turbines at once.

The script runs the shared two-turbine simulation ONCE, forcing the slug at
WT1's SS frequency (0.234 Hz). It then compares the two turbines' responses
and looks for turbine-to-turbine interaction: WT1 (on resonance) rings up and
pumps a 0.234 Hz power oscillation into the shared rig grid, while WT2 (off
resonance) only responds weakly -- and at the FORCING frequency, not its own
0.25 Hz -- i.e. WT2 is driven through the grid, it does not self-resonate.

Usage
-----
  .\.venv\Scripts\python.exe casestudies\dyn_sim\_run_2wt_detuning.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\_run_2wt_detuning.py --force
  .\.venv\Scripts\python.exe casestudies\dyn_sim\_run_2wt_detuning.py --show
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIM = PROJECT_ROOT / "casestudies" / "dyn_sim" / "test_WT_LEOGO_2WT_sim.py"
CSV_DIR = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "csv_files"
FIG_DIR = PROJECT_ROOT / "results" / "em_interaction_sweep"

F_WT1 = 0.234        # WT1 tower SS natural frequency [Hz]
F_WT2 = 0.250        # WT2 tower SS natural frequency [Hz]
F_WT1_CTRL = 0.250   # WT1 de-tuned to WT2's frequency for the control run [Hz]
AMP_MW = 2.0         # slug pulsation peak [MW]
ONSET_S = 10.0       # slug onset [s]
T_END = 200.0        # simulation horizon [s]
WARMUP_S = 30.0      # pre-t0 relaxation for a flat baseline [s]
WIND_MPS = 10.0      # identical wind for both turbines (isolate de-tuning)

C_WT1 = "#1f5fb0"    # blue  (WT1, 0.234 Hz)
C_WT2 = "#c0392b"    # red   (WT2, 0.250 Hz)
C_LOAD = "#2c3e50"

# Two-way SS <-> drivetrain reciprocal coupling (optional, off by default).
# Symmetric 2x2 mass matrix [[J_e, -c], [-c, 1]] with a single coefficient c
# (positive definite for c^2 < J_e). The coupled SS eigenfrequency shifts up by
# sqrt(J_e/(J_e - c^2)), so the base f_ss handed to the model is pre-divided by
# that factor to hold each turbine's *coupled* tower resonance exactly at its
# target (0.234 / 0.250 Hz). Forward-only and two-way runs then share identical
# actual resonances and the ONLY difference is the reciprocal coupling itself.
C_FB = 0.05                                     # coupling coefficient [-]
J_E_PU = 0.076748                               # generator inertia 2*H_e [pu]
_FB_DETUNE = float(np.sqrt(1.0 - C_FB ** 2 / J_E_PU))
F_WT1_FB = F_WT1 * _FB_DETUNE                    # base so coupled WT1 = 0.234 Hz
F_WT2_FB = F_WT2 * _FB_DETUNE                    # base so coupled WT2 = 0.250 Hz


def _run_one(freq_hz: float, f_wt1: float, f_wt2: float,
             out_csv: Path, force: bool,
             fb_c: float = 0.0, fb_target: str = "both") -> None:
    """Run the two-turbine sim once (slug at freq_hz, per-turbine SS freqs).

    When ``fb_c`` > 0 the optional two-way SS<->drivetrain coupling is enabled
    on ``fb_target`` turbines with coefficient ``fb_c``.
    """
    if out_csv.exists() and not force:
        print(f"  [skip] {out_csv.name} exists (use --force to re-run)")
        return
    cmd = [
        sys.executable, str(SIM),
        "--osc-load-mw", str(AMP_MW),
        "--osc-freq-hz", f"{freq_hz:g}",
        "--osc-start", str(ONSET_S),
        "--f-ss-hz", f"{f_wt1:g}",
        "--f-ss-hz-wt2", f"{f_wt2:g}",
        "--droop-wt1", "0", "--droop-wt2", "0",
        "--wind-wt1", f"{WIND_MPS:g}", "--wind-wt2", f"{WIND_MPS:g}",
        "--warmup-s", f"{WARMUP_S:g}",
        "--t-end", f"{T_END:g}", "--dt", "0.01",
        "--headroom", "0.05",
        "--out", str(out_csv),
    ]
    if fb_c > 0.0:
        cmd += ["--ss-feedback",
                "--ss-feedback-c", f"{fb_c:g}",
                "--ss-feedback-target", fb_target]
    fb_note = f", toveis c={fb_c:g} ({fb_target})" if fb_c > 0.0 else ""
    print(f"  [run ] slug @ {freq_hz:g} Hz, WT1 f_ss={f_wt1:g}, "
          f"WT2 f_ss={f_wt2:g}{fb_note} -> {out_csv.name}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _steady_amp(a: np.ndarray) -> float:
    """Half peak-to-peak of the settled tail (last 40 %) of a signal."""
    tail = a[int(0.6 * len(a)):]
    return 0.5 * float(np.ptp(tail))


def _amp_spectrum(t: np.ndarray, x: np.ndarray, t0: float):
    """Single-sided amplitude spectrum of x over the window t >= t0.

    A Hann window is applied and the amplitude is scaled so a pure sine of
    amplitude A reads ~A at its frequency.
    """
    m = t >= t0
    xs = x[m] - np.mean(x[m])
    n = xs.size
    dt = float(np.median(np.diff(t[m])))
    w = np.hanning(n)
    amp = np.abs(np.fft.rfft(xs * w)) * (2.0 / np.sum(w))
    freq = np.fft.rfftfreq(n, dt)
    return freq, amp


def main() -> None:
    p = argparse.ArgumentParser(
        description="Two-turbine SS interaction under a single slug at "
                    "WT1's resonance (0.234 Hz).")
    p.add_argument("--force", action="store_true",
                   help="Re-run the simulation even if the CSV already exists.")
    p.add_argument("--feedback", action="store_true",
                   help="Also run the two-way SS<->drivetrain coupling "
                        "comparison (isolates the WT1->WT2 path).")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    csv = CSV_DIR / "2wt_detuning_force0p234.csv"
    csv_ctrl = CSV_DIR / "2wt_detuning_ctrl_wt1detuned.csv"
    print("Two-turbine interaction (WT1 f_ss=0.234 Hz on resonance, "
          "WT2 f_ss=0.250 Hz off resonance),")
    print(f"single slug +/-{AMP_MW:g} MW @ {F_WT1:g} Hz at the LEOGO main bus:")
    _run_one(F_WT1, F_WT1, F_WT2, csv, args.force)
    # Control: WT1 de-tuned to WT2's frequency so neither turbine resonates;
    # isolates the WT1 -> WT2 feedback path through the shared grid.
    _run_one(F_WT1, F_WT1_CTRL, F_WT2, csv_ctrl, args.force)

    df = pd.read_csv(csv)
    t = df["t"].to_numpy()
    ss1 = df["ss_accel_mps2_wt1"].to_numpy()
    ss2 = df["ss_accel_mps2_wt2"].to_numpy()
    p1 = df["P_uic_bus_sys_pu_wt1"].to_numpy()
    p2 = df["P_uic_bus_sys_pu_wt2"].to_numpy()
    fgrid = df["grid_freq_hz"].to_numpy()

    a1 = _steady_amp(ss1)
    a2 = _steady_amp(ss2)
    ratio = a1 / max(a2, 1e-9)

    # Spectra of the SS acceleration over a developed window (skip the ring-up)
    t0 = min(40.0, 0.3 * float(t[-1]))
    f1, A1 = _amp_spectrum(t, ss1, t0)
    f2, A2 = _amp_spectrum(t, ss2, t0)
    band = f2 <= 0.5
    fband = f2[band]
    dom_wt2 = float(fband[np.argmax(A2[band])])

    plt.rcParams.update({"font.size": 10.5, "axes.grid": True,
                         "grid.alpha": 0.3})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # A: SS acceleration time series (on- vs off-resonance)
    axA.plot(t, ss1, color=C_WT1, lw=1.2,
             label=f"WT1  (0,234 Hz, på res.)  A={a1:.2e}")
    axA.plot(t, ss2, color=C_WT2, lw=1.2, ls=(0, (5, 2)),
             label=f"WT2  (0,250 Hz, av res.)  A={a2:.2e}")
    axA.axvline(ONSET_S, color="#888888", lw=0.8, ls=":")
    axA.set_xlabel("Tid [s]")
    axA.set_ylabel("Tårn SS-akselerasjon  [m/s²]")
    axA.set_title(f"a) Tårnrespons — WT1/WT2 amplitudeforhold ≈ {ratio:.0f}×",
                  fontsize=10)
    axA.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # B: amplitude spectra of the SS acceleration
    axB.plot(f1[band], A1[band], color=C_WT1, lw=1.3, label="WT1 SS")
    axB.plot(f2[band], A2[band], color=C_WT2, lw=1.3, ls=(0, (5, 2)),
             label="WT2 SS")
    axB.axvline(F_WT1, color=C_WT1, lw=0.8, ls=":",
                label="0,234 Hz (WT1 res. / slug)")
    axB.axvline(F_WT2, color=C_WT2, lw=0.8, ls=":", label="0,250 Hz (WT2 res.)")
    axB.set_xlim(0.15, 0.35)
    axB.set_xlabel("Frekvens [Hz]")
    axB.set_ylabel("Amplitude  [m/s²]")
    axB.set_title("b) Spektrum: WT2 svarer ved 0,234 Hz (nett-drevet), "
                  "ikke 0,25 Hz", fontsize=9.5)
    axB.legend(loc="upper right", fontsize=8)

    # C: electrical power injected by each turbine (deviation from its mean)
    axC.plot(t, (p1 - np.mean(p1)) * 1e3, color=C_WT1, lw=1.1, label="WT1")
    axC.plot(t, (p2 - np.mean(p2)) * 1e3, color=C_WT2, lw=1.1, ls=(0, (5, 2)),
             label="WT2")
    axC.axvline(ONSET_S, color="#888888", lw=0.8, ls=":")
    axC.set_xlabel("Tid [s]")
    axC.set_ylabel("Δ elektrisk effekt  [10⁻³ pu]")
    axC.set_title("c) WT1s resonans pumper 0,234 Hz effekt inn i riggnettet",
                  fontsize=9.5)
    axC.legend(loc="upper left", fontsize=8.5)

    # D: shared grid frequency (the channel that couples the turbines)
    axD.plot(t, (fgrid - 50.0) * 1e3, color=C_LOAD, lw=1.1)
    axD.axvline(ONSET_S, color="#888888", lw=0.8, ls=":")
    axD.set_xlabel("Tid [s]")
    axD.set_ylabel("Δ nettfrekvens  [mHz]")
    axD.set_title("d) Delt nettfrekvens — kanalen som kobler turbinene",
                  fontsize=9.5)

    fig.suptitle(
        "To turbiner på LEOGO, felles slug ved 0,234 Hz (WT1s resonans): "
        "kun WT1 ringer opp;\nWT2 (av res.) svarer svakt og ved slug-frekvensen "
        "— altså drevet via nettet, ikke egenresonans", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = FIG_DIR / "2wt_ss_interaction.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\n  Lagret figur: {out}")

    print("\n  Oppsummering (slug @ 0,234 Hz):")
    print(f"   WT1 SS-amplitude (pa res.)  : {a1:.3e} m/s^2")
    print(f"   WT2 SS-amplitude (av res.)  : {a2:.3e} m/s^2")
    print(f"   forhold WT1/WT2             : {ratio:.1f}x")
    print(f"   WT2 dominerende svarfrekvens: {dom_wt2:.3f} Hz "
          f"(slug 0,234 Hz, WT2 egenres. 0,250 Hz)")

    # -- Control comparison: WT1 de-tuned so it no longer resonates ----------
    dfc = pd.read_csv(csv_ctrl)
    tc = dfc["t"].to_numpy()
    ss1_ctrl = dfc["ss_accel_mps2_wt1"].to_numpy()
    ss2_ctrl = dfc["ss_accel_mps2_wt2"].to_numpy()

    a2_main = a2                       # WT2 with WT1 on resonance (from above)
    a2_ctrl = _steady_amp(ss2_ctrl)    # WT2 with WT1 de-tuned
    a1_ctrl = _steady_amp(ss1_ctrl)    # sanity: WT1 should stay quiet now
    add_pct = 100.0 * (a2_main - a2_ctrl) / max(a2_ctrl, 1e-9)

    # Pure WT1 -> WT2 contribution: the extra WT2 motion the resonating WT1
    # adds on top of the common slug (both runs share slug phase and grid).
    ss2_ctrl_i = np.interp(t, tc, ss2_ctrl)
    diff = ss2 - ss2_ctrl_i
    a_diff = _steady_amp(diff)

    fig2, (bx1, bx2) = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True)
    bx1.plot(t, ss1, color="#bbbbbb", lw=1.0,
             label=f"WT1 (på res., referanse)  A={a1:.2e}")
    bx1.plot(t, ss2, color=C_WT2, lw=1.4,
             label=f"WT2 | WT1 på resonans  A={a2_main:.2e}")
    bx1.plot(tc, ss2_ctrl, color=C_WT1, lw=1.4, ls=(0, (5, 2)),
             label=f"WT2 | WT1 av-stemt  A={a2_ctrl:.2e}")
    bx1.axvline(ONSET_S, color="#888888", lw=0.8, ls=":")
    bx1.set_ylabel("Tårn SS-akselerasjon  [m/s²]")
    bx1.set_title(f"a) WT2s respons: WT1 på resonans vs WT1 av-stemt "
                  f"— WT1s resonans endrer WT2 med {add_pct:+.0f}%",
                  fontsize=10)
    bx1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    bx2.plot(t, diff, color="#8e44ad", lw=1.1,
             label=f"WT2(WT1 på res.) − WT2(WT1 av-stemt)  A={a_diff:.2e}")
    bx2.axvline(ONSET_S, color="#888888", lw=0.8, ls=":")
    bx2.set_xlabel("Tid [s]")
    bx2.set_ylabel("Δ SS-akselerasjon  [m/s²]")
    bx2.set_title("b) Ren turbin-til-turbin-forplantning (WT1→WT2 via nettet)",
                  fontsize=10)
    bx2.legend(loc="upper left", fontsize=8.5)

    fig2.suptitle(
        "Kontrollkjøring: WT1 av-stemt til 0,250 Hz så den ikke ringer opp "
        "(slug fortsatt 0,234 Hz).\nDifferansen i WT2s tårnrespons isolerer "
        "WT1→WT2-koblingen gjennom riggnettet", fontsize=11.5)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    out2 = FIG_DIR / "2wt_ss_interaction_control.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    print(f"  Lagret figur: {out2}")

    print("\n  Kontroll (WT1 av-stemt til 0,250 Hz, slug fortsatt 0,234 Hz):")
    print(f"   WT2 SS | WT1 pa resonans : {a2_main:.3e} m/s^2")
    print(f"   WT2 SS | WT1 av-stemt    : {a2_ctrl:.3e} m/s^2")
    print(f"   WT1 SS i kontroll (sjekk): {a1_ctrl:.3e} m/s^2 (skal vaere lav)")
    print(f"   WT1-resonans endrer WT2  : {add_pct:+.1f}%")
    print(f"   ren WT1->WT2 differanse  : {a_diff:.3e} m/s^2")

    # -- Two-way coupling: does the reciprocal SS<->drivetrain path open a
    #    genuine WT1 -> WT2 channel that the forward-only model lacks? --------
    if args.feedback:
        csv_fb_main = CSV_DIR / "2wt_detuning_fb_main.csv"
        csv_fb_ctrl = CSV_DIR / "2wt_detuning_fb_ctrl.csv"
        print(f"\n  Toveis kobling (c={C_FB:g}): WT1 basis f_ss={F_WT1_FB:.4f} "
              f"-> koblet 0,234 Hz; WT2 basis f_ss={F_WT2_FB:.4f} "
              f"-> koblet 0,250 Hz")
        # Two-way main: WT1 resonates (coupled 0.234), WT2 off (coupled 0.250).
        _run_one(F_WT1, F_WT1_FB, F_WT2_FB, csv_fb_main, args.force,
                 fb_c=C_FB, fb_target="both")
        # Two-way control: WT1 de-tuned to 0.250 (coupled) so it stays quiet.
        _run_one(F_WT1, F_WT2_FB, F_WT2_FB, csv_fb_ctrl, args.force,
                 fb_c=C_FB, fb_target="both")

        dfm = pd.read_csv(csv_fb_main)
        dfk = pd.read_csv(csv_fb_ctrl)
        tm = dfm["t"].to_numpy()
        ss1_fb = dfm["ss_accel_mps2_wt1"].to_numpy()
        ss2_fb = dfm["ss_accel_mps2_wt2"].to_numpy()
        ss2_fb_ctrl = np.interp(tm, dfk["t"].to_numpy(),
                                dfk["ss_accel_mps2_wt2"].to_numpy())

        a1_fb = _steady_amp(ss1_fb)          # WT1 still resonates at 0.234
        diff_fb = ss2_fb - ss2_fb_ctrl       # two-way WT1 -> WT2 path
        a_diff_fb = _steady_amp(diff_fb)

        ff, Ad = _amp_spectrum(tm, diff_fb, t0)
        bnd = ff <= 0.5
        dom_diff = float(ff[bnd][np.argmax(Ad[bnd])])

        fig3, (cx1, cx2) = plt.subplots(2, 1, figsize=(11, 7.4))
        cx1.plot(t, diff, color="#9aa0a6", lw=1.2,
                 label=f"forover-koblet (dagens modell)  A={a_diff:.2e}")
        cx1.plot(tm, diff_fb, color="#8e44ad", lw=1.3,
                 label=f"toveis kobling c={C_FB:g}  A={a_diff_fb:.2e}")
        cx1.axvline(ONSET_S, color="#888888", lw=0.8, ls=":")
        cx1.set_xlabel("Tid [s]")
        cx1.set_ylabel("WT1→WT2  Δ SS-akselerasjon  [m/s²]")
        cx1.set_title("a) Ren WT1→WT2-forplantning: forover-koblet (≈ 0) vs "
                      "toveis (liten, men ≠ 0)", fontsize=9.5)
        cx1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

        cx2.plot(ff[bnd], Ad[bnd], color="#8e44ad", lw=1.3)
        cx2.axvline(F_WT1, color=C_WT1, lw=0.8, ls=":",
                    label="0,234 Hz (WT1s resonans)")
        cx2.set_xlim(0.15, 0.35)
        cx2.set_xlabel("Frekvens [Hz]")
        cx2.set_ylabel("Amplitude  [m/s²]")
        cx2.set_title(f"b) Spektrum av WT1→WT2-signalet — topp ved "
                      f"{dom_diff:.3f} Hz (WT1s resonans forplanter seg)",
                      fontsize=9.5)
        cx2.legend(loc="upper right", fontsize=8)

        fig3.suptitle(
            "Toveis tårn-SS ↔ drivverk-kobling åpner en ekte "
            "turbin-til-turbin-vei.\nForover-koblet modell: WT1→WT2 = 0. "
            f"Toveis (c={C_FB:g}): liten, men målbar kobling ved WT1s "
            "resonansfrekvens (Type-4-omformer demper den)", fontsize=11)
        fig3.tight_layout(rect=(0, 0, 1, 0.94))
        out3 = FIG_DIR / "2wt_ss_interaction_feedback.png"
        fig3.savefig(out3, dpi=200, bbox_inches="tight")
        print(f"  Lagret figur: {out3}")

        print("\n  Toveis vs forover (ren WT1->WT2 differanse):")
        print(f"   WT1 SS (toveis, pa res.)   : {a1_fb:.3e} m/s^2 "
              f"(forover {a1:.3e})")
        print(f"   WT1->WT2 forover-koblet    : {a_diff:.3e} m/s^2 (~0)")
        print(f"   WT1->WT2 toveis c={C_FB:g}      : {a_diff_fb:.3e} m/s^2 (!= 0)")
        print(f"   toveis dominerende frekvens: {dom_diff:.3f} Hz "
              f"(WT1s resonans 0,234 Hz)")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
