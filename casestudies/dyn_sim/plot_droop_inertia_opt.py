r"""
Analyse the droop + virtual-inertia optimisation sweep (sweep_droop_inertia_opt.py).

main block -> two heatmaps over (K_droop, K_inertia): frequency nadir and RoCoF,
              a Pareto scatter (RoCoF vs nadir, marker size = peak WT power), and a
              ranked table flagging configs that over-rate the 15 MW turbine.
lpf block  -> RoCoF and the 4-7 Hz generator-torque RMS (the ~5.3 Hz LEOGO mode
              pumped into the torque) vs the frequency-LPF corner: the
              inertia-responsiveness vs mode-rejection trade-off.

Usage:
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_droop_inertia_opt.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "results" / "em_interaction_sweep" / "droop_inertia_opt"
EVENT_TIME = 20.0
RATING_MW = 15.0


def metrics(df: pd.DataFrame) -> dict:
    t = df["t"].to_numpy()
    f = df["f_grid_hz"].to_numpy()
    pre = f[(t >= EVENT_TIME - 5) & (t < EVENT_TIME)].mean()
    win = (t >= EVENT_TIME) & (t <= EVENT_TIME + 2.0)
    rocof = float(np.max(np.abs(np.gradient(f[win], t[win])))) * 1000.0 if win.sum() > 3 else np.nan
    post = t >= EVENT_TIME
    nadir = (float(np.min(f[post])) - pre) * 1000.0
    settled = (f[t >= t.max() - 8.0].mean() - pre) * 1000.0
    peak_mw = 100.0 * df["P_e_sys_pu"][post].max() if "P_e_sys_pu" in df else np.nan
    # 4-7 Hz generator-torque RMS after the event (the ~5.3 Hz mode content)
    band = np.nan
    if "fmu_GenTq" in df:
        m = t >= EVENT_TIME + 2
        tt, s = t[m], df["fmu_GenTq"].to_numpy()[m]
        if tt.size > 32:
            dt = np.median(np.diff(tt))
            s = s - s.mean()
            fr = np.fft.rfftfreq(s.size, dt)
            amp = np.abs(np.fft.rfft(s)) / s.size * 2.0
            sel = (fr >= 4.0) & (fr <= 7.0)
            band = float(np.sqrt(np.sum(amp[sel] ** 2)))
    return dict(rocof=rocof, nadir=nadir, settled=settled,
                peak_mw=peak_mw, band53=band)


def _num(s: str) -> float:
    return float(s.replace("p", ".").replace("m", "-"))


def analyse_main() -> None:
    folder = ROOT / "main"
    files = sorted(folder.glob("opt_d*_i*.csv"))
    if not files:
        print("[main] no CSVs, skipping.")
        return
    rows = []
    for fp in files:
        mt = re.search(r"opt_d([0-9emp]+)_i([0-9emp]+)", fp.stem)
        if not mt:
            continue
        kd, ki = _num(mt.group(1)), _num(mt.group(2))
        m = metrics(pd.read_csv(fp))
        rows.append(dict(kd=kd, ki=ki, **m))
    d = pd.DataFrame(rows).sort_values(["kd", "ki"])
    droops = sorted(d.kd.unique())
    inertias = sorted(d.ki.unique())

    def grid(col):
        g = np.full((len(droops), len(inertias)), np.nan)
        for _, r in d.iterrows():
            g[droops.index(r.kd), inertias.index(r.ki)] = r[col]
        return g

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title, cmap in [
        (axes[0], "nadir", "Frequency nadir [mHz]  (higher = shallower)", "viridis"),
        (axes[1], "rocof", "RoCoF [mHz/s]  (lower = gentler)", "magma_r")]:
        g = grid(col)
        im = ax.imshow(g, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(inertias)))
        ax.set_xticklabels([f"{v:.0e}" for v in inertias])
        ax.set_yticks(range(len(droops)))
        ax.set_yticklabels([f"{v:.0e}" for v in droops])
        ax.set_xlabel("K_inertia [Nm.s/Hz]")
        ax.set_ylabel("K_droop [Nm/Hz]")
        ax.set_title(title)
        for i in range(len(droops)):
            for j in range(len(inertias)):
                over = grid("peak_mw")[i, j] > RATING_MW
                ax.text(j, i, f"{g[i, j]:.0f}" + ("*" if over else ""),
                        ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Droop x inertia optimisation (15.8 MW GT trip, de-loaded)\n"
                 "* = WT peak power exceeds 15 MW rating", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ROOT / "opt_heatmaps.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[main] saved {out}")

    # Pareto scatter
    fig2, ax = plt.subplots(figsize=(7.5, 5.5))
    ok = d[d.peak_mw <= RATING_MW]
    bad = d[d.peak_mw > RATING_MW]
    ax.scatter(bad.rocof, bad.nadir, s=40, c="#d62728", marker="x",
               label="over-rating (>15 MW)")
    sc = ax.scatter(ok.rocof, ok.nadir, s=60, c=ok.ki, cmap="viridis",
                    edgecolors="k", label="within rating")
    fig2.colorbar(sc, ax=ax, label="K_inertia [Nm.s/Hz]")
    ax.set_xlabel("RoCoF [mHz/s]  (lower better)")
    ax.set_ylabel("Nadir [mHz]  (higher/less negative better)")
    ax.set_title("Pareto view: RoCoF vs nadir")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(ROOT / "opt_pareto.png", dpi=200, bbox_inches="tight")
    print(f"[main] saved {ROOT / 'opt_pareto.png'}")

    print("\n  K_droop   K_inertia   RoCoF   nadir  settled  peak_MW")
    for _, r in d.iterrows():
        flag = "  OVER-RATING" if r.peak_mw > RATING_MW else ""
        print(f"  {r.kd:.0e}  {r.ki:.0e}   {r.rocof:6.1f}  {r.nadir:6.1f}  "
              f"{r.settled:6.1f}  {r.peak_mw:6.2f}{flag}")


def analyse_lpf() -> None:
    folder = ROOT / "lpf"
    files = sorted(folder.glob("lpf_*.csv"))
    if not files:
        print("[lpf] no CSVs, skipping.")
        return
    rows = []
    for fp in files:
        mt = re.search(r"lpf_([0-9p]+)", fp.stem)
        if not mt:
            continue
        rows.append(dict(lpf=_num(mt.group(1)), **metrics(pd.read_csv(fp))))
    d = pd.DataFrame(rows).sort_values("lpf")

    fig, ax1 = plt.subplots(figsize=(7.5, 5.0))
    ax1.plot(d.lpf, d.rocof, "o-", color="#1f77b4", label="RoCoF")
    ax1.set_xlabel("Frequency-LPF corner [Hz]")
    ax1.set_ylabel("RoCoF [mHz/s]", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(d.lpf, d.band53, "s--", color="#d62728", label="4-7 Hz GenTq RMS")
    ax2.set_ylabel("4-7 Hz GenTq RMS [Nm]  (~5.3 Hz pumping)", color="#d62728")
    ax1.set_title("LPF trade-off: inertia responsiveness vs 5.3 Hz mode rejection")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    out = ROOT / "opt_lpf_tradeoff.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[lpf] saved {out}")
    print("\n  LPF[Hz]   RoCoF   4-7Hz-RMS")
    for _, r in d.iterrows():
        print(f"  {r.lpf:5.1f}   {r.rocof:6.1f}   {r.band53:8.1f}")


def main() -> None:
    analyse_main()
    analyse_lpf()


if __name__ == "__main__":
    main()
