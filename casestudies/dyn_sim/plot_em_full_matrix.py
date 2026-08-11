r"""
Analyse and plot the OpenFAST electromechanical-interaction full-matrix sweep
produced by sweep_em_full_matrix.py.

  ss         settled tower side-to-side acceleration vs process-load frequency
             (with fore-aft overlay), droop OFF vs ON  ->  resonance curve + Q.
  torsion    settled shaft-torque amplitude vs process-load frequency.
  stability  post-event generator-torque RMS as a droop-gain x freq-LPF map
             (bright = the ~5.3 Hz LEOGO mode pumped into the torque).

Usage:
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_em_full_matrix.py
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
OUT_ROOT = PROJECT_ROOT / "results" / "em_interaction_sweep" / "full_matrix"


def lockin(t, sig, f, t_lo, t_hi) -> float:
    """Single-frequency amplitude of sig at f over [t_lo, t_hi] (quadrature)."""
    m = (t >= t_lo) & (t <= t_hi)
    tt, s = t[m], np.asarray(sig[m], dtype=float)
    if tt.size < 8:
        return float("nan")
    s = s - s.mean()
    w = 2.0 * np.pi * f
    c = np.trapezoid(s * np.cos(w * tt), tt)
    q = np.trapezoid(s * np.sin(w * tt), tt)
    span = tt[-1] - tt[0]
    return 2.0 * np.hypot(c, q) / span if span > 0 else float("nan")


def _f_from_tag(name: str) -> float:
    m = re.search(r"_f([0-9]+p[0-9]+)_", name)
    return float(m.group(1).replace("p", ".")) if m else float("nan")


def _resonance_curve(block: str, signal: str, fa_signal: str | None,
                     title: str, xlabel: str, ylabel: str, out_png: Path) -> None:
    folder = OUT_ROOT / block
    files = sorted(folder.glob(f"{'ss' if block == 'ss' else 'tors'}_f*.csv"))
    if not files:
        print(f"[{block}] no CSVs in {folder}, skipping.")
        return

    data: dict[str, list[tuple[float, float, float]]] = {"off": [], "on": []}
    for fp in files:
        sup = "on" if fp.stem.endswith("_on") else "off"
        f = _f_from_tag(fp.stem)
        df = pd.read_csv(fp)
        if signal not in df or "t" not in df:
            continue
        t = df["t"].to_numpy()
        t_lo, t_hi = 0.6 * t.max(), t.max()      # settled tail
        a = lockin(t, df[signal].to_numpy(), f, t_lo, t_hi)
        a_fa = (lockin(t, df[fa_signal].to_numpy(), f, t_lo, t_hi)
                if fa_signal and fa_signal in df else float("nan"))
        data[sup].append((f, a, a_fa))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for sup, colour in (("off", "#1f77b4"), ("on", "#d62728")):
        pts = sorted(data[sup])
        if not pts:
            continue
        fs = [p[0] for p in pts]
        amp = [p[1] for p in pts]
        ax.plot(fs, amp, "o-", color=colour, label=f"support {sup}")
    # fore-aft reference (support off) to show selectivity
    pts = sorted(data["off"])
    if pts and np.isfinite([p[2] for p in pts]).any():
        ax.plot([p[0] for p in pts], [p[2] for p in pts], "s--",
                color="#7f7f7f", alpha=0.7, label="fore-aft (off)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[{block}] saved {out_png}")


def _stability_map() -> None:
    folder = OUT_ROOT / "stability"
    files = sorted(folder.glob("stab_d*_lpf*.csv"))
    if not files:
        print("[stability] no CSVs, skipping.")
        return

    droops, lpfs, cells = set(), set(), {}
    for fp in files:
        m = re.search(r"stab_d([0-9emp]+)_lpf([0-9p]+)", fp.stem)
        if not m:
            continue
        droop = float(m.group(1).replace("p", ".").replace("m", "-"))
        lpf = float(m.group(2).replace("p", "."))
        df = pd.read_csv(fp)
        if "fmu_GenTq" not in df or "t" not in df:
            continue
        t = df["t"].to_numpy()
        sig = df["fmu_GenTq"].to_numpy().astype(float)
        mask = t > 12.0                              # after the load kick
        seg = sig[mask]
        rms = float(np.std(seg - seg.mean())) if seg.size > 8 else float("nan")
        droops.add(droop)
        lpfs.add(lpf)
        cells[(droop, lpf)] = rms

    droops = sorted(droops)
    lpfs = sorted(lpfs)
    grid = np.full((len(droops), len(lpfs)), np.nan)
    for i, d in enumerate(droops):
        for j, l in enumerate(lpfs):
            grid[i, j] = cells.get((d, l), np.nan)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks(range(len(lpfs)))
    ax.set_xticklabels([f"{l:g}" for l in lpfs])
    ax.set_yticks(range(len(droops)))
    ax.set_yticklabels([f"{d:.0e}" for d in droops])
    ax.set_xlabel("frequency-LPF corner [Hz]  (0 = off)")
    ax.set_ylabel("droop gain [Nm/Hz]")
    ax.set_title("Post-event generator-torque RMS\n"
                 "(bright = ~5.3 Hz LEOGO mode pumped into the torque)")
    for i in range(len(droops)):
        for j in range(len(lpfs)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2e}", ha="center", va="center",
                        color="w", fontsize=7)
    fig.colorbar(im, ax=ax, label="GenTq RMS [Nm]")
    fig.tight_layout()
    out_png = OUT_ROOT / "stability" / "stability_map.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[stability] saved {out_png}")


def main() -> None:
    _resonance_curve(
        "ss", "fmu_YawBrTAyp", "fmu_YawBrTAxp",
        "OpenFAST tower side-to-side resonance (grid process load)",
        "Process-load frequency [Hz]",
        "Settled side-to-side acceleration\nat drive frequency  [m/s$^2$]",
        OUT_ROOT / "ss" / "ss_resonance_curve.png")
    _resonance_curve(
        "torsion", "fmu_HSShftTq", None,
        "OpenFAST drivetrain torsion resonance (grid process load)",
        "Process-load frequency [Hz]",
        "Settled shaft-torque amplitude\nat drive frequency  [kN$\\cdot$m]",
        OUT_ROOT / "torsion" / "torsion_resonance_curve.png")
    _stability_map()


if __name__ == "__main__":
    main()
