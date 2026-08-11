r"""
UIC-control parameter sweep on the coupled WT + LEOGO model.

Motivation
----------
The interaction analysis (interaction_WT_LEOGO.py) showed that at nominal
tuning the wind turbine and the LEOGO grid are modally decoupled: the only
grid-coupled turbine mode is the heavily damped ~1.445 Hz converter/genset
mode (UIC vi_x + generator e_q_st, zeta ~ 96 %), and the drivetrain torsional
mode (3.49 Hz, zeta = 4.37 %) is WT-isolated.

This script asks the follow-up question directly: how does the damping (and
frequency) of the grid-coupled converter mode move as we retune the UIC grid-
side converter, and can any converter/grid mode be driven toward the 3.49 Hz
torsional frequency (which would open a genuine grid <-> drivetrain resonance)?

For every parameter value the coupled model is rebuilt, linearised, and three
modes are tracked by their dominant participation (robust to frequency shifts):

    torsional : max participation on WT drivetrain omega_e / theta_e (3.49 Hz)
    vi_x mode : max participation on UIC vi_x   (the grid-coupled ~1.445 Hz mode)
    vi_y mode : max participation on UIC vi_y   (the converter ~7.998 Hz mode)

Outputs a CSV and three PNGs (damping vs param, frequency vs param, s-plane
trajectories).  Headless.

Run (default sweeps the filter reactance xf):
    .\.venv\Scripts\python.exe casestudies\modal_analysis\param_sweep_uic_coupling_WT_LEOGO.py
    .\.venv\Scripts\python.exe casestudies\modal_analysis\param_sweep_uic_coupling_WT_LEOGO.py --param Ki
    .\.venv\Scripts\python.exe casestudies\modal_analysis\param_sweep_uic_coupling_WT_LEOGO.py --param T_filter
"""

from __future__ import annotations

import argparse
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

F_TORSIONAL_HZ = 3.4909

# UIC_sig parameters that can be swept, with (nominal value, default range).
PARAM_SPECS = {
    "xf":       {"nominal": 0.10,  "range": (0.02, 0.30), "label": "UIC filter reactance $x_f$ (pu)"},
    "Ki":       {"nominal": 0.03,  "range": (0.005, 0.12), "label": "UIC current gain $K_i$"},
    "Kv":       {"nominal": 0.00,  "range": (0.0, 0.25),  "label": "UIC voltage gain $K_v$"},
    "T_filter": {"nominal": 0.01,  "range": (0.002, 0.06), "label": "UIC measurement filter $T_{filter}$ (s)"},
}

TRACK = {
    "torsional": {"keys": ("WT1 omega_e", "WT1 theta_e"), "color": "#c0392b",
                  "label": "Torsional (drivetrain)"},
    "vi_x":      {"keys": ("WT1_LEOGO vi_x",),            "color": "#2c6fbb",
                  "label": "Grid-coupled converter (vi$_x$)"},
    "vi_y":      {"keys": ("WT1_LEOGO vi_y",),            "color": "#5f7d6a",
                  "label": "Converter (vi$_y$)"},
}


def state_label(desc) -> str:
    if isinstance(desc, (tuple, list, np.ndarray)):
        return " ".join(str(d) for d in desc)
    return str(desc)


def track_modes(ps, ps_lin) -> dict[str, tuple[float, float, complex]]:
    """Return {name: (f_hz, zeta, lambda)} for each tracked mode, identified by
    the oscillatory mode with the largest participation on its key states."""
    eigs = ps_lin.eigs
    pfs = ps_lin.lev.T * ps_lin.rev
    pfs_abs = np.abs(pfs) / np.max(np.abs(pfs), axis=0)
    state_desc = ps.state_desc

    osc = [i for i, lam in enumerate(eigs) if lam.imag > 1e-6]
    out: dict[str, tuple[float, float, complex]] = {}
    for name, spec in TRACK.items():
        idx = [
            i for i, d in enumerate(state_desc)
            if any(k in state_label(d) for k in spec["keys"])
        ]
        if not idx:
            continue
        best_i, best_part = None, -1.0
        for i in osc:
            part = float(np.sum(pfs_abs[idx, i]))
            if part > best_part:
                best_part, best_i = part, i
        lam = eigs[best_i]
        f_hz = lam.imag / (2.0 * np.pi)
        zeta = -lam.real / abs(lam)
        out[name] = (f_hz, zeta, lam)
    return out


def build_ps(param_name: str, value: float):
    """Build, set the UIC parameter, run power flow + init (matches the
    interaction-analysis operating point: UIC p_ref = 0)."""
    model = build_model()
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)
    ps.vsc["UIC_sig"].par[param_name] = value
    ps.power_flow()
    ps.init_dyn_sim()
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()
    return ps, ps_lin


def run_sweep(param_name: str, values: np.ndarray) -> pd.DataFrame:
    records = []
    for v in values:
        ps, ps_lin = build_ps(param_name, float(v))
        tracked = track_modes(ps, ps_lin)
        row = {param_name: float(v)}
        for name, (f_hz, zeta, lam) in tracked.items():
            row[f"{name}_f_hz"] = f_hz
            row[f"{name}_zeta_pct"] = 100.0 * zeta
            row[f"{name}_real"] = lam.real
            row[f"{name}_imag"] = lam.imag
        records.append(row)
        tors = tracked.get("torsional")
        vix = tracked.get("vi_x")
        print(f"  {param_name}={v:7.4f}  "
              f"torsional {tors[0]:6.3f} Hz / {100*tors[1]:5.2f}%   "
              f"vi_x {vix[0]:6.3f} Hz / {100*vix[1]:6.2f}%")
    return pd.DataFrame.from_records(records)


def make_plots(df: pd.DataFrame, param_name: str) -> None:
    spec = PARAM_SPECS[param_name]
    x = df[param_name].to_numpy()
    nominal = spec["nominal"]

    # 1) damping vs parameter
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, tspec in TRACK.items():
        col = f"{name}_zeta_pct"
        if col in df:
            ax.plot(x, df[col], "-o", ms=3, color=tspec["color"],
                    label=tspec["label"])
    ax.axvline(nominal, color="0.6", ls=":", lw=1.0, label="nominal")
    ax.set_xlabel(spec["label"])
    ax.set_ylabel("Damping ratio $\\zeta$ (%)")
    ax.set_title(f"Modal damping vs {param_name}")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = PLOT_DIR / f"uic_sweep_{param_name}_damping.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")

    # 2) frequency vs parameter
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, tspec in TRACK.items():
        col = f"{name}_f_hz"
        if col in df:
            ax.plot(x, df[col], "-o", ms=3, color=tspec["color"],
                    label=tspec["label"])
    ax.axhline(F_TORSIONAL_HZ, color="#c0392b", ls="--", lw=1.0,
               label="3.49 Hz torsional")
    ax.axvline(nominal, color="0.6", ls=":", lw=1.0)
    ax.set_xlabel(spec["label"])
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Modal frequency vs {param_name}")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = PLOT_DIR / f"uic_sweep_{param_name}_frequency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")

    # 3) s-plane trajectories (color = parameter value)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    norm = plt.Normalize(x.min(), x.max())
    cmap = plt.get_cmap("viridis")
    for name, tspec in TRACK.items():
        re = df.get(f"{name}_real")
        im = df.get(f"{name}_imag")
        if re is None or im is None:
            continue
        ax.plot(re, im, "-", color=tspec["color"], lw=0.8, alpha=0.6)
        ax.scatter(re, im, c=x, cmap=cmap, norm=norm, s=18, zorder=3)
        ax.annotate(tspec["label"], (re.iloc[-1], im.iloc[-1]),
                    fontsize=7, color=tspec["color"])
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Real part (1/s)")
    ax.set_ylabel("Imag part (rad/s)")
    ax.set_title(f"Mode trajectories as {param_name} varies")
    ax.grid(True, ls="--", alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label=param_name)
    fig.tight_layout()
    out = PLOT_DIR / f"uic_sweep_{param_name}_splane.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--param", choices=list(PARAM_SPECS), default="xf",
                    help="UIC_sig parameter to sweep")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--stop", type=float, default=None)
    ap.add_argument("--num", type=int, default=15)
    args = ap.parse_args()

    spec = PARAM_SPECS[args.param]
    lo = args.start if args.start is not None else spec["range"][0]
    hi = args.stop if args.stop is not None else spec["range"][1]
    values = np.linspace(lo, hi, args.num)

    print(f"Sweeping UIC_sig['{args.param}'] over [{lo}, {hi}] "
          f"({args.num} points); nominal = {spec['nominal']}")
    df = run_sweep(args.param, values)

    csv_out = PLOT_DIR / f"uic_sweep_{args.param}_modes.csv"
    df.to_csv(csv_out, index=False)
    print(f"saved {csv_out}")

    make_plots(df, args.param)


if __name__ == "__main__":
    main()
