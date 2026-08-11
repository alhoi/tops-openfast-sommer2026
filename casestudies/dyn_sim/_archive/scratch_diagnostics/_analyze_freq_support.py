"""Compare a GT-trip with WT frequency support OFF vs ON (headline scenario).

Overlays the two runs produced by test_WT_LEOGO_tower_sim.py (same GT-trip, same
de-loaded operating point; only the WT frequency-support droop differs) and
quantifies how much the support action amplifies the side-to-side (SS) tower
loading. This is the frequency-support trade-off: enabling grid support helps
the grid frequency but increases the mechanical loading on the tower.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", default="results/tower_test/gt_trip_support_off.csv")
    ap.add_argument("--on", default="results/tower_test/gt_trip_support_on.csv")
    ap.add_argument("--event-time", type=float, default=5.0)
    ap.add_argument("--event-duration", type=float, default=50.0)
    ap.add_argument("--out", default="results/tower_test/gt_trip_support.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    off = pd.read_csv(args.off)
    on = pd.read_csv(args.on)

    t = off["t"].to_numpy()
    post = t >= args.event_time
    # Transient window: the fast frequency response right after the load step.
    trans = (t >= args.event_time) & (t <= args.event_time + 12.0)
    # Plateau window: sustained load-on interval just before the load clears
    # (this is where the droop's steady frequency benefit shows).
    t_off = args.event_time + args.event_duration
    plateau = (t >= t_off - 15.0) & (t <= t_off - 1.0)

    def stats(df):
        ss = df["ss_accel_mps2"].to_numpy()
        fa = df["fa_accel_mps2"].to_numpy()
        f = df["grid_freq_hz"].to_numpy()
        p = df["P_uic_bus_sys_pu"].to_numpy()
        return {
            "ss_pk": float(np.max(np.abs(ss[post]))),
            "ss_rms": rms(ss[post]),
            "fa_pk": float(np.max(np.abs(fa[post]))),
            "f_dip_trans": 50.0 - float(np.min(f[trans])),
            "f_dip_plateau": 50.0 - float(np.mean(f[plateau])),
            "p_peak": float(np.max(p[post])),
            "p_plateau": float(np.mean(p[plateau])),
        }

    s_off = stats(off)
    s_on = stats(on)

    print("=" * 66)
    print("HEADLINE: LOAD STEP  --  WT FREQUENCY SUPPORT  OFF  vs  ON")
    print("=" * 66)
    print(f"{'':26s}{'OFF':>12s}{'ON':>12s}{'ratio ON/OFF':>16s}")
    print("-" * 66)
    print(f"{'Transient freq dip [Hz]':26s}{s_off['f_dip_trans']:>12.4f}"
          f"{s_on['f_dip_trans']:>12.4f}"
          f"{s_on['f_dip_trans']/max(s_off['f_dip_trans'],1e-9):>16.3f}")
    print(f"{'Plateau freq dip [Hz]':26s}{s_off['f_dip_plateau']:>12.4f}"
          f"{s_on['f_dip_plateau']:>12.4f}"
          f"{s_on['f_dip_plateau']/max(s_off['f_dip_plateau'],1e-9):>16.3f}")
    print(f"{'WT power peak [pu sys]':26s}{s_off['p_peak']:>12.4f}{s_on['p_peak']:>12.4f}")
    print(f"{'WT power plateau [pu sys]':26s}{s_off['p_plateau']:>12.4f}{s_on['p_plateau']:>12.4f}")
    print("-" * 66)
    print(f"{'SS accel peak [m/s2]':26s}{s_off['ss_pk']:>12.4f}{s_on['ss_pk']:>12.4f}"
          f"{s_on['ss_pk']/max(s_off['ss_pk'],1e-9):>16.2f}")
    print(f"{'SS accel RMS  [m/s2]':26s}{s_off['ss_rms']:>12.4f}{s_on['ss_rms']:>12.4f}"
          f"{s_on['ss_rms']/max(s_off['ss_rms'],1e-9):>16.2f}")
    print(f"{'FA accel peak [m/s2]':26s}{s_off['fa_pk']:>12.4f}{s_on['fa_pk']:>12.4f}")
    print("=" * 66)
    print(f"Trade-off: WT frequency support cuts the plateau freq dip "
          f"{100*(1 - s_on['f_dip_plateau']/max(s_off['f_dip_plateau'],1e-9)):.0f} %, "
          f"but rings the SS tower mode "
          f"{s_on['ss_rms']/max(s_off['ss_rms'],1e-9):.1f}x harder (RMS).")
    print("=" * 66)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(t, off["grid_freq_hz"], color="tab:blue", label="support OFF")
    ax[0].plot(t, on["grid_freq_hz"], color="tab:red", label="support ON")
    ax[0].axvline(args.event_time, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[0].set_ylabel("Grid freq [Hz]")
    ax[0].set_title("Temporary load step: WT frequency support OFF vs ON")
    ax[0].legend(loc="upper right")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, off["P_uic_bus_sys_pu"], color="tab:blue", label="support OFF")
    ax[1].plot(t, on["P_uic_bus_sys_pu"], color="tab:red", label="support ON")
    ax[1].axvline(args.event_time, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[1].set_ylabel("WT power [pu sys]")
    ax[1].legend(loc="upper right")
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(t, off["ss_accel_mps2"], color="tab:blue", label="SS accel OFF")
    ax[2].plot(t, on["ss_accel_mps2"], color="tab:red", label="SS accel ON")
    ax[2].axvline(args.event_time, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[2].set_ylabel("SS accel [m/s$^2$]")
    ax[2].set_xlabel("Time [s]")
    ax[2].legend(loc="upper right")
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Figure written to: {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
