from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# LEOGO system base [MVA]; converts system-pu power to MW.
SYS_MVA = 100.0


def _nearest(df, t_query):
    """Return the row of df whose time column is closest to t_query."""
    idx = int((df['t'] - t_query).abs().idxmin())
    return df.loc[idx]


def _summary(df, event_time, settle_window):
    """Print a short before/after comparison for both turbines."""
    t_end = float(df['t'].max())
    pre = _nearest(df, max(0.0, event_time - 1.0))
    settle = df[df['t'] >= t_end - settle_window]

    print('\n=== To-turbin interaksjon: sammendrag ===')
    for wt in ('wt1', 'wt2'):
        om = df[f'omega_m_pu_{wt}']
        p_mw = df[f'P_uic_bus_sys_pu_{wt}'] * SYS_MVA
        v = df[f'V_LV_pu_{wt}']
        print(f'\n{wt.upper()}:')
        print(f"  vind:            {pre[f'wind_speed_mps_{wt}']:.1f} m/s")
        print(f"  omega_m foer:    {pre[f'omega_m_pu_{wt}']:.5f} pu")
        print(f"  omega_m min/max: {om.min():.5f} / {om.max():.5f} pu")
        print(f"  omega_m svingn.: {1e3 * (om.max() - om.min()):.3f} mpu (topp-topp)")
        print(f"  |V| min:         {v.min():.4f} pu")
        print(f"  P foer:          {pre[f'P_uic_bus_sys_pu_{wt}'] * SYS_MVA:.3f} MW")
        print(f"  P min/max:       {p_mw.min():.3f} / {p_mw.max():.3f} MW")
        print(f"  omega_m settlet: {settle[f'omega_m_pu_{wt}'].mean():.5f} pu")


def make_figure(df, event_time, out_path, show, gust_time=None, title=None):
    t = df['t'].to_numpy()

    fig, axes = plt.subplots(8, 1, figsize=(10.5, 18.0), sharex=True)

    c1 = '#1f5fbf'   # WT1 (droop paa)
    c2 = '#c1272d'   # WT2 (droop av)

    # Panel 0: grid (centre-of-inertia) frequency
    ax = axes[0]
    ax.plot(t, df['grid_freq_hz'], color='0.25', lw=1.4)
    ax.set_ylabel('Nettfrekvens\n[Hz]')
    ax.set_title(title or 'To-turbin respons: WT1 (droop på) vs WT2 (uten droop)',
                 fontsize=10.5)

    # Panel 1: wind speed (the WT1 gust vs constant WT2)
    ax = axes[1]
    ax.plot(t, df['wind_speed_mps_wt1'], color=c1, lw=1.4, label='WT1 (droop på)')
    ax.plot(t, df['wind_speed_mps_wt2'], color=c2, lw=1.4, label='WT2 (droop av)')
    ax.set_ylabel('Vindhastighet\n[m/s]')
    ax.legend(loc='best', fontsize=8)

    # Panel 2: rotor speed (mechanical)
    ax = axes[2]
    ax.plot(t, df['omega_m_pu_wt1'], color=c1, lw=1.4)
    ax.plot(t, df['omega_m_pu_wt2'], color=c2, lw=1.4)
    ax.set_ylabel('Rotorhastighet\nω_m [pu]')

    # Panel 3: torsional twist (omega_m - omega_e) - the excited drivetrain mode
    ax = axes[3]
    ax.plot(t, 1e3 * (df['omega_m_pu_wt1'] - df['omega_e_pu_wt1']), color=c1, lw=1.2)
    ax.plot(t, 1e3 * (df['omega_m_pu_wt2'] - df['omega_e_pu_wt2']), color=c2, lw=1.2)
    ax.set_ylabel('Torsjon\nω_m−ω_e [mpu]')

    # Panel 4: electrical power (solid) vs droop power reference P_ref (dashed)
    ax = axes[4]
    ax.plot(t, df['P_uic_bus_sys_pu_wt1'] * SYS_MVA, color=c1, lw=1.3, label='WT1 P')
    ax.plot(t, df['P_uic_bus_sys_pu_wt2'] * SYS_MVA, color=c2, lw=1.3, label='WT2 P')
    if 'P_ref_sys_pu_wt1' in df.columns:
        ax.plot(t, df['P_ref_sys_pu_wt1'] * SYS_MVA, color=c1, lw=1.0, ls='--',
                label='WT1 P_ref (droop)')
        ax.plot(t, df['P_ref_sys_pu_wt2'] * SYS_MVA, color=c2, lw=1.0, ls='--',
                label='WT2 P_ref')
    ax.set_ylabel('Effekt\n[MW]')
    ax.legend(loc='best', fontsize=7, ncol=2)

    # Panel 5: tower side-to-side acceleration (grid/torque -> SS E->M path)
    ax = axes[5]
    if 'ss_accel_mps2_wt1' in df.columns:
        ax.plot(t, df['ss_accel_mps2_wt1'], color=c1, lw=1.0)
        ax.plot(t, df['ss_accel_mps2_wt2'], color=c2, lw=1.0)
    ax.set_ylabel('Tårn side-side\na_ss [m/s²]')

    # Panel 6: tower fore-aft acceleration (wind/thrust -> FA path)
    ax = axes[6]
    if 'fa_accel_mps2_wt1' in df.columns:
        ax.plot(t, df['fa_accel_mps2_wt1'], color=c1, lw=1.0)
        ax.plot(t, df['fa_accel_mps2_wt2'], color=c2, lw=1.0)
    ax.set_ylabel('Tårn for-akter\na_fa [m/s²]')

    # Panel 7: terminal voltage at each LV bus
    ax = axes[7]
    ax.plot(t, df['V_LV_pu_wt1'], color=c1, lw=1.2)
    ax.plot(t, df['V_LV_pu_wt2'], color=c2, lw=1.2)
    ax.set_ylabel('Klemmespenning\n|V| [pu]')
    ax.set_xlabel('Tid [s]')

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.axvline(event_time, color='k', ls='--', lw=0.9, alpha=0.7)
        if gust_time is not None:
            ax.axvline(gust_time, color='#2a8a3e', ls=':', lw=1.0, alpha=0.8)
    axes[0].annotate('nett-hendelse', xy=(event_time, 0.5),
                     xycoords=('data', 'axes fraction'),
                     xytext=(6, 0), textcoords='offset points',
                     fontsize=8, color='k', va='center')
    if gust_time is not None:
        axes[1].annotate('vindkast', xy=(gust_time, 0.5),
                         xycoords=('data', 'axes fraction'),
                         xytext=(6, 0), textcoords='offset points',
                         fontsize=8, color='#2a8a3e', va='center')

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f'\nFigur lagret: {out_path}')
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True,
                        help='Path to WT_LEOGO_2wt_results.csv')
    parser.add_argument('--event-time', type=float, default=5.0)
    parser.add_argument('--gust-time', type=float, default=None,
                        help='WT1 wind-gust start time [s] to mark (optional).')
    parser.add_argument('--title', default=None,
                        help='Optional figure title override.')
    parser.add_argument('--settle-window', type=float, default=5.0)
    parser.add_argument('--out', default=None)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    out_path = Path(args.out) if args.out else csv_path.with_name('2wt_interaction.png')

    _summary(df, args.event_time, args.settle_window)
    make_figure(df, args.event_time, out_path, args.show, gust_time=args.gust_time,
                title=args.title)


if __name__ == '__main__':
    main()
