"""Generate a uniform (.hh) wind file with a sinusoidal wind-DIRECTION oscillation.

Alternative 1 for exciting the side-to-side tower mode: an oscillating yaw
misalignment produces a lateral aerodynamic force on the rotor that forces the
SS mode directly (bypasses the ROSCO-owned generator-torque channel).

Columns (OpenFAST InflowWind uniform format):
    Time  HorSpd  WndDir  VerSpd  HorShr  VerShr  LnVShr  GstSpd  Upflow
"""
import argparse
import numpy as np
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mean-speed", type=float, default=8.0, help="Mean horizontal wind speed (m/s)")
    p.add_argument("--dir-amplitude", type=float, default=4.0, help="Wind-direction oscillation amplitude (deg)")
    p.add_argument("--freq-hz", type=float, default=0.234, help="Oscillation frequency (Hz)")
    p.add_argument("--start", type=float, default=0.0, help="Time to start the oscillation (s)")
    p.add_argument("--t-end", type=float, default=900.0, help="End time (s)")
    p.add_argument("--dt", type=float, default=0.05, help="Time resolution of the wind file (s)")
    p.add_argument("--vert-shear", type=float, default=0.14, help="Vertical (power-law) shear exponent")
    p.add_argument("--chirp", action="store_true",
                   help="Linear frequency sweep (chirp) instead of a fixed-frequency sine.")
    p.add_argument("--f0", type=float, default=0.10, help="Chirp start frequency (Hz)")
    p.add_argument("--f1", type=float, default=2.00, help="Chirp end frequency (Hz)")
    p.add_argument("--out", type=str,
                   default="test1002/WindData/winddir_sine_0p234Hz.hh",
                   help="Output .hh file path (relative to repo root)")
    args = p.parse_args()

    t = np.arange(0.0, args.t_end + 0.5 * args.dt, args.dt)
    tau = np.maximum(t - args.start, 0.0)
    if args.chirp:
        t_sweep = max(args.t_end - args.start, 1e-6)
        # Linear instantaneous frequency f(t) = f0 + (f1-f0)*tau/t_sweep,
        # phase = 2*pi*(f0*tau + 0.5*(f1-f0)/t_sweep * tau^2).
        phase = 2.0 * np.pi * (args.f0 * tau + 0.5 * (args.f1 - args.f0) / t_sweep * tau ** 2)
        wnd_dir = np.where(t >= args.start, args.dir_amplitude * np.sin(phase), 0.0)
    else:
        wnd_dir = np.where(
            t >= args.start,
            args.dir_amplitude * np.sin(2.0 * np.pi * args.freq_hz * tau),
            0.0,
        )
    hor_spd = np.full_like(t, args.mean_speed)
    ver_spd = np.zeros_like(t)
    hor_shr = np.zeros_like(t)
    ver_shr = np.full_like(t, args.vert_shear)
    lnv_shr = np.zeros_like(t)
    gst_spd = np.zeros_like(t)
    upflow = np.zeros_like(t)

    out_path = Path(__file__).resolve().parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "! Uniform wind file with sinusoidal wind-DIRECTION oscillation.\n"
        f"! Mean speed {args.mean_speed} m/s, dir amplitude {args.dir_amplitude} deg, "
        + (f"CHIRP {args.f0}->{args.f1} Hz" if args.chirp else f"freq {args.freq_hz} Hz")
        + f", start {args.start} s (Alternative 1: excite SS mode via lateral aero force).\n"
        "!\n"
        "!       Time          HorSpd           WndDir          VerSpd          HorShr          "
        "VerShr          LnVShr          GstSpd          Upflow\n"
        "!      (sec)          (m/s)            (deg)           (m/s)           (-)             "
        "(-)             (-)            (m/s)            (deg)\n"
    )

    with open(out_path, "w", newline="\n") as f:
        f.write(header)
        for i in range(len(t)):
            f.write(
                f"  {t[i]:14.6E}  {hor_spd[i]:14.6E}  {wnd_dir[i]:14.6E}  {ver_spd[i]:14.6E}  "
                f"{hor_shr[i]:14.6E}  {ver_shr[i]:14.6E}  {lnv_shr[i]:14.6E}  {gst_spd[i]:14.6E}  "
                f"{upflow[i]:14.6E}\n"
            )

    print(f"Wrote {len(t)} rows to {out_path}")
    print(f"WndDir range: {wnd_dir.min():.3f} .. {wnd_dir.max():.3f} deg")


if __name__ == "__main__":
    main()
