"""Standalone ZeroMQ REP server for ROSCO's ZMQ interface (ROSCO 2.10.1).

ROSCO (compiled with ZMQ_CLIENT) connects as a REQ client to ZMQ_CommAddress
(default tcp://localhost:5555) every ZMQ_UpdatePeriod and sends 17 turbine
measurements as a comma-separated "%.6e" string:

  [0] ZMQ_ID   [1] iStatus   [2] Time        [3] VS_MechGenPwr [4] VS_GenPwr
  [5] GenSpeed [6] RotSpeed  [7] GenTqMeas   [8] NacHeading    [9] NacVane
  [10] HorWindV [11..13] rootMOOP1..3 [14] FA_Acc_TT [15] NacIMU_FA_RAcc [16] Azimuth

and expects 8 setpoints back (comma-separated), applied by ROSCO as:

  [0] TorqueOffset  [1] YawOffset  [2..4] PitOffset1..3
  [5] R_Speed       [6] R_Torque   [7] R_Pitch

This proof-of-concept replies with a sinusoidal GENERATOR-TORQUE OFFSET at a
chosen frequency (default the tower side-to-side mode, 0.234 Hz), R_* = 1.0 and
all other offsets 0. It logs (time, torque_offset_sent, GenTqMeas, GenSpeed,
RotSpeed) so the generator-torque response can be checked directly.

Start this FIRST (it binds the socket), then launch the co-simulation driver.
The server exits automatically once ROSCO stops sending (idle timeout).
"""

import argparse
import csv
import math
from pathlib import Path

import zmq

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--amp-nm", type=float, default=1.1e6,
                   help="Torque-offset amplitude [Nm] (~10%% of Region-2 torque).")
    p.add_argument("--freq-hz", type=float, default=0.234)
    p.add_argument("--start", type=float, default=10.0,
                   help="Sim time [s] at which the offset switches on.")
    p.add_argument("--ramp-s", type=float, default=5.0,
                   help="Smoothstep ramp-on duration [s].")
    p.add_argument("--idle-timeout", type=float, default=20.0,
                   help="Exit after this many seconds without a request.")
    p.add_argument("--out", type=str,
                   default=str(PROJECT_ROOT / "results" / "sweep" / "zmq_server_log.csv"))
    args = p.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.RCVTIMEO, int(args.idle_timeout * 1000))
    sock.bind(f"tcp://*:{args.port}")
    print(f"[zmq-server] bound tcp://*:{args.port}; offset {args.amp_nm:.3e} Nm "
          f"@ {args.freq_hz} Hz from t={args.start}s. Waiting for ROSCO...",
          flush=True)

    rows = []
    n = 0
    while True:
        try:
            msg = sock.recv()
        except zmq.Again:
            print("[zmq-server] idle timeout -> stopping.", flush=True)
            break

        parts = msg.decode("ascii", "ignore").split(",")
        try:
            t = float(parts[2])
            gen_speed = float(parts[5])
            rot_speed = float(parts[6])
            gen_tq = float(parts[7])
        except (IndexError, ValueError):
            t = gen_speed = rot_speed = gen_tq = float("nan")

        # Sinusoidal torque offset with a smoothstep ramp-on (avoids an onset kick).
        if math.isnan(t) or t < args.start:
            offset = 0.0
        else:
            u = min((t - args.start) / args.ramp_s, 1.0) if args.ramp_s > 0 else 1.0
            env = u * u * (3.0 - 2.0 * u)
            offset = args.amp_nm * env * math.sin(
                2.0 * math.pi * args.freq_hz * (t - args.start))

        setpoints = [offset, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        # Null-terminate so ROSCO's C strtok stops cleanly at the end of the reply.
        reply = ",".join(f"{s:.6e}" for s in setpoints).encode("ascii") + b"\x00"
        sock.send(reply)

        rows.append((t, offset, gen_tq, gen_speed, rot_speed))
        n += 1
        if n % 500 == 0:
            print(f"[zmq-server] t={t:8.2f}s  offset={offset:+.3e} Nm  "
                  f"GenTqMeas={gen_tq:.3e} Nm", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "torque_offset_nm", "gen_tq_meas_nm",
                    "gen_speed_rpm", "rot_speed_rpm"])
        w.writerows(rows)
    print(f"[zmq-server] wrote {len(rows)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
