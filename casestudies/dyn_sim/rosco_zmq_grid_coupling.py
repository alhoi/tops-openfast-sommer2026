"""In-process grid-driven generator-torque coupling for ROSCO over ZeroMQ.

This closes the genuine electrical -> mechanical loop between the LEOGO grid and
the OpenFAST wind turbine WITHOUT rebuilding the FMU.

Background
----------
ROSCO (compiled with the ZMQ client and patched to APPLY ``ZMQ_TorqueOffset``)
runs inside ``fmu.doStep`` as a ZeroMQ REQ client. Every ``ZMQ_UpdatePeriod`` it
connects to ``ZMQ_CommAddress`` (tcp://localhost:5555), sends 17 turbine
measurements and blocks waiting for 8 setpoints back. Setpoint 0 is a
generator-torque offset [Nm] that is added to the demanded generator torque.

``rosco_zmq_server.py`` answered that request with a *prescribed* sinusoid - an
open-loop probe. Here we instead answer with the turbine's **frequency-support
response to the live grid frequency**:

    dT = -K_droop * (f_grid - f_nom)  -  K_inertia * d(f_grid)/dt      [Nm]

so that a real disturbance IN the LEOGO grid (a load step, a GT trip, a
process-load oscillation) changes the grid frequency at the WTG bus, which the
droop/inertia law converts into a generator-torque command, which drives the
drivetrain and tower - and whose changed electrical power flows back into the
islanded grid. A genuine two-way electromechanical interaction.

Sign convention: when the grid frequency drops (f_grid < f_nom) the torque
offset is positive, i.e. the generator extracts more power to support the grid
(fast frequency response / synthetic inertia from rotor kinetic energy).

Threading model
---------------
ROSCO's REQ is issued from inside ``fmu.doStep`` on the main thread, so the REP
socket is serviced from a background daemon thread. The main simulation loop
calls :meth:`update` each macro-step with the freshly solved grid frequency; the
control law runs there (deterministic, no data race), storing a ready-to-send
torque offset that the socket thread simply returns. The socket thread only
does recv -> reply -> log, which is microseconds, so it never stalls ``doStep``.

Start the responder BEFORE ``ps.init_dyn_sim()`` (which primes the FMU and may
already trigger a ROSCO ZMQ exchange), and call :meth:`close` at the end.
"""

from __future__ import annotations

import csv
import math
import threading
from pathlib import Path

import zmq


class GridTorqueResponder:
    """ZeroMQ REP server that streams a grid-frequency-derived torque offset."""

    def __init__(
        self,
        port: int = 5555,
        f_nom_hz: float = 50.0,
        droop_nm_per_hz: float = 2.0e7,
        inertia_nm_s_per_hz: float = 0.0,
        deload_nm: float = 0.0,
        support_start_s: float = 10.0,
        ramp_s: float = 5.0,
        deadband_hz: float = 0.0,
        max_offset_nm: float = 3.0e6,
        max_over_nm: float | None = None,
        freq_lpf_hz: float = 0.5,
        freq_lpf_order: int = 2,
        notch_hz: float = 0.0,
        notch_q: float = 2.0,
        log_path: str | Path | None = None,
    ):
        self._port = int(port)
        self._f_nom = float(f_nom_hz)
        self._droop = float(droop_nm_per_hz)
        self._inertia = float(inertia_nm_s_per_hz)
        self._deload = float(deload_nm)
        self._start = float(support_start_s)
        self._ramp = float(ramp_s)
        self._deadband = float(deadband_hz)
        self._max = float(max_offset_nm)
        # Asymmetric UPPER clip: caps the (positive) inertia burst so WT power
        # does not run far above rating. Defaults to the symmetric +max.
        self._max_over = (float(max_over_nm) if max_over_nm is not None
                          else float(max_offset_nm))
        self._fc = float(freq_lpf_hz)
        self._lpf_order = max(0, int(freq_lpf_order))
        self._notch_f0 = float(notch_hz)
        self._notch_q = max(1e-3, float(notch_q))
        self._log_path = Path(log_path) if log_path else None

        # Shared state (guarded by _lock): the control law writes, the socket
        # thread reads.
        self._offset = 0.0          # current torque offset [Nm]
        self._f = self._f_nom       # last grid frequency [Hz]
        self._t = float("nan")      # last sim time [s]
        self._lock = threading.Lock()

        # Derivative memory (main thread only).
        self._prev_t: float | None = None
        self._prev_f_filt: float | None = None

        # Frequency low-pass filter state (cascaded first-order sections).
        self._lpf_state: list[float] | None = None

        # Notch (band-stop) biquad state [x1, x2, y1, y2].
        self._notch_state: list[float] | None = None

        # Logging (socket thread appends; list append is atomic under the GIL).
        self._rows: list[tuple[float, float, float, float]] = []

        self._stop = threading.Event()
        self._ctx = zmq.Context()
        self._sock = None
        self._thread: threading.Thread | None = None
        self._n_requests = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Bind the socket and launch the background REP thread."""
        self._sock = self._ctx.socket(zmq.REP)
        # Short receive timeout so the thread can poll the stop flag.
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://*:{self._port}")
        self._thread = threading.Thread(
            target=self._serve, name="GridTorqueResponder", daemon=True
        )
        self._thread.start()
        _notch = (f"notch={self._notch_f0:.3f} Hz (Q {self._notch_q:.1f})  "
                  if self._notch_f0 > 0.0 else "")
        print(
            f"[grid-zmq] bound tcp://*:{self._port}  "
            f"droop={self._droop:.3e} Nm/Hz  inertia={self._inertia:.3e} Nm.s/Hz  "
            f"deload={self._deload:.3e} Nm  "
            f"freq-LPF={self._fc:.2f} Hz (order {self._lpf_order})  "
            f"{_notch}"
            f"support from t={self._start}s  |dT|<= {self._max:.3e} Nm",
            flush=True,
        )

    def close(self) -> None:
        """Stop the thread, close the socket and write the log."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sock is not None:
            self._sock.close(0)
        self._ctx.term()
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    ["t_rosco", "torque_offset_nm", "gen_tq_meas_nm", "f_grid_hz"]
                )
                w.writerows(self._rows)
            print(
                f"[grid-zmq] {self._n_requests} requests served; "
                f"wrote {len(self._rows)} rows to {self._log_path}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Control law (main thread, once per macro-step)
    # ------------------------------------------------------------------
    def _envelope(self, t: float) -> float:
        """Smoothstep ramp-in of the support action (0 before support_start)."""
        if t < self._start:
            return 0.0
        if self._ramp > 0.0 and t < self._start + self._ramp:
            u = (t - self._start) / self._ramp
            return u * u * (3.0 - 2.0 * u)
        return 1.0

    def _lpf(self, x: float, dt: float) -> float:
        """Cascaded first-order low-pass on the frequency measurement.

        Removes fast grid content (e.g. the lightly damped ~5 Hz LEOGO
        generator mode) that the high-gain droop would otherwise pump into the
        generator torque. fc<=0 or order<=0 disables it (pass-through).
        """
        if self._fc <= 0.0 or self._lpf_order <= 0:
            return x
        if self._lpf_state is None:
            self._lpf_state = [x] * self._lpf_order
        tau = 1.0 / (2.0 * math.pi * self._fc)
        a = dt / (tau + dt)
        y = x
        for i in range(self._lpf_order):
            self._lpf_state[i] += a * (y - self._lpf_state[i])
            y = self._lpf_state[i]
        return y

    def _notch(self, x: float, dt: float) -> float:
        """Second-order band-stop (notch) on the frequency measurement.

        Rejects a narrow band around ``notch_hz`` (the tower mode) so the
        droop/inertia support does not modulate the generator torque at that
        frequency and pump the tower resonance, while leaving the slow
        frequency-support response (DC/low frequency) at unity gain. RBJ
        biquad, Direct Form I. ``notch_hz<=0`` disables it (pass-through).
        """
        if self._notch_f0 <= 0.0:
            return x
        fs = 1.0 / dt if dt > 0.0 else 100.0
        w0 = 2.0 * math.pi * self._notch_f0 / fs
        cw = math.cos(w0)
        alpha = math.sin(w0) / (2.0 * self._notch_q)
        a0 = 1.0 + alpha
        b0 = 1.0 / a0
        b1 = -2.0 * cw / a0
        b2 = 1.0 / a0
        a1 = -2.0 * cw / a0
        a2 = (1.0 - alpha) / a0
        if self._notch_state is None:
            # Start settled at the current DC so there is no switch-on kick.
            self._notch_state = [x, x, x, x]   # x1, x2, y1, y2
        x1, x2, y1, y2 = self._notch_state
        y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        self._notch_state = [x, x1, y, y1]
        return y

    def update(self, t: float, f_grid_hz: float) -> float:
        """Recompute the torque offset from the live grid frequency.

        Called by the simulation driver each macro-step, just before the FMU
        (and therefore ROSCO's ZMQ request) is advanced. Returns the offset for
        convenience/logging.
        """
        # Time step since the previous update (nominal on the first call).
        if self._prev_t is not None and t > self._prev_t:
            dt = t - self._prev_t
        else:
            dt = 0.01

        # Low-pass filter the grid-frequency measurement before the droop, so a
        # high-gain droop does not amplify fast grid modes into the torque.
        # An optional notch then removes the tower-mode band so the support does
        # not pump the tower resonance.
        f_meas = self._notch(self._lpf(f_grid_hz, dt), dt)
        df = f_meas - self._f_nom

        # Optional deadband (soft): ignore |df| below the threshold.
        if self._deadband > 0.0:
            if abs(df) <= self._deadband:
                df_eff = 0.0
            else:
                df_eff = df - math.copysign(self._deadband, df)
        else:
            df_eff = df

        # Rate of change of the FILTERED grid frequency (synthetic inertia).
        if self._prev_t is not None and t > self._prev_t and self._prev_f_filt is not None:
            dfdt = (f_meas - self._prev_f_filt) / dt
        else:
            dfdt = 0.0
        self._prev_t = t
        self._prev_f_filt = f_meas

        # Standing de-load (curtailment) plus frequency-support droop and
        # synthetic inertia. The de-load is a constant negative torque offset
        # that over-speeds the rotor off the Cp-max point, creating a power
        # reserve; the droop unwinds it on a frequency dip to release the
        # reserve as extra grid power.
        raw = -self._deload - (self._droop * df_eff) - (self._inertia * dfdt)
        raw = max(-self._max, min(self._max_over, raw))
        offset = self._envelope(t) * raw

        with self._lock:
            self._offset = offset
            self._f = f_grid_hz
            self._t = t
        return offset

    def current_offset(self) -> float:
        with self._lock:
            return self._offset

    def set_manual_offset(self, value: float) -> None:
        """Directly set the reply torque offset [Nm], bypassing the droop law.

        Used for open-loop excitation (e.g. a generator-torque impulse to ring a
        mechanical mode for modal identification).
        """
        with self._lock:
            self._offset = float(value)

    # ------------------------------------------------------------------
    # Socket thread
    # ------------------------------------------------------------------
    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._sock.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break

            # Parse ROSCO's Time (idx 2) and GenTqMeas (idx 7) for logging only.
            t_rosco = gen_tq = float("nan")
            try:
                parts = msg.decode("ascii", "ignore").split(",")
                t_rosco = float(parts[2])
                gen_tq = float(parts[7])
            except (IndexError, ValueError):
                pass

            with self._lock:
                offset = self._offset
                f_grid = self._f

            setpoints = [offset, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
            # Null-terminate so ROSCO's C strtok stops cleanly at the reply end.
            reply = ",".join(f"{s:.6e}" for s in setpoints).encode("ascii") + b"\x00"
            try:
                self._sock.send(reply)
            except zmq.ZMQError:
                break

            self._n_requests += 1
            self._rows.append((t_rosco, offset, gen_tq, f_grid))
