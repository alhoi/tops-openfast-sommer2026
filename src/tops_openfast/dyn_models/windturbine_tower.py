"""
Wind-turbine model with an optional tower side-to-side (SS) structural mode.

This module is a superset of ``windturbine.py``: the drivetrain, aerodynamics,
pitch controller, MPT/Cp handling and UIC/droop interface are reproduced
unchanged, and an optional first tower **side-to-side** bending mode is added on
top. The SS mode can be switched on/off with the ``ss_enable`` parameter; when
off (or absent) the model behaves identically to the original ``WindTurbine``.

------------------------------------------------------------------------------
Theory: why the generator torque excites the tower side-to-side mode
------------------------------------------------------------------------------
The generator/air-gap electromagnetic torque ``Te`` acts about the (nominally
horizontal) low-speed-shaft axis. By Newton's third law the stator -- i.e. the
nacelle -- feels the equal and opposite reaction torque about that same axis. A
reaction torque about the fore-aft axis at the nacelle rolls the nacelle in the
vertical-lateral plane, which bends the tower top **side to side** (lateral
motion). Fluctuations in the electromagnetic torque therefore drive the tower
side-to-side mode. This is exactly the mechanism exploited by active
side-to-side tower-damping controllers, and by ROSCO's Tower Resonance Avoidance
(TRA) logic, both of which modulate generator torque to influence tower motion.

    Bossanyi, E. A. (2003). "Wind Turbine Control for Load Reduction."
        Wind Energy 6(3): 229-244.  (generator torque <-> side-to-side damping)
    Burton, Jenkins, Sharpe, Bossanyi (2011). "Wind Energy Handbook", 2nd ed.,
        Wiley. (tower structural dynamics, thrust/aerodynamics)

Because the drivetrain electromagnetic torque ``Te`` here carries the grid
disturbance (``Te = Pe / (omega_e * eta)``, with ``Pe`` the delivered electrical
power), this gives a genuine, *open* electrical -> mechanical path:
grid -> Pe -> Te -> nacelle reaction moment -> tower side-to-side mode.

------------------------------------------------------------------------------
Reduced modal representation
------------------------------------------------------------------------------
Each tower mode is represented as a single-DOF modal (assumed-mode) oscillator,
exactly as ElastoDyn/FAST represents tower bending via a Rayleigh-Ritz mode
shape:

    q'' + 2*zeta_ss*omega_ss*q' + omega_ss^2 * q = g_ss * Te                (1)

    Jonkman, J. M. (2007). "Dynamics Modeling and Loads Analysis of an Offshore
        Floating Wind Turbine." NREL/TP-500-41958. (ElastoDyn modal tower DOFs)
    Jonkman & Buhl (2005). "FAST User's Guide." NREL/EL-500-38230.

Here ``q`` is the tower-top side-to-side displacement, ``omega_ss = 2*pi*f_ss``
the modal natural frequency, ``zeta_ss`` the modal damping ratio, and ``g_ss``
the torque-to-modal-force coupling gain [m/s^2 per pu torque]. The tower-top
side-to-side **acceleration** (analogous to the OpenFAST output ``YawBrTAyp``)
is the second derivative

    a_ss = q'' = g_ss*Te - 2*zeta_ss*omega_ss*q' - omega_ss^2*q.            (2)

Modal frequency/damping for the IEA 15 MW monopile tower first SS mode:

    Gaertner, E. et al. (2020). "Definition of the IEA 15-Megawatt Offshore
        Reference Wind Turbine." NREL/TP-5000-75698.
    -> f_ss ~ 0.234 Hz.

------------------------------------------------------------------------------
Calibration of g_ss (against the high-fidelity OpenFAST-FMU sweep)
------------------------------------------------------------------------------
For the oscillator (1) driven at its natural frequency, the steady-state
acceleration amplitude is (from the transfer function of (2) at omega=omega_ss)

    |a_ss(omega_ss)| = (g_ss / (2*zeta_ss)) * |dTe|.                        (3)

The OpenFAST-FMU torque sweep (sweep_resonance.py) measured a peak side-to-side
acceleration of ~0.259 m/s^2 (YawBrTAyp at 0.234 Hz) for a +/-10 % generator-
torque modulation (|dTe| ~ 0.10 pu) at zeta ~ 0.05. Solving (3) for g_ss:

    g_ss = 2*zeta_ss*|a_ss| / |dTe| = 2*0.05*0.259 / 0.10 ~ 0.26 m/s^2/pu.

This analytic estimate is then refined by an equivalent reduced-model torque
sweep (sweep_ss_reduced.py), which injects the same +/-10 % modulation on Te and
measures the SS acceleration. That sweep gives a peak of 0.2499 m/s^2 at
0.234 Hz for g_ss = 0.259; scaling linearly to hit the FMU peak,

    g_ss = 0.259 * (0.259 / 0.2499) = 0.2684 m/s^2/pu,

which was the originally calibrated value at the nominal zeta_ss = 0.05.

Effective SS damping (measured from the OpenFAST ring-down)
----------------------------------------------------------
The nominal zeta_ss = 0.05 above was a placeholder. The *effective* side-to-side
damping is measured from the OpenFAST ``YawBrTAyp`` free-decay ring-down (a
generator-torque pulse at constant wind, then free decay), exactly as zeta_fa is
obtained for the fore-aft mode. Two independent estimators (log decrement on the
band-passed peaks and an exponential fit of the Hilbert amplitude envelope)
agree at

    zeta_ss ~ 0.0034 (0.34 %),   tau ~ 187 s,   Q ~ 145,

confirming that the tower side-to-side mode is very lightly damped (almost no
aerodynamic damping, structural only). Because the calibration anchors the
*transfer gain* g_ss/(2*zeta_ss) to the FMU resonant peak (3), correcting
zeta_ss requires rescaling g_ss by the same factor so the forced-resonance
amplitude stays matched to OpenFAST:

    g_ss = 0.2684 * (0.0034 / 0.05) ~ 0.01825 m/s^2/pu.

These two (zeta_ss = 0.0034, g_ss = 0.01825) are the calibrated defaults. (The
small scale factor 1.036 in the g_ss anchor reflects that the reduced-model
operating-point torque Te0 ~ 0.965 pu differs slightly from unity; the
peak-amplitude match folds the Region-2/Region-3 operating-point difference
into g_ss.)

------------------------------------------------------------------------------
Theory: the tower fore-aft mode and why the grid barely excites it
------------------------------------------------------------------------------
The first tower **fore-aft** mode bends the tower top in the wind direction and
is driven by the rotor **thrust** force

    F_T = 0.5 * rho * A * v^2 * C_T(lambda, beta),                          (4)

with A the rotor swept area, v the wind speed, C_T the thrust coefficient,
lambda = omega_m*R/v the tip-speed ratio and beta the blade pitch. Unlike the
side-to-side mode, the fore-aft mode is *not* driven by the generator/air-gap
torque: a grid disturbance reaches the thrust only *indirectly*, through the
drivetrain speed,

    grid -> Pe -> Te -> omega_m -> lambda -> C_T -> F_T -> fore-aft,

i.e. a second-order path via the (very large) rotor inertia. At a fixed wind
speed the thrust therefore hardly moves when the grid is perturbed, so the
fore-aft response to an electrical disturbance is expected to be much smaller
than the side-to-side response -- the very point of adding this DOF.

Aerodynamic damping. In operation the fore-aft mode is far more heavily damped
than the side-to-side mode: when the tower top moves downwind the wind seen by
the rotor drops, lowering the thrust and opposing the motion (the well-known
*aerodynamic damper*). Side-to-side sees almost no such damping, which is why it
-- not fore-aft -- is the resonance-prone mode. Rather than resolve this
velocity-dependent aerodynamic force explicitly, it is folded into an *effective*
modal damping ratio zeta_fa (structural + aerodynamic), calibrated from the
OpenFAST-FMU fore-aft ring-down exactly as the effective SS damping is. This
keeps the two tower modes represented consistently and directly comparable. In
practice the measured effective zeta_fa is small (~1.2-1.9 % from the FMU
ring-down at v=11 m/s, where the rotor is pitching and the aerodynamic damping
is modest), so fore-aft is only lightly damped here -- the reason it is *not*
the resonance-prone mode is the weak thrust drive path, not heavy damping.

    Kuhn, M. (2001). "Dynamics and Design Optimisation of Offshore Wind Energy
        Conversion Systems." PhD thesis, TU Delft. (aerodynamic tower damping)
    Salzmann & van der Tempel (2005). "Aerodynamic damping in the design of
        support structures for offshore wind turbines." Copenhagen Offshore Wind.

The fore-aft modal oscillator mirrors the side-to-side one (1):

    q_fa'' + 2*zeta_fa*omega_fa*q_fa' + omega_fa^2 * q_fa = g_fa * F_T_hat,  (5)

where F_T_hat = C_T(lambda,beta)*v^2 normalised by its equilibrium value (so a
steady thrust produces no ring), omega_fa = 2*pi*f_fa, and g_fa is the
thrust-to-modal-acceleration gain. At a fixed wind speed F_T_hat moves only
through omega_m -> lambda -> C_T, i.e. the weak indirect grid path. f_fa ~ f_ss
for this tower (the two first tower modes are nearly degenerate); f_fa, zeta_fa
and g_fa are calibrated against the FMU fore-aft acceleration output
``YawBrTAxp`` (free-decay/step) exactly as g_ss was calibrated against
``YawBrTAyp``.

------------------------------------------------------------------------------
Modelling choices / limitations (current version)
------------------------------------------------------------------------------
- The first side-to-side (torque-driven) and first fore-aft (thrust-driven)
  tower modes are implemented, each as an independent 1-DOF modal oscillator,
  toggled by ``ss_enable`` / ``fa_enable``.
- Both modes are *forward-coupled*: they are driven by the turbine states (Te
  for SS, thrust for FA) but do not feed back onto the drivetrain/electrical
  states, so enabling them does not change the electrical/drivetrain solution
  and on/off runs stay directly comparable.
- The fore-aft aerodynamic damping is represented by an effective (calibrated)
  modal damping zeta_fa rather than resolved from the relative wind, keeping it
  consistent with the SS representation.
- f_fa (0.235 Hz), zeta_fa (0.0125 effective) and g_fa (0.102 m/s^2) are
  calibrated against the FMU YawBrTAxp: f_fa/zeta_fa from a free-decay
  ring-down and g_fa from a forced-resonance run (wind sine at f_fa,
  TTDspFA=0 so there is no ring-down contamination, 300 s so the response
  is fully settled).
- Higher tower modes, blade edgewise/flap DOFs and the drivetrain torsion DOF
  are not represented here (torsion is covered separately in the FMU studies).
"""


from tops.dyn_models.utils import DAEModel
from tops_openfast.dyn_models.speed_lpf import (
    apply_speed_lpf_dynamics,
    resolve_speed_lpf_params,
    speed_pu_for_use,
)
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import brentq
from pathlib import Path


class WindTurbineTower(DAEModel):
    """
    'windturbine': {
        'WindTurbineTower': [
            ['name', 'UIC', 'S_n', 'V_n', 'J_m', 'J_e', 'K', 'D', 'Kp_pitch', 'Ki_pitch', 'T_pitch', 'max_pitch', 'min_pitch', 'max_pitch_rate', 'rho', 'R', 'P_rated', 'omega_m_rated', 'wind_rated', 'efficiency', 'MPT_filename', 'Cp_filename', 'speed_lpf_type', 'speed_lpf_corner_rad_s', 'speed_lpf_damping', 'f_nom_hz', 'droop_enable', 'K_droop_pu_per_hz', 'headroom_pu', 'ss_enable', 'f_ss_hz', 'zeta_ss', 'g_ss', 'fa_enable', 'f_fa_hz', 'zeta_fa', 'g_fa'],
            ['WT1', 'WT1_LEOGO', 15.0, 0.69, 352460500.0, 1836784.0, 697376449.0, 71186519.0, 0.6738, 0.06, 2.2, 30.0, 0.0, 10.0, 1.225, 120.97, 1.0, 7.56, 10.6, 0.95756, 'MPT_Kopt2150.csv', 'Cp_Ct_Cq.IEA15MW.ROSCO.txt', 2, 1.00810, 0.70000, 50.0, 0, 0.75, 0.05, 1, 0.234, 0.0034, 0.01825, 1, 0.235, 0.0125, 0.102],
            # [ ...same units as WindTurbine..., ss_enable 0/1, f_ss_hz Hz, zeta_ss -, g_ss m/s^2/pu,
            #   fa_enable 0/1, f_fa_hz Hz, zeta_fa - (effective), g_fa m/s^2/pu]
        ],
    }

    Superset of ``WindTurbine`` with an optional tower side-to-side (SS)
    structural mode. See the module docstring for theory, sources and the
    calibration of ``g_ss``. The four trailing parameters (``ss_enable``,
    ``f_ss_hz``, ``zeta_ss``, ``g_ss``) are optional; sensible defaults are used
    when they are absent, and ``ss_enable=0`` reproduces ``WindTurbine`` exactly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sn = self.par['S_n']
        sn[sn == 0] = self.sys_par['s_n']
        self.par['S_n'] = sn
        self._sys_to_local = self.sys_par['s_n'] / self.par['S_n']
        self._local_to_sys = self.par['S_n'] / self.sys_par['s_n']

        # Convert omega_m_rated from RPM to rad/s
        RPM_to_rad_per_s = 2 * np.pi / 60  # 1 RPM = 2π/60 rad/s
        self.par['omega_m_rated'] = self.par['omega_m_rated'] * RPM_to_rad_per_s

        self._debug_counter = 0

        # Load wind data from .hh file for variable wind speed
        wind_file_path = Path(__file__).parents[3] / 'wind_data' / '10mps_NTM_3xDTU10MW_IECKAI_VS_T1.hh'
        wind_data = np.loadtxt(wind_file_path, skiprows=1, usecols=(0, 1))
        wind_times = wind_data[:, 0]  # First column: time in seconds
        wind_speeds = wind_data[:, 1]  # Second column: wind speed in m/s
        self._wind_interp = interp1d(wind_times, wind_speeds, kind='linear',
                                     bounds_error=False, fill_value=(wind_speeds[0], wind_speeds[-1]))

        # convert all WT params to pu:
        w_m_base = self.par['omega_m_rated']  # rad/s
        T_base = self.par['S_n'] * 1e6 / w_m_base  # Nm

        # Calculate H_m and H_e from J_m and J_e as instance variables (arrays, one per unit)
        self.H_m = 0.5 * self.par['J_m'] * w_m_base**2 / (self.par['S_n'] * 1e6)
        self.H_e = 0.5 * self.par['J_e'] * w_m_base**2 / (self.par['S_n'] * 1e6)
        self.par['K'] = self.par['K'] / T_base
        self.par['D'] = self.par['D'] * w_m_base / T_base
        self.par['max_pitch'] = self.par['max_pitch'] * np.pi / 180
        self.par['min_pitch'] = self.par['min_pitch'] * np.pi / 180
        self.par['max_pitch_rate'] = self.par['max_pitch_rate'] * np.pi / 180

        n_units = int(np.asarray(self.par['S_n']).size)
        (
            self._speed_lpf_type,
            self._speed_lpf_corner_rad_s,
            self._speed_lpf_damping,
        ) = resolve_speed_lpf_params(self.par, n_units)

        # Gen-speed LPF in Te denominator (ROSCO); keeps MPT on raw omega_m when speed_lpf_type=0.
        self._te_speed_lpf_type = 2
        self._grid_frequency_hz = float(
            np.asarray(self.par['f_nom_hz']).ravel()[0]
        )

        # --- Tower side-to-side (SS) modal parameters (optional) ---
        # Read with defaults so old parameter templates (without these fields)
        # still construct; ss_enable=0 => identical to WindTurbine.
        self._ss_enable = bool(int(self._get_par('ss_enable', 0)))
        self._f_ss_hz = self._get_par('f_ss_hz', 0.234)
        self._zeta_ss = self._get_par('zeta_ss', 0.05)
        self._g_ss = self._get_par('g_ss', 0.2684)
        self._omega_ss = 2.0 * np.pi * self._f_ss_hz

        # Optional reciprocal tower-SS <-> drivetrain feedback (default OFF).
        # ss_feedback_enable=0 keeps the SS mode forward-coupled only, so the
        # electrical/drivetrain solution is unchanged and earlier results are
        # reproduced bit-for-bit. When enabled, a single energy-consistent
        # inertia coupling c (ss_feedback_c) links the generator-rotor and
        # tower-SS accelerations through a symmetric 2x2 mass matrix (see
        # state_derivatives). c is intended to be calibrated against the
        # OpenFAST FMU.
        self._ss_feedback_enable = bool(
            int(self._get_par('ss_feedback_enable', 0))
        )
        self._ss_feedback_c = float(self._get_par('ss_feedback_c', 0.0))

        # Optional test modulation of the SS driving torque, used only for
        # calibration/verification sweeps. When set to a callable m(t) it makes
        # the SS forcing Te*(1 + m(t)), reproducing the FMU sweep protocol
        # GenTq = T0*(1 + amp*sin(...)). None (default) => no effect.
        self._ss_test_mod = None

        # --- Tower fore-aft (FA) modal parameters (optional, thrust-driven) ---
        # Read with defaults so old parameter templates (without these fields)
        # still construct; fa_enable=0 => identical to the SS-only model.
        self._fa_enable = bool(int(self._get_par('fa_enable', 0)))
        self._f_fa_hz = self._get_par('f_fa_hz', 0.235)  # FMU YawBrTAxp ring-down
        self._zeta_fa = self._get_par('zeta_fa', 0.0125)  # effective (struct+aero), FMU-calibrated
        self._g_fa = self._get_par('g_fa', 0.102)
        self._omega_fa = 2.0 * np.pi * self._f_fa_hz
        # Equilibrium thrust normaliser Ct0*v0^2 (set in init_from_connections)
        # so that the fore-aft forcing F_T_hat starts at 1.0.
        self._fa_thrust0 = None
        # Optional test modulation of the FA thrust forcing (calibration sweeps).
        self._fa_test_mod = None

    def set_ss_test_modulation(self, func):
        """Set a callable m(t) -> fractional SS-forcing torque modulation.

        Pass None to disable. Used by calibration sweeps only; it does not
        affect the electrical/drivetrain states (SS is forward-coupled).
        """
        self._ss_test_mod = func

    def set_ss_feedback(self, enable, c=None):
        """Enable/disable the reciprocal tower-SS <-> drivetrain feedback.

        Parameters
        ----------
        enable : bool
            Turn the two-way inertia coupling on or off.
        c : float, optional
            Coupling coefficient (symmetric 2x2 mass-matrix off-diagonal).
            When given it also sets ss_feedback_c. For the mass matrix to stay
            positive-definite it must satisfy c**2 < 2*H_e (== J_e in pu).
        """
        self._ss_feedback_enable = bool(enable)
        if c is not None:
            self._ss_feedback_c = float(c)

    def _ss_test_factor(self):
        """Return (1 + m(t)) for the SS test modulation, or 1.0 if disabled."""
        mod = self._ss_test_mod
        if mod is None:
            return 1.0
        t = getattr(self, '_sim_time', 0.0)
        return 1.0 + float(mod(t))

    def set_fa_test_modulation(self, func):
        """Set a callable m(t) -> fractional FA-forcing (thrust) modulation.

        Pass None to disable. Used by fore-aft calibration sweeps only; it does
        not affect the electrical/drivetrain states (FA is forward-coupled).
        """
        self._fa_test_mod = func

    def _fa_test_factor(self):
        """Return (1 + m(t)) for the FA test modulation, or 1.0 if disabled."""
        mod = self._fa_test_mod
        if mod is None:
            return 1.0
        t = getattr(self, '_sim_time', 0.0)
        return 1.0 + float(mod(t))

    def _get_par(self, name, default):
        """Return par[name] as a float, or ``default`` if the field is absent."""
        try:
            return float(np.asarray(self.par[name]).ravel()[0])
        except (ValueError, KeyError, IndexError, TypeError):
            return float(default)

    def connections(self):
        return [
            {
                'input': 'P_e',
                'source': {
                    'container': 'vsc',
                    'mdl': '*',
                    'id': self.par['UIC'],
                },
                'output': 'p_e',
            },
            {
                'input': 'S_n_UIC',
                'source': {
                    'container': 'vsc',
                    'mdl': '*',
                    'id': self.par['UIC'],
                },
                'output': 'S_n',
            },
            {
                'output': 'P_ref',
                'destination': {
                    'container': 'vsc',
                    'mdl': '*',
                    'id': self.par['UIC'],
                },
                'input': 'p_ref',
            }
        ]

    def state_list(self):
        # speed_lpf_* applies to omega_m_filt only (rotor / MPT path).
        # q_ss / q_ss_dot are the tower side-to-side modal displacement and
        # velocity; they are inert (derivatives zero) when ss_enable=0.
        return [
            'omega_m',
            'omega_e',
            'theta_m',
            'theta_e',
            'pitch_PI_integral_state',
            'pitch_angle',
            'omega_m_filt',
            'omega_m_filt_dot',
            'omega_e_filt',
            'omega_e_filt_dot',
            'q_ss',
            'q_ss_dot',
            'q_fa',
            'q_fa_dot',
        ]

    def input_list(self):
        return ['P_e', 'S_n_UIC']

    def output_list(self):
        return ['P_ref']

    def _electromagnetic_torque(self, x, v):
        """Generator electromagnetic torque Te [WT pu] (mirrors state_derivatives)."""
        X = self.local_view(x)
        par = self.par
        eta = float(np.asarray(par['efficiency']).ravel()[0])
        lpf_type = int(np.asarray(self._speed_lpf_type).ravel()[0])
        Pe = self.P_e(x, v) * self.S_n_UIC(x, v) / par['S_n']  # UIC pu -> WT pu
        omega_e_filtered = speed_pu_for_use(X, 'omega_e', 'omega_e_filt', lpf_type)
        omega_e_filtered = np.asarray(omega_e_filtered, dtype=float)
        omega_e_filtered = np.where(
            np.isfinite(omega_e_filtered) & (omega_e_filtered > 1e-3),
            omega_e_filtered, 1e-3,
        )
        Te = np.where(np.isfinite(Pe), Pe / (omega_e_filtered * eta), 0.0)
        return np.asarray(Te).ravel()

    def ss_acceleration(self, x, v):
        """Tower-top side-to-side acceleration [m/s^2], analogous to YawBrTAyp.

        Returns zeros when the SS mode is disabled. With ss_feedback_enable=1
        the reciprocal inertia coupling is applied so the reported acceleration
        is consistent with the integrated q_ss_dot dynamics.
        """
        X = self.local_view(x)
        q = np.asarray(X['q_ss']).ravel()
        if not self._ss_enable:
            return np.zeros_like(q)
        q_dot = np.asarray(X['q_ss_dot']).ravel()
        Te_raw = np.asarray(self._electromagnetic_torque(x, v)).ravel()
        a_fwd = (
            self._g_ss * Te_raw * self._ss_test_factor()
            - 2.0 * self._zeta_ss * self._omega_ss * q_dot
            - self._omega_ss**2 * q
        )
        if self._ss_feedback_enable and self._ss_feedback_c != 0.0:
            # Coupled tower-SS acceleration (see state_derivatives): the reverse
            # inertia coupling c mixes in the rotor RHS b1 = T_shaft - Te.
            par = self.par
            theta_s = np.asarray(X['theta_m'] - X['theta_e']).ravel()
            omega_s = np.asarray(X['omega_m'] - X['omega_e']).ravel()
            T_shaft = (np.asarray(par['K']).ravel() * theta_s
                       + np.asarray(par['D']).ravel() * omega_s)
            b1 = T_shaft - Te_raw
            J_e_pu = np.asarray(2.0 * self.H_e, dtype=float).ravel()
            c = float(self._ss_feedback_c)
            det = J_e_pu - c * c
            return (c * b1 + J_e_pu * a_fwd) / det
        return a_fwd

    def _thrust_ct_v2(self, x, v):
        """Rotor thrust proxy C_T(lambda, beta) * v^2 [per unit turbine].

        rho and A cancel when this is later normalised by its equilibrium
        value, so they are omitted here. Evaluated at the free wind speed;
        the aerodynamic damping is folded into the effective zeta_fa.
        """
        self.load_and_set_Ct(x, v)
        par = self.par
        X = self.local_view(x)
        w_rated = float(np.asarray(par['omega_m_rated']).ravel()[0])
        R = float(np.asarray(par['R']).ravel()[0])
        v_free = float(np.asarray(self.wind_speed(x, v)).ravel()[0])
        v_free = max(v_free, 1e-3)
        omega_m_rad_s = np.asarray(X['omega_m'], dtype=float).ravel() * w_rated
        pa = np.asarray(X['pitch_angle'], dtype=float).ravel() * 180.0 / np.pi
        tsr = omega_m_rad_s * R / v_free
        pa_c = np.clip(pa, self._ct_interp.grid[0].min(), self._ct_interp.grid[0].max())
        tsr_c = np.clip(tsr, self._ct_interp.grid[1].min(), self._ct_interp.grid[1].max())
        Ct = np.asarray(self._ct_interp(np.column_stack([pa_c, tsr_c]))).ravel()
        return Ct * v_free**2

    def _fa_forcing(self, x, v):
        """Normalised fore-aft thrust forcing F_T_hat (=1 at equilibrium).

        At fixed wind this moves only through omega_m -> lambda -> C_T, i.e. the
        weak indirect grid path (see module docstring, eq. (4)-(5)).
        """
        ct_v2 = np.asarray(self._thrust_ct_v2(x, v), dtype=float).ravel()
        thrust0 = self._fa_thrust0
        if thrust0 is None:
            thrust0 = self._thrust_ct_v2(x, v)  # fallback: assume ~equilibrium
        thrust0 = np.asarray(thrust0, dtype=float).ravel()
        thrust0 = np.where(np.isfinite(thrust0) & (thrust0 > 1e-9), thrust0, 1e-9)
        return ct_v2 / thrust0

    def fa_acceleration(self, x, v):
        """Tower-top fore-aft acceleration [m/s^2], analogous to YawBrTAxp.

        Returns zeros when the fore-aft mode is disabled.
        """
        X = self.local_view(x)
        q = np.asarray(X['q_fa']).ravel()
        if not self._fa_enable:
            return np.zeros_like(q)
        q_dot = np.asarray(X['q_fa_dot']).ravel()
        F_fa = np.asarray(self._fa_forcing(x, v)).ravel() * self._fa_test_factor()
        return (
            self._g_fa * F_fa
            - 2.0 * self._zeta_fa * self._omega_fa * q_dot
            - self._omega_fa**2 * q
        )

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par

        # P and T calculations
        P_aero = self.P_aero(x, v)
        Pe = self.P_e(x, v) * self.S_n_UIC(x, v) / par['S_n']  # UIC pu -> WT pu
        Ta = P_aero / X['omega_m'] if X['omega_m'] > 0 else 0
        eta = float(np.asarray(par['efficiency']).ravel()[0])
        Tm = Ta  # could have rotor efficiency here

        # Speed LPF params
        lpf_type = int(np.asarray(self._speed_lpf_type).ravel()[0])
        omega_c = float(np.asarray(self._speed_lpf_corner_rad_s).ravel()[0])
        zeta = float(np.asarray(self._speed_lpf_damping).ravel()[0])

        # --- Torque coupling: Te = Pe / (omega_e_filt * eta) ---
        omega_e_filtered = speed_pu_for_use(X, 'omega_e', 'omega_e_filt', lpf_type)
        if not np.isfinite(omega_e_filtered) or omega_e_filtered <= 1e-3:
            omega_e_filtered = 1e-3
        Te = Pe / (omega_e_filtered * eta) if np.isfinite(Pe) else 0.0

        # shaft torque
        theta_s = X['theta_m'] - X['theta_e']
        omega_s = X['omega_m'] - X['omega_e']
        T_shaft = par['K'] * theta_s + par['D'] * omega_s

        # dynamic equations for drivetrain
        dX['omega_m'] = (1/(2*self.H_m)) * (Tm - T_shaft)
        dX['theta_m'] = X['omega_m']
        dX['theta_e'] = X['omega_e']

        # --- Tower side-to-side modal oscillator (toggleable) ---
        # q'' + 2*zeta*omega*q' + omega^2*q = g_ss*Te   (see module docstring).
        # By default forward-coupled ONLY: the tower does not feed back onto the
        # drivetrain, so the electrical/drivetrain solution is unchanged and
        # ss_feedback_enable=0 reproduces earlier results bit-for-bit.
        #
        # With ss_feedback_enable=1 a reciprocal (energy-consistent) inertia
        # coupling c links the generator-rotor and tower-SS accelerations
        # through the symmetric 2x2 mass matrix
        #     [ J_e  -c ] [ d(omega_e) ]   [ T_shaft - Te                   ]
        #     [ -c    1 ] [   dd(q_ss)  ] = [ g_ss*Te - 2 zeta w qd - w^2 q  ]
        # (J_e = 2*H_e in pu). Tower motion then reacts back on the rotor, and
        # hence on Te and the grid. c=0 recovers the diagonal (forward-only)
        # equations exactly. The symmetric, positive-definite mass matrix
        # (requires c**2 < J_e) makes the coupling conservative -- it injects
        # no spurious energy.
        J_e_pu = 2.0 * self.H_e
        b1 = T_shaft - Te                    # rotor RHS (== J_e * d(omega_e))
        q_ss = np.asarray(X['q_ss']).ravel()
        q_ss_dot = np.asarray(X['q_ss_dot']).ravel()
        if self._ss_enable:
            Te_arr = np.asarray(Te).ravel() * self._ss_test_factor()
            b2 = (
                self._g_ss * Te_arr
                - 2.0 * self._zeta_ss * self._omega_ss * q_ss_dot
                - self._omega_ss**2 * q_ss
            )
            if self._ss_feedback_enable and self._ss_feedback_c != 0.0:
                c = float(self._ss_feedback_c)
                Je = np.asarray(J_e_pu, dtype=float).ravel()
                det = Je - c * c
                b1r = np.asarray(b1, dtype=float).ravel()
                dX['omega_e'] = (b1r + c * b2) / det
                dX['q_ss'] = q_ss_dot
                dX['q_ss_dot'] = (c * b1r + Je * b2) / det
            else:
                dX['omega_e'] = (1.0 / J_e_pu) * b1
                dX['q_ss'] = q_ss_dot
                dX['q_ss_dot'] = b2
        else:
            dX['omega_e'] = (1.0 / J_e_pu) * b1
            dX['q_ss'] = 0.0
            dX['q_ss_dot'] = 0.0

        # --- Tower fore-aft modal oscillator (toggleable, thrust-driven) ---
        # q_fa'' + 2*zeta_fa*omega_fa*q_fa' + omega_fa^2*q_fa = g_fa*F_T_hat.
        # The thrust forcing carries the grid disturbance only indirectly, via
        # omega_m -> lambda -> Ct; the aerodynamic damping is folded into the
        # effective zeta_fa. Forward-coupled: does NOT feed back onto the
        # drivetrain/electrical states.
        q_fa = np.asarray(X['q_fa']).ravel()
        q_fa_dot = np.asarray(X['q_fa_dot']).ravel()
        if self._fa_enable:
            F_fa = np.asarray(self._fa_forcing(x, v)).ravel() * self._fa_test_factor()
            dX['q_fa'] = q_fa_dot
            dX['q_fa_dot'] = (
                self._g_fa * F_fa
                - 2.0 * self._zeta_fa * self._omega_fa * q_fa_dot
                - self._omega_fa**2 * q_fa
            )
        else:
            dX['q_fa'] = 0.0
            dX['q_fa_dot'] = 0.0

        # pitch controller
        max_pitch = par['max_pitch'][0]
        min_pitch = par['min_pitch'][0]
        max_pitch_rate = par['max_pitch_rate'][0]
        omega_ref = 1.0  # Rated speed in pu.

        # Pitch control must not react to the unfiltered torsional speed.
        omega_pitch = float(np.asarray(omega_e_filtered).ravel()[0])
        e_omega = omega_pitch - omega_ref

        wind_now = float(np.asarray(self.wind_speed(x, v)).ravel()[0])
        wind_rated = float(np.asarray(par["wind_rated"]).ravel()[0])
        use_region_2 = wind_now < wind_rated

        pitch_reference_pi = 0.0

        if use_region_2:
            # Region 2: below-rated wind, MPPT and minimum pitch.
            dX_pitch_integral = 0.0
            pitch_reference_pi = min_pitch
        else:  # Region 3: at/above rated wind, pitch to limit power
            PIctrl_integral_term = par['Ki_pitch'][0] * X['pitch_PI_integral_state'][0]
            PIctrl_proportional_term = par['Kp_pitch'][0] * e_omega
            pitch_reference_unclamped = PIctrl_integral_term + PIctrl_proportional_term

            # Anti-windup -> stops integration term when reference is at limit
            if pitch_reference_unclamped >= max_pitch or pitch_reference_unclamped <= min_pitch:
                dX_pitch_integral = 0.0  # Stop integrating when output hits limits
            else:
                dX_pitch_integral = e_omega  # Normal integration

            # Clamp pitch_reference to max and min pitch angle
            pitch_reference_pi = np.clip(pitch_reference_unclamped, min_pitch, max_pitch)

        dX['pitch_PI_integral_state'] = dX_pitch_integral
        # pitch servo: T_pitch drives pitch_angle toward PI demand, subject to rate limit
        delta_pitch_angle = (1/par['T_pitch'][0]) * (pitch_reference_pi - X['pitch_angle'][0])
        dX['pitch_angle'] = np.clip(delta_pitch_angle, -max_pitch_rate, max_pitch_rate)

        # apply speed LPF for omega_e
        omega_e = np.asarray(X['omega_e']).ravel()
        apply_speed_lpf_dynamics(dX, X, omega_e, 'omega_e_filt', 'omega_e_filt_dot', lpf_type, omega_c, zeta)
        if self._debug_counter > 100 and X['omega_m'][0] > 10:
            print('solution blowing up, omega_m:', X['omega_m'][0])

        return

    def init_from_connections(self, x_0, v_0, S):
        X = self.local_view(x_0)
        par = self.par
        self._input_values['P_e'] = self.P_e(x_0, v_0)
        self._input_values['S_n_UIC'] = self.S_n_UIC(x_0, v_0)

        w_rated = float(np.asarray(par['omega_m_rated']).ravel()[0])
        u_rated = float(np.asarray(par['wind_rated']).ravel()[0])
        u_start = float(np.asarray(self.wind_speed(x_0, v_0)).ravel()[0])
        K = float(np.asarray(par['K']).ravel()[0])

        self._load_MPT_table()
        eta = float(np.asarray(par['efficiency']).ravel()[0])

        if u_start >= u_rated * 0.99:
            omega_m_init_pu = 1.0
        else:
            # Region 2: P_aero = T_mech*omega_pu (MPPT mechanical target).
            def _res(om):
                X['omega_m'] = om
                X['omega_e'] = om
                X['pitch_angle'] = 0.0
                return float(self.P_aero(x_0, v_0).ravel()[0]) - self._mpt_power_mech_pu(
                    om * w_rated, om
                )

            try:
                omega_m_init_pu = float(brentq(_res, 0.05, 1.0))
            except ValueError:
                omega_m_init_pu = float(np.clip(u_start / u_rated, 0.05, 0.98))
                print(
                    'Brentq omega_m init failed; using u/u_rated =',
                    omega_m_init_pu,
                )
        X['omega_m'] = omega_m_init_pu
        X['omega_e'] = omega_m_init_pu
        X['pitch_angle'] = max(0.0, float(np.asarray(par['min_pitch']).ravel()[0]))

        pe_loc_pu = (self.P_e(x_0, v_0) * self.S_n_UIC(x_0, v_0) / par['S_n']).ravel()[0]  # UIC pu -> WT pu

        if omega_m_init_pu >= 0.99:
            # Region 3: rated speed + pitch for P_aero = P_e/eta.
            min_pitch = float(np.asarray(par['min_pitch']).ravel()[0])
            max_pitch = float(np.asarray(par['max_pitch']).ravel()[0])
            Ki = float(np.asarray(par['Ki_pitch']).ravel()[0])
            X['omega_m'] = 1.0
            X['omega_e'] = 1.0
            self.load_and_set_Cp(x_0, v_0)

            def _res_pitch(pitch_rad):
                X['pitch_angle'] = pitch_rad
                return float(self.P_aero(x_0, v_0).ravel()[0]) - pe_loc_pu / eta
            try:
                pitch_eq = brentq(_res_pitch, min_pitch, max_pitch)
            except ValueError:
                r_min = _res_pitch(min_pitch)
                r_max = _res_pitch(max_pitch)
                pitch_eq = min_pitch if abs(r_min) <= abs(r_max) else max_pitch
                print('Brentq pitch init failed, using endpoint closest to power balance')
            pitch_eq = np.clip(pitch_eq, min_pitch, max_pitch)
            X['pitch_angle'] = pitch_eq
            X['pitch_PI_integral_state'] = pitch_eq / Ki if Ki > 0 else 0.0
            self._pitch_angle = pitch_eq
            Te = pe_loc_pu / (float(np.asarray(X['omega_e']).ravel()[0]) * eta) if float(np.asarray(X['omega_e']).ravel()[0]) > 0 else 0.0
            theta_s = Te / K
            X['theta_m'] = 0.0
            X['theta_e'] = -theta_s
        else:
            # Region 2: MPPT, pitch at minimum (typically 0)
            X['pitch_PI_integral_state'] = 0.0
            X['pitch_angle'] = max(0.0, float(np.asarray(par['min_pitch']).ravel()[0]))
            self._pitch_angle = float(np.asarray(X['pitch_angle']).ravel()[0])
            om_e0 = float(np.asarray(X['omega_e']).ravel()[0])
            Te = pe_loc_pu / (om_e0 * eta) if om_e0 > 0 else 0.0
            theta_s = Te / K
            X['theta_m'] = 0.0
            X['theta_e'] = -theta_s

        om_m = np.asarray(X['omega_m'], dtype=float).ravel()
        X['omega_m_filt'] = om_m.copy()
        X['omega_m_filt_dot'] = np.zeros_like(om_m)
        om_e = np.asarray(X['omega_e'], dtype=float).ravel()
        X['omega_e_filt'] = om_e.copy()
        X['omega_e_filt_dot'] = np.zeros_like(om_e)

        # --- Tower side-to-side modal state: start at static equilibrium ---
        # q_eq = g_ss*Te0/omega_ss^2 gives zero initial acceleration, so a
        # constant torque produces no spurious startup transient; a torque
        # disturbance then makes the mode ring.
        om_e0 = float(np.asarray(X['omega_e']).ravel()[0])
        Te0 = pe_loc_pu / (om_e0 * eta) if om_e0 > 0 else 0.0
        if self._ss_enable:
            X['q_ss'] = self._g_ss * Te0 / (self._omega_ss**2)
        else:
            X['q_ss'] = 0.0
        X['q_ss_dot'] = 0.0

        # --- Tower fore-aft modal state: start at static equilibrium ---
        # Store the equilibrium thrust Ct0*v0^2 so the forcing F_T_hat starts at
        # 1.0 (no spurious startup ring); then q_fa_eq = g_fa/omega_fa^2.
        self._fa_thrust0 = np.asarray(
            self._thrust_ct_v2(x_0, v_0), dtype=float
        ).ravel()
        if self._fa_enable:
            X['q_fa'] = self._g_fa / (self._omega_fa**2)
        else:
            X['q_fa'] = 0.0
        X['q_fa_dot'] = 0.0

        return

    def reset_tower_modes(self, x, v):
        """Reset the tower SS/FA modal states to the *current* static equilibrium.

        Mirrors the tower-state initialisation in init_from_connections, but uses
        the present (e.g. warmed-up, settled) electromagnetic torque and thrust
        instead of the initial guess. With q_ss_dot = q_fa_dot = 0 and each modal
        displacement placed at its equilibrium (zero net modal acceleration), a
        settled baseline produces no residual ring-down. The tower modes are
        forward-coupled, so this does not perturb the electrical/drivetrain
        solution.
        """
        X = self.local_view(x)
        if self._ss_enable:
            Te = float(np.asarray(
                self._electromagnetic_torque(x, v), dtype=float
            ).ravel()[0])
            X['q_ss'] = self._g_ss * Te / (self._omega_ss**2)
        else:
            X['q_ss'] = 0.0
        X['q_ss_dot'] = 0.0

        # Re-reference the fore-aft thrust normaliser to the settled thrust, so
        # F_T_hat = 1 at the settled point and q_fa_eq = g_fa/omega_fa^2.
        self._fa_thrust0 = np.asarray(
            self._thrust_ct_v2(x, v), dtype=float
        ).ravel()
        if self._fa_enable:
            X['q_fa'] = self._g_fa / (self._omega_fa**2)
        else:
            X['q_fa'] = 0.0
        X['q_fa_dot'] = 0.0

        return

    def P_aero(self, x, v):
        # aerodynamic power calculation
        par = self.par
        wind_speed = self.wind_speed(x, v)
        Cp = self.load_and_set_Cp(x, v)

        # Aerodynamic power from wind: P = 0.5 * rho * A * v^3 * Cp
        P_aero_watts = 0.5 * par['rho'] * np.pi * par['R']**2 * wind_speed**3 * Cp

        # Convert to per-unit using local base power
        S_base_watts = self.par['S_n'] * 1e6  # Convert MVA to Watts, WT local base
        P_aero_pu = P_aero_watts / S_base_watts

        return P_aero_pu  # WT pu

    def set_grid_frequency_hz(self, f_grid_hz):
        """Midlertidig inngang for målt nettfrekvens."""
        self._grid_frequency_hz = float(f_grid_hz)

    def _droop_command(self, p_available_uic_pu):
        """Compute the active-power reference on the UIC base."""
        par = self.par

        f_nom = float(np.asarray(par["f_nom_hz"]).ravel()[0])
        droop_enabled = bool(int(np.asarray(par["droop_enable"]).ravel()[0]))
        k_droop = float(np.asarray(par["K_droop_pu_per_hz"]).ravel()[0])
        headroom = float(np.asarray(par["headroom_pu"]).ravel()[0])

        p_available_uic_pu = float(max(0.0, p_available_uic_pu))

        # De-loaded MPT operating point. No scheduled-power mode exists.
        p_base = float(
            np.clip(p_available_uic_pu - headroom, 0.0, p_available_uic_pu)
        )

        delta_p_requested = (
            k_droop * (f_nom - self._grid_frequency_hz)
            if droop_enabled
            else 0.0
        )

        p_ref_cmd = float(
            np.clip(p_base + delta_p_requested, 0.0, p_available_uic_pu)
        )

        delta_p_delivered = p_ref_cmd - p_base

        return p_ref_cmd, p_base, delta_p_delivered

    def P_ref_components(self, x, v):
        """Returnerer MPT-grense, basisreferanse, droop-bidrag og total P_ref (UIC-base)."""
        X = self.local_view(x)
        par = self.par

        lpf_type = int(np.asarray(self._speed_lpf_type).ravel()[0])

        omega_e_pu = speed_pu_for_use(X, 'omega_e', 'omega_e_filt', lpf_type)

        omega_rated = float(np.asarray(par['omega_m_rated']).ravel()[0])
        omega_e_rad_s = omega_e_pu * omega_rated

        # MPT-basert tilgjengelig elektrisk effekt på WT-base
        p_mpt_wt_pu = self._mpt_power_elec_pu(omega_e_rad_s, omega_e_pu)

        p_available_uic_pu = float(
            np.asarray(p_mpt_wt_pu * par["S_n"] / self.S_n_UIC(x, v)).ravel()[0]
        )

        p_rated_uic_pu = float(
            np.asarray(par["P_rated"] * par["S_n"] / self.S_n_UIC(x, v)).ravel()[0]
        )

        p_available_uic_pu = np.clip(p_available_uic_pu, 0.0, p_rated_uic_pu)

        p_ref_cmd, p_base, delta_p = self._droop_command(p_available_uic_pu)

        return {
            'p_available_uic_pu': p_available_uic_pu,
            'p_base_uic_pu': p_base,
            'p_droop_delta_uic_pu': delta_p,
            'p_ref_uic_pu': p_ref_cmd,
        }

    def P_ref(self, x, v):
        """Aktiv-effektreferanse sendt fra vindturbinmodellen til UIC-en."""
        ref = self.P_ref_components(x, v)
        return np.atleast_1d(ref['p_ref_uic_pu'])

    def P_ref_from_wind(self, wind_speed_mps, S_n_UIC):
        """P_ref in UIC pu. Same MPPT root as init (P_aero = T_mech*omega_pu)."""
        wind_speed_mps = float(np.asarray(wind_speed_mps).ravel()[0])
        self._load_MPT_table()
        self.load_and_set_Cp(None, None)  # load Cp table only
        par = self.par
        w_rated = float(np.asarray(par['omega_m_rated']).ravel()[0])
        R = float(np.asarray(par['R']).ravel()[0])
        rho = float(np.asarray(par['rho']).ravel()[0])
        S_n = float(np.asarray(par['S_n']).ravel()[0])
        eta = float(np.asarray(par['efficiency']).ravel()[0])
        eta = eta if np.isfinite(eta) and eta > 0 else 1.0

        def _res(om):
            omega_rad = om * w_rated
            tsr = omega_rad * R / wind_speed_mps if wind_speed_mps > 0 else 0
            pa = np.clip(0.0, self._cp_interp.grid[0].min(), self._cp_interp.grid[0].max())
            tsr_c = np.clip(tsr, self._cp_interp.grid[1].min(), self._cp_interp.grid[1].max())
            Cp = float(self._cp_interp(np.array([pa, tsr_c]))[0])
            P_aero = 0.5 * rho * np.pi * R**2 * wind_speed_mps**3 * Cp / (S_n * 1e6)
            return P_aero - self._mpt_power_mech_pu(omega_rad, om)

        try:
            omega_init = float(brentq(_res, 0.05, 1.0))
        except ValueError:
            lam_ref = R * w_rated / float(np.asarray(par['wind_rated']).ravel()[0])
            omega_init = float(np.clip(lam_ref * wind_speed_mps / R / w_rated, 0.05, 1.0))
        P_elec_wt_pu = self._mpt_power_elec_pu(omega_init * w_rated, omega_init)
        p_available_uic_pu = (
            P_elec_wt_pu * S_n / float(np.asarray(S_n_UIC).ravel()[0])
        )
        p_rated_uic_pu = float(
            np.asarray(par["P_rated"]).ravel()[0]
        ) * S_n / float(np.asarray(S_n_UIC).ravel()[0])

        p_available_uic_pu = float(
            np.clip(p_available_uic_pu, 0.0, p_rated_uic_pu)
        )

        p_ref_cmd, _, _ = self._droop_command(float(p_available_uic_pu))

        return p_ref_cmd

    def _mpt_power_mech_pu(self, omega_rad_s, omega_pu):
        return self._mpt_torque_mech_pu(omega_rad_s) * float(omega_pu)

    def _mpt_power_elec_pu(self, omega_rad_s, omega_pu):
        eta = float(np.asarray(self.par['efficiency']).ravel()[0])
        eta = eta if np.isfinite(eta) and eta > 0 else 1.0
        return eta * self._mpt_power_mech_pu(omega_rad_s, omega_pu)

    def _mpt_torque_mech_pu(self, omega_rad_s):
        self._load_MPT_table()
        return float(self._mpt_torque_interp(omega_rad_s))

    def _load_MPT_table(self):
        # load MPT_T_* torque (pu on WT shaft base). P_e = eta*T_mech*omega_pu.
        if hasattr(self, '_mpt_torque_interp'):
            return
        project_root = Path(__file__).parents[3]
        mpt_filename = self.par['MPT_filename'][0] if isinstance(self.par['MPT_filename'], np.ndarray) else self.par['MPT_filename']
        mpt_t_filename = str(mpt_filename).replace('MPT_', 'MPT_T_', 1)
        path = project_root / 'wind_data' / mpt_t_filename
        data = np.loadtxt(path, delimiter='\t')
        rotor_speed_RPM = data[2:, 0]
        torque_mech_pu = data[2:, 1]
        rotor_speed_rad_s = rotor_speed_RPM * (2 * np.pi / 60)
        self._mpt_torque_interp = interp1d(
            rotor_speed_rad_s,
            torque_mech_pu,
            kind='linear',
            bounds_error=False,
            fill_value=(0.0, torque_mech_pu[-1]),
        )

    def load_and_set_Cp(self, x, v):
        par = self.par
        # Load Cp data if not already loaded
        if not hasattr(self, '_cp_data'):
            project_root = Path(__file__).parents[3]
            cp_filename = self.par['Cp_filename'][0] if isinstance(self.par['Cp_filename'], np.ndarray) else self.par['Cp_filename']
            path = project_root / 'wind_data' / cp_filename
            with open(path, 'r') as f:
                lines = f.readlines()
            pitch_line = lines[4].strip()
            if pitch_line.startswith('#'):
                pitch_line = pitch_line[1:].strip()
            pitch_angles = np.array([float(x) for x in pitch_line.split()])
            tsr_line = lines[6].strip()
            if tsr_line.startswith('#'):
                tsr_line = tsr_line[1:].strip()
            tip_speed_ratios = np.array([float(x) for x in tsr_line.split()])
            cp_start_idx = None
            for i, line in enumerate(lines):
                if '# Power coefficient' in line:
                    cp_start_idx = i + 1
                    break
            if cp_start_idx is None:
                raise ValueError("Could not find '# Power coefficient' section in Cp file")
            while cp_start_idx < len(lines) and not lines[cp_start_idx].strip():
                cp_start_idx += 1
            cp_values = []
            for i in range(len(tip_speed_ratios)):
                if cp_start_idx + i >= len(lines):
                    break
                line = lines[cp_start_idx + i].strip()
                if not line:
                    continue
                cp_row = np.array([float(x) for x in line.split() if x.strip()])
                cp_values.append(cp_row)
            if len(cp_values) > 0:
                expected_length = len(pitch_angles)
                cp_values_fixed = []
                for row in cp_values:
                    if len(row) > expected_length:
                        cp_values_fixed.append(row[:expected_length])
                    elif len(row) < expected_length:
                        padded = np.zeros(expected_length)
                        padded[:len(row)] = row
                        cp_values_fixed.append(padded)
                    else:
                        cp_values_fixed.append(row)
                cp_values = np.array(cp_values_fixed)
            else:
                raise ValueError("No Cp data found in Cp file")
            self._cp_interp = RegularGridInterpolator(
                (pitch_angles, tip_speed_ratios),
                cp_values.T,
                method='linear', bounds_error=False, fill_value=0.0
            )
            self._cp_data = True
        if x is None:
            return 0.0

        X = self.local_view(x)
        # omega_m is stored in per-unit (base = omega_m_rated in rad/s)
        omega_m_rad_s = X['omega_m'] * par['omega_m_rated']  # pu speed * base speed -> rad/s
        wind_speed = self.wind_speed(x, v)  # m/s
        tip_speed_ratio = np.where(wind_speed > 0, omega_m_rad_s * par['R'] / wind_speed, 0)

        tsr = float(tip_speed_ratio) if np.isscalar(tip_speed_ratio) else float(tip_speed_ratio.item())
        X = self.local_view(x)
        pitch_angle_val = X['pitch_angle']
        pa = float(pitch_angle_val*180/np.pi) if np.isscalar(pitch_angle_val) else float(pitch_angle_val.item()*180/np.pi)

        pa_clamped = np.clip(pa, self._cp_interp.grid[0].min(), self._cp_interp.grid[0].max())
        tsr_clamped = np.clip(tsr, self._cp_interp.grid[1].min(), self._cp_interp.grid[1].max())

        point = np.array([pa_clamped, tsr_clamped], dtype=np.float64)
        Cp_table = float(self._cp_interp(point)[0])

        return Cp_table

    def load_and_set_Ct(self, x, v):
        """Load the thrust-coefficient table Ct(pitch, TSR) once into _ct_interp.

        Parses the '# Thrust coefficient' block of the ROSCO Cp/Ct/Cq file (same
        pitch/TSR grid as the power-coefficient table used by load_and_set_Cp).
        """
        if hasattr(self, '_ct_interp'):
            return
        par = self.par
        project_root = Path(__file__).parents[3]
        cp_filename = par['Cp_filename'][0] if isinstance(par['Cp_filename'], np.ndarray) else par['Cp_filename']
        path = project_root / 'wind_data' / cp_filename
        with open(path, 'r') as f:
            lines = f.readlines()
        pitch_line = lines[4].strip().lstrip('#').strip()
        pitch_angles = np.array([float(a) for a in pitch_line.split()])
        tsr_line = lines[6].strip().lstrip('#').strip()
        tip_speed_ratios = np.array([float(a) for a in tsr_line.split()])
        ct_start_idx = None
        for i, line in enumerate(lines):
            if 'Thrust coefficient' in line:
                ct_start_idx = i + 1
                break
        if ct_start_idx is None:
            raise ValueError("Could not find 'Thrust coefficient' section in Cp/Ct/Cq file")
        while ct_start_idx < len(lines) and not lines[ct_start_idx].strip():
            ct_start_idx += 1
        ct_values = []
        for i in range(len(tip_speed_ratios)):
            if ct_start_idx + i >= len(lines):
                break
            line = lines[ct_start_idx + i].strip()
            if not line:
                break
            ct_values.append(np.array([float(a) for a in line.split() if a.strip()]))
        expected = len(pitch_angles)
        fixed = []
        for row in ct_values:
            if len(row) > expected:
                fixed.append(row[:expected])
            elif len(row) < expected:
                pad = np.zeros(expected)
                pad[:len(row)] = row
                fixed.append(pad)
            else:
                fixed.append(row)
        ct_values = np.array(fixed)
        self._ct_interp = RegularGridInterpolator(
            (pitch_angles, tip_speed_ratios),
            ct_values.T,
            method='linear', bounds_error=False, fill_value=0.0,
        )

    def wind_speed_init(self):
        """Wind speed at t=0 (m/s)."""
        return 11.0

    def wind_speed(self, x, v):
        return 11.0


class WindTurbineTower2(WindTurbineTower):
    """
    Alias of WindTurbineTower for placing a second, independent single-unit
    turbine on the same grid (mirrors WindTurbine2). Adds no new behaviour.
    """
    pass
