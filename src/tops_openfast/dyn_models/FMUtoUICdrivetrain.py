import os
import numpy as np
from tops.dyn_models.utils import DAEModel
from tops_openfast.dyn_models.speed_lpf import (
    apply_speed_lpf_dynamics,
    resolve_speed_lpf_params,
    speed_pu_for_use,
)
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave


class FMUtoUICdrivetrain(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """ 
        'FMUtoUICdrivetrain': {
            'FMUtoUICdrivetrain': [
                ['name', 'UIC', 'S_n', 'V_n', 'FMU_path', 'fmu_filename', 'control_mode', 'wd_path', 'openfast_test_dir',
                 'J_m', 'J_e', 'K', 'D', 'omega_m_rated', 'fmu_dt', 'ElecPwrCom_kW', 'efficiency',
                 'speed_lpf_type', 'speed_lpf_corner_rad_s', 'speed_lpf_damping'],
                ['FMUtoUICdrivetrain1', 'UIC1', 15, 22, 'FMU_path1', 'fmu_filename1', 3, 'wd_path1', 'openfast_test_dir1',
                 1.0e7, 1.0e6, 7.0e8, 7.0e7, 7.55, 0.01, 20000.0, 0.95756, 2, 1.00810, 0.70000],
            ],
        }

        """
        # extract params and make sure the format is correct
        par = self.par
        sn = par['S_n']
        sn[sn == 0] = self.sys_par['s_n']
        par['S_n'] = sn
        eta = float(np.asarray(par['efficiency']).ravel()[0])
        self._efficiency = eta if np.isfinite(eta) and eta > 0.0 else 1.0
        self._fmu_dt = float(np.asarray(par['fmu_dt']).ravel()[0])
        omega_rated_rpm = float(np.asarray(par['omega_m_rated']).ravel()[0])
        J_m = float(np.asarray(par['J_m']).ravel()[0])
        J_e = float(np.asarray(par['J_e']).ravel()[0])
        K_SI = float(np.asarray(par['K']).ravel()[0])
        D_SI = float(np.asarray(par['D']).ravel()[0])

        # create useful conversion factors
        self._sys_to_local = self.sys_par['s_n'] / par['S_n']
        self._local_to_sys = par['S_n'] / self.sys_par['s_n']
        rpm_to_rad_s = 2.0 * np.pi / 60.0
        self._omega_base_rpm = omega_rated_rpm
        self._omega_base_rad_s = self._omega_base_rpm * rpm_to_rad_s
        self._T_base_Nm = sn * 1e6 / self._omega_base_rad_s

        # convert drivetrain params to pu
        # H_pu = 1/2 * J_SI * omega_base^2 / S_base
        self.H_m = 0.5 * J_m * self._omega_base_rad_s**2 / (sn * 1e6)
        self.H_e = 0.5 * J_e * self._omega_base_rad_s**2 / (sn * 1e6)
        #K_pu = K/T_base, D_pu = D*omega_base/T_base
        par['K'] = K_SI / self._T_base_Nm
        par['D'] = D_SI * self._omega_base_rad_s / self._T_base_Nm

        # Resolve FMU file path:
        # - If FMU_path already points to a .fmu, use it directly.
        # - Otherwise join FMU_path and fmu_filename.
        fmu_filename = str(np.atleast_1d(par['fmu_filename']).ravel()[0])
        fmu_path = str(np.atleast_1d(par['FMU_path']).ravel()[0])
        fmu_file = None
        if fmu_path and fmu_path.lower().endswith('.fmu'):
            fmu_file = fmu_path
        elif fmu_path and fmu_filename:
            fmu_file = os.path.join(fmu_path, fmu_filename)
        elif fmu_filename:
            # Backwards compatibility: allow callers that pass a full path as fmu_filename.
            fmu_file = fmu_filename
        else:
            raise KeyError("FMUtoUICdrivetrain requires 'FMU_path' and/or 'fmu_filename' to locate the .fmu.")

        if not os.path.isfile(fmu_file):
            raise FileNotFoundError(f"FMU file not found: {fmu_file}")

        model_description = read_model_description(fmu_file, validate=False)

        vrs = {}
        for variable in model_description.modelVariables:
            vrs[variable.name] = variable.valueReference

        print("Value References: \n")
        for name, vr in vrs.items():
            print(f"Variable: {name}, Value Reference: {vr}")

        unzipdir = extract(fmu_file)

        # OpenFAST fmu reads wd.txt from inside the extracted fmu resources folder
        # Writing to an arbitrary path in the repo (par['wd_path']) will not affect what the fmu reads
        new_directory = str(np.atleast_1d(par['openfast_test_dir']).ravel()[0])
        wd_file_path_in_fmu = os.path.join(unzipdir, 'resources', 'wd.txt')
        os.makedirs(os.path.dirname(wd_file_path_in_fmu), exist_ok=True)
        with open(wd_file_path_in_fmu, 'w') as f:
            f.write(new_directory)

        # also write to the user-provided path for visibility/debugging
        try:
            wd_file_path = str(np.atleast_1d(par['wd_path']).ravel()[0])
            if wd_file_path:
                os.makedirs(os.path.dirname(wd_file_path), exist_ok=True)
                with open(wd_file_path, 'w') as f:
                    f.write(new_directory)
        except Exception:
            pass

        fmu = FMU2Slave(guid=model_description.guid,
                        unzipDirectory=unzipdir,
                        modelIdentifier=model_description.coSimulation.modelIdentifier,
                        instanceName='instance1')

        if 'control_mode' not in par.dtype.names:
            raise KeyError("FMUtoUICdrivetrain requires parameter 'control_mode'.")
        control_mode = int(np.atleast_1d(par['control_mode']).ravel()[0])

        if 'testNr' not in par.dtype.names:
            raise KeyError("FMUtoUICdrivetrain requires parameter 'testNr'.")
        testNr = int(np.atleast_1d(par['testNr']).ravel()[0])

        fmu.instantiate()
        fmu.setReal([vrs['testNr']], [int(testNr)])
        fmu.setReal([vrs['Mode']], [int(control_mode)])

        print(f"[FMUtoUICdrivetrain] Using FMU: {fmu_file}", flush=True)
        fmu.setupExperiment(startTime=0.0)
        fmu.enterInitializationMode()

        fmu.exitInitializationMode()

        self.fmu = fmu
        self.vrs = vrs
        if not np.isfinite(self._fmu_dt) or self._fmu_dt <= 0.0:
            raise ValueError(f"Invalid 'fmu_dt'={self._fmu_dt}. Must be a positive finite float (s).")
        self._last_fmu_comm_point = None
        self._fmu_warm_stepped = False
        self._primed_at_init = False
        # Cached fmu measurements (updated once per tops step in step_fmu)
        # Avoid reading fmu inside state_derivatives() since the solver may call it multiple times per step
        self._omega_m_pu_meas = None
        self._Te_pu_cmd = None
        self._gen_speed_rpm_meas = None
        self._gen_tq_kNm_meas = None
        self._gen_spdortrq_kNm_set = None
        self._genpwr_kW_set = None

        # Startup workaround for OpenFAST FMU internal offset compensation:
        # For the first N FMU macro-steps, echo the FMU's own measured GenTq back into GenSpdOrTrq.
        # This makes (ideal - input) ~ 0 inside the FMU during the "activate==false" window.
        self._startup_echo_steps_total = 10
        self._startup_echo_steps_left = int(self._startup_echo_steps_total)

        # Electrical power command to controller (kW)
        if 'ElecPwrCom_kW' not in par.dtype.names:
            raise KeyError("FMUtoUICdrivetrain requires parameter 'ElecPwrCom_kW' (kW).")
        self._elec_pwr_com_kW = float(np.atleast_1d(par['ElecPwrCom_kW']).ravel()[0])
        if not np.isfinite(self._elec_pwr_com_kW) or self._elec_pwr_com_kW < 0.0:
            raise ValueError(
                f"Invalid 'ElecPwrCom_kW'={self._elec_pwr_com_kW}. Must be a finite float >= 0 (kW)."
            )

        n_units = int(np.asarray(par["S_n"]).size)
        (
            self._speed_lpf_type,
            self._speed_lpf_corner_rad_s,
            self._speed_lpf_damping,
        ) = resolve_speed_lpf_params(par, n_units)

    def _pe_local_pu(self, x, v) -> float:
        par = self.par
        pe_uic = float(np.asarray(self.P_e(x, v)).ravel()[0])
        s_n_uic = float(np.asarray(self.S_n_UIC(x, v)).ravel()[0])
        s_n_loc = float(np.asarray(par["S_n"]).ravel()[0])
        return pe_uic * (s_n_uic / s_n_loc) if s_n_loc > 0 else pe_uic

    def _te_pu(self, X, pe_pu: float) -> float:
        lpf_type = int(np.asarray(self._speed_lpf_type).ravel()[0])
        omega_for_te = speed_pu_for_use(X, "omega_e", "omega_e_filt", lpf_type)
        if not np.isfinite(omega_for_te) or abs(omega_for_te) <= 1e-6:
            return 0.0
        return float(pe_pu) / (self._efficiency * omega_for_te)

    def connections(self):
        # tops convention of input and output to connected model; here uic is connected
        # P_e is elec power output from uic, S_n_UIC is UIC base power: UIC -> FMUtoUICdrivetrain
        # P_ref is the power reference for the UIC: FMUtoUICdrivetrain -> UIC
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
        # Grid coupling (same as WindTurbine): omega_e_filt for Te = Pe/(omega_e_filt*eta).
        return [
            "omega_e",
            "theta_s",
            "omega_e_filt",
            "omega_e_filt_dot",
        ]

    def input_list(self):
        return ['P_e', 'S_n_UIC'] 

    def output_list(self):
        return ['P_ref']
    
    def init_from_connections(self, x_0, v_0, S):
        self._input_values["P_e"] = self.P_e(x_0, v_0)
        self._input_values["S_n_UIC"] = self.S_n_UIC(x_0, v_0)

        # Initialize drivetrain states (do not read FMU outputs before a doStep()).
        X = self.local_view(x_0)
        par = self.par

        # Electrical torque requested by coupling at init (local pu); shaft angle set after FMU prime.
        Pe_pu = self._pe_local_pu(x_0, v_0)
        omega_e0 = 1.0  # guess until FMU GenSpeed is read after prime
        Te_pu = Pe_pu / (self._efficiency * omega_e0) if abs(omega_e0) > 1e-6 else 0.0
        self._Te_pu_cmd = Te_pu
        om_guess = np.asarray(omega_e0, dtype=float).ravel()
        X["omega_e_filt"] = om_guess.copy()
        X["omega_e_filt_dot"] = np.zeros_like(om_guess)

        # Prime the FMU once before the dynamic simulation loop.
        # OpenFAST FMU outputs are often zeros until the first doStep().
        # We advance the FMU from 0 -> dt using the initial coupling inputs.
        # Set env var FMU_CACHE_AFTER_PRIME=1 to latch FMU outputs after the prime step.
        if not self._primed_at_init:
            dt0 = float(self._fmu_dt)
            if not np.isfinite(dt0) or dt0 <= 0.0:
                raise ValueError(f"FMUtoUICdrivetrain: invalid fmu_dt={dt0}.")

            # Electrical power (kW) on WT base: Pe_pu is local (WT) pu.
            s_n_loc_mva = float(np.asarray(par['S_n']).ravel()[0])
            P_e_kW0 = float(np.asarray(Pe_pu).ravel()[0]) * s_n_loc_mva * 1e3
            self._genpwr_kW_set = float(P_e_kW0)
            #self.fmu.setReal([self.vrs['GenPwr']], [float(P_e_kW0)]) # unsure if these should be set here as the fmu should get readback vals the first iterations

            Te_kNm_cmd0 = float(np.asarray(Te_pu).ravel()[0]) * float(np.asarray(self._T_base_Nm).ravel()[0]) / 1e3
            self._gen_spdortrq_kNm_set = float(Te_kNm_cmd0)
            #self.fmu.setReal([self.vrs['GenSpdOrTrq']], [float(Te_kNm_cmd0)])

            self._elec_pwr_com_kW_last = float(self._elec_pwr_com_kW)
            #self.fmu.setReal([self.vrs['ElecPwrCom']], [float(self._elec_pwr_com_kW_last)])

            # Advance FMU from t=0 to t=dt0.
            self.fmu.doStep(currentCommunicationPoint=0.0, communicationStepSize=dt0)
            self._last_fmu_comm_point = 0.0
            self._primed_at_init = True
            
            rot_rpm = float(self.fmu.getReal([self.vrs['RotSpeed']])[0])
            self._omega_m_pu_meas = rot_rpm / self._omega_base_rpm
            self._gen_speed_rpm_meas = float(self.fmu.getReal([self.vrs['GenSpeed']])[0])
            self._gen_tq_kNm_meas = float(self.fmu.getReal([self.vrs['GenTq']])[0])

        X["omega_e"] = self._gen_speed_rpm_meas / self._omega_base_rpm
        om_e = np.asarray(X["omega_e"], dtype=float).ravel()
        X["omega_e_filt"] = om_e.copy()
        X["omega_e_filt_dot"] = np.zeros_like(om_e)
        Pe_pu = self._pe_local_pu(x_0, v_0)
        self._Te_pu_cmd = self._te_pu(X, Pe_pu)
        K_pu = float(np.asarray(par['K']).ravel()[0])
        X['theta_s'] = self._Te_pu_cmd / K_pu if K_pu > 0.0 else 0.0

        return

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par

        omega_e = float(np.asarray(X['omega_e']).ravel()[0]) # current tops state of generator speed
        omega_m = omega_e if self._omega_m_pu_meas is None else float(self._omega_m_pu_meas) # from OpenFAST (cached)
        theta_s = float(np.asarray(X['theta_s']).ravel()[0]) # current tops state of shaft twist angle

        if not np.isfinite(omega_e):
            raise ValueError(f"FMUtoUICdrivetrain: omega_e is not finite (omega_e={omega_e}).")

        Pe_pu = self._pe_local_pu(x, v)
        Te_pu = self._te_pu(X, Pe_pu)
        self._Te_pu_cmd = float(Te_pu)

        # shaft torque
        omega_s = omega_m - omega_e
        K_pu = float(np.asarray(par['K']).ravel()[0])
        D_pu = float(np.asarray(par['D']).ravel()[0])
        T_shaft = (K_pu * theta_s + D_pu * omega_s) # shaft twist torque

        # swing eqs for drivetrain dynamics (pu)
        if not np.isfinite(T_shaft): # avoid simulations with NaN values
            raise ValueError(f"FMUtoUICdrivetrain: T_shaft is not finite (T_shaft={T_shaft}).")
        if not np.isfinite(Te_pu):
            raise ValueError(f"FMUtoUICdrivetrain: Te_pu is not finite (Te_pu={Te_pu}).")

        d_omega_e = (1 / (2.0 * self.H_e)) * (T_shaft - Te_pu)
        if not np.isfinite(d_omega_e):
            raise ValueError(
                "FMUtoUICdrivetrain: d(omega_e) is not finite "
                f"(d_omega_e={d_omega_e}, H_e={self.H_e}, T_shaft={T_shaft}, Te_pu={Te_pu})."
            )
        dX["omega_e"] = d_omega_e
        dX["theta_s"] = omega_s

        lpf_type = int(np.asarray(self._speed_lpf_type).ravel()[0])
        omega_c = float(np.asarray(self._speed_lpf_corner_rad_s).ravel()[0])
        zeta = float(np.asarray(self._speed_lpf_damping).ravel()[0])
        apply_speed_lpf_dynamics(
            dX, X, np.asarray(X["omega_e"]).ravel(), "omega_e_filt", "omega_e_filt_dot", lpf_type, omega_c, zeta
        )

        return

    # FMU output names from modelDescription.xml (causality="output")
    FMU_OUTPUT_NAMES = [
        'Time', 'HSShftTq', 'GenTq', 'Wind1VelX', 'RtVAvgxh', 'BldPitch1',
        'NacYaw', 'RefGenSpd', 'GenSpeed', 'RotSpeed', 'LSSGagPxa', 'Azimuth',
        'GenAccel', 'YawBrTAxp', 'YawBrTAyp', 'RtAeroMxh',
    ]

    def get_all_fmu_outputs(self):
        """Return dict of all FMU outputs (from modelDescription.xml)."""
        names = [n for n in self.FMU_OUTPUT_NAMES if n in self.vrs]
        if not names:
            return {}
        vrefs = [self.vrs[n] for n in names]
        vals = self.fmu.getReal(vrefs)
        return dict(zip(names, vals))

    def P_ref(self, x, v):
        # P_ref (UIC pu, electrical) = eta * P_mech; GenTq is mechanical (kN·m), same as WindTurbine MPT.
        # Use cached OpenFAST outputs to avoid repeated FMU reads during one solver step.
        X = self.local_view(x)
        lpf_type = int(np.asarray(self._speed_lpf_type).ravel()[0])
        omega_for_pref = speed_pu_for_use(X, 'omega_e', 'omega_e_filt', lpf_type)
        #omega_e_pu = float(np.asarray(X['omega_e']).ravel()[0])
        if self._gen_tq_kNm_meas is None:
            return np.atleast_1d(0.0)
        P_mech_kW = float(self._gen_tq_kNm_meas) * omega_for_pref * self._omega_base_rad_s
        S_n_MVA = float(np.asarray(self.S_n_UIC(x, v)).ravel()[0])
        p_ref_pu = (P_mech_kW / (S_n_MVA * 1e3)) * self._efficiency
        return np.atleast_1d(p_ref_pu)
    
    def step_fmu(self, x, v, t, dt):
        # NB! Because of a built-in function in fmu, any offset in the first 10 iterations will be compensated.
        # any disturbance should be applied after the first 10 iterations.
        par = self.par
        X = self.local_view(x)

        # Provide measured electrical power from the grid/UIC (kW on UIC base).
        P_e_uic_pu = float(np.asarray(self.P_e(x, v)).ravel()[0])
        S_n_uic_MVA = float(np.asarray(self.S_n_UIC(x, v)).ravel()[0])
        P_e_kW = P_e_uic_pu * S_n_uic_MVA * 1e3
        input_power_kW = float(P_e_kW)

        Pe_pu = self._pe_local_pu(x, v)
        Te_pu = self._te_pu(X, Pe_pu)
        self._Te_pu_cmd = Te_pu

        # Send coupling torque (kN·m) into OpenFAST-FMU
        Te_kNm_cmd = float(Te_pu) * self._T_base_Nm / 1e3
        input_torque = float(Te_kNm_cmd[0])

        # Startup: echo measured FMU torque back as input for a few steps
        # to avoid the FMU's internal offset compensation latching a non-zero bias.
        if self._startup_echo_steps_left > 0:
            # Read the FMU's current "ideal" torque directly right before sending the input.
            # This is limited to the first few steps only, so the extra FMU read cost is negligible.
            tq_ideal = float(self.fmu.getReal([self.vrs['GenTq']])[0])
            if np.isfinite(tq_ideal):
                input_torque = tq_ideal
            input_power_kW = tq_ideal * self._omega_base_rad_s / 1e3
            # Keep GenPwr driven from TOPS (grid measurement) unless we explicitly want to echo power as well.
            # (Computing an "ideal power" from torque requires a reliable GenSpeed sample; cached value may be None/stale.)

        # Optional diagnostic modulation of the coupling (generator) torque
        # command. This directly forces the OpenFAST drivetrain torque at a
        # chosen frequency (e.g. to probe the side-to-side tower mode),
        # bypassing the weak grid -> torque path. The driver sets
        # self._te_mod_factor before each step_fmu() call; the default of 1.0
        # (when the attribute is absent) leaves the coupling torque unchanged.
        # It is not applied during the startup echo window.
        te_mod_factor = float(getattr(self, "_te_mod_factor", 1.0))
        if te_mod_factor != 1.0 and self._startup_echo_steps_left <= 0:
            input_torque = float(input_torque) * te_mod_factor

        self.fmu.setReal([self.vrs['GenSpdOrTrq']], [input_torque])
        self._gen_spdortrq_kNm_set = input_torque
        self._genpwr_kW_set = input_power_kW
        self.fmu.setReal([self.vrs['GenPwr']], [input_power_kW])
        
        # Demanded electrical power (kW) for controller.
        self._elec_pwr_com_kW_last = float(self._elec_pwr_com_kW)

        # Optional diagnostic modulation of the demanded electrical power
        # (ElecPwrCom) command. With VSContrl=5 (ROSCO), the external generator
        # torque input is ignored, but ElecPwrCom is the grid-side power demand
        # the controller does act on. Modulating it at the side-to-side tower
        # frequency represents a grid/converter power-command oscillation that
        # the controller translates into a generator-torque variation, which
        # reacts on the nacelle and can excite the side-to-side tower mode.
        # The driver sets self._epc_mod_factor before each step_fmu() call;
        # the default of 1.0 leaves the command unchanged. Not applied during
        # the startup echo window.
        epc_mod_factor = float(getattr(self, "_epc_mod_factor", 1.0))
        if epc_mod_factor != 1.0 and self._startup_echo_steps_left <= 0:
            self._elec_pwr_com_kW_last = float(self._elec_pwr_com_kW_last) * epc_mod_factor

        self.fmu.setReal([self.vrs['ElecPwrCom']], [float(self._elec_pwr_com_kW_last)])

        if abs(float(dt) - self._fmu_dt) > 1e-12:
            raise ValueError(f"FMUtoUICdrivetrain requires dt == fmu_dt. Got dt={dt}, fmu_dt={self._fmu_dt}.")
        comm_point = float(t) - float(dt)
        if comm_point < -1e-12:
            raise ValueError(f"Invalid FMU communication point t-dt={comm_point} (t={t}, dt={dt}).")

        # If we primed the FMU in init_from_connections() with doStep(0, dt),
        # then the first runtime call occurs at t=dt with comm_point=0.
        # In that case, skip doStep to avoid advancing 0->dt twice, but latch outputs so caches are set.
        if self._primed_at_init and abs(comm_point) <= 1e-12:
            rot_rpm = float(self.fmu.getReal([self.vrs['RotSpeed']])[0])
            self._omega_m_pu_meas = rot_rpm / self._omega_base_rpm
            self._gen_speed_rpm_meas = float(self.fmu.getReal([self.vrs['GenSpeed']])[0])
            self._gen_tq_kNm_meas = float(self.fmu.getReal([self.vrs['GenTq']])[0])
            self._primed_at_init = False
            if self._startup_echo_steps_left > 0:
                self._startup_echo_steps_left -= 1
            return
        # check if the FMU time is monotone -> t+1 > t
        if self._last_fmu_comm_point is not None and comm_point < self._last_fmu_comm_point - 1e-12:
            raise ValueError(f"Non-monotone FMU time: {comm_point} < last {self._last_fmu_comm_point}.")
        self._last_fmu_comm_point = comm_point

        self.fmu.doStep(currentCommunicationPoint=comm_point, communicationStepSize=dt)
        if self._startup_echo_steps_left > 0:
            self._startup_echo_steps_left -= 1

        # Cache measurements for the next solver step
        rot_rpm = float(self.fmu.getReal([self.vrs['RotSpeed']])[0])
        self._omega_m_pu_meas = rot_rpm / self._omega_base_rpm
        self._gen_speed_rpm_meas = float(self.fmu.getReal([self.vrs['GenSpeed']])[0])
        self._gen_tq_kNm_meas = float(self.fmu.getReal([self.vrs['GenTq']])[0])

        return

    def terminate_fmu(self):
        self.fmu.terminate()
        self.fmu.freeInstance()