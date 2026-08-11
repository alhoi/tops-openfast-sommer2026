def load():
    return {
        'base_mva': 10,
        'f': 50,

        # The only electrical network bus
        'slack_bus': 'WT1_LEOGO',

        'buses': [
            ['name',        'V_n'],
            ['WT1_LEOGO',   22],
        ],

        # No transmission lines remain
        'lines': [
            ['name', 'from_bus', 'to_bus', 'length', 'S_n', 'V_n',
             'unit', 'R', 'X', 'B'],
        ],

        # No separate load remains
        'loads': [
            ['name', 'bus', 'P', 'Q', 'model'],
        ],

        # Keep this only if WT1_LEOGO is connected to an external/infinite grid.
        # It represents the grid behind the LEOGO connection point.
        'generators': {
            'GEN': [
                ['name', 'bus', 'S_n', 'V_n', 'P', 'V', 'H', 'D',
                 'X_d', 'X_q', 'X_d_t', 'X_q_t',
                 'X_d_st', 'X_q_st',
                 'T_d0_t', 'T_q0_t', 'T_d0_st', 'T_q0_st'],

                ['IB_LEOGO', 'WT1_LEOGO', 10e8, 22, 0, 1,
                 1e5, 0,
                 1.05, 0.66, 0.328, 0.66,
                 1e-5, 1e-5,
                 1e5, 10000, 1e5, 1e5],
            ],
        },

        'vsc': {
            'UIC_sig': [
                ['name', 'bus', 'S_n', 'V_n', 'v_ref', 'p_ref', 'q_ref',
                 'Ki', 'Kv', 'xf', 'perfect_tracking', 'T_filter'],

                # Changed only: B2 -> WT1_LEOGO
                ['UIC1', 'WT1_LEOGO', 20, 22,
                 1.0, 0.5, 0.0,
                 0.03, 0.0, 0.1, 1, 0.1],
            ],
        },

        'windturbine': {
            'WindTurbine': [
                [
                    'name', 'UIC', 'S_n', 'V_n',
                    'J_m', 'J_e', 'K', 'D',
                    'Kp_pitch', 'Ki_pitch', 'T_pitch',
                    'max_pitch', 'min_pitch', 'max_pitch_rate',
                    'rho', 'R', 'P_rated', 'omega_m_rated',
                    'wind_rated', 'efficiency',
                    'MPT_filename', 'Cp_filename',
                    'speed_lpf_type', 'speed_lpf_corner_rad_s',
                    'speed_lpf_damping',
                    'f_nom_hz',
                    'droop_enable',
                    'K_droop_pu_per_hz',
                    'headroom_pu',
                    'scheduled_power_enable',
                    'P_schedule_uic_pu',
                ],

                [
                    'WT1', 'UIC1', 15, 22,
                    352460500., 1836784.,
                    69737644900./100., 35698200.0/10.,
                    0.6738, 0.06, 2.2,
                    30.0, 0.0, 10.0,
                    1.225, 120.97, 1.0,
                    7.559987120819503,
                    10.6, 0.95756,
                    'MPT_Kopt2150.csv',
                    'Cp_Ct_Cq.IEA15MW.ROSCO.txt',
                    2, 1.00810, 0.70000,

                    # Droop-parametere
                    50.0,   # f_nom_hz
                    0,      # droop_enable
                    0.0,    # K_droop_pu_per_hz
                    0.0,    # headroom_pu
                    1.0,    # scheduled_power_enable
                    0.40,   # P_schedule_uic_pu = 8 MW / 20 MVA 
                ],
            ],
        },
    }
