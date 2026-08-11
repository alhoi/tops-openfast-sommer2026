def load():
    return {
        'base_mva': 100,
        'f': 50,
        'slack_bus': 'Main Bus A',

        'buses': [
            ['name',    'V_n'],
            ['Busbar WTG3 HV', 33], # idx 0
            ['Busbar WTG2 HV', 33],
            ['Busbar WTG1 HV', 33],
            ['Busbar WTG3 LV', 0.69],
            ['Busbar WTG2 LV', 0.69],
            ['Busbar WTG1 LV', 0.69],
            ['Terminal WindPark', 33],
            ['SG Terminal 1', 11],
            ['SG Terminal 2', 11],
            ['SG Terminal 3', 11],
            ['Main Bus A', 11],
            #['Main Bus B', 11],
            ['Drilling AC Bus A', 11],
            ['Drilling AC Bus B', 11],
            ['Terminal AC High Drilling A', 11],
            ['Terminal AC High Drilling B', 11],
            ['Terminal AC High PEC_VSDc_OEX_01', 11],
            ['Terminal AC Low Drilling A', 0.69],
            ['Terminal AC Low Drilling B', 0.69],
            ['Terminal AC High PEC_VSDc_OEX_02', 11],
            ['Terminal SWL_01', 11],
            ['Terminal ACO_01', 11],
            ['Terminal ACO_02', 11],
            ['Terminal AC High PEC_VSDc_GEX_01', 11],
            ['Terminal TRA_UTL690_02', 11],
            ['Terminal TRA_UTL690_01', 11],
            ['Terminal AC High PEC_VSDc_WIN_01', 11],
            ['Terminal AC High PEC_VSDc_WIN_02', 11],
            ['Terminal AC High PEC_VSDc_WIN_03', 11],
            ['Terminal AC High PEC_VSDc_REC', 11],
            ['Terminal AC High PEC_VSDc_GEX_02', 11],
            ['Terminal AC High PEC_VSDc_GEX_03', 11],
            ['Terminal AC Low PEC_VSDc_OEX_01', 0.69],
            ['Terminal AC Low PEC_VSDc_OEX_02', 0.69],
            ['Terminal AC Low PEC_VSDc_GEX_01', 3.3],
            ['Terminal AC Low PEC_VSDc_WIN_01', 3.3],
            ['Terminal AC Low PEC_VSDc_WIN_02', 3.3],
            ['Terminal AC Low PEC_VSDc_WIN_03', 3.3],
            ['Terminal AC Low PEC_VSDc_GEX_02', 3.3],
            ['Terminal AC Low PEC_VSDc_GEX_03', 3.3],
            ['Terminal AC Low PEC_VSDc_REC', 3.3],
            ['Utility690 Bus B', 0.69],
            ['Utility690 Bus A', 0.69],
            ['Terminal TRA_UTL400_02', 0.69],
            ['Terminal TRA_UTL400_01', 0.69],
            ['Utility400 SWBD/BusB', 0.4],
            ['Utility400 SWBD/BusA', 0.4] # idx 46

                        

        ],

        'lines': [
            ['name', 'from_bus', 'to_bus', 'length', 'S_n', 'V_n', 'unit', 'R', 'X', 'B'],
            ['Cable WTG1', 'Busbar WTG1 HV', 'Terminal WindPark', 9, 100, 33, 'pf', 0.0786, 0.102541, 0.3], # 0
            ['Cable WTG2', 'Busbar WTG2 HV', 'Busbar WTG1 HV', 1.3, 100, 33, 'pf', 0.0786, 0.102541, 0.3],
            ['Cable WTG3', 'Busbar WTG3 HV', 'Busbar WTG2 HV', 1.3, 100, 33, 'pf', 0.0786, 0.102541, 0.3],
            ['Line GEX_02', 'Main Bus A', 'Terminal AC High PEC_VSDc_GEX_02', 0.25, 100, 11, 'pf', 0.0786, 0.094247, 0.37],
            ['Line GEX_03', 'Main Bus A', 'Terminal AC High PEC_VSDc_GEX_03', 0.25, 100, 11, 'pf', 0.0786, 0.094247, 0.37],
            ['Line OEX_02', 'Main Bus A', 'Terminal AC High PEC_VSDc_OEX_02', 0.25, 100, 11, 'pf', 0.1555, 0.103672, 0.28],
            ['Line WIN_03', 'Main Bus A', 'Terminal AC High PEC_VSDc_WIN_03', 0.15, 100, 11, 'pf', 0.1555, 0.103672, 0.28],
            ['Line GEX_01', 'Main Bus A', 'Terminal AC High PEC_VSDc_GEX_01', 0.05, 100, 11, 'pf', 0.0786, 0.094247, 0.37],
            ['Line OEX_01', 'Main Bus A', 'Terminal AC High PEC_VSDc_OEX_01', 0.05, 100, 11, 'pf', 0.1555, 0.103672, 0.28],
            ['Line WIN_01', 'Main Bus A', 'Terminal AC High PEC_VSDc_WIN_01', 0.15, 100, 11, 'pf', 0.1555, 0.103672, 0.28],
            ['Line WIN_02', 'Main Bus A', 'Terminal AC High PEC_VSDc_WIN_02', 0.15, 100, 11, 'pf', 0.1555, 0.103672, 0.28],
            ['Line REC', 'Main Bus A', 'Terminal AC High PEC_VSDc_REC', 0.2, 100, 11, 'pf', 0.1555, 0.103672, 0.28],
            ['Line ACO_02', 'Main Bus A', 'Terminal ACO_02', 0.25, 100, 11, 'pf', 0.1555, 0.103672, 0.28],
            ['Line SWL_01', 'Main Bus A', 'Terminal SWL_01', 0.25, 100, 11, 'pf', 0.1555, 0.103672, 0.28],
            ['Line ACO_01', 'Main Bus A', 'Terminal ACO_01', 0.25, 100, 11, 'pf', 0.1555, 0.103672, 0.28],
            ['Line Utility B', 'Main Bus A', 'Terminal TRA_UTL690_02', 0.15, 100, 11, 'pf', 0.1555, 0.103672, 0.28],
            ['Line Utility A', 'Main Bus A', 'Terminal TRA_UTL690_01', 0.2, 100, 11, 'pf', 0.1555, 0.103672, 0.28],
            ['Line 690 Trafo B', 'Utility690 Bus B', 'Terminal TRA_UTL400_02', 0.15, 100, 0.69, 'pf', 0.0786, 0.094247, 0.37],
            ['Line 690 Trafo A', 'Utility690 Bus A', 'Terminal TRA_UTL400_01', 0.15, 100, 0.69, 'pf', 0.0786, 0.094247, 0.37],
            ['Line SG1', 'SG Terminal 1', 'Main Bus A', 0.05, 100, 11, 'pf', 0.0786, 0.094247, 0.37],
            ['Line SG2', 'SG Terminal 2', 'Main Bus A', 0.05, 100, 11, 'pf', 0.0786, 0.094247, 0.37],
            ['Line SG3', 'SG Terminal 3', 'Main Bus A', 0.05, 100, 11, 'pf', 0.0786, 0.094247, 0.37], 
            ['Line Drilling B', 'Main Bus A', 'Drilling AC Bus B', 0.25, 100, 11, 'pf', 0.1555, 0.103672, 0.28],
            ['Line Drilling A', 'Main Bus A', 'Drilling AC Bus A', 0.2, 100, 11, 'pf', 0.1555, 0.103672, 0.28],
            ['Line AC Drilling B', 'Drilling AC Bus B', 'Terminal AC High Drilling B', 0.05, 100, 11, 'pf', 0.1555, 0.103672, 0.28],
            ['Line AC Drilling A', 'Drilling AC Bus A', 'Terminal AC High Drilling A', 0.05, 100, 11, 'pf', 0.1555, 0.103672, 0.28]
            #['Switch Main A Main B', 'Main Bus A', 'Main Bus A', 1, 100, 11, 'pf', 0.0786, 0.094247, 0.37] #6.538

        ],

        'transformers': [
            ['name', 'from_bus', 'to_bus', 'S_n', 'V_n_from', 'V_n_to', 'R', 'X'],
            ['Trafo WTG1', 'Busbar WTG1 LV', 'Busbar WTG1 HV', 10, 0.69, 33, 0.005, 0.0597913],
            ['Trafo WTG2', 'Busbar WTG2 LV', 'Busbar WTG2 HV', 10, 0.69, 33, 0.005, 0.0597913],
            ['Trafo WTG3', 'Busbar WTG3 LV', 'Busbar WTG3 HV', 10, 0.69, 33, 0.005, 0.0597913],
            ['Trafo WindPark', 'Terminal WindPark', 'Main Bus A', 25, 33, 11, 0.002, 0.08997778],
            ['Trafo GEX_01', 'Terminal AC High PEC_VSDc_GEX_01', 'Terminal AC Low PEC_VSDc_GEX_01', 10.5, 11, 3.3, 0.0047619, 0.05981074],
            ['Trafo GEX_02', 'Terminal AC High PEC_VSDc_GEX_02', 'Terminal AC Low PEC_VSDc_GEX_02', 10.5, 11, 3.3, 0.0047619, 0.05981074],
            ['Trafo GEX_03', 'Terminal AC High PEC_VSDc_GEX_03', 'Terminal AC Low PEC_VSDc_GEX_03', 10.5, 11, 3.3, 0.0047619, 0.05981074],
            ['Trafo OEX_01', 'Terminal AC High PEC_VSDc_OEX_01', 'Terminal AC Low PEC_VSDc_OEX_01', 1.6, 11, 0.69, 0.0079375, 0.05947265],
            ['Trafo OEX_02', 'Terminal AC High PEC_VSDc_OEX_02', 'Terminal AC Low PEC_VSDc_OEX_02', 1.6, 11, 0.69, 0.0079375, 0.05947265],
            ['Trafo REC', 'Terminal AC High PEC_VSDc_REC', 'Terminal AC Low PEC_VSDc_REC', 5, 11, 3.3, 0.006, 0.05969924],
            ['Trafo WIN_01', 'Terminal AC High PEC_VSDc_WIN_01', 'Terminal AC Low PEC_VSDc_WIN_01', 5, 11, 3.3, 0.006, 0.05969924],
            ['Trafo WIN_02', 'Terminal AC High PEC_VSDc_WIN_02', 'Terminal AC Low PEC_VSDc_WIN_02', 5, 11, 3.3, 0.006, 0.05969924],
            ['Trafo WIN_03', 'Terminal AC High PEC_VSDc_WIN_03', 'Terminal AC Low PEC_VSDc_WIN_03', 5, 11, 3.3, 0.006, 0.05969924],
            ['TRA_UTL690_02', 'Terminal TRA_UTL690_02', 'Utility690 Bus B', 3.3, 11, 0.69, 0.00606061, 0.1098329],
            ['TRA_UTL690_01', 'Terminal TRA_UTL690_01', 'Utility690 Bus A', 3.3, 11, 0.69, 0.00606061, 0.1098329],
            ['TRA_UTL400_02', 'Terminal TRA_UTL400_02', 'Utility400 SWBD/BusB', 0.6, 0.69, 0.4, 0.00833333, 0.05941848],
            ['TRA_UTL400_01', 'Terminal TRA_UTL400_01', 'Utility400 SWBD/BusA', 0.6, 0.69, 0.4, 0.00833333, 0.05941848],
            ['TRA_Dril_02', 'Terminal AC High Drilling B', 'Terminal AC Low Drilling B', 3.3, 11, 0.69, 0.00606061, 0.1098329],
            ['TRA_Dril_01', 'Terminal AC High Drilling A', 'Terminal AC Low Drilling A', 3.3, 11, 0.69, 0.00606061, 0.1098329]

        ],

        'loads': [
            ['name', 'bus', 'P', 'Q', 'model'],
            ['DC Load - VSD_GEX_02', 'Terminal AC Low PEC_VSDc_GEX_02', 9.7, 0, 'Z'],
            ['DC Load - VSD_GEX_03', 'Terminal AC Low PEC_VSDc_GEX_03', 9.7, 0, 'Z'],
            #['Additional DC Load 1', 'Main Bus A', 15, 0, 'Z'],
            ['DC Load - VSD_OEX_02', 'Terminal AC Low PEC_VSDc_OEX_02', 0.39, 0, 'Z'], 
            ['DC Load - VSD_WIN_03', 'Terminal AC Low PEC_VSDc_WIN_03', 3, 0, 'Z'],
            ['DC Load - VSD_GEX_01', 'Terminal AC Low PEC_VSDc_GEX_01', 9.7, 0, 'Z'],
            ['DC Load - VSD_OEX_01', 'Terminal AC Low PEC_VSDc_OEX_01', 0.39, 0, 'Z'],
            ['DC Load - VSD_WIN_01', 'Terminal AC Low PEC_VSDc_WIN_01', 3, 0, 'Z'],
            ['DC Load - VSD_WIN_02', 'Terminal AC Low PEC_VSDc_WIN_02', 3, 0, 'Z'],
            ['DC Load - VSD_REC', 'Terminal AC Low PEC_VSDc_REC', 3.8, 0, 'Z'],
            ['LOD2 - Utility Load 690V B', 'Utility690 Bus B', 1.05, 0.5, 'Z'],
            ['LOD4 - Utility Load 400V B', 'Utility400 SWBD/BusB', 0.25, 0, 'Z'],
            ['LOD1 - Utility Load 690V A', 'Utility690 Bus A', 1.05, 0.5, 'Z'],
            ['LOD3 - Utility Load 400V A', 'Utility400 SWBD/BusA', 0.25, 0, 'Z'],
            ['ASM 690V B', 'Utility690 Bus B', 0.2, 0.1, 'Z'],
            ['ASM 690V A', 'Utility690 Bus A', 0.2, 0.1, 'Z'],
            ['ASM_SWL_01', 'Terminal SWL_01', 0.5, 0.3, 'Z'],
            ['ASM_ACO_02', 'Terminal ACO_02', 0.5, 0.4, 'Z'],
            ['ASM ACO_01', 'Terminal ACO_01', 0.5, 0.4, 'Z']
    
        ],

        'shunts': [
            ['name', 'bus', 'V_n', 'Q', 'model'],
            ['LCL-Kondensator Drilling B', 'Terminal AC Low Drilling B', 0.69, 0.139, 'Z'],
            ['LCL-Kondensator GEX_01', 'Terminal AC Low PEC_VSDc_GEX_01', 3.3, 0.435, 'Z'],
            ['LCL-Kondensator GEX_02','Terminal AC Low PEC_VSDc_GEX_02', 3.3, 0.435, 'Z'],
            ['LCL-Kondensator GEX_03', 'Terminal AC Low PEC_VSDc_GEX_03', 3.3, 0.435, 'Z'],
            ['LCL-Kondensator OEX_01', 'Terminal AC Low PEC_VSDc_OEX_01', 0.69, 0.0672, 'Z'],
            ['LCL-Kondensator OEX_02', 'Terminal AC Low PEC_VSDc_OEX_02', 0.69, 0.0672, 'Z'],
            ['LCL-Kondensator WIN_01', 'Terminal AC Low PEC_VSDc_WIN_01', 3.3, 0.21, 'Z'],
            ['LCL-Kondensator WIN_02', 'Terminal AC Low PEC_VSDc_WIN_02', 3.3, 0.21, 'Z'],
            ['LCL-Kondensator WIN_03', 'Terminal AC Low PEC_VSDc_WIN_03', 3.3, 0.21, 'Z'],
            ['LCL-Kondensator REC', 'Terminal AC Low PEC_VSDc_REC', 3.3, 0.21, 'Z'],
            ['LCL-Kondensator Drilling A', 'Terminal AC Low Drilling A', 0.69, 0.139, 'Z']
            #['test', 'Busbar WTG1 LV', 33, 3, 'Z']

        ],

        'generators': {
            'GEN': [
                ['name',                        'bus', 'S_n','V_n',     'P',          'V',    'H',  'D',    'X_d',  'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st',   'T_A',   'T_B',   'T_C'],
                ['Synchronous Generator 1', 'Main Bus A', 28, 11, 15.803972283333332, 1.02, 7.0074, 5, 2.33, 2.1, 0.173, 0.01, 0.159, 0.159, 0.822, 1, 0.03, 0.013],
                ['Synchronous Generator 2', 'Main Bus A', 28, 11, 15.803972283333332, 1.02, 7.0074, 5, 2.33, 2.1, 0.173, 0.01, 0.159, 0.159, 0.822, 1, 0.03, 0.013],
                ['Synchronous Generator 3', 'Main Bus A', 28, 11, 15.803972283333332, 1.02, 7.0074, 5, 2.33, 2.1, 0.173, 0.01, 0.159, 0.159, 0.822, 1, 0.03, 0.013]
            ]
        },

        'gov': {'TGOV1': [ ## GAST -> TGOV1
            ['name',    'gen',  'R',    'D_t',  'V_min',    'V_max',    'T_1',  'T_2',  'T_3'],
            ['GOV1',     'Synchronous Generator 1',   0.05,   0.02,   0, 1,    0.1,    0.09,   0.2],
            ['GOV2',     'Synchronous Generator 2',   0.05,   0.02,   0, 1,    0.1,    0.09,   0.2],
            ['GOV3',     'Synchronous Generator 3',   0.05,   0.02,   0, 1,    0.1,    0.09,   0.2]
            ]
        },

        'avr': {
            'SEXS': [
                ['name',   'gen',      'K',    'T_a',  'T_b',  'T_e',  'E_min',    'E_max'],
                ['Static Excitation System', 'Synchronous Generator 1',       100,    2,    10,   0.5,    0,         3],
                ['Static Excitation System', 'Synchronous Generator 2',       100,    2,    10,   0.5,    0,         3],
                ['Static Excitation System', 'Synchronous Generator 3',       100,    2,    10,   0.5,    0,         3]
            ]
        },
        'vsc': {
            'GridSideConverter_PV': [ 
                ['name',   'bus',                           'S_n',      "p_ref_grid",      "v_ref_grid",   'k_p',      'k_v',    'T_p',     'T_v',     'k_pll',   'T_pll',    'T_i',      "i_max"],
                ['WT1_LEOGO',    'Busbar WTG1 LV',            20,         0.0,               1.05,           5,          20,        0.1,        100,        5,        0.1,         0.01,      1.2]
            ]
        },
        
        'pss': {
            'STAB1': [
                ['name',    'gen',  'K',    'T',    'T_1',  'T_2',  'T_3',  'T_4',  'H_lim'],
                ['PSS1',    'Synchronous Generator 1',   50,     10.0,   0.5,    0.5,    0.05,   0.05,   0.03],
                ['PSS2',    'Synchronous Generator 2',   50,     10.0,   0.5,    0.5,    0.05,   0.05,   0.03],
                ['PSS3',    'Synchronous Generator 3',   50,     10.0,   0.5,    0.5,    0.05,   0.05,   0.03]
            ]
        }
    }


""" 
'avr': {
            'SEXS': [
                ['name',   'gen',      'K',    'T_a',  'T_b',  'T_e',  'E_min',    'E_max'],
                ['Static Excitation System', 'Synchronous Generator 1',       50,    5,    15,   0.1,    0,         3],
                ['Static Excitation System', 'Synchronous Generator 2',       50,    5,    15,   0.1,    0,         3],
                ['Static Excitation System', 'Synchronous Generator 3',       50,    5,    15,   0.1,    0,         3]
            ]
        },


'avr': {
            'SEXS': [
                ['name',   'gen',      'K',    'T_a',  'T_b',  'T_e',  'E_min',    'E_max'],
                ['Static Excitation System', 'Synchronous Generator 3',       100,    2.0,    10.0,   0.5,    -3,         3]
            ]
            'avr': {
            'SEXS': [ ## ext_ST
                ['name',   'gen',      'K',    'T_a',  'T_b',  'T_e',  'E_min',    'E_max'],
                ['Static Excitation System', 'Synchronous Generator 1', 50, 0.02, 10, 0.02, -1, 5], #T_b = 0?
                ['Static Excitation System', 'Synchronous Generator 2', 50, 0.02, 10, 0.02, -1, 5],
                ['Static Excitation System', 'Synchronous Generator 3', 50, 0.02, 10, 0.02, -1, 5]

                ['name',   'gen',      'K',    'T_a',  'T_b',  'T_e',  'E_min',    'E_max'],
                ['Static Excitation System', 'Synchronous Generator 1',       120,    1,    2,   0.02,    -1,         5],
                ['Static Excitation System', 'Synchronous Generator 2',       120,    1,    2,   0.02,    -1,         5],
                ['Static Excitation System', 'Synchronous Generator 3',       120,    1,    2,   0.02,    -1,         5]
                ['GOV1',     'G1',   0.05,   0,     0,          1,          0.2,    1,      2],
            ],
        }
} """
""" 'vsc': {
            'GridSideConverter_PV': [ 
                ['name',   'bus',    'S_n',      "p_ref_grid",      "v_ref_grid",   'k_p',      'k_v',    'T_p',     'T_v',     'k_pll',   'T_pll',    'T_i',      "i_max"],
                ['WT1_leogo',    'Busbar WTG1 LV',      20,         0.0,               1.05,           5,          10,        0.1,        100,        5,        0.1,         0.01,      1.2],
            ]
        }, 
        'vsc': {
            'GridSideConverter_PQ': [ 
                ['name',   'bus',    'S_n',      "p_ref_grid",      "q_ref_grid",   'k_p',      'k_q',    'T_p',     'T_q',     'k_pll',   'T_pll',    'T_i',      "i_max"],
                ['WT1_leogo',    'Busbar WTG1 LV',      20,         0.0,               0.0,           5,          1,        0.1,        0.1,        5,        0.1,         0.01,      1.2],
            ]
        },"""
""" ['Main Bus A', 11],
            ['Terminal AC High PEC_VSDc_GEX_01', 11],
            ['Terminal AC Low PEC_VSDc_GEX_01', 3.3], 
            ['Terminal AC High PEC_VSDc_OEX_01', 11],
            ['Terminal AC Low PEC_VSDc_OEX_01', 0.69],
            ['Terminal AC High PEC_VSDc_WIN_01', 11],
            ['Terminal AC Low PEC_VSDc_WIN_01', 3.3],
            ['Terminal AC High PEC_VSDc_WIN_02', 11],
            ['Terminal AC Low PEC_VSDc_WIN_02', 3.3],
            ['Terminal AC High PEC_VSDc_REC', 11],
            ['Terminal AC Low PEC_VSDc_REC', 3.3],
            ['Main Bus A', 11],
            ['Busbar WTG1 LV', 0.69],
            ['Busbar WTG2 LV', 0.69],
            ['Busbar WTG3 LV', 0.69],     
            ['Busbar WTG1 HV', 33], 
            ['Busbar WTG2 HV', 33],
            ['Busbar WTG3 HV', 33],
            ['Terminal WindPark', 33],
            ['Drilling AC Bus B', 11],
            ['Drilling AC Bus A', 11],
            ['Terminal AC High Drilling B', 11],
            ['Terminal AC Low Drilling B', 0.69],
            ['Terminal AC High Drilling A', 11],
            ['Terminal AC Low Drilling A', 0.69],
            ['SG Terminal 3', 11],
            ['SG Terminal 2', 11],
            ['SG Terminal 1', 11],
            ['Terminal ACO_02', 11],
            ['Terminal SWL_01', 11],
            ['Terminal ACO_01', 11],
            ['Terminal AC High PEC_VSDc_GEX_02', 11],
            ['Terminal AC Low PEC_VSDc_GEX_02', 3.3],
            ['Terminal AC High PEC_VSDc_GEX_03', 11],
            ['Terminal AC Low PEC_VSDc_GEX_03', 3.3],
            ['Terminal AC High PEC_VSDc_OEX_02', 11],
            ['Terminal AC Low PEC_VSDc_OEX_02', 0.69],
            ['Terminal AC High PEC_VSDc_WIN_03', 11],
            ['Terminal AC Low PEC_VSDc_WIN_03', 3.3],
            ['Terminal TRA_UTL690_02', 11],
            ['Terminal TRA_UTL690_01', 11],
            ['Utility690 Bus B',  0.69],
            ['Utility690 Bus A',  0.69],
            ['Terminal TRA_UTL400_02', 0.69],
            ['Terminal TRA_UTL400_01', 0.69],
            ['Utility400 SWBD/BusB', 0.4],
            ['Utility400 SWBD/BusA', 0.4] """