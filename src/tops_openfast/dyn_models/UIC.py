from tops.dyn_models.blocks import DAEModel
import tops.utility_functions as dps_uf
import numpy as np


class UIC_sig(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        fix_idx = self.par['V_n'] == 0
        bus_idx = dps_uf.lookup_strings(self.par['bus'], self.sys_par['bus_names'])
        self.par['V_n'][fix_idx] = self.sys_par['bus_v_n'][bus_idx][fix_idx]
        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        
    def load_flow_pv(self):
        return self.bus_idx['terminal'], -self.par['p_ref']*self.par['S_n'], self.par['v_ref']

    def int_par_list(self):
        return ['f']

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}
    
    def state_list(self):
        """Return list of states for this VSC"""
        return ['vi_x', 'vi_y', 'x_filter']

    def input_list(self):
        """Input list for VSC"""
        return ['p_ref', 'q_ref', 'v_ref']
    
    def output_list(self):
        """Output list for VSC"""
        return ['S_n']

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par
        s_ref = self.p_ref(x, v) + 1j*self.q_ref(x, v)
        v_ref = self.v_ref(x, v)
        v_t = v[self.bus_idx_red['terminal']]
        vi = X['vi_x'] + 1j*X['vi_y']

        i_ref = np.conj(s_ref/vi) 
        i_max_pu = 1.9  # 20% overrating 
        i_ref_clamped = np.minimum(abs(i_ref), i_max_pu) * np.exp(1j*np.angle(i_ref))
        if np.any(np.abs(i_ref_clamped) >= i_max_pu):
            print('NB! i_ref is clamped by 1.9 pu limit')
        i_a = self.i_a(x, v)
        theta = np.angle(vi, deg=False)
        
        i_error = (i_ref_clamped - i_a)
        v_error = (v_ref - abs(vi))*np.exp(1j*(theta))

        dvi = 1j * 2 * np.pi * self.sys_par['f_n'] * par['Ki'] * i_error + 2 *np.pi *  self.sys_par['f_n'] * par['Kv'] * v_error + 1j * vi * X['x_filter'] * par['perfect_tracking']
        
        delta_omega = (dvi/vi).imag
        dX['x_filter'] = (1/par['T_filter'])*(delta_omega-X['x_filter'])
        perfect_tracking_addition = 1j * vi * X['x_filter'] * par['perfect_tracking']
        
        dX['vi_x'] = np.real(dvi)
        dX['vi_y'] = np.imag(dvi)
        
        # Debug
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter == 1 or self._debug_counter == 7500 or self._debug_counter == 100 or self._debug_counter == 500 or self._debug_counter == 1000 or (self._debug_counter % 5000 == 0 and self._debug_counter <= 60000): 
            """ print('Debug values (iteration', self._debug_counter, '):')
            print('  X[vi_x]:', X['vi_x'])
            print('  X[vi_y]:', X['vi_y'])
            print('vi: ', vi)
            print('theta: ', theta)
            print('v_t: ', v_t)
            print('  p_ref:', s_ref.real)
            print('  q_ref:', s_ref.imag)
            print('p_e: ', self.p_e(x, v))
            print('q_e: ', self.q_e(x, v))
            print('  v_ref:', v_ref)
            print('  i_ref:', i_ref)
            print('i_a: ', self.i_a(x, v))
            print('i_inj: ', self.i_inj(x, v))
            print('dvi: ', dvi)
            print('dX[vi_x]:', dX['vi_x'])
            print('dX[vi_y]:', dX['vi_y'])
            print('v_error: ', v_error)
            print('i_error: ', i_error)
            print('x_filter: ', X['x_filter'])
            print('delta_omega: ', delta_omega)
            print('dX[x_filter]:', dX['x_filter'])
            print('perfect_tracking_addition: ', perfect_tracking_addition) """
            #print('perfect_tracking_addition: ', perfect_tracking_addition)

        return

    def i_inj(self, x, v):
        X = self.local_view(x)
        vi = X['vi_x'] + 1j*X['vi_y']
        xf = self.par['xf'][0]
        current = vi/(1j*xf)
        return current
    
    def current_injections(self, x, v):
        i_n_r = self.par['S_n'][0] / self.sys_par['s_n'] 
        return self.bus_idx_red['terminal'], self.i_inj(x, v) * i_n_r
    
    def init_from_load_flow(self, x_0, v_0, S):
        par = self.par
        X = self.local_view(x_0)

        v_t = v_0[self.bus_idx_red['terminal']]
        S_local = S/ par['S_n'] # local pu
        current = np.conj(S_local/v_t)
        vi = v_t + 1j*current*par['xf']
        S_internal = vi * np.conj(current)

        self._input_values['v_ref'] = abs(vi)
        self._input_values['p_ref'] = S_internal.real 
        self._input_values['q_ref'] = S_internal.imag
        
        X['vi_x'] = np.real(vi)
        X['vi_y'] = np.imag(vi)
        X['x_filter'] = 0.0

        #print('vi_x init', X['vi_x'])
        #print('vi_y init', X['vi_y'])
        #print('p_ref init', S_internal.real)
        #print('q_ref init', S_internal.imag)
        #print('p_e init', self.p_e(x_0, v_0))
        #print('q_e init', self.q_e(x_0, v_0))
        #print('v_ref init', abs(vi))
        #print('v_t init', v_t)
        return

    def dyn_const_adm(self):
        par = self.par
        idx_bus = self.bus_idx['terminal']
        bus_v_n = self.sys_par['bus_v_n'][idx_bus]
        z_base_global = bus_v_n ** 2 / self.sys_par['s_n']
        z_base_local = par['V_n'] ** 2 / par['S_n']
        z_local = 1j*par['xf']
        z_global = z_local*z_base_local/z_base_global
        Y = 1/z_global
        return Y, (idx_bus,)*2

    def v_t(self, x, v):
        return v[self.bus_idx_red['terminal']]

    def v_q(self,x,v):
        return (self.v_t(x,v)*np.exp(-1j*self.local_view(x)['angle'])).imag
    
    def i_a(self, x, v):
        par = self.par
        X = self.local_view(x)
        v_t = v[self.bus_idx_red['terminal']]
        vi = X['vi_x'] + 1j*X['vi_y']
        i_a = -(v_t-vi)/(1j*par['xf'])
        return i_a

    def s_e(self, x, v):
        return self.v_t(x, v)*np.conj(self.i_a(x, v))

    def p_e(self, x, v):
        return self.s_e(x, v).real

    def q_e(self, x, v):
        return self.s_e(x, v).imag

    def S_n(self, x, v):
        return self.par['S_n']


class UIC_sig2(UIC_sig):
    """
    Alias of UIC_sig used to place a second, independent single-unit UIC
    converter on the same grid (e.g. WT2 at Busbar WTG2 LV).

    The base UIC_sig model has single-unit assumptions (scalar current-limit
    check, par['xf'][0], par['S_n'][0]). Registering the two converters under
    two different model keys ('UIC_sig' and 'UIC_sig2') keeps each converter as
    its own single-unit model object, so the existing math is reused unchanged.
    It adds no new behaviour.
    """
    pass