# set the initial value of the model and the initial parameters
import numpy as np
import torch

step_fct = lambda x: (np.tanh(5.0 * x) + 1.0) * 0.5
Ps = lambda P, T, Tmin: step_fct(Tmin - T) * P
Pr = lambda P, T, Tmin: step_fct(T - Tmin) * P
M = lambda S0, T, Df, Tmax: step_fct(T - Tmax) * step_fct(S0) * np.minimum(S0, Df * (T - Tmax))
PET = lambda T, Lday: 29.8 * Lday * 0.611 * np.exp((17.3 * T) / (T + 237.3)) / (T + 273.2)
ET = lambda S1, T, Lday, Smax: step_fct(S1) * step_fct(S1 - Smax) * PET(T, Lday) + \
                               step_fct(S1) * step_fct(Smax - S1) * PET(T, Lday) * (S1 / Smax)
Qb = lambda S1, f, Smax, Qmax: step_fct(S1) * step_fct(S1 - Smax) * Qmax + step_fct(S1) * step_fct(
    Smax - S1) * Qmax * np.exp(-f * (Smax - S1))
Qs = lambda S1, Smax: step_fct(S1) * step_fct(S1 - Smax) * (S1 - Smax)


class M0:

    def __init__(self, precp_interp, temp_interp, lday_interp):
        self.precp_interp = precp_interp
        self.temp_interp = temp_interp
        self.lday_interp = lday_interp

    @staticmethod
    def exp_hydro_single_step(t, S, args):
        """
        ordinary differential equations of the model for step time point solving
        :param t: time idx
        :param S: stores value
        :param args: model param (Dict)
        :return: dS1, dS2
        """
        f, Smax, Qmax, Df, Tmax, Tmin, precp, temp, Lday = args
        S1, S2 = S
        Q_out = Qb(S2, f, Smax, Qmax) + Qs(S2, Smax)
        dS1 = Ps(precp, temp, Tmin) - M(S1, temp, Df, Tmax)
        dS2 = Pr(precp, temp, Tmin) + M(S1, temp, Df, Tmax) - ET(S2, temp, Lday, Smax) - Q_out
        return [dS1, dS2]

    @staticmethod
    def exp_hydro_total_series(S, t, f, Smax, Qmax, Df, Tmax, Tmin, precp_interp, temp_interp, lday_interp):
        """
        ordinary differential equations of the model for total series solving with interpolating
        :param t: time idx
        :param S: stores value
        :return: dS1, dS2
        """
        S1, S2 = S
        # precp = precp_interp.evaluate(torch.tensor(t)).item()
        # temp = temp_interp.evaluate(torch.tensor(t)).item()
        # lday = lday_interp.evaluate(torch.tensor(t)).item()
        precp = precp_interp(t)
        temp = temp_interp(t)
        lday = lday_interp(t)
        Q_out = Qb(S2, f, Smax, Qmax) + Qs(S2, Smax)
        dS1 = Ps(precp, temp, Tmin) - M(S1, temp, Df, Tmax)
        dS2 = Pr(precp, temp, Tmin) + M(S1, temp, Df, Tmax) - ET(S2, temp, lday, Smax) - Q_out
        return [dS1, dS2]

    def run(self, S0, param, precp_series, temp_series, Lday_series):
        # step time points solve ode
        from scipy.integrate import ode

        ode_solver = ode(f=self.exp_hydro_single_step).set_integrator('dopri5', rtol=1e-6, atol=1e-6)
        f, Smax, Qmax, Df, Tmax, Tmin = param
        ode_solver.set_initial_value(y=S0, t=0.)
        total_t = len(precp_series) - 1
        dt = 1
        S_series = [np.array(S0)]
        while ode_solver.t < total_t:
            tmp_idx = int(ode_solver.t)
            tmp_prec, tmp_temp, tmp_Lday = precp_series[tmp_idx], temp_series[tmp_idx], Lday_series[tmp_idx]
            ode_solver.set_f_params(param + (tmp_prec, tmp_temp, tmp_Lday))
            S_t = ode_solver.integrate(ode_solver.t + dt)
            S_series.append(S_t)
        S_series = np.array(S_series)
        # two store series in model (S_snow and S_water)
        S_snow_series = S_series[:, 0]
        S_water_series = S_series[:, 1]
        Qb_series = Qb(S_water_series, f, Smax, Qmax)
        Qs_series = Qs(S_water_series, Smax)
        return S_snow_series, S_water_series, Qb_series, Qs_series

    def run_v2(self, S0, param, time_series_idx):
        # total series solving with interpolating
        from scipy.integrate import odeint
        f, Smax, Qmax, Df, Tmax, Tmin = param
        sol = odeint(self.exp_hydro_total_series, y0=S0, t=time_series_idx,
                     args=(param + (self.precp_interp, self.temp_interp, self.lday_interp)))
        S_snow_series = sol[:, 0]
        S_water_series = sol[:, 1]
        Qb_series = Qb(S_water_series, f, Smax, Qmax)
        Qs_series = Qs(S_water_series, Smax)
        return S_snow_series, S_water_series, Qb_series, Qs_series

    def get_flux(self, S0, param, S_snow_series, S_water_series, precp_series, temp_series, Lday_series):
        f, Smax, Qmax, Df, Tmax, Tmin = param
        # evapotranspiration
        ET_mech = ET(S_water_series, temp_series, Lday_series, Smax)
        ET_mech = np.array([x if x > 0.0 else 0.000000001 for x in ET_mech])
        # melt
        temp_T_mech = Df * (temp_series - Tmax)
        temp_T_mech = np.array([np.minimum(t, S0[1]) for t in temp_T_mech])
        M_mech = temp_T_mech * step_fct(temp_series - Tmax)
        # discharge
        Q_mech = Qb(S_water_series, f, Smax, Qmax) + Qs(S_water_series, Smax)
        Q_mech = np.array([x if x > 0.0 else 0.000000001 for x in Q_mech])
        # snow precipitation
        Ps_mech = Ps(precp_series, temp_series, Tmin)
        # rain precipitation
        Pr_mech = Pr(precp_series, temp_series, Tmin)
        return ET_mech, M_mech, Q_mech, Ps_mech, Pr_mech
