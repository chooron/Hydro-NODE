import numpy as np
import spotpy
from spotpy import algorithms, parameter, analyser
from sklearn.metrics import mean_squared_error as mse


def nse(real, pred):
    return (np.sum((pred - real) ** 2) / np.sum((real - np.mean(real)) ** 2))


class model_setup(object):
    def __init__(self, model, data, params_names, params_low_bounds, params_up_bounds):
        self.model = model
        self.precp_series, self.temp_series, self.Lday_series, self.Q_obs_series = data
        self.params_names = params_names
        self.params_low_bounds = params_low_bounds
        self.params_up_bounds = params_up_bounds
        self.params = self.parse_search_params()

    def parameters(self):
        return parameter.generate(self.params)

    def simulation(self, x):
        S0 = (x[0], x[1])
        param = (x[2], x[3], x[4], x[5], x[6], x[7])
        S_snow_series, S_water_series, Qb_series, Qs_series = self.model.run(
            S0, param, self.precp_series, self.temp_series, self.Lday_series)
        ET_mech, M_mech, Q_mech, Ps_mech, Pr_mech = self.model.get_flux(
            S0, param, S_snow_series, S_water_series, self.precp_series, self.temp_series, self.Lday_series)
        return Q_mech

    def evaluation(self):
        return self.Q_obs_series

    def objectivefunction(self, simulation, evaluation):
        return mse(simulation, evaluation)

    def parse_search_params(self):
        temp_params = []
        for param_name, param_low, param_up in zip(self.params_names, self.params_low_bounds, self.params_up_bounds):
            temp_param = parameter.Uniform(name=param_name, low=param_low, high=param_up)
            temp_params.append(temp_param)
        return temp_params


def optimization(model, data, params_names, params_low_bounds, params_up_bounds, epochs=100):
    sampler = algorithms.sceua(model_setup(model, data, params_names, params_low_bounds, params_up_bounds),
                               dbname='none', dbformat='ram', random_state=42)
    sampler.sample(epochs)  # Sample 100.000 parameter combinations
    results = sampler.getdata()
    best_params = analyser.get_best_parameterset(results, maximize=False)[0]
    return best_params
