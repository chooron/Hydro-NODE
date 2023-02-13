import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy.integrate import ode

# load data
# determine data source path (download from https://ral.ucar.edu/solutions/products/camels)
data_path = r'E:\CAMELS\basin_dataset_public_v1p2'
# determine the basin id, area, huc
basin_id = "01013500"
area = 2260093113
basin_huc = "01"
# read the forcing data and streamflow data, show in the pandas DataFrame
path_forcing_data = os.path.join(data_path, "basin_mean_forcing", "daymet", basin_huc,
                                 "{}_lump_cida_forcing_leap.txt".format(basin_id))
path_flow_data = os.path.join(data_path, "usgs_streamflow", basin_huc, "{}_streamflow_qc.txt".format(basin_id))
header = ['Date', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']
forcing_df = pd.read_csv(path_forcing_data, skiprows=4, sep='\t', header=None)
raw_flow_df = pd.read_csv(path_flow_data, header=None)
forcing_df.columns = header
# set datetime of the data
forcing_df['Date'] = pd.to_datetime(forcing_df['Date'])
flow_series = []
for flow in raw_flow_df.values:
    flow_series.append(np.float32(flow[0].split(' ')[-2]))
flow_series = np.array(flow_series)
# handle data exceptions
flow_series[np.where(flow_series == -999)] = np.nan
# change data unit
flow_series = flow_series * (304.8 ** 3) / (area * 10 ** 6) * 86400
flow_df = pd.DataFrame(flow_series, columns=['flow'])
flow_df['Date'] = forcing_df['Date']
flow_df = flow_df[(flow_df['Date'] >= datetime(1980, 10, 1))
                  & (flow_df['Date'] <= datetime(2000, 9, 30))].drop('Date', axis=1)
# save the streamflow DataFrame
flow_df.to_csv(r'data/flow_target.csv', index=False)
forcing_df = forcing_df[(forcing_df['Date'] >= datetime(1980, 10, 1)) & (forcing_df['Date'] <= datetime(2000, 9, 30))]
# extract the precipitation, temperature, and day length
precp_series = forcing_df.loc[:, 'prcp(mm/day)'].values
temp_series = forcing_df.loc[:, ['tmax(C)', 'tmin(C)']].values.mean(axis=1)
Lday_series = forcing_df.loc[:, 'dayl(s)'].values / 3600

# ----------------------------------------------------------------------------------------------------------------------
# set the initial value of the model and the initial parameters
S0 = [0.0, 1303.0042478479704]
param = f, Smax, Qmax, Df, Tmax, Tmin = (0.017, 1709.46, 18.47, 2.67, 0.176, -2.09)
# Note: I don't know the meaning of the last param 0.8137
# define model function in numpy
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


def exp_hydro(t, S, args):
    """
    ordinary differential equations of the model
    :param t: time idx
    :param S: stores value
    :param args: model param (Dict)
    :return: dS1, dS2
    """
    f, Smax, Qmax, Df, Tmax, Tmin = args
    S1, S2 = S
    t = int(t)
    precp = precp_series[t]
    temp = temp_series[t]
    Lday = Lday_series[t]
    Q_out = Qb(S2, f, Smax, Qmax) + Qs(S2, Smax)
    dS1 = Ps(precp, temp, Tmin) - M(S1, temp, Df, Tmax)
    dS2 = Pr(precp, temp, Tmin) + M(S1, temp, Df, Tmax) - ET(S2, temp, Lday, Smax) - Q_out
    return [dS1, dS2]


# solve the ordinary differential equations by using scipy
ode_solver = ode(f=exp_hydro).set_integrator('dopri5', rtol=1e-6, atol=1e-6)
ode_solver.set_initial_value(y=S0, t=0.).set_f_params(param)
total_t = len(precp_series) - 1
dt = 1
S_series = [np.array(S0)]
while ode_solver.t < total_t:
    S_t = ode_solver.integrate(ode_solver.t + dt)
    S_series.append(S_t)
S_series = np.array(S_series)
# two store series in model (S_snow and S_water)
S_snow_series = S_series[:, 0]
S_water_series = S_series[:, 1]

# calculate the variable of qb and qs
qb = Qb(S_water_series, f, Smax, Qmax)
qs = Qs(S_water_series, Smax)

# Save the model prediction results
result_df = pd.DataFrame({'S_snow': S_snow_series, 'S_water': S_water_series,
                          'precp': precp_series, 'temp': temp_series, 'Lday': Lday_series})
result_df.to_csv(r'data/exp_hydro_output.csv', index=False)

# ----------------------------------------------------------------------------------------------------------------------
# prepare pre-trained data
S_snow_bucket = result_df['S_snow'].values
S_water_bucket = result_df['S_water'].values
P_bucket = result_df['precp'].values
T_bucket = result_df['temp'].values
Lday_bucket = result_df['Lday'].values

# evapotranspiration
ET_mech = ET(S_water_bucket, T_bucket, Lday_bucket, Smax)
ET_mech = np.array([x if x > 0.0 else 0.000000001 for x in ET_mech])
# melt
temp_T_mech = Df * (T_bucket - Tmax)
temp_T_mech = np.array([np.minimum(t, S0[1]) for t in temp_T_mech])
M_mech = temp_T_mech * step_fct(T_bucket - Tmax)
# discharge
Q_mech = Qb(S_water_bucket, f, Smax, Qmax) + Qs(S_water_bucket, Smax)
Q_mech = np.array([x if x > 0.0 else 0.000000001 for x in Q_mech])
# snow precipitation
Ps_mech = Ps(P_bucket, T_bucket, Tmin)
# rain precipitation
Pr_mech = Pr(P_bucket, T_bucket, Tmin)

# pretrain DataFrame for M50
ET_df = pd.DataFrame(
    {'S_snow': S_snow_bucket, 'S_water': S_water_bucket,
     'Temp': T_bucket, 'ET': np.log(ET_mech / Lday_bucket)}
).to_csv(r'data/pretrain_dataset(M50-ET).csv', index=False)
Q_df = pd.DataFrame({'S_water': S_water_bucket, 'Precp': P_bucket, 'Q': np.log(Q_mech)}).to_csv(
    r'data/pretrain_dataset(M50-Q).csv', index=False)

# pretrain DataFrame for M100
M100_df = pd.DataFrame({'S_snow': S_snow_bucket, 'S_water': S_water_bucket, 'Precp': P_bucket, 'Temp': T_bucket,
                        'ET_mech': np.log(ET_mech / Lday_bucket), 'Q_mech': np.log(Q_mech),
                        'M_mech': np.arcsinh(M_mech), 'Ps_mech': np.arcsinh(Ps_mech), 'Pr_mech': np.arcsinh(Pr_mech)})
M100_df.to_csv(r'data/pretrain_dataset(M100).csv', index=False)
