import numpy as np
import os
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from models.M0_models import M0
from utils.data_utils import load_data, prepare_data

# load data
basin_id = 6431500
forcing_df, flow_df = load_data(basin_id)

train_data_df, test_data_df, train_flow_df, test_flow_df, precp_interp, temp_interp, lday_interp = \
    prepare_data(forcing_df, flow_df)

train_precp_series, test_precp_series = train_data_df['Precp'].values, test_data_df['Precp'].values
train_temp_series, test_temp_series = train_data_df['Temp'].values, test_data_df['Temp'].values
train_Lday_series, test_Lday_series = train_data_df['Lday'].values, test_data_df['Lday'].values

params_df = pd.read_csv(r'checkpoint/bucket_opt_init.csv')
best_params = params_df[params_df['basin id'] == basin_id].values.squeeze()[1:-1]
S0 = best_params[:2].tolist()
param = f, Smax, Qmax, Df, Tmax, Tmin = tuple(best_params[2:].tolist())
# S0 = [963.0510489149701,217.79595099306835]
# param = f, Smax, Qmax, Df, Tmax, Tmin =0.016162871409461377,1357.9758639379108,34.2571623863836,0.05589328756698195,0.3044146285980963,-1.0094946926758326
model = M0(precp_interp, temp_interp, lday_interp)
# train result
train_time_series = np.linspace(0, len(train_data_df) - 1, len(train_data_df))
# S_snow_series, S_water_series, Qb_series, Qs_series = model.run2(S0, param, train_time_series)
S_snow_series, S_water_series, Qb_series, Qs_series = model.run_v2(S0, param, train_time_series)

temp_S0 = [S_snow_series[-1], S_water_series[-1]]
train_ET_mech, train_M_mech, train_Q_mech, train_Ps_mech, train_Pr_mech = model.get_flux(
    S0, param, S_snow_series, S_water_series,
    train_precp_series, train_temp_series, train_Lday_series)

# test result
test_time_series = np.linspace(len(train_data_df), len(train_data_df) + len(test_data_df) - 1, len(test_data_df))
# test_S_snow_series, test_S_water_series, test_Qb_series, test_Qs_series = model.run(temp_S0, param, test_precp_series,
#                                                                                    test_temp_series, test_Lday_series)
test_S_snow_series, test_S_water_series, test_Qb_series, test_Qs_series = model.run_v2(temp_S0, param, test_time_series)
test_ET_mech, test_M_mech, test_Q_mech, test_Ps_mech, test_Pr_mech = model.get_flux(
    temp_S0, param, test_S_snow_series, test_S_water_series,
    test_precp_series, test_temp_series, test_Lday_series)

# save file
train_data_df = pd.DataFrame({'S_snow': S_snow_series, 'S_water': S_water_series,
                              'Precp': train_precp_series, 'Temp': train_temp_series,
                              'Lday': train_Lday_series, 'ET_mech': np.log(train_ET_mech / train_Lday_series),
                              'Q_mech': np.log(train_Q_mech), 'M_mech': np.arcsinh(train_M_mech),
                              'Ps_mech': np.arcsinh(train_Ps_mech), 'Pr_mech': np.arcsinh(train_Pr_mech),
                              'Q_obs': train_flow_df.values.squeeze()})

test_data_df = pd.DataFrame({'S_snow': test_S_snow_series, 'S_water': test_S_water_series,
                             'Precp': test_precp_series, 'Temp': test_temp_series,
                             'Lday': test_Lday_series, 'ET_mech': np.log(test_ET_mech / test_Lday_series),
                             'Q_mech': np.log(test_Q_mech), 'M_mech': np.arcsinh(test_M_mech),
                             'Ps_mech': np.arcsinh(test_Ps_mech), 'Pr_mech': np.arcsinh(test_Pr_mech),
                             'Q_obs': test_flow_df.values.squeeze()})
test_data_df.index = test_data_df.index + len(train_data_df)
if not os.path.exists(r'data/{}/'.format(basin_id)):
    os.makedirs(r'data/{}/'.format(basin_id))
train_data_df.to_csv(r'data/{}/train_data_df.csv'.format(basin_id), index=True)
test_data_df.to_csv(r'data/{}/test_data_df.csv'.format(basin_id), index=True)

print('train r2:' + str(r2_score(train_flow_df.values.squeeze(), train_Q_mech)))
print('test r2:' + str(r2_score(test_flow_df.values.squeeze(), test_Q_mech)))

print('train mse:' + str(mean_squared_error(train_flow_df.values.squeeze(), train_Q_mech)))
print('test mse:' + str(mean_squared_error(test_flow_df.values.squeeze(), test_Q_mech)))
