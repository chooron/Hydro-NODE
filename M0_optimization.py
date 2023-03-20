import os.path

import numpy as np
import pandas as pd

from models.M0_models import M0
from models.spotpy_optimization import optimization
from utils.data_utils import load_data, prepare_data

# load data
basin_id = 6431500
forcing_df, flow_df = load_data(basin_id)
train_data_df, test_data_df, train_flow_df, test_flow_df, precp_interp, temp_interp, lday_interp = \
    prepare_data(forcing_df, flow_df)
data = train_data_df['Precp'].values, train_data_df['Temp'].values, train_data_df['Lday'].values, train_flow_df.values

model = M0(precp_interp, temp_interp, lday_interp)
params_names = ['S1', 'S2', 'f', 'Smax', 'Qmax', 'Df', 'Tmax', 'Tmin']
params_low_bounds = [0.01, 100.0, 0.0, 100.0, 10.0, 0.01, 0.0, -3.0]
params_up_bounds = [1500.0, 1500.0, 0.1, 1500.0, 50.0, 5.0, 3.0, 0.0]
best_params = optimization(model, data, params_names, params_low_bounds, params_up_bounds, epochs=100)
best_params = list(best_params)
best_params_df = pd.DataFrame({nm: vl for nm, vl in zip(params_names, best_params)}, index=[0])
save_path = r'checkpoint/{}'.format(basin_id)
if not os.path.exists(save_path):
    os.makedirs(save_path)
best_params_df.to_csv(r'checkpoint/{}/best_params_df.csv'.format(basin_id), index=False)
