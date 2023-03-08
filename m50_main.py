import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from models.NODE_models import M50
from scipy.interpolate import PchipInterpolator
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error

from models.customer_dataset import TrainDataset
from utils.training_utils import BaseLearner, NSELoss1D, forecast, train

# project info
basin_id = 1013500
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_path = os.path.join(os.getcwd(), 'checkpoint')
loss_metric = NSELoss1D().to(device)
# loss_metric = torch.nn.MSELoss()
solver_lib = 'torchdiffeq'

# static model param
f, Smax, Qmax, Df, Tmax, Tmin = 0.017, 1709.46, 18.47, 2.67, 0.176, -2.09  # 1013500
# f, Smax, Qmax, Df, Tmax, Tmin = 963.051, 217.796, 34.2572, 0.0558933, 0.304415, -1.00949  # 6431500

# load data
# train data
train_data_df = pd.read_csv(r'data/{}/train_data_df.csv'.format(basin_id), index_col=0)

# test data
test_data_df = pd.read_csv(r'data/{}/test_data_df.csv'.format(basin_id), index_col=0)

means = train_data_df[['S_snow', 'S_water', 'Precp', 'Temp', 'Lday']].mean().values
stds = train_data_df[['S_snow', 'S_water', 'Precp', 'Temp', 'Lday']].std().values

# interpolate the time series for solve ode
all_data_df = pd.concat([train_data_df, test_data_df], join='inner')
precp_series = all_data_df['Precp'].values
temp_series = all_data_df['Temp'].values
lday_series = all_data_df['Lday'].values

t_series = np.linspace(0, len(precp_series) - 1, len(precp_series))
precp_interp = PchipInterpolator(t_series, precp_series)
temp_interp = PchipInterpolator(t_series, temp_series)
lday_interp = PchipInterpolator(t_series, lday_series)

# M50 model train

# load pretrain model
et_save_path = os.path.join(save_path, str(basin_id), 'pretrain', 'M50-ET', 'model_state.pt')
q_save_path = os.path.join(save_path, str(basin_id), 'pretrain', 'M50-Q', 'model_state.pt')
et_pretrained_model = torch.load(et_save_path).to(device)
q_pretrained_model = torch.load(q_save_path).to(device)

# M50 train
# 1.prepare the train dataset
m50_train_dataloader = DataLoader(
    TrainDataset(
        train_data_df,
        input_cols=['S_snow', 'S_water', 'Precp', 'Temp', 'Lday'],
        target_cols=['Q_obs']),
    batch_size=len(train_data_df), shuffle=False)

m50_test_dataloader = DataLoader(
    TrainDataset(
        test_data_df,
        input_cols=['S_snow', 'S_water', 'Precp', 'Temp', 'Lday'],
        target_cols=['Q_obs']),
    batch_size=len(test_data_df), shuffle=False)

# 2.build the model
m50_model = M50(ET_net=et_pretrained_model, Q_net=q_pretrained_model, ode_lib=solver_lib,
                params=(f, Smax, Qmax, Df, Tmax, Tmin), interps=(precp_interp, temp_interp, lday_interp))

# 3.train the model based on pytorch-lightning
optimizer = torch.optim.Adam(m50_model.parameters(), lr=0.001)
m50_leaner = BaseLearner(m50_model, loss_metric=loss_metric, optimizer=optimizer)
m50_trained_model, m50_trained_learner = train(
    m50_leaner, m50_train_dataloader,
    os.path.join(save_path, str(basin_id), 'train', 'M50-NSE'), max_epochs=100)

# 4.test the trained model
train_real_arr, train_pred_arr = forecast(m50_trained_learner, m50_train_dataloader)
test_real_arr, test_pred_arr = forecast(m50_trained_learner, m50_test_dataloader)

print('train r2 ' + str(r2_score(train_real_arr, train_pred_arr)))
print('test r2 ' + str(r2_score(test_real_arr, test_pred_arr)))

print('train mse ' + str(mean_squared_error(train_real_arr, train_pred_arr)))
print('test mse ' + str(mean_squared_error(test_real_arr, test_pred_arr)))

# 5.plot the train and test result
plt.plot(train_real_arr, '--')
plt.plot(train_pred_arr, '--')
plt.show()

plt.plot(test_real_arr, '--')
plt.plot(test_pred_arr, '--')
plt.show()
