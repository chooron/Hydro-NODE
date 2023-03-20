import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from models.NODE_models_Batch import M100_Func, ODESolver, M50_Func
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import r2_score, mean_squared_error

from models.common_net import M100_NN, M50_NN
from models.customer_dataset import BatchTrainDataset, TrainDataset
from utils.loss_utils import AdaptiveNSE, NSELoss, QuantileLoss
from utils.training_utils import BaseLearner, forecast, train

# project info
basin_id = 1013500
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_path = os.path.join(r'F:\pycharm\My Project\Some Implemets\Hydro-NODE', 'checkpoint')
loss_metric = torch.nn.MSELoss()
solver_lib = 'torchdiffeq'
use_pretrain = True
all_input = True
params_df = pd.read_csv(r'../checkpoint/bucket_opt_init.csv')
best_params = params_df[params_df['basin id'] == basin_id].values.squeeze()[1:-1]
# static model param
S0, S1, f, Smax, Qmax, Df, Tmax, Tmin = tuple(best_params.tolist())  # 1013500
# f, Smax, Qmax, Df, Tmax, Tmin = 963.051, 217.796, 34.2572, 0.0558933, 0.304415, -1.00949  # 6431500

# load data
# train data
train_data_df = pd.read_csv(r'../data/{}/train_data_df.csv'.format(basin_id), index_col=0)
# test data
test_data_df = pd.read_csv(r'../data/{}/test_data_df.csv'.format(basin_id), index_col=0)

means = train_data_df[['S_snow', 'S_water', 'Precp', 'Temp', 'Lday']].mean().values
stds = train_data_df[['S_snow', 'S_water', 'Precp', 'Temp', 'Lday']].std().values

# interpolate the time series for solve ode
all_data_df = pd.concat([train_data_df, test_data_df], join='inner')
precp_series = all_data_df['Precp'].values
temp_series = all_data_df['Temp'].values
lday_series = all_data_df['Lday'].values

# get loss metric
# 30: 0.78, 0.82

# torchcubicspline 插值库
from torchcubicspline import (natural_cubic_spline_coeffs, NaturalCubicSpline)

t_series = torch.linspace(0, len(precp_series) - 1, len(precp_series))
precp_spline = NaturalCubicSpline(natural_cubic_spline_coeffs(t_series, torch.from_numpy(precp_series).unsqueeze(1)))
temp_spline = NaturalCubicSpline(natural_cubic_spline_coeffs(t_series, torch.from_numpy(temp_series).unsqueeze(1)))
lday_spline = NaturalCubicSpline(natural_cubic_spline_coeffs(t_series, torch.from_numpy(lday_series).unsqueeze(1)))

# M50 model train

# load pretrain model
if use_pretrain:
    net_save_path = os.path.join(save_path, str(basin_id), 'pretrain', 'M100', 'model_state.pt')
    net_pretrained_model = torch.load(net_save_path).to(device)
else:
    net_pretrained_model = M100_NN(input_dim=4, output_dim=5, hidden_units=32,
                                   means=means[[0, 1, 2, 3]], stds=stds[[0, 1, 2, 3]]).to(device)

# M50 train
# 1.prepare the train dataset
if all_input:
    train_val_dataset = BatchTrainDataset(
        train_data_df,
        input_cols=['S_snow', 'S_water', 'Precp', 'Temp', 'Lday', 'Q_mech'],
        target_cols=['Q_obs'],
        time_len=len(train_data_df),
    )
    m50_train_val_dataloader = m50_train_dataloader = DataLoader(train_val_dataset, batch_size=1, shuffle=False)
    m50_val_dataloader = None
else:
    train_val_dataset = BatchTrainDataset(
        train_data_df,
        input_cols=['S_snow', 'S_water', 'Precp', 'Temp', 'Lday', 'Q_mech'],
        target_cols=['Q_obs'],
        time_len=365,
    )
    train_dataset, val_dataset = random_split(
        dataset=train_val_dataset,
        lengths=[int(len(train_val_dataset) * 0.8),
                 len(train_val_dataset) - int(len(train_val_dataset) * 0.8)],
        generator=torch.Generator().manual_seed(42))

    m50_train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    m50_val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    m50_train_val_dataloader = DataLoader(train_val_dataset, batch_size=1, shuffle=False)

m50_test_dataloader = DataLoader(
    TrainDataset(
        test_data_df,
        input_cols=['S_snow', 'S_water', 'Precp', 'Temp', 'Lday', 'Q_mech'],
        target_cols=['Q_obs']),
    batch_size=len(test_data_df), shuffle=False)

# 2.build the model
m50_func = M100_Func(net=net_pretrained_model, ode_lib=solver_lib,
                     params=(f, Smax, Qmax, Df, Tmax, Tmin),
                     interps=(precp_spline, temp_spline, lday_spline))
# 3.train the model based on pytorch-lightning
optimizer = torch.optim.SGD(m50_func.parameters(), lr=0.001, momentum=0.5)
learner_kwarg = {'solve_func': m50_func, 'loss_metric': loss_metric, 'optimizer': optimizer}
m50_leaner = ODESolver(m50_func, loss_metric=loss_metric, optimizer=optimizer, ode_lib=solver_lib)
m50_trained_model, m50_trained_learner = train(
    m50_leaner, m50_train_dataloader,
    os.path.join(save_path, str(basin_id), 'train', 'M100-Total', 'MSE'),
    val_dataloaders=m50_val_dataloader, max_epochs=100, **learner_kwarg)
# 4.test the trained model
train_real_arr, train_pred_arr = forecast(m50_trained_learner, m50_train_val_dataloader)
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
