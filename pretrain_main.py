import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from models.common_net import M100_NN, M50_NN
from torch.utils.data import Dataset, DataLoader
from utils.training_utils import BaseLearner, forecast, train
from models.customer_dataset import PretrainDataset

# project info
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_path = os.path.join(os.getcwd(), 'checkpoint')
basin_id = 1013500

# load data
# train data
train_data_df = pd.read_csv(r'data/{}/train_data_df.csv'.format(basin_id))
# test data
test_data_df = pd.read_csv(r'data/{}/test_data_df.csv'.format(basin_id))

means = train_data_df[['S_snow', 'S_water', 'Precp', 'Temp', 'Lday']].mean().values
stds = train_data_df[['S_snow', 'S_water', 'Precp', 'Temp', 'Lday']].std().values




ET_train_dataloader = DataLoader(dataset=PretrainDataset(
    train_data_df, input_cols=['S_snow', 'S_water', 'Temp'], target_cols=['ET_mech']),
    batch_size=32, shuffle=True)

ET_test_dataloader = DataLoader(dataset=PretrainDataset(
    test_data_df, input_cols=['S_snow', 'S_water', 'Temp'], target_cols=['ET_mech']),
    batch_size=32, shuffle=False)

Q_train_dataloader = DataLoader(dataset=PretrainDataset(
    train_data_df, input_cols=['S_water', 'Precp'], target_cols=['Q_mech']),
    batch_size=32, shuffle=False)

Q_test_dataloader = DataLoader(dataset=PretrainDataset(
    test_data_df, input_cols=['S_water', 'Precp'], target_cols=['Q_mech']),
    batch_size=32, shuffle=False)

net_train_dataloader = DataLoader(dataset=PretrainDataset(
    train_data_df, input_cols=['S_snow', 'S_water', 'Precp', 'Temp'],
    target_cols=['ET_mech', 'Q_mech', 'M_mech', 'Ps_mech', 'Pr_mech']),
    batch_size=32, shuffle=False)

net_test_dataloader = DataLoader(dataset=PretrainDataset(
    test_data_df, input_cols=['S_snow', 'S_water', 'Precp', 'Temp'],
    target_cols=['ET_mech', 'Q_mech', 'M_mech', 'Ps_mech', 'Pr_mech']),
    batch_size=32, shuffle=False)

ET_model = M50_NN(input_dim=3, output_dim=1, hidden_units=16, means=means[[0, 1, 3]], stds=stds[[0, 1, 3]]).to(device)
ET_learner = BaseLearner(ET_model, loss_metric=torch.nn.MSELoss())
ET_pretrain_save_path = os.path.join(save_path, str(basin_id), 'pretrain', 'M50-ET')
ET_trained_model, trained_ET_learner = train(ET_learner, ET_train_dataloader, ET_pretrain_save_path)

et_real_arr, et_pred_arr = forecast(trained_ET_learner, ET_test_dataloader)
print('test acc ' + str(r2_score(et_real_arr, et_pred_arr)))

Q_model = M50_NN(input_dim=2, output_dim=1, hidden_units=16, means=means[[1, 2]], stds=stds[[1, 2]]).to(device)
Q_learner = BaseLearner(Q_model, loss_metric=torch.nn.MSELoss())
Q_pretrain_save_path = os.path.join(save_path, str(basin_id), 'pretrain', 'M50-Q')
Q_trained_model, trained_Q_learner = train(Q_learner, Q_train_dataloader, Q_pretrain_save_path)

net_model = M100_NN(input_dim=4, output_dim=5, hidden_units=32,
                    means=means[[0, 1, 2, 3]], stds=stds[[0, 1, 2, 3]]).to(device)
net_learner = BaseLearner(net_model, loss_metric=torch.nn.MSELoss(), lr=1e-3)
net_pretrain_save_path = os.path.join(save_path, str(basin_id), 'pretrain', 'M100')
net_trained_model, trained_net_learner = train(net_learner, net_train_dataloader, net_pretrain_save_path)

# q_real_arr, q_pred_arr = forecast(trained_Q_learner, Q_test_dataloader)
# q_real_arr, q_pred_arr = forecast(trained_Q_learner, Q_train_dataloader)
# Q_obs = train_data_df['Q_obs'].values
# q_pred = np.exp(q_pred_arr)
# print('test acc ' + str(r2_score(q_real_arr, q_pred_arr)))
# print('test acc ' + str(r2_score(Q_obs, q_pred)))
