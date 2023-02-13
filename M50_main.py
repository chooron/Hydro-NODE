import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.M50_models import M50, M50_NN, M50_ODEFunc
from models.my_dataset import TrainDataset, TorchDataModule
from utils.training_utils import BaseLearner, forecast, train

# project info
save_path = os.path.join(os.getcwd(), 'checkpoint')
proj_seed = 42
# static param
f, Smax, Qmax, Df, Tmax, Tmin = 0.017, 1709.46, 18.47, 2.67, 0.176, -2.09

# load data
ET_df = pd.read_csv(r'data/pretrain_dataset(ET).csv')
Q_df = pd.read_csv(r'data/pretrain_dataset(Q).csv')

# ET model pretrain
ET_train_module = TorchDataModule(ET_df, time_idx=None,
                                  feature_cols=['S_snow', 'S_water', 'Temp'], target_cols=['ET'],
                                  feature_scaler=StandardScaler(), target_scaler=None,
                                  dataset=TrainDataset())

ET_model = M50_NN(input_dim=3, output_dim=1, hidden_units=16)
ET_learner = BaseLearner(ET_model)
ET_trained_model, ET_learner = train(ET_model, ET_learner, ET_train_module,
                                     os.path.join(save_path, 'pretrain', 'M50-ET', 'seed_{}'.format(proj_seed)))
et_real_arr, et_pred_arr = forecast(ET_learner, ET_train_module)

# Q model pretrain
Q_train_module = TorchDataModule(Q_df, time_idx=None,
                                 feature_cols=['S_water', 'Precp'], target_cols=['Q'],
                                 feature_scaler=StandardScaler(), target_scaler=None,
                                 dataset=TrainDataset())
Q_model = M50_NN(input_dim=2, output_dim=1, hidden_units=16)
Q_learner = BaseLearner(Q_model)
Q_trained_model, Q_learner = train(Q_model, Q_learner, Q_train_module,
                                   os.path.join(save_path, 'pretrain', 'M50-Q', 'seed_{}'.format(proj_seed)))
q_real_arr, q_pred_arr = forecast(Q_learner, Q_train_module)

# M50 model train
exp_output_df = pd.read_csv(r'data/exp_hydro_output.csv')
means = exp_output_df[['precp', 'temp', 'Lday']].mean()
stds = exp_output_df[['precp', 'temp', 'Lday']].std()

exp_target_df = pd.read_csv(r'data/flow_target.csv')
train_df = pd.concat([exp_output_df, exp_target_df], axis=1)
m50_train_module = TorchDataModule(train_df, time_idx=None,
                                   feature_scaler=StandardScaler(), target_scaler=None, shuffle=False, batch_size=64,
                                   feature_cols=['S_snow', 'S_water', 'precp', 'temp', 'Lday'], target_cols=['flow'],
                                   dataset=TrainDataset())
m50_func = M50_ODEFunc(ET_model, Q_model, f, Smax, Qmax, Df, Tmax, Tmin)
m50_leaner = M50(model=m50_func)
trained_m50_func, m50_leaner = train(m50_func, m50_leaner, m50_train_module,
                                     checkpoint_path=os.path.join(save_path, 'train', 'M50',
                                                                  'seed_{}'.format(proj_seed)))
trained_m50_learner = M50(model=trained_m50_func)
real_arr, pred_arr = forecast(trained_m50_learner, m50_train_module)
