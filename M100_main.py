import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.M100_models import M100, M100_NN, M100_ODEFunc
from models.my_dataset import TrainDataset, TorchDataModule
from utils.training_utils import BaseLearner, forecast, train

# project info
save_path = os.path.join(os.getcwd(), 'checkpoint')
proj_seed = 42
# static param
f, Smax, Qmax, Df, Tmax, Tmin = 0.017, 1709.46, 18.47, 2.67, 0.176, -2.09

# load data
pretrain_df = pd.read_csv(r'data/pretrain_dataset(M100).csv')

# model pretrain
pretrain_module = TorchDataModule(pretrain_df, time_idx=None,
                                  feature_cols=['S_snow', 'S_water', 'Precp', 'Temp'],
                                  target_cols=['ET_mech', 'Q_mech', 'M_mech', 'Ps_mech', 'Pr_mech'],
                                  feature_scaler=StandardScaler(), target_scaler=None,
                                  dataset=TrainDataset())

pretrain_model = M100_NN(input_dim=4, output_dim=5, hidden_units=32)
pretrain_learner = BaseLearner(pretrain_model)
pretrain_model, pretrain_learner = train(pretrain_model, pretrain_learner, pretrain_module,
                                         os.path.join(save_path, 'pretrain', 'M100', 'seed_{}'.format(proj_seed)))
pretrain_real_arr, pretrain_pred_arr = forecast(pretrain_learner, pretrain_module)

# M50 model train
exp_output_df = pd.read_csv(r'data/exp_hydro_output.csv')
means = exp_output_df[['precp', 'temp', 'Lday']].mean()
stds = exp_output_df[['precp', 'temp', 'Lday']].std()

exp_target_df = pd.read_csv(r'data/flow_target.csv')
train_df = pd.concat([exp_output_df, exp_target_df], axis=1)
m100_train_module = TorchDataModule(train_df, time_idx=None,
                                    feature_scaler=StandardScaler(), target_scaler=None, shuffle=False, batch_size=64,
                                    feature_cols=['S_snow', 'S_water', 'precp', 'temp', 'Lday'], target_cols=['flow'],
                                    dataset=TrainDataset())
m100_func = M100_ODEFunc(pretrain_model, f, Smax, Qmax, Df, Tmax, Tmin)
m100_leaner = M100(model=m100_func)
trained_m100_func, m100_leaner = train(m100_func, m100_leaner, m100_train_module,
                                       checkpoint_path=os.path.join(save_path, 'train', 'M100',
                                                                    'seed_{}'.format(proj_seed)))
trained_m50_learner = M100(model=trained_m100_func)
real_arr, pred_arr = forecast(trained_m50_learner, m100_train_module)
