from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def nse_origin(real, pred, obs_mean):
    denominator = np.sum(np.power(real - obs_mean, 2))
    numerator = np.sum(np.power(pred - real, 2))
    return numerator / denominator


def nse_beta(real, pred, obs_mean):
    denominator = np.sum(np.abs(real - obs_mean))
    numerator = np.sum(np.abs(pred - real))
    return numerator / denominator


df = pd.read_csv(r'F:\pycharm\My Project\Some Implemets\Hydro-NODE\data\1013500\train_data_df.csv')
Q_mech = np.exp(df['Q_mech'].values)
df2 = pd.read_csv(r'F:\pycharm\My Project\Some Implemets\Hydro-NODE\data\M0_output_julia.csv')[:-1]

Q_obs = df['Q_obs'].values
Q_pred = df2['Q'].values
i = 0
batch_idx = 1
nse_list = []
total_mean = np.mean(Q_obs)
time_step = 300
# temp_nse = r2_score(Q_obs[i:end_idx], Q_pred[i:end_idx])
# print('batch {} nse: ' + str(temp_nse) + ", weight: " + str(temp_weight) + str(
#     ' weight nse:' + str(temp_nse * temp_weight)))
while i < len(Q_obs):
    if i + time_step < len(Q_obs):
        end_idx = i + time_step
    else:
        break
    temp_weight = np.mean(Q_obs[i:end_idx]) ** 2 / total_mean ** 2
    # temp_nse = nse_origin(Q_obs[i:end_idx], Q_pred[i:end_idx], np.mean(Q_obs[i:end_idx]))
    temp_nse = mean_absolute_error(Q_obs[i:end_idx], Q_pred[i:end_idx])
    print('batch {} rse: '.format(batch_idx) + str(temp_nse))
    i = i + time_step
    batch_idx = batch_idx + 1
    nse_list.append(temp_nse)
print('batch mean rse: ' + str(np.mean(nse_list)))
print('batch std rse: ' + str(np.std(nse_list) / np.mean(nse_list)))
# temp_df = pd.DataFrame({'Q_mech': Q_mech, 'Q_obs': Q_obs, 'Q_pred': Q_pred})

