import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any
from torchdiffeq import odeint_adjoint as odeint
from torchmetrics import MeanSquaredError

from models.exp_hydro_function import *
from utils.training_utils import BaseLearner


# Fast Neural ode
# --------------------------------------------------------------
def rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()


def make_norm(state):
    state_size = state.numel()

    def norm(aug_state):
        y = aug_state[1:1 + state_size]
        adj_y = aug_state[1 + state_size:1 + 2 * state_size]
        return max(rms_norm(y), rms_norm(adj_y))

    return norm


# --------------------------------------------------------------


class M50_NN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=16):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_features=input_dim, out_features=hidden_units),
                                   nn.Tanh(),
                                   nn.Linear(in_features=hidden_units, out_features=hidden_units),
                                   nn.LeakyReLU(),
                                   nn.Linear(in_features=hidden_units, out_features=output_dim))

    def forward(self, x):
        x = self.model(x).squeeze()
        return x


class M50_ODEFunc(nn.Module):
    def __init__(self, ET_net, Q_net, f, Smax, Qmax, Df, Tmax, Tmin):
        super().__init__()
        self.ET_net = ET_net
        self.Q_net = Q_net
        self.f = f
        self.Smax = Smax
        self.Qmax = Qmax
        self.Df = Df
        self.Tmax = Tmax
        self.Tmin = Tmin

        # inner model series
        self.Precp_series = None
        self.Temp_series = None
        self.Lday_series = None

        # means and stds
        self.means = None
        self.stds = None

    def refresh_series(self, batch_ode_input):
        self.Precp_series = batch_ode_input[:, 0]
        self.Temp_series = batch_ode_input[:, 1]
        self.Lday_series = batch_ode_input[:, 2]

    def forward(self, t, S):
        S_snow, S_water = S[0], S[1]
        t = min(torch.floor(t).to(torch.int).item(), self.Precp_series.shape[0] - 1)
        t = max(0, t)
        Precp, Temp, Lday = self.Precp_series[t], self.Temp_series[t], self.Lday_series[t]
        # 归一化处理
        S_snow_norm = (S_snow - self.means[0]) / self.stds[0]
        S_water_norm = (S_water - self.means[1]) / self.stds[1]
        Precp_norm = (Precp - self.means[2]) / self.stds[2]
        Temp_norm = (Temp - self.means[3]) / self.stds[3]

        ET_output = self.ET_net(torch.tensor([S_snow_norm, S_water_norm, Temp_norm]).unsqueeze(0))
        Q_output = self.Q_net(torch.tensor([S_water_norm, Precp_norm]).unsqueeze(0))
        melt_output = M(S_snow, Temp, self.Df, self.Tmax)
        dS_1 = Ps(Precp, Temp, self.Tmin) - melt_output
        dS_2 = Pr(Precp, Temp, self.Tmin) + melt_output - step_fct(S_water) * Lday * torch.exp(ET_output) - step_fct(
            S_water) * torch.exp(Q_output)
        return dS_1, dS_2


class M50(BaseLearner):
    def __init__(self, model: M50_ODEFunc, loss_metric=MeanSquaredError(), eval_metric_list=None):
        super().__init__(model, loss_metric, eval_metric_list)
        self.model = model
        output = pd.read_csv(r'data/exp_hydro_output.csv')
        self.factor_std = self.model.stds = torch.from_numpy(output.std().values.astype(np.float32))
        self.factor_mean = self.model.means = torch.from_numpy(output.mean().values.astype(np.float32))

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch_x, batch_y = batch
        batch_ode_input = batch_x[:, 2:5] * self.factor_std[2:5] + self.factor_mean[2:5]
        S_snow0 = batch_x[0, 0] * self.factor_std[0] + self.factor_mean[0]
        S_water0 = batch_x[0, 1] * self.factor_std[1] + self.factor_mean[1]
        batch_t = torch.linspace(0, batch_x.shape[0] - 1, steps=batch_x.shape[0]).to(torch.device('cpu'))
        # 统一输入
        self.model.refresh_series(batch_ode_input)
        sol = odeint(self.model, y0=(S_snow0, S_water0), t=batch_t,
                     rtol=1e-3, atol=1e-3, adjoint_options=dict(norm='seminorm'))
        sol_1 = (sol[1].unsqueeze(1) - self.factor_mean[1]) / self.factor_std[1]
        y_out = torch.exp(self.model.Q_net(
            torch.concat([sol_1, batch_x[:, 2].unsqueeze(1)], dim=1))).unsqueeze(1)
        return batch_y, y_out
