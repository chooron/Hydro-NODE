import torch.nn as nn
import torch
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

step_fct = lambda x: (torch.tanh(5.0 * x) + 1.0) * 0.5
Ps = lambda P, T, Tmin: step_fct(Tmin - T) * P
Pr = lambda P, T, Tmin: step_fct(T - Tmin) * P
M = lambda S0, T, Df, Tmax: step_fct(T - Tmax) * step_fct(S0) * torch.minimum(S0, Df * (T - Tmax))
PET = lambda T, Lday: 29.8 * Lday * 0.611 * torch.exp((17.3 * T) / (T + 237.3)) / (T + 273.2)
ET = lambda S1, T, Lday, Smax: step_fct(S1) * step_fct(S1 - Smax) * PET(T, Lday) + \
                               step_fct(S1) * step_fct(Smax - S1) * PET(T, Lday) * (S1 / Smax)
Qb = lambda S1, f, Smax, Qmax: step_fct(S1) * step_fct(S1 - Smax) * Qmax + step_fct(S1) * step_fct(
    Smax - S1) * Qmax * torch.exp(-f * (Smax - S1))
Qs = lambda S1, Smax: step_fct(S1) * step_fct(S1 - Smax) * (S1 - Smax)


class M50_NN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=16, means=None, stds=None):
        super().__init__()
        self.means = torch.tensor(means.astype(np.float32)).to(device)
        self.stds = torch.tensor(stds.astype(np.float32)).to(device)
        self.model = nn.Sequential(nn.Linear(in_features=input_dim, out_features=hidden_units),
                                   nn.Tanh(),
                                   nn.Linear(in_features=hidden_units, out_features=hidden_units),
                                   nn.LeakyReLU(),
                                   nn.Linear(in_features=hidden_units, out_features=output_dim)).to(device)

    def forward(self, x, t=None):
        x = (x - self.means) / self.stds
        x = self.model(x)
        return x


class M100_NN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_units=32, means=None, stds=None):
        super().__init__()
        self.means = torch.tensor(means.astype(np.float32)).to(device)
        self.stds = torch.tensor(stds.astype(np.float32)).to(device)
        self.model = nn.Sequential(nn.Linear(in_features=input_dim, out_features=hidden_units),
                                   nn.Tanh(),
                                   nn.Linear(in_features=hidden_units, out_features=hidden_units),
                                   nn.LeakyReLU(),
                                   nn.Linear(in_features=hidden_units, out_features=hidden_units),
                                   nn.LeakyReLU(),
                                   nn.Linear(in_features=hidden_units, out_features=hidden_units),
                                   nn.LeakyReLU(),
                                   nn.Linear(in_features=hidden_units, out_features=hidden_units),
                                   nn.LeakyReLU(),
                                   nn.Linear(in_features=hidden_units, out_features=output_dim)).to(device)

    def forward(self, x, t=None):
        x = (x - self.means) / self.stds
        x = self.model(x)
        return x
