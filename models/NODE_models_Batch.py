import numpy as np
import torch
import torch.nn as nn
import torchcde
from torchdiffeq import odeint
from models.common_net import Ps, Pr, M, step_fct

from utils.training_utils import BaseLearner

# --------------------------------------------------------------
# NODE version 4.0 meta 是面向于batch输入的模型
# --------------------------------------------------------------

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class M50_Func(nn.Module):
    def __init__(self, ET_net, Q_net, params, interps, ode_lib='torchdiffeq'):
        super().__init__()
        self.S0, self.S1, self.f, self.Smax, self.Qmax, self.Df, self.Tmax, self.Tmin = params
        self.ET_net = ET_net.to(device)
        self.ode_lib = ode_lib
        self.Q_net = Q_net.to(device)
        self.precp_interp, self.temp_interp, self.lday_interp = interps

    def forward(self, t, S):
        S_snow, S_water = S

        precp = self.precp_interp.evaluate(t).to(torch.float32).to(device)
        temp = self.temp_interp.evaluate(t).to(torch.float32).to(device)
        lday = self.lday_interp.evaluate(t).to(torch.float32).to(device)

        ET_output = self.ET_net(torch.tensor([S_snow, S_water, temp]).to(device))
        Q_output = self.Q_net(torch.tensor([S_water, precp]).to(device))

        melt_output = M(S_snow, temp, self.Df, self.Tmax)
        dS_1 = Ps(precp, temp, self.Tmin) - melt_output
        dS_2 = Pr(precp, temp, self.Tmin) + melt_output - step_fct(S_water) * lday * torch.exp(
            ET_output) - step_fct(S_water) * torch.exp(Q_output)
        return dS_1, dS_2

    def output(self, x, sol):
        sol_1 = sol[1]
        y_hat = torch.exp(self.Q_net(torch.concat([sol_1.unsqueeze(1), x[:, 2].unsqueeze(1).to(device)], dim=1)))
        return y_hat


class M100_Func(nn.Module):
    def __init__(self, net, params, interps, ode_lib='torchdiffeq'):
        super().__init__()
        self.net = net
        self.net.train()
        self.ode_lib = ode_lib
        self.f, self.Smax, self.Qmax, self.Df, self.Tmax, self.Tmin = params
        self.precp_interp, self.temp_interp, self.lday_interp = interps

    def forward(self, t, S):
        S_snow, S_water = S

        precp = self.precp_interp.evaluate(t).to(torch.float32).to(device)
        temp = self.temp_interp.evaluate(t).to(torch.float32).to(device)
        lday = self.lday_interp.evaluate(t).to(torch.float32).to(device)

        net_output = self.net(torch.tensor([S_snow, S_water, precp, temp]).to(device))
        melt_output = torch.relu(step_fct(S_snow)) * torch.sinh(net_output[2])
        dS_1 = torch.relu(torch.sinh(net_output[3]) * step_fct(-temp)) - melt_output
        dS_2 = torch.relu(torch.sinh(net_output[4])) + melt_output - step_fct(
            S_water) * lday * torch.exp(net_output[0]) - step_fct(S_water) * torch.exp(net_output[1])
        return dS_1, dS_2

    def output(self, x, sol):
        sol_0, sol_1 = sol[0], sol[1]
        y_hat = torch.exp(self.net(torch.concat(
            [sol_0.unsqueeze(1), sol_1.unsqueeze(1), x[:, 2:4]], dim=1).to(device))[:, 1:2])
        return y_hat


class ODESolver(BaseLearner):
    def __init__(self, solve_func: nn.Module, rtol=1e-6, atol=1e-6, ode_lib='torchdiffeq',
                 loss_metric=torch.nn.L1Loss(), eval_metric_list=None, lr=0.01, optimizer=None):
        super().__init__(solve_func, loss_metric, eval_metric_list, lr, optimizer)
        self.solve_func = solve_func
        self.solve_func.train()
        self.ode_lib = ode_lib
        self.rtol = rtol
        self.atol = atol

    def forward(self, x, t_eval):
        if len(x.shape) > 2:
            x = x[0]
        if len(t_eval.shape) > 1:
            t_eval = t_eval[0]
        t_eval = t_eval.to(torch.float32)
        if self.ode_lib == 'torchdiffeq':
            sol = odeint(self.solve_func.forward, y0=(x[0, 0], x[0, 1]), t=t_eval, rtol=self.rtol, atol=self.atol)
        elif self.ode_lib == 'torchode':
            import torchode as to
            y0 = torch.tensor([[x[0, 0], x[0, 1]]])
            t_eval = t_eval.unsqueeze(0)
            term = to.ODETerm(self.solve_func.forward)
            step_method = to.Dopri5(term=term)
            step_size_controller = to.IntegralController(atol=self.atol, rtol=self.rtol, term=term)
            adjoint = to.AutoDiffAdjoint(step_method, step_size_controller)
            problem = to.InitialValueProblem(y0=y0, t_eval=t_eval)
            sol = adjoint.solve(problem)
            sol_result = sol.ys
            sol_1 = sol_result[0, :, 1]
        elif self.ode_lib == 'torchcde':
            coeffs = torchcde.natural_cubic_spline_coeffs(x.unsqueeze(0), t_eval)
            X = torchcde.CubicSpline(coeffs)
            sol = torchcde.cdeint(X=X, func=self.solve_func, rtol=self.rtol, atol=self.atol,
                                  z0=(x[0, 0], x[0, 1]), t=t_eval, adjoint_options={"norm": "seminorm"})
            sol_1 = sol[0, :, 1]
        else:
            raise NotImplementedError
        y_hat = self.solve_func.output(x, sol)
        return y_hat
