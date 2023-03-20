import numpy as np
import torch
import torch.nn as nn
import torchcde
from torchdiffeq import odeint, odeint_adjoint

from utils.training_utils import BaseLearner

# --------------------------------------------------------------
# NODE meta: optimal param 是面向于bacth输入的模型, 在meta模型研究基础上
# --------------------------------------------------------------

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class M50_Func(nn.Module):
    def __init__(self, ET_net, Q_net, params, interps, ode_lib='torchdiffeq'):
        super().__init__()
        self.f = nn.Parameter(torch.tensor(params[0]), requires_grad=True)
        self.Smax = nn.Parameter(torch.tensor(params[1]), requires_grad=True)
        self.Qmax = nn.Parameter(torch.tensor(params[2]), requires_grad=True)
        self.Df = nn.Parameter(torch.tensor(params[3]), requires_grad=True)
        self.Tmax = nn.Parameter(torch.tensor(params[4]), requires_grad=True)
        self.Tmin = nn.Parameter(torch.tensor(params[5]), requires_grad=True)
        self.ET_net = ET_net
        self.ET_net.train()
        self.ode_lib = ode_lib
        self.Q_net = Q_net
        self.Q_net.train()
        self.precp_interp, self.temp_interp, self.lday_interp = interps
        # self.cde_linear = nn.Linear(2, 2 * 5)

    def clamp_parameters(self):
        self.f.data.clamp_(0.0, 1.0)
        self.Smax.data.clamp_(100.0, 1500.0)
        self.Qmax.data.clamp_(10.0, 50.0)
        self.Df.data.clamp_(0.01, 5.0)
        self.Tmax.data.clamp_(0.0, 3.0)
        self.Tmin.data.clamp_(-3.0, 0.0)

    def forward(self, t, S):
        from models.common_net import Ps, Pr, M, step_fct

        self.clamp_parameters()
        S_snow, S_water = S[0][0], S[0][1]
        precp = self.precp_interp.evaluate(t).to(torch.float32)
        temp = self.temp_interp.evaluate(t).to(torch.float32)
        lday = self.lday_interp.evaluate(t).to(torch.float32)
        ET_output = self.ET_net(torch.tensor([S_snow, S_water, temp]))
        Q_output = self.Q_net(torch.tensor([S_water, precp]))

        melt_output = M(S_snow, temp, self.Df, self.Tmax)
        dS_1 = Ps(precp, temp, self.Tmin) - melt_output
        dS_2 = Pr(precp, temp, self.Tmin) + melt_output - step_fct(S_water) * lday * torch.exp(
            ET_output) - step_fct(S_water) * torch.exp(Q_output)
        # return self.cde_linear(torch.tensor([dS_1, dS_2]).unsqueeze(0)).view(1, 2, 5)
        return torch.tensor([dS_1, dS_2]).unsqueeze(0)


class M50_Solver(BaseLearner):
    def __init__(self, solve_func: nn.Module, rtol=1e-6, atol=1e-6, ode_lib='torchdiffeq',
                 loss_metric=torch.nn.MSELoss(), eval_metric_list=None, lr=0.01, optimizer=None):
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
        y0 = torch.tensor([[x[0, 0], x[0, 1]]])

        if self.ode_lib == 'torchdiffeq':
            sol = odeint(self.solve_func.forward, y0=y0, t=t_eval, rtol=self.rtol, atol=self.atol)
            sol_1 = sol[:, 0, 1]
        elif self.ode_lib == 'torchode':
            import torchode as to
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
            sol = torchcde.cdeint(X=X, func=self.solve_func, rtol=self.rtol, atol=self.atol, z0=y0, t=t_eval,
                                  adjoint_options={"norm": "seminorm"})
            sol_1 = sol[0, :, 1]
        else:
            raise NotImplementedError
        y_hat = torch.exp(self.solve_func.Q_net(torch.concat([sol_1.unsqueeze(1), x[:, 2].unsqueeze(1)], dim=1)))
        return y_hat
