import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

# --------------------------------------------------------------
# NODE version4.0是面向于输入为完整步长的模型
# --------------------------------------------------------------

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class M50(nn.Module):
    def __init__(self, ET_net, Q_net, params, interps, ode_lib='torchdiffeq'):
        super().__init__()
        self.f, self.Smax, self.Qmax, self.Df, self.Tmax, self.Tmin = params
        self.ET_net = ET_net
        self.ode_lib = ode_lib
        self.Q_net = Q_net
        self.precp_interp, self.temp_interp, self.lday_interp = interps

    def forward(self, x, t_eval):
        def solve_ode(t, S):
            from models.common_net import Ps, Pr, M, step_fct
            S_snow, S_water = S

            precp = self.precp_interp.evaluate(t).to(torch.float32)
            temp = self.temp_interp.evaluate(t).to(torch.float32)
            lday = self.lday_interp.evaluate(t).to(torch.float32)

            # t = t.detach().cpu()
            # precp = torch.from_numpy(self.precp_interp(t.numpy()).astype(np.float32)).to(device)
            # temp = torch.from_numpy(self.temp_interp(t.numpy()).astype(np.float32)).to(device)
            # lday = torch.from_numpy(self.lday_interp(t.numpy()).astype(np.float32)).to(device)

            ET_output = self.ET_net(torch.tensor([S_snow, S_water, temp]))
            Q_output = self.Q_net(torch.tensor([S_water, precp]))

            melt_output = M(S_snow, temp, self.Df, self.Tmax)
            dS_1 = Ps(precp, temp, self.Tmin) - melt_output
            dS_2 = Pr(precp, temp, self.Tmin) + melt_output - step_fct(S_water) * lday * torch.exp(
                ET_output) - step_fct(S_water) * torch.exp(Q_output)
            return dS_1, dS_2

        if self.ode_lib == 'torchdiffeq':
            t_eval = t_eval.to(torch.float32)
            y0 = (x[0, 0], x[0, 1])
            sol = odeint(solve_ode, y0=y0, t=t_eval, rtol=1e-6, atol=1e-6)
            sol_1 = sol[1]
        elif self.ode_lib == 'torchode':
            import torchode as to
            y0 = torch.tensor([[x[0, 0]], [x[0, 1]]])
            t_eval = t_eval.unsqueeze(0).expand(y0.shape[0], len(t_eval))
            term = to.ODETerm(solve_ode)
            step_method = to.Dopri5(term=term)
            step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-6, term=term, dt_max=1.)
            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            sol = solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))
            print('complete!')
        else:
            raise NotImplementedError
        y_hat = torch.exp(self.Q_net(torch.concat([sol_1.unsqueeze(1), x[:, 2].unsqueeze(1)], dim=1)))
        return y_hat


class M100(nn.Module):
    def __init__(self, net, params, interps, ode_lib='torchdiffeq'):
        super().__init__()
        self.net = net
        self.net.train()
        self.ode_lib = ode_lib
        self.f, self.Smax, self.Qmax, self.Df, self.Tmax, self.Tmin = params
        self.precp_interp, self.temp_interp, self.lday_interp = interps

    def forward(self, x, t_eval):
        def solve_ode(t, S):
            from models.common_net import Ps, Pr, M, step_fct
            S_snow, S_water = S
            t = t.detach().cpu()
            precp = torch.from_numpy(self.precp_interp(t.numpy()).astype(np.float32)).to(device)
            temp = torch.from_numpy(self.temp_interp(t.numpy()).astype(np.float32)).to(device)
            lday = torch.from_numpy(self.lday_interp(t.numpy()).astype(np.float32)).to(device)
            net_output = self.net(torch.tensor([S_snow, S_water, precp, temp]).to(device))
            melt_output = torch.relu(step_fct(S_snow)) * torch.sinh(net_output[2])
            dS_1 = torch.relu(torch.sinh(net_output[3]) * step_fct(-temp)) - melt_output
            dS_2 = torch.relu(torch.sinh(net_output[4])) + melt_output - step_fct(
                S_water) * lday * torch.exp(net_output[0]) - step_fct(S_water) * torch.exp(net_output[1])
            return dS_1, dS_2

        if self.ode_lib == 'torchdiffeq':
            # 1.using torchdiffeq
            t_eval = t_eval.to(torch.float32)
            y0 = (x[0, 0], x[0, 1])
            sol = odeint(solve_ode, y0=y0, t=t_eval, rtol=1e-6, atol=1e-6)
            sol_0 = sol[0]
            sol_1 = sol[1]
        # 2.using torchode
        elif self.ode_lib == 'torchode':
            import torchode as to
            y0 = torch.tensor([[x[0, 0]], [x[0, 1]]])
            t_eval = t_eval.unsqueeze(0).expand(y0.shape[0], len(t_eval))
            term = to.ODETerm(solve_ode)
            step_method = to.Dopri5(term=term)
            step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-6, term=term, dt_max=1.)
            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            sol = solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))
            sol_0 = sol[0]
            sol_1 = sol[1]
            print('complete!')
        else:
            raise NotImplementedError
        y_hat = torch.exp(
            self.net(torch.concat([sol_0.unsqueeze(1), sol_1.unsqueeze(1), x[:, 2:4]], dim=1).to(device))[:, 1:2])
        return y_hat
