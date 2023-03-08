import torch.nn as nn
import torch


class DiffeqSolver(nn.Module):
    def __init__(self, diffeq_func, method='dopri8', rtol=1e-3, atol=1e-3, lib='torchdiffeq'):
        super(DiffeqSolver, self).__init__()

        self.diffeq_func = diffeq_func
        self.rtol = rtol
        self.atol = atol
        self.ode_method = method
        self.lib = lib

    def forward(self, y0, t, X=None):
        if self.lib == "torchdiffeq":
            from torchdiffeq import odeint

            sol = odeint(self.diffeq_func, y0=y0, t=t, rtol=self.rtol, atol=self.atol, method=self.ode_method)
        elif self.lib == "torchcde":
            import torchcde
            z0 = torch.concat([y0[0].unsqueeze(1), y0[1].unsqueeze(1)], axis=1)
            sol = torchcde.cdeint(X=X, func=self.diffeq_func, rtol=self.rtol, atol=self.atol, z0=z0, t=t)
        elif self.lib == "torchode":
            import torchode
            y0 = torch.concat([y0[0].unsqueeze(1), y0[1].unsqueeze(1)], axis=1)
            ode_term = torchode.ODETerm(self.diffeq_func.forward)
            ode_step_method = torchode.Dopri5(term=ode_term)
            step_size_controller = torchode.IntegralController(atol=self.atol, rtol=self.rtol, term=ode_term)
            solver = torchode.AutoDiffAdjoint(ode_step_method, step_size_controller)
            sol = solver.solve(torchode.InitialValueProblem(y0=y0, t_eval=torch.tensor([0., 1.])))
        else:
            raise NotImplementedError
        return sol
