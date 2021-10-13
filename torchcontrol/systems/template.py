import torch
import torch.nn as nn
from torchdyn.numerics.odeint import odeint

class ControlledSystemTemplate(nn.Module):
    """
    Template Model
    """
    def __init__(self, u, 
                 solver='euler', 
                 retain_u=False,
                 **odeint_kwargs):
        super().__init__()
        self.u = u
        self.solver = solver
        self.retain_u = retain_u # use for retaining control input (e.g. MPC simulation)
        self.nfe = 0 # count number of function evaluations of the vector field
        self.cur_f = None # current dynamics evaluation
        self.cur_u = None # current controller value
        self._retain_flag = False # temporary flag for evaluating the controller only the first time
        self.odeint_kwargs = odeint_kwargs

    def forward(self, x0, t_span):
        x = [x0[None]]
        xt = x0
        if self.retain_u:
            # Iterate over the t_span: evaluate the controller the first time only and then retain it
            # this is useful to simulate control with MPC
            for i in range(len(t_span)-1):
                self._retain_flag = False
                diff_span = torch.linspace(t_span[i], t_span[i+1], 2)
                odeint(self.dynamics, xt, diff_span, solver=self.solver, **self.odeint_kwargs)[1][-1]
                x.append(xt[None])
            traj = torch.cat(x)
        else:
            # Compute trajectory with odeint and base solvers
            traj = odeint(self.dynamics, xt, t_span, solver=self.solver, **self.odeint_kwargs)[1]
        return traj

    def reset_nfe(self):
        """Return number of function evaluation and reset"""
        cur_nfe = self.nfe; self.nfe = 0
        return cur_nfe

    def _evaluate_controller(self, t, x):
        '''
        If we wish not to re-evaluate the control input, we set the retain
        flag to True so we do not re-evaluate next time
        '''
        if self.retain_u:
            if not self._retain_flag:
                self.cur_u = self.u(t, x)
                self._retain_flag = True
            else: 
                pass # We do not re-evaluate the control input
        else:
            self.cur_u = self.u(t, x)
        return self.cur_u
    
        
    def dynamics(self, t, x):
        '''
        Model dynamics in the form xdot = f(t, x, u)
        '''
        raise NotImplementedError