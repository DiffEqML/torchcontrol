import torch
from warnings import warn
from torch import cos, sin
from .template import ControlledSystemTemplate


class ForceMass(ControlledSystemTemplate):
    '''System of a force acting on a mass with unitary weight'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   
        
    def dynamics(self, t, x):
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x)

        # States
        p = x[...,1:]

        # Differential Equations
        dq = p
        dp = u 
        # trick for broadcasting into the same dimension
        self.cur_f = torch.cat(torch.broadcast_tensors(dq, dp), -1)
        return self.cur_f


class LTISystem(ControlledSystemTemplate):
    """Linear Time Invariant System
    Args:
        A (Tensor): dynamics matrix
        B (Tensor): controller weights
    """
    def __init__(self, A=None, B=None, *args, **kwargs):
        super().__init__(*args, **kwargs)   
        if A is None:
            raise ValueError("Matrix A was not declared")
        self.A = A
        self.dim = A.shape[0]
        if B is None:
            warn("Controller weigth matrix B not specified;" 
                 " using default identity matrix")
            self.B = torch.eye(self.dim).to(A)
        else:
            self.B = B.to(A)
            
    def dynamics(self, t, x):
        """The system is described by the ODE:
        dx = Ax + BU(t,x)
        We perform the operations in batches via torch.einsum()
        """
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x)

        # Differential equations        
        dx = torch.einsum('jk, ...bj -> ...bk', self.A, x) + \
            torch.einsum('ij, ...bj -> ...bi', self.B, u)
        return dx
    

class SpringMass(ControlledSystemTemplate):
    """
    Spring Mass model
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.m  = 1. 
        self.k  = 0.5

    def dynamics(self, t, x):
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x)

        # States
        q, p = x[..., :1], x[..., 1:]

        # Differential equations
        dq = p/self.m
        dp = -self.k*q + u
        self.cur_f = torch.cat([dq, dp], -1)
        return self.cur_f


class Pendulum(ControlledSystemTemplate):
    """
    Inverted pendulum with torsional spring
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.m  = 1. 
        self.k  = 0.5
        self.l  = 1
        self.qr = 0
        self.beta  = 0.01
        self.g  = 9.81

    def dynamics(self, t, x):
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x)

        # States
        q, p = x[..., :1], x[..., 1:]

        # Differential equations
        dq = p/self.m
        dp = -self.k*(q - self.qr) - self.m*self.g*self.l*sin(q)- self.beta*p/self.m + u
        self.cur_f = torch.cat([dq, dp], -1)
        return self.cur_f
    

class Acrobot(ControlledSystemTemplate):
    """
    Acrobot: underactuated 2dof manipulator
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.m1 = 1.
        self.m2 = 1.
        self.l1 = 1.
        self.l2 = 1.
        self.b1 = 1
        self.b2 = 1
        self.g  = 9.81

    def dynamics(self, t, x):
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x)

        with torch.set_grad_enabled(True):
            # States
            q1, q2, p1, p2 = x[:, :1], x[:, 1:2], x[:, 2:3], x[:, 3:4]

            # Variables
            s1, s2 = sin(q1), sin(q2)
            c2, c2 = cos(q1), cos(q2)
            s12, c12, s212 = sin(q1-q2), cos(q1-q2), sin(2*(q1-q2))
            h1 = p1*p2*s12/(self.l1*self.l2*(self.m1 + self.m2*(s12**2)))    
            h2 = self.m2*(self.l2**2)*(p1**2) + (self.m1+self.m2)*(self.l1**2)*(p2**2) - 2*self.m2*self.l1*self.l2*p1*p2*c12
            h2 = h2/(2*((self.l1*self.l2)**2)*(self.m1 + self.m2*(s12**2))**2)

            # Differential Equations
            dqdt = torch.cat([
                (self.l2*p1 - self.l1*p2*c12)/((self.l1**2)*self.l2*(self.m1 + self.m2*(s12**2))),
                (-self.m2*self.l2*p1*c12 + (self.m1+self.m2)*self.l1*p2)/(self.m2*(self.l2**2)*self.l1*(self.m1 + self.m2*(s12**2)))
                ], 1)
            dpdt = torch.cat([
                -(self.m1+self.m2)*self.g*self.l1*s1 - h1 + h2*s212 - self.b1*dqdt[:,:1],
                -self.m2*self.g*self.l2*s2 + h1 - h2*s212 - self.b2*dqdt[:,1:]], 1)
            self.cur_f = torch.cat([dqdt, dpdt+u], 1)
        return self.cur_f


class CartPole(ControlledSystemTemplate):
    '''Continuous version of the OpenAI Gym cartpole
    Inspired by: https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.gravity = 9.81
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)
        
    def dynamics(self, t, x_):
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x_) # controller
        
        # States
        x   = x_[..., 0:1]
        dx  = x_[..., 1:2]
        θ   = x_[..., 2:3]
        dθ  = x_[..., 3:4]
        
        # Auxiliary variables
        cosθ, sinθ = cos(θ), sin(θ)
        temp = (u + self.polemass_length * dθ**2 * sinθ) / self.total_mass
        
        # Differential Equations
        ddθ = (self.gravity * sinθ - cosθ * temp) / \
                (self.length * (4.0/3.0 - self.masspole * cosθ**2 / self.total_mass))
        ddx = temp - self.polemass_length * ddθ * cosθ / self.total_mass
        self.cur_f = torch.cat([dx, ddx, dθ, ddθ], -1)
        return self.cur_f

    def render(self):
        raise NotImplementedError("TODO: add the rendering from OpenAI Gym")
