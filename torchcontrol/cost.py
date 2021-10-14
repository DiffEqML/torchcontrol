import torch
import torch.nn as nn


class IntegralCost(nn.Module):
    '''Integral cost function
    Args:
        x_star: torch.tensor, target position
        u_star: torch.tensor / float, controller with no cost
        P: terminal cost weights
        Q: state weights
        R: controller regulator weights
    '''
    def __init__(self, x_star, u_star=0, P=0, Q=1, R=0):
        super().__init__()
        self.x_star = x_star
        self.u_star = u_star
        self.P, self.Q, self.R, = P, Q, R
        
    def forward(self, x, u=torch.Tensor([0.])):
        """
        x: trajectories
        u: control inputs
        """
        cost = torch.norm(self.P*(x[-1] - self.x_star), p=2, dim=-1).mean()
        cost += torch.norm(self.Q*(x - self.x_star), p=2, dim=-1).mean()
        cost += torch.norm(self.R*(u - self.u_star), p=2, dim=-1).mean()
        return cost

    
def circle_loss(z, a=1):
    """Make the system follow a circle with radius a"""
    x, y = z[...,:1], z[...,1:]
    loss = torch.abs(x**2 + y**2 - a)
    return loss.mean()


def circus_loss(z, a=1., k=2.1):
    """Make the system follow an elongated circus-like shape with
    curve a and length k"""
    x, y = z[...,:1], z[...,1:]
    
    a1 = torch.sqrt((x + a)**2 + y**2)
    a2 = torch.sqrt((x - a)**2 + y**2)
    return torch.abs(a1*a2 - k).mean()
