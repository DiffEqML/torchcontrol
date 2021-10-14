import torch
import torch.nn as nn
from torch.autograd import grad
from warnings import warn
from torch import cos, sin, sign, norm
from .template import ControlledSystemTemplate


class CSTR(ControlledSystemTemplate):
    '''
    Controlled Continuous Stirred Tank Reactor
    Reference: https://www.do-mpc.com/en/latest/example_gallery/CSTR.html
    '''
    def __init__(self, *args, alpha=1, beta=1, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Parameters
        self.α = alpha # empirical parameter, may vary
        self.β = beta # empirical parameter, may vary
        self.K0_ab = 1.287e12 # K0 [h^-1]
        self.K0_bc = 1.287e12 # K0 [h^-1]
        self.K0_ad = 9.043e9 # K0 [l/mol.h]
        self.R_gas = 8.3144621e-3 # Universal gas constant
        self.E_A_ab = 9758.3*1.00 #* R_gas# [kj/mol]
        self.E_A_bc = 9758.3*1.00 #* R_gas# [kj/mol]
        self.E_A_ad = 8560.0*1.0 #* R_gas# [kj/mol]
        self.H_R_ab = 4.2 # [kj/mol A]
        self.H_R_bc = -11.0 # [kj/mol B] Exothermic
        self.H_R_ad = -41.85 # [kj/mol A] Exothermic
        self.Rou = 0.9342 # Density [kg/l]
        self.Cp = 3.01 # Specific Heat capacity [kj/Kg.K]
        self.Cp_k = 2.0 # Coolant heat capacity [kj/kg.k]
        self.A_R = 0.215 # Area of reactor wall [m^2]
        self.V_R = 10.01 #0.01 # Volume of reactor [l]
        self.m_k = 5.0 # Coolant mass[kg]
        self.T_in = 130.0 # Temp of inflow [Celsius]
        self.K_w = 4032.0 # [kj/h.m^2.K]
        self.C_A0 = (5.7+4.5)/2.0*1.0 # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]

    def dynamics(self, t, x):
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x)
        
        # States
        C_a = x[..., 0:1]
        C_b = x[..., 1:2]
        T_R = x[..., 2:3]
        T_K = x[..., 3:4] 
        
        # Controller
        F, dQ = u[..., :1], u[..., 1:] 

        # Auxiliary variables
        K_1 = self.β * self.K0_ab * torch.exp((-self.E_A_ab)/((T_R+273.15)))
        K_2 =  self.K0_bc * torch.exp((-self.E_A_bc)/((T_R+273.15)))
        K_3 = self.K0_ad * torch.exp((-self.α*self.E_A_ad)/((T_R+273.15)))
        T_dif = T_R - T_K
        
        # Differential equations
        dC_a = F*(self.C_A0 - C_a) -K_1*C_a - K_3*(C_a**2)
        dC_b = -F*C_b + K_1*C_a - K_2*C_b
        dT_R = ((K_1*C_a*self.H_R_ab + K_2*C_b*self.H_R_bc + K_3*(C_a**2)*self.H_R_ad)/(-self.Rou*self.Cp)) \
                    + F*(self.T_in-T_R) +(((self.K_w*self.A_R)*(-T_dif))/(self.Rou*self.Cp*self.V_R))
        dT_K = (dQ + self.K_w*self.A_R*(T_dif))/(self.m_k*self.Cp_k)
        self.cur_f = torch.cat([dC_a, dC_b, dT_R, dT_K], -1)
        return self.cur_f