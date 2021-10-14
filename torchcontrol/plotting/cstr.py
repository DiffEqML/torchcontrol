import matplotlib.pyplot as plt
import torch

def plot_cstr_trajectories_controls(traj, controls, tf=0.5):
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    alpha = 1
    dummy=torch.linspace(0, tf, controls.shape[0])
    axs[0].plot(traj[:, 0], color='blue', alpha=alpha, label=r'$C_a$')
    axs[0].plot(traj[:, 1], color='orange', alpha=alpha, label=r'$C_b$')
    axs[1].plot(traj[:, 2], color='blue', alpha=alpha, label=r'$T_R$')
    axs[1].plot(traj[:, 3], color='orange', alpha=alpha, label=r'$T_K$')
    axs[2].step(dummy, controls[:, 0], alpha=alpha, label=r'$F$')
    axs[3].step(dummy, controls[:, 1], alpha=alpha, label=r'$\dot{Q}$')

    axs[0].set_ylabel('$Concentration~[mol/l]$')
    axs[1].set_ylabel('$Temperature~[°C]$')
    axs[2].set_ylabel('$Flow~[l/h]$')
    axs[3].set_xlabel('$Time~[h]$')
    axs[3].set_ylabel('$Heat~[kW]$')

    for ax in axs:
        ax.legend()
        ax.label_outer()