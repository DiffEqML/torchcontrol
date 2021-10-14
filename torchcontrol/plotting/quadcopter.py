from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
from ..systems.quadcopter import euler_matrix

# Cube util function
def cuboid_data2(pos, size=(1,1,1), rotation=None):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    if rotation is not None:
        for i in range(4):
            X[:,i,:] = np.dot(rotation, X[:,i,:].T).T
    X += pos
    return X

# Plot cube for drone body
def plot_cube(position,size=None,rotation=None,color=None, **kwargs):
    if not isinstance(color,(list,np.ndarray)): color=["C0"]*len(position)
    if not isinstance(size,(list,np.ndarray)): size=[(1,1,1)]*len(position)
    g = cuboid_data2(position, size=size, rotation=rotation)
    return Poly3DCollection(g,  
                            facecolor=np.repeat(color,6), **kwargs)



def plot_quadcopter_trajectories_3d(traj, x_star, i=0):
    '''
    Plot trajectory of the drone up to the i-th element
    Args
        traj: drone trajectory
        x_star: target state
        i: plot until i-th frame
    '''
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(projection='3d')  
    if isinstance(traj, torch.Tensor): traj = traj.numpy()
    # For visualization
    scale = 1.5
    s = 50
    dxm = scale*0.16      # arm length (m)
    dym = scale*0.16      # arm length (m)
    dzm = scale*0.05      # motor height (m)
    s_drone = scale*10 # drone body dimension
    lw = scale
    drone_size = [dxm/2, dym/2, dzm]
    drone_color = ["royalblue"]

    lim = [0, x_star[2]*1.2]
    ax.set_xlim3d(lim[0], lim[1])
    ax.set_ylim3d(lim[0], lim[1])
    ax.set_zlim3d(lim[0], lim[1])

    l1, = ax.plot([], [], [], lw=lw, color='red')
    l2, = ax.plot([], [], [], lw=lw, color='green')

    body, = ax.plot([], [], [], marker='o', markersize=s_drone, color='black', markerfacecolor='grey')

    initial = traj[0]


    init = ax.scatter(initial[0], initial[1], initial[2], marker='^', color='blue', label='Initial Position', s=s)
    fin = ax.scatter(x_star[0], x_star[1], x_star[2], marker='*', color='red', label='Target', s=s) # set linestyle to none

    ax.plot(traj[:i, 0], traj[:i, 1], traj[:i, 2],  alpha=1, linestyle='-')
    pos = traj[i-1]
    x = pos[0]
    y = pos[1]
    z = pos[2]

    # Trick to reuse the same function
    R = euler_matrix(torch.Tensor([pos[3]]), torch.Tensor([pos[4]]), torch.Tensor([pos[5]])).numpy().squeeze(0)
    motorPoints = np.array([[dxm, -dym, dzm], [0, 0, 0], [dxm, dym, dzm], [-dxm, dym, dzm], [0, 0, 0], [-dxm, -dym, dzm],  [-dxm, -dym, -dzm]])
    motorPoints = np.dot(R, np.transpose(motorPoints))
    motorPoints[0,:] += x 
    motorPoints[1,:] += y 
    motorPoints[2,:] += z

    # Motors
    l1.set_data(motorPoints[0,0:3], motorPoints[1,0:3])
    l1.set_3d_properties(motorPoints[2,0:3])
    l2.set_data(motorPoints[0,3:6], motorPoints[1,3:6])
    l2.set_3d_properties(motorPoints[2,3:6])

    # Body
    pos = ((motorPoints[:, 6] + 2*motorPoints[:, 1])/3)
    body = plot_cube(pos, drone_size, rotation=R, edgecolor="k")
    ax.add_collection3d(body)

    ax.legend()
    ax.set_xlabel(f'$x~[m]$')
    ax.set_ylabel(f'$y~[m]$')
    ax.set_zlabel(f'$z~[m]$')

    ax.legend(loc='upper center', bbox_to_anchor=(0.52, -0.05),
            fancybox=True, shadow=False, ncol=3)


def animate_quadcopter_3d(traj, x_star, t_span, path='quadcopter_animation.gif', html_embed=False):
    '''
    Animate drone and save gif
    Args
        traj: drone trajectory
        x_star: target position
        t_span: time vector corresponding to each trajectory
        path: save path for 
        html_embed: embed mp4 video in the page
    '''

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    # For visualization
    scale = 1.5
    s = 50
    dxm = scale*0.16      # arm length (m)
    dym = scale*0.16      # arm length (m)
    dzm = scale*0.05      # motor height (m)
    s_drone = scale*10 # drone body dimension
    lw = scale
    drone_size = [dxm/2, dym/2, dzm]
    drone_color = ["royalblue"]

    lim = [0, x_star[2]*1.2]
    ax.set_xlim3d(lim[0], lim[1])
    ax.set_ylim3d(lim[0], lim[1])
    ax.set_zlim3d(lim[0], lim[1])
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')

    lines1, lines2 = [], []
    l1, = ax.plot([], [], [], lw=2, color='red')
    l2, = ax.plot([], [], [], lw=2, color='green')

    body, = ax.plot([], [], [], marker='o', markersize=s_drone, color='black', markerfacecolor='black')

    initial = traj[0]
    tr = traj

    # Single frame plotting
    def get_frame(i):
        del ax.collections[:] # remove previous 3D elements 
        init = ax.scatter(initial[0], initial[1], initial[2], marker='^', color='blue', label='Initial Position', s=s)
        fin = ax.scatter(x_star[0], x_star[1], x_star[2], marker='*', color='red', label='Target', s=s) # set linestyle to none
        ax.plot(tr[:i, 0], tr[:i, 1], tr[:i, 2],  alpha=0.1, linestyle='-.', color='tab:blue')
        time = t_span[i]
        pos = tr[i]
        x = pos[0]
        y = pos[1]
        z = pos[2]

        x_from0 = tr[0:i,0]
        y_from0 = tr[0:i,1]
        z_from0 = tr[0:i,2]

        # Trick to reuse the same function
        R = euler_matrix(torch.Tensor([pos[3]]), torch.Tensor([pos[4]]), torch.Tensor([pos[5]])).numpy().squeeze(0)
        motorPoints = np.array([[dxm, -dym, dzm], [0, 0, 0], [dxm, dym, dzm], [-dxm, dym, dzm], [0, 0, 0], [-dxm, -dym, dzm],  [-dxm, -dym, -dzm]])
        motorPoints = np.dot(R, np.transpose(motorPoints))
        motorPoints[0,:] += x 
        motorPoints[1,:] += y 
        motorPoints[2,:] += z

        # Motors
        l1.set_data(motorPoints[0,0:3], motorPoints[1,0:3])
        l1.set_3d_properties(motorPoints[2,0:3])
        l2.set_data(motorPoints[0,3:6], motorPoints[1,3:6])
        l2.set_3d_properties(motorPoints[2,3:6])

        # Body
        pos = ((motorPoints[:, 6] + 2*motorPoints[:, 1])/3)
        body = plot_cube(pos, drone_size, rotation=R, edgecolor="k")
        ax.add_collection3d(body)

        ax.set_title("Quadcopter Trajectory, t = {:.2f} s".format(time))
        
    # Unused for now
    def anim_callback(i, get_world_frame):
        frame = get_world_frame(i)
        set_frame(frame)
        
    # Frame setting
    def set_frame(frame):
        # convert 3x6 world_frame matrix into three line_data objects which is 3x2 (row:point index, column:x,y,z)
        lines_data = [frame[:,[0,2]], frame[:,[1,3]], frame[:,[4,5]]]
        ax = plt.gca()
        lines = ax.get_lines()
        for line, line_data in zip(lines[:3], lines_data):
            x, y, z = line_data
            line.set_data(x, y)
            line.set_3d_properties(z)
            
    an = FuncAnimation(fig,
                        get_frame,
                        init_func=None,
                        frames=len(t_span)-1, interval=20, blit=False)

    an.save(path, dpi=80, writer='imagemagick', fps=20)

    if html_embed: HTML(an.to_html5_video())


def plot_quadcopter_trajectories(traj):
    '''
    Simple plot with all variables in time
    '''

    fig, axs = plt.subplots(12, 1, figsize=(10, 10))

    axis_labels = ['$x$', '$y$', '$z$', '$\phi$', r'$\theta$', '$\psi$', '$\dot x$', '$\dot y$', '$\dot z$', '$\dot \phi$', '$\dot \theta$', '$\dot \psi$']

    for ax, i, axis_label in zip(axs, range(len(axs)), axis_labels):
        ax.plot(traj[:, i].cpu().detach(), color='tab:red')
        ax.label_outer()
        ax.set_ylabel(axis_label)

    fig.suptitle('Trajectories', y=0.92, fontweight='bold')


def plot_quadcopter_controls(controls):
    '''
    Simple plot with all variables in time
    '''

    fig, axs = plt.subplots(4, 1, figsize=(10, 5))

    axis_labels = ['$u_0$ RPM', '$u_1$ RPM','$u_2$ RPM','$u_3$ RPM']

    for ax, i, axis_label in zip(axs, range(len(axs)), axis_labels):
        ax.plot(controls[:, i].cpu().detach(), color='tab:red')
        ax.label_outer()
        ax.set_ylabel(axis_label)

    fig.suptitle('Control inputs', y=0.94, fontweight='bold')