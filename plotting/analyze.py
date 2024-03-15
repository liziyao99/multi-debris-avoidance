import matplotlib.pyplot as plt
import numpy as np
import torch
import math

from plotting.dataplot import moving_average
from agent.agent import rlAgent

def criticGrid2D(agent:rlAgent, dims=(0,1), span=(-2,2), step=21, singles=0.):
    x0 = np.linspace(span[0], span[1], step)
    xs = tuple([x0]*2)
    XY = np.meshgrid(*xs)
    if not hasattr(singles, "__iter__"):
        singles = [singles]*(agent.obs_dim-2)
    coords = []
    idx_d = 0
    idx_s = 0
    for i in range(agent.obs_dim):
        if i in dims:
            coords.append(XY[idx_d].reshape((-1,1)))
            idx_d += 1
        else:
            coords.append(np.ones((step*step,1))*singles[idx_s])
            idx_s += 1
    points = np.hstack(coords)
    points = torch.from_numpy(points).float().to(agent.device)
    values = agent.critic(points).detach().cpu().numpy()
    values = values.reshape([step]*2)
    return XY, values

def criticContour(agent:rlAgent, dims=(0,1), span=(-2,2), step=21, singles=0.):
    XY, Z = criticGrid2D(agent, dims=dims, span=span, step=step, singles=singles)
    plt.contourf(XY[0], XY[1], Z)
    plt.colorbar()
    plt.show()

def historyFile(trans_dict:dict, agent:rlAgent, stage=-1, n_debris=0):
    fig_num = 5+n_debris
    col = math.ceil(fig_num/5)
    fig, axs = plt.subplots(5, col, sharex=True, figsize=plt.figaspect(fig_num/(2*col)))
    axs = axs.flatten()
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    # primal position
    d2o = np.linalg.norm(trans_dict["states"][:stage,:3], axis=1)
    for i in range(3):
        axs[0].plot(trans_dict["states"][:stage,i], label=f"r{i}", color=colors[i])
    axs[0].plot(d2o, label="d2o", color=colors[-1])
    axs[0].legend()
    axs[0].set_title("primal position")

    # primal velocity
    for i in range(3):
        axs[1].plot(trans_dict["states"][:stage,i+3], label=f"v{i}", color=colors[i])
    axs[1].legend()
    axs[1].set_title("primal velocity")

    # primal thrust
    for i in range(3):
        axs[2].plot(trans_dict["actions"][:stage,i], color=colors[i], alpha=0.3)
        axs[2].plot(moving_average(trans_dict["actions"][:stage, i], 21), label=f"a{i}", color=colors[i], alpha=0.7)
    axs[2].legend()
    axs[2].set_title("primal thrust")

    # rewards
    axs[3].plot(trans_dict["rewards"][:stage])
    axs[3].set_title("reward")

    # critic
    obss = torch.from_numpy(trans_dict["obss"][:stage]).float().to(agent.device)
    critic = agent.critic(obss).detach().cpu().numpy()
    axs[4].plot(critic)
    axs[4].set_title("critic")

    for j in range(n_debris):
        debris_pos = trans_dict["states"][:stage, 6*(j+1):6*(j+1)+3]
        for i in range(3):
            axs[5+j].plot(debris_pos[:, i], color=colors[i])
        d2d = np.linalg.norm(debris_pos-trans_dict["states"][:stage, :3], axis=1)
        axs[5+j].plot(d2d, color=colors[-1])
        axs[5+j].set_title(f"debris{j}")

    plt.show()
    return fig, axs