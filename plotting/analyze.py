import matplotlib.pyplot as plt
import numpy as np
import torch
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
