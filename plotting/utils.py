import matplotlib.pyplot as plt
import numpy as np

def sphere(loc=(0,0,0), rad=1, ax=None, devide=100, color="r"):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    u = np.linspace(0, 2 * np.pi, devide)
    v = np.linspace(0, np.pi, devide)
    x = rad * np.outer(np.cos(u), np.sin(v)) + loc[0]
    y = rad * np.outer(np.sin(u), np.sin(v)) + loc[1]
    z = rad * np.outer(np.ones(np.size(u)), np.cos(v)) + loc[2]
    s = ax.plot_surface(x, y, z, color=color)
    return ax, s

def cuboid(o:np.ndarray, x:np.ndarray, y:np.ndarray, z:np.ndarray, ax=None, device=100, color="r"):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    vertices = np.zeros((8, 3))
    vertices[0,:] = o
    vertices[1,:] = o + x
    vertices[2,:] = o + y
    vertices[3,:] = o + z
    vertices[4,:] = o + x + y
    vertices[5,:] = o + x + z
    vertices[6,:] = o + y + z
    vertices[7,:] = o + x + y + z

    return ax